"""Virtual Observatory (VO) Lightcurve Parsing and Formatting Module.

This module provides tools, classes, and helper functions to ingest, parse, and format
astronomical lightcurve data files complying with VO standards (e.g., VOTable) or
heuristic text/ASCII formats. It implements data models for coordinate systems (CooSys),
time systems (TimeSys), and photometric calibrations (PhotDM, PhotCal, PhotometryFilter)
and facilitates unit-safe conversions between astronomical magnitudes and fluxes.
"""

import io
import re
import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.table import MaskedColumn
from astropy.io.votable import is_votable
from astropy.io import ascii

from gavo.votable import tableparser, votparse
from gavo.votable.model import VOTable as V

import logging

from volightcurve.vo_unit_codec import (
    rewrite_astropy_empty_unit_attributes,
    to_internal,
    to_wire,
)
from volightcurve.photdm import (
    MagnitudeSystem,
    PogsonZeroPoint,
    ZeroPoint,
)

logger = logging.getLogger(__name__)


class PhotometryFilter:
    """Represents a photometric filter with its identifier and physical spectral location.

    This class corresponds to IVOA photDM:PhotometryFilter data model component, which
    is used to describe a bandpass/filter in astronomical observations.
    """

    def __init__(
        self,
        filter_id=None,
        name: str | None = None,
        spectral_location=None,
        spectral_location_unit: str = None,
    ):
        """Initialises a PhotometryFilter instance.

        Args:
            filter_id (str, optional): Unique passport
                (``photDM:PhotometryFilter.identifier``), e.g. ``Generic/Bessell.I``.
            name (str, optional): Human-readable label
                (``photDM:PhotometryFilter.name``), e.g. ``I`` for plot axes.
            spectral_location (float or astropy.units.Quantity, optional): The physical
                spectral location (e.g., central wavelength) of the filter. Defaults to None.
            spectral_location_unit (str, optional): The physical unit of the spectral
                location (e.g., 'nm', 'AA', 'um') if spectral_location is not already
                an astropy.units.Quantity. Defaults to None.
        """
        self._filter_id = filter_id  # photDM:PhotometryFilter.identifier
        self._name = name  # photDM:PhotometryFilter.name

        # Build the physical quantity
        if spectral_location is None:
            self.spectral_location = None
        else:
            if isinstance(spectral_location, u.Quantity):
                self.spectral_location = spectral_location
            else:
                # Pair the value with the unit (defaulting to dimensionless if None)
                unit = u.dimensionless_unscaled
                if spectral_location_unit is not None:
                    try:
                        unit = u.Unit(spectral_location_unit)
                    except Exception as e:
                        logger.warning(f'Inappropriate spectral_location_unit {spectral_location_unit}: {type(e)}')

                self.spectral_location = spectral_location * unit
        # todo: blah-blah-blah

    @property
    def filter_id(self):
        """Gets or sets the filter identifier (PhotDM passport).

        Returns:
            str: The filter identifier.
        """
        return self._filter_id

    @filter_id.setter
    def filter_id(self, value):
        if not isinstance(value, str):
            raise ValueError("Filter ID must be a string.")
        self._filter_id = value.strip()

    @property
    def name(self):
        """Gets or sets the human-readable filter name (PhotDM ``name``).

        Returns:
            str or None: Display label for plots and UI.
        """
        return self._name

    @name.setter
    def name(self, value):
        if value is None:
            self._name = None
            return
        if not isinstance(value, str):
            raise ValueError("Filter name must be a string.")
        stripped = value.strip()
        self._name = stripped or None

    def __repr__(self):
        return (
            f"<PhotometryFilter: filter_id={self.filter_id} name={self.name} "
            f"spectralLocation={self.spectral_location}>"
        )


class PhotCal:
    """photDM:PhotCal façade — filter calibration with nested ZeroPoint.

    Callers use ``mag_to_flux`` / ``flux_to_mag`` (and error helpers) on this
    object only. Scale-specific maths live on ``self.zero_point``
    (``PogsonZeroPoint`` today; Asinh/Linear stubs for later).

    Flat constructor args (``zp_flux``, ``zp_mag``, ``mag_sys``) remain supported
    and build a ``PogsonZeroPoint`` + ``MagnitudeSystem`` (compressed PhotDM).
    """

    def __init__(
        self,
        zp_flux=1.0,
        zp_flux_unit=None,
        zp_mag=0.0,
        zp_mag_unit=None,
        mag_sys="Vega",
        *,
        zero_point: ZeroPoint | None = None,
        magnitude_system: MagnitudeSystem | None = None,
        photometry_filter: PhotometryFilter | None = None,
        identifier: str | None = None,
    ):
        """Initialises a PhotCal instance.

        Args:
            zp_flux: Zero-point flux (used when ``zero_point`` is omitted).
            zp_flux_unit: Unit for ``zp_flux`` when not a Quantity.
            zp_mag: Reference magnitude (used when ``zero_point`` is omitted).
            zp_mag_unit: Unit for ``zp_mag`` when not a Quantity.
            mag_sys: Magnitude system type string when ``magnitude_system`` omitted.
            zero_point: Nested PhotDM zero-point (Pogson/Asinh/Linear).
            magnitude_system: Nested ``MagnitudeSystem``.
            photometry_filter: Optional ``PhotometryFilter``.
            identifier: Optional PhotCal identifier.
        """
        if zero_point is not None:
            self.zero_point = zero_point
        else:
            self.zero_point = PogsonZeroPoint(
                zp_flux=zp_flux,
                zp_flux_unit=zp_flux_unit,
                zp_mag=zp_mag,
                zp_mag_unit=zp_mag_unit,
            )
        self.magnitude_system = magnitude_system or MagnitudeSystem(type=mag_sys or "Vega")
        self.photometry_filter = photometry_filter
        self.identifier = identifier

    @property
    def _zp_flux(self):
        """Compatibility alias for nested zero-point flux."""
        return self.zero_point._zp_flux

    @_zp_flux.setter
    def _zp_flux(self, value):
        self.zero_point._zp_flux = value

    @property
    def _zp_mag(self):
        """Compatibility alias for nested zero-point magnitude."""
        return self.zero_point._zp_mag

    @_zp_mag.setter
    def _zp_mag(self, value):
        self.zero_point._zp_mag = value

    @property
    def _zp_flux_unit_text(self):
        """Compatibility alias for nested flux unit text."""
        return self.zero_point._zp_flux_unit_text

    @_zp_flux_unit_text.setter
    def _zp_flux_unit_text(self, value):
        self.zero_point._zp_flux_unit_text = value

    @property
    def _zp_flux_unit_unparsed(self):
        """Compatibility alias for unparsed flux-unit flag."""
        return self.zero_point._zp_flux_unit_unparsed

    @_zp_flux_unit_unparsed.setter
    def _zp_flux_unit_unparsed(self, value):
        self.zero_point._zp_flux_unit_unparsed = value

    @property
    def _zp_mag_unit_text(self):
        """Compatibility alias for nested mag unit text."""
        return self.zero_point._zp_mag_unit_text

    @_zp_mag_unit_text.setter
    def _zp_mag_unit_text(self, value):
        self.zero_point._zp_mag_unit_text = value

    @property
    def _zp_mag_unit_unparsed(self):
        """Compatibility alias for unparsed mag-unit flag."""
        return self.zero_point._zp_mag_unit_unparsed

    @_zp_mag_unit_unparsed.setter
    def _zp_mag_unit_unparsed(self, value):
        self.zero_point._zp_mag_unit_unparsed = value

    @property
    def zp_flux(self):
        """Gets or sets the zero-point flux.

        Returns:
            astropy.units.Quantity: The zero-point flux with its unit.
        """
        return self.zero_point.zp_flux

    @zp_flux.setter
    def zp_flux(self, value):
        self.zero_point.zp_flux = value

    @property
    def zp_mag(self):
        """Gets or sets the zero-point magnitude.

        Returns:
            astropy.units.Quantity: The zero-point magnitude with its unit.
        """
        return self.zero_point.zp_mag

    @zp_mag.setter
    def zp_mag(self, value):
        self.zero_point.zp_mag = value

    @property
    def mag_sys(self):
        """Gets or sets the magnitude system name.

        Returns:
            str: The magnitude system type.
        """
        return self.magnitude_system.type

    @mag_sys.setter
    def mag_sys(self, value):
        if not isinstance(value, str):
            raise ValueError("Magnitude system must be a string.")
        self.magnitude_system.type = value.strip()

    def mag_to_flux(self, mag):
        """Converts magnitude to flux via the nested ``zero_point``.

        Args:
            mag (astropy.units.Quantity): Magnitude values.

        Returns:
            astropy.units.Quantity: Flux values.
        """
        return self.zero_point.mag_to_flux(mag)

    def flux_to_mag(self, flux):
        """Converts flux to magnitude via the nested ``zero_point``.

        Args:
            flux (astropy.units.Quantity): Flux values.

        Returns:
            astropy.units.Quantity: Magnitude values.
        """
        return self.zero_point.flux_to_mag(flux)

    def mag_err_to_flux_err(self, mag, mag_err):
        """Propagates magnitude uncertainties to flux via ``zero_point``.

        Args:
            mag (astropy.units.Quantity): Magnitude values.
            mag_err (astropy.units.Quantity): Magnitude uncertainties.

        Returns:
            astropy.units.Quantity: Flux uncertainties.
        """
        return self.zero_point.mag_err_to_flux_err(mag, mag_err)

    def flux_err_to_mag_err(self, flux, flux_err):
        """Propagates flux uncertainties to magnitude via ``zero_point``.

        Args:
            flux (astropy.units.Quantity): Flux values.
            flux_err (astropy.units.Quantity): Flux uncertainties.

        Returns:
            astropy.units.Quantity: Magnitude uncertainties.
        """
        return self.zero_point.flux_err_to_mag_err(flux, flux_err)

    def __repr__(self):
        return (
            f"<PhotCal zeroPoint={self.zero_point!r} "
            f"magnitudeSystem={self.magnitude_system!r}>"
        )


class PhotDM:
    """Photometry Data Model (photDM) hub combining filter and calibration information.

    Acts as a container linking a specific `PhotometryFilter` and its `PhotCal` calibration.
    """

    def __init__(self, photcal: PhotCal = None, photometry_filter: PhotometryFilter = None):
        """Initialises a PhotDM instance.

        Args:
            photcal (PhotCal, optional): The photometric calibration zero points.
                Defaults to None.
            photometry_filter (PhotometryFilter, optional): The photometric filter
                associated with this data model. Defaults to None.
        """
        self.photcal = photcal
        self.filter = photometry_filter
        # todo: blah-blah-blah

    @property
    def filter_id(self):
        """Gets or sets the filter identifier shortcut.

        Returns:
            str: The identifier of the filter, or "Unknown" if no filter is set.
        """
        return self.filter.filter_id if self.filter else "Unknown"

    @filter_id.setter
    def filter_id(self, value):
        self.filter.filter_id = value

    @property
    def filter_name(self):
        """Gets or sets the human-readable filter name on the nested filter.

        Returns:
            str or None: ``PhotometryFilter.name``, or None if no filter is set.
        """
        return self.filter.name if self.filter else None

    @filter_name.setter
    def filter_name(self, value):
        if self.filter is None:
            self.filter = PhotometryFilter(name=value)
        else:
            self.filter.name = value

    @property
    def mag0(self):
        """Gets or sets the zero-point magnitude shortcut.

        Returns:
            astropy.units.Quantity or float: The zero-point magnitude of the calibration,
                or 0.0 if no calibration is set.
        """
        return self.photcal.zp_mag if self.photcal else 0.0

    @mag0.setter
    def mag0(self, value):
        self.photcal.zp_mag = value

    def __repr__(self):
        # f_id = self.filter_id
        # sys = self.photcal.mag_sys if self.photcal else "None"
        # return f"<PhotDM Filter={f_id} Sys={sys}>"
        return f"<PhotDM Filter={self.filter} photcal={self.photcal}>"


class CooSys:
    """Represents a coordinate system reference frame for spatial coordinates.

    Used to model VOTable ``<COOSYS>`` metadata (reference frame and epoch).
    """

    def __init__(self, *, epoch=None, system='ICRS', coosys_id='system'):
        """Initialises a CooSys instance.

        Args:
            epoch (float, int, or str, optional): Reference epoch of the coordinates
                (e.g. proper-motion epoch for Gaia). ``None`` when the source VOTable
                omits ``COOSYS/@epoch``.
            system (str, optional): Reference coordinate system/frame. Defaults to ``ICRS``.
            coosys_id (str, optional): VOTable ``COOSYS/@ID`` value. Defaults to ``system``.
        """
        self.epoch = epoch
        self.system = system
        self.coosys_id = coosys_id

    def __repr__(self):
        return f"<CooSys id={self.coosys_id} epoch={self.epoch}: system={self.system}>"


class TimeSys:
    """Represents a time system standard for astronomical timing metadata.

    Captures reference position, origin (such as JD0/MJD0 offset), and timescale.
    """

    def __init__(self, refposition='HELIOCENTER', timeorigin=0.0, timescale='UTC'):
        """Initialises a TimeSys instance.

        Args:
            refposition (str, optional): Time reference position (e.g., 'HELIOCENTER',
                'BARYCENTER'). Defaults to 'HELIOCENTER'.
            timeorigin (float, optional): The origin offset of the time scale, i.e.,
                the JD0 or MJD0 value. Defaults to 0.0.
            timescale (str, optional): The timing scale used (e.g., 'UTC', 'TDB', 'TCB').
                Defaults to 'UTC'.
        """
        self.refposition = refposition  # HELIOCENTER OR BARYCENTER ...
        self.timeorigin = timeorigin  # JD0, f.e. 2400000.5
        self.timescale = timescale  # UTC, TCB, TBD etc

    @property
    def jd0(self):
        """Gets the reference time origin (JD0 offset).

        Returns:
            float: The reference time origin value.
        """
        return self.timeorigin

    def __repr__(self):
        return f"<TimeSys: timescale={self.timescale} refposition={self.refposition} timeorigin={self.timeorigin}>"


def _apply_gavo_votable_metadata(volc_instance, gavo_tree) -> None:
    """Populates TIMESYS, COOSYS, and PhotDM metadata from a GAVO VOTable tree.

    Args:
        volc_instance (VOLightCurve): Target instance to mutate in place.
        gavo_tree: GAVO stanxml root from ``_gavo_votable_metadata_tree``.
    """
    from volightcurve.time_reference import extract_timesys_metadata_from_gavo

    ts_meta = extract_timesys_metadata_from_gavo(gavo_tree)
    volc_instance.timesys_by_id = ts_meta.registry
    volc_instance.field_timesys_ref = ts_meta.field_refs
    volc_instance.param_timesys_ref = ts_meta.param_refs
    volc_instance.timesys = ts_meta.default_timesys
    volc_instance.coosys = extract_coosys(gavo_tree)
    volc_instance.photdms = extract_photdm(gavo_tree)


def _normalize_photdm_column_key(name: str) -> str:
    """Normalises a photometry column name for ``photdms`` lookup.

    Args:
        name (str): VOTable ``FIELD/@name`` or ``FIELDref/@ref`` value.

    Returns:
        str: Lowercase column key.
    """
    return str(name).strip().lower()


def _link_photdm_field_refs(tree, photcal_groups: dict[str, PhotDM], dm_map: dict[str, PhotDM]) -> None:
    """Links ``photcal`` GROUP instances to TABLE fields via ``FIELD/@ref``.

    Gaia and other archives attach ``FIELD/@ref`` to a ``photcal`` GROUP ``@ID``
    instead of embedding ``FIELDref`` inside the GROUP (UPJS/DaCHS style).

    Args:
        tree: GAVO stanxml root from ``votparse.parse``.
        photcal_groups (dict): ``photcal`` GROUP ``ID`` to ``PhotDM`` instances.
        dm_map (dict): Target mapping mutated in place (column name → ``PhotDM``).
    """

    def walk(node, text, attrs, childIter):
        if node.name_ == "TABLE":
            for child in childIter:
                if child.name_ != "FIELD":
                    continue
                field_name = getattr(child, "name", None)
                group_ref = getattr(child, "ref", None)
                if not field_name or not group_ref:
                    continue
                photdm = photcal_groups.get(str(group_ref))
                if photdm is None:
                    continue
                key = _normalize_photdm_column_key(field_name)
                dm_map.setdefault(key, photdm)
            return
        for child in childIter:
            if hasattr(child, "apply"):
                child.apply(walk)

    tree.apply(walk)


def extract_photdm(tree):
    """GAVO tree walker for IVOA PhotDM ``photcal`` GROUP metadata.

    Traverses a GAVO-parsed VOTable tree and resolves photometric calibration
    groups linked to photometry columns. Handles inline ``PARAM`` entries and
    ``PARAMref`` indirection (the pattern used by modern VO publishers such as
    DaCHS/UPJS TAP services), plus ``FIELD/@ref`` links to ``photcal`` GROUP
    ``@ID`` values (Gaia DR3 ARI epoch photometry).

    Args:
        tree: The GAVO VOTable tree node/element to parse.

    Returns:
        dict: A mapping of target column references (strings) to PhotDM instances.
    """
    id_map = {}

    def map_ids(node, text, attrs, childIter):
        node_id = getattr(node, "id", None) or getattr(node, "ID", None)
        if node_id:
            id_map[node_id] = node
        for child in childIter:
            if hasattr(child, "apply"):
                child.apply(map_ids)

    tree.apply(map_ids)

    dm_map = {}
    photcal_groups: dict[str, PhotDM] = {}

    UT_FLUX = "photDM:PhotCal.zeroPoint.flux.value"
    UT_MAG = "photDM:PhotCal.zeroPoint.referenceMagnitude.value"
    UT_MAG_SYS = "photDM:PhotCal.magnitudeSystem.type"
    UT_FILTER = "photDM:PhotometryFilter.identifier"
    UT_FILTER_NAME = "photDM:PhotometryFilter.name"
    UT_FILTER_SPEC = "photDM:PhotometryFilter.spectralLocation.value"

    def process_group(node, text, attrs, childIter):
        if node.name_ == "GROUP" and getattr(node, "name", None) == "photcal":
            cal_params = {
                "zp_flux": 1.0,
                "zp_mag": 0.0,
                "zp_mag_unit": "mag",
                "zp_flux_unit": None,
            }
            filter_params = {
                "filter_id": "",
                "name": None,
                "spectral_location": 0.0,
                "spectral_location_unit": None,
            }
            target_col = None
            group_id = getattr(node, "id", None) or getattr(node, "ID", None)
            for child in childIter:
                target_param = None
                role_utype = (getattr(child, "utype", None) or "").lower()

                if child.name_ == "PARAM":
                    target_param = child
                elif child.name_ == "PARAMref":
                    target_param = id_map.get(getattr(child, "ref", None))
                elif child.name_ == "FIELDref":
                    target_col = getattr(child, "ref", None)

                if target_param is not None:
                    ut = role_utype or (getattr(target_param, "utype", None) or "").lower()

                    if ut == UT_FLUX.lower():
                        cal_params["zp_flux"] = float(target_param.value)
                        cal_params["zp_flux_unit"] = to_internal(
                            getattr(target_param, "unit", None)
                        )
                    elif ut == UT_MAG.lower():
                        cal_params["zp_mag"] = float(target_param.value)
                        cal_params["zp_mag_unit"] = target_param.unit
                    elif ut == UT_MAG_SYS.lower():
                        cal_params["mag_sys"] = target_param.value
                    elif ut == UT_FILTER.lower():
                        filter_params["filter_id"] = target_param.value
                    elif ut == UT_FILTER_NAME.lower():
                        filter_params["name"] = target_param.value
                    elif (
                        not ut
                        and (getattr(target_param, "name", None) or "").lower()
                        in ("filter", "filtername")
                    ):
                        if not filter_params.get("name"):
                            filter_params["name"] = target_param.value
                    elif ut == UT_FILTER_SPEC.lower():
                        filter_params["spectral_location"] = float(target_param.value)
                        filter_params["spectral_location_unit"] = getattr(
                            target_param, "unit", None
                        )

            phot_filter = PhotometryFilter(**filter_params)
            photcal = PhotCal(**cal_params, photometry_filter=phot_filter)
            photdm = PhotDM(photcal=photcal, photometry_filter=phot_filter)
            if group_id:
                photcal_groups[str(group_id)] = photdm
            if target_col:
                dm_map[_normalize_photdm_column_key(target_col)] = photdm

        for child in childIter:
            if hasattr(child, "apply"):
                child.apply(process_group)

    tree.apply(process_group)
    _link_photdm_field_refs(tree, photcal_groups, dm_map)
    return dm_map


def _list_votable_table_ids(payload: bytes) -> list[str | int]:
    """Lists Astropy VOTable table identifiers for a byte payload.

    Args:
        payload (bytes): Raw VOTable content.

    Returns:
        list: Table ``ID``, ``name``, or zero-based index for each embedded table.
    """
    import astropy.io.votable as vot

    vot_file = vot.parse(io.BytesIO(payload))
    table_ids: list[str | int] = []
    for index, table in enumerate(vot_file.iter_tables()):
        table_ids.append(table.ID or table.name or index)
    return table_ids


def _read_votable_payload(file_path) -> bytes:
    """Reads a VOTable path or stream into bytes for a single-pass ingest.

    Args:
        file_path (str or file-like): Path or binary stream.

    Returns:
        bytes: Raw VOTable content.
    """
    if hasattr(file_path, "read"):
        if hasattr(file_path, "seek"):
            file_path.seek(0)
        return file_path.read()
    with open(file_path, "rb") as handle:
        return handle.read()


def _gavo_votable_metadata_tree(payload: bytes):
    """Builds a GAVO ``VOTABLE`` tree for metadata walkers without row decode.

    ``votparse.parse`` always yields ``tableparser.Rows`` at ``<DATA>``; that
    iterator must not be consumed (no ``list(rows)``), which avoids BINARY decode
    failures while preserving PARAMref resolution in photcal GROUPs.

    Args:
        payload (bytes): Raw VOTable bytes.

    Returns:
        GAVO ``V.VOTABLE`` root for ``extract_photdm`` / TIMESYS walkers.

    Raises:
        ValueError: When parsing completes without a ``VOTABLE`` root node.
    """
    votable_node = None
    for element in votparse.parse(io.BytesIO(payload), {V.VOTABLE}):
        if isinstance(element, tableparser.Rows):
            continue
        if type(element).__name__ == "VOTABLE":
            votable_node = element
    if votable_node is None:
        raise ValueError("GAVO metadata parse did not yield a VOTABLE root.")
    return votable_node


def _gavo_votable_tree_from_source(file_path):
    """Parses VOTable metadata into a GAVO stanxml tree (no row materialisation).

    Args:
        file_path (str or file-like): Path or stream containing VOTable bytes.

    Returns:
        GAVO stanxml tree root for metadata walkers.
    """
    return _gavo_votable_metadata_tree(_read_votable_payload(file_path))


def extract_timesys(tree):
    """GAVO tree walker for the default ``TIMESYS`` metadata block.

    Args:
        tree: The GAVO VOTable tree node/element to parse.

    Returns:
        TimeSys: The first ``TIMESYS`` element in document order, or an empty default.
    """
    from volightcurve.time_reference import extract_timesys_metadata_from_gavo

    return extract_timesys_metadata_from_gavo(tree).default_timesys


def extract_coosys(tree):
    """GAVO tree walker for IVOA ``COOSYS`` metadata.

    Preserves absent ``@epoch`` attributes as ``None`` rather than inventing a
    default proper-motion epoch.

    Args:
        tree: The GAVO VOTable tree node/element to parse.

    Returns:
        CooSys or None: Populated coordinate-system metadata when present.
    """
    data = {"coosys_id": "system", "epoch": None, "system": None}

    def find_cs(node, text, attrs, childIter):
        if node.name_ == "COOSYS":
            data["coosys_id"] = attrs.get("ID", "system")
            data["system"] = attrs.get("system")
            epoch = attrs.get("epoch")
            if epoch is not None:
                try:
                    data["epoch"] = float(epoch)
                except (TypeError, ValueError):
                    data["epoch"] = epoch
        for child in childIter:
            if hasattr(child, "apply"):
                child.apply(find_cs)

    tree.apply(find_cs)
    if not data["system"]:
        return None
    return CooSys(
        coosys_id=data["coosys_id"],
        system=data["system"],
        epoch=data["epoch"],
    )


def find_columns_by_ucd(table, ucd_fragment):
    """Retrieves all column names from an Astropy table that contain the specified UCD fragment.

    Args:
        table (astropy.table.Table): The table to search.
        ucd_fragment (str): The Unified Content Descriptor (UCD) substring/fragment to look for.

    Returns:
        list of str: A list of column names containing the given UCD fragment in their metadata.
    """
    matches = []
    for colname in table.colnames:
        col_ucd = table[colname].info.meta.get('ucd', '') if table[colname].info.meta else ''
        if ucd_fragment in col_ucd:
            matches.append(colname)
    return matches


def get_time_colnames(table):
    """Retrieves names of columns containing time epoch data from the table.

    Args:
        table (astropy.table.Table): The table to search.

    Returns:
        list of str: Column names matching time epoch UCDs (e.g., 'time.epoch').
    """
    return find_columns_by_ucd(table, 'time.epoch')


def get_mag_colnames(table):
    """Retrieves names of columns containing primary magnitudes from the table, excluding error columns.

    Args:
        table (astropy.table.Table): The table to search.

    Returns:
        list of str: Column names matching magnitude UCDs (e.g., 'phot.mag') that do not represent errors.
    """
    # Returns primary magnitudes, excludes errors
    all_mags = find_columns_by_ucd(table, 'phot.mag')
    return [c for c in all_mags if 'stat.error' not in table[c].info.meta.get('ucd', '')]


def get_flux_colnames(table):
    """Retrieves names of columns containing primary fluxes from the table, excluding error columns.

    Args:
        table (astropy.table.Table): The table to search.

    Returns:
        list of str: Column names matching flux UCDs (e.g., 'phot.flux') that do not represent errors.
    """
    all_flux = find_columns_by_ucd(table, 'phot.flux')
    return [c for c in all_flux if 'stat.error' not in table[c].info.meta.get('ucd', '')]


def is_mag_column(table: Table, colname: str | None):
    """Checks whether the specified column in the table is a magnitude column.

    Args:
        table (astropy.table.Table): The table containing the column.
        colname (str, optional): The column name to check.

    Returns:
        bool: True if the column is a magnitude column, False otherwise or if colname is None.
    """
    if colname is None: return False
    return 'phot.mag' in table[colname].info.meta.get('ucd', '')


def is_flux_column(table: Table, colname: str | None):
    """Checks whether the specified column in the table is a flux column.

    Args:
        table (astropy.table.Table): The table containing the column.
        colname (str, optional): The column name to check.

    Returns:
        bool: True if the column is a flux column, False otherwise or if colname is None.
    """
    if colname is None: return False
    return 'phot.flux' in table[colname].info.meta.get('ucd', '')


LABEL_UCD_FRAGMENTS = ("meta.code", "meta.id")
DEFAULT_LABEL_UCD = "meta.id"


def _column_ucd(table: Table, colname: str) -> str:
    """Returns the UCD string stored on a column, or an empty string.

    Args:
        table (astropy.table.Table): Source table.
        colname (str): Column name.

    Returns:
        str: UCD text from ``info.meta``, or ``""``.
    """
    meta = table[colname].info.meta
    if not meta:
        return ""
    return str(meta.get("ucd") or "")


def column_main_role(table: Table, colname: str) -> str | None:
    """Classifies a column as time, magnitude, flux, or a matching error.

    Uses UCD fragments only. Columns without a science UCD return ``None``.

    Args:
        table (astropy.table.Table): Source table.
        colname (str): Column name.

    Returns:
        str or None: ``time``, ``mag``, ``flux``, ``mag_err``, ``flux_err``, or None.
    """
    ucd = _column_ucd(table, colname)
    if "time.epoch" in ucd:
        return "time"
    if "stat.error" in ucd and "phot.mag" in ucd:
        return "mag_err"
    if "stat.error" in ucd and "phot.flux" in ucd:
        return "flux_err"
    if "phot.mag" in ucd:
        return "mag"
    if "phot.flux" in ucd:
        return "flux"
    return None


def get_label_colnames(table: Table) -> list[str]:
    """Returns the designated per-epoch label column, if any.

    Prefers the leftmost column whose UCD contains ``meta.code`` or
    ``meta.id``. Otherwise returns the leftmost column that is not time,
    magnitude, flux, or an error. See ``docs/io_contract.md`` §8.

    Args:
        table (astropy.table.Table): Product or file table.

    Returns:
        list of str: Zero or one column name.
    """
    labelled = [
        name
        for name in table.colnames
        if any(fragment in _column_ucd(table, name) for fragment in LABEL_UCD_FRAGMENTS)
    ]
    if labelled:
        return [labelled[0]]
    for name in table.colnames:
        if column_main_role(table, name) is None:
            return [name]
    return []


def assign_label_column_ucd(table: Table) -> None:
    """Stores a label-role UCD on the designated label column when missing.

    Does not rename the column. If the column already has a UCD containing
    ``meta.code`` or ``meta.id``, it is left unchanged. Otherwise the UCD is
    set to ``meta.id``. See ``docs/io_contract.md`` §8.

    Args:
        table (astropy.table.Table): Product table after role promotion.
    """
    names = get_label_colnames(table)
    if not names:
        return
    name = names[0]
    ucd = _column_ucd(table, name)
    if any(fragment in ucd for fragment in LABEL_UCD_FRAGMENTS):
        return
    column = table[name]
    if column.info.meta is None:
        column.info.meta = {}
    column.info.meta["ucd"] = DEFAULT_LABEL_UCD


def get_error_colnames(table, base_ucd=None):
    """Retrieves names of columns containing statistical errors from the table.

    If a base UCD is specified, narrows down to error columns associated with that base UCD.

    Args:
        table (astropy.table.Table): The table to search.
        base_ucd (str, optional): The base UCD (e.g., 'phot.mag', 'phot.flux') to filter errors by.
            Defaults to None.

    Returns:
        list of str: Column names matching the statistical error UCD and optional base UCD.
    """
    errors = find_columns_by_ucd(table, 'stat.error')
    if base_ucd:
        return [c for c in errors if base_ucd in table[c].info.meta.get('ucd', '')]
    return errors


def is_magnitude_phot_column(table: Table, colname: str = "phot") -> bool:
    """Detects whether a ``phot`` column holds magnitudes rather than flux.

    Uses column unit and UCD metadata following VO photometry conventions.
    When unit and UCD are both absent (ambiguous non-VO ``phot``), defaults to
    **magnitude** (Ticket 8 Phase 2b).

    Args:
        table (astropy.table.Table): Source table.
        colname (str): Photometry column name.

    Returns:
        bool: True when the column represents magnitudes.
    """
    if colname not in table.colnames:
        return False
    col = table[colname]
    if col.unit is not None:
        try:
            if col.unit.is_equivalent(u.mag):
                col.unit.to(u.mag)
                return True
        except (u.UnitsError, u.UnitTypeError, TypeError, ValueError):
            pass
    ucd = (col.info.meta or {}).get("ucd", "")
    if "phot.mag" in ucd:
        return True
    if "phot.flux" in ucd:
        return False
    # Non-magnitude unit present (e.g. Jy) ⇒ flux.
    if col.unit is not None:
        return False
    # Ambiguous bare ``phot`` (no unit, no UCD) ⇒ magnitude.
    return True


def assign_photometry_column_semantics(
    table: Table,
    phot_col: str = "phot",
    error_col: str | None = "flux_error",
    *,
    force_magnitude: bool | None = None,
) -> Table:
    """Assigns VO UCD metadata to generic ``phot`` and error columns.

    Args:
        table (astropy.table.Table): Export or upload table to annotate.
        phot_col (str): Primary photometry column name.
        error_col (str, optional): Uncertainty column name.
        force_magnitude (bool, optional): Override automatic domain detection.

    Returns:
        astropy.table.Table: The same table with UCD metadata updated in place.
    """
    if phot_col not in table.colnames:
        return table
    is_mag = force_magnitude if force_magnitude is not None else is_magnitude_phot_column(table, phot_col)
    phot_ucd = "phot.mag" if is_mag else "phot.flux;em.opt"
    err_ucd = "stat.error;phot.mag" if is_mag else "stat.error;phot.flux;em.opt"
    if table[phot_col].info.meta is None:
        table[phot_col].info.meta = {}
    table[phot_col].info.meta["ucd"] = phot_ucd
    if error_col and error_col in table.colnames:
        if table[error_col].info.meta is None:
            table[error_col].info.meta = {}
        table[error_col].info.meta["ucd"] = err_ucd
    return table


def resolve_votable_phot_field_labels(table: Table, phot_col: str = "phot", error_col: str = "flux_error") -> dict:
    """Returns VOTable field UCDs and descriptions for a ``phot`` column.

    Args:
        table (astropy.table.Table): Export table containing ``phot``.
        phot_col (str): Photometry column name.
        error_col (str): Uncertainty column name.

    Returns:
        dict: Keys ``phot_ucd``, ``error_ucd``, ``phot_description``, ``error_description``.
    """
    is_mag = is_magnitude_phot_column(table, phot_col)
    if is_mag:
        return {
            "phot_ucd": "phot.mag",
            "error_ucd": "stat.error;phot.mag",
            "phot_description": "Photometry (magnitude)",
            "error_description": "Statistical uncertainty of magnitude",
        }
    return {
        "phot_ucd": "phot.flux;em.opt",
        "error_ucd": "stat.error;phot.flux;em.opt",
        "phot_description": "Photometry (flux)",
        "error_description": "Statistical uncertainty of flux",
    }


def _promote_to_vo_standards(table):
    """Heuristically assigns Unified Content Descriptors (UCDs) and physical units to table columns.

    This function scans column names for common patterns (e.g., 'mag', 'flux', 'time')
    and assigns appropriate Astropy units and UCD metadata values without renaming columns.

    Args:
        table (astropy.table.Table): The table to process.

    Returns:
        astropy.table.Table: The table with updated column units and UCD metadata.
    """
    for colname in table.colnames:
        col = table[colname]
        name_low = colname.lower()
        if not col.unit:
            if 'mag' in name_low:
                col.unit = u.mag
            elif 'flux' in name_low:
                # col.unit = u.Jy  # default assumption
                col.unit = 'electron s-1'  # default assumption
            elif any(k in name_low for k in ['time', 'jd', 'mjd']):
                col.unit = u.d

        if not col.info.meta or not col.info.meta.get('ucd'):
            if col.info.meta is None:
                col.info.meta = {}
            if name_low == 'phot':
                is_mag_phot = is_magnitude_phot_column(table, colname)
                ucd = 'phot.mag' if is_mag_phot else 'phot.flux;em.opt'
                if is_mag_phot and not col.unit:
                    col.unit = u.mag
            elif 'mag' in name_low:
                ucd = 'phot.mag'
            elif 'flux' in name_low:
                ucd = 'phot.flux'
            elif any(k in name_low for k in ['time', 'jd', 'mjd']):
                ucd = 'time.epoch'
            else:
                continue

            if any(k in name_low for k in ['err', 'uncert', 'sigma']) and name_low != 'phot':
                ucd = f"stat.error;{ucd}"
            col.info.meta['ucd'] = ucd
    return table


def _pickup_jd0_from_table(table):
    """Parses metadata comments to locate the time origin value (JD0).

    Searches the table comments for patterns like "JD0 = value".

    Args:
        table (astropy.table.Table): The table to scan.

    Returns:
        float: The parsed JD0 value if found, or 0.0 otherwise.
    """
    jd0_pattern = re.compile(r"JD0\s*=\s*([+-]?\d*\.?\d+)")
    for line in table.meta.get('comments', []):
        match = jd0_pattern.search(line.upper())
        if match: return float(match.group(1))
    return 0.0


def _pickup_mag0_from_table(table):
    """Parses metadata comments to locate the reference magnitude zero point (MAG0).

    Searches the table comments for patterns like "MAG0 = value".

    Args:
        table (astropy.table.Table): The table to scan.

    Returns:
        float: The parsed MAG0 value if found, or 0.0 otherwise.
    """
    jd0_pattern = re.compile(r"MAG0\s*=\s*([+-]?\d*\.?\d+)")
    for line in table.meta.get('comments', []):
        match = jd0_pattern.search(line.upper())
        if match: return float(match.group(1))
    return 0.0


def _pickup_period_from_table(table):
    """Parses ``PERIOD=`` from comment metadata (folding period in days).

    Args:
        table (astropy.table.Table): The table to scan.

    Returns:
        float or None: Period in days if a matching comment line exists.
    """
    pattern = re.compile(
        r"PERIOD\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)",
        re.IGNORECASE,
    )
    for line in table.meta.get("comments", []):
        match = pattern.search(line)
        if match:
            return float(match.group(1))
    return None


def _pickup_epoch_from_table(table):
    """Parses ``EPOCH=`` from comment metadata (same time scale as the time column).

    The value is stored in ``table.meta['epoch']`` and converted to absolute JD when
    packing transport JSON (respecting ``JD0`` / TIMESYS, same as VOTable PARAMs).

    Args:
        table (astropy.table.Table): The table to scan.

    Returns:
        float or None: Epoch coordinate if a matching comment line exists.
    """
    pattern = re.compile(
        r"EPOCH\s*=\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)",
        re.IGNORECASE,
    )
    for line in table.meta.get("comments", []):
        match = pattern.search(line)
        if match:
            return float(match.group(1))
    return None


_DAT_METADATA_COMMENT = re.compile(
    r"(?:JD0|MAG0|PERIOD|EPOCH|FILTER|BAND|ZP_FLUX|ZP_MAG|MAG_SYS)\s*=",
    re.IGNORECASE,
)


def _is_dat_metadata_comment_line(line: str) -> bool:
    """Return True when a ``#`` comment line carries ``KEY=value`` metadata, not column names.

    Args:
        line (str): One entry from ``table.meta['comments']`` (without leading ``#``).

    Returns:
        bool: True for first-class calibration assignments.
    """
    from volightcurve.io_keywords import is_metadata_assignment_line

    text = line.strip().lstrip("#").strip()
    return is_metadata_assignment_line(text) or bool(_DAT_METADATA_COMMENT.search(text))


def _pickup_filter_from_table(table):
    """Scans the table comments for filter/band identification.

    Matches comments containing patterns like: FILTER=Gaia/GAIA3.G or BAND = r.

    Args:
        table (astropy.table.Table): The table to scan.

    Returns:
        str or None: Filter or band identifier if found.
    """
    filter_pattern = re.compile(
        r"(?:FILTER|BAND)\s*=\s*(\S+)",
        re.IGNORECASE,
    )

    for line in table.meta.get("comments", []):
        match = filter_pattern.search(line)
        if match:
            return match.group(1).strip()

    return None


def _recover_lc_colnames(table):
    """Strictly renames columns for unlabelled 'colN' tables based on comments or positional fallback.

    Uses the same header-selection rules as ``read_dat_table`` (``docs/io_contract.md``
    §4b). Otherwise applies a rigid positional fallback: column 1 becomes
    ``obs_time``, column 2 ``mag``, column 3 ``mag_err``.

    Args:
        table (astropy.table.Table): The table whose columns are to be renamed.

    Returns:
        astropy.table.Table: The renamed table.
    """
    from volightcurve.io_dat import resolve_dat_column_names, select_dat_header_names

    comments = table.meta.get('comments', []) or []
    num_cols = len(table.colnames)
    generic_cols = any(
        re.match(r"^col\d+$", name, re.IGNORECASE) for name in table.colnames
    )
    broken_header = "=" in table.colnames
    if not generic_cols and not broken_header:
        return table

    found_header = select_dat_header_names(comments, num_cols)
    if found_header is not None:
        found_header, _ = resolve_dat_column_names(
            found_header, num_cols, first_data_line_no=0
        )

    # Apply names
    for i, colname in enumerate(table.colnames):
        if found_header:
            new_name = found_header[i]
        elif generic_cols or broken_header:
            # RIGID POSITIONAL FALLBACK
            if i == 0:
                new_name = 'obs_time'
            elif i == 1:
                new_name = 'mag'
            elif i == 2:
                new_name = 'mag_err'
            else:
                new_name = f'col{i + 1}'
        else:
            continue

        if colname != new_name:
            table.rename_column(colname, new_name)
    return table


def apply_non_votable_heuristics(volc: "VOLightCurve") -> None:
    """Apply column promotion and comment/flat-meta calibration for non-VOTable tables.

    Mutates ``volc.table``, ``volc.timesys``, and ``volc.photdms`` in place.
    Uses the shared Ticket 8 vocabulary (``JD0``, ``ZP_*``, ``FILTER``, …) from
    ``#`` comments and flat ECSV ``meta`` keys. Legacy ``MAG0`` alone still implies
    instrumental ``ZP_FLUX = 1`` dimensionless on ingest only.

    Args:
        volc (VOLightCurve): Parsed instance with ``table`` already assigned.
    """
    from volightcurve.io_keywords import (
        KEY_EFFECTIVE_WAVELENGTH,
        KEY_EFFECTIVE_WAVELENGTH_UNIT,
        KEY_EPOCH,
        KEY_FILTER,
        KEY_FILTER_NAME,
        KEY_JD0,
        KEY_MAG_SYS,
        KEY_PERIOD,
        KEY_ZP_FLUX,
        KEY_ZP_FLUX_UNIT,
        KEY_ZP_MAG,
        KEY_ZP_MAG_UNIT,
    )
    from volightcurve.io_meta import merge_calibration_sources

    volc.table = _recover_lc_colnames(volc.table)
    volc.table = _promote_to_vo_standards(volc.table)
    if volc.table.meta is None:
        volc.table.meta = {}

    calibration = merge_calibration_sources(
        (volc.table.meta or {}).get("comments"),
        volc.table.meta,
    )

    if KEY_JD0 in calibration:
        volc.timesys.timeorigin = float(calibration[KEY_JD0])
    else:
        volc.timesys.timeorigin = _pickup_jd0_from_table(volc.table)

    if KEY_PERIOD in calibration:
        volc.table.meta["period"] = float(calibration[KEY_PERIOD])
    else:
        period = _pickup_period_from_table(volc.table)
        if period is not None:
            volc.table.meta["period"] = period

    if KEY_EPOCH in calibration:
        volc.table.meta["epoch"] = float(calibration[KEY_EPOCH])
    else:
        epoch = _pickup_epoch_from_table(volc.table)
        if epoch is not None:
            volc.table.meta["epoch"] = epoch

    heur_filter_id = calibration.get(KEY_FILTER) or _pickup_filter_from_table(volc.table)
    if heur_filter_id:
        volc.table.meta["filter"] = heur_filter_id
    if KEY_FILTER_NAME in calibration:
        volc.table.meta["filter_name"] = str(calibration[KEY_FILTER_NAME])

    has_zp_mag = KEY_ZP_MAG in calibration
    has_zp_flux = KEY_ZP_FLUX in calibration
    # Mag-only keyword (MAG0 or ZP_MAG without ZP_FLUX) → instrumental flux ZP 1.
    mag_only = has_zp_mag and not has_zp_flux
    build_photcal = (has_zp_mag and has_zp_flux) or mag_only
    if build_photcal:
        zp_mag = float(calibration[KEY_ZP_MAG])
        zp_mag_unit = calibration.get(KEY_ZP_MAG_UNIT) or "mag"
        mag_sys = calibration.get(KEY_MAG_SYS) or "Vega"
        if has_zp_flux:
            zp_flux = float(calibration[KEY_ZP_FLUX])
            zp_flux_unit = calibration.get(KEY_ZP_FLUX_UNIT)
        else:
            zp_flux = 1.0
            zp_flux_unit = None
    else:
        zp_mag = zp_flux = zp_flux_unit = zp_mag_unit = mag_sys = None

    spectral_location = calibration.get(KEY_EFFECTIVE_WAVELENGTH)
    spectral_unit = calibration.get(KEY_EFFECTIVE_WAVELENGTH_UNIT)

    for colname in volc.get_flux_colnames() + volc.get_mag_colnames():
        photdm = volc.photdms.get(colname, None)
        heur_filter_name = None
        if KEY_FILTER_NAME in calibration:
            heur_filter_name = str(calibration[KEY_FILTER_NAME])
        new_filter = PhotometryFilter(
            filter_id=heur_filter_id,
            name=heur_filter_name,
            spectral_location=spectral_location,
            spectral_location_unit=spectral_unit,
        )
        photcal = None
        if build_photcal:
            photcal = PhotCal(
                zp_flux=zp_flux,
                zp_flux_unit=zp_flux_unit,
                zp_mag=zp_mag,
                zp_mag_unit=zp_mag_unit,
                mag_sys=mag_sys,
                photometry_filter=new_filter,
            )

        if photdm is None:
            volc.photdms[colname] = PhotDM(photcal=photcal, photometry_filter=new_filter)
        else:
            if heur_filter_id and (
                photdm.filter_id is None or photdm.filter_id == "Unknown"
            ):
                photdm.filter_id = heur_filter_id
            if heur_filter_name and not photdm.filter_name:
                photdm.filter_name = heur_filter_name
            if build_photcal:
                if photdm.photcal is None:
                    photdm.photcal = photcal
                else:
                    if has_zp_mag:
                        photdm.mag0 = zp_mag
                    if has_zp_flux:
                        photdm.photcal.zp_flux = zp_flux

    assign_label_column_ucd(volc.table)


def _mag0_declared_in_comments(table) -> bool:
    """Return True when a ``MAG0=`` assignment appears in table comment metadata.

    Args:
        table (astropy.table.Table): Table with optional ``meta['comments']``.

    Returns:
        bool: True when legacy ``MAG0`` is present in comments.
    """
    pattern = re.compile(r"MAG0\s*=\s*([+-]?\d*\.?\d+)")
    for line in table.meta.get("comments", []):
        if pattern.search(line.upper()):
            return True
    return False


class VOLightCurve:
    """Represents a Virtual Observatory (VO) lightcurve container.

    Encapulates an Astropy Table containing timing and photometric observations,
    along with associated coordinate system, time system, and photometric calibration
    metadata mapped to specific columns. It supports files in both VOTable and ASCII
    heuristic formats.
    """

    def __init__(self, file_path, *, table_id: str | int | None = None):
        """Initialises a VOLightCurve instance and ingests the specified data file.

        Args:
            file_path (str or file-like object): Path to the input file or an active file-like stream.
            table_id (str or int, optional): When the VOTable embeds multiple ``TABLE``
                elements, selects one by Astropy table ``ID``, name, or zero-based index.
        """
        self.file_path = file_path
        self.table = None
        self.timesys = TimeSys()
        self.timesys_by_id: dict[str, TimeSys] = {}
        self.field_timesys_ref: dict[str, str] = {}
        self.param_timesys_ref: dict[str, str | None] = {}
        self.coosys = None
        self.photdms = {}  # maps column_name -> PhotDM instance
        self._table_id = table_id

        self._ingest(file_path, table_id=table_id)

    def _ingest_votable(self, payload: bytes, *, table_id: str | int | None = None) -> None:
        """Ingests a confirmed VOTable byte payload into table and VO metadata.

        Table rows come from Astropy; TIMESYS, COOSYS, and PhotDM (including
        standard ``PARAMref`` resolution) come from GAVO metadata walkers on a
        parse that skips ``<DATA>`` row consumption.

        Args:
            payload (bytes): Raw VOTable content.
            table_id (str or int, optional): Selects one table when several are present.

        Raises:
            ValueError: When multiple tables are present and ``table_id`` is omitted.
        """
        import astropy.io.votable as vot

        table_ids = _list_votable_table_ids(payload)
        if len(table_ids) > 1 and table_id is None:
            names = ", ".join(str(item) for item in table_ids)
            raise ValueError(
                f"Multiple VOTable tables found ({names}). "
                "Pass table_id= to VOLightCurve to select one lightcurve table."
            )

        read_kwargs: dict = {"format": "votable"}
        if table_id is not None:
            read_kwargs["table_id"] = table_id
        self.table = Table.read(io.BytesIO(payload), **read_kwargs)

        astro_tree = vot.parse(io.BytesIO(payload))
        if table_id is not None:
            if isinstance(table_id, int):
                selected_table = astro_tree.get_table_by_index(table_id)
            else:
                selected_table = astro_tree.get_table_by_id(table_id)
        else:
            selected_table = astro_tree.get_first_table()
        for param in selected_table.params:
            if param.name:
                self.table.meta[param.name] = param.value

        gavo_tree = _gavo_votable_metadata_tree(payload)
        _apply_gavo_votable_metadata(self, gavo_tree)

        self.table = _promote_to_vo_standards(self.table)
        assign_label_column_ucd(self.table)
        if not self.timesys.timeorigin:
            self.timesys.timeorigin = _pickup_jd0_from_table(self.table)

    @classmethod
    def from_table(cls, table):
        """Build a ``VOLightCurve`` from an already parsed Astropy table.

        Applies the same non-VOTable heuristics as file ingest (column promotion,
        ``.dat`` comment metadata, PhotDM stubs).

        Args:
            table (astropy.table.Table): Tabular lightcurve data.

        Returns:
            VOLightCurve: Instance with ``table``, ``timesys``, and ``photdms`` populated.
        """
        instance = cls.__new__(cls)
        instance.file_path = None
        instance.table = table
        instance.timesys = TimeSys()
        instance.timesys_by_id = {}
        instance.field_timesys_ref = {}
        instance.param_timesys_ref = {}
        instance.coosys = None
        instance.photdms = {}
        instance._table_id = None
        apply_non_votable_heuristics(instance)
        return instance

    def _ingest(self, file_path, *, table_id: str | int | None = None):
        """Main ingestion flow that loads and processes the input file.

        VOTable products: one byte read, Astropy for table data and TABLE PARAM
        values, GAVO metadata parse (no BINARY row decode) for TIMESYS, COOSYS,
        and PhotDM including ``PARAMref``. Non-VOTable inputs fall back to generic
        tabular or heuristic ASCII parsing.

        Args:
            file_path (str or file-like object): Path to the input file or an active file-like stream.
            table_id (str or int, optional): Selects one embedded VOTable table.
        """
        if hasattr(file_path, "seek"):
            file_path.seek(0)

        if is_votable(file_path):
            self._ingest_votable(_read_votable_payload(file_path), table_id=table_id)
            return

        try:
            if hasattr(file_path, "seek"):
                file_path.seek(0)
            self.table = Table.read(file_path)
            logger.info(
                "Read %s with Table.read; file is not a VOTable.",
                file_path,
            )

        except Exception:
            logger.warning("Standard read failed, trying heuristic ASCII...")
            if hasattr(file_path, "seek"):
                file_path.seek(0)
            self.table = ascii.read(file_path)
            self.table = _recover_lc_colnames(self.table)

        apply_non_votable_heuristics(self)

    def __repr__(self):
        return f"<VOLightCurve: {len(self.table)} rows, {len(self.photdms)} PhotCals, jd0={self.jd0}>"

    def __getitem__(self, key):
        """Allows row/column indexing directly on the underlying Astropy Table.

        Args:
            key (str or int or slice): Key to access columns or rows in the table.

        Returns:
            astropy.table.Column or astropy.table.Row or astropy.table.Table: The requested data.
        """
        return self.table[key]

    def __getattr__(self, name):
        """Allows direct attribute access delegate to the underlying Astropy Table.

        This enables accessing table properties such as 'colnames', 'meta', 'row_groups', etc.
        directly on the VOLightCurve instance.

        Args:
            name (str): The name of the attribute.

        Returns:
            Any: The attribute value from the underlying table.

        Raises:
            AttributeError: If 'table' is not yet initialized or the attribute is not found.
        """
        # Avoid infinite recursion if table isn't initialized yet
        if name == "table":
            raise AttributeError("Table not yet initialized")

        try:
            return getattr(self.table, name)
        except AttributeError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __len__(self):
        """Returns the number of rows in the underlying table.

        Returns:
            int: The row count of the table.
        """
        return len(self.table)

    @property
    def jd0(self):
        """Gets the reference time origin (JD0) from the time system metadata.

        Returns:
            float: The reference time origin value.
        """
        # Yes, I realise that I ought to have a set of timesys connected to time columns,
        # but I'm too lazy to implement this (yet)
        return self.timesys.jd0

    def add_flux_column_from_mag(self, mag_col_name, new_col_name=None):
        """Generates and adds a new flux column converted from the specified magnitude column.

        Retrieves the associated PhotCal calibration and performs a unit-safe conversion to 
        physical flux density. If no calibration is available, fills the new column with NaN values.

        Args:
            mag_col_name (str): The name of the source magnitude column in the table.
            new_col_name (str, optional): The name for the newly created flux column. 
                Defaults to None (which yields 'flux_from_<mag_col_name>').

        Returns:
            str: The name of the newly added flux column.
        """

        colname_out = new_col_name or f"flux_from_{mag_col_name}"
        photdm = self.photdms.get(mag_col_name, None)

        if photdm is None or photdm.photcal is None:
            logger.warning(f'No PhotCal for {mag_col_name}. Filling {new_col_name} with None/NaN')
            flux = np.full(len(self.table), np.nan)
            self.table[new_col_name] = MaskedColumn(data=flux, mask=np.isnan(flux))
        else:
            # Unit safety:
            try:
                flux = photdm.photcal.mag_to_flux(self.table[mag_col_name])
            except (u.UnitConversionError, u.UnitTypeError, u.UnitsError, TypeError, ValueError) as e:
                logger.error(f"Unit conversion failed for {mag_col_name} [{self.table[mag_col_name].unit}]: {e}")
                data = np.full(len(self.table), np.nan)
                flux = MaskedColumn(data=data, mask=np.ones(len(data), dtype=bool))

            # Check physical equivalence: Flux Density is power/area/freq (or wavelength)
            ucd = 'phot.flux'
            if hasattr(flux, 'unit') and flux.unit is not None:
                u_flux_nu = u.W / (u.m ** 2 * u.Hz)
                u_flux_lam = u.W / (u.m ** 2 * u.m)
                is_flux_density = False
                for ref_unit in (u_flux_nu, u_flux_lam):
                    if flux.unit.is_equivalent(ref_unit):
                        flux.unit.to(ref_unit)
                        is_flux_density = True
                        break
                if is_flux_density:
                    ucd = 'phot.flux.density'

            self.table[colname_out] = flux
            self.table[colname_out].info.meta = {'ucd': ucd}

            # Link the same DataModel
            self.photdms[colname_out] = photdm

        return colname_out

    def add_mag_column_from_flux(self, flux_col_name, new_col_name='mag_from_flux'):
        """Generates and adds a new magnitude column converted from the specified flux column.

        Retrieves the associated PhotCal calibration and performs a unit-safe conversion to 
        astronomical magnitudes. If no calibration is available, fills the new column with NaN values.

        Args:
            flux_col_name (str): The name of the source flux column in the table.
            new_col_name (str, optional): The name for the newly created magnitude column. 
                Defaults to 'mag_from_flux' (or 'mag_from_<flux_col_name>' if none provided).

        Returns:
            str: The name of the newly added magnitude column.
        """
        colname_out = new_col_name or f"mag_from_{flux_col_name}"
        photdm = self.photdms.get(flux_col_name, None)

        if photdm is None or photdm.photcal is None:
            logger.warning(f'No PhotCal for {flux_col_name}. Filling {new_col_name} with None/NaN')
            mag = np.full(len(self.table), np.nan)
            self.table[new_col_name] = MaskedColumn(data=mag, mask=np.isnan(mag))
        else:
            # Unit safety:
            try:
                mag = photdm.photcal.flux_to_mag(self.table[flux_col_name])
            except (u.UnitConversionError, u.UnitTypeError, u.UnitsError, TypeError, ValueError) as e:
                logger.error(
                    f"Unit conversion failed for {flux_col_name} [{self.table[flux_col_name].unit}]: {e}")
                data = np.full(len(self.table), np.nan)
                mag = MaskedColumn(data=data, mask=np.ones(len(data), dtype=bool))

            # Check physical equivalence: Flux Density is power/area/freq (or wavelength)
            ucd = 'phot.mag'
            self.table[colname_out] = mag
            self.table[colname_out].info.meta = {'ucd': ucd}

            # Link the same DataModel
            self.photdms[colname_out] = photdm

        return colname_out

    def get_time_colnames(self):
        """Retrieves list of table column names containing time data.

        Returns:
            list of str: Column names associated with time coordinate standards.
        """
        return get_time_colnames(self.table)

    def get_mag_colnames(self):
        """Retrieves list of table column names containing primary magnitudes.

        Returns:
            list of str: Column names containing magnitude data.
        """
        return get_mag_colnames(self.table)

    def get_flux_colnames(self):
        """Retrieves list of table column names containing primary fluxes.

        Returns:
            list of str: Column names containing flux data.
        """
        return get_flux_colnames(self.table)

    def get_mag_error_colnames(self):
        """Retrieves list of table column names containing magnitude statistical errors.

        Returns:
            list of str: Column names representing statistical errors on magnitudes.
        """
        return get_error_colnames(self.table, base_ucd='phot.mag')

    def get_flux_error_colnames(self):
        """Retrieves list of table column names containing flux statistical errors.

        Returns:
            list of str: Column names representing statistical errors on fluxes.
        """
        return get_error_colnames(self.table, base_ucd='phot.flux')

    def get_label_colnames(self):
        """Returns the designated per-epoch label column, if any.

        Returns:
            list of str: Zero or one column name. See ``get_label_colnames``.
        """
        return get_label_colnames(self.table)

    def write_votable(
        self,
        output_stream_or_path,
        table_name: str,
        filter_identifier: str | None = None,
        refposition: str = "HELIOCENTER",
        timescale: str = "UTC", # could we reasonable presume this for old-fashioned handmade scripts?
        timeorigin: float = 0,
        votable_description: str | None = None,
        creator: str | None = None,
        zero_point_flux: float | None = None,
        # Internal: ``None`` = dimensionless (wire form is ``VO_DIMENSIONLESS_WIRE``).
        zero_point_flux_unit: str | None = None,
        zero_point_ref_mag: float | None = None,
        zero_point_ref_mag_unit: str = "mag",
        magnitude_system: str = "Vega",
        effective_wavelength: float | None = None,
        effective_wavelength_unit: str = "m",
        table_description: str | None = None,
        ra: float | None = None,
        dec: float | None = None,
        filter_name: str | None = None,
        period: float | None = None,
        epoch: float | None = None,
        binary: bool = True,
        coosys_id: str = "system",
        coosys_system: str | None = None,
        coosys_epoch: float | str | None = None,
        publication_id: str | None = None,
    ):
        """Writes this lightcurve to a compliant IVOA VOTable XML file.

        Refer to `write_vo_lightcurve` for detailed argument descriptions.
        """
        write_kwargs = dict(
            output_stream_or_path=output_stream_or_path,
            table_data=self.table,
            table_name=table_name,
            filter_identifier=filter_identifier,
            refposition=refposition,
            timescale=timescale,
            timeorigin=timeorigin,
            votable_description=votable_description,
            creator=creator,
            zero_point_flux=zero_point_flux,
            zero_point_flux_unit=zero_point_flux_unit,
            zero_point_ref_mag=zero_point_ref_mag,
            zero_point_ref_mag_unit=zero_point_ref_mag_unit,
            magnitude_system=magnitude_system,
            effective_wavelength=effective_wavelength,
            effective_wavelength_unit=effective_wavelength_unit,
            table_description=table_description,
            ra=ra,
            dec=dec,
            filter_name=filter_name,
            period=period,
            epoch=epoch,
            binary=binary,
        )
        if self.coosys is not None:
            write_kwargs.update(
                coosys_id=self.coosys.coosys_id,
                coosys_system=self.coosys.system,
                coosys_epoch=self.coosys.epoch,
            )
        elif coosys_system is not None:
            write_kwargs.update(
                coosys_id=coosys_id,
                coosys_system=coosys_system,
                coosys_epoch=coosys_epoch,
            )
        if publication_id is not None:
            write_kwargs["publication_id"] = publication_id
        return write_vo_lightcurve(**write_kwargs)



def _repair_votable_xml_char_param_ampersands(payload: bytes) -> bytes:
    """Corrects double-escaped ampersands in astropy VOTable XML char PARAM values.

    Astropy 6.x may serialise ``&`` in char PARAM ``value`` attributes as
    ``&amp;amp;`` instead of ``&amp;``. Archives and VO clients expect a single
    XML escape (e.g. bibcodes containing ``A&A``).

    Args:
        payload (bytes): Raw VOTable XML document bytes.

    Returns:
        bytes: XML with ``&amp;amp;`` normalised to ``&amp;``.
    """
    return payload.replace(b"&amp;amp;", b"&amp;")


def print_col_ucd(lc: VOLightCurve):
    """Prints the column names, units, and UCD metadata for a given lightcurve.

    Args:
        lc (VOLightCurve): The lightcurve object whose columns should be printed.
    """
    for colname in lc.colnames:
        logger.info(
            "Col: %-10s Unit: %s UCD: %s",
            colname,
            lc[colname].unit,
            lc[colname].info.meta.get('ucd', 'None'),
        )


def write_vo_lightcurve(
    output_stream_or_path,
    table_data,
    table_name: str,
    filter_identifier: str | None = None,
    refposition: str = "BARYCENTER",
    timescale: str = "TCB",
    timeorigin: float = 0,
    votable_description: str | None = None,
    creator: str | None = None,
    zero_point_flux: float | None = None,
    zero_point_flux_unit: str | None = None,
    zero_point_ref_mag: float | None = None,
    zero_point_ref_mag_unit: str = "mag",
    magnitude_system: str = "Vega",
    effective_wavelength: float | None = None,
    effective_wavelength_unit: str = "m",
    table_description: str | None = None,
    ra: float | None = None,
    dec: float | None = None,
    filter_name: str | None = None,
    period: float | None = None,
    epoch: float | None = None,
    binary: bool = True,
    coosys_id: str = "system",
    coosys_system: str | None = None,
    coosys_epoch: float | str | None = None,
    publication_id: str | None = None,
    facility_name: str | None = None,
    instrument_name: str | None = None,
):
    """Writes a lightcurve to a compliant IVOA VOTable (v1.4) XML file/stream.

    Column names are preserved. UCDs already stored on columns are written
    through; this function does not rename fields or drop columns
    (``docs/io_contract.md`` §8). Time-role columns are linked to TIMESYS.
    Photometry columns that already carry a photometry UCD are linked from
    the photcal GROUP.

    Args:
        output_stream_or_path (str or file-like object): Path or stream to write the output to.
        table_data (astropy.table.Table or pandas.DataFrame or VOLightCurve):
            The source lightcurve containing timing and photometry.
        table_name (str): Value for the `<TABLE name="...">` attribute (Obligatory).
        filter_identifier (str, optional): Value for the ``filterIdentifier`` PARAM.
            Omitted from the photcal GROUP when absent.
        refposition (str, optional): Time reference position (e.g. 'BARYCENTER', 'HELIOCENTER').
            Defaults to "BARYCENTER" (Obligatory).
        timescale (str, optional): Time scale. Defaults to "TCB" (Optional).
        timeorigin (float, optional): TIMESYS origin (``JD0`` sense) added to
            time-role columns to obtain absolute Julian Date. Defaults to ``0``.
        votable_description (str, optional): High-level global description. Defaults to None.
        creator (str, optional): Pipeline or entity creator name. Defaults to None.
        zero_point_flux (float, optional): Zero point flux value. Defaults to None.
        zero_point_flux_unit (str, optional): Internal unit of zeroPointFlux
            (``None`` = dimensionless; written as ``VO_DIMENSIONLESS_WIRE``).
        zero_point_ref_mag (float, optional): Reference magnitude zero point. Defaults to None.
        zero_point_ref_mag_unit (str, optional): Unit of zeroPointReferenceMagnitude. Defaults to "mag".
        magnitude_system (str, optional): Type of magnitude system. Defaults to "Vega".
        effective_wavelength (float, optional): Effective wavelength value. Defaults to None.
        effective_wavelength_unit (str, optional): Unit of effectiveWavelength. Defaults to "m".
        table_description (str, optional): Table block description. Defaults to None.
        ra (float, optional): RA of target in degrees. Defaults to None.
        dec (float, optional): Dec of target in degrees. Defaults to None.
        filter_name (str, optional): Human-readable filter name written as
            ``photDM:PhotometryFilter.name`` in the photcal GROUP (and a legacy
            table PARAM ``filter``). Defaults to None.
        period (float, optional): Variability period in days. Defaults to None.
        epoch (float, optional): Reference time epoch in days. Defaults to None.
        binary (bool, optional): If True, encodes table data in BINARY format.
            If False, encodes in TABLEDATA XML format. Defaults to True.
        coosys_id (str, optional): ``COOSYS/@ID`` when writing coordinate metadata.
        coosys_system (str, optional): ``COOSYS/@system`` (e.g. ``ICRS``). When omitted,
            no ``<COOSYS>`` element is written unless supplied via a ``VOLightCurve``.
        coosys_epoch (float or str, optional): ``COOSYS/@epoch`` (proper-motion epoch).
        publication_id (str, optional): Publication bibcode written as TABLE ``bibcode`` PARAM.
        facility_name (str, optional): Observatory or facility name TABLE PARAM (may be empty).
        instrument_name (str, optional): Instrument name TABLE PARAM (may be empty).
    """
    import astropy.io.votable as vot
    from astropy.io.votable.tree import Group, Param, Info, FieldRef, TimeSys, CooSys as VOTableCooSys
    import pandas as pd

    # Extract astropy Table
    if isinstance(table_data, Table):
        t = table_data.copy()
    elif hasattr(table_data, 'table') and isinstance(table_data.table, Table):
        t = table_data.table.copy()
    elif isinstance(table_data, pd.DataFrame):
        t = Table.from_pandas(table_data)
    else:
        raise TypeError(
            "table_data must be an astropy Table, VOLightCurve, or pandas DataFrame."
        )

    t_out = t

    # Convert to VOTableFile structure
    vot_file = vot.from_table(t_out)
    vot_file.version = '1.4'
    if votable_description:
        vot_file.description = votable_description

    res = vot_file.resources[0]
    tab = res.tables[0]
    tab.name = table_name
    if table_description:
        tab.description = table_description

    # Add creator info if provided
    if creator:
        info = Info(name='creator', value=creator)
        info.ucd = 'meta.bib.author'
        info.content = 'Pipeline or contributing resource creator'
        res.infos.append(info)

    # Add TimeSys metadata element
    ts = TimeSys(
        ID='ts',
        refposition=refposition,
        timescale=timescale,
        timeorigin=str(timeorigin),
        config={'version_1_4_or_later': True}
    )
    res.time_systems.append(ts)

    if coosys_system is not None:
        cs = VOTableCooSys(
            ID=coosys_id or "system",
            system=coosys_system,
            epoch=str(coosys_epoch) if coosys_epoch is not None else None,
        )
        res.coordinate_systems.append(cs)

    # Preserve column UCDs. Link time columns to TIMESYS. Do not invent roles.
    phot_field_ids: list[str] = []
    for f in tab.fields:
        f.ID = f.name
        col = t_out[f.name]
        ucd = ""
        if col.info.meta:
            ucd = str(col.info.meta.get("ucd") or "")
        if ucd:
            f.ucd = ucd
        if "time.epoch" in ucd:
            f.ref = "ts"
        if ucd and ("phot.mag" in ucd or "phot.flux" in ucd) and "stat.error" not in ucd:
            f.ref = "phot_def"
            phot_field_ids.append(f.name)
    if not phot_field_ids:
        for name in t_out.colnames:
            if name in ("phot", "mag", "flux"):
                phot_field_ids.append(name)

    # Add GROUP ID="phot_def" name="photcal"
    g = Group(vot_file, ID='phot_def', name='photcal')

    if filter_identifier:
        p_fid = Param(
            vot_file,
            name="filterIdentifier",
            value=filter_identifier,
            datatype="char",
            arraysize="*",
        )
        p_fid.utype = "photDM:PhotometryFilter.identifier"
        p_fid.ucd = "meta.id;instr.filter"
        g.entries.append(p_fid)

    # zeroPointFlux (Optional PARAM)
    if zero_point_flux is not None:
        zpf_unit_wire = to_wire(to_internal(zero_point_flux_unit))
        p_zpf = Param(
            vot_file,
            name='zeroPointFlux',
            value=float(zero_point_flux),
            datatype='double',
            unit=zpf_unit_wire,
        )
        p_zpf.utype = 'photDM:PhotCal.zeroPoint.flux.value'
        p_zpf.ucd = 'phot.flux;arith.zp'
        g.entries.append(p_zpf)

    # zeroPointReferenceMagnitude (Optional PARAM)
    if zero_point_ref_mag is not None:
        p_zpm = Param(vot_file, name='zeroPointReferenceMagnitude', value=float(zero_point_ref_mag), datatype='double', unit=zero_point_ref_mag_unit)
        p_zpm.utype = 'photDM:PhotCal.zeroPoint.referenceMagnitude.value'
        p_zpm.ucd = 'phot.mag;arith.zp'
        g.entries.append(p_zpm)

    # magnitudeSystem (PARAM with default "Vega")
    p_mgs = Param(vot_file, name='magnitudeSystem', value=magnitude_system, datatype='char', arraysize='*')
    p_mgs.utype = 'photDM:PhotCal.magnitudeSystem.type'
    p_mgs.ucd = 'meta.code'
    g.entries.append(p_mgs)

    # filterName — PhotometryFilter.name inside photcal GROUP
    if filter_name is not None:
        p_fn = Param(
            vot_file,
            name='filterName',
            value=str(filter_name),
            datatype='char',
            arraysize='*',
        )
        p_fn.utype = 'photDM:PhotometryFilter.name'
        p_fn.ucd = 'meta.id;instr.filter'
        p_fn.description = 'Human-readable photometric filter name'
        g.entries.append(p_fn)

    # effectiveWavelength (Optional PARAM)
    if effective_wavelength is not None:
        p_wl = Param(vot_file, name='effectiveWavelength', value=float(effective_wavelength), datatype='double', unit=effective_wavelength_unit)
        p_wl.utype = 'photDM:PhotometryFilter.spectralLocation.value'
        p_wl.ucd = 'em.wl.effective'
        g.entries.append(p_wl)

    for phot_name in phot_field_ids:
        g.entries.append(
            FieldRef(
                vot_file,
                ref=phot_name,
                utype="adhoc:location",
                config={"version_1_2_or_later": True},
            )
        )

    res.groups.append(g)

    # Table-level optional metadata PARAMs
    if ra is not None:
        p_ra = Param(vot_file, name='ra', value=float(ra), datatype='double')
        p_ra.ucd = 'pos.eq.ra'
        p_ra.description = 'RA of source object'
        tab.params.append(p_ra)

    if dec is not None:
        p_dec = Param(vot_file, name='dec', value=float(dec), datatype='double')
        p_dec.ucd = 'pos.eq.dec'
        p_dec.description = 'Dec of source object'
        tab.params.append(p_dec)

    if filter_name is not None:
        p_filt = Param(vot_file, name='filter', value=str(filter_name), datatype='char', arraysize='*')
        p_filt.ucd = 'meta.id;instr.filter'
        p_filt.description = 'Photometric filter name'
        tab.params.append(p_filt)

    if period is not None:
        p_per = Param(vot_file, name='period', value=float(period), datatype='double', unit='d')
        p_per.ucd = 'src.var;time.period'
        p_per.description = 'Period of the variable star'
        tab.params.append(p_per)

    if epoch is not None:
        p_ep = Param(vot_file, name='epoch', value=float(epoch), datatype='double', unit='d')
        p_ep.ucd = 'time.epoch'
        p_ep.ref = 'ts'
        p_ep.description = 'Reference time'
        tab.params.append(p_ep)

    if publication_id:
        p_bib = Param(
            vot_file,
            name='bibcode',
            value=str(publication_id),
            datatype='char',
            arraysize='*',
        )
        p_bib.ucd = 'meta.bib.bibcode'
        p_bib.utype = 'ssa:Curation.Reference'
        p_bib.description = 'URL or bibcode of a publication describing this data.'
        tab.params.append(p_bib)

    for param_name, param_value in (
        ("facility_name", facility_name),
        ("instrument_name", instrument_name),
    ):
        p_fac = Param(
            vot_file,
            name=param_name,
            value="" if param_value is None else str(param_value),
            datatype="char",
            arraysize="*",
        )
        p_fac.description = f"Observatory {param_name.replace('_', ' ')}"
        tab.params.append(p_fac)

    meta_src = getattr(t, 'meta', None) or {}
    optional_char_params = (
        ('sectors', 'meta.id;meta.dataset', 'TESS sector identifiers present in this lightcurve'),
        ('flux_origins', 'meta.code', 'Photometry extraction method (e.g. pdcsap, sap)'),
        ('authors', 'meta.bib.author', 'Pipeline author(s)'),
        ('title', 'meta.note', 'Display title for the lightcurve figure'),
        ('name', 'meta.id', 'Target identifier'),
        ('stitched', 'meta.code', 'True when sectors were stitched and flux calibration is relative'),
        ('cutout_source', 'meta.id;instr', 'Cutout data source: FFI or TPF'),
        ('mask_mode', 'meta.code', 'Aperture mask mode: handmade, threshold, or pipeline'),
    )
    for param_name, param_ucd, param_desc in optional_char_params:
        param_val = meta_src.get(param_name)
        if param_val is None:
            continue
        if isinstance(param_val, (list, tuple)):
            param_val = ','.join(str(v) for v in param_val)
        param_val = str(param_val).strip()
        if not param_val:
            continue
        p_extra = Param(vot_file, name=param_name, value=param_val, datatype='char', arraysize='*')
        p_extra.ucd = param_ucd
        p_extra.description = param_desc
        tab.params.append(p_extra)

    # Write to target path or stream. Astropy emits unit="---" for empty units;
    # always rewrite to VO_DIMENSIONLESS_WIRE (see vo_unit_codec).
    tabledata_format = 'binary' if binary else 'tabledata'
    needs_ampersand_repair = bool(publication_id and "&" in str(publication_id))
    buffer = io.BytesIO()
    vot_file.to_xml(buffer, tabledata_format=tabledata_format)
    payload = rewrite_astropy_empty_unit_attributes(buffer.getvalue())
    if needs_ampersand_repair:
        payload = _repair_votable_xml_char_param_ampersands(payload)
    if hasattr(output_stream_or_path, "write"):
        output_stream_or_path.write(payload)
    else:
        with open(output_stream_or_path, "wb") as handle:
            handle.write(payload)



def main():
    """Main execution block to test and demonstrate ingestion and conversion functionality."""
    for filename in [
        # 'data/lc_tess_HD182144_TIC_406949643_sector__40_author__SPOC_methods__pdcsap.vot',
        'data/lc_tess_HD182144_TIC_406949643_sector__40_author__SPOC_methods__pdcsap.ecsv'
        # 'data/OGLE-SMC-CEP-0325-I.vot',
        # 'data/6009363278148078848-G.vot',
        # 'data/AY_Lac-R.vot',
        # 'data/g2_jk.vot',
        # 'data/my_g3.vot',
        # 'data/ASas19pm.dat'
    ]:
        logger.info('Ingesting %s', filename)
        lc = VOLightCurve(file_path=filename)
        print_col_ucd(lc)
        logger.info('%s', lc.photdms)
        logger.info('%s', lc.timesys)
        logger.info('time columns: %s', get_time_colnames(lc))
        logger.info('flux columns: %s', get_flux_colnames(lc))
        logger.info('mag columns: %s', get_mag_colnames(lc))

        logger.info('%s', lc[0])
        logger.info('All flux columns. Convert into magnitudes')
        for colname in lc.get_flux_colnames():
            out_colname = f'magnitude_from_{colname}'
            logger.info('flux:%s --> mag:%s', colname, out_colname)
            lc.add_mag_column_from_flux(colname, out_colname)
            logger.info('%s', lc[out_colname, colname][0])
        logger.info('All magnitude columns. Convert into flux')
        for colname in lc.get_mag_colnames():
            out_colname = f'flux_from_{colname}'
            logger.info('mag:%s --> flux:%s', colname, out_colname)
            lc.add_flux_column_from_mag(colname, out_colname)
            logger.info('%s', lc[out_colname, colname][0])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
