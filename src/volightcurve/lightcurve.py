"""Virtual Observatory (VO) Lightcurve Parsing and Formatting Module.

This module provides tools, classes, and helper functions to ingest, parse, and format
astronomical lightcurve data files complying with VO standards (e.g., VOTable) or
heuristic text/ASCII formats. It implements data models for coordinate systems (CooSys),
time systems (TimeSys), and photometric calibrations (PhotDM, PhotCal, PhotometryFilter)
and facilitates unit-safe conversions between astronomical magnitudes and fluxes.
"""

import io
import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.io.votable import is_votable
from astropy.io import ascii

import logging

from volightcurve.photcal_check import PhotCalError

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

# The strict core: what a file declares. Interpretation (guesses from names, comments or
# conventions) lives in ``volightcurve.interpretation`` and is not listed here.
__all__ = [
    "VOLightCurve",
    "PhotCal",
    "PhotDM",
    "PhotometryFilter",
    "CooSys",
    "TimeSys",
    "extract_photdm",
    "extract_timesys",
    "extract_coosys",
    "write_vo_lightcurve",
    "find_columns_by_ucd",
    "get_time_colnames",
    "get_mag_colnames",
    "get_flux_colnames",
    "get_error_colnames",
    "get_label_colnames",
    "is_mag_column",
    "is_flux_column",
    "column_main_role",
    "assign_label_column_ucd",
    "assign_photometry_column_semantics",
    "print_col_ucd",
    "DEFAULT_TIME_EPOCH_UCD",
    "DEFAULT_LABEL_UCD",
    "LABEL_UCD_FRAGMENTS",
]


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
    Omitted zero points stay missing. They are not filled with a usable pair.
    Conversion then raises :class:`PhotCalError`.
    """

    def __init__(
        self,
        zp_flux=None,
        zp_flux_unit=None,
        zp_mag=None,
        zp_mag_unit=None,
        mag_sys=None,
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
                ``None`` stays ``None`` (no system is assumed).
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
        self.magnitude_system = magnitude_system or MagnitudeSystem(type=mag_sys or None)
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

    def conversion_problems(self, *, flux_unit=None, mag_unit=None) -> list[str]:
        """Lists why this calibration cannot convert magnitude and flux.

        Does not insert default zero points.

        Args:
            flux_unit: Flux column unit when a flux Quantity is in play.
            mag_unit: Magnitude unit when a magnitude Quantity is in play.

        Returns:
            list[str]: Empty when conversion may proceed.
        """
        from volightcurve.photcal_check import inspect_conversion_photcal

        zp = self.zero_point
        zp_flux = None if zp._zp_flux is None else zp._zp_flux.value
        zp_mag = None if zp._zp_mag is None else zp._zp_mag.value
        return inspect_conversion_photcal(
            zp_flux=zp_flux,
            zp_mag=zp_mag,
            zp_flux_unit=zp._zp_flux_unit_text,
            zp_mag_unit=zp._zp_mag_unit_text,
            flux_unit=flux_unit,
            mag_unit=mag_unit,
            zp_flux_unit_unparsed=bool(zp._zp_flux_unit_unparsed),
            zp_mag_unit_unparsed=bool(zp._zp_mag_unit_unparsed),
        )

    def _require_conversion(self, *, flux_unit=None, mag_unit=None) -> None:
        """Raises when this calibration cannot convert.

        Args:
            flux_unit: Flux column unit, when relevant.
            mag_unit: Magnitude unit, when relevant.

        Raises:
            PhotCalError: One or more specific reasons.
        """
        from volightcurve.photcal_check import PhotCalError

        problems = self.conversion_problems(flux_unit=flux_unit, mag_unit=mag_unit)
        if problems:
            raise PhotCalError(problems)

    def mag_to_flux(self, mag):
        """Converts magnitude to flux via the nested ``zero_point``.

        Args:
            mag (astropy.units.Quantity): Magnitude values.

        Returns:
            astropy.units.Quantity: Flux values.

        Raises:
            PhotCalError: When the zero points or units cannot support conversion.
        """
        self._require_conversion(mag_unit=getattr(mag, "unit", None))
        return self.zero_point.mag_to_flux(mag)

    def flux_to_mag(self, flux):
        """Converts flux to magnitude via the nested ``zero_point``.

        Args:
            flux (astropy.units.Quantity): Flux values.

        Returns:
            astropy.units.Quantity: Magnitude values.

        Raises:
            PhotCalError: When the zero points or units cannot support conversion.
        """
        self._require_conversion(flux_unit=getattr(flux, "unit", None))
        return self.zero_point.flux_to_mag(flux)

    def mag_err_to_flux_err(self, mag, mag_err):
        """Propagates magnitude uncertainties to flux via ``zero_point``.

        Args:
            mag (astropy.units.Quantity): Magnitude values.
            mag_err (astropy.units.Quantity): Magnitude uncertainties.

        Returns:
            astropy.units.Quantity: Flux uncertainties.

        Raises:
            PhotCalError: When the zero points or units cannot support conversion.
        """
        self._require_conversion(mag_unit=getattr(mag, "unit", None))
        return self.zero_point.mag_err_to_flux_err(mag, mag_err)

    def flux_err_to_mag_err(self, flux, flux_err):
        """Propagates flux uncertainties to magnitude via ``zero_point``.

        Args:
            flux (astropy.units.Quantity): Flux values.
            flux_err (astropy.units.Quantity): Flux uncertainties.

        Returns:
            astropy.units.Quantity: Magnitude uncertainties.

        Raises:
            PhotCalError: When the zero points or units cannot support conversion.
        """
        self._require_conversion(flux_unit=getattr(flux, "unit", None))
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
            str or None: The identifier of the filter, or ``None`` when no filter is set
                or the filter has no identifier.
        """
        return self.filter.filter_id if self.filter else None

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
            astropy.units.Quantity or None: The zero-point magnitude of the calibration,
                or ``None`` when no calibration is set or it declares none.
        """
        return self.photcal.zp_mag if self.photcal else None

    @mag0.setter
    def mag0(self, value):
        self.photcal.zp_mag = value

    def conversion_problems(self, *, flux_unit=None, mag_unit=None) -> list[str]:
        """Lists why this calibration cannot convert magnitude and flux.

        Args:
            flux_unit: Flux column unit, when relevant.
            mag_unit: Magnitude column unit, when relevant.

        Returns:
            list[str]: Empty when conversion may proceed; one sentence per problem.
        """
        if self.photcal is None:
            return ["No photometric calibration (photcal) is declared."]
        return self.photcal.conversion_problems(flux_unit=flux_unit, mag_unit=mag_unit)

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

    def __init__(self, refposition=None, timeorigin=None, timescale=None):
        """Initialises a TimeSys instance.

        Args:
            refposition (str, optional): Time reference position (e.g., 'HELIOCENTER',
                'BARYCENTER'). ``None`` when not declared.
            timeorigin (float, optional): The origin offset of the time scale, i.e.,
                the JD0 or MJD0 value. ``None`` when not declared (0.0 is a declared value).
            timescale (str, optional): The timing scale used (e.g., 'UTC', 'TDB', 'TCB').
                ``None`` when not declared.
        """
        self.refposition = refposition  # HELIOCENTER OR BARYCENTER ...
        self.timeorigin = timeorigin  # JD0, f.e. 2400000.5
        self.timescale = timescale  # UTC, TCB, TBD etc

    @property
    def jd0(self):
        """Gets the reference time origin (JD0 offset).

        Returns:
            float or None: The reference time origin value, ``None`` when not declared.
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


def _declared_float(value) -> float | None:
    """Reads a numeric PARAM value, treating the absent-value wire forms as ``None``.

    Files of earlier versions wrote ``NaN`` (double) or an empty string for an absent
    value; both are read as "not declared".

    Args:
        value: Raw PARAM value (number, string, masked value or ``None``).

    Returns:
        float or None: The declared number, or ``None`` when absent.
    """
    if value is None or np.ma.is_masked(value):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    number = float(value)
    return None if np.isnan(number) else number


def _declared_text(value) -> str | None:
    """Reads a text PARAM value, treating an empty string as absent.

    Args:
        value: Raw PARAM value (string, masked value or ``None``).

    Returns:
        str or None: The stripped text, or ``None`` when empty or absent.
    """
    if value is None or np.ma.is_masked(value):
        return None
    text = str(value).strip()
    return text or None


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
            cal_params = {}
            filter_params = {
                "filter_id": None,
                "name": None,
                "spectral_location": None,
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
                        value = _declared_float(target_param.value)
                        if value is not None:
                            cal_params["zp_flux"] = value
                            cal_params["zp_flux_unit"] = to_internal(
                                getattr(target_param, "unit", None)
                            )
                    elif ut == UT_MAG.lower():
                        value = _declared_float(target_param.value)
                        if value is not None:
                            cal_params["zp_mag"] = value
                            cal_params["zp_mag_unit"] = target_param.unit
                    elif ut == UT_MAG_SYS.lower():
                        cal_params["mag_sys"] = _declared_text(target_param.value)
                    elif ut == UT_FILTER.lower():
                        filter_params["filter_id"] = _declared_text(target_param.value)
                    elif ut == UT_FILTER_NAME.lower():
                        filter_params["name"] = _declared_text(target_param.value)
                    elif (
                        not ut
                        and (getattr(target_param, "name", None) or "").lower()
                        in ("filter", "filtername")
                    ):
                        if not filter_params.get("name"):
                            filter_params["name"] = _declared_text(target_param.value)
                    elif ut == UT_FILTER_SPEC.lower():
                        value = _declared_float(target_param.value)
                        if value is not None:
                            filter_params["spectral_location"] = value
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


def _clean_text(value) -> str | None:
    """Returns stripped text, or ``None`` when the value is empty.

    Args:
        value (object): A description or INFO text.

    Returns:
        str or None: The stripped text, ``None`` for ``None`` or blank input.
    """
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _declared_infos(astro_tree, selected_table) -> list[dict]:
    """Lists the INFO elements that apply to the selected table, in file order.

    Collects the INFOs of the VOTABLE root, of the RESOURCE that holds the table and of
    the table itself. Nothing is dropped or merged, so repeated names (several
    ``creator`` entries) are all kept.

    Args:
        astro_tree (astropy.io.votable.tree.VOTableFile): Parsed VOTable.
        selected_table (astropy.io.votable.tree.TableElement): The table in use.

    Returns:
        list[dict]: One dict per INFO with ``name`` and ``value`` and, when declared,
            ``ID``, ``ucd``, ``utype``, ``unit``, ``xtype``, ``ref`` and ``content``.
    """
    holders = [astro_tree]
    for resource in astro_tree.resources:
        if any(table is selected_table for table in resource.tables):
            holders.append(resource)
    holders.append(selected_table)
    found: list[dict] = []
    for holder in holders:
        for info in getattr(holder, "infos", []) or []:
            entry = {"name": info.name, "value": info.value}
            for attribute in ("ID", "ucd", "utype", "unit", "xtype", "ref"):
                text = getattr(info, attribute, None)
                if text:
                    entry[attribute] = str(text)
            content = _clean_text(getattr(info, "content", None))
            if content:
                entry["content"] = content
            found.append(entry)
    return found


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
    from gavo.votable import tableparser, votparse
    from gavo.votable.model import VOTable as V

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


DEFAULT_TIME_EPOCH_UCD = "time.epoch"


# UCD words of a column that describes the measurements without being one: a limit
# (``stat.max`` / ``stat.min``), a signal-to-noise ratio (``stat.snr``) or a bare ``stat``
# (a statistic of the flux, such as flux over error). Such a column keeps its photometric UCD
# words but has no photometric calibration of its own, so it is not a photometry column.
NON_MEASUREMENT_UCD_WORDS = frozenset({"stat", "stat.max", "stat.min", "stat.snr"})


def is_measurement_ucd(ucd: str) -> bool:
    """Tells whether a UCD describes a measurement, not a statistic about the measurements.

    Args:
        ucd (str): UCD string of a column, words separated by ``;``.

    Returns:
        bool: False when the UCD carries a word of ``NON_MEASUREMENT_UCD_WORDS``.
    """
    words = {word.strip() for word in ucd.split(";")}
    return not (words & NON_MEASUREMENT_UCD_WORDS)


def get_mag_colnames(table):
    """Retrieves names of columns containing primary magnitudes from the table, excluding error columns.

    Args:
        table (astropy.table.Table): The table to search.

    Returns:
        list of str: Column names matching magnitude UCDs (e.g., 'phot.mag') that do not represent errors.
    """
    # Returns primary magnitudes, excludes errors
    all_mags = find_columns_by_ucd(table, 'phot.mag')
    return [
        c
        for c in all_mags
        if 'stat.error' not in table[c].info.meta.get('ucd', '')
        and is_measurement_ucd(table[c].info.meta.get('ucd', ''))
    ]


def get_flux_colnames(table):
    """Retrieves names of columns containing primary fluxes from the table, excluding error columns.

    Args:
        table (astropy.table.Table): The table to search.

    Returns:
        list of str: Column names matching flux UCDs (e.g., 'phot.flux') that do not represent errors.
    """
    all_flux = find_columns_by_ucd(table, 'phot.flux')
    return [
        c
        for c in all_flux
        if 'stat.error' not in table[c].info.meta.get('ucd', '')
        and is_measurement_ucd(table[c].info.meta.get('ucd', ''))
    ]


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
    if not is_measurement_ucd(ucd):
        return None
    if "phot.mag" in ucd:
        return "mag"
    if "phot.flux" in ucd:
        return "flux"
    return None


def _is_per_row_stamp(table: Table, colname: str) -> bool:
    """True when a column has a different value on nearly every row.

    A timestamp such as ``UT Date`` must not be the grouping label. Small
    tables are never treated as stamps.

    Args:
        table (astropy.table.Table): Product table.
        colname (str): Candidate label column.

    Returns:
        bool: True when the column is unique per row on a long table.
    """
    n_rows = len(table)
    if n_rows < 30:
        return False
    values = np.asarray(table[colname], dtype=object)
    distinct = len(set(values.tolist()))
    return distinct >= 30 and distinct / n_rows > 0.8


def get_label_colnames(table: Table) -> list[str]:
    """Returns the designated per-epoch label column, if any.

    Prefers the leftmost column whose UCD contains ``meta.code`` or
    ``meta.id``. Otherwise returns the leftmost column that is not time,
    magnitude, flux, or an error. Columns that are unique on almost every
    row (observation dates) are skipped. See ``docs/io_contract.md`` §8.

    Args:
        table (astropy.table.Table): Product or file table.

    Returns:
        list of str: Zero or one column name.
    """
    labelled = [
        name
        for name in table.colnames
        if any(fragment in _column_ucd(table, name) for fragment in LABEL_UCD_FRAGMENTS)
        and not _is_per_row_stamp(table, name)
    ]
    if labelled:
        return [labelled[0]]
    for name in table.colnames:
        if column_main_role(table, name) is None and not _is_per_row_stamp(table, name):
            return [name]
    return []


def assign_label_column_ucd(table: Table) -> str | None:
    """Stores a label-role UCD on the designated label column when missing.

    Does not rename the column. If the column already has a UCD containing
    ``meta.code`` or ``meta.id``, it is left unchanged. Otherwise the UCD is
    set to ``meta.id``. See ``docs/io_contract.md`` §8.

    Args:
        table (astropy.table.Table): Product table after role promotion.

    Returns:
        str or None: Name of the column that received the UCD, ``None`` when nothing
            was assigned.
    """
    names = get_label_colnames(table)
    if not names:
        return None
    name = names[0]
    ucd = _column_ucd(table, name)
    if any(fragment in ucd for fragment in LABEL_UCD_FRAGMENTS):
        return None
    column = table[name]
    if column.info.meta is None:
        column.info.meta = {}
    column.info.meta["ucd"] = DEFAULT_LABEL_UCD
    return name


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


def assign_photometry_column_semantics(
    table: Table,
    phot_col: str,
    error_col: str | None,
    *,
    force_magnitude: bool,
) -> Table:
    """Assigns VO UCD metadata to a photometry column and its error column.

    The caller states whether the column holds magnitudes; nothing is guessed here.

    Args:
        table (astropy.table.Table): Export or upload table to annotate.
        phot_col (str): Primary photometry column name.
        error_col (str, optional): Uncertainty column name.
        force_magnitude (bool): ``True`` for magnitudes, ``False`` for flux.

    Returns:
        astropy.table.Table: The same table with UCD metadata updated in place.
    """
    if phot_col not in table.colnames:
        return table
    phot_ucd = "phot.mag" if force_magnitude else "phot.flux;em.opt"
    err_ucd = "stat.error;phot.mag" if force_magnitude else "stat.error;phot.flux;em.opt"
    if table[phot_col].info.meta is None:
        table[phot_col].info.meta = {}
    table[phot_col].info.meta["ucd"] = phot_ucd
    if error_col and error_col in table.colnames:
        if table[error_col].info.meta is None:
            table[error_col].info.meta = {}
        table[error_col].info.meta["ucd"] = err_ucd
    return table


def _unit_text_for_check(unit) -> str:
    """Returns the text of an Astropy unit as it would appear on the VOTable wire.

    Args:
        unit (astropy.units.UnitBase): Unit of a table column.

    Returns:
        str: ``1`` for dimensionless, the raw text for an unrecognised unit, the VOUnits
            form otherwise (the plain text when Astropy cannot express it in VOUnits).
    """
    if isinstance(unit, u.UnrecognizedUnit):
        return str(unit)
    if unit == u.dimensionless_unscaled:
        return "1"
    try:
        return unit.to_string("vounit")
    except (ValueError, u.UnitsError):
        return str(unit)


class VOLightCurve:
    """Represents a Virtual Observatory (VO) lightcurve container.

    Encapulates an Astropy Table containing timing and photometric observations,
    along with associated coordinate system, time system, and photometric calibration
    metadata mapped to specific columns. It supports files in both VOTable and ASCII
    heuristic formats.
    """

    def __init__(self, file_path, *, table_id: str | int | None = None, strict: bool = False):
        """Initialises a VOLightCurve instance and ingests the specified data file.

        Args:
            file_path (str or file-like object): Path to the input file or an active file-like stream.
            table_id (str or int, optional): When the VOTable embeds multiple ``TABLE``
                elements, selects one by Astropy table ``ID``, name, or zero-based index.
            strict (bool, optional): ``True`` reads only what a VOTable declares: no column
                roles from names, no ``JD0`` from comments, and a file that is not a VOTable
                is rejected. Used to validate issued products. Defaults to ``False``
                (tolerant: the interpretation module runs on what the file carries loosely).

        Raises:
            ValueError: In strict mode, when the file is not a VOTable.
        """
        self.strict = bool(strict)
        self.file_path = file_path
        self.table = None
        self.timesys = TimeSys()
        self.timesys_by_id: dict[str, TimeSys] = {}
        self.field_timesys_ref: dict[str, str] = {}
        self.param_timesys_ref: dict[str, str | None] = {}
        self.coosys = None
        self.photdms = {}  # maps column_name -> PhotDM instance
        self.votable_description: str | None = None  # VOTABLE-level DESCRIPTION
        self.infos: list[dict] = []  # INFO elements as the file declared them
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
        self.votable_description = _clean_text(astro_tree.description)
        self.infos = _declared_infos(astro_tree, selected_table)

        gavo_tree = _gavo_votable_metadata_tree(payload)
        _apply_gavo_votable_metadata(self, gavo_tree)

        if self.strict:
            return
        from volightcurve.interpretation import (
            _pickup_jd0_from_table,
            _promote_to_vo_standards,
            add_interpretation_note,
        )

        self.table = _promote_to_vo_standards(self.table)
        label = assign_label_column_ucd(self.table)
        if label:
            add_interpretation_note(
                self.table, "from the column values", [f"column {label} was marked as the label column"]
            )
        if self.timesys.timeorigin is None:
            self.timesys.timeorigin = _pickup_jd0_from_table(self.table)
            if self.timesys.timeorigin is not None:
                add_interpretation_note(
                    self.table,
                    "from comments or metadata keywords",
                    [f"time origin {self.timesys.timeorigin}"],
                )

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
        instance.votable_description = None
        instance.infos = []
        instance._table_id = None
        instance.strict = False
        from volightcurve.interpretation import apply_non_votable_heuristics

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

        if self.strict:
            raise ValueError(
                "Strict mode reads VOTable files only; this file is not a VOTable."
            )

        from volightcurve.interpretation import (
            _recover_lc_colnames,
            apply_non_votable_heuristics,
        )

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

    def check_units(self) -> dict:
        """Checks every unit on the product against VOUnits 1.1 (informative, never raises).

        Covers the table columns and the declared photcal PARAM units (zero-point flux,
        zero-point magnitude, effective wavelength). A column without a unit and a
        calibration without a declared unit are fine. Only findings are returned.

        Returns:
            dict[str, UnitReport]: ``column:<name>`` or ``photcal:<column>:<PARAM name>``
                to the report; empty when every declared unit is clean.
        """
        from volightcurve.vo_unit_codec import inspect_unit_string

        findings: dict = {}

        def add(key, text):
            report = inspect_unit_string(text)
            if report.errors or report.notes:
                findings[key] = report

        for name in self.table.colnames:
            unit = self.table[name].unit
            if unit is None:
                continue
            add(f"column:{name}", _unit_text_for_check(unit))
        for column, photdm in self.photdms.items():
            photcal = photdm.photcal
            if photcal is not None:
                if photcal.zp_flux is not None and photcal._zp_flux_unit_text is not None:
                    add(f"photcal:{column}:zeroPointFlux", photcal._zp_flux_unit_text)
                if photcal.zp_mag is not None and photcal._zp_mag_unit_text is not None:
                    add(f"photcal:{column}:zeroPointReferenceMagnitude", photcal._zp_mag_unit_text)
            spectral = photdm.filter.spectral_location if photdm.filter is not None else None
            if spectral is not None and getattr(spectral, "unit", None) is not None:
                add(f"photcal:{column}:effectiveWavelength", _unit_text_for_check(spectral.unit))
        return findings

    def conversion_problems(self, column: str) -> list[str]:
        """Lists why a flux or magnitude column cannot be converted. Never raises on data.

        Does not insert default zero points. Use before a conversion to ask the user for
        what is missing.

        Args:
            column (str): Name of a flux or magnitude column of the table.

        Returns:
            list[str]: Empty when the conversion may proceed; otherwise one sentence per
                missing or inconsistent item.

        Raises:
            KeyError: When ``column`` is not a column of the table.
        """
        if column not in self.table.colnames:
            raise KeyError(f"No column {column!r} in the table.")
        photdm = self.photdms.get(column)
        if photdm is None or photdm.photcal is None:
            return [f"Column {column!r} has no photometric calibration (photcal)."]
        unit = self.table[column].unit
        if column in self.get_mag_colnames():
            return photdm.photcal.conversion_problems(mag_unit=unit)
        return photdm.photcal.conversion_problems(flux_unit=unit)

    def time_problems(self) -> list[str]:
        """Lists what is missing in the time description. Never raises on data.

        Offsets (values below the absolute-JD threshold) need a declared time origin to
        become absolute Julian Dates. The time scale and the reference position are
        reported when they are not declared.

        Returns:
            list[str]: Empty when the time description is complete; otherwise one sentence
                per missing item.
        """
        from volightcurve.time_reference import TIME_OFFSET_ABSOLUTE_JD_THRESHOLD

        problems: list[str] = []
        time_cols = self.get_time_colnames()
        if not time_cols:
            problems.append("No observation time column was found.")
        timesys = self.timesys
        if timesys.timeorigin is None:
            has_offsets = False
            for name in time_cols:
                values = np.asarray(self.table[name], dtype=float)
                finite = values[np.isfinite(values)]
                if finite.size and float(np.min(finite)) < TIME_OFFSET_ABSOLUTE_JD_THRESHOLD:
                    has_offsets = True
            if has_offsets:
                problems.append(
                    "The time origin (TIMESYS timeorigin) is not declared, so time "
                    "offsets cannot be converted to absolute Julian Date."
                )
        if timesys.timescale is None:
            problems.append("The time scale (TIMESYS timescale) is not declared.")
        if timesys.refposition is None:
            problems.append("The time reference position (TIMESYS refposition) is not declared.")
        return problems

    def add_flux_column_from_mag(self, mag_col_name, new_col_name=None):
        """Generates and adds a new flux column converted from the specified magnitude column.

        Uses the ``PhotCal`` of the column. Nothing is filled in: a missing or unusable
        calibration raises. Ask ``conversion_problems`` first to learn what is missing.

        Args:
            mag_col_name (str): The name of the source magnitude column in the table.
            new_col_name (str, optional): The name for the newly created flux column.
                Defaults to None (which yields 'flux_from_<mag_col_name>').

        Returns:
            str: The name of the newly added flux column.

        Raises:
            PhotCalError: When the column has no calibration, or the zero points or units
                cannot support the conversion.
        """
        colname_out = new_col_name or f"flux_from_{mag_col_name}"
        problems = self.conversion_problems(mag_col_name)
        if problems:
            raise PhotCalError(problems)
        photdm = self.photdms[mag_col_name]
        flux = photdm.photcal.mag_to_flux(self.table[mag_col_name])

        # Physical equivalence: flux density is power/area/frequency (or wavelength).
        ucd = 'phot.flux'
        if hasattr(flux, 'unit') and flux.unit is not None:
            u_flux_nu = u.W / (u.m ** 2 * u.Hz)
            u_flux_lam = u.W / (u.m ** 2 * u.m)
            if any(flux.unit.is_equivalent(ref_unit) for ref_unit in (u_flux_nu, u_flux_lam)):
                ucd = 'phot.flux.density'

        self.table[colname_out] = flux
        self.table[colname_out].info.meta = {'ucd': ucd}
        self.photdms[colname_out] = photdm
        return colname_out

    def add_mag_column_from_flux(self, flux_col_name, new_col_name=None):
        """Generates and adds a new magnitude column converted from the specified flux column.

        Uses the ``PhotCal`` of the column. Nothing is filled in: a missing or unusable
        calibration raises. Ask ``conversion_problems`` first to learn what is missing.

        Args:
            flux_col_name (str): The name of the source flux column in the table.
            new_col_name (str, optional): The name for the newly created magnitude column.
                Defaults to None (which yields 'mag_from_<flux_col_name>').

        Returns:
            str: The name of the newly added magnitude column.

        Raises:
            PhotCalError: When the column has no calibration, or the zero points or units
                cannot support the conversion.
        """
        colname_out = new_col_name or f"mag_from_{flux_col_name}"
        problems = self.conversion_problems(flux_col_name)
        if problems:
            raise PhotCalError(problems)
        photdm = self.photdms[flux_col_name]
        mag = photdm.photcal.flux_to_mag(self.table[flux_col_name])
        self.table[colname_out] = mag
        self.table[colname_out].info.meta = {'ucd': 'phot.mag'}
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
        refposition: str | None = None,
        timescale: str | None = None,
        timeorigin: float | None = None,
        votable_description: str | None = None,
        creator: str | None = None,
        zero_point_flux: float | None = None,
        # Internal: ``None`` = dimensionless (wire form is ``VO_DIMENSIONLESS_WIRE``).
        zero_point_flux_unit: str | None = None,
        zero_point_ref_mag: float | None = None,
        zero_point_ref_mag_unit: str = "mag",
        magnitude_system: str | None = None,
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

        Time system and calibration come from this instance (``timesys`` and
        ``photdms``). ``refposition``, ``timescale`` and ``timeorigin`` override the
        instance values when given. The flat calibration arguments
        (``filter_identifier``, ``zero_point_*``, ``magnitude_system``,
        ``effective_wavelength*``, ``filter_name``) are DEPRECATED: when any is given the
        instance ``photdms`` are not used. Refer to `write_vo_lightcurve` for detailed
        argument descriptions.
        """
        if refposition is None:
            refposition = self.timesys.refposition
        if timescale is None:
            timescale = self.timesys.timescale
        if timeorigin is None:
            timeorigin = self.timesys.timeorigin
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
        flat_calibration_given = any(
            value is not None
            for value in (
                filter_identifier,
                zero_point_flux,
                zero_point_flux_unit,
                zero_point_ref_mag,
                magnitude_system,
                effective_wavelength,
                filter_name,
            )
        )
        if not flat_calibration_given:
            write_kwargs["photdms"] = self.photdms
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


def _photcal_params_from_photdm(photdm) -> dict:
    """Collects the declared photcal values of one ``PhotDM`` for writing.

    Args:
        photdm (PhotDM): Calibration of one or more columns.

    Returns:
        dict: Keys ``filter_identifier``, ``filter_name``, ``zero_point_flux``,
            ``zero_point_flux_unit`` (internal text or ``None``), ``zero_point_ref_mag``,
            ``zero_point_ref_mag_unit``, ``magnitude_system``, ``effective_wavelength``
            and ``effective_wavelength_unit``; ``None`` for everything not declared.
    """
    params = {
        "filter_identifier": None,
        "filter_name": None,
        "zero_point_flux": None,
        "zero_point_flux_unit": None,
        "zero_point_ref_mag": None,
        "zero_point_ref_mag_unit": "mag",
        "magnitude_system": None,
        "effective_wavelength": None,
        "effective_wavelength_unit": "m",
    }
    phot_filter = photdm.filter
    if phot_filter is not None:
        params["filter_identifier"] = phot_filter.filter_id or None
        params["filter_name"] = phot_filter.name
        spectral = phot_filter.spectral_location
        if spectral is not None:
            params["effective_wavelength"] = float(getattr(spectral, "value", spectral))
            unit = getattr(spectral, "unit", None)
            if unit is not None:
                params["effective_wavelength_unit"] = str(unit)
    photcal = photdm.photcal
    if photcal is not None:
        if photcal.zp_flux is not None:
            params["zero_point_flux"] = float(getattr(photcal.zp_flux, "value", photcal.zp_flux))
            params["zero_point_flux_unit"] = to_internal(photcal._zp_flux_unit_text)
        if photcal.zp_mag is not None:
            params["zero_point_ref_mag"] = float(getattr(photcal.zp_mag, "value", photcal.zp_mag))
            params["zero_point_ref_mag_unit"] = photcal._zp_mag_unit_text or "mag"
        params["magnitude_system"] = photcal.mag_sys or None
    return params


def _photcal_group_specs(photdms: dict, phot_field_ids: list[str], error_field_ids: list[str]) -> list:
    """Groups table columns by the ``PhotDM`` object that describes them.

    Columns mapped to the same object share one group. Photometry columns that are not
    mapped share one empty calibration. Error columns that are not mapped are left out.

    Args:
        photdms (dict): Column name to ``PhotDM`` (names compared case-insensitively).
        phot_field_ids (list[str]): Flux and magnitude columns in table order.
        error_field_ids (list[str]): Flux and magnitude error columns in table order.

    Returns:
        list: ``(group_id, params, columns)`` triples in table order.
    """
    lowered = {str(key).lower(): value for key, value in photdms.items()}
    groups: dict[int, tuple[object, list[str]]] = {}
    unmapped: list[str] = []
    for name in phot_field_ids + error_field_ids:
        photdm = lowered.get(name.lower())
        if photdm is None:
            if name in phot_field_ids:
                unmapped.append(name)
            continue
        groups.setdefault(id(photdm), (photdm, []))[1].append(name)
    specs = [
        (f"photcal_{columns[0]}", _photcal_params_from_photdm(photdm), columns)
        for photdm, columns in groups.values()
    ]
    if unmapped:
        empty = _photcal_params_from_photdm(PhotDM())
        specs.append((f"photcal_{unmapped[0]}", empty, unmapped))
    return specs


def _append_photcal_group(vot_file, res, tab, group_id: str, params: dict, columns: list[str]) -> None:
    """Adds one ``GROUP name="photcal"`` to the resource and links its columns.

    Only declared values become PARAMs. Every column gets ``FIELD/@ref`` and a
    ``FIELDref`` inside the group.

    Args:
        vot_file: Astropy ``VOTableFile`` being built.
        res: Resource that receives the group.
        tab: Table whose FIELDs are linked.
        group_id (str): ``GROUP/@ID``.
        params (dict): Output of ``_photcal_params_from_photdm``.
        columns (list[str]): Column names described by this calibration.
    """
    from astropy.io.votable.tree import Group, Param, FieldRef

    g = Group(vot_file, ID=group_id, name='photcal')

    def add(name, value, datatype, utype, ucd, unit=None, arraysize=None, description=None):
        kwargs = {"name": name, "value": value, "datatype": datatype}
        if unit is not None:
            kwargs["unit"] = unit
        if arraysize is not None:
            kwargs["arraysize"] = arraysize
        param = Param(vot_file, **kwargs)
        param.utype = utype
        param.ucd = ucd
        if description:
            param.description = description
        g.entries.append(param)

    if params["filter_identifier"]:
        add("filterIdentifier", params["filter_identifier"], "char",
            "photDM:PhotometryFilter.identifier", "meta.id;instr.filter", arraysize="*")
    if params["zero_point_flux"] is not None:
        add("zeroPointFlux", float(params["zero_point_flux"]), "double",
            "photDM:PhotCal.zeroPoint.flux.value", "phot.flux;arith.zp",
            unit=to_wire(to_internal(params["zero_point_flux_unit"])))
    if params["zero_point_ref_mag"] is not None:
        add("zeroPointReferenceMagnitude", float(params["zero_point_ref_mag"]), "double",
            "photDM:PhotCal.zeroPoint.referenceMagnitude.value", "phot.mag;arith.zp",
            unit=params["zero_point_ref_mag_unit"])
    if params["magnitude_system"]:
        add("magnitudeSystem", str(params["magnitude_system"]), "char",
            "photDM:PhotCal.magnitudeSystem.type", "meta.code", arraysize="*")
    if params["filter_name"] is not None:
        add("filterName", str(params["filter_name"]), "char",
            "photDM:PhotometryFilter.name", "meta.id;instr.filter", arraysize="*",
            description="Human-readable photometric filter name")
    if params["effective_wavelength"] is not None:
        add("effectiveWavelength", float(params["effective_wavelength"]), "double",
            "photDM:PhotometryFilter.spectralLocation.value", "em.wl.effective",
            unit=params["effective_wavelength_unit"])

    for name in columns:
        g.entries.append(
            FieldRef(
                vot_file,
                ref=name,
                utype="adhoc:location",
                config={"version_1_2_or_later": True},
            )
        )
        for f in tab.fields:
            if f.name == name:
                f.ref = group_id
    res.groups.append(g)


def write_vo_lightcurve(
    output_stream_or_path,
    table_data,
    table_name: str,
    filter_identifier: str | None = None,
    refposition: str | None = None,
    timescale: str | None = None,
    timeorigin: float | None = None,
    votable_description: str | None = None,
    creator: str | None = None,
    zero_point_flux: float | None = None,
    zero_point_flux_unit: str | None = None,
    zero_point_ref_mag: float | None = None,
    zero_point_ref_mag_unit: str = "mag",
    magnitude_system: str | None = None,
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
    photdms: dict | None = None,
    infos: list[dict] | None = None,
):
    """Writes a lightcurve to a compliant IVOA VOTable (v1.4) XML file/stream.

    Nothing is invented: a value that is not given is not written. The photometric
    calibration comes from ``photdms`` (column name to ``PhotDM``): one
    ``GROUP name="photcal"`` is written per distinct ``PhotDM`` object, referencing every
    column mapped to it (a flux and its error may share one). A ``PhotDM`` with nothing
    declared still gives a group, without value PARAMs. The flat keyword arguments
    (``filter_identifier``, ``zero_point_*``, ``magnitude_system``,
    ``effective_wavelength*``, ``filter_name``) are DEPRECATED: they describe one
    calibration for all photometry columns and are kept for one release. They cannot be
    combined with ``photdms``.

    Column names are preserved. UCDs already stored on columns are written
    through; this function does not rename fields or drop columns
    (``docs/io_contract.md`` §8). A TIME-role column with no UCD receives
    ``time.epoch``. A non-empty time UCD is not rewritten. ``FIELD/@ref``
    (and the epoch PARAM) point at the ``TIMESYS/@ID`` of the time system
    actually written on this resource, and are omitted if none exists.
    Photometry columns that already carry a photometry UCD are linked from
    the photcal GROUP.

    Args:
        output_stream_or_path (str or file-like object): Path or stream to write the output to.
        table_data (astropy.table.Table or pandas.DataFrame or VOLightCurve):
            The source lightcurve containing timing and photometry.
        table_name (str): Value for the `<TABLE name="...">` attribute (Obligatory).
        filter_identifier (str, optional): DEPRECATED (use ``photdms``). Value for the
            ``filterIdentifier`` PARAM. Omitted from the photcal GROUP when absent.
        refposition (str, optional): Time reference position (e.g. 'BARYCENTER',
            'HELIOCENTER'). ``None`` when unknown.
        timescale (str, optional): Time scale. ``None`` when unknown.
        timeorigin (float, optional): TIMESYS origin (``JD0`` sense) added to
            time-role columns to obtain absolute Julian Date. ``None`` when unknown;
            then no ``TIMESYS`` is written (a declared ``refposition`` or ``timescale``
            is dropped with a warning). When given and ``refposition`` or ``timescale``
            is ``None``, the IVOA vocabulary term ``UNKNOWN`` is written for it.
        votable_description (str, optional): High-level global description. Defaults to None.
        creator (str, optional): Pipeline or entity creator name. Defaults to None.
        zero_point_flux (float, optional): DEPRECATED (use ``photdms``). Zero point flux value.
        zero_point_flux_unit (str, optional): DEPRECATED. Internal unit of zeroPointFlux
            (``None`` = dimensionless; written as ``VO_DIMENSIONLESS_WIRE``).
        zero_point_ref_mag (float, optional): DEPRECATED. Reference magnitude zero point.
        zero_point_ref_mag_unit (str, optional): DEPRECATED. Unit of zeroPointReferenceMagnitude.
        magnitude_system (str, optional): DEPRECATED. Type of magnitude system; not written
            when ``None``.
        effective_wavelength (float, optional): DEPRECATED. Effective wavelength value.
        effective_wavelength_unit (str, optional): DEPRECATED. Unit of effectiveWavelength.
        table_description (str, optional): Table block description. Defaults to None.
        infos (list[dict], optional): INFO elements to write on the RESOURCE, each a dict
            with ``name`` and ``value`` and optionally ``ID``, ``ucd``, ``utype``, ``unit``,
            ``xtype``, ``ref`` and ``content``. Written as given, repeated names included.
        ra (float, optional): RA of target in degrees. Defaults to None.
        dec (float, optional): Dec of target in degrees. Defaults to None.
        filter_name (str, optional): DEPRECATED (use ``photdms``). Human-readable filter
            name written as ``photDM:PhotometryFilter.name`` in the photcal GROUP (and a
            legacy table PARAM ``filter``). Defaults to None.
        period (float, optional): Variability period in days. Defaults to None.
        epoch (float, optional): Reference time epoch in days. Defaults to None.
        binary (bool, optional): If True, encodes table data in BINARY format.
            If False, encodes in TABLEDATA XML format. Defaults to True.
        coosys_id (str, optional): ``COOSYS/@ID`` when writing coordinate metadata.
        coosys_system (str, optional): ``COOSYS/@system`` (e.g. ``ICRS``). When omitted,
            no ``<COOSYS>`` element is written unless supplied via a ``VOLightCurve``.
        coosys_epoch (float or str, optional): ``COOSYS/@epoch`` (proper-motion epoch).
        publication_id (str, optional): Publication bibcode written as TABLE ``bibcode`` PARAM.
        facility_name (str, optional): Observatory or facility name TABLE PARAM; not
            written when ``None`` or empty.
        instrument_name (str, optional): Instrument name TABLE PARAM; not written when
            ``None`` or empty.
        photdms (dict, optional): Column name to ``PhotDM``. Columns that share one
            ``PhotDM`` object share one photcal GROUP. Photometry columns that are not
            listed are tied to one shared empty calibration; error columns that are not
            listed stay unlinked.

    Raises:
        ValueError: When ``photdms`` is combined with the deprecated flat calibration
            arguments.
    """
    import astropy.io.votable as vot
    from astropy.io.votable.tree import Param, Info, TimeSys, CooSys as VOTableCooSys
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

    for entry in infos or []:
        info = Info(name=entry["name"], value=entry["value"])
        for attribute in ("ID", "ucd", "utype", "unit", "xtype", "ref"):
            if entry.get(attribute):
                setattr(info, attribute, entry[attribute])
        if entry.get("content"):
            info.content = entry["content"]
        res.infos.append(info)

    # Add creator info if provided
    if creator:
        info = Info(name='creator', value=creator)
        info.ucd = 'meta.bib.author'
        info.content = 'Pipeline or contributing resource creator'
        res.infos.append(info)

    # TIMESYS is written only when the time origin is known (VOTable 1.4: the origin is
    # required unless the time is a calendar date). FIELD/@ref below uses its ID.
    timesys_id = None
    if timeorigin is None:
        if refposition is not None or timescale is not None:
            logger.warning(
                "No time origin given: TIMESYS is not written; refposition=%r and "
                "timescale=%r are dropped.",
                refposition,
                timescale,
            )
    else:
        ts = TimeSys(
            ID='ts',
            refposition=refposition or "UNKNOWN",
            timescale=timescale or "UNKNOWN",
            timeorigin=str(timeorigin),
            config={'version_1_4_or_later': True}
        )
        res.time_systems.append(ts)
        timesys_id = ts.ID

    if coosys_system is not None:
        cs = VOTableCooSys(
            ID=coosys_id or "system",
            system=coosys_system,
            epoch=str(coosys_epoch) if coosys_epoch is not None else None,
        )
        res.coordinate_systems.append(cs)

    flat_given = any(
        value is not None
        for value in (
            filter_identifier,
            zero_point_flux,
            zero_point_ref_mag,
            magnitude_system,
            effective_wavelength,
            filter_name,
        )
    )
    if photdms is not None and flat_given:
        raise ValueError(
            "photdms cannot be combined with the deprecated flat calibration arguments."
        )
    if photdms is None and not flat_given and hasattr(table_data, "photdms"):
        photdms = table_data.photdms

    # Preserve column UCDs. Link TIME-role fields to the TIMESYS we just wrote.
    phot_field_ids: list[str] = []
    error_field_ids: list[str] = []
    time_field_ids: list[str] = []
    for f in tab.fields:
        f.ID = f.name
        col = t_out[f.name]
        ucd = ""
        if col.info.meta:
            ucd = str(col.info.meta.get("ucd") or "")
        if DEFAULT_TIME_EPOCH_UCD in ucd:
            time_field_ids.append(f.name)
            f.ucd = ucd
            if timesys_id:
                f.ref = timesys_id
        elif ucd:
            f.ucd = ucd
        if ucd and ("phot.mag" in ucd or "phot.flux" in ucd):
            if "stat.error" in ucd:
                error_field_ids.append(f.name)
            else:
                phot_field_ids.append(f.name)

    if not time_field_ids or not phot_field_ids:
        raise ValueError(
            "The table has no column with a time UCD (time.epoch) or no photometry "
            "column with a phot.mag or phot.flux UCD. The writer does not guess roles "
            "from column names; build the product with VOLightCurve.from_table or "
            "assign the UCDs first."
        )

    if photdms is None:
        # Deprecated flat route: one calibration for every photometry column.
        flat_params = {
            "filter_identifier": filter_identifier,
            "filter_name": filter_name,
            "zero_point_flux": zero_point_flux,
            "zero_point_flux_unit": zero_point_flux_unit,
            "zero_point_ref_mag": zero_point_ref_mag,
            "zero_point_ref_mag_unit": zero_point_ref_mag_unit,
            "magnitude_system": magnitude_system,
            "effective_wavelength": effective_wavelength,
            "effective_wavelength_unit": effective_wavelength_unit,
        }
        group_specs = []
        shared_columns = phot_field_ids + error_field_ids
        if shared_columns:
            group_specs.append((f"photcal_{shared_columns[0]}", flat_params, shared_columns))
        legacy_filter_name = filter_name
    else:
        group_specs = _photcal_group_specs(photdms, phot_field_ids, error_field_ids)
        legacy_filter_name = next(
            (params["filter_name"] for _, params, _ in group_specs if params["filter_name"]),
            None,
        )

    for group_id, params, columns in group_specs:
        _append_photcal_group(vot_file, res, tab, group_id, params, columns)

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

    if legacy_filter_name is not None:
        p_filt = Param(vot_file, name='filter', value=str(legacy_filter_name), datatype='char', arraysize='*')
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
        if timesys_id:
            p_ep.ref = timesys_id
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
        if param_value is None or not str(param_value).strip():
            continue
        p_fac = Param(
            vot_file,
            name=param_name,
            value=str(param_value),
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


_MOVED_TO_INTERPRETATION = frozenset(
    {
        "_promote_to_vo_standards",
        "_pickup_jd0_from_table",
        "_pickup_period_from_table",
        "_pickup_epoch_from_table",
        "_pickup_filter_from_table",
        "_DAT_METADATA_COMMENT",
        "_is_dat_metadata_comment_line",
        "_recover_lc_colnames",
        "apply_non_votable_heuristics",
        "_mag0_declared_in_comments",
        "is_magnitude_phot_column",
    }
)


def __getattr__(name):
    """Serves the names moved to ``volightcurve.interpretation`` (compatibility, one release).

    Args:
        name (str): Attribute requested from this module.

    Returns:
        object: The moved function or constant.

    Raises:
        AttributeError: When ``name`` was not moved.
    """
    if name in _MOVED_TO_INTERPRETATION:
        from volightcurve import interpretation

        return getattr(interpretation, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
