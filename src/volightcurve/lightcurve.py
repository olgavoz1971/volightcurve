"""The ``VOLightCurve`` class: one lightcurve table with its declared VO metadata.

The former single module was split by responsibility; the old import paths still work
for one release through ``__getattr__`` at the end of this file:

* ``volightcurve.model``: ``PhotometryFilter``, ``PhotCal``, ``PhotDM``, ``CooSys``,
  ``TimeSys``.
* ``volightcurve.votable_reader``: extraction of declared metadata (PhotDM, TIMESYS,
  COOSYS, INFO) from the GAVO metadata tree.
* ``volightcurve.columns``: column discovery by UCD and column roles.
* ``volightcurve.votable_writer``: ``write_vo_lightcurve`` and its helpers.
* ``volightcurve.interpretation``: everything that guesses.
"""

import io
import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.io.votable import is_votable
from astropy.io import ascii

import logging

from volightcurve.photcal_check import PhotCalError


from volightcurve.columns import (
    DEFAULT_LABEL_UCD,
    DEFAULT_TIME_EPOCH_UCD,
    LABEL_UCD_FRAGMENTS,
    assign_photometry_column_semantics,
    column_main_role,
    find_columns_by_ucd,
    get_error_colnames,
    get_flux_colnames,
    get_label_colnames,
    get_mag_colnames,
    get_time_colnames,
    is_flux_column,
    is_mag_column,
    print_col_ucd,
)
from volightcurve.model import CooSys, PhotCal, PhotDM, PhotometryFilter, TimeSys
from volightcurve.votable_reader import (
    _apply_gavo_votable_metadata,
    _clean_text,
    _declared_infos,
    _gavo_votable_metadata_tree,
    _list_votable_table_ids,
    _read_votable_payload,
    extract_coosys,
    extract_photdm,
    extract_timesys,
)
from volightcurve.votable_writer import (
    _unit_text_for_check,
    write_vo_lightcurve,
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
    "assign_photometry_column_semantics",
    "print_col_ucd",
    "DEFAULT_TIME_EPOCH_UCD",
    "DEFAULT_LABEL_UCD",
    "LABEL_UCD_FRAGMENTS",
]


def _origin_of_offsets(values, timesys):
    """Returns, per value, the time origin to add (0 for a full Julian Date).

    Args:
        values (numpy.ndarray): Offsets from ``timesys.timeorigin`` or full Julian Dates.
        timesys (TimeSys): Time system of the values.

    Returns:
        numpy.ndarray: Origin for each value; ``0.0`` where the value is already a full
            Julian Date (at or above ``TIME_OFFSET_ABSOLUTE_JD_THRESHOLD``) or not finite.

    Raises:
        ValueError: When the time origin is not declared although a value is an offset.
    """
    from volightcurve.time_reference import TIME_OFFSET_ABSOLUTE_JD_THRESHOLD

    is_offset = np.isfinite(values) & (values < TIME_OFFSET_ABSOLUTE_JD_THRESHOLD)
    if is_offset.any() and timesys.timeorigin is None:
        raise ValueError(
            "The time origin (TIMESYS timeorigin) is not declared, so time offsets "
            "cannot be converted to absolute Julian Date; it is not assumed."
        )
    return np.where(is_offset, float(timesys.timeorigin or 0.0), 0.0)


def _jd_from_offsets(values, timesys):
    """Returns absolute Julian Dates (floats) for time values and their time system.

    The declared time origin is added to every offset; nothing else is done and no time
    scale is needed, because adding the origin does not depend on the scale.

    Args:
        values (numpy.ndarray): Offsets from ``timesys.timeorigin`` or full Julian Dates.
        timesys (TimeSys): Time system of the values.

    Returns:
        numpy.ndarray: Julian Dates, float64.

    Raises:
        ValueError: When the time origin is not declared although a value is an offset.
    """
    values = np.asarray(values, dtype=float)
    return _origin_of_offsets(values, timesys) + values


def _time_from_offsets(values, timesys):
    """Builds an Astropy ``Time`` from time values and the time system that governs them.

    Origin and value stay in two parts, so no precision is lost. A ``Time`` carries a time
    scale, so the scale must be declared. Nothing is assumed.

    Args:
        values (numpy.ndarray): Offsets from ``timesys.timeorigin`` or full Julian Dates.
        timesys (TimeSys): Time system of the values.

    Returns:
        astropy.time.Time: Times, ``format="jd"``, in the declared time scale.

    Raises:
        ValueError: When the time scale is not declared or is not an Astropy time scale,
            or the time origin is not declared although a value is an offset.
    """
    from astropy.time import Time

    if timesys.timescale is None:
        raise ValueError("The time scale (TIMESYS timescale) is not declared; it is not assumed.")
    scale = str(timesys.timescale).strip().lower()
    if scale not in Time.SCALES:
        raise ValueError(
            f"The declared time scale {timesys.timescale!r} is not an Astropy time scale "
            f"({', '.join(Time.SCALES)})."
        )
    values = np.asarray(values, dtype=float)
    return Time(_origin_of_offsets(values, timesys), values, format="jd", scale=scale)


class VOLightCurve:
    """Represents a Virtual Observatory (VO) lightcurve container.

    Encapulates an Astropy Table containing timing and photometric observations,
    along with associated coordinate system, time system, and photometric calibration
    metadata mapped to specific columns. It supports files in both VOTable and ASCII
    heuristic formats.

    Time: a time column holds offsets from ``timesys.timeorigin``. Get absolute times with
    ``get_observation_times`` (Astropy ``Time``), ``get_observation_jd`` or
    ``get_observation_mjd`` (float arrays). The source of the data (TESS, Gaia, ZTF, ...)
    never decides a time shift; only the declared time system does.
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

        from volightcurve.interpretation import interpret

        interpret(self, source="votable")

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
        from volightcurve.interpretation import interpret

        interpret(instance, source="table")
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

        from volightcurve.interpretation import interpret, _recover_lc_colnames

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

        interpret(self, source="table")

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
                if photcal.zp_flux is not None and photcal.zero_point._zp_flux_unit_text is not None:
                    add(f"photcal:{column}:zeroPointFlux", photcal.zero_point._zp_flux_unit_text)
                if photcal.zp_mag is not None and photcal.zero_point._zp_mag_unit_text is not None:
                    add(f"photcal:{column}:zeroPointReferenceMagnitude", photcal.zero_point._zp_mag_unit_text)
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
                    "offsets cannot be converted to absolute Julian Date "
                    "(get_observation_times needs it)."
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

    def get_observation_times(self, column=None):
        """Returns the observation times of a time column as an Astropy ``Time``.

        The only supported way to obtain absolute times from a product. The column holds
        offsets from the declared ``TIMESYS`` time origin, so the origin is added here. The
        source (TESS, Gaia, ZTF, ...) never decides a shift: only the declared time origin and
        time scale do, whatever the archive's native convention. A value at or above
        ``TIME_OFFSET_ABSOLUTE_JD_THRESHOLD`` is taken as a full Julian Date, as on ingest.
        Origin and values are kept in two parts, so no precision is lost.

        The reference position (``BARYCENTER``, ``HELIOCENTER``, ...) is not part of an
        Astropy ``Time``; read it from ``self.timesys.refposition`` and report it.

        Args:
            column (str, optional): Time column name. Defaults to the first column found by
                ``get_time_colnames``.

        Returns:
            astropy.time.Time: Times in the declared time scale, ``format="jd"``. A
                non-finite value stays non-finite.

        Raises:
            ValueError: When there is no time column, the time origin is not declared for an
                offset, the time scale is not declared, or it is not an Astropy time scale.
        """
        name, timesys = self._time_column_and_system(column)
        return _time_from_offsets(np.asarray(self.table[name], dtype=float), timesys)

    def _time_column_and_system(self, column=None):
        """Resolves a time column and the ``TIMESYS`` that governs it.

        Args:
            column (str, optional): Time column name. Defaults to the first time column.

        Returns:
            tuple[str, TimeSys]: Column name and its time system.

        Raises:
            ValueError: When there is no time column, ``column`` is not one, or its
                ``TIMESYS`` reference is unknown or ambiguous.
        """
        from volightcurve.time_reference import resolve_observation_timesys

        time_cols = self.get_time_colnames()
        if not time_cols:
            raise ValueError("No observation time column (UCD time.epoch) was found.")
        name = time_cols[0] if column is None else column
        if name not in time_cols:
            raise ValueError(f"{name!r} is not a time column; time columns: {time_cols}.")
        ref_id = (self.field_timesys_ref or {}).get(name)
        if ref_id:
            timesys = self.timesys_by_id.get(ref_id)
            if timesys is None:
                raise ValueError(f"Time column {name!r} references unknown TIMESYS {ref_id!r}.")
        else:
            timesys = resolve_observation_timesys(self)
        return name, timesys

    def get_observation_jd(self, column=None):
        """Returns the observation times as absolute Julian Dates, plain floats.

        The declared ``TIMESYS`` time origin is added to the column offsets, nothing else
        (the source never decides a shift). No time scale is needed for the number; the scale
        is in ``timesys.timescale`` and ``time_problems()`` reports it when it is missing. A
        float64 Julian Date resolves about 40 microseconds; use ``get_observation_times`` when
        that matters.

        Args:
            column (str, optional): Time column name. Defaults to the first time column.

        Returns:
            numpy.ndarray: Julian Dates, float64.

        Raises:
            ValueError: When there is no time column, or the time origin is not declared for
                an offset.
        """
        name, timesys = self._time_column_and_system(column)
        return _jd_from_offsets(self.table[name], timesys)

    def get_observation_mjd(self, column=None):
        """Returns the observation times as Modified Julian Dates, plain floats.

        The declared ``TIMESYS`` time origin is added to the column offsets, nothing else
        (the source never decides a shift); then ``JD_TO_MJD`` is subtracted. No time scale is
        needed for the number; see ``timesys.timescale`` and ``time_problems()``.

        Args:
            column (str, optional): Time column name. Defaults to the first time column.

        Returns:
            numpy.ndarray: Modified Julian Dates, float64.

        Raises:
            ValueError: As ``get_observation_jd``.
        """
        from volightcurve.time_reference import JD_TO_MJD

        return self.get_observation_jd(column) - JD_TO_MJD

    def _epoch_and_system(self):
        """Returns the raw folding epoch and the ``TIMESYS`` that governs it.

        Returns:
            tuple[float, TimeSys] or None: ``None`` when the product declares no epoch.

        Raises:
            ValueError: When the time system of the epoch is ambiguous or unknown.
        """
        from volightcurve.time_reference import resolve_timesys_for_table_param

        raw = self.table.meta.get("epoch")
        if raw is None:
            return None
        return float(raw), resolve_timesys_for_table_param(self, "epoch")

    def get_epoch_time(self):
        """Returns the folding epoch of the product as an Astropy ``Time``, or ``None``.

        The epoch is the table PARAM ``epoch`` (``table.meta["epoch"]``). Like a time column
        it is an offset from the declared ``TIMESYS`` time origin (the PARAM's own ``ref``,
        else the system of the time column); the declared origin is added, nothing else. The
        source of the data never decides a shift. A value at or above
        ``TIME_OFFSET_ABSOLUTE_JD_THRESHOLD`` is taken as a full Julian Date.

        Returns:
            astropy.time.Time or None: Scalar epoch in the declared time scale, ``None``
                when the product declares no epoch.

        Raises:
            ValueError: When the time system of the epoch is ambiguous or unknown, its time
                scale is not declared, or its time origin is not declared for an offset.
        """
        found = self._epoch_and_system()
        if found is None:
            return None
        raw, timesys = found
        return _time_from_offsets(np.asarray(raw), timesys)

    def get_epoch_jd(self):
        """Returns the folding epoch as an absolute Julian Date, or ``None``.

        The declared time origin is added, nothing else. No time scale is needed for the
        number (see ``timesys.timescale`` and ``time_problems()``).

        Returns:
            float or None: Julian Date; ``None`` when the product declares no epoch.

        Raises:
            ValueError: When the time system of the epoch is ambiguous or unknown, or its
                time origin is not declared for an offset.
        """
        found = self._epoch_and_system()
        if found is None:
            return None
        raw, timesys = found
        return float(_jd_from_offsets(np.asarray(raw), timesys))

    def get_epoch_mjd(self):
        """Returns the folding epoch as an absolute Modified Julian Date, or ``None``.

        The declared time origin is added, nothing else; then ``JD_TO_MJD`` is subtracted. No
        time scale is needed for the number.

        Returns:
            float or None: Modified Julian Date; ``None`` when the product declares no epoch.

        Raises:
            ValueError: As ``get_epoch_jd``.
        """
        from volightcurve.time_reference import JD_TO_MJD

        jd = self.get_epoch_jd()
        return None if jd is None else jd - JD_TO_MJD

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
        "assign_label_column_ucd",
        "guess_label_colnames",
        "is_magnitude_phot_column",
    }
)


_MOVED_TO_SPLIT_MODULES = {
    "volightcurve.model": (
        "PhotometryFilter", "PhotCal", "PhotDM", "CooSys", "TimeSys",
    ),
    "volightcurve.votable_reader": (
        "_apply_gavo_votable_metadata", "_normalize_photdm_column_key",
        "_link_photdm_field_refs", "_declared_float", "_declared_text",
        "extract_photdm", "_list_votable_table_ids", "_read_votable_payload",
        "_clean_text", "_declared_infos", "_gavo_votable_metadata_tree",
        "_gavo_votable_tree_from_source", "extract_timesys", "extract_coosys",
    ),
    "volightcurve.columns": (
        "find_columns_by_ucd", "get_time_colnames", "DEFAULT_TIME_EPOCH_UCD",
        "NON_MEASUREMENT_UCD_WORDS", "is_measurement_ucd", "get_mag_colnames",
        "get_flux_colnames", "is_mag_column", "is_flux_column",
        "LABEL_UCD_FRAGMENTS", "DEFAULT_LABEL_UCD", "_column_ucd",
        "column_main_role", "_is_per_row_stamp", "get_label_colnames",
        "get_error_colnames",
        "assign_photometry_column_semantics", "print_col_ucd",
    ),
    "volightcurve.votable_writer": (
        "_unit_text_for_check", "_force_vounits_on_wire",
        "_repair_votable_xml_char_param_ampersands", "_photcal_params_from_photdm",
        "_photcal_group_specs", "_append_photcal_group", "write_vo_lightcurve",
    ),
}
_MOVED_LOOKUP = {
    name: module for module, names in _MOVED_TO_SPLIT_MODULES.items() for name in names
}


def __getattr__(name):
    """Serves names moved to the split modules (compatibility, one release).

    Names moved to ``volightcurve.interpretation`` and to ``volightcurve.model``,
    ``votable_reader``, ``columns`` and ``votable_writer`` stay importable from here.

    Args:
        name (str): Attribute requested from this module.

    Returns:
        object: The moved function, class or constant.

    Raises:
        AttributeError: When ``name`` was not moved.
    """
    import importlib

    if name in _MOVED_TO_INTERPRETATION:
        return getattr(importlib.import_module("volightcurve.interpretation"), name)
    if name in _MOVED_LOOKUP:
        return getattr(importlib.import_module(_MOVED_LOOKUP[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
