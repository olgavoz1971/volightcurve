"""Column discovery by UCD and column role helpers (no guessing from names)."""

import logging
from typing import TYPE_CHECKING

import numpy as np
from astropy.table import Table


if TYPE_CHECKING:
    from volightcurve.lightcurve import VOLightCurve

logger = logging.getLogger(__name__)


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

    The column holds offsets from the declared ``TIMESYS`` time origin. For absolute times
    use ``VOLightCurve.get_observation_times`` (Astropy ``Time``), ``get_observation_jd`` or
    ``get_observation_mjd``; do not add a mission offset by hand.

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
    """Returns the declared per-epoch label column, if any.

    Returns the leftmost column whose UCD contains ``meta.code`` or ``meta.id``.
    Columns that are unique on almost every row (observation dates) are skipped.
    Nothing is guessed: a table without a declared label column gives an empty
    list. The positional guess lives in ``volightcurve.interpretation``
    (``guess_label_colnames``). See ``docs/io_contract.md`` §8.

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
    return [labelled[0]] if labelled else []


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


def print_col_ucd(lc: "VOLightCurve"):
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
