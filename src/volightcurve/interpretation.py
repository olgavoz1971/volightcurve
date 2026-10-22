"""Interpretation (part B): bringing data that are not VO-compliant into the core model.

Everything here uses column names, ``#`` comments, ``KEY=value`` conventions or positions,
that is, evidence the file carries only in a loose form. Nothing is invented: a value that
no evidence supports stays absent (``None``). The strict core in ``lightcurve.py`` does not
import this module at import time; it calls it lazily from the tolerant reading path.

The old private names stay importable from ``volightcurve.lightcurve`` for one release.
"""

from __future__ import annotations

import logging
import re

from typing import TYPE_CHECKING

import astropy.units as u

from volightcurve.lightcurve import (
    PhotCal,
    PhotDM,
    PhotometryFilter,
    assign_label_column_ucd,
    is_magnitude_phot_column,
)

if TYPE_CHECKING:
    from volightcurve.lightcurve import VOLightCurve

logger = logging.getLogger(__name__)


def add_interpretation_note(table, how: str, items: list[str]) -> None:
    """Records what interpretation did as one plain sentence in the table description.

    The sentence is appended to ``table.meta["description"]`` (the VOTable table
    DESCRIPTION; the text formats re-emit it as a comment line). It carries no
    ``KEY = value`` pattern, so a later read does not take it for a calibration keyword.
    A sentence already present is not added twice.

    Args:
        table (astropy.table.Table): Table whose description receives the sentence.
        how (str): Where the evidence came from, for example ``"from column names"``.
        items (list[str]): What was set, one short phrase each; nothing is written when empty.
    """
    if not items:
        return
    sentence = f"Interpreted from the file, not declared in it ({how}): {'; '.join(items)}."
    if table.meta is None:
        table.meta = {}
    current = str(table.meta.get("description") or "").strip()
    if sentence in current:
        return
    table.meta["description"] = f"{current} {sentence}".strip()


def _promote_to_vo_standards(table):
    """Heuristically assigns Unified Content Descriptors (UCDs) and physical units to table columns.

    This function scans column names for common patterns (e.g., 'mag', 'flux', 'time')
    and assigns UCDs, and units for ``mag`` and time columns, without renaming
    columns. A flux column without a unit is never given one. Every assignment is
    logged as a warning and recorded as a sentence in the table description
    (``add_interpretation_note``).

    Args:
        table (astropy.table.Table): The table to process.

    Returns:
        astropy.table.Table: The table with updated column units and UCD metadata.
    """
    promoted: list[str] = []
    for colname in table.colnames:
        col = table[colname]
        name_low = colname.lower()
        if not col.unit:
            if 'mag' in name_low:
                col.unit = u.mag
                promoted.append(f"column {colname} was given the unit mag")
            elif 'flux' in name_low:
                # A flux column without a unit stays without a unit: the
                # physical unit is never guessed (it may be normalised flux).
                pass
            elif any(k in name_low for k in ['time', 'jd', 'mjd']):
                col.unit = u.d
                promoted.append(f"column {colname} was given the unit d")

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
            promoted.append(f"column {colname} was given the UCD {ucd}")
    if promoted:
        add_interpretation_note(table, "from column names", promoted)
        logger.warning(
            "Column roles/units were assigned from column names, not from the "
            "file: %s",
            "; ".join(promoted),
        )
    return table


def _pickup_jd0_from_table(table):
    """Parses metadata comments to locate the time origin value (JD0).

    Searches the table comments for patterns like "JD0 = value".

    Args:
        table (astropy.table.Table): The table to scan.

    Returns:
        float or None: The parsed JD0 value if found, or ``None`` otherwise.
    """
    jd0_pattern = re.compile(r"JD0\s*=\s*([+-]?\d*\.?\d+)")
    for line in table.meta.get('comments', []):
        match = jd0_pattern.search(line.upper())
        if match: return float(match.group(1))
    return None


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
    renamed: list[str] = []
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
            renamed.append(f"column {colname} was named {new_name}")
    add_interpretation_note(
        table, "from the header comment" if found_header else "from column positions", renamed
    )
    return table


def apply_non_votable_heuristics(volc: "VOLightCurve") -> None:
    """Apply column promotion and comment/flat-meta calibration for non-VOTable tables.

    Mutates ``volc.table``, ``volc.timesys``, and ``volc.photdms`` in place.
    Uses the shared Ticket 8 vocabulary (``JD0``, ``ZP_*``, ``FILTER``, …) from
    ``#`` comments and flat ECSV ``meta`` keys. A declared ``MAG0`` or
    ``ZP_MAG`` is stored as the magnitude zero point. A missing ``ZP_FLUX``
    stays missing.

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
    from volightcurve.io_keywords import KEY_SENTINEL
    from volightcurve.io_meta import apply_declared_sentinels, coerce_censored_tokens

    sentinels = list(calibration.get(KEY_SENTINEL) or [])
    if sentinels:
        volc.table.meta[KEY_SENTINEL] = sentinels
    coerce_censored_tokens(volc.table)
    apply_declared_sentinels(volc.table, sentinels)

    read_items: list[str] = []
    if KEY_JD0 in calibration:
        volc.timesys.timeorigin = float(calibration[KEY_JD0])
    else:
        volc.timesys.timeorigin = _pickup_jd0_from_table(volc.table)
    if volc.timesys.timeorigin is not None:
        read_items.append(f"time origin {volc.timesys.timeorigin}")

    if KEY_PERIOD in calibration:
        volc.table.meta["period"] = float(calibration[KEY_PERIOD])
    else:
        period = _pickup_period_from_table(volc.table)
        if period is not None:
            volc.table.meta["period"] = period
    if volc.table.meta.get("period") is not None:
        read_items.append(f"period {volc.table.meta['period']}")

    if KEY_EPOCH in calibration:
        volc.table.meta["epoch"] = float(calibration[KEY_EPOCH])
    else:
        epoch = _pickup_epoch_from_table(volc.table)
        if epoch is not None:
            volc.table.meta["epoch"] = epoch
    if volc.table.meta.get("epoch") is not None:
        read_items.append(f"epoch {volc.table.meta['epoch']}")

    heur_filter_id = calibration.get(KEY_FILTER) or _pickup_filter_from_table(volc.table)
    if heur_filter_id:
        volc.table.meta["filter"] = heur_filter_id
        read_items.append(f"filter {heur_filter_id}")
    if KEY_FILTER_NAME in calibration:
        volc.table.meta["filter_name"] = str(calibration[KEY_FILTER_NAME])
        read_items.append(f"filter name {calibration[KEY_FILTER_NAME]}")

    has_zp_mag = KEY_ZP_MAG in calibration
    has_zp_flux = KEY_ZP_FLUX in calibration
    build_photcal = has_zp_mag or has_zp_flux
    if has_zp_mag:
        read_items.append(f"zero-point magnitude {calibration[KEY_ZP_MAG]}")
    if has_zp_flux:
        read_items.append(f"zero-point flux {calibration[KEY_ZP_FLUX]}")
    if build_photcal and calibration.get(KEY_MAG_SYS):
        read_items.append(f"magnitude system {calibration[KEY_MAG_SYS]}")
    if calibration.get(KEY_EFFECTIVE_WAVELENGTH) is not None:
        read_items.append(f"effective wavelength {calibration[KEY_EFFECTIVE_WAVELENGTH]}")
    if build_photcal:
        zp_mag = float(calibration[KEY_ZP_MAG]) if has_zp_mag else None
        zp_mag_unit = calibration.get(KEY_ZP_MAG_UNIT) or "mag"
        mag_sys = calibration.get(KEY_MAG_SYS) or None
        if has_zp_flux:
            zp_flux = float(calibration[KEY_ZP_FLUX])
            zp_flux_unit = calibration.get(KEY_ZP_FLUX_UNIT)
        else:
            zp_flux = None
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
            if heur_filter_id and not photdm.filter_id:
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

    add_interpretation_note(volc.table, "from comments or metadata keywords", read_items)
    label = assign_label_column_ucd(volc.table)
    if label:
        add_interpretation_note(
            volc.table, "from the column values", [f"column {label} was marked as the label column"]
        )


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
