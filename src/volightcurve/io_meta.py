"""Calibration keyword extract / apply helpers for non-VO lightcurve I/O.

Implements the shared vocabulary in ``io_keywords`` for comment lines and flat
ECSV ``meta`` keys. See ``docs/volightcurve_io_contract.md``.
"""

from __future__ import annotations

import logging
from typing import Any

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
    WRITE_KEYWORD_ORDER,
    is_metadata_assignment_line,
    normalise_keyword,
    parse_keyword_assignment,
)
from volightcurve.vo_unit_codec import to_internal, to_wire

logger = logging.getLogger(__name__)


def _coerce_float(raw: str | float | int | None) -> float | None:
    """Parses a numeric metadata value.

    Args:
        raw: Raw string or number from a keyword assignment.

    Returns:
        float | None: Parsed float, or ``None`` when empty / invalid.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        logger.warning("Ignoring non-numeric calibration value %r", raw)
        return None


def collect_calibration_from_comments(comments: list[str] | None) -> dict[str, Any]:
    """Collects first-class calibration keywords from ``#`` comment lines.

    Later assignments override earlier ones. Alias resolution uses
    ``parse_keyword_assignment`` (``MAG0`` → ``ZP_MAG``, ``BAND`` → ``FILTER``).

    Args:
        comments (list, optional): Comment texts without leading ``#``.

    Returns:
        dict: Canonical keyword → raw/parsed value (floats where numeric).
    """
    collected: dict[str, Any] = {}
    if not comments:
        return collected
    for line in comments:
        parsed = parse_keyword_assignment(line)
        if parsed is None:
            continue
        key, value = parsed
        if key in {
            KEY_JD0,
            KEY_EPOCH,
            KEY_PERIOD,
            KEY_ZP_MAG,
            KEY_ZP_FLUX,
            KEY_EFFECTIVE_WAVELENGTH,
        }:
            number = _coerce_float(value)
            if number is not None:
                collected[key] = number
        elif key == KEY_ZP_FLUX_UNIT:
            collected[key] = to_internal(value if value != "" else None)
        else:
            collected[key] = value
    return collected


def collect_calibration_from_flat_meta(meta: dict | None) -> dict[str, Any]:
    """Collects calibration keywords from flat table ``meta`` (ECSV).

    Args:
        meta (dict, optional): Astropy ``table.meta`` mapping.

    Returns:
        dict: Canonical keyword → value.
    """
    collected: dict[str, Any] = {}
    if not meta:
        return collected
    # Prefer ZP_MAG over MAG0 when both present (iterate aliases last via order).
    items = list(meta.items())
    for raw_key, raw_val in items:
        key_up = normalise_keyword(str(raw_key))
        # Reuse assignment parser by synthesising a line.
        line = f"{key_up} = {raw_val if raw_val is not None else ''}"
        parsed = parse_keyword_assignment(line)
        if parsed is None:
            continue
        key, value = parsed
        if key in {
            KEY_JD0,
            KEY_EPOCH,
            KEY_PERIOD,
            KEY_ZP_MAG,
            KEY_ZP_FLUX,
            KEY_EFFECTIVE_WAVELENGTH,
        }:
            number = _coerce_float(value)
            if number is not None:
                collected[key] = number
        elif key == KEY_ZP_FLUX_UNIT:
            collected[key] = to_internal(value if value != "" else None)
        else:
            collected[key] = value
    return collected


def merge_calibration_sources(
    comments: list[str] | None,
    meta: dict | None,
) -> dict[str, Any]:
    """Merges comment and flat-meta calibration; comments win on key clash.

    Args:
        comments (list, optional): ``table.meta['comments']`` lines.
        meta (dict, optional): Flat table meta (ECSV keys).

    Returns:
        dict: Merged canonical calibration map.
    """
    merged = collect_calibration_from_flat_meta(meta)
    from_comments = collect_calibration_from_comments(comments)
    merged.update(from_comments)
    return merged


def free_text_comments(comments: list[str] | None) -> list[str]:
    """Returns comment lines that are not first-class calibration assignments.

    Column-header-like lines (multiple words, not ``KEY=value``) are included so
    writers can re-emit them; callers that only want narrative may filter further.

    Args:
        comments (list, optional): Comment texts without leading ``#``.

    Returns:
        list[str]: Non-calibration comment lines.
    """
    if not comments:
        return []
    return [line for line in comments if not is_metadata_assignment_line(line)]


def narrative_file_comments(
    comments: list[str] | None,
    colnames: list[str] | None = None,
) -> list[str]:
    """Returns free-text comments excluding calibration and column-header lines.

    These are the lines the I/O contract stores as description and re-emits on
    download (not ``KEY = value``, not the ``# jd mag …`` header).

    Args:
        comments (list, optional): Comment texts without leading ``#``.
        colnames (list, optional): Table column names used to detect header lines.

    Returns:
        list[str]: Narrative comment lines only.
    """
    names = list(colnames) if colnames else None
    out: list[str] = []
    for line in free_text_comments(comments):
        parts = line.split()
        if names is not None and parts == names:
            continue
        if (
            names is not None
            and len(parts) == len(names)
            and set(parts) == set(names)
        ):
            continue
        out.append(line)
    return out


def format_keyword_comment_lines(calibration: dict[str, Any]) -> list[str]:
    """Formats calibration as ``KEY = value`` strings (no leading ``#``).

    Args:
        calibration (dict): Canonical keyword → value (omit absent keys).

    Returns:
        list[str]: Ordered assignment lines for write.
    """
    lines: list[str] = []
    for key in WRITE_KEYWORD_ORDER:
        if key not in calibration:
            continue
        value = calibration[key]
        if value is None and key != KEY_ZP_FLUX_UNIT:
            continue
        if key == KEY_ZP_FLUX_UNIT:
            wire = to_wire(to_internal(value))
            lines.append(f"{key} = {wire}")
            continue
        lines.append(f"{key} = {value}")
    return lines


def calibration_dict_from_volc(volc) -> dict[str, Any]:
    """Builds a calibration keyword dict from a ``VOLightCurve`` instance.

    Args:
        volc (VOLightCurve): Parsed or assembled lightcurve.

    Returns:
        dict: Canonical keywords present on the product (no invented values).
    """
    calibration: dict[str, Any] = {}
    timesys = getattr(volc, "timesys", None)
    if timesys is not None and timesys.timeorigin is not None:
        calibration[KEY_JD0] = float(timesys.timeorigin)

    meta = getattr(volc, "table", None)
    table_meta = (meta.meta if meta is not None else None) or {}
    if table_meta.get("period") is not None:
        calibration[KEY_PERIOD] = float(table_meta["period"])
    if table_meta.get("epoch") is not None:
        calibration[KEY_EPOCH] = float(table_meta["epoch"])

    photdms = getattr(volc, "photdms", None) or {}
    photdm = next(iter(photdms.values()), None)
    if photdm is None:
        if table_meta.get("filter"):
            calibration[KEY_FILTER] = str(table_meta["filter"])
        return calibration

    filter_id = getattr(photdm, "filter_id", None)
    if filter_id and str(filter_id) not in ("", "Unknown"):
        calibration[KEY_FILTER] = str(filter_id)

    filter_name = getattr(photdm, "filter_name", None)
    if filter_name:
        calibration[KEY_FILTER_NAME] = str(filter_name)

    phot_filter = getattr(photdm, "filter", None)
    if phot_filter is not None and getattr(phot_filter, "spectral_location", None) is not None:
        spectral = phot_filter.spectral_location
        try:
            calibration[KEY_EFFECTIVE_WAVELENGTH] = float(spectral.value)
            if spectral.unit is not None:
                calibration[KEY_EFFECTIVE_WAVELENGTH_UNIT] = str(spectral.unit)
        except Exception:
            logger.debug("Could not serialise spectral_location for write", exc_info=True)

    if KEY_FILTER_NAME not in calibration and table_meta.get("filter_name"):
        calibration[KEY_FILTER_NAME] = str(table_meta["filter_name"])

    photcal = getattr(photdm, "photcal", None)
    if photcal is None:
        return calibration

    if getattr(photcal, "zp_flux", None) is not None:
        try:
            calibration[KEY_ZP_FLUX] = float(photcal.zp_flux.value)
        except Exception:
            calibration[KEY_ZP_FLUX] = float(photcal.zp_flux)
        unit_text = getattr(photcal, "_zp_flux_unit_text", None)
        calibration[KEY_ZP_FLUX_UNIT] = to_internal(unit_text)
    if getattr(photcal, "zp_mag", None) is not None:
        try:
            calibration[KEY_ZP_MAG] = float(photcal.zp_mag.value)
        except Exception:
            calibration[KEY_ZP_MAG] = float(photcal.zp_mag)
        mag_unit = getattr(photcal, "_zp_mag_unit_text", None)
        calibration[KEY_ZP_MAG_UNIT] = str(mag_unit) if mag_unit else "mag"
    mag_sys = getattr(photcal, "mag_sys", None)
    if mag_sys:
        calibration[KEY_MAG_SYS] = str(mag_sys)
    return calibration


def apply_calibration_to_table_meta(table, calibration: dict[str, Any]) -> None:
    """Writes flat calibration keys onto ``table.meta`` for ECSV export.

    Args:
        table (astropy.table.Table): Table whose meta is updated in place.
        calibration (dict): Canonical keyword map.
    """
    if table.meta is None:
        table.meta = {}
    for key in WRITE_KEYWORD_ORDER:
        if key not in calibration:
            continue
        value = calibration[key]
        if key == KEY_ZP_FLUX_UNIT:
            table.meta[key] = to_wire(to_internal(value))
        elif value is not None:
            table.meta[key] = value
