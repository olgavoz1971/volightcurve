"""Public lightcurve file read/write API (Ticket 8 Phase 1).

``read_lightcurve`` / ``write_lightcurve`` own the first and last file steps.
No UI framework imports. See ``docs/io_contract.md``.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

from astropy.table import Table

from volightcurve.io_dat import read_dat_table
from volightcurve.io_errors import LightcurveIOError
from volightcurve.io_meta import (
    apply_calibration_to_table_meta,
    calibration_dict_from_volc,
    format_keyword_comment_lines,
    free_text_comments,
    narrative_file_comments,
)
from volightcurve.lightcurve import VOLightCurve, write_vo_lightcurve
from volightcurve.time_reference import JD_TO_MJD

import numpy as np

logger = logging.getLogger(__name__)

_EXT_TO_FORMAT = {
    "vot": "votable",
    "xml": "votable",
    "dat": "dat",
    "csv": "csv",
    "ecsv": "ecsv",
}

_FORMAT_ALIASES = {
    "votable": "votable",
    "votable_binary": "votable",
    "votable_text": "votable",
    "dat": "dat",
    "ascii.commented_header": "dat",
    "csv": "csv",
    "ecsv": "ecsv",
    "ascii.ecsv": "ecsv",
}


def normalise_io_format(format_id: str | None, *, filename: str | None = None) -> str:
    """Resolves a write/read format id or filename extension to a codec name.

    Args:
        format_id (str, optional): UI or Astropy-style format string.
        filename (str, optional): Used when ``format_id`` is omitted.

    Returns:
        str: One of ``votable``, ``dat``, ``csv``, ``ecsv``.

    Raises:
        LightcurveIOError: When the format cannot be determined.
    """
    if format_id:
        key = str(format_id).strip().lower()
        if key in _FORMAT_ALIASES:
            return _FORMAT_ALIASES[key]
        raise LightcurveIOError(f"Unsupported lightcurve format '{format_id}'.")
    if filename:
        ext = Path(filename).suffix.lower().lstrip(".")
        if ext in _EXT_TO_FORMAT:
            return _EXT_TO_FORMAT[ext]
        raise LightcurveIOError(
            f"Cannot determine lightcurve format from filename '{filename}'."
        )
    raise LightcurveIOError("Lightcurve format or filename is required.")


def _split_hash_preamble(text: str) -> tuple[list[str], str]:
    """Splits leading ``#`` comment lines from the remainder of a text file.

    Args:
        text (str): Full file text.

    Returns:
        tuple: ``(comments_without_hash, remaining_text)``.
    """
    comments: list[str] = []
    body_lines: list[str] = []
    in_preamble = True
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if in_preamble and (not stripped or stripped.startswith("#")):
            if stripped.startswith("#"):
                comments.append(stripped.lstrip("#").strip())
            continue
        in_preamble = False
        body_lines.append(line)
    return comments, "".join(body_lines)


def _read_csv_with_comments(file_source) -> Table:
    """Reads CSV text, preserving a leading ``#`` calibration preamble.

    Args:
        file_source: Path or stream.

    Returns:
        astropy.table.Table: Table with ``meta['comments']`` when present.
    """
    if hasattr(file_source, "seek"):
        file_source.seek(0)
    if hasattr(file_source, "read"):
        raw = file_source.read()
    else:
        raw = Path(file_source).read_bytes()
    text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
    comments, body = _split_hash_preamble(text)
    table = Table.read(io.BytesIO(body.encode("utf-8")), format="csv")
    if comments:
        if table.meta is None:
            table.meta = {}
        table.meta["comments"] = comments
    return table


def read_lightcurve(
    source,
    *,
    filename: str | None = None,
    format: str | None = None,
    table_id: str | int | None = None,
) -> VOLightCurve:
    """Reads a lightcurve file into a ``VOLightCurve`` (first file step).

    Args:
        source: Path or readable binary/text stream.
        filename (str, optional): Original name for extension detection.
        format (str, optional): Explicit format id (overrides extension).
        table_id (str or int, optional): Embedded VOTable table selector.

    Returns:
        VOLightCurve: Parsed lightcurve with timescale and photometry metadata.

    Raises:
        LightcurveIOError: On unsupported format or parse failure.
    """
    codec = normalise_io_format(format, filename=filename)
    logger.info("read_lightcurve codec=%s filename=%s", codec, filename)

    if codec == "votable":
        if hasattr(source, "seek"):
            source.seek(0)
        return VOLightCurve(source, table_id=table_id)

    if codec == "dat":
        if hasattr(source, "seek"):
            source.seek(0)
        return VOLightCurve.from_table(read_dat_table(source))

    if codec == "csv":
        if hasattr(source, "seek"):
            source.seek(0)
        return VOLightCurve.from_table(_read_csv_with_comments(source))

    if codec == "ecsv":
        if hasattr(source, "seek"):
            source.seek(0)
        table = Table.read(source, format="ascii.ecsv")
        return VOLightCurve.from_table(table)

    raise LightcurveIOError(f"Unsupported lightcurve codec '{codec}'.")


def _votable_binary_flag(format_id: str | None, *, binary: bool) -> bool:
    """Resolves VOTable BINARY vs TABLEDATA encoding.

    Args:
        format_id (str, optional): Original format string from the caller.
        binary (bool): Default when format does not specify encoding.

    Returns:
        bool: True for binary VOTable encoding.
    """
    if format_id:
        key = str(format_id).strip().lower()
        if key == "votable_text":
            return False
        if key == "votable_binary":
            return True
    return binary


def _write_comment_header_table(
    table: Table,
    calibration: dict[str, Any],
    *,
    delimiter: str,
) -> bytes:
    """Serialises a table with ``# KEY = value`` preamble and data rows.

    Args:
        table (Table): Data columns to write.
        calibration (dict): Canonical calibration keywords.
        delimiter (str): Field delimiter (space for ``.dat``, comma for CSV).

    Returns:
        bytes: UTF-8 file payload.
    """
    lines: list[str] = []
    for assignment in format_keyword_comment_lines(calibration):
        lines.append(f"# {assignment}")

    comments = free_text_comments((table.meta or {}).get("comments"))
    # Skip column-header duplicate: if a free comment matches colnames, keep it once.
    col_header = " ".join(table.colnames) if delimiter == " " else ",".join(table.colnames)
    emitted_header = False
    for comment in comments:
        if comment.split() == list(table.colnames) or comment.replace(" ", "") == col_header.replace(
            " ", ""
        ):
            lines.append(f"# {comment}")
            emitted_header = True
            continue
        # Skip free-text that looks like a bare column header already handled.
        if set(comment.split()) == set(table.colnames) and len(comment.split()) == len(
            table.colnames
        ):
            if not emitted_header:
                lines.append(f"# {comment}")
                emitted_header = True
            continue
        lines.append(f"# {comment}")

    if not emitted_header and delimiter == " ":
        lines.append(f"# {' '.join(table.colnames)}")

    buf = io.StringIO()
    # Write data without Astropy rewriting comments into the body.
    data_only = table.copy()
    if data_only.meta is not None:
        data_only.meta = {}
    if delimiter == " ":
        data_only.write(buf, format="ascii.no_header", delimiter=" ")
    else:
        # Include CSV header row.
        data_only.write(buf, format="csv")
    body = buf.getvalue()
    if delimiter == "," and not emitted_header:
        # CSV already has a header row from Astropy; preamble only has KEY lines.
        pass
    payload = "\n".join(lines) + ("\n" if lines else "") + body
    if not payload.endswith("\n"):
        payload += "\n"
    return payload.encode("utf-8")


def assemble_volightcurve(
    table: Table,
    *,
    timeorigin: float = 0.0,
    period: float | None = None,
    epoch: float | None = None,
    filter_id: str | None = None,
    filter_name: str | None = None,
    zp_flux: float | None = None,
    zp_flux_unit: str | None = None,
    zp_mag: float | None = None,
    zp_mag_unit: str | None = None,
    mag_sys: str | None = None,
    effective_wavelength: float | None = None,
    effective_wavelength_unit: str | None = None,
    free_comments: list[str] | None = None,
) -> VOLightCurve:
    """Builds a ``VOLightCurve`` from a table and explicit calibration fields.

    Writes flat meta keys, then runs ``VOLightCurve.from_table`` so ingest and
    export share one calibration path. Does not invent missing zero points.

    Args:
        table (Table): Photometry table (copied).
        timeorigin (float): ``JD0`` matching the time column.
        period (float, optional): Folding period in days.
        epoch (float, optional): Epoch in the same origin as the time column.
        filter_id (str, optional): Filter identifier.
        filter_name (str, optional): Human-readable filter name.
        zp_flux (float, optional): Zero-point flux.
        zp_flux_unit (str, optional): Zero-point flux unit (``None`` = dimensionless).
        zp_mag (float, optional): Zero-point magnitude.
        zp_mag_unit (str, optional): Zero-point magnitude unit.
        mag_sys (str, optional): Magnitude system.
        effective_wavelength (float, optional): Filter spectral location.
        effective_wavelength_unit (str, optional): Unit for that wavelength.
        free_comments (list, optional): Narrative ``#`` lines (not calibration).

    Returns:
        VOLightCurve: Assembled instance ready for ``write_lightcurve``.
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
    from volightcurve.vo_unit_codec import to_wire

    tab = table.copy()
    if tab.meta is None:
        tab.meta = {}
    if free_comments:
        existing = list(tab.meta.get("comments") or [])
        tab.meta["comments"] = existing + [str(c) for c in free_comments if str(c).strip()]
    tab.meta[KEY_JD0] = float(timeorigin)
    if period is not None:
        tab.meta[KEY_PERIOD] = float(period)
    if epoch is not None:
        tab.meta[KEY_EPOCH] = float(epoch)
    if filter_id:
        tab.meta[KEY_FILTER] = str(filter_id)
    if filter_name:
        tab.meta[KEY_FILTER_NAME] = str(filter_name)
    if zp_flux is not None:
        tab.meta[KEY_ZP_FLUX] = float(zp_flux)
        tab.meta[KEY_ZP_FLUX_UNIT] = to_wire(zp_flux_unit)
    if zp_mag is not None:
        tab.meta[KEY_ZP_MAG] = float(zp_mag)
        if zp_mag_unit:
            tab.meta[KEY_ZP_MAG_UNIT] = str(zp_mag_unit)
    if mag_sys:
        tab.meta[KEY_MAG_SYS] = str(mag_sys)
    if effective_wavelength is not None:
        tab.meta[KEY_EFFECTIVE_WAVELENGTH] = float(effective_wavelength)
        if effective_wavelength_unit:
            tab.meta[KEY_EFFECTIVE_WAVELENGTH_UNIT] = str(effective_wavelength_unit)
    return VOLightCurve.from_table(tab)


def _photometry_columns_for_votable(tab: Table) -> tuple[str, str | None]:
    """Resolves photometry and error column names for VOTable export.

    Args:
        tab (Table): Source table (``phot``/``flux``/``mag`` naming).

    Returns:
        tuple: ``(phot_col, err_col_or_None)``.

    Raises:
        LightcurveIOError: When no photometry column is present.
    """
    if "phot" in tab.colnames:
        err = "flux_error" if "flux_error" in tab.colnames else None
        if err is None and "phot_err" in tab.colnames:
            err = "phot_err"
        return "phot", err
    if "mag" in tab.colnames:
        err = "mag_err" if "mag_err" in tab.colnames else None
        return "mag", err
    if "flux" in tab.colnames:
        err = "flux_err" if "flux_err" in tab.colnames else None
        return "flux", err
    raise LightcurveIOError("Cannot export VOTable: no photometry column.")


def _table_for_votable_codec(volc: VOLightCurve) -> tuple[Table, float]:
    """Builds a VOTable-shaped table and TIMESYS timeorigin from a product.

    Canonical assembly uses absolute JD (``JD0 = 0``) with ``jd`` / ``flux|mag``
    columns. VOTable writing remaps to ``obs_time`` (MJD) and ``phot`` here —
    the only format-specific table transform, at write time.

    Args:
        volc (VOLightCurve): Assembled lightcurve (any supported column layout).

    Returns:
        tuple: ``(table, timeorigin)`` for ``write_vo_lightcurve``.

    Raises:
        LightcurveIOError: When a time or photometry column is missing.
    """
    src = volc.table
    meta = dict(src.meta or {})
    stored_origin = 0.0
    if volc.timesys is not None and volc.timesys.timeorigin is not None:
        stored_origin = float(volc.timesys.timeorigin)

    if "obs_time" in src.colnames:
        out = src.copy()
        out.meta = meta
        return out, stored_origin

    time_col = None
    for name in ("jd", "time"):
        if name in src.colnames:
            time_col = name
            break
    if time_col is None:
        raise LightcurveIOError("Cannot export VOTable: no time column.")

    phot_col, err_col = _photometry_columns_for_votable(src)
    times = np.asarray(src[time_col], dtype=float)

    # Absolute JD (JD0 ≈ 0) → MJD column with standard MJD timeorigin.
    if abs(stored_origin) < 1e-9:
        obs_time = times - JD_TO_MJD
        write_origin = float(JD_TO_MJD)
        if meta.get("epoch") is not None:
            try:
                meta["epoch"] = float(meta["epoch"]) - JD_TO_MJD
            except (TypeError, ValueError):
                pass
    else:
        obs_time = times
        write_origin = stored_origin

    out = Table()
    out["obs_time"] = obs_time
    out["phot"] = np.asarray(src[phot_col], dtype=float)
    if src[phot_col].unit is not None:
        out["phot"].unit = src[phot_col].unit
    if err_col is not None:
        out["flux_error"] = np.asarray(src[err_col], dtype=float)
        if src[err_col].unit is not None:
            out["flux_error"].unit = src[err_col].unit
    if "label" in src.colnames:
        out["label"] = src["label"]
    out.meta = meta
    return out, write_origin


def _votable_kwargs_from_volc(
    volc: VOLightCurve,
    calibration: dict[str, Any],
    table_data: Table,
    timeorigin: float,
) -> dict[str, Any]:
    """Infers ``write_vo_lightcurve`` kwargs from the assembled product only.

    Args:
        volc (VOLightCurve): Source product.
        calibration (dict): Keyword map from ``calibration_dict_from_volc``.
        table_data (Table): VO-shaped table (may carry adjusted epoch).
        timeorigin (float): TIMESYS time origin for the ``obs_time`` column.

    Returns:
        dict: Keyword arguments for ``write_vo_lightcurve``.
    """
    meta = dict(table_data.meta or {})
    envelope = dict(meta.get("vo_envelope") or {})
    comments = narrative_file_comments(meta.get("comments"))
    description = (
        envelope.get("votable_description")
        or envelope.get("table_description")
        or (" ".join(comments) if comments else None)
    )
    table_name = (
        envelope.get("table_name")
        or meta.get("name")
        or meta.get("ID")
        or "lightcurve"
    )
    kwargs: dict[str, Any] = {
        "table_name": str(table_name),
        "timeorigin": float(timeorigin),
    }
    filt = calibration.get("FILTER")
    if not filt:
        raise LightcurveIOError(
            "Cannot export VOTable: filter identifier is missing."
        )
    kwargs["filter_identifier"] = str(filt)

    if envelope.get("refposition"):
        kwargs["refposition"] = str(envelope["refposition"])
    elif volc.timesys is not None and volc.timesys.refposition:
        kwargs["refposition"] = str(volc.timesys.refposition)

    if envelope.get("timescale"):
        kwargs["timescale"] = str(envelope["timescale"]).upper()
    elif volc.timesys is not None and volc.timesys.timescale:
        kwargs["timescale"] = str(volc.timesys.timescale).upper()

    if description:
        kwargs["votable_description"] = description
        kwargs["table_description"] = (
            envelope.get("table_description") or description
        )
    if envelope.get("creator"):
        kwargs["creator"] = str(envelope["creator"])
    if meta.get("ra") is not None:
        kwargs["ra"] = meta["ra"]
    if meta.get("dec") is not None:
        kwargs["dec"] = meta["dec"]
    if meta.get("period") is not None:
        kwargs["period"] = float(meta["period"])
    if meta.get("epoch") is not None:
        kwargs["epoch"] = float(meta["epoch"])
    if calibration.get("FILTER_NAME"):
        kwargs["filter_name"] = str(calibration["FILTER_NAME"])
    if "ZP_FLUX" in calibration:
        kwargs["zero_point_flux"] = calibration["ZP_FLUX"]
        kwargs["zero_point_flux_unit"] = calibration.get("ZP_FLUX_UNIT")
    if "ZP_MAG" in calibration:
        kwargs["zero_point_ref_mag"] = calibration["ZP_MAG"]
        kwargs["zero_point_ref_mag_unit"] = calibration.get("ZP_MAG_UNIT", "mag")
    if "MAG_SYS" in calibration:
        kwargs["magnitude_system"] = calibration["MAG_SYS"]
    if "EFFECTIVE_WAVELENGTH" in calibration:
        kwargs["effective_wavelength"] = calibration["EFFECTIVE_WAVELENGTH"]
        if calibration.get("EFFECTIVE_WAVELENGTH_UNIT"):
            kwargs["effective_wavelength_unit"] = calibration[
                "EFFECTIVE_WAVELENGTH_UNIT"
            ]
    if envelope.get("publication_id"):
        kwargs["publication_id"] = str(envelope["publication_id"])
    if envelope.get("coosys_system"):
        kwargs["coosys_system"] = envelope["coosys_system"]
        if envelope.get("coosys_epoch") is not None:
            kwargs["coosys_epoch"] = envelope["coosys_epoch"]
        if envelope.get("coosys_id"):
            kwargs["coosys_id"] = envelope["coosys_id"]
    return kwargs


def write_lightcurve(
    volc: VOLightCurve,
    format: str,
    destination=None,
    *,
    binary: bool = True,
    **votable_kwargs: Any,
) -> bytes:
    """Writes a ``VOLightCurve`` to the requested format (last file step).

    Photcal, narrative comments, and provenance envelope must already live on
    ``volc``. Format branching (VOTable column remap vs ``#`` comments) happens
    only here.

    Args:
        volc (VOLightCurve): Lightcurve to serialise.
        format (str): Format id (``votable_binary``, ``ascii.ecsv``, ``csv``,
            ``ascii.commented_header``, …).
        destination: Optional path or binary stream; when omitted, bytes are
            returned only.
        binary (bool, optional): Default VOTable encoding when format is generic
            ``votable``.
        **votable_kwargs: Optional overrides for ``write_vo_lightcurve`` (tests /
            advanced callers). Prefer storing fields on ``volc`` instead.

    Returns:
        bytes: Serialised file content (also written to ``destination`` when set).

    Raises:
        LightcurveIOError: On unsupported format or missing VOTable requirements.
    """
    if not isinstance(volc, VOLightCurve):
        raise LightcurveIOError("write_lightcurve expects a VOLightCurve instance.")

    codec = normalise_io_format(format)
    calibration = calibration_dict_from_volc(volc)
    logger.info("write_lightcurve codec=%s keys=%s", codec, sorted(calibration))

    if codec == "votable":
        table_data, timeorigin = _table_for_votable_codec(volc)
        kwargs = _votable_kwargs_from_volc(
            volc, calibration, table_data, timeorigin
        )
        kwargs.update({k: v for k, v in votable_kwargs.items() if v is not None})
        kwargs.setdefault("binary", _votable_binary_flag(format, binary=binary))
        buf = io.BytesIO()
        write_vo_lightcurve(
            output_stream_or_path=buf,
            table_data=table_data,
            **kwargs,
        )
        payload = buf.getvalue()
    elif codec == "ecsv":
        tab = volc.table.copy()
        narrative = free_text_comments((tab.meta or {}).get("comments"))
        # Flat calibration keys only — drop leftover non-contract meta.
        tab.meta = {}
        apply_calibration_to_table_meta(tab, calibration)
        if narrative:
            tab.meta["comments"] = narrative
        buf = io.StringIO()
        tab.write(buf, format="ascii.ecsv", overwrite=True)
        payload = buf.getvalue().encode("utf-8")
    elif codec == "dat":
        payload = _write_comment_header_table(
            volc.table, calibration, delimiter=" "
        )
    elif codec == "csv":
        payload = _write_comment_header_table(
            volc.table, calibration, delimiter=","
        )
    else:
        raise LightcurveIOError(f"Unsupported lightcurve codec '{codec}'.")

    if destination is not None:
        if hasattr(destination, "write"):
            destination.write(payload)
        else:
            Path(destination).write_bytes(payload)
    return payload
