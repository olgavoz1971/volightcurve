"""Strict ``.dat`` / commented-header table reader.

Parses whitespace-separated lightcurve ASCII with ``#`` comment metadata.
Raises ``LightcurveIOError`` on invalid input.
"""

from __future__ import annotations

import logging
from pathlib import Path

from astropy.table import Table

from volightcurve.io_errors import LightcurveIOError
from volightcurve.io_keywords import is_metadata_assignment_line

logger = logging.getLogger(__name__)

_LEGACY_DAT_COLUMN_COUNT = 3
_LEGACY_DAT_COLUMN_HINT = "jd, mag, mag_err"
_DAT_FREE_TEXT_COLUMNS = frozenset({"label", "sector", "flag"})


def _dat_column_label_hint(col_names: list[str] | None) -> str:
    """Builds a parenthetical column hint for strict ``.dat`` row errors.

    Args:
        col_names (list, optional): Header names from a ``#`` comment line.

    Returns:
        str: Comma-separated names or the legacy default hint.
    """
    if col_names:
        return ", ".join(col_names)
    return _LEGACY_DAT_COLUMN_HINT + " (legacy default)"


def _dat_column_accepts_free_text(col_name: str) -> bool:
    """True when a declared ``.dat`` column holds arbitrary strings.

    Args:
        col_name (str): Column name from the header comment.

    Returns:
        bool: True for ``label``, ``sector``, and ``flag`` (case-insensitive).
    """
    return col_name.lower() in _DAT_FREE_TEXT_COLUMNS


def _parse_dat_cell(col_name: str, token: str, line_no: int, stripped: str):
    """Parses one whitespace-separated ``.dat`` field for table construction.

    Args:
        col_name (str): Header column name for this field.
        token (str): Field text from the row.
        line_no (int): Source line number for errors.
        stripped (str): Full stripped row text for errors.

    Returns:
        float or str: Cell value for the Astropy column.

    Raises:
        LightcurveIOError: When a numeric column is not numeric.
    """
    if _dat_column_accepts_free_text(col_name):
        return token
    if token.lower() == "nan":
        return float("nan")
    try:
        return float(token)
    except ValueError as exc:
        raise LightcurveIOError(
            f"Invalid .dat row at line {line_no}: column {col_name!r} expects a "
            f"numeric value, got {token!r}.\n"
            f"Offending line: {stripped}"
        ) from exc


def _read_text_source(file_source) -> str:
    """Reads a path or stream as UTF-8 text.

    Args:
        file_source: Path or readable binary/text stream.

    Returns:
        str: Decoded file text.

    Raises:
        LightcurveIOError: When the payload is not valid UTF-8.
    """
    if hasattr(file_source, "seek"):
        file_source.seek(0)
    if hasattr(file_source, "read"):
        raw = file_source.read()
    else:
        raw = Path(file_source).read_bytes()
    if isinstance(raw, str):
        return raw
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise LightcurveIOError(
            "Invalid .dat file: content is not valid UTF-8 text."
        ) from exc


def read_dat_table(file_source) -> Table:
    """Reads a ``.dat`` upload with strict row validation and full ``#`` metadata.

    Comment lines are stored in ``table.meta['comments']`` (text without ``#``).
    Data rows must have a fixed token count: from the first non-metadata ``#``
    header whose word count matches the data width, or exactly three columns for
    legacy files. Column typing follows the header names (``label`` / ``sector`` /
    ``flag`` stay strings); promotion to VO standards runs in
    ``VOLightCurve.from_table``.

    Args:
        file_source: Path or readable binary stream.

    Returns:
        astropy.table.Table: Parsed rows with ``meta['comments']`` populated.

    Raises:
        LightcurveIOError: On encoding issues, empty files, or ragged rows.
    """
    text = _read_text_source(file_source)

    comments: list[str] = []
    data_rows: list[tuple[int, list[str], str]] = []

    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            comments.append(stripped.lstrip("#").strip())
            continue
        tokens = stripped.split()
        data_rows.append((line_no, tokens, stripped))

    if not data_rows:
        raise LightcurveIOError("Invalid .dat file: no data rows found.")

    first_line_no, first_tokens, _ = data_rows[0]
    n_data = len(first_tokens)
    for line_no, tokens, stripped in data_rows[1:]:
        if len(tokens) != n_data:
            raise LightcurveIOError(
                f"Invalid .dat row at line {line_no}: expected {n_data} columns "
                f"(same as first data row at line {first_line_no}), got {len(tokens)}.\n"
                f"Offending line: {stripped}"
            )

    header_names: list[str] | None = None
    header_candidates: list[list[str]] = []
    for comment in comments:
        if is_metadata_assignment_line(comment):
            continue
        parts = comment.split()
        if parts:
            header_candidates.append(parts)

    for parts in header_candidates:
        if len(parts) == n_data:
            header_names = parts
            break

    if header_names is None and header_candidates:
        declared = header_candidates[0]
        raise LightcurveIOError(
            f"Invalid .dat file: column header declares {len(declared)} columns "
            f"({', '.join(declared)}) but data rows have {n_data} columns "
            f"(first data row at line {first_line_no})."
        )

    if header_names is None:
        if n_data != _LEGACY_DAT_COLUMN_COUNT:
            raise LightcurveIOError(
                f"Invalid .dat file: no column header comment with {n_data} names "
                f"and data rows have {n_data} columns; legacy .dat expects "
                f"{_LEGACY_DAT_COLUMN_COUNT} ({_LEGACY_DAT_COLUMN_HINT}). "
                f"Add a line such as '# jd mag mag_err' or fix the column count."
            )
        col_names = [f"col{i + 1}" for i in range(_LEGACY_DAT_COLUMN_COUNT)]
    else:
        col_names = header_names

    n_expected = len(col_names)
    hint = _dat_column_label_hint(header_names)
    columns: dict[str, list] = {name: [] for name in col_names}

    for line_no, tokens, stripped in data_rows:
        if len(tokens) != n_expected:
            raise LightcurveIOError(
                f"Invalid .dat row at line {line_no}: expected {n_expected} columns "
                f"({hint}), got {len(tokens)}.\n"
                f"Offending line: {stripped}"
            )
        for name, token in zip(col_names, tokens):
            columns[name].append(_parse_dat_cell(name, token, line_no, stripped))

    table = Table([columns[name] for name in col_names], names=col_names)
    table.meta["comments"] = comments
    logger.debug("Read .dat table with %s rows and %s columns", len(table), len(col_names))
    return table
