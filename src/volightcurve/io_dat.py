"""Strict ``.dat`` / commented-header table reader.

Parses whitespace-separated lightcurve ASCII with ``#`` comment metadata.
Header selection, name→role typing, and structural failback follow
``docs/io_contract.md`` §4b. Raises ``LightcurveIOError`` on invalid input.
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
_FAILBACK_NAMES_3 = ("jd", "mag", "mag_err")
_FAILBACK_NAMES_4 = ("jd", "mag", "mag_err", "label")

DatColumnRole = str  # time | mag | flux | error | other


def dat_column_role(col_name: str) -> DatColumnRole:
    """Classifies a ``.dat`` header token into a photometric role by name sense.

    Args:
        col_name (str): Column name from a header comment.

    Returns:
        str: One of ``time``, ``mag``, ``flux``, ``error``, ``other``.
    """
    name_low = col_name.lower()
    if (
        name_low == "error"
        or "err" in name_low
        or "uncert" in name_low
        or "sigma" in name_low
    ):
        return "error"
    if any(k in name_low for k in ("time", "jd", "mjd")):
        return "time"
    if name_low == "phot" or "mag" in name_low or "magnitude" in name_low:
        return "mag"
    if "flux" in name_low:
        return "flux"
    return "other"


def _dat_layout_is_valid(roles: list[DatColumnRole]) -> bool:
    """True when role counts match the §4b structural check.

    Args:
        roles (list): Per-column roles in header order.

    Returns:
        bool: True when exactly one time, exactly one mag or flux, ≤1 error.
    """
    n_time = roles.count("time")
    n_mag = roles.count("mag")
    n_flux = roles.count("flux")
    n_error = roles.count("error")
    if n_time != 1:
        return False
    if (n_mag == 1 and n_flux == 0) or (n_flux == 1 and n_mag == 0):
        return n_error <= 1
    return False


def _is_role_canonical_header(parts: list[str]) -> bool:
    """True when all words except the last have a scientific role.

    Args:
        parts (list): Header tokens.

    Returns:
        bool: Role-canonical per ``docs/io_contract.md`` §4b.
    """
    if len(parts) < 2:
        return False
    return all(dat_column_role(name) != "other" for name in parts[:-1])


def select_dat_header_names(
    comments: list[str], n_data: int
) -> list[str] | None:
    """Selects the winning ``.dat`` column header from comment lines.

    Prefers the last role-canonical width-matching candidate; otherwise the
    last width-matching candidate. See ``docs/io_contract.md`` §4b.

    Args:
        comments (list): Comment texts without leading ``#``.
        n_data (int): Number of fields per data row.

    Returns:
        list or None: Winning header names, or None if no candidate matches.
    """
    width_matches: list[list[str]] = []
    for comment in comments:
        if is_metadata_assignment_line(comment):
            continue
        parts = comment.split()
        if len(parts) == n_data:
            width_matches.append(parts)
    if not width_matches:
        return None
    role_canonical = [p for p in width_matches if _is_role_canonical_header(p)]
    if role_canonical:
        return list(role_canonical[-1])
    return list(width_matches[-1])


def resolve_dat_column_names(
    header_names: list[str] | None, n_data: int, *, first_data_line_no: int
) -> tuple[list[str], set[str]]:
    """Resolves final column names and which columns are free-text.

    Applies structural validation and 3/4-column failback when needed.

    Args:
        header_names (list, optional): Winning header from comments.
        n_data (int): Data width.
        first_data_line_no (int): Line number for error messages.

    Returns:
        tuple: ``(column_names, free_text_names)``.

    Raises:
        LightcurveIOError: When layout is invalid and width is not 3 or 4.
    """
    if header_names is None:
        if n_data != _LEGACY_DAT_COLUMN_COUNT:
            raise LightcurveIOError(
                f"Invalid .dat file: no column header comment with {n_data} names "
                f"and data rows have {n_data} columns; legacy .dat expects "
                f"{_LEGACY_DAT_COLUMN_COUNT} ({_LEGACY_DAT_COLUMN_HINT}). "
                f"Add a line such as '# jd mag mag_err' or fix the column count."
            )
        col_names = [f"col{i + 1}" for i in range(_LEGACY_DAT_COLUMN_COUNT)]
        return col_names, set()

    roles = [dat_column_role(name) for name in header_names]
    if not _dat_layout_is_valid(roles):
        if n_data == 3:
            col_names = list(_FAILBACK_NAMES_3)
            logger.info(
                "Invalid .dat header roles %s at data width 3; failback to %s "
                "(first data line %s)",
                list(zip(header_names, roles)),
                col_names,
                first_data_line_no,
            )
        elif n_data == 4:
            col_names = list(_FAILBACK_NAMES_4)
            logger.info(
                "Invalid .dat header roles %s at data width 4; failback to %s "
                "(first data line %s)",
                list(zip(header_names, roles)),
                col_names,
                first_data_line_no,
            )
        else:
            raise LightcurveIOError(
                f"Invalid .dat file: header columns {header_names!r} do not form "
                f"exactly one time, one mag or flux, and at most one error column "
                f"(roles={[dat_column_role(n) for n in header_names]}). "
                f"Failback applies only for width 3 or 4; this file has {n_data} "
                f"columns (first data line {first_data_line_no})."
            )
    else:
        col_names = list(header_names)

    free_text = {name for name in col_names if dat_column_role(name) == "other"}
    return col_names, free_text


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


def _parse_dat_cell(
    col_name: str,
    token: str,
    line_no: int,
    stripped: str,
    *,
    free_text_names: set[str],
):
    """Parses one whitespace-separated ``.dat`` field for table construction.

    Args:
        col_name (str): Header column name for this field.
        token (str): Field text from the row.
        line_no (int): Source line number for errors.
        stripped (str): Full stripped row text for errors.
        free_text_names (set): Column names that store arbitrary strings.

    Returns:
        float or str: Cell value for the Astropy column.

    Raises:
        LightcurveIOError: When a numeric column is not numeric.
    """
    if col_name in free_text_names:
        return token
    if token == "" or token.lower() == "nan":
        return float("nan")
    if token[:1] in "<>":
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
    Column header selection, role typing, and failback follow
    ``docs/io_contract.md`` §4b. Promotion to VO UCDs runs in
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

    header_names = select_dat_header_names(comments, n_data)
    if header_names is None:
        non_meta_parts = [
            comment.split()
            for comment in comments
            if not is_metadata_assignment_line(comment) and comment.split()
        ]
        if non_meta_parts and not any(len(p) == n_data for p in non_meta_parts):
            declared = non_meta_parts[0]
            raise LightcurveIOError(
                f"Invalid .dat file: column header declares {len(declared)} columns "
                f"({', '.join(declared)}) but data rows have {n_data} columns "
                f"(first data row at line {first_line_no})."
            )

    col_names, free_text_names = resolve_dat_column_names(
        header_names, n_data, first_data_line_no=first_line_no
    )

    n_expected = len(col_names)
    hint = _dat_column_label_hint(col_names)
    columns: dict[str, list] = {name: [] for name in col_names}

    for line_no, tokens, stripped in data_rows:
        if len(tokens) != n_expected:
            raise LightcurveIOError(
                f"Invalid .dat row at line {line_no}: expected {n_expected} columns "
                f"({hint}), got {len(tokens)}.\n"
                f"Offending line: {stripped}"
            )
        for name, token in zip(col_names, tokens):
            columns[name].append(
                _parse_dat_cell(
                    name,
                    token,
                    line_no,
                    stripped,
                    free_text_names=free_text_names,
                )
            )

    table = Table([columns[name] for name in col_names], names=col_names)
    table.meta["comments"] = comments
    logger.debug(
        "Read .dat table with %s rows and %s columns (free_text=%s)",
        len(table),
        len(col_names),
        sorted(free_text_names),
    )
    return table
