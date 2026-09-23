"""Shared lightcurve file-metadata keyword vocabulary (Ticket 8 Phase 0).

Non-VO formats (``.dat``, CSV with ``#`` preamble, ECSV flat ``meta``) use the
same plain ``KEY = value`` names. VOTable keeps IVOA PhotDM / TIMESYS and is
mapped to these concepts only at the bridge.

See ``docs/volightcurve_io_contract.md``. Writers emit a key only when the
value is present; they must not invent calibration or time origins.
"""

from __future__ import annotations

import re

# --- Canonical keyword names (write form) ---------------------------------

KEY_JD0 = "JD0"
KEY_EPOCH = "EPOCH"
KEY_PERIOD = "PERIOD"
KEY_MAG0 = "MAG0"
KEY_ZP_MAG = "ZP_MAG"
KEY_ZP_MAG_UNIT = "ZP_MAG_UNIT"
KEY_ZP_FLUX = "ZP_FLUX"
KEY_ZP_FLUX_UNIT = "ZP_FLUX_UNIT"
KEY_MAG_SYS = "MAG_SYS"
KEY_FILTER = "FILTER"
KEY_BAND = "BAND"
KEY_FILTER_NAME = "FILTER_NAME"
KEY_EFFECTIVE_WAVELENGTH = "EFFECTIVE_WAVELENGTH"
KEY_EFFECTIVE_WAVELENGTH_UNIT = "EFFECTIVE_WAVELENGTH_UNIT"
KEY_SENTINEL = "SENTINEL"

# Keywords that participate in photometric or time calibration round-trip.
CALIBRATION_KEYWORDS: frozenset[str] = frozenset(
    {
        KEY_JD0,
        KEY_EPOCH,
        KEY_PERIOD,
        KEY_MAG0,
        KEY_ZP_MAG,
        KEY_ZP_MAG_UNIT,
        KEY_ZP_FLUX,
        KEY_ZP_FLUX_UNIT,
        KEY_MAG_SYS,
        KEY_FILTER,
        KEY_BAND,
        KEY_FILTER_NAME,
        KEY_EFFECTIVE_WAVELENGTH,
        KEY_EFFECTIVE_WAVELENGTH_UNIT,
        KEY_SENTINEL,
    }
)

# Read aliases: secondary name -> preferred canonical name.
KEYWORD_READ_ALIASES: dict[str, str] = {
    KEY_MAG0: KEY_ZP_MAG,
    KEY_BAND: KEY_FILTER,
}

# Ordered keys for deterministic non-VO write headers (canonical only).
WRITE_KEYWORD_ORDER: tuple[str, ...] = (
    KEY_JD0,
    KEY_PERIOD,
    KEY_EPOCH,
    KEY_ZP_FLUX,
    KEY_ZP_FLUX_UNIT,
    KEY_ZP_MAG,
    KEY_ZP_MAG_UNIT,
    KEY_MAG_SYS,
    KEY_FILTER,
    KEY_FILTER_NAME,
    KEY_EFFECTIVE_WAVELENGTH,
    KEY_EFFECTIVE_WAVELENGTH_UNIT,
    KEY_SENTINEL,
)

_METADATA_ASSIGNMENT = re.compile(
    r"^(?P<key>[A-Za-z][A-Za-z0-9_]*)\s*=\s*(?P<value>.*)$",
    re.UNICODE,
)


def normalise_keyword(name: str) -> str:
    """Returns the uppercase canonical form of a metadata keyword name.

    Args:
        name (str): Raw keyword token from a comment line or ECSV meta key.

    Returns:
        str: Uppercase keyword without surrounding whitespace.
    """
    return str(name).strip().upper()


def canonical_keyword(name: str) -> str:
    """Maps a keyword (including read aliases) to its preferred canonical name.

    Args:
        name (str): Raw or uppercase keyword.

    Returns:
        str: Preferred keyword for storage (e.g. ``MAG0`` → ``ZP_MAG``).
    """
    key = normalise_keyword(name)
    return KEYWORD_READ_ALIASES.get(key, key)


def is_calibration_keyword(name: str) -> bool:
    """Returns whether ``name`` is a first-class calibration / timing keyword.

    Args:
        name (str): Raw or uppercase keyword.

    Returns:
        bool: True if the keyword is in ``CALIBRATION_KEYWORDS``.
    """
    return normalise_keyword(name) in CALIBRATION_KEYWORDS


def parse_keyword_assignment(line: str) -> tuple[str, str] | None:
    """Parses a ``KEY = value`` assignment from a comment or meta line.

    Args:
        line (str): Text without a leading ``#`` (caller strips comment markers).

    Returns:
        tuple[str, str] | None: ``(canonical_key, raw_value)`` when the line is
        an assignment to a known calibration keyword; otherwise ``None``.
        Unknown ``KEY = value`` lines are not treated as calibration (callers
        may keep them as free-text description).
    """
    stripped = line.strip()
    if not stripped:
        return None
    match = _METADATA_ASSIGNMENT.match(stripped)
    if match is None:
        return None
    raw_key = normalise_keyword(match.group("key"))
    if raw_key not in CALIBRATION_KEYWORDS:
        return None
    return canonical_keyword(raw_key), match.group("value").strip()


def is_metadata_assignment_line(line: str) -> bool:
    """Returns True when ``line`` is a recognised calibration ``KEY = value`` line.

    Args:
        line (str): Comment text without a leading ``#``.

    Returns:
        bool: True for first-class calibration assignments.
    """
    return parse_keyword_assignment(line) is not None
