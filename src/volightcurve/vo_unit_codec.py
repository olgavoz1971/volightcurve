"""Single codec for VO / Astropy unit strings ↔ internal unitless ``None``.

**Internal model** (``PhotCal``, photcal dicts, table metadata):
``None`` means dimensionless / unitless.

**VOTable wire** (IVOA VOUnits): the empty string denotes dimensionless, so the
attribute must be present as ``unit=""``. Change ``VO_DIMENSIONLESS_WIRE`` here
only if that agreement changes.

**Astropy quirk:** ``Param(unit="")`` is serialised as ``unit="---"`` and re-read
as ``UnrecognizedUnit(---)``. Ingest maps that sentinel back to ``None``; export
post-processes ``unit="---"`` back to ``VO_DIMENSIONLESS_WIRE``.
"""

from __future__ import annotations

import re
from typing import Any

import astropy.units as u

# VOUnits empty string = dimensionless. Change only here for file export/import.
VO_DIMENSIONLESS_WIRE = ""

# Astropy VOTable stand-in for an empty unit attribute (not a physical unit).
_ASTROPY_EMPTY_UNIT_SENTINELS = frozenset({"---"})

_UNIT_ATTR_EMPTY_SENTINEL_RE = re.compile(
    rb'\bunit\s*=\s*([\'"])---\1',
    flags=re.IGNORECASE,
)


def is_dimensionless_wire(value: Any) -> bool:
    """Returns whether a wire/Astropy unit value means dimensionless.

    Args:
        value: ``None``, string, or Astropy unit instance from a PARAM/FIELD.

    Returns:
        bool: ``True`` for unitless / empty / Astropy ``---`` sentinel.
    """
    if value is None:
        return True
    if isinstance(value, u.UnitBase):
        if isinstance(value, u.UnrecognizedUnit):
            return str(value).strip() in _ASTROPY_EMPTY_UNIT_SENTINELS
        try:
            return value.is_equivalent(u.dimensionless_unscaled) and (
                value == u.dimensionless_unscaled
                or str(value).strip() in ("", "1")
            )
        except (ValueError, u.UnitsError, u.UnitTypeError):
            return False
    text = str(value).strip()
    if text == "" or text in _ASTROPY_EMPTY_UNIT_SENTINELS:
        return True
    return False


def to_internal(value: Any) -> str | None:
    """Maps a file/Astropy/UI unit value to internal storage.

    Args:
        value: Raw unit from VOTable, metadata, or UI.

    Returns:
        str or None: Non-empty unit label, or ``None`` when dimensionless.
    """
    if is_dimensionless_wire(value):
        return None
    if isinstance(value, u.UnitBase):
        try:
            if value.is_equivalent(u.dimensionless_unscaled) and (
                value == u.dimensionless_unscaled
            ):
                return None
            return value.to_string("vounit")
        except (ValueError, u.UnitsError, u.UnitTypeError):
            text = str(value).strip()
            return text or None
    text = str(value).strip()
    return text or None


def to_wire(internal: str | None) -> str:
    """Maps internal storage to the VOTable ``unit`` attribute string.

    Args:
        internal (str, optional): Internal unit (``None`` = dimensionless).

    Returns:
        str: Wire token; ``VO_DIMENSIONLESS_WIRE`` when unitless.
    """
    if internal is None or str(internal).strip() == "":
        return VO_DIMENSIONLESS_WIRE
    return str(internal).strip()


def to_display(internal: str | None) -> str:
    """Maps internal storage to a UI label (never ``None`` or ``---``).

    Args:
        internal (str, optional): Internal unit.

    Returns:
        str: British English display label.
    """
    if internal is None or str(internal).strip() == "":
        return "dimensionless"
    text = str(internal).strip()
    if text in _ASTROPY_EMPTY_UNIT_SENTINELS:
        return "dimensionless"
    return text


def rewrite_astropy_empty_unit_attributes(xml_payload: bytes) -> bytes:
    """Rewrites Astropy ``unit="---"`` attributes to ``VO_DIMENSIONLESS_WIRE``.

    Call after ``VOTableFile.to_xml`` so on-disk VOTables keep a present
    ``unit`` attribute with the agreed empty-string encoding.

    Args:
        xml_payload (bytes): Serialised VOTable XML.

    Returns:
        bytes: XML with empty-unit sentinels normalised.
    """
    if not xml_payload:
        return xml_payload
    wire = VO_DIMENSIONLESS_WIRE.encode("ascii")
    replacement = b'unit="' + wire + b'"'

    def _replace(match: re.Match[bytes]) -> bytes:
        return replacement

    return _UNIT_ATTR_EMPTY_SENTINEL_RE.sub(_replace, xml_payload)
