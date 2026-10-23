"""Single codec for VO / Astropy unit strings ↔ internal unitless ``None``.

**Internal model** (``PhotCal``, photcal dicts, table metadata):
``None`` means dimensionless / unitless.

**VOTable wire** (IVOA VOUnits 1.1, REC 2023-12-15): ``1`` denotes dimensionless. The
empty string is NOT a valid unit string in 1.1, so it is never written. An absent or
unknown unit is written as no ``unit`` attribute at all. On read, ``1``, the empty string
(files of earlier versions), the Astropy ``---`` sentinel and a dimensionless ``Unit`` all
map to ``None``.

**Astropy quirk:** a dimensionless ``Param(unit="1")`` is serialised as ``unit="---"`` and
re-read as ``UnrecognizedUnit(---)``. Ingest maps that sentinel back to ``None``; export
post-processes ``unit="---"`` to ``VO_DIMENSIONLESS_WIRE``.

``inspect_unit_string`` reports what a unit string breaks in VOUnits 1.1 (informative, it
never raises); the codec itself does not reject unit strings.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Any

import astropy.units as u

# VOUnits 1.1: "1" = dimensionless. Change only here for file export/import.
VO_DIMENSIONLESS_WIRE = "1"

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
    if text in ("", VO_DIMENSIONLESS_WIRE) or text in _ASTROPY_EMPTY_UNIT_SENTINELS:
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
        internal (str, optional): Internal unit (``None`` = dimensionless). A caller that
            has an absent or unknown unit writes no ``unit`` attribute and does not call
            this function.

    Returns:
        str: Wire token; ``VO_DIMENSIONLESS_WIRE`` (``1``) when unitless.
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
    """Rewrites Astropy ``unit="---"`` attributes to ``VO_DIMENSIONLESS_WIRE`` (``1``).

    Call after ``VOTableFile.to_xml`` so on-disk VOTables carry the VOUnits 1.1
    token for dimensionless quantities.

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


_RESERVED_UNIT_WORDS = frozenset({"unknown"})


@dataclass
class UnitReport:
    """Findings about one unit string against VOUnits 1.1.

    Attributes:
        unit (str or None): The text that was inspected.
        errors (list[str]): Breaches of the VOUnits 1.1 syntax or reserved words.
        notes (list[str]): Informative remarks (for example a unit outside the VOUnits
            table, which the standard allows).
    """

    unit: str | None
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Returns whether the unit string has no error (notes do not count).

        Returns:
            bool: ``True`` when ``errors`` is empty.
        """
        return not self.errors


def inspect_unit_string(value: str | None) -> UnitReport:
    """Reports what a unit string breaks in VOUnits 1.1. Never raises.

    ``None`` (no unit declared) is fine. The checks: empty string or the Astropy ``---``
    sentinel, the reserved word ``unknown``, whitespace, ``*`` used as a product sign, a
    string Astropy cannot parse, and (as a note, not an error) a unit that is not in the
    VOUnits table, such as ``electron``.

    Args:
        value (str, optional): Unit text as found on the wire or in metadata.

    Returns:
        UnitReport: Errors and notes; both empty for a clean unit.
    """
    report = UnitReport(unit=value)
    if value is None:
        return report
    text = str(value)
    if text == "" or text.strip() in _ASTROPY_EMPTY_UNIT_SENTINELS:
        report.errors.append(
            "an empty unit string is not valid in VOUnits 1.1: write '1' for "
            "dimensionless and no unit attribute when the unit is unknown"
        )
        return report
    if text.strip().lower() in _RESERVED_UNIT_WORDS:
        report.errors.append(
            f"{text!r} is a reserved word in VOUnits 1.1: write no unit attribute "
            "when the unit is unknown"
        )
        return report
    if re.search(r"\s", text):
        report.errors.append(
            "whitespace is not allowed in a VOUnits 1.1 unit string; the product sign is '.'"
        )
    # "**" is the VOUnits 1.1 power operator (s**-1); only a lone "*" is a product sign.
    if "*" in text.replace("**", ""):
        report.errors.append("'*' is not a product sign in VOUnits 1.1; use '.'")
    if report.errors:
        return report
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            u.Unit(text, format="vounit")
        except (ValueError, TypeError) as exc:
            report.errors.append(f"cannot be parsed as a VOUnits 1.1 unit: {exc}")
    for item in caught:
        if issubclass(item.category, u.UnitsWarning):
            report.notes.append(str(item.message).strip())
    return report
