"""Readiness of a photometric calibration for magnitude↔flux conversion.

Inspect reports problems and does not insert default zero points. Conversion
methods raise :class:`PhotCalError` with the same sentences.
"""

from __future__ import annotations

import astropy.units as u


class PhotCalError(ValueError):
    """Calibration cannot support a magnitude↔flux conversion.

    Args:
        problems (list[str]): Specific reasons. Joined into the exception text.
    """

    def __init__(self, problems: list[str]):
        self.problems = [str(item) for item in problems if str(item).strip()]
        if not self.problems:
            self.problems = ["Photometric calibration cannot convert magnitude and flux."]
        super().__init__("\n".join(self.problems))


def _as_unit(value) -> u.Unit | None:
    """Parses a unit text or Quantity unit.

    Args:
        value: Unit string, ``astropy.units.Unit``, or ``None``.

    Returns:
        astropy.units.Unit or None: ``None`` when the text is not a unit.
    """
    if value is None:
        return u.dimensionless_unscaled
    if isinstance(value, u.UnitBase):
        return value
    text = str(value).strip()
    if text in ("", "---"):
        return u.dimensionless_unscaled
    try:
        return u.Unit(text)
    except (ValueError, TypeError, u.UnitsError, u.UnitTypeError):
        return None


def inspect_conversion_photcal(
    *,
    zp_flux,
    zp_mag,
    zp_flux_unit=None,
    zp_mag_unit=None,
    flux_unit=None,
    mag_unit=None,
    zp_flux_unit_unparsed: bool = False,
    zp_mag_unit_unparsed: bool = False,
) -> list[str]:
    """Lists why this calibration cannot convert magnitude and flux.

    A missing flux-column unit is not treated as dimensionless. Magnitude-only
    curves often have no flux unit yet, and that must not be reported as a
    mismatch against a real zero-point unit such as Jy.

    Args:
        zp_flux: Zero-point flux number, or ``None`` when absent.
        zp_mag: Zero-point magnitude number, or ``None`` when absent.
        zp_flux_unit: Zero-point flux unit text. Empty means dimensionless.
        zp_mag_unit: Zero-point magnitude unit text. Empty means ``mag``.
        flux_unit: Flux column unit when the column exists. ``None`` skips
            the column comparison.
        mag_unit: Magnitude column unit when a magnitude Quantity is being
            converted. ``None`` skips that comparison.
        zp_flux_unit_unparsed (bool): The flux unit text was kept but is not
            a unit.
        zp_mag_unit_unparsed (bool): The magnitude unit text was kept but is
            not a unit.

    Returns:
        list[str]: Empty when conversion may proceed.
    """
    problems: list[str] = []
    if zp_flux is None:
        problems.append(
            "Zero-point flux (zp_flux) is missing, so magnitude and flux "
            "cannot be converted."
        )
    if zp_mag is None:
        problems.append(
            "Zero-point magnitude (zp_mag) is missing, so magnitude and flux "
            "cannot be converted."
        )

    flux_zp_unit = None
    if zp_flux_unit_unparsed or (
        zp_flux_unit not in (None, "") and _as_unit(zp_flux_unit) is None
    ):
        problems.append(
            f"Zero-point flux unit ({zp_flux_unit}) is not a valid unit."
        )
    else:
        flux_zp_unit = _as_unit(zp_flux_unit)

    mag_zp_unit = None
    if zp_mag_unit_unparsed or (
        zp_mag_unit not in (None, "") and _as_unit(zp_mag_unit) is None
    ):
        problems.append(
            f"Zero-point magnitude unit ({zp_mag_unit}) is not a valid unit."
        )
    else:
        mag_zp_unit = u.mag if zp_mag_unit in (None, "") else _as_unit(zp_mag_unit)

    if flux_unit is not None and flux_zp_unit is not None:
        column = _as_unit(flux_unit)
        if column is None:
            problems.append(
                f"Flux column unit ({flux_unit}) is not a valid unit, so it "
                "cannot be compared with the zero-point flux unit."
            )
        elif not column.is_equivalent(flux_zp_unit):
            problems.append(
                f"Flux column unit ({flux_unit}) is not equivalent to the "
                f"zero-point flux unit ({zp_flux_unit or 'dimensionless'})."
            )

    if mag_unit is not None and mag_zp_unit is not None:
        column = _as_unit(mag_unit)
        if column is None:
            problems.append(
                f"Magnitude unit ({mag_unit}) is not a valid unit, so it "
                "cannot be compared with the zero-point magnitude unit."
            )
        elif not column.is_equivalent(mag_zp_unit):
            problems.append(
                f"Magnitude unit ({mag_unit}) is not equivalent to the "
                f"zero-point magnitude unit ({zp_mag_unit or 'mag'})."
            )
    return problems
