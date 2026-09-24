"""Conversion refuses an empty PhotCal and reports specific reasons."""

import astropy.units as u
import pytest

from volightcurve.lightcurve import PhotCal
from volightcurve.photcal_check import PhotCalError, inspect_conversion_photcal


def test_empty_photcal_is_not_a_usable_zero_point():
    """Omitted constructor arguments stay missing."""
    pc = PhotCal()
    problems = pc.conversion_problems()
    assert any("zp_flux" in item for item in problems)
    assert any("zp_mag" in item for item in problems)
    with pytest.raises(PhotCalError, match="zp_flux"):
        pc.mag_to_flux(16.0 * u.mag)


def test_explicit_instrumental_pair_converts():
    """A caller who passes 1 and 0 still has a Pogson calibration."""
    pc = PhotCal(zp_flux=1.0, zp_mag=0.0)
    flux = pc.mag_to_flux(0.0 * u.mag)
    assert flux.unit.is_equivalent(u.dimensionless_unscaled)


def test_inspect_reports_unit_mismatch_without_rewriting():
    """Inspect names both units and does not invent a zero point."""
    problems = inspect_conversion_photcal(
        zp_flux=3631.0,
        zp_mag=0.0,
        zp_flux_unit="Jy",
        flux_unit="m/s",
    )
    assert problems
    assert "Jy" in problems[0]
    assert "m/s" in problems[0]
