"""Pure tests for PhotCal façade and nested PogsonZeroPoint conversion."""

from __future__ import annotations

import astropy.units as u
import numpy as np
import pytest

from volightcurve import PhotCal, PogsonZeroPoint
from volightcurve.photdm import AsinhZeroPoint


def test_photcal_default_uses_pogson_zero_point():
    """Flat PhotCal constructor nests a PogsonZeroPoint."""
    pc = PhotCal(zp_flux=3631.0, zp_flux_unit="Jy", zp_mag=0.0, mag_sys="Vega")
    assert isinstance(pc.zero_point, PogsonZeroPoint)
    assert pc.mag_sys == "Vega"
    assert pc.zp_flux.unit.is_equivalent(u.Jy)


def test_mag_flux_roundtrip_pogson():
    """mag→flux→mag round-trips for a known Pogson PhotCal."""
    pc = PhotCal(zp_flux=3631.0, zp_flux_unit="Jy", zp_mag=0.0)
    mag = np.array([0.0, 2.5, 5.0]) * u.mag
    flux = pc.mag_to_flux(mag)
    back = pc.flux_to_mag(flux)
    np.testing.assert_allclose(back.to_value(u.mag), mag.to_value(u.mag), rtol=1e-12)


def test_flux_err_to_mag_err_matches_snr_form():
    """σ_m = (2.5/ln10)·(σ_F/F) via PhotCal, not a local constant."""
    pc = PhotCal(zp_flux=1.0, zp_flux_unit=None, zp_mag=0.0)
    flux = np.array([10.0, 100.0]) * u.dimensionless_unscaled
    flux_err = np.array([1.0, 2.0]) * u.dimensionless_unscaled
    mag_err = pc.flux_err_to_mag_err(flux, flux_err)
    expected = (2.5 / np.log(10.0)) * np.abs(flux_err / flux)
    np.testing.assert_allclose(mag_err.to_value(u.mag), expected.to_value(u.dimensionless_unscaled), rtol=1e-12)


def test_unit_mismatch_raises():
    """Incompatible magnitude units raise UnitsError."""
    pc = PhotCal(zp_flux=1.0, zp_mag=0.0)
    with pytest.raises(u.UnitsError):
        pc.mag_to_flux(1.0 * u.Jy)


def test_asinh_stub_raises():
    """AsinhZeroPoint conversion is reserved and fails visibly."""
    zp = AsinhZeroPoint(softening_parameter=1.0, zp_flux=1.0, zp_mag=0.0)
    pc = PhotCal(zero_point=zp)
    with pytest.raises(NotImplementedError, match="Asinh"):
        pc.mag_to_flux(12.0 * u.mag)
