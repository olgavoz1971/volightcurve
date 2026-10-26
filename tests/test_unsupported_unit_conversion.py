"""Units that VOUnits does not support must not break the flux and magnitude conversion."""

import io

import astropy.units as u
import numpy as np
import pytest
from astropy.table import Column, Table

from volightcurve import PhotCal, PhotDM, PhotometryFilter, read_lightcurve, write_vo_lightcurve


def _tess_like_payload(zero_point_unit: str = "count/s") -> bytes:
    """Writes a table whose flux and zero point carry the same unsupported unit text."""
    t = Table()
    t["jd"] = Column(np.array([1.0, 2.0, 3.0]), unit="d")
    t["flux"] = Column(np.array([100.0, 200.0, 400.0]), unit=u.UnrecognizedUnit("electron.s-1"))
    t["flux_err"] = Column(np.array([1.0, 2.0, 4.0]), unit=u.UnrecognizedUnit("electron.s-1"))
    t["jd"].info.meta = {"ucd": "time.epoch"}
    t["flux"].info.meta = {"ucd": "phot.flux;em.opt"}
    t["flux_err"].info.meta = {"ucd": "stat.error;phot.flux;em.opt"}
    photdm = PhotDM(
        photcal=PhotCal(
            zp_flux=1.0, zp_flux_unit="electron.s-1", zp_mag=20.0, zp_mag_unit="mag", mag_sys="Vega"
        ),
        photometry_filter=PhotometryFilter(filter_id="X", spectral_location=7e-7 * u.m),
    )
    buffer = io.BytesIO()
    write_vo_lightcurve(
        buffer, t, "x", timeorigin=0.0, timescale="TDB", refposition="BARYCENTER",
        binary=False, photdms={"flux": photdm, "flux_err": photdm},
    )
    return buffer.getvalue()


def test_flux_in_an_unsupported_unit_converts_when_the_zero_point_declares_the_same_text():
    """electron.s-1 on the FIELD and on the zero point: the conversion succeeds."""
    lc = read_lightcurve(io.BytesIO(_tess_like_payload()), filename="x.vot")
    photcal = lc.photdms["flux"].photcal
    flux = lc.table["flux"].quantity
    error = lc.table["flux_err"].quantity
    mag = photcal.flux_to_mag(flux)
    assert mag.value == pytest.approx(20.0 - 2.5 * np.log10([100.0, 200.0, 400.0]))
    mag_err = photcal.flux_err_to_mag_err(flux, error)
    assert mag_err.value == pytest.approx(2.5 / np.log(10.0) * 0.01)


def test_different_unsupported_units_are_refused_with_both_texts():
    """electron/s against electron/min is not converted and the error names both units."""
    photcal = PhotCal(zp_flux=1.0, zp_flux_unit="electron.s-1", zp_mag=20.0, zp_mag_unit="mag")
    other = u.Unit("electron.min-1", format="vounit", parse_strict="silent")
    with pytest.raises(u.UnitsError, match="electron"):
        photcal.flux_to_mag(np.array([1.0, 2.0]) * other)
