"""Dash-free round-trip tests for ``read_lightcurve`` / ``write_lightcurve``."""

from __future__ import annotations

import io

import numpy as np
import pytest
from astropy.table import Table

from volightcurve.io import (
    assemble_volightcurve,
    read_lightcurve,
    write_lightcurve,
)
from volightcurve.io_errors import LightcurveIOError
from volightcurve.io_keywords import KEY_PERIOD


def _sample_volc():
    """Builds a small mag-domain VOLightCurve with full calibration."""
    table = Table(
        {
            "jd": np.array([2459000.0, 2459000.5, 2459001.0]),
            "mag": np.array([12.1, 12.2, 12.0]),
            "mag_err": np.array([0.01, 0.01, 0.02]),
        }
    )
    return assemble_volightcurve(
        table,
        timeorigin=0.0,
        period=1.23,
        epoch=2459000.25,
        filter_id="TESS/TESS.Red",
        zp_flux=1.0,
        zp_flux_unit=None,
        zp_mag=20.0,
        zp_mag_unit="mag",
        mag_sys="Vega",
    )


@pytest.mark.parametrize(
    "fmt,filename",
    [
        ("ascii.commented_header", "lc.dat"),
        ("csv", "lc.csv"),
        ("ascii.ecsv", "lc.ecsv"),
    ],
)
def test_tabular_round_trip_preserves_calibration(fmt, filename):
    """Export then re-ingest keeps JD0, period, epoch, filter, and ZP_MAG."""
    volc = _sample_volc()
    payload = write_lightcurve(volc, fmt)
    text = payload.decode("utf-8")
    assert "JD0" in text or "jd0" in text.lower()
    assert "ZP_MAG" in text or "zp_mag" in text.lower()
    assert "PERIOD" in text or "period" in text.lower()

    restored = read_lightcurve(io.BytesIO(payload), filename=filename)
    assert restored.timesys.timeorigin == pytest.approx(0.0)
    assert restored.table.meta.get("period") == pytest.approx(1.23)
    assert restored.table.meta.get("epoch") == pytest.approx(2459000.25)
    assert restored.table.meta.get("filter") == "TESS/TESS.Red"
    photdm = next(iter(restored.photdms.values()))
    assert float(photdm.photcal.zp_mag.value) == pytest.approx(20.0)
    np.testing.assert_allclose(restored.table["mag"], [12.1, 12.2, 12.0])


def test_read_dat_rejects_ragged_rows():
    """Strict ``.dat`` reader raises ``LightcurveIOError`` with a line number."""
    dat = b"""# jd mag mag_err
52500.1 12.3 0.01
52501.2 12.4
"""
    with pytest.raises(LightcurveIOError, match="line 3"):
        read_lightcurve(io.BytesIO(dat), filename="bad.dat")


def test_write_dat_emits_extended_photcal_keywords():
    """``.dat`` write includes ZP_* lines from assembled photcal."""
    volc = _sample_volc()
    text = write_lightcurve(volc, "ascii.commented_header").decode("utf-8")
    assert "# JD0 =" in text
    assert "ZP_FLUX" in text
    assert "ZP_MAG" in text
    assert "FILTER" in text
    assert KEY_PERIOD in text or "PERIOD" in text


def test_narrative_comment_survives_dat_round_trip():
    """Free-text ``#`` lines (not KEY=value) are re-emitted on ``.dat`` write."""
    dat = b"""# JD0 = 0
# Observer notes: night was clear
# ZP_MAG = 20.0
# ZP_FLUX = 1.0
# FILTER = V
# jd mag mag_err
2459000.0 12.1 0.01
2459001.0 12.2 0.02
"""
    volc = read_lightcurve(io.BytesIO(dat), filename="notes.dat")
    out = write_lightcurve(volc, "ascii.commented_header").decode("utf-8")
    assert "Observer notes: night was clear" in out
    assert "ZP_MAG" in out
    restored = read_lightcurve(io.BytesIO(out.encode("utf-8")), filename="notes.dat")
    comments = restored.table.meta.get("comments") or []
    assert any("Observer notes" in c for c in comments)
