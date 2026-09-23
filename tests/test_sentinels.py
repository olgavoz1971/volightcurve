"""Declared SENTINEL values and censored tokens (I/O contract §9)."""

import io

import numpy as np
import pytest

from volightcurve import read_lightcurve, write_lightcurve


def test_sentinel_values_become_nan_and_round_trip():
    """Repeated SENTINEL lines replace close floats and are written back."""
    payload = b"""# JD0 = 0
# SENTINEL = 99.99
# SENTINEL = 99.999
# jd mag mag_err
2459000.0 12.1 0.01
2459001.0 99.99 0.02
2459002.0 99.999 0.02
"""
    volc = read_lightcurve(io.BytesIO(payload), filename="lc.dat")
    assert np.isnan(volc.table["mag"][1])
    assert np.isnan(volc.table["mag"][2])
    assert volc.table["mag"][0] == pytest.approx(12.1)
    text = write_lightcurve(volc, "ascii.commented_header").decode("utf-8")
    assert "SENTINEL = 99.99" in text
    assert "SENTINEL = 99.999" in text


def test_censored_magnitude_becomes_nan():
    """A leading ``>`` in a numeric column is missing data, not a file error."""
    payload = b"""jd,mag,mag_err,flux,flux_err
2459000.0,16.453,0.175,1.007,0.160
2459001.0,>16.766,0.160,0.617,0.151
"""
    volc = read_lightcurve(io.BytesIO(payload), filename="lc.csv")
    assert volc.table["mag"][0] == pytest.approx(16.453)
    assert np.isnan(volc.table["mag"][1])
    assert volc.table["flux"][1] == pytest.approx(0.617)
