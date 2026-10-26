"""``get_observation_times`` adds the declared time origin; nothing else decides a shift."""

import io

import astropy.units as u
import numpy as np
import pytest
from astropy.table import Column, Table

from volightcurve import read_lightcurve, write_vo_lightcurve


def _lightcurve(origin, scale="TDB", values=(10.0, 11.0)):
    """Writes and reads back a table with a time column and one magnitude column."""
    t = Table()
    t["time"] = Column(np.array(values), unit="d")
    t["mag"] = Column(np.ones(len(values)), unit="mag")
    t["time"].info.meta = {"ucd": "time.epoch"}
    t["mag"].info.meta = {"ucd": "phot.mag"}
    buffer = io.BytesIO()
    write_vo_lightcurve(buffer, t, "x", timeorigin=origin, timescale=scale,
                        refposition="BARYCENTER", binary=False)
    return read_lightcurve(io.BytesIO(buffer.getvalue()), filename="x.vot")


def test_offsets_get_the_declared_origin_whatever_the_source():
    """Origin 2400000.5 (MJD) gives JD offset + 2400000.5; nothing mission specific is added."""
    times = _lightcurve(2400000.5).get_observation_times()
    assert times.scale == "tdb"
    assert times.jd[0] == pytest.approx(2400010.5)
    assert times.mjd[0] == pytest.approx(10.0)


def test_a_full_julian_date_is_not_shifted_again():
    """A value that is already a Julian Date keeps its value."""
    times = _lightcurve(0.0, values=(2460000.5, 2460001.5)).get_observation_times()
    assert times.jd[1] == pytest.approx(2460001.5)


def test_an_undeclared_time_scale_is_refused():
    """The time scale is not assumed."""
    lc = _lightcurve(2400000.5)
    lc.timesys.timescale = None
    with pytest.raises(ValueError, match="time scale"):
        lc.get_observation_times()


def test_an_undeclared_time_origin_is_refused_for_offsets():
    """The origin is not assumed for an offset."""
    lc = _lightcurve(2400000.5)
    lc.timesys.timeorigin = None
    with pytest.raises(ValueError, match="time origin"):
        lc.get_observation_times()
