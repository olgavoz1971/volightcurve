"""``get_observation_times`` adds the declared time origin; nothing else decides a shift."""

import io

import astropy.units as u
import numpy as np
import pytest
from astropy.table import Column, Table

from volightcurve import read_lightcurve, write_vo_lightcurve


def _lightcurve(origin, scale="TDB", values=(10.0, 11.0), epoch=None):
    """Writes and reads back a table with a time column and one magnitude column."""
    t = Table()
    t["time"] = Column(np.array(values), unit="d")
    t["mag"] = Column(np.ones(len(values)), unit="mag")
    t["time"].info.meta = {"ucd": "time.epoch"}
    t["mag"].info.meta = {"ucd": "phot.mag"}
    buffer = io.BytesIO()
    write_vo_lightcurve(buffer, t, "x", timeorigin=origin, timescale=scale,
                        refposition="BARYCENTER", binary=False, epoch=epoch)
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


def test_jd_and_mjd_arrays_follow_the_declared_origin_for_any_source():
    """The float methods give origin + value; a TESS-like origin needs no BTJD shift."""
    lc = _lightcurve(2400000.5, values=(58000.25, 58001.25))
    assert lc.get_observation_mjd() == pytest.approx([58000.25, 58001.25])
    assert lc.get_observation_jd() == pytest.approx([2458000.75, 2458001.75])
    assert lc.get_observation_jd().dtype == float


def test_jd_method_refuses_an_undeclared_origin():
    """The float methods raise like get_observation_times."""
    lc = _lightcurve(2400000.5)
    lc.timesys.timeorigin = None
    with pytest.raises(ValueError, match="time origin"):
        lc.get_observation_mjd()


def test_epoch_follows_the_declared_origin_like_a_time_column():
    """An epoch offset gets the declared origin; the JD and MJD floats agree with the Time."""
    lc = _lightcurve(2400000.5, epoch=58000.25)
    assert lc.get_epoch_mjd() == pytest.approx(58000.25)
    assert lc.get_epoch_jd() == pytest.approx(2458000.75)
    assert lc.get_epoch_time().scale == "tdb"


def test_a_full_julian_date_epoch_is_not_shifted_again():
    """An epoch that is already a Julian Date keeps its value."""
    lc = _lightcurve(2400000.5, epoch=2458000.75)
    assert lc.get_epoch_jd() == pytest.approx(2458000.75)


def test_no_epoch_gives_none_and_an_undeclared_origin_is_refused():
    """No declared epoch is None, not an invented value; an offset needs its origin."""
    assert _lightcurve(2400000.5).get_epoch_jd() is None
    lc = _lightcurve(2400000.5, epoch=58000.25)
    lc.timesys.timeorigin = None
    lc.timesys_by_id = {key: value for key, value in lc.timesys_by_id.items()}
    for ts in lc.timesys_by_id.values():
        ts.timeorigin = None
    with pytest.raises(ValueError, match="time origin"):
        lc.get_epoch_mjd()


def test_float_methods_do_not_need_a_time_scale_but_time_methods_do():
    """JD and MJD floats add the declared origin without a scale; Time needs the scale."""
    lc = _lightcurve(2400000.5, epoch=58000.25, values=(58000.25, 58001.25))
    lc.timesys.timescale = None
    for ts in lc.timesys_by_id.values():
        ts.timescale = None
    assert lc.get_observation_mjd() == pytest.approx([58000.25, 58001.25])
    assert lc.get_epoch_jd() == pytest.approx(2458000.75)
    with pytest.raises(ValueError, match="time scale"):
        lc.get_observation_times()
    with pytest.raises(ValueError, match="time scale"):
        lc.get_epoch_time()
    assert any("time scale" in problem for problem in lc.time_problems())
