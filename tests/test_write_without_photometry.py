"""A table with a time column and no photometry column can be written."""

import io

import numpy as np
import pytest
from astropy.table import Column, Table

from volightcurve.lightcurve import write_vo_lightcurve


def _table() -> Table:
    """Builds a table with a time column and one plain column."""
    t = Table()
    t["time"] = Column(np.arange(3.0), unit="d")
    t["quality"] = Column(np.arange(3, dtype=np.int64))
    t["time"].info.meta = {"ucd": "time.epoch"}
    return t


def test_table_without_photometry_is_written_without_photcal_group():
    """No phot.mag or phot.flux column gives no photcal GROUP and no error."""
    buffer = io.BytesIO()
    write_vo_lightcurve(buffer, _table(), "x", timeorigin=0.0, timescale="TDB",
                        refposition="BARYCENTER", binary=False)
    xml = buffer.getvalue().decode("utf-8")
    assert 'name="quality"' in xml
    assert 'name="photcal"' not in xml


def test_table_without_time_column_is_refused():
    """A table with no time UCD is still refused, naming the cause."""
    t = _table()
    t["time"].info.meta = {}
    with pytest.raises(ValueError, match="time UCD"):
        write_vo_lightcurve(io.BytesIO(), t, "x")
