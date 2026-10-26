"""Units are written as VOUnits text, not as Astropy's CDS spelling."""

import io
import re

import astropy.units as u
import numpy as np
from astropy.table import Column, Table

from volightcurve.lightcurve import PhotCal, PhotDM, PhotometryFilter, write_vo_lightcurve


def _units(payload: bytes) -> dict:
    """Maps FIELD and PARAM names to their written unit text."""
    found = re.findall(rb'<(?:FIELD|PARAM) [^>]*', payload)
    out = {}
    for tag in found:
        name = re.search(rb'name="([^"]*)"', tag).group(1).decode()
        unit = re.search(rb'unit="([^"]*)"', tag)
        out[name] = unit.group(1).decode() if unit else None
    return out


def test_units_are_written_as_vounits_and_verbatim_text_is_kept(caplog):
    """A recognised unit uses VOUnits; the zero point and an unrecognised unit keep their text."""
    t = Table()
    t["time"] = Column(np.arange(3.0), unit="d")
    t["f"] = Column(np.ones(3), unit=u.UnrecognizedUnit("count/s"))
    t["g"] = Column(np.ones(3), unit=u.Unit("1/s"))
    t["time"].info.meta = {"ucd": "time.epoch"}
    t["f"].info.meta = {"ucd": "phot.flux;em.opt"}
    photdm = PhotDM(
        photcal=PhotCal(zp_flux=1.0, zp_flux_unit="count/s", zp_mag=0.0, zp_mag_unit="mag", mag_sys="Vega"),
        photometry_filter=PhotometryFilter(filter_id="X", spectral_location=500e-9 * u.m),
    )
    buffer = io.BytesIO()
    write_vo_lightcurve(buffer, t, "x", timeorigin=0.0, timescale="TDB",
                        refposition="BARYCENTER", photdms={"f": photdm})
    units = _units(buffer.getvalue())
    assert units["f"] == "count/s"
    assert units["zeroPointFlux"] == "count/s"
    assert units["g"] == "s**-1"
    assert units["time"] == "d"
    assert "W50" not in caplog.text
