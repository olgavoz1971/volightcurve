"""Tests for the shared VO unit codec (internal None ↔ wire empty string)."""

from __future__ import annotations

import io

import numpy as np
from astropy.table import Table
from astropy import units as u

from volightcurve import write_vo_lightcurve
from volightcurve.vo_unit_codec import (
    VO_DIMENSIONLESS_WIRE,
    rewrite_astropy_empty_unit_attributes,
    to_display,
    to_internal,
    to_wire,
)


def test_codec_maps_empty_and_astropy_sentinel_to_none():
    """Empty wire tokens and Astropy ``---`` become internal ``None``."""
    assert to_internal(None) is None
    assert to_internal("") is None
    assert to_internal("---") is None
    assert to_internal(u.dimensionless_unscaled) is None
    assert to_internal("Jy") == "Jy"


def test_codec_wire_and_display_never_expose_none():
    """Wire and UI helpers never return ``None`` for unitless values."""
    assert to_wire(None) == VO_DIMENSIONLESS_WIRE
    assert to_wire("") == VO_DIMENSIONLESS_WIRE
    assert to_display(None) == "dimensionless"
    assert to_display("---") == "dimensionless"
    assert to_display("Jy") == "Jy"


def test_rewrite_astropy_empty_unit_attributes():
    """Post-processing replaces Astropy ``unit=\"---\"`` with the wire token."""
    raw = b'<PARAM name="zeroPointFlux" unit="---" value="1"/>'
    fixed = rewrite_astropy_empty_unit_attributes(raw)
    expected = f'unit="{VO_DIMENSIONLESS_WIRE}"'.encode("ascii")
    assert expected in fixed
    assert b'unit="---"' not in fixed


def test_write_vo_dimensionless_zp_uses_empty_unit_attribute():
    """Exported VOTables keep ``unit=""`` for dimensionless zero-point flux."""
    t = Table()
    t["obs_time"] = np.array([59000.1, 59000.2])
    t["phot"] = np.array([1.0, 1.1]) * u.dimensionless_unscaled
    t["flux_error"] = np.array([0.01, 0.02]) * u.dimensionless_unscaled

    buf = io.BytesIO()
    write_vo_lightcurve(
        output_stream_or_path=buf,
        table_data=t,
        table_name="Unitless ZP test",
        filter_identifier="TESS/TESS.Red",
        zero_point_flux=1.0,
        zero_point_flux_unit=None,
        zero_point_ref_mag=20.0,
        binary=False,
    )
    xml = buf.getvalue().decode("utf-8")
    assert 'name="zeroPointFlux"' in xml
    assert 'unit="---"' not in xml
    assert f'unit="{VO_DIMENSIONLESS_WIRE}"' in xml

    # Re-ingest: PhotCal must store internal None, not UnrecognizedUnit(---).
    from volightcurve import VOLightCurve

    buf.seek(0)
    lc = VOLightCurve(buf)
    photdm = next(iter(lc.photdms.values()), None)
    assert photdm is not None and photdm.photcal is not None
    assert photdm.photcal._zp_flux_unit_text is None
    assert not photdm.photcal._zp_flux_unit_unparsed
