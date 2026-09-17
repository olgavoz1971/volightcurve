"""VOTable photcal GROUP filter identifier + name round-trip."""

from __future__ import annotations

import io

import numpy as np
from astropy.table import Table

from volightcurve import VOLightCurve, write_vo_lightcurve


def test_votable_photcal_group_writes_filter_name_utype():
    """``filterName`` PARAM with PhotDM utype is emitted inside photcal GROUP."""
    table = Table(
        {
            "obs_time": np.array([59000.1, 59000.2]),
            "phot": np.array([12.5, 12.6]),
            "flux_error": np.array([0.01, 0.02]),
        }
    )
    buf = io.BytesIO()
    write_vo_lightcurve(
        output_stream_or_path=buf,
        table_data=table,
        table_name="Filter name test",
        filter_identifier="Generic/Bessell.I",
        filter_name="I",
        zero_point_flux=2415.65,
        zero_point_flux_unit="Jy",
        zero_point_ref_mag=0.0,
        binary=False,
    )
    xml = buf.getvalue().decode("utf-8")
    assert 'name="filterIdentifier"' in xml
    assert 'utype="photDM:PhotometryFilter.identifier"' in xml
    assert "Generic/Bessell.I" in xml
    assert 'name="filterName"' in xml
    assert 'utype="photDM:PhotometryFilter.name"' in xml
    # Still inside photcal GROUP context (before FIELDref / after magnitudeSystem)
    assert xml.index('name="filterName"') < xml.index('ref="phot"')


def test_votable_photcal_filter_name_roundtrip():
    """Ingest restores PhotometryFilter.identifier and ``name`` from the GROUP."""
    table = Table(
        {
            "obs_time": np.array([59000.1]),
            "phot": np.array([12.5]),
            "flux_error": np.array([0.01]),
        }
    )
    buf = io.BytesIO()
    write_vo_lightcurve(
        output_stream_or_path=buf,
        table_data=table,
        table_name="Roundtrip",
        filter_identifier="Generic/Bessell.I",
        filter_name="I",
        zero_point_flux=1.0,
        zero_point_ref_mag=0.0,
        binary=False,
    )
    buf.seek(0)
    volc = VOLightCurve(buf)
    photdm = next(iter(volc.photdms.values()))
    assert photdm.filter_id == "Generic/Bessell.I"
    assert photdm.filter_name == "I"
    assert photdm.filter.name == "I"
    assert photdm.photcal.photometry_filter is photdm.filter
