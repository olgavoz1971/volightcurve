"""Unit tests for the VOTable writing and ingestion round-trip.

This module verifies that lightcurves written by `write_vo_lightcurve` match the required
VOTable specification, include all metadata Groups, Params, and Fields, and can be
successfully ingested back into a `VOLightCurve` instance.
"""

import io
import re
import numpy as np
from fixtures_votable import annotate_vo_roles
import pytest
from astropy.table import Table
from volightcurve import VOLightCurve, write_vo_lightcurve


def test_volightcurve_write():
    """Performs a comprehensive write, metadata validation, and round-trip ingestion test."""
    # 1. Create a dummy lightcurve table
    t = Table()
    t['obs_time'] = np.array([59000.1, 59000.2, 59000.3])
    t['phot'] = np.array([12.5, 12.6, 12.4])
    t['flux_error'] = np.array([0.01, 0.02, 0.015])
    
    # 2. Test writing to an in-memory stream using the standalone function
    buf = io.BytesIO()
    write_vo_lightcurve(
        output_stream_or_path=buf,
        table_data=annotate_vo_roles(t),
        table_name="Test Sim Source",
        filter_identifier="TESS/TESS.Red",
        votable_description="Simulated test lightcurve",
        creator="UPJS skvo_veb pipeline",
        zero_point_flux=2632.0,
        zero_point_ref_mag=20.44,
        effective_wavelength=7.45e-7,
        ra=256.698,
        dec=-54.050,
        period=4.39327,
        epoch=2184.403,
        binary=True
    )
    
    xml_content = buf.getvalue().decode('utf-8')
    
    # Assert obligatory and optional attributes are present in generated XML
    assert "Test Sim Source" in xml_content
    assert "filterIdentifier" in xml_content
    assert "TESS/TESS.Red" in xml_content
    assert "zeroPointFlux" in xml_content
    assert "effectiveWavelength" in xml_content
    assert "ra" in xml_content
    assert "dec" in xml_content
    assert "period" in xml_content
    assert "epoch" in xml_content
    print("✓ Standalone write_vo_lightcurve test passed successfully!")

    # 3. Test round-trip parsing of the written VOTable using VOLightCurve
    buf.seek(0)
    lc = VOLightCurve(buf)
    
    # Assert the ingested VOLightCurve matches the expected structure
    assert len(lc) == 3
    assert 'obs_time' in lc.colnames
    assert 'phot' in lc.colnames
    print("✓ VOLightCurve ingest roundtrip test passed successfully!")


def test_votable_export_keeps_column_names_and_extra_columns():
    """VOTable write does not rename jd/mag or drop a non-science column."""
    from volightcurve import assemble_volightcurve, write_lightcurve
    from volightcurve.time_reference import JD_TO_MJD

    table = Table()
    table["jd"] = np.array([2459000.0, 2459001.0])
    table["mag"] = np.array([12.1, 12.2])
    table["mag_err"] = np.array([0.01, 0.02])
    table["note"] = np.array(["a", "b"])
    volc = assemble_volightcurve(
        table,
        timeorigin=0.0,
        filter_id="TESS/TESS.Red",
        zp_flux=1.0,
        zp_mag=20.0,
    )
    payload = write_lightcurve(volc, "votable")
    xml = payload.decode("utf-8")
    assert 'name="jd"' in xml
    assert 'name="mag"' in xml
    assert 'name="note"' in xml
    assert 'name="obs_time"' not in xml

    restored = VOLightCurve(io.BytesIO(payload))
    assert list(restored.table.colnames) == ["jd", "mag", "mag_err", "note"]
    np.testing.assert_allclose(
        restored.table["jd"],
        np.array([2459000.0, 2459001.0]) - JD_TO_MJD,
    )
    assert restored.timesys.timeorigin == JD_TO_MJD
    ts_id = _timesys_id(xml)
    jd_field = _named_field(xml, "jd")
    assert 'ucd="time.epoch"' in jd_field
    assert f'ref="{ts_id}"' in jd_field


def _timesys_id(xml: str) -> str:
    """Returns ``TIMESYS/@ID`` from serialised VOTable XML."""
    import re

    match = re.search(r"<TIMESYS\b([^>]*)>", xml)
    assert match, "VOTable has no TIMESYS"
    id_match = re.search(r'\bID="([^"]+)"', match.group(1))
    assert id_match, "TIMESYS has no ID"
    return id_match.group(1)


def _named_field(xml: str, name: str) -> str:
    """Returns the FIELD start tag whose ``name`` matches ``name``."""
    import re

    match = re.search(rf'<FIELD\b[^>]*\bname="{re.escape(name)}"[^>]*>', xml)
    assert match, f"missing FIELD name={name}"
    return match.group(0)


def test_votable_write_preserves_existing_time_ucd_through_jd_shift():
    """JD→MJD rewrite must not drop a TIME UCD already on the column."""
    from volightcurve import assemble_volightcurve, write_lightcurve

    table = Table()
    table["jd"] = np.array([2459000.0, 2459001.0])
    table["mag"] = np.array([12.1, 12.2])
    table["jd"].info.meta = {"ucd": "time.epoch;obs.exposure"}
    volc = assemble_volightcurve(table, timeorigin=0.0, filter_id="TESS/TESS.Red")
    xml = write_lightcurve(volc, "votable").decode("utf-8")
    jd_field = _named_field(xml, "jd")
    assert 'ucd="time.epoch;obs.exposure"' in jd_field
    assert f'ref="{_timesys_id(xml)}"' in jd_field


def test_write_vo_lightcurve_rejects_a_table_without_role_ucds():
    """The writer does not guess roles from names: a bare table is rejected."""
    t = Table()
    t["obs_time"] = np.array([59000.1, 59000.2])
    t["phot"] = np.array([12.5, 12.6])
    with pytest.raises(ValueError, match="does not guess roles"):
        write_vo_lightcurve(output_stream_or_path=io.BytesIO(), table_data=t, table_name="t")


def test_write_vo_lightcurve_links_time_epoch_to_the_written_timesys():
    """A TIME-role column carries ``time.epoch`` and ``ref`` is the written TIMESYS ID."""
    t = Table()
    t["obs_time"] = np.array([59000.1, 59000.2])
    t["phot"] = np.array([12.5, 12.6])
    t = annotate_vo_roles(t)
    buf = io.BytesIO()
    write_vo_lightcurve(
        output_stream_or_path=buf,
        table_data=t,
        table_name="t",
        filter_identifier="TESS/TESS.Red",
        refposition="BARYCENTER",
        timescale="TDB",
        timeorigin=2400000.5,
        epoch=59000.0,
        binary=False,
    )
    xml = buf.getvalue().decode("utf-8")
    ts_id = _timesys_id(xml)
    time_field = _named_field(xml, "obs_time")
    assert 'ucd="time.epoch"' in time_field
    assert f'ref="{ts_id}"' in time_field
    assert f'name="epoch"' in xml
    assert f'ref="{ts_id}"' in xml


if __name__ == "__main__":
    test_volightcurve_write()


def test_write_votable_method_writes_the_calibration_of_the_instance():
    """``VOLightCurve.write_votable`` passes the instance ``photdms`` to the writer."""
    import re

    from fixtures_votable import annotate_vo_roles
    from volightcurve import PhotCal, PhotDM, read_lightcurve

    table = Table()
    table["obs_time"] = np.array([59000.1, 59000.2])
    table["phot"] = np.array([12.5, 12.6])
    table["phot"].info.meta = {"ucd": "phot.mag;em.opt"}
    source = io.BytesIO()
    write_vo_lightcurve(
        source,
        annotate_vo_roles(table),
        table_name="t",
        photdms={"phot": PhotDM(photcal=PhotCal(zp_mag=0.0, zp_flux=3631.0, zp_flux_unit="Jy"))},
        refposition="BARYCENTER",
        timescale="TDB",
        timeorigin=2400000.5,
    )
    source.seek(0)
    volc = read_lightcurve(source, filename="t.vot", strict=True)
    again = io.BytesIO()
    volc.write_votable(again, table_name="t")
    text = again.getvalue().decode()
    assert re.search(r'name="zeroPointFlux"[^>]*value="3631', text)
    assert 'name="zeroPointReferenceMagnitude"' in text


def test_table_params_are_written_as_given_and_duplicates_are_refused():
    """Generic TABLE PARAMs keep name, datatype, unit, UCD and value; a repeated name fails."""
    from fixtures_votable import annotate_vo_roles

    table = Table()
    table["obs_time"] = np.array([59000.1, 59000.2])
    table["phot"] = np.array([12.5, 12.6])
    table["phot"].info.meta = {"ucd": "phot.mag;em.opt"}
    entries = [
        {"name": "source_id", "value": "10655814178816", "datatype": "long", "ucd": "meta.id"},
        {"name": "author", "value": "SPOC"},
        {"name": "exptime", "value": "120.0", "datatype": "double", "unit": "s"},
    ]
    out = io.BytesIO()
    write_vo_lightcurve(
        out,
        annotate_vo_roles(table),
        table_name="t",
        timeorigin=2400000.5,
        refposition="BARYCENTER",
        timescale="TDB",
        table_params=entries,
    )
    text = out.getvalue().decode()
    assert re.search(r'name="source_id"[^>]*value="10655814178816"', text) or re.search(
        r'value="10655814178816"[^>]*name="source_id"', text
    )
    assert 'name="author"' in text and 'value="SPOC"' in text
    assert re.search(r'name="exptime"[^>]*unit="s"', text) or re.search(r'unit="s"[^>]*name="exptime"', text)
    with pytest.raises(ValueError, match="already has a PARAM"):
        write_vo_lightcurve(
            io.BytesIO(),
            annotate_vo_roles(table),
            table_name="t",
            ra=1.0,
            table_params=[{"name": "ra", "value": "2.0", "datatype": "double"}],
        )
    with pytest.raises(ValueError, match="needs a 'name' and a 'value'"):
        write_vo_lightcurve(
            io.BytesIO(),
            annotate_vo_roles(table),
            table_name="t",
            table_params=[{"name": "x"}],
        )


def test_writer_gives_no_photcal_group_to_a_non_measurement_column():
    """A signal-to-noise column keeps its UCD but is not linked to any photcal group."""
    from fixtures_votable import annotate_vo_roles
    from volightcurve import PhotCal, PhotDM

    table = Table()
    table["obs_time"] = np.array([59000.1, 59000.2])
    table["mag"] = np.array([12.5, 12.6])
    table["mag"].info.meta = {"ucd": "phot.mag;em.opt"}
    table["snr"] = np.array([50.0, 51.0])
    table["snr"].info.meta = {"ucd": "stat.snr;phot.flux"}
    out = io.BytesIO()
    write_vo_lightcurve(
        out,
        annotate_vo_roles(table),
        table_name="t",
        timeorigin=2400000.5,
        photdms={"mag": PhotDM(photcal=PhotCal(zp_mag=0.0, zp_flux=3631.0, zp_flux_unit="Jy"))},
    )
    text = out.getvalue().decode()
    assert text.count("<GROUP ") == 1
    assert re.findall(r'<FIELDref ref="([^"]+)"', text) == ["mag"]
    assert 'ucd="stat.snr;phot.flux"' in text
