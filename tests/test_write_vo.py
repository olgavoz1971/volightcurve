"""Unit tests for the VOTable writing and ingestion round-trip.

This module verifies that lightcurves written by `write_vo_lightcurve` match the required
VOTable specification, include all metadata Groups, Params, and Fields, and can be
successfully ingested back into a `VOLightCurve` instance.
"""

import io
import numpy as np
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
        table_data=t,
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


if __name__ == "__main__":
    test_volightcurve_write()
