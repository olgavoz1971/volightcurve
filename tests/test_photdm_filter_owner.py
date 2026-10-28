"""PhotDM 1.1: the filter belongs to the calibration; the hub only forwards to it."""

import io

import pytest
from astropy.table import Table

from volightcurve import PhotCal, PhotDM, PhotometryFilter, VOLightCurve, write_vo_lightcurve


def test_hub_filter_is_the_calibration_filter():
    """A filter given to the hub is stored on the calibration."""
    flt = PhotometryFilter(filter_id="Palomar/ZTF.g")
    cal = PhotCal(zp_flux=1.0, zp_flux_unit="Jy")
    hub = PhotDM(photcal=cal, photometry_filter=flt)
    assert cal.photometry_filter is flt
    assert hub.filter is flt
    hub.filter_id = "Palomar/ZTF.r"
    assert cal.photometry_filter.filter_id == "Palomar/ZTF.r"


def test_filter_without_calibration_is_handed_over_later():
    """A filter set before any calibration moves to the calibration assigned later."""
    flt = PhotometryFilter(filter_id="GAIA/GAIA2.G")
    hub = PhotDM(photometry_filter=flt)
    assert hub.photcal is None and hub.filter is flt
    cal = PhotCal(zp_flux=1.0, zp_flux_unit="Jy")
    hub.photcal = cal
    assert cal.photometry_filter is flt


def test_two_different_filters_are_refused():
    """The hub never keeps a second filter next to the calibration's own."""
    cal = PhotCal(photometry_filter=PhotometryFilter(filter_id="A"))
    with pytest.raises(ValueError, match="different filters"):
        PhotDM(photcal=cal, photometry_filter=PhotometryFilter(filter_id="B"))


def test_reference_magnitude_is_written_with_the_photdm_11_name_and_read_back():
    """The writer uses ``referenceMagnitudeValue``; the reader also takes the old spelling."""
    table = Table({"time": [1.0, 2.0], "mag": [10.0, 11.0]})
    table["time"].info.meta = {"ucd": "time.epoch"}
    table["mag"].info.meta = {"ucd": "phot.mag"}
    photdm = PhotDM(photcal=PhotCal(zp_flux=3631.0, zp_flux_unit="Jy", zp_mag=0.0, mag_sys="AB"))
    buffer = io.BytesIO()
    write_vo_lightcurve(buffer, table, "t", timescale="TDB", timeorigin=0.0, photdms={"mag": photdm})
    text = buffer.getvalue().decode()
    assert "photDM:PhotCal.zeroPoint.referenceMagnitudeValue" in text
    assert "referenceMagnitude.value" not in text
    back = VOLightCurve(io.BytesIO(buffer.getvalue()))
    assert back.photdms["mag"].photcal.zp_mag.value == 0.0


def _table_with_filter_comment(text: str) -> Table:
    """Builds a two-column table with a ``FILTER`` keyword in its comments.

    Args:
        text (str): Value of the ``FILTER`` assignment.

    Returns:
        astropy.table.Table: Table with ``jd`` and ``mag`` columns.
    """
    import numpy as np

    table = Table()
    table["jd"] = np.array([100.0, 101.0])
    table["mag"] = np.array([12.0, 12.1])
    table.meta["comments"] = ["JD0 = 2450000", f"FILTER = {text}"]
    return table


def test_interpretation_takes_a_service_code_as_identifier():
    """A value with a slash (Facility/Subcategory.Band) is read as the filter identifier."""
    volc = VOLightCurve.from_table(_table_with_filter_comment("TESS/TESS.Red"))
    flt = volc.photdms["mag"].filter
    assert flt.filter_id == "TESS/TESS.Red"
    assert flt.name is None


def test_interpretation_takes_a_bare_band_label_as_name_only():
    """A bare label such as V is a name; the identifier stays empty."""
    volc = VOLightCurve.from_table(_table_with_filter_comment("V"))
    flt = volc.photdms["mag"].filter
    assert flt.name == "V"
    assert not flt.filter_id
    assert "filter name V" in volc.table.meta["description"]
