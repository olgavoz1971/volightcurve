"""Characterisation tests: pin what the package does TODAY (baseline ``v0.1.0``).

Purpose (see ``docs/TODO.md`` of the host repository, Ticket 32, addenda A and B):
the cleaning of ``volightcurve`` separates the strict core from interpretation and
removes metadata that the code invents. Before anything changes, these tests record the
present behaviour, including the behaviour that is going to be removed.

Two groups:

* ``KEEP``: behaviour that agrees with the agreed principles. It must stay true.
* ``INVENTED``: behaviour that conflicts with "absent stays absent". The test passes today
  and is expected to FAIL when Ticket 32 phase 2 removes the invention. At that point the
  test is rewritten to assert the new behaviour (``None`` / no element written), with the
  reason noted. A failing ``INVENTED`` test is therefore a signal, not a regression.

The numbers in the test names refer to addendum A, section 2 of Ticket 32.
"""

from __future__ import annotations

import io
import logging
import re
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table

from volightcurve import (
    PhotCal,
    PhotCalError,
    PhotDM,
    VOLightCurve,
    read_lightcurve,
    write_vo_lightcurve,
)
from volightcurve.lightcurve import PhotometryFilter, TimeSys, _pickup_jd0_from_table
from volightcurve.time_reference import (
    extract_timesys_metadata_from_gavo,
    time_offset_to_absolute_jd,
)
from volightcurve.lightcurve import _gavo_votable_tree_from_source
from fixtures_votable import annotate_vo_roles

logger = logging.getLogger(__name__)


def _votable(
    *,
    timesys: str = "",
    groups: str = "",
    fields: str,
    rows: str,
) -> bytes:
    """Builds a small VOTable document.

    Args:
        timesys (str): ``TIMESYS`` element(s) placed in the RESOURCE.
        groups (str): ``GROUP`` element(s) placed in the RESOURCE.
        fields (str): ``FIELD`` elements of the TABLE.
        rows (str): ``TR`` rows of the TABLEDATA.

    Returns:
        bytes: UTF-8 VOTable document.
    """
    return f"""<?xml version="1.0" encoding="utf-8"?>
<VOTABLE xmlns="http://www.ivoa.net/xml/VOTable/v1.3" version="1.4">
  <RESOURCE>
    {timesys}
    {groups}
    <TABLE name="characterisation">
      {fields}
      <DATA><TABLEDATA>{rows}</TABLEDATA></DATA>
    </TABLE>
  </RESOURCE>
</VOTABLE>
""".encode()


_PHOT_GROUP_TEMPLATE = """<GROUP ID="{gid}" name="photcal">
      <PARAM name="zeroPointFlux" datatype="double" utype="photDM:PhotCal.zeroPoint.flux.value" value="{zp}"/>
      <PARAM name="zeroPointReferenceMagnitude" datatype="double" utype="photDM:PhotCal.zeroPoint.referenceMagnitude.value" value="0.0" unit="mag"/>
      <PARAM name="magnitudeSystem" datatype="char" arraysize="*" utype="photDM:PhotCal.magnitudeSystem.type" value="Vega"/>
      <FIELDref ref="{col}"/>
    </GROUP>"""


def _mag_table() -> Table:
    """Returns a two-row table with a time column and a magnitude column.

    Returns:
        astropy.table.Table: Table with ``obs_time`` and ``phot``.
    """
    table = Table()
    table["obs_time"] = np.array([59000.1, 59000.2])
    table["phot"] = np.array([12.5, 12.6])
    table["flux_error"] = np.array([0.01, 0.02])
    return table


def _write_bytes(table: Table, **kwargs) -> bytes:
    """Writes a table with ``write_vo_lightcurve`` and returns the bytes.

    Args:
        table (astropy.table.Table): Table to write.
        **kwargs: Extra keyword arguments for the writer.

    Returns:
        bytes: VOTable document.
    """
    buffer = io.BytesIO()
    write_vo_lightcurve(buffer, annotate_vo_roles(table), table_name="characterisation", **kwargs)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# KEEP: behaviour that agrees with the agreed principles
# ---------------------------------------------------------------------------


def test_keep_conversion_without_zero_points_raises_and_names_what_is_missing():
    """KEEP: magnitude to flux without a calibration raises ``PhotCalError``."""
    photcal = PhotCal()
    with pytest.raises(PhotCalError) as caught:
        photcal.mag_to_flux(np.array([12.5]) * u.mag)
    message = str(caught.value)
    assert "zp_flux" in message
    assert "zp_mag" in message


def test_keep_conversion_problems_lists_missing_zero_points_without_raising():
    """KEEP: ``conversion_problems`` is a check method that does not raise."""
    problems = PhotCal().conversion_problems()
    assert problems
    assert any("zp_flux" in text for text in problems)


def test_keep_votable_without_coosys_gives_none():
    """KEEP: an undeclared coordinate system stays ``None``."""
    payload = _votable(
        fields='<FIELD name="obs_time" datatype="double" ucd="time.epoch" unit="d"/>',
        rows="<TR><TD>1.0</TD></TR>",
    )
    assert VOLightCurve(io.BytesIO(payload)).coosys is None


def test_keep_writer_without_coosys_argument_writes_no_coosys_element():
    """KEEP: the writer does not invent a ``COOSYS`` element."""
    assert b"<COOSYS" not in _write_bytes(_mag_table())


def test_keep_zero_points_are_not_written_when_absent():
    """KEEP: absent zero points are not written (no ``zeroPoint`` PARAM)."""
    text = _write_bytes(_mag_table()).decode()
    assert "zeroPointFlux" not in text
    assert "zeroPointReferenceMagnitude" not in text


def test_keep_one_calibration_group_per_declared_column_is_read():
    """KEEP: several photcal groups are read as one calibration per column."""
    groups = _PHOT_GROUP_TEMPLATE.format(gid="photcal_sap", zp="1.0", col="sap_flux") + (
        _PHOT_GROUP_TEMPLATE.format(gid="photcal_pdcsap", zp="2.0", col="pdcsap_flux")
    )
    fields = (
        '<FIELD name="obs_time" datatype="double" ucd="time.epoch" unit="d"/>'
        '<FIELD name="sap_flux" datatype="double" ucd="phot.flux;em.opt" unit="electron.s-1" ref="photcal_sap"/>'
        '<FIELD name="pdcsap_flux" datatype="double" ucd="phot.flux;em.opt" unit="electron.s-1" ref="photcal_pdcsap"/>'
    )
    payload = _votable(
        groups=groups,
        fields=fields,
        rows="<TR><TD>1.0</TD><TD>10.0</TD><TD>11.0</TD></TR>",
    )
    volc = VOLightCurve(io.BytesIO(payload))
    assert volc.photdms["sap_flux"].photcal.zp_flux.value == pytest.approx(1.0)
    assert volc.photdms["pdcsap_flux"].photcal.zp_flux.value == pytest.approx(2.0)


def test_keep_declared_timesys_attributes_are_read_as_declared():
    """KEEP: a fully declared ``TIMESYS`` is read exactly."""
    payload = _votable(
        timesys='<TIMESYS ID="ts" refposition="BARYCENTER" timeorigin="2455197.5" timescale="TCB"/>',
        fields='<FIELD name="obs_time" datatype="double" ucd="time.epoch" unit="d" ref="ts"/>',
        rows="<TR><TD>1.0</TD></TR>",
    )
    volc = VOLightCurve(io.BytesIO(payload))
    assert volc.timesys.refposition == "BARYCENTER"
    assert volc.timesys.timescale == "TCB"
    assert volc.timesys.timeorigin == pytest.approx(2455197.5)


def test_keep_comment_keywords_are_read_as_evidence():
    """KEEP: ``JD0=`` in a comment is read into the time origin (non-VO input)."""
    table = Table()
    table["jd"] = np.array([2459000.0])
    table["mag"] = np.array([12.0])
    table.meta["comments"] = ["JD0=2400000.5"]
    assert VOLightCurve.from_table(table).timesys.timeorigin == pytest.approx(2400000.5)


def test_keep_zero_point_comment_builds_calibration_with_declared_values():
    """KEEP: ``ZP_FLUX=`` in a comment becomes the flux zero point."""
    table = Table()
    table["jd"] = np.array([2459000.0])
    table["mag"] = np.array([12.0])
    table.meta["comments"] = ["ZP_FLUX=2.5"]
    photcal = VOLightCurve.from_table(table).photdms["mag"].photcal
    assert photcal.zp_flux.value == pytest.approx(2.5)
    assert photcal.zp_mag is None


# ---------------------------------------------------------------------------
# INVENTED: behaviour expected to change in Ticket 32 phase 2
# ---------------------------------------------------------------------------


def test_absent_1_timesys_constructor_declares_nothing():
    """REWRITTEN (2.1, step 1): ``TimeSys()`` holds ``None`` for every attribute."""
    timesys = TimeSys()
    assert timesys.refposition is None
    assert timesys.timeorigin is None
    assert timesys.timescale is None
    assert timesys.jd0 is None


def test_absent_1_curve_without_a_declaration_reports_no_time_system():
    """REWRITTEN (2.1, step 1): a VOTable with no ``TIMESYS`` reports ``None`` values."""
    payload = _votable(
        fields='<FIELD name="obs_time" datatype="double" ucd="time.epoch" unit="d"/>',
        rows="<TR><TD>1.0</TD></TR>",
    )
    volc = VOLightCurve(io.BytesIO(payload))
    assert volc.timesys_by_id == {}
    assert (volc.timesys.refposition, volc.timesys.timescale, volc.timesys.timeorigin) == (
        None,
        None,
        None,
    )
    tree = _gavo_votable_tree_from_source(io.BytesIO(payload))
    assert extract_timesys_metadata_from_gavo(tree).registry == {}


def test_absent_2_declared_timesys_with_missing_attributes_keeps_origin_absent():
    """REWRITTEN (2.2, step 1): a ``TIMESYS`` with only an ID has no time origin.

    The strings ``UNKNOWN`` come from the GAVO parser (the VOTable attribute defaults) and
    ``UNKNOWN`` is a term of the IVOA vocabularies, so they are kept as they come. The time
    origin, which has no parser default, is ``None``.
    """
    payload = _votable(
        timesys='<TIMESYS ID="ts"/>',
        fields='<FIELD name="obs_time" datatype="double" ucd="time.epoch" unit="d" ref="ts"/>',
        rows="<TR><TD>1.0</TD></TR>",
    )
    timesys = VOLightCurve(io.BytesIO(payload)).timesys_by_id["ts"]
    assert (timesys.refposition, timesys.timescale, timesys.timeorigin) == (
        "UNKNOWN",
        "UNKNOWN",
        None,
    )


def test_absent_3_declared_zero_origin_and_absent_origin_are_distinguished():
    """REWRITTEN (2.3, step 1): ``timeorigin="0"`` reads as 0.0, no ``timeorigin`` as ``None``."""
    template = (
        '<TIMESYS ID="ts" refposition="BARYCENTER" timescale="TCB"{extra}/>'
    )
    fields = '<FIELD name="obs_time" datatype="double" ucd="time.epoch" unit="d" ref="ts"/>'
    declared = VOLightCurve(
        io.BytesIO(
            _votable(
                timesys=template.format(extra=' timeorigin="0"'),
                fields=fields,
                rows="<TR><TD>1.0</TD></TR>",
            )
        )
    )
    absent = VOLightCurve(
        io.BytesIO(
            _votable(
                timesys=template.format(extra=""),
                fields=fields,
                rows="<TR><TD>1.0</TD></TR>",
            )
        )
    )
    assert declared.timesys.timeorigin == 0.0
    assert absent.timesys.timeorigin is None


def test_absent_3_comment_guess_returns_none_when_nothing_is_found():
    """REWRITTEN (2.3, step 1): ``_pickup_jd0_from_table`` returns ``None`` for 'not found'."""
    table = Table()
    table.meta["comments"] = ["nothing useful here"]
    assert _pickup_jd0_from_table(table) is None


def test_absent_4_photdm_without_filter_or_calibration_returns_none():
    """REWRITTEN (2.5, step 1): ``PhotDM().filter_id`` and ``mag0`` are ``None``."""
    photdm = PhotDM()
    assert photdm.filter_id is None
    assert photdm.mag0 is None


def test_absent_5_zero_point_comment_without_system_has_no_system():
    """REWRITTEN (2.6, step 1): a ``ZP_FLUX`` comment alone leaves the magnitude system ``None``."""
    table = Table()
    table["jd"] = np.array([2459000.0])
    table["mag"] = np.array([12.0])
    table.meta["comments"] = ["ZP_FLUX=2.5"]
    photcal = VOLightCurve.from_table(table).photdms["mag"].photcal
    assert photcal.mag_sys is None


def test_written_6_no_time_origin_means_no_timesys():
    """REWRITTEN (2.7, step 2): without a time origin no ``TIMESYS`` is written."""
    text = _write_bytes(_mag_table()).decode()
    assert "<TIMESYS" not in text
    assert 'ref="ts"' not in text


def test_written_6_declared_time_system_is_written_exactly():
    """Step 2: a declared origin, position and scale are written as given."""
    text = _write_bytes(
        _mag_table(), refposition="BARYCENTER", timescale="TDB", timeorigin=2400000.5
    ).decode()
    element = re.findall(r"<TIMESYS[^>]*>", text)
    assert len(element) == 1
    assert 'refposition="BARYCENTER"' in element[0]
    assert 'timescale="TDB"' in element[0]
    assert 'timeorigin="2400000.5"' in element[0]


def test_written_6_zero_origin_is_a_declared_value():
    """Step 2: ``timeorigin=0`` is written, unlike an absent origin."""
    text = _write_bytes(_mag_table(), timeorigin=0.0).decode()
    element = re.findall(r"<TIMESYS[^>]*>", text)
    assert len(element) == 1
    assert 'timeorigin="0.0"' in element[0]
    assert 'refposition="UNKNOWN"' in element[0]
    assert 'timescale="UNKNOWN"' in element[0]


def test_written_6_declared_scale_without_origin_is_dropped_with_a_warning(caplog):
    """Step 2: a scale without an origin cannot be written; the loss is logged."""
    with caplog.at_level(logging.WARNING):
        text = _write_bytes(_mag_table(), timescale="TDB").decode()
    assert "<TIMESYS" not in text
    assert "TIMESYS is not written" in caplog.text


def test_written_6_method_uses_the_time_system_of_the_instance():
    """Step 2: ``write_votable`` and the function agree; the instance values are written."""
    volc = VOLightCurve.from_table(_mag_table())
    volc.timesys = TimeSys(refposition="BARYCENTER", timeorigin=2400000.5, timescale="TDB")
    buffer = io.BytesIO()
    volc.write_votable(buffer, table_name="characterisation")
    element = re.findall(r"<TIMESYS[^>]*>", buffer.getvalue().decode())
    assert 'refposition="BARYCENTER"' in element[0]
    assert 'timescale="TDB"' in element[0]
    assert 'timeorigin="2400000.5"' in element[0]


def test_written_6_no_magnitude_system_is_written_without_a_declaration():
    """Step 2: ``magnitudeSystem`` is not written when it is not declared."""
    text = _write_bytes(_mag_table()).decode()
    assert "magnitudeSystem" not in text


def test_written_6_absent_calibration_still_gives_an_empty_photcal_group():
    """Step 2: an empty calibration is a group with the column link and no value PARAMs."""
    text = _write_bytes(_mag_table()).decode()
    assert len(re.findall(r"<GROUP ", text)) == 1
    group = re.search(r"<GROUP .*?</GROUP>", text, re.S).group(0)
    assert "<PARAM" not in group
    assert '<FIELDref ref="phot"' in group


def test_written_6_absent_calibration_round_trip_stays_absent():
    """Step 2: nothing declared on write is ``None`` on read."""
    volc = VOLightCurve(io.BytesIO(_write_bytes(_mag_table())))
    photdm = volc.photdms["phot"]
    assert photdm.photcal.mag_sys is None
    assert photdm.photcal.zp_flux is None
    assert photdm.photcal.zp_mag is None
    assert photdm.filter_id is None
    assert volc.timesys.timeorigin is None
    assert volc.timesys.refposition is None


def test_written_6_empty_observatory_params_are_not_written():
    """Step 2: ``facility_name`` and ``instrument_name`` are written only with a value."""
    text = _write_bytes(_mag_table()).decode()
    assert "facility_name" not in text
    assert "instrument_name" not in text
    text = _write_bytes(_mag_table(), facility_name="Brno", instrument_name="  ").decode()
    assert re.search(r'name="facility_name"[^>]*value="Brno"', text)
    assert "instrument_name" not in text


# ---------------------------------------------------------------------------
# Limits relevant to Ticket 33 (writer) and interpretation on compliant input
# ---------------------------------------------------------------------------


def test_limit_writer_emits_one_photcal_group_for_all_photometry_columns():
    """LIMIT (4): the writer cannot write one calibration per column."""
    table = Table()
    table["obs_time"] = np.array([59000.1, 59000.2])
    table["sap_flux"] = np.array([1.0, 1.1])
    table["sap_flux"].info.meta = {"ucd": "phot.flux;em.opt"}
    table["pdcsap_flux"] = np.array([2.0, 2.1])
    table["pdcsap_flux"].info.meta = {"ucd": "phot.flux;em.opt"}
    text = _write_bytes(table, zero_point_flux=1.0, zero_point_ref_mag=0.0).decode()
    assert len(re.findall(r"<GROUP ", text)) == 1
    assert text.count("<FIELDref") == 2


def test_units_dimensionless_is_written_as_one():
    """REWRITTEN (units, step 3): a dimensionless zero point flux is written as ``unit="1"``."""
    from volightcurve.vo_unit_codec import VO_DIMENSIONLESS_WIRE

    photdm = PhotDM(photcal=PhotCal(zp_flux=1.0, zp_mag=0.0))
    text = _write_bytes(_mag_table(), photdms={"phot": photdm}).decode()
    assert VO_DIMENSIONLESS_WIRE == "1"
    assert re.search(r'name="zeroPointFlux"[^>]*unit="1"', text) or re.search(
        r'unit="1"[^>]*name="zeroPointFlux"', text
    )
    assert 'unit=""' not in text
    assert 'unit="---"' not in text


def test_units_absent_zero_point_has_no_unit_attribute_and_no_param():
    """Step 3: nothing declared writes neither the PARAM nor a unit attribute."""
    text = _write_bytes(_mag_table()).decode()
    assert "zeroPointFlux" not in text
    assert 'unit=""' not in text


def test_units_codec_reads_every_dimensionless_form_as_none():
    """REWRITTEN (units, step 3): ``1``, ``""``, ``---`` and a dimensionless Unit are ``None``."""
    from volightcurve.vo_unit_codec import to_internal, to_wire

    assert to_internal(u.dimensionless_unscaled) is None
    for form in ("", "1", " 1 ", "---", None):
        assert to_internal(form) is None
    assert to_internal("electron.s-1") == "electron.s-1"
    assert to_wire(None) == "1"
    assert to_wire("electron.s-1") == "electron.s-1"


def test_units_old_empty_unit_attribute_is_read_as_dimensionless():
    """Step 3: files of earlier versions with ``unit=""`` still read as dimensionless."""
    groups = """
    <GROUP ID="photcal_x" name="photcal">
      <PARAM name="zeroPointFlux" datatype="double" unit="" utype="photdm:PhotCal.zeroPoint.flux.value" value="1.0"/>
      <FIELDref ref="mag" utype="adhoc:location"/>
    </GROUP>
    """
    payload = _votable(
        groups=groups,
        fields='<FIELD ID="mag" name="mag" datatype="double" ucd="phot.mag" unit="mag"/>',
        rows="<TR><TD>12.0</TD></TR>",
    )
    photcal = VOLightCurve(io.BytesIO(payload)).photdms["mag"].photcal
    assert photcal.zp_flux.value == pytest.approx(1.0)
    assert photcal.zero_point._zp_flux_unit_text is None


def test_units_round_trip_of_dimensionless_and_electron_per_second():
    """Step 3: ``1`` and ``electron.s-1`` survive a write and read."""
    for unit, expected in ((None, None), ("electron.s-1", "electron.s-1")):
        photdm = PhotDM(photcal=PhotCal(zp_flux=1.0, zp_flux_unit=unit))
        payload = _write_bytes(_mag_table(), photdms={"phot": photdm})
        back = VOLightCurve(io.BytesIO(payload)).photdms["phot"].photcal
        assert back.zero_point._zp_flux_unit_text == expected


def test_units_inspect_unit_string_reports_vounit_problems():
    """Step 3: the check names whitespace, ``*``, empty, ``unknown``; notes outside units."""
    from volightcurve import inspect_unit_string

    assert inspect_unit_string(None).ok
    assert inspect_unit_string("1").ok
    assert inspect_unit_string("mag").ok and not inspect_unit_string("mag").notes
    assert not inspect_unit_string("").ok
    assert not inspect_unit_string("---").ok
    assert not inspect_unit_string("unknown").ok
    assert not inspect_unit_string("UNKNOWN").ok
    assert any("whitespace" in e for e in inspect_unit_string("electron s-1").errors)
    assert any("'*'" in e for e in inspect_unit_string("electron*s-1").errors)
    electron = inspect_unit_string("electron.s-1")
    assert electron.ok
    assert any("electron" in note for note in electron.notes)


def test_units_check_units_of_a_curve_lists_only_findings():
    """Step 3: ``check_units`` is empty for clean units and keyed by column or PARAM."""
    clean = VOLightCurve.from_table(_mag_table())
    assert clean.check_units() == {}
    volc = VOLightCurve.from_table(_mag_table())
    volc.photdms["phot"].photcal = PhotCal(zp_flux=1.0, zp_flux_unit="electron.s-1")
    findings = volc.check_units()
    assert list(findings) == ["photcal:phot:zeroPointFlux"]
    assert findings["photcal:phot:zeroPointFlux"].ok
    volc.photdms["phot"].photcal.zero_point._zp_flux_unit_text = "electron s-1"
    assert not volc.check_units()["photcal:phot:zeroPointFlux"].ok


def test_written_flat_route_links_the_error_column_to_the_group():
    """Step 2: a flux or magnitude error shares the group of the photometry (flat route)."""
    table = Table()
    table["obs_time"] = np.array([1.0, 2.0])
    table["mag"] = np.array([12.0, 12.1])
    table["mag"].info.meta = {"ucd": "phot.mag;em.opt"}
    table["mag_err"] = np.array([0.01, 0.01])
    table["mag_err"].info.meta = {"ucd": "stat.error;phot.mag;em.opt"}
    text = _write_bytes(table).decode()
    assert re.findall(r"<FIELDref ref=\"([^\"]+)\"", text) == ["mag", "mag_err"]
    group_id = re.search(r'<GROUP ID="([^"]+)"', text).group(1)
    assert re.search(rf'name="mag_err"[^>]*ref="{group_id}"', text)


def _two_flux_table() -> Table:
    """Returns a table with two fluxes and their errors, with photometry UCDs.

    Returns:
        astropy.table.Table: Columns ``obs_time``, ``sap_flux``, ``sap_flux_err``,
            ``pdcsap_flux`` and ``pdcsap_flux_err``.
    """
    table = Table()
    table["obs_time"] = np.array([1.0, 2.0])
    table["obs_time"].info.meta = {"ucd": "time.epoch"}
    for name in ("sap_flux", "pdcsap_flux"):
        table[name] = np.array([1.0, 1.1])
        table[name].info.meta = {"ucd": "phot.flux;em.opt"}
        table[f"{name}_err"] = np.array([0.1, 0.1])
        table[f"{name}_err"].info.meta = {"ucd": "stat.error;phot.flux;em.opt"}
    return table


def test_written_photdms_one_group_per_distinct_calibration():
    """Step 2: columns that share a ``PhotDM`` share a group; others get their own."""
    table = _two_flux_table()
    shared = PhotDM(photcal=PhotCal(zp_flux=1.0, zp_flux_unit="electron.s-1", zp_mag=20.0))
    other = PhotDM(photcal=PhotCal(zp_flux=2.0, zp_flux_unit="electron.s-1"))
    photdms = {
        "sap_flux": shared,
        "sap_flux_err": shared,
        "pdcsap_flux": other,
        "pdcsap_flux_err": other,
    }
    buffer = io.BytesIO()
    write_vo_lightcurve(buffer, table, table_name="t", photdms=photdms, timeorigin=0.0)
    text = buffer.getvalue().decode()
    groups = re.findall(r"<GROUP .*?</GROUP>", text, re.S)
    assert len(groups) == 2
    assert groups[0].count("<FIELDref") == 2
    assert groups[1].count("<FIELDref") == 2
    assert 'name="zeroPointReferenceMagnitude"' in groups[0]
    assert 'name="zeroPointReferenceMagnitude"' not in groups[1]
    assert "magnitudeSystem" not in text
    volc = VOLightCurve(io.BytesIO(buffer.getvalue()))
    assert volc.photdms["sap_flux_err"] is volc.photdms["sap_flux"]
    assert volc.photdms["pdcsap_flux"].photcal.zp_flux.value == pytest.approx(2.0)
    assert volc.photdms["sap_flux"].photcal.zp_flux.value == pytest.approx(1.0)


def test_written_photdms_unlisted_error_column_stays_unlinked():
    """Step 2: an error column that no ``PhotDM`` names is not linked to any group."""
    table = _two_flux_table()
    photdm = PhotDM(photcal=PhotCal(zp_flux=1.0, zp_flux_unit="electron.s-1"))
    buffer = io.BytesIO()
    write_vo_lightcurve(
        buffer, table, table_name="t", photdms={"sap_flux": photdm, "pdcsap_flux": photdm}
    )
    text = buffer.getvalue().decode()
    assert re.findall(r"<FIELDref ref=\"([^\"]+)\"", text) == ["sap_flux", "pdcsap_flux"]
    assert not re.search(r'name="sap_flux_err"[^>]*ref=', text)


def test_written_photdms_cannot_be_combined_with_the_flat_arguments():
    """Step 2: the deprecated flat arguments and ``photdms`` exclude each other."""
    with pytest.raises(ValueError, match="photdms"):
        _write_bytes(_mag_table(), photdms={}, zero_point_flux=1.0)


def test_interpretation_runs_on_a_compliant_votable_and_assigns_ucd_from_a_name():
    """MIXED (3): a VOTable FIELD without a UCD gets one from its name on ingest."""
    payload = _votable(
        fields=(
            '<FIELD name="obs_time" datatype="double" ucd="time.epoch" unit="d"/>'
            '<FIELD name="my_flux_col" datatype="double"/>'
        ),
        rows="<TR><TD>1.0</TD><TD>5.0</TD></TR>",
    )
    volc = VOLightCurve(io.BytesIO(payload))
    assert volc.table["my_flux_col"].info.meta.get("ucd") == "phot.flux"
    assert "column my_flux_col was given the UCD phot.flux" in volc.table.meta["description"]


def test_interpretation_name_promotion_is_recorded_for_non_votable_tables():
    """KEEP (agreed): roles from names are applied and recorded in the table meta."""
    table = Table()
    table["jd"] = np.array([2459000.0])
    table["mag"] = np.array([12.0])
    volc = VOLightCurve.from_table(table)
    assert volc.table["mag"].unit == u.mag
    assert "column mag was given the unit mag" in volc.table.meta["description"]
    assert "promoted_by_name" not in volc.table.meta


def test_heuristic_absolute_jd_threshold_in_offset_conversion():
    """MIXED (1): an offset at or above 2400000.5 is treated as an absolute JD."""
    assert time_offset_to_absolute_jd(2450000.0, 100.0) == pytest.approx(2450000.0)
    assert time_offset_to_absolute_jd(5000.0, 100.0) == pytest.approx(5100.0)


def test_photometry_filter_without_arguments_has_no_identifier():
    """KEEP: a bare ``PhotometryFilter`` has ``None`` as its identifier."""
    assert PhotometryFilter().filter_id is None


# ---------------------------------------------------------------------------
# Round 6: photcal given by PARAMref (DaCHS file, VOTable 1.5, with MIVOT)
# ---------------------------------------------------------------------------

_PARAMREF_FILE = Path(__file__).parent / "data" / "ogle_dachs_paramref.xml"


def test_keep_photcal_group_with_paramref_is_resolved():
    """KEEP (round 6): values given by ``PARAMref`` to table PARAMs are read.

    The file is a DaCHS export (OGLE). The group holds one inline PARAM and three
    PARAMrefs; the MIVOT block at the end is ignored.
    """
    volc = read_lightcurve(_PARAMREF_FILE, filename=_PARAMREF_FILE.name)
    assert volc.table.colnames == ["obs_time", "phot", "mag_err", "ogle_phase"]
    photdm = volc.photdms["phot"]
    photcal = photdm.photcal
    assert photcal.zero_point.zp_flux.to_value(u.Jy) == pytest.approx(2415.6497711876)
    assert photcal.zero_point.zp_mag is None
    assert photcal.magnitude_system.type == "Vega"
    assert photdm.filter.filter_id == "Generic/Bessell.I"
    assert photdm.filter.spectral_location.to_value(u.m) == pytest.approx(8.020142e-07)


def test_keep_paramref_file_time_system_is_read_exactly():
    """KEEP (round 6): the resource-level TIMESYS is read as declared."""
    volc = read_lightcurve(_PARAMREF_FILE, filename=_PARAMREF_FILE.name)
    assert volc.timesys.timescale == "UTC"
    assert volc.timesys.refposition == "HELIOCENTER"
    assert volc.timesys.timeorigin == pytest.approx(2400000.5)


def test_limit_paramref_file_error_column_has_no_calibration():
    """LIMIT (round 6): ``mag_err`` has no link to the group, so it has no entry."""
    volc = read_lightcurve(_PARAMREF_FILE, filename=_PARAMREF_FILE.name)
    assert "mag_err" not in volc.photdms


# ---------------------------------------------------------------------------
# Step 1 additions: absent stays absent on read
# ---------------------------------------------------------------------------


def test_absent_offset_conversion_without_origin_raises():
    """Step 1: an offset cannot become an absolute JD when the origin is not declared."""
    with pytest.raises(ValueError, match="not declared"):
        time_offset_to_absolute_jd(100.0, None)
    assert time_offset_to_absolute_jd(2459000.5, None) == pytest.approx(2459000.5)
    assert time_offset_to_absolute_jd(100.0, 0.0) == pytest.approx(100.0)


def test_absent_old_wire_forms_are_read_as_none():
    """Step 1: ``NaN`` and empty PARAMs written by earlier versions read as ``None``."""
    groups = """
    <GROUP ID="photcal_x" name="photcal">
      <PARAM name="zeroPointFlux" datatype="double" utype="photdm:PhotCal.zeroPoint.flux.value" value="NaN"/>
      <PARAM name="zeroPointReferenceMagnitude" datatype="double" utype="photdm:PhotCal.zeroPoint.referenceMagnitude.value" value="NaN"/>
      <PARAM name="magnitudeSystem" datatype="char" arraysize="*" utype="photdm:PhotCal.magnitudeSystem.type" value=""/>
      <PARAM name="filterIdentifier" datatype="char" arraysize="*" utype="photdm:PhotometryFilter.identifier" value=""/>
      <FIELDref ref="mag" utype="adhoc:location"/>
    </GROUP>
    """
    payload = _votable(
        groups=groups,
        fields='<FIELD ID="mag" name="mag" datatype="double" ucd="phot.mag" unit="mag"/>',
        rows="<TR><TD>12.0</TD></TR>",
    )
    photdm = VOLightCurve(io.BytesIO(payload)).photdms["mag"]
    assert photdm.photcal.zp_flux is None
    assert photdm.photcal.zp_mag is None
    assert photdm.photcal.mag_sys is None
    assert photdm.filter_id is None
    assert photdm.filter.spectral_location is None


# ---------------------------------------------------------------------------
# Step 4: check methods report, operations raise
# ---------------------------------------------------------------------------


def _calibrated_mag_curve(*, zp_flux=3631.0, zp_mag=0.0) -> VOLightCurve:
    """Builds a curve with a magnitude column and a given calibration.

    Args:
        zp_flux (float, optional): Zero-point flux in Jy; ``None`` leaves it absent.
        zp_mag (float, optional): Zero-point magnitude; ``None`` leaves it absent.

    Returns:
        VOLightCurve: Curve whose ``phot`` column carries the calibration.
    """
    volc = VOLightCurve.from_table(_mag_table())
    volc.photdms["phot"] = PhotDM(
        photcal=PhotCal(zp_flux=zp_flux, zp_flux_unit="Jy", zp_mag=zp_mag)
    )
    return volc


def test_checked_conversion_problems_name_the_missing_items():
    """Step 4: ``conversion_problems`` lists what is missing and changes nothing."""
    volc = VOLightCurve.from_table(_mag_table())
    assert any("no photometric calibration" in p for p in volc.conversion_problems("phot"))
    volc.photdms["phot"] = PhotDM(photcal=PhotCal(zp_flux=None, zp_mag=None))
    problems = volc.conversion_problems("phot")
    assert any("zp_flux" in p for p in problems)
    assert any("zp_mag" in p for p in problems)
    assert volc.table.colnames == ["obs_time", "phot", "flux_error"]
    assert _calibrated_mag_curve().conversion_problems("phot") == []
    with pytest.raises(KeyError):
        volc.conversion_problems("nothing")


def test_checked_conversion_without_calibration_raises_and_adds_no_column():
    """Step 4: no NaN column is invented; the conversion raises ``PhotCalError``."""
    volc = VOLightCurve.from_table(_mag_table())
    volc.photdms.pop("phot", None)
    with pytest.raises(PhotCalError):
        volc.add_flux_column_from_mag("phot")
    assert "flux_from_phot" not in volc.table.colnames
    volc.photdms["phot"] = PhotDM(photcal=PhotCal(zp_flux=None, zp_mag=0.0))
    with pytest.raises(PhotCalError) as caught:
        volc.add_flux_column_from_mag("phot", new_col_name="f")
    assert any("zp_flux" in p for p in caught.value.problems)
    assert "f" not in volc.table.colnames


def test_checked_conversion_with_calibration_adds_the_column_and_links_it():
    """Step 4: a complete calibration converts, names the column and shares the PhotDM."""
    volc = _calibrated_mag_curve()
    name = volc.add_flux_column_from_mag("phot")
    assert name == "flux_from_phot"
    assert volc.photdms[name] is volc.photdms["phot"]
    assert volc.table[name].unit.is_equivalent(u.Jy)
    back = volc.add_mag_column_from_flux(name)
    assert back == f"mag_from_{name}"
    assert volc.table[back].info.meta["ucd"] == "phot.mag"
    assert np.allclose(np.asarray(volc.table[back]), np.asarray(volc.table["phot"]))


def test_checked_photdm_conversion_problems():
    """Step 4: ``PhotDM.conversion_problems`` reports an absent calibration."""
    assert PhotDM().conversion_problems() == ["No photometric calibration (photcal) is declared."]
    assert PhotDM(photcal=PhotCal(zp_flux=1.0, zp_mag=0.0)).conversion_problems() == []


def test_checked_time_problems_name_the_missing_items():
    """Step 4: ``time_problems`` reports the absent origin, scale and position."""
    volc = VOLightCurve.from_table(_mag_table())
    problems = volc.time_problems()
    assert any("time origin" in p for p in problems)
    assert any("time scale" in p for p in problems)
    assert any("reference position" in p for p in problems)
    volc.timesys = TimeSys(refposition="BARYCENTER", timeorigin=2400000.5, timescale="TDB")
    assert volc.time_problems() == []


def test_checked_absolute_jd_values_need_no_origin():
    """Step 4: values at or above the absolute-JD threshold do not ask for an origin."""
    table = Table()
    table["jd"] = np.array([2459000.5, 2459001.5])
    table["mag"] = np.array([12.0, 12.1])
    volc = VOLightCurve.from_table(table)
    assert not any("time origin" in p for p in volc.time_problems())


# ---------------------------------------------------------------------------
# Step 5: interpretation split, strict mode, suggestions
# ---------------------------------------------------------------------------


def test_interpretation_functions_live_in_their_own_module_and_stay_importable():
    """Step 5: the guessing code is in ``interpretation``; old private names still resolve."""
    import volightcurve.lightcurve as core
    from volightcurve import apply_non_votable_heuristics, interpretation

    assert apply_non_votable_heuristics is interpretation.apply_non_votable_heuristics
    assert core._pickup_jd0_from_table is interpretation._pickup_jd0_from_table
    assert "_promote_to_vo_standards" not in vars(core)
    with pytest.raises(AttributeError):
        core.not_a_moved_name


def test_strict_votable_read_does_not_interpret_names_or_comments():
    """Step 5: strict mode leaves a FIELD without UCD alone and takes no ``JD0`` from comments."""
    payload = _votable(
        fields=(
            '<FIELD name="obs_time" datatype="double" ucd="time.epoch" unit="d"/>'
            '<FIELD name="my_flux_col" datatype="double"/>'
        ),
        rows="<TR><TD>1.0</TD><TD>5.0</TD></TR>",
    )
    tolerant = VOLightCurve(io.BytesIO(payload))
    strict = VOLightCurve(io.BytesIO(payload), strict=True)
    assert tolerant.table["my_flux_col"].info.meta.get("ucd") == "phot.flux"
    assert not strict.table["my_flux_col"].info.meta.get("ucd")
    assert "promoted_by_name" not in strict.table.meta
    assert "Interpreted" not in str(strict.table.meta.get("description") or "")
    assert strict.timesys.timeorigin is None


def test_strict_reading_rejects_what_is_not_a_votable():
    """Step 5: strict mode refuses ASCII input instead of interpreting it."""
    from volightcurve import LightcurveIOError

    with pytest.raises(ValueError, match="VOTable"):
        VOLightCurve(io.BytesIO(b"1 2\n3 4\n"), strict=True)
    with pytest.raises(LightcurveIOError, match="Strict"):
        read_lightcurve(io.StringIO("1 2\n"), filename="x.dat", strict=True)


def test_strict_reading_of_the_paramref_file_keeps_the_declared_calibration():
    """Step 5: a compliant file reads the same in strict mode."""
    volc = read_lightcurve(_PARAMREF_FILE, filename=_PARAMREF_FILE.name, strict=True)
    assert volc.photdms["phot"].photcal.zp_flux.to_value(u.Jy) == pytest.approx(2415.6497711876)
    assert volc.timesys.timeorigin == pytest.approx(2400000.5)


def test_core_import_does_not_need_gavo():
    """Step 5: importing and writing need no GAVO; only reading a VOTable does."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "sys.modules['gavo'] = None\n"
        "import volightcurve\n"
        "from astropy.table import Table\n"
        "import numpy as np, io\n"
        "t = Table(); t['obs_time'] = np.array([1.0]); t['phot'] = np.array([2.0])\n"
        "t['obs_time'].info.meta = {'ucd': 'time.epoch'}; t['phot'].info.meta = {'ucd': 'phot.mag'}\n"
        "volightcurve.write_vo_lightcurve(io.BytesIO(), t, table_name='x')\n"
        "print('ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_suggestions_are_plain_data_with_a_reason_each():
    """Step 5: the suggestion table holds the agreed values and changes nothing."""
    from volightcurve import SUGGESTION_REASONS, suggested_calibration, suggested_time_system

    assert suggested_time_system() == {
        "refposition": "HELIOCENTER",
        "timescale": "UTC",
        "timeorigin": 0.0,
    }
    calibration = suggested_calibration()
    assert calibration["magnitude_system"] == "Vega"
    assert calibration["zp_mag"] == 0.0
    assert calibration["zp_flux"] == 1.0
    assert calibration["zp_flux_unit"] is None
    suggested_time_system()["timeorigin"] = 5.0
    assert suggested_time_system()["timeorigin"] == 0.0
    for key in list(calibration) + list(suggested_time_system()):
        assert SUGGESTION_REASONS[key]


def test_suggestions_for_missing_name_only_the_absent_fields():
    """Step 5: only absent fields get a suggestion; declared ones are left alone."""
    from volightcurve import suggestions_for_missing

    volc = VOLightCurve.from_table(_mag_table())
    volc.timesys = TimeSys(timeorigin=2400000.5)
    volc.photdms["phot"] = PhotDM(photcal=PhotCal(zp_flux=5.0, zp_flux_unit="Jy", mag_sys="AB"))
    found = suggestions_for_missing(volc)
    assert set(found["time_system"]) == {"refposition", "timescale"}
    assert set(found["calibration"]["phot"]) == {"zp_mag", "zp_mag_unit"}
    assert volc.timesys.refposition is None
    assert volc.photdms["phot"].photcal.zp_mag is None


# ---------------------------------------------------------------------------
# Step 5b: what interpretation did is recorded in the table description
# ---------------------------------------------------------------------------


def _dat_table_with_comments() -> Table:
    """Returns a table with a loose ``JD0``, ``PERIOD`` and ``FILTER`` in its comments.

    Returns:
        astropy.table.Table: Two columns, ``jd`` and ``mag``, with keyword comments.
    """
    table = Table()
    table["jd"] = np.array([100.0, 101.0])
    table["mag"] = np.array([12.0, 12.1])
    table.meta["comments"] = ["JD0 = 2450000", "PERIOD = 0.5", "FILTER = Generic/Bessell.V"]
    return table


def test_described_interpretation_is_one_sentence_per_kind_in_the_description():
    """Step 5b: names and comment keywords are recorded as plain sentences in the description."""
    volc = VOLightCurve.from_table(_dat_table_with_comments())
    description = volc.table.meta["description"]
    assert "(from column names)" in description
    assert "column mag was given the unit mag" in description
    assert "(from comments or metadata keywords)" in description
    assert "time origin 2450000.0" in description
    assert "period 0.5" in description
    assert "filter identifier Generic/Bessell.V" in description
    assert "promoted_by_name" not in volc.table.meta


def test_described_nothing_interpreted_adds_no_sentence():
    """Step 5b: a compliant VOTable read strictly, and tolerantly, gets no sentence."""
    for strict in (True, False):
        volc = read_lightcurve(_PARAMREF_FILE, filename=_PARAMREF_FILE.name, strict=strict)
        assert "Interpreted" not in volc.table.meta["description"]


def test_described_sentence_has_no_keyword_pattern_and_is_not_added_twice():
    """Step 5b: the sentence cannot be re-read as a keyword, and a second pass adds nothing."""
    from volightcurve.io_keywords import is_metadata_assignment_line
    from volightcurve.interpretation import add_interpretation_note

    volc = VOLightCurve.from_table(_dat_table_with_comments())
    description = volc.table.meta["description"]
    assert not is_metadata_assignment_line(description)
    assert "=" not in description
    add_interpretation_note(
        volc.table,
        "from comments or metadata keywords",
        ["time origin 2450000.0", "period 0.5", "filter identifier Generic/Bessell.V"],
    )
    assert volc.table.meta["description"] == description


def test_described_sentences_reach_the_written_votable_description():
    """Step 5b: the written VOTable carries the sentences in its DESCRIPTION."""
    from volightcurve.io import write_lightcurve

    volc = VOLightCurve.from_table(_dat_table_with_comments())
    text = write_lightcurve(volc, "votable").decode()
    assert "Interpreted from the file, not declared in it" in text
    assert "time origin 2450000.0" in text


def test_described_sentences_survive_next_to_narrative_comments_in_every_format():
    """Step 5b: sentences and the author's own comment both reach all four written formats."""
    from volightcurve.io import write_lightcurve

    table = _dat_table_with_comments()
    table.meta["comments"] = ["JD0 = 2450000", "Observed at Site X."]
    volc = VOLightCurve.from_table(table)
    for fmt in ("votable", "dat", "csv", "ecsv"):
        out = write_lightcurve(volc, fmt)
        text = out.decode() if isinstance(out, bytes) else out
        assert "Interpreted from the file, not declared in it" in text, fmt
        assert "Observed at Site X." in text, fmt


# ---------------------------------------------------------------------------
# Step 6: VOTABLE description and INFO elements are read and written back
# ---------------------------------------------------------------------------

_OGLE_FULL = Path(__file__).parent / "data" / "ogle_lmc_rrlyr_13820_full.vot"


def _infos_named(volc, name):
    """Returns the values of every INFO called ``name``.

    Args:
        volc (VOLightCurve): Product to look at.
        name (str): INFO name.

    Returns:
        list[str]: Values in file order.
    """
    return [entry["value"] for entry in volc.infos if entry["name"] == name]


def test_declared_votable_description_and_infos_are_read_separately():
    """Step 6: the VOTABLE text, the table text and every INFO are kept, none merged."""
    volc = read_lightcurve(_OGLE_FULL, filename=_OGLE_FULL.name)
    assert volc.votable_description.startswith("The **Optical Gravitational Lensing Experiment")
    assert volc.table.meta["description"] == "The OGLE lightcurve for OGLE-LMC-RRLYR-13820 in I passband"
    assert len(_infos_named(volc, "creator")) == 19
    assert _infos_named(volc, "legal")
    assert _infos_named(volc, "rights")
    assert _infos_named(volc, "publisher")


def test_votable_round_trip_keeps_both_descriptions_and_every_info():
    """Step 6: a VOTable written from a VOTable read loses neither description nor INFO."""
    from volightcurve.io import write_lightcurve

    volc = read_lightcurve(_OGLE_FULL, filename=_OGLE_FULL.name)
    again = read_lightcurve(io.BytesIO(write_lightcurve(volc, "votable")), filename="x.vot")
    # The XML writer re-wraps long text (even after a hyphen), so compare without whitespace.
    assert "".join(again.votable_description.split()) == "".join(volc.votable_description.split())
    assert again.table.meta["description"] == volc.table.meta["description"]
    assert _infos_named(again, "creator") == _infos_named(volc, "creator")
    assert _infos_named(again, "legal") == _infos_named(volc, "legal")


def test_text_formats_carry_no_info_elements():
    """Step 6: ECSV, DAT and CSV stay concise, with no INFO or VOTABLE text."""
    from volightcurve.io import write_lightcurve

    volc = read_lightcurve(_OGLE_FULL, filename=_OGLE_FULL.name)
    for fmt in ("dat", "csv", "ecsv"):
        out = write_lightcurve(volc, fmt)
        text = out.decode() if isinstance(out, bytes) else out
        assert "Optical Gravitational" not in text, fmt
        assert "legal" not in text.lower(), fmt


def test_text_formats_keep_only_the_interpretation_sentences():
    """Step 6b: the source description is not copied into dat, csv or ecsv; the sentences are."""
    from volightcurve.io import write_lightcurve

    table = _dat_table_with_comments()
    volc = VOLightCurve.from_table(table)
    volc.table.meta["description"] = "Long source text. " + volc.table.meta["description"]
    for fmt in ("dat", "csv", "ecsv"):
        out = write_lightcurve(volc, fmt)
        text = out.decode() if isinstance(out, bytes) else out
        assert "Long source text" not in text, fmt
        assert "Interpreted from the file, not declared in it" in text, fmt
