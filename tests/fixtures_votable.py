"""Shared VOTable byte fixtures for pure package tests."""

from __future__ import annotations


def gaia_style_votable(*, include_epoch: bool = True, second_time_col: bool = False) -> bytes:
    """Builds a Gaia-like VOTable with Gaia-style timeorigin and optional epoch PARAM.

    Args:
        include_epoch: When True, include an unreferenced ``epoch`` PARAM.
        second_time_col: When True, add a second time FIELD with a different TIMESYS.

    Returns:
        UTF-8 VOTable document as bytes.
    """
    epoch_param = ""
    if include_epoch:
        epoch_param = (
            '<PARAM name="epoch" datatype="double" ucd="time.epoch" unit="d" '
            'value="2207.1263399818404"/>'
        )
    second_field = ""
    second_row_cell = ""
    if second_time_col:
        second_field = (
            '<FIELD name="aux_time" datatype="double" ucd="time.epoch" unit="d" ref="ts2"/>'
        )
        second_row_cell = "<TD>1.0</TD>"

    return f"""<?xml version="1.0" encoding="utf-8"?>
<VOTABLE xmlns="http://www.ivoa.net/xml/VOTable/v1.3" version="1.4">
  <RESOURCE>
    <TIMESYS ID="ts" refposition="BARYCENTER" timeorigin="2455197.5" timescale="TCB"/>
    <TIMESYS ID="ts2" refposition="BARYCENTER" timeorigin="2400000.5" timescale="TCB"/>
    <GROUP ID="phot_def" name="photcal">
      <PARAM name="filterIdentifier" datatype="char" arraysize="*" utype="photDM:PhotometryFilter.identifier" value="GAIA/GAIA3.G"/>
      <PARAM name="zeroPointFlux" datatype="double" utype="photDM:PhotCal.zeroPoint.flux.value" value="1.0"/>
      <PARAM name="zeroPointReferenceMagnitude" datatype="double" utype="photDM:PhotCal.zeroPoint.referenceMagnitude.value" value="0.0" unit="mag"/>
      <PARAM name="magnitudeSystem" datatype="char" arraysize="*" utype="photDM:PhotCal.magnitudeSystem.type" value="Vega"/>
      <FIELDref ref="phot"/>
    </GROUP>
    <TABLE name="Gaia epoch test">
      <DESCRIPTION>Epoch TIMESYS linkage test product.</DESCRIPTION>
      <FIELD name="obs_time" datatype="double" ucd="time.epoch" unit="d" ref="ts"/>
      {second_field}
      <FIELD name="phot" datatype="double" ucd="phot.mag;em.opt" unit="mag" ref="phot_def"/>
      <FIELD name="flux_error" datatype="double" ucd="stat.error;phot.mag" unit="mag"/>
      {epoch_param}
    <DATA>
        <TABLEDATA>
          <TR><TD>55197.0</TD>{second_row_cell}<TD>12.34</TD><TD>0.01</TD></TR>
        </TABLEDATA>
      </DATA>
    </TABLE>
  </RESOURCE>
</VOTABLE>
""".encode()


def annotate_vo_roles(table):
    """Gives a bare test table the UCDs the writer now requires (it never guesses roles).

    Time columns named ``obs_time``, ``time``, ``jd`` or ``mjd`` get ``time.epoch``;
    ``phot`` and ``mag`` get ``phot.mag``; ``flux`` gets ``phot.flux``; ``flux_error``
    and ``mag_err`` get the matching error UCD. A column that already has a UCD is
    left alone.

    Args:
        table (astropy.table.Table): Table built by a test.

    Returns:
        astropy.table.Table: The same table, annotated in place.
    """
    ucds = {
        "obs_time": "time.epoch",
        "time": "time.epoch",
        "jd": "time.epoch",
        "mjd": "time.epoch",
        "phot": "phot.mag",
        "mag": "phot.mag",
        "flux": "phot.flux;em.opt",
        "mag_err": "stat.error;phot.mag",
        "flux_error": "stat.error;phot.mag",
    }
    for name in table.colnames:
        if name in ucds and not (table[name].info.meta or {}).get("ucd"):
            table[name].info.meta = {**(table[name].info.meta or {}), "ucd": ucds[name]}
    return table
