"""Pure-package tests for TIMESYS helpers and VOLightCurve ingest."""

from __future__ import annotations

import io

import pytest

from fixtures_votable import gaia_style_votable
from volightcurve import VOLightCurve
from volightcurve.lightcurve import _gavo_votable_tree_from_source
from volightcurve.time_reference import (
    absolute_jd_to_time_offset,
    extract_timesys_metadata_from_gavo,
    extract_timesys_registry_from_gavo,
    normalise_table_epoch_to_absolute_jd,
    time_offset_to_absolute_jd,
)


def test_gavo_timesys_registry_extracts_multiple_systems():
    """GAVO TIMESYS walker builds a registry keyed by ``TIMESYS/@ID``."""
    tree = _gavo_votable_tree_from_source(io.BytesIO(gaia_style_votable()))
    registry = extract_timesys_registry_from_gavo(tree)
    assert set(registry) == {"ts", "ts2"}
    assert registry["ts"].timeorigin == pytest.approx(2455197.5)
    assert registry["ts2"].timeorigin == pytest.approx(2400000.5)
    assert registry["ts"].timescale == "TCB"


def test_gavo_timesys_metadata_maps_table_field_and_param_refs():
    """GAVO walker resolves TABLE FIELD/PARAM ``ref`` attributes."""
    tree = _gavo_votable_tree_from_source(io.BytesIO(gaia_style_votable()))
    metadata = extract_timesys_metadata_from_gavo(tree)
    assert metadata.field_refs["obs_time"] == "ts"
    assert metadata.param_refs["epoch"] is None
    assert metadata.default_timesys.timeorigin == pytest.approx(2455197.5)


def test_volightcurve_ingest_uses_gavo_timesys_registry():
    """VOLightCurve ingest populates TIMESYS registry via GAVO walkers."""
    volc = VOLightCurve(io.BytesIO(gaia_style_votable()))
    assert set(volc.timesys_by_id) == {"ts", "ts2"}
    assert volc.field_timesys_ref["obs_time"] == "ts"
    assert volc.timesys.timeorigin == pytest.approx(2455197.5)


def test_time_offset_roundtrip_helpers():
    """Offset and absolute Julian Date helpers are inverse for a given timeorigin."""
    origin = 2455197.5
    offset = 2207.1263399818404
    absolute = time_offset_to_absolute_jd(offset, origin)
    assert absolute == pytest.approx(origin + offset)
    assert absolute_jd_to_time_offset(absolute, origin) == pytest.approx(offset)


def test_normalise_epoch_inherits_obs_time_timesys_when_unreferenced():
    """Unreferenced epoch PARAM uses the observation column TIMESYS."""
    volc = VOLightCurve(io.BytesIO(gaia_style_votable()))
    absolute = normalise_table_epoch_to_absolute_jd(
        volc,
        volc.table.meta["epoch"],
    )
    assert absolute == pytest.approx(2455197.5 + 2207.1263399818404)
