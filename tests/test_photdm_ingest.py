"""Tests for GAVO-based PhotDM ingestion in VOLightCurve."""

from __future__ import annotations

import io

import pytest
from gavo.votable import votparse

from volightcurve import VOLightCurve
from volightcurve.lightcurve import extract_photdm
from fixtures_votable import gaia_style_votable as _gaia_style_votable


def test_extract_photdm_resolves_inline_params():
    """Inline photcal PARAM entries (Gaia-style) map to the phot column."""
    payload = _gaia_style_votable()
    photdms = extract_photdm(votparse.readRaw(io.BytesIO(payload)))
    assert "phot" in photdms
    photdm = photdms["phot"]
    assert photdm.filter.filter_id == "GAIA/GAIA3.G"
    assert float(photdm.photcal.zp_flux.value) == pytest.approx(1.0)


def test_volightcurve_ingest_uses_gavo_photdm_for_inline_params():
    """VOLightCurve attaches Gaia-style photcal metadata via the GAVO walker."""
    volc = VOLightCurve(io.BytesIO(_gaia_style_votable()))
    photdm = volc.photdms["phot"]
    assert photdm.filter.filter_id == "GAIA/GAIA3.G"
