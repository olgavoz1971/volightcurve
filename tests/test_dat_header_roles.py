"""Tests for ``.dat`` header selection, role typing, and failback (§4b)."""

from __future__ import annotations

import io

import pytest

from volightcurve import read_lightcurve
from volightcurve.io_dat import dat_column_role, read_dat_table, select_dat_header_names
from volightcurve.io_errors import LightcurveIOError


def test_dat_column_role_sense():
    """Name→role map matches the I/O contract sense rules."""
    assert dat_column_role("JDhel2400000+") == "time"
    assert dat_column_role("Rmag") == "mag"
    assert dat_column_role("error") == "error"
    assert dat_column_role("mag_err") == "error"
    assert dat_column_role("flux_error") == "error"
    assert dat_column_role("Observ") == "other"
    assert dat_column_role("label") == "other"
    assert dat_column_role("phot") == "mag"


def test_select_dat_header_prefers_last_role_canonical():
    """Last role-canonical width match wins over an earlier legacy header."""
    comments = [
        "JDhel2400000+ Rmag error Observ",
        "ASASSN17pm",
        "JD0=2400000",
        "jd mag mag_err label",
    ]
    assert select_dat_header_names(comments, 4) == [
        "jd",
        "mag",
        "mag_err",
        "label",
    ]


def test_dual_header_label_column_is_string():
    """Shugarov-style dual headers: canonical last header; label stays text."""
    text = """# JDhel2400000+ Rmag  error  Observ
# ASASSN17pm
# JD0=2400000
# FILTER=R
# MAG0=10
# jd         mag    mag_err label
58079.54425 12.416  0.000  00
58079.92294 12.888  0.006  TR
"""
    table = read_dat_table(io.BytesIO(text.encode("utf-8")))
    assert list(table.colnames) == ["jd", "mag", "mag_err", "label"]
    assert table["label"][0] == "00"
    assert table["label"][1] == "TR"
    assert table["mag"][0] == pytest.approx(12.416)


def test_observ_last_column_is_free_text_by_role():
    """Last header token may be any name; role other → string."""
    text = """# JDhel2400000+ Rmag error Observ
58079.54425 12.416  0.000  00
58079.92294 12.888  0.006  TR
"""
    table = read_dat_table(io.BytesIO(text.encode("utf-8")))
    assert list(table.colnames) == ["JDhel2400000+", "Rmag", "error", "Observ"]
    assert table["Observ"][1] == "TR"


def test_invalid_roles_width4_failback():
    """Structural failure at width 4 renames to jd mag mag_err label."""
    text = """# mag1 mag2 mag3 Observ
58079.54425 12.416  0.000  00
58079.92294 12.888  0.006  TR
"""
    table = read_dat_table(io.BytesIO(text.encode("utf-8")))
    assert list(table.colnames) == ["jd", "mag", "mag_err", "label"]
    assert table["label"][1] == "TR"


def test_invalid_roles_width5_hard_fail():
    """Failback is only for width 3 or 4."""
    text = """# a b c d e
1 2 3 4 5
"""
    with pytest.raises(LightcurveIOError, match="Failback applies only"):
        read_dat_table(io.BytesIO(text.encode("utf-8")))


def test_read_lightcurve_shugarov_style_smoke():
    """End-to-end ingest keeps string labels through VOLightCurve."""
    text = """# JDhel2400000+ Rmag error Observ
# jd mag mag_err label
# JD0 = 2400000
58079.54425 12.416  0.000  00
58079.92294 12.888  0.006  TR
"""
    volc = read_lightcurve(io.BytesIO(text.encode("utf-8")), filename="R.dat")
    assert "label" in volc.table.colnames
    assert list(volc.table["label"]) == ["00", "TR"]
