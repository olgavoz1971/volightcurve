"""Unit tests for Ticket 8 Phase 0 keyword vocabulary helpers."""

from volightcurve.io_keywords import (
    KEY_FILTER,
    KEY_JD0,
    KEY_ZP_MAG,
    canonical_keyword,
    is_metadata_assignment_line,
    parse_keyword_assignment,
)


def test_canonical_aliases():
    """MAG0 and BAND map to preferred ZP_MAG and FILTER names."""
    assert canonical_keyword("mag0") == KEY_ZP_MAG
    assert canonical_keyword("BAND") == KEY_FILTER
    assert canonical_keyword("jd0") == KEY_JD0


def test_parse_calibration_assignment():
    """Recognised KEY = value lines return canonical key and raw value."""
    assert parse_keyword_assignment("JD0 = 2400000.5") == (KEY_JD0, "2400000.5")
    assert parse_keyword_assignment("MAG0=20.0") == (KEY_ZP_MAG, "20.0")
    assert parse_keyword_assignment("FILTER = TESS/TESS.Red") == (
        KEY_FILTER,
        "TESS/TESS.Red",
    )


def test_unknown_assignment_is_not_calibration():
    """Non-vocabulary KEY = value lines stay free-text (not calibration)."""
    assert parse_keyword_assignment("NAME = Target A") is None
    assert is_metadata_assignment_line("NAME = Target A") is False
    assert is_metadata_assignment_line("# column names only") is False
