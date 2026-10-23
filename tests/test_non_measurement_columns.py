"""Columns that describe the measurements (limits, signal-to-noise) are not photometry columns."""

from __future__ import annotations

import pytest
from astropy.table import Table

from volightcurve.lightcurve import (
    column_main_role,
    get_flux_colnames,
    get_mag_colnames,
    is_measurement_ucd,
)


@pytest.mark.parametrize(
    "ucd, expected",
    [
        ("phot.mag;em.opt", True),
        ("phot.flux;em.opt", True),
        ("stat.error;phot.flux;em.opt", True),
        ("phot.mag;em.opt;stat.max;instr.sensitivity;obs.exposure", False),
        ("phot.flux;em.opt;stat", False),
        ("stat.snr;phot.flux", False),
    ],
)
def test_is_measurement_ucd(ucd, expected):
    """Limits, bare ``stat`` and ``stat.snr`` words mark a non-measurement."""
    assert is_measurement_ucd(ucd) is expected


def _table() -> Table:
    table = Table({"mag": [1.0], "limitmag": [2.0], "flux": [3.0], "flux_over_error": [4.0], "snr": [5.0]})
    table["mag"].info.meta = {"ucd": "phot.mag;em.opt"}
    table["limitmag"].info.meta = {"ucd": "phot.mag;em.opt;stat.max;instr.sensitivity"}
    table["flux"].info.meta = {"ucd": "phot.flux;em.opt"}
    table["flux_over_error"].info.meta = {"ucd": "phot.flux;em.opt;stat"}
    table["snr"].info.meta = {"ucd": "stat.snr;phot.flux"}
    return table


def test_photometry_columns_exclude_non_measurements():
    """Only the measurement columns are listed as magnitude and flux."""
    table = _table()
    assert get_mag_colnames(table) == ["mag"]
    assert get_flux_colnames(table) == ["flux"]


def test_column_main_role_of_non_measurements_is_none():
    """A limit or a signal-to-noise column has no photometry role."""
    table = _table()
    assert column_main_role(table, "mag") == "mag"
    assert column_main_role(table, "limitmag") is None
    assert column_main_role(table, "flux_over_error") is None
    assert column_main_role(table, "snr") is None
