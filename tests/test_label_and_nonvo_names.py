"""Label discovery and non-VO export column names (I/O contract §8)."""

import io

import numpy as np
from astropy.table import Table

from volightcurve import (
    assemble_volightcurve,
    get_label_colnames,
    read_lightcurve,
    write_lightcurve,
)


def _ucd(table: Table, name: str, ucd: str) -> None:
    """Sets a column UCD in place.

    Args:
        table (astropy.table.Table): Table to annotate.
        name (str): Column name.
        ucd (str): UCD string.
    """
    if table[name].info.meta is None:
        table[name].info.meta = {}
    table[name].info.meta["ucd"] = ucd


def test_label_prefers_leftmost_meta_id_ucd():
    """A ``meta.id`` column wins over an earlier free-text column."""
    table = Table()
    table["note"] = ["a", "b"]
    table["sector_id"] = ["40", "41"]
    table["jd"] = np.array([2459000.0, 2459001.0])
    _ucd(table, "jd", "time.epoch")
    _ucd(table, "sector_id", "meta.id;meta.dataset")
    assert get_label_colnames(table) == ["sector_id"]


def test_label_falls_back_to_leftmost_non_science_column():
    """Without a label UCD, the leftmost non-science column is the label."""
    table = Table()
    table["jd"] = np.array([2459000.0, 2459001.0])
    table["mag"] = np.array([12.0, 12.1])
    table["plate"] = ["A", "B"]
    table["note"] = ["x", "y"]
    _ucd(table, "jd", "time.epoch")
    _ucd(table, "mag", "phot.mag")
    assert get_label_colnames(table) == ["plate"]


def test_per_row_date_is_not_the_label():
    """A unique timestamp column is skipped; a repeated camera name is the label."""
    n = 40
    table = Table()
    table["jd"] = np.linspace(2459000.0, 2459040.0, n)
    table["mag"] = np.full(n, 16.2)
    table["UT Date"] = [f"2012-02-{i:02d}.1" for i in range(1, n + 1)]
    table["Camera"] = ["ba" if i < 20 else "bb" for i in range(n)]
    _ucd(table, "jd", "time.epoch")
    _ucd(table, "mag", "phot.mag")
    assert get_label_colnames(table) == ["Camera"]


def test_dat_ingest_assigns_meta_id_without_renaming():
    """A free-text column becomes the label and receives ``meta.id``."""
    payload = b"""# JD0 = 0
# jd mag mag_err plate
2459000.0 12.1 0.01 A
2459001.0 12.2 0.02 B
"""
    volc = read_lightcurve(io.BytesIO(payload), filename="lc.dat")
    assert "plate" in volc.table.colnames
    assert volc.get_label_colnames() == ["plate"]
    assert volc.table["plate"].info.meta["ucd"] == "meta.id"


def test_existing_label_ucd_is_kept():
    """Ingest does not replace a UCD that is already in the label-role list."""
    table = Table()
    table["jd"] = np.array([2459000.0, 2459001.0])
    table["mag"] = np.array([12.0, 12.1])
    table["flag"] = ["x", "y"]
    _ucd(table, "jd", "time.epoch")
    _ucd(table, "mag", "phot.mag")
    _ucd(table, "flag", "meta.code")
    volc = assemble_volightcurve(table, filter_id="Generic/Bessell.V")
    assert volc.table["flag"].info.meta["ucd"] == "meta.code"


def test_csv_export_numbers_repeated_magnitude_columns():
    """Two magnitude columns become ``mag-1`` and ``mag-2``; extras stay."""
    table = Table()
    table["obs_time"] = np.array([2459000.0, 2459001.0])
    table["g"] = np.array([12.0, 12.1])
    table["r"] = np.array([11.0, 11.1])
    table["plate"] = ["A", "B"]
    _ucd(table, "obs_time", "time.epoch")
    _ucd(table, "g", "phot.mag")
    _ucd(table, "r", "phot.mag")
    volc = assemble_volightcurve(table, filter_id="Generic/Bessell.V", zp_flux=1.0, zp_mag=0.0)
    text = write_lightcurve(volc, "csv").decode("utf-8")
    header = next(line for line in text.splitlines() if line.startswith("obs_time") or line.startswith("jd"))
    assert header.split(",")[:4] == ["jd", "mag-1", "mag-2", "plate"]
    restored = read_lightcurve(io.BytesIO(text.encode()), filename="lc.csv")
    assert "mag-1" in restored.table.colnames
    assert "plate" in restored.table.colnames
