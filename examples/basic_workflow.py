"""Minimal volightcurve workflow: assemble, plot mag/flux, export.

Run from the package root with an editable install (or ``PYTHONPATH=src``)::

    python examples/basic_workflow.py

Uses matplotlib only for display. No web UI types.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.table import Table

from volightcurve import (
    PhotCal,
    assemble_volightcurve,
    read_lightcurve,
    write_lightcurve,
)

logger = logging.getLogger(__name__)


def _demo_volc():
    """Builds a short magnitude lightcurve with a coherent PhotCal pair.

    Returns:
        VOLightCurve: Sample product with absolute JD and Vega ZPs.
    """
    table = Table(
        {
            "jd": np.array([2459000.0, 2459000.5, 2459001.0, 2459001.5]),
            "mag": np.array([12.10, 12.25, 12.05, 12.18]) * u.mag,
            "mag_err": np.array([0.01, 0.012, 0.011, 0.01]) * u.mag,
        }
    )
    return assemble_volightcurve(
        table,
        timeorigin=0.0,
        filter_id="Generic/Johnson.V",
        filter_name="V",
        zp_flux=3631.0,
        zp_flux_unit="Jy",
        zp_mag=0.0,
        zp_mag_unit="mag",
        mag_sys="Vega",
        free_comments=["Example lightcurve for volightcurve basic_workflow."],
    )


def main() -> None:
    """Runs mag↔flux↔mag (PhotCal), plot, export, and re-read demo."""
    logging.basicConfig(level=logging.INFO)

    volc = _demo_volc()
    logger.info("Assembled %s", volc)

    time_col = volc.get_time_colnames()[0]
    t = np.asarray(volc.table[time_col], dtype=float)
    mag = np.asarray(volc.table["mag"], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)

    axes[0].errorbar(t, mag, yerr=np.asarray(volc.table["mag_err"], dtype=float), fmt="o")
    axes[0].invert_yaxis()
    axes[0].set_xlabel("JD")
    axes[0].set_ylabel("Magnitude")
    axes[0].set_title("Magnitude view")

    # Direct PhotCal façade: mag → flux → mag round-trip (+ error helpers)
    photdm = next(iter(volc.photdms.values()))
    pc: PhotCal = photdm.photcal
    mag_q = volc.table["mag"]
    mag_err_q = volc.table["mag_err"]
    flux_q = pc.mag_to_flux(mag_q)
    mag_back = pc.flux_to_mag(flux_q)
    flux_err_q = pc.mag_err_to_flux_err(mag_q, mag_err_q)
    mag_err_back = pc.flux_err_to_mag_err(flux_q, flux_err_q)

    logger.info(
        "PhotCal %s: mag[0]=%s → flux[0]=%s → mag[0]=%s",
        type(pc.zero_point).__name__,
        mag_q[0],
        flux_q[0],
        mag_back[0],
    )
    np.testing.assert_allclose(
        mag_back.to_value(u.mag),
        np.asarray(mag_q, dtype=float),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        mag_err_back.to_value(u.mag),
        np.asarray(mag_err_q, dtype=float),
        rtol=1e-10,
    )
    logger.info(
        "Error round-trip OK: mag_err[0]=%s ↔ flux_err[0]=%s",
        mag_err_q[0],
        flux_err_q[0],
    )

    flux_col = volc.add_flux_column_from_mag("mag", new_col_name="flux")
    flux = np.asarray(volc.table[flux_col], dtype=float)
    axes[1].errorbar(
        t,
        flux,
        yerr=np.asarray(flux_err_q.value, dtype=float),
        fmt="o",
    )
    axes[1].set_xlabel("JD")
    axes[1].set_ylabel(f"Flux ({volc.table[flux_col].unit})")
    axes[1].set_title("Flux view (PhotCal mag→flux)")

    out_dir = Path(tempfile.mkdtemp(prefix="volightcurve_demo_"))
    dat_path = out_dir / "demo.dat"
    vot_path = out_dir / "demo.vot"
    write_lightcurve(volc, "ascii.commented_header", destination=dat_path)
    write_lightcurve(volc, "votable_binary", destination=vot_path)
    logger.info("Wrote %s and %s", dat_path, vot_path)

    restored = read_lightcurve(dat_path, filename=str(dat_path.name))
    logger.info("Re-read .dat → %s", restored)

    fig.suptitle("volightcurve basic workflow")
    plt.show()


if __name__ == "__main__":
    main()
