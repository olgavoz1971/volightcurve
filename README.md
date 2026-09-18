# volightcurve

IVOA-oriented lightcurve I/O and photometric calibration for Python.

Annotation of time, photometry, and related metadata largely follows the
IVOA Note *[Time Series: Annotation of light curves in
VOTable](https://www.ivoa.net/documents/Notes/LightCurveTimeSeries/index.html)*
(Nebot et al., 22 November 2023, v0.1). That document is an **IVOA Note**
(guidance and proposed conventions), not an endorsed Recommendation; this
package implements those ideas **closely but not exhaustively** (for
example TIMESYS, PhotDM/PhotCal, and lightcurve-oriented VOTable usage).
Support for everyday non-VOTable files is covered below — it is pragmatic,
not a proposed standard.

Standalone **src-layout** package (`import volightcurve`). No web UI
dependency. Day-to-day edits belong in this tree; install into the host app
venv with ``pip install -e``.

## What it does

- Read and write lightcurves as **VOTable**, and also **ECSV**, **CSV**, and
  commented ASCII (``.dat``) — see [Non-VOTable formats](#non-votable-formats-dat-csv-ecsv)
  for what that compromise means.
- Carry **TIMESYS** (time origin / timescale / reference position) and
  **PhotDM / PhotCal** (filter, zero points, magnitude system) where present.
- Convert magnitude ↔ flux with unit-aware ``PhotCal`` when calibration exists
  (nested PhotDM: ``PhotCal`` → ``ZeroPoint``; Pogson today, Asinh/Linear stubs).
- Fail explicitly when calibration is incomplete — writers do **not** invent
  zero points.

## PhotDM / ``PhotCal`` conversion API

Photometric calibration is shaped after the IVOA Recommendation
*[Photometry Data Model (PhotDM) 1.1](https://www.ivoa.net/documents/PHOTDM/20221101/PhotDMv1-1.html)*
(Salgado et al.). We try to stay **close** to that model — nested
``PhotCal`` → ``MagnitudeSystem`` + ``ZeroPoint`` (Pogson / Asinh /
Linear) — without claiming a full PhotDM implementation (filters,
transmission curves, and several PhotDM types remain simplified or
absent). A handy structural sketch is the IVOA sample client
[photcal.py](https://github.com/ivoa/modelinstanceinvot-code/blob/merge-syntax/python/client/photdm/photcal.py).

Note: our class named ``PhotDM`` is a **per-column hub** on
``VOLightCurve`` (filter + photcal). PhotDM 1.1 has no object type of
that name; its binding class for calibration is ``PhotCal``.

```text
VOLightCurve.photdms[column]
  └── PhotDM                 ← hub per photometry column
        ├── photometryFilter
        └── photcal (PhotCal)  ← façade — call mag_to_flux / … here
              ├── magnitudeSystem
              └── zeroPoint
                    ├── PogsonZeroPoint      ← live mag↔flux + error helpers
                    ├── AsinhZeroPoint       ← stub (softeningParameter reserved)
                    └── LinearFluxZeroPoint  ← stub
```

```python
from volightcurve import PhotCal
import astropy.units as u

pc = PhotCal(zp_flux=3631.0, zp_flux_unit="Jy", zp_mag=0.0, mag_sys="Vega")
flux = pc.mag_to_flux(12.5 * u.mag)
mag = pc.flux_to_mag(flux)  # round-trip
flux_err = pc.mag_err_to_flux_err(12.5 * u.mag, 0.01 * u.mag)
mag_err = pc.flux_err_to_mag_err(flux, flux_err)
```

See also ``examples/basic_workflow.py`` (full mag↔flux↔mag + error round-trip
on an assembled lightcurve).

Do **not** embed Pogson / asinh formulae in host apps — always call ``PhotCal``.
Per-column PhotCals on ``VOLightCurve.photdms`` are the long-term model (mag and
flux columns may differ).

## Install (editable, recommended)

From a virtualenv that already has (or will install) the declared dependencies:

```bash
pip install -e /path/to/volightcurve
```

```python
from volightcurve import (
    read_lightcurve,
    write_lightcurve,
    assemble_volightcurve,
    VOLightCurve,
    PhotCal,
)
```

## Cursor Agent skill

``pip install`` does **not** install Cursor skills. The Agent skill lives at
``.cursor/skills/volightcurve/`` in this checkout (self-contained:
``SKILL.md`` plus ``references/``).

To use it in a **fresh host project** (or globally on this machine), copy or
symlink the whole skill directory:

```bash
# into a consumer project
mkdir -p /path/to/host-app/.cursor/skills
cp -a /path/to/volightcurve/.cursor/skills/volightcurve \
  /path/to/host-app/.cursor/skills/

# or for all local Cursor projects
mkdir -p ~/.cursor/skills
cp -a /path/to/volightcurve/.cursor/skills/volightcurve ~/.cursor/skills/
```

After editing ``docs/io_contract.md`` or ``examples/basic_workflow.py``,
re-copy those files into ``.cursor/skills/volightcurve/references/`` so the
bundled skill stays in sync.

## Quick start

See ``examples/basic_workflow.py``:

```bash
# with editable install, or PYTHONPATH=src
python examples/basic_workflow.py
```

## Public entry points

| Symbol | Role |
|--------|------|
| ``read_lightcurve`` | First file step (ingest any supported format) |
| ``write_lightcurve`` | Last file step (format chosen only here) |
| ``assemble_volightcurve`` | Build a product from a table + explicit calibration |
| ``VOLightCurve`` | In-memory table + TIMESYS + PhotDM map |
| ``PhotCal`` | PhotDM façade: ``mag_to_flux`` / ``flux_to_mag`` (+ err helpers) |
| ``PogsonZeroPoint`` | Nested zero-point implementing Pogson scale |
| ``AsinhZeroPoint`` / ``LinearFluxZeroPoint`` | Stubs for later scales |
| ``write_vo_lightcurve`` | Low-level VOTable writer (prefer ``write_lightcurve``) |

## Non-VOTable formats (``.dat``, CSV, ECSV)

Getting everyday lightcurve files to carry *enough* calibration metadata
without inventing yet another “standard” was awkward. People stick to
simple tables and comment lines.

What we ship for non-VOTable formats is a **practical compromise for real
use** — a shared keyword vocabulary and comment / flat-meta conventions so
uploads and downloads do not throw away photcal and time origin when the
user is not on VOTable. It is **not** a proposed IVOA (or any other)
standard, and should not be cited as one. Prefer VOTable when you need the
full annotation model; treat ASCII/ECSV as a best-effort carrier for the
same ideas.

Keyword and codec details: [docs/io_contract.md](docs/io_contract.md).

## I/O contract

Calibration keywords, column naming, and “never invent ZPs on write” rules:
[docs/io_contract.md](docs/io_contract.md).

## Tests

Pure package regression tests live under ``tests/``. See [TESTING.md](TESTING.md).

```bash
pytest -q
```

## Dependencies

- ``numpy``, ``astropy``, ``pyOpenSSL`` (required by GAVO at import)
- GAVO VOTable utilities (``gavoutils`` / ``gavostc`` / ``gavovot`` tarballs)

Optional for the example plots: ``matplotlib``.

## Layout

```text
volightcurve/
  pyproject.toml
  README.md
  TESTING.md
  docs/io_contract.md
  examples/basic_workflow.py
  .cursor/skills/volightcurve/   # Cursor Agent skill (+ references/)
  src/volightcurve/
    __init__.py
    lightcurve.py
    photdm.py          # ZeroPoint hierarchy (PhotDM-aligned)
    io.py
    …
  tests/
```
