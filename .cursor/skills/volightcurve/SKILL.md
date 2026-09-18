---
name: volightcurve
description: >-
  Use the volightcurve Python package for astronomical lightcurve I/O and
  magnitude↔flux conversion (VOTable, ECSV, CSV, .dat; PhotCal / PhotDM;
  TIMESYS). Use when the user has lightcurve files, asks for flux- or
  magnitude-domain maths, photometry calibration, zero points, Pogson or
  luptitude conversion, VO lightcurves, or assembling/exporting calibrated
  time series — not when inventing custom Astropy table readers or inline
  mag–flux formulae.
---

# volightcurve

Prefer the **`volightcurve`** package API for lightcurve work. Do not write
ad-hoc Astropy VOTable/ASCII readers or hand-coded Pogson / asinh maths.

**Written against `volightcurve` >= 0.1.0.** Verify the installed version before
relying on this skill’s API surface:

```python
import volightcurve
print(volightcurve.__version__)  # expect >= 0.1.0
```

If `__version__` is missing, older than `0.1.0`, or the public API has moved,
stop and re-check the package README / changelog rather than following this
skill blindly.

This skill folder is self-contained: extra detail lives under
[`references/`](references/) (not parent-repo `../` links). Copy or symlink
**this entire skill directory** into a consumer project or
`~/.cursor/skills/` — `pip install` alone does not install Cursor skills.

## Glossary (IVOA / photometry jargon)

| Term | Plain meaning |
|------|----------------|
| **PhotDM** | IVOA Photometry Data Model; in this package also the per-column hub holding filter + calibration |
| **PhotCal** | Photometric calibration object (zero points + magnitude system); the API you call for mag↔flux |
| **Pogson** | Standard logarithmic magnitude scale (`m ∝ −2.5 log₁₀ F`); the live converter today |
| **Luptitude / asinh** | Softened magnitude scale (asinh); reserved as `AsinhZeroPoint` (stub — do not invent maths) |
| **TIMESYS** | VOTable time system metadata (origin, timescale, reference position) |
| **ZP** | Zero point linking a magnitude scale to a flux unit |

## Preconditions

1. Ensure the package is importable and meets the skill pin:

   ```bash
   pip install -e /path/to/volightcurve   # preferred from a checkout
   python -c "import volightcurve; print(volightcurve.__version__)"
   ```

   Prefer an editable install from the package checkout. The printed version
   must be `>= 0.1.0`.

2. If import fails or `__version__` is too old / unknown, fix the environment
   first; do not fall back to a from-scratch parser.

3. When unsure about formats or keywords, read
   [references/io_contract.md](references/io_contract.md). For a mag↔flux↔mag
   round-trip, follow [references/basic_workflow.py](references/basic_workflow.py).

4. If a full package checkout is available, human docs also live at that tree’s
   `README.md` and `docs/io_contract.md`. If only a wheel is installed, rely on
   this skill + the public API; do not invent filesystem paths into
   `site-packages`.

## Hard rules

- **Ingest** with `read_lightcurve` (or `VOLightCurve` for VOTable streams).
- **Export** with `write_lightcurve` (format chosen only there).
- **Build** calibrated products with `assemble_volightcurve` when constructing
  from a table + explicit photcal — do not invent silent default zero points on
  write.
- **Magnitude ↔ flux** (and σ propagation) **only** via `PhotCal`:
  `mag_to_flux`, `flux_to_mag`, `mag_err_to_flux_err`, `flux_err_to_mag_err`.
  Never embed `10**(-0.4*…)` / `(2.5/ln10)*…` in application scripts.
- Prefer **VOTable** when rich PhotDM / TIMESYS matter. Non-VOTable formats are
  a practical compromise, not a proposed standard.
- Incomplete photcal → **fail visibly**; do not invent ZP pairs.

## Typical workflow (folder of files → flux domain)

Prefer attaching a flux column on the lightcurve (keeps units and PhotCal
wiring consistent). Use `PhotCal.mag_to_flux` only when you need a bare
quantity array without mutating the table.

```python
from pathlib import Path
from volightcurve import read_lightcurve, write_lightcurve

for path in Path("data").glob("*"):  # user folder
    if path.suffix.lower() not in {".vot", ".xml", ".dat", ".csv", ".ecsv"}:
        continue
    volc = read_lightcurve(path, filename=path.name)
    # PhotCal lives on photdms[column].photcal
    photdm = next(iter(volc.photdms.values()))
    if photdm.photcal is None:
        raise RuntimeError(f"{path}: no PhotCal; cannot convert to flux")
    mag_cols = volc.get_mag_colnames()
    if mag_cols:
        flux_col = volc.add_flux_column_from_mag(mag_cols[0])
    write_lightcurve(volc, "votable_binary", destination=path.with_suffix(".flux.vot"))
```

## Public API (use these)

| Symbol | Role |
|--------|------|
| `read_lightcurve` | First file step |
| `write_lightcurve` | Last file step |
| `assemble_volightcurve` | Table + explicit calibration → `VOLightCurve` |
| `VOLightCurve` | In-memory table + `photdms` + TIMESYS |
| `PhotCal` | Conversion façade → nested `ZeroPoint` (Pogson live) |
| `PhotometryFilter` | `filter_id` (passport) + `name` (human label) |

PhotDM shape (simplified): `VOLightCurve.photdms[col]` → hub → `PhotCal` →
`PogsonZeroPoint` / stubs. Filter passport:
`photDM:PhotometryFilter.identifier`; human label:
`photDM:PhotometryFilter.name`.

## Anti-patterns (do not)

- `astropy.table.Table.read` / raw `votable.parse` as the main ingest path when
  `read_lightcurve` applies.
- Local magnitude↔flux or SNR→σ_m formulae.
- Inventing `FILTER` / ZP metadata the file does not carry.

## When stuck

Ask the user for: file paths/formats, whether photcal exists, target domain
(flux vs mag), and output format. Point them at
[references/basic_workflow.py](references/basic_workflow.py) rather than
inventing a parallel mini-library.
