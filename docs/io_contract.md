# volightcurve file I/O contract

**PhotCal / PhotDM:** Magnitude↔flux conversion uses the nested PhotDM shape
(``PhotCal`` façade → ``ZeroPoint`` subtype). Application code must call
``PhotCal.mag_to_flux`` / ``flux_to_mag`` (and error helpers), never local
Pogson/asinh formulae. See package README and
``src/volightcurve/photdm.py``.


Canonical rules for ``read_lightcurve`` / ``write_lightcurve``. Do not invent
alternate formats or nested application schemas.

## 1. Goals

1. Parse and serialise lightcurve products in this package only (no UI types).
2. Preserve photometric and time calibration metadata the product already holds.
3. Do **not** invent private file formats. Non-VO carriers reuse one plain
   keyword vocabulary; VOTable keeps IVOA PhotDM / TIMESYS.

## 2. Formats in scope

| Format id | Metadata carrier |
|-----------|------------------|
| ``votable_binary`` / ``votable_text`` / ``votable`` | TIMESYS, PhotDM GROUP, PARAMs |
| ``ascii.ecsv`` | Flat keys in standard ECSV ``meta:`` YAML |
| ``ascii.commented_header`` (``.dat``) | ``# KEY = value`` comments |
| ``csv`` | ``# KEY = value`` preamble (same vocabulary) |

``ascii.commented_header`` and ``.dat`` share one codec.

## 3. First-class vs description

**First-class** (must round-trip when present): photometric calibration and
time calibration (including fold period / epoch that share the time origin).

**Not first-class on non-VO formats:** timescale, refposition, COOSYS,
facility, instrument, publication id, rich VO descriptions as structured fields.

**Free text:** other ``#`` comment lines are stored as narrative comments and
re-emitted on download. They must not be parsed as calibration keywords.

## 4. Shared keyword vocabulary (non-VO)

Case-insensitive on read. Write using the canonical uppercase names below.
Emit a key **only when the value is present** — never invent zero points or
origins on write.

| Keyword | Meaning |
|---------|---------|
| ``JD0`` | Time origin added to the time column to obtain absolute JD |
| ``EPOCH`` | Reference epoch in the **same units and origin as the time column** |
| ``PERIOD`` | Folding period in days |
| ``MAG0`` | Legacy compact reference magnitude (read alias of ``ZP_MAG``) |
| ``ZP_MAG`` / ``ZP_MAG_UNIT`` | Zero-point magnitude |
| ``ZP_FLUX`` / ``ZP_FLUX_UNIT`` | Zero-point flux (empty / omitted unit ⇒ dimensionless) |
| ``MAG_SYS`` | Magnitude system (e.g. ``Vega``, ``AB``) |
| ``FILTER`` / ``BAND`` | Filter identifier (``BAND`` is a read alias) |
| ``FILTER_NAME`` | Human-readable filter name (optional) |
| ``EFFECTIVE_WAVELENGTH`` / ``EFFECTIVE_WAVELENGTH_UNIT`` | Optional passband location |

### Write preference

When rich photcal is present, write the full ZP set plus ``FILTER`` and optional
filter extras. Legacy files with only ``MAG0`` may still be read; incomplete
products must not gain invented ZPs on write.

### ECSV

Use **flat** ``meta:`` entries with the same keyword names. Do not nest a
private ``photcal:`` mapping.

### CSV / ``.dat`` shape

```text
# JD0 = 0
# ZP_MAG = 20.0
# ZP_FLUX = 1.0
# ZP_FLUX_UNIT =
# FILTER = TESS/TESS.Red
# PERIOD = 1.23
# EPOCH = 2459000.25
# optional free-text description lines
# jd mag mag_err
2459000.0 12.1 0.01
```

Empty ``ZP_FLUX_UNIT`` means dimensionless on the wire (see ``vo_unit_codec.py``).

## 5. Photometry domain (non-VO)

**No ``DOMAIN=`` keyword.** Domain is carried by **column names** (and by
VOTable unit/UCD on the VO path).

| Domain | Time | Photometry | Error |
|--------|------|------------|-------|
| magnitude | ``jd`` | ``mag`` | ``mag_err`` |
| flux | ``jd`` | ``flux`` | ``flux_err`` |

Do not write ambiguous ``phot`` / ``flux_error`` for these formats.

### Read

1. Explicit ``mag`` / ``mag_err`` → magnitude.
2. Explicit ``flux`` / ``flux_err`` → flux.
3. Ambiguous cases → default **magnitude**.
4. Never infer domain from ``ZP_*`` alone.

## 6. Time-origin contract

1. The time column is relative to an explicit origin (``JD0`` / TIMESYS
   ``@timeorigin``).
2. ``EPOCH`` must use that same origin as the time column in the file.
3. Missing ``JD0`` on legacy ``.dat`` ingest defaults to ``0`` (documented
   heuristic). Writers must not invent a non-zero origin.

VOTable writing may serialise ``obs_time`` as MJD with
``timeorigin = 2400000.5`` provided epoch PARAMs use that same origin. That
remap happens only inside ``write_lightcurve``.

## 7. Public API

```text
read_lightcurve(source, *, filename | format) -> VOLightCurve
write_lightcurve(volc, format, destination=None, ...) -> bytes
assemble_volightcurve(table, **calibration) -> VOLightCurve
```

Keyword constants: ``io_keywords.py``.
