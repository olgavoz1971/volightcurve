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

## 4b. ``.dat`` column header selection and typing

These rules apply to the shared ``.dat`` / ``ascii.commented_header`` codec
(``read_dat_table``). Do **not** infer string vs float from cell content
(e.g. ``TR`` vs ``00``).

### Header candidates

1. Take every ``#`` comment that is **not** a ``KEY = value`` metadata
   assignment.
2. Split on whitespace. A line is a **header candidate** when its word count
   equals the number of fields in every data row.

### Name → role (by sense)

Classify each header **token** (case-insensitive) into exactly one role:

| Role | Sense (name contains / equals) |
|------|--------------------------------|
| ``error`` | ``err``, ``uncert``, ``sigma``, or the whole name is ``error`` (checked first so ``mag_err`` is error, not mag) |
| ``time`` | ``time``, ``jd``, or ``mjd`` |
| ``mag`` | ``mag`` or ``magnitude``, or the ambiguous name ``phot`` (defaults to magnitude domain) |
| ``flux`` | ``flux`` |
| ``other`` | anything else (e.g. ``Observ``, ``label``, ``sector``, ``flag``, ``site``) |

Substring checks use the lowered token (so ``Rmag`` → mag, ``JDhel2400000+`` →
time, ``flux_error`` → error).

### Choosing the winning header

1. A candidate is **role-canonical** when it has at least two words and
   **every word except the last** has role ``time``, ``mag``, ``flux``, or
   ``error`` (the last word may be any user label name).
2. If any role-canonical candidates exist, the winner is the **last** of those
   (file order).
3. Otherwise the winner is the **last** width-matching candidate.
4. If there is no width-matching candidate: legacy three-column files may use
   positional ``col1``…``col3`` (later promoted); other widths fail ingest.

### Structural check and failback

After the winning header is chosen, classify all column names with the same
role map. The layout is valid only when:

- exactly **one** ``time`` column;
- exactly **one** photometry column that is either ``mag`` **or** ``flux``
  (not both, not neither);
- at most **one** ``error`` column (zero or one).

If that check **fails**:

| Data width | Failback column names |
|------------|------------------------|
| 3 | ``jd``, ``mag``, ``mag_err`` |
| 4 | ``jd``, ``mag``, ``mag_err``, ``label`` |
| otherwise | hard-fail (no silent invent) |

### Cell typing

- Role ``other`` → free-text **string** column (including failback ``label``).
- All other roles → numeric (``float``; token ``nan`` allowed).
- There is **no** closed allowlist of free-text names (``label`` / ``sector`` /
  ``flag`` alone). Any ``other`` column is a string.

## 5. Photometry domain (non-VO)

**No ``DOMAIN=`` keyword.** Domain is carried by **column names** (and by
VOTable unit/UCD on the VO path).

| Domain | Time | Photometry | Error |
|--------|------|------------|-------|
| magnitude | ``jd`` | ``mag`` | ``mag_err`` |
| flux | ``jd`` | ``flux`` | ``flux_err`` |

Do not write ambiguous ``phot`` / ``flux_error`` for these formats.
When several columns share one role, see §8 (``mag-1``, ``mag-2``, …).

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

VOTable writing may change **time values** to MJD and record the origin on
TIMESYS (the same sense as ``JD0``). Field **names** and **UCDs** stay as on
the product. See §8.

## 7. Public API

```text
read_lightcurve(source, *, filename | format) -> VOLightCurve
write_lightcurve(volc, format, destination=None, ...) -> bytes
assemble_volightcurve(table, **calibration) -> VOLightCurve
```

Keyword constants: ``io_keywords.py``.

## 8. Export (describe the product; do not re-validate)

**UCD is the source of truth** for column roles. Ingest assigns and promotes
UCDs. Export does **not** add validation: it must not refuse a file because
several time, magnitude, flux, or error columns are present, because photcal
is incomplete, or because names are not a preferred spelling.

**PhotCal:** when ``VOLightCurve.photdms`` has an entry for a column, write
that calibration with **that** column. Do not drop extra science columns so
that only one photometry column remains.

### VOTable (UCD-capable)

Write the product table **as stored**:

- Keep column **names**.
- Write each column’s **UCD** and unit.
- Do **not** rename fields to ``obs_time`` / ``phot`` / ``flux_error``, and
  do **not** drop columns that are not in that short list.

**Time values:** the writer may convert the time column to MJD and set
TIMESYS ``@timeorigin`` (``JD0`` sense) so absolute JD is recoverable.
``EPOCH`` uses that same origin (§6). The field name and UCD are unchanged.

``write_vo_lightcurve`` must follow this section. The legacy behaviour that
renames columns and keeps only ``obs_time``, ``phot``, ``flux_error``, and
``label`` is **retired**.

### Non-VO (CSV, ``.dat``, ECSV)

These files do not carry per-column UCDs. Export the columns that exist.
Encode **main** roles in column names (from UCDs already on the product):

| Role | One column | Several columns (left to right) |
|------|------------|----------------------------------|
| time | ``jd`` | ``jd-1``, ``jd-2``, … |
| magnitude | ``mag`` | ``mag-1``, ``mag-2``, … |
| flux | ``flux`` | ``flux-1``, ``flux-2``, … |
| magnitude error | ``mag_err`` | ``mag_err-1``, ``mag_err-2``, … |
| flux error | ``flux_err`` | ``flux_err-1``, ``flux_err-2``, … |

Do not drop a column because another column has the same role. Do not invent
UCDs in the file. Other columns keep a stable non-science name (see labels).

### Per-epoch label column

Discovery (for hosts and for choosing which string column is the label).
Export still writes **every** column; this rule only **selects** the label.

1. If any column’s UCD is in the label-role list, take the **leftmost** such
   column. Initial list (extend later): ``meta.code``, ``meta.id``.
2. Otherwise (no matching UCD, or the format has no UCDs), take the
   **leftmost** column that is **not** time, flux, magnitude, or error.

Matching is a UCD **fragment** (same style as ``find_columns_by_ucd``).
Several label-like columns may exist; only the leftmost is the designated
label. The others are still exported.

**Ingest:** when that column has no UCD from the label-role list, store
``meta.id`` on it. Do not rename the column. An existing ``meta.code`` or
``meta.id`` UCD is kept.
