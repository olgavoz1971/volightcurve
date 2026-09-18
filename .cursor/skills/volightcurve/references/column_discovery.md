# Column discovery

After ``read_lightcurve``, **do not guess column names**. Roles come from
UCDs (VOTable) or from the non-VO naming conventions in ``io_contract.md``.

## Prefer these APIs

On a ``VOLightCurve`` instance:

| Method | Finds |
|--------|--------|
| ``get_time_colnames()`` | Time (``time.epoch``) |
| ``get_mag_colnames()`` | Primary magnitudes (``phot.mag``, errors excluded) |
| ``get_flux_colnames()`` | Primary fluxes (``phot.flux``, errors excluded) |
| ``get_mag_error_colnames()`` | ``stat.error`` + ``phot.mag`` |
| ``get_flux_error_colnames()`` | ``stat.error`` + ``phot.flux`` |

Module-level equivalents: ``find_columns_by_ucd``, ``get_time_colnames``,
``get_mag_colnames``, ``get_flux_colnames``, ``get_error_colnames(table,
base_ucd=…)``.

```python
volc = read_lightcurve(path, filename=path.name)
flux_cols = volc.get_flux_colnames()       # e.g. ["phot"]
err_cols = volc.get_flux_error_colnames()  # e.g. ["flux_error"]
if not flux_cols:
    raise RuntimeError("no flux column (UCD phot.flux)")
if len(err_cols) != 1:
    raise RuntimeError(f"ambiguous flux errors: {err_cols}")
phot_col, err_col = flux_cols[0], err_cols[0]
```

## Why names mislead

A column named ``phot`` may be flux or magnitude. A column named
``flux_error`` may be the uncertainty of either, depending on its UCD.
Always check discovery results (and units) before converting or plotting.

## Known limitation

There is no IVOA-standard 1:1 FIELD link from an error column to its parent
photometry column when several bands coexist. This package returns
**domain-filtered lists**, not a paired map. Single-band files are fine;
multi-band ambiguity → fail or ask the user — do not invent pairing.
