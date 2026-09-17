# Testing volightcurve

Pure package tests live under ``tests/`` in this repository. Prefer tests that
import **only** ``volightcurve`` (and Astropy / NumPy / GAVO).

## Pure package tests

| File | Covers |
|------|--------|
| ``test_io_roundtrip.py`` | ``read_lightcurve`` / ``write_lightcurve`` |
| ``test_io_keywords.py`` | Keyword vocabulary |
| ``test_vo_unit_codec.py`` | Dimensionless unit wire format |
| ``test_write_vo.py`` | Low-level VOTable writer |
| ``test_photdm_ingest.py`` | PhotDM GROUP ingest |
| ``test_photcal_conversion.py`` | ``PhotCal`` / PogsonZeroPoint mag↔flux |
| ``test_time_reference.py`` | TIMESYS helpers and VOLightCurve ingest |
| ``fixtures_votable.py`` | Shared Gaia-style VOTable bytes |

Host application / CurveDash / bridge / mission tests remain in ``skvo_veb_2``
(``skvo_veb/tests/volightcurve/``) and are **not** part of this suite.

## Run pure suite

From this project root (editable install or ``pythonpath`` via ``pyproject.toml``):

```bash
pytest -q
```
