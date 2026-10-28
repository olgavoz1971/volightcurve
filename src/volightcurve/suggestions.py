"""Table of SUGGESTED values for data that declare nothing (plain data, never applied here).

The library never writes these values on its own. A host UI or an agent shows them, with
the reason, and the user applies, edits or ignores them through an explicit action. They
replace the old ``photcal_defaults`` constants (module removed).

The values (agreed in Ticket 32): for observers' files with calibrated magnitudes,
Vega magnitudes and a relative flux without a physical unit (a magnitude-0 star has flux
1), and for hand-made legacy data a heliocentric UTC time with origin 0.
"""

from __future__ import annotations

# Why each suggestion exists, for display next to the value.
SUGGESTION_REASONS: dict[str, str] = {
    "refposition": "Hand-made legacy data are usually heliocentric.",
    "timescale": "Hand-made legacy data are usually in UTC.",
    "timeorigin": "Legacy times are usually full Julian Dates, so the origin is 0.",
    "magnitude_system": "Observers' magnitudes are usually on the Vega scale.",
    "zp_mag": "Reference magnitude 0: a magnitude-0 star has relative flux 1.",
    "zp_mag_unit": "Magnitudes are in mag.",
    "zp_flux": (
        "Relative flux: a magnitude-0 star has flux 1, comparable only within this "
        "light curve; no physical calibration is claimed."
    ),
    "zp_flux_unit": "Relative flux has no physical unit (dimensionless, written as 1).",
}

_TIME_SYSTEM = {"refposition": "HELIOCENTER", "timescale": "UTC", "timeorigin": 0.0}
_CALIBRATION = {
    "magnitude_system": "Vega",
    "zp_mag": 0.0,
    "zp_mag_unit": "mag",
    "zp_flux": 1.0,
    "zp_flux_unit": None,
}


def suggested_time_system() -> dict:
    """Returns the suggested time system for legacy data (a new dict each call).

    Returns:
        dict: ``refposition``, ``timescale`` and ``timeorigin`` with their suggested values.
    """
    return dict(_TIME_SYSTEM)


def suggested_calibration() -> dict:
    """Returns the suggested calibration for observers' files (a new dict each call).

    ``zp_flux_unit`` is ``None``, which means dimensionless (relative flux).

    Returns:
        dict: ``magnitude_system``, ``zp_mag``, ``zp_mag_unit``, ``zp_flux`` and
            ``zp_flux_unit`` with their suggested values.
    """
    return dict(_CALIBRATION)


def suggestions_for_missing(volc) -> dict:
    """Lists the suggestions for exactly the fields a curve leaves absent. Changes nothing.

    Args:
        volc (VOLightCurve): The curve to look at.

    Returns:
        dict: ``time_system`` (suggested values for the absent ``TimeSys`` fields) and
            ``calibration`` (column name to the suggested values for the absent
            calibration fields of each flux and magnitude column). Empty parts are omitted.
    """
    result: dict = {}
    timesys = volc.timesys
    time_missing = {
        key: value
        for key, value in _TIME_SYSTEM.items()
        if getattr(timesys, key) is None
    }
    if time_missing:
        result["time_system"] = time_missing
    calibration: dict = {}
    for column in volc.get_flux_colnames() + volc.get_mag_colnames():
        photdm = volc.photdms.get(column)
        photcal = photdm.photcal if photdm is not None else None
        current = {
            "magnitude_system": None if photcal is None else photcal.mag_sys,
            "zp_mag": None if photcal is None else photcal.zp_mag,
            "zp_flux": None if photcal is None else photcal.zp_flux,
        }
        absent = {key: _CALIBRATION[key] for key, value in current.items() if value is None}
        if "zp_mag" in absent:
            absent["zp_mag_unit"] = _CALIBRATION["zp_mag_unit"]
        if "zp_flux" in absent:
            absent["zp_flux_unit"] = _CALIBRATION["zp_flux_unit"]
        if absent:
            calibration[column] = absent
    if calibration:
        result["calibration"] = calibration
    return result
