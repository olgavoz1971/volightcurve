"""Virtual Observatory (VO) Lightcurve Processing Package.

This package provides a comprehensive data model and parser for lightcurves, supporting
standard VOTable files as well as heuristic ASCII text formats. It maps columns to
astronomical concepts like coordinate systems, time systems, and photometric calibrations.

Start here. Most tasks need only these names:

* ``read_lightcurve(path, filename=...)``: ingest a file into a ``VOLightCurve``.
* ``write_lightcurve(volc, format, destination=...)``: save it (format chosen only there).
* ``VOLightCurve``: ``table`` + ``timesys`` + ``photdms``. Its methods do the work:
  ``get_time_colnames`` / ``get_mag_colnames`` / ``get_flux_colnames`` (find columns by UCD),
  ``get_observation_times`` / ``get_observation_jd`` / ``get_observation_mjd`` (absolute
  times: the declared ``timeorigin`` is added; the source of the data never decides a
  shift; ``get_epoch_time`` / ``get_epoch_jd`` / ``get_epoch_mjd`` do the same for the folding
  epoch), ``time_problems``, ``check_units``, ``conversion_problems``,
  ``add_flux_column_from_mag`` / ``add_mag_column_from_flux``.
* ``PhotCal``, ``PhotDM``, ``PhotometryFilter``: the photometric calibration, and the
  magnitude and flux conversions (``mag_to_flux``, ``flux_to_mag`` and the error helpers).
* ``assemble_volightcurve``: build a product from a table and an explicit calibration.
* ``PhotCalError``, ``LightcurveIOError``: what is raised when something cannot be done.

Everything else exported here (zero-point classes, unit text helpers, suggestions,
heuristics, low-level writers) is for advanced use. Keyword constants and the unit wire
helpers live in their submodules (``volightcurve.io_keywords``, ``volightcurve.vo_unit_codec``).
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("volightcurve")
except PackageNotFoundError:  # pragma: no cover - editable / source tree without metadata
    __version__ = "0.0.0+unknown"

from .io import (
    assemble_volightcurve,
    read_lightcurve,
    write_lightcurve,
)
from .io_dat import read_dat_table
from .io_errors import LightcurveIOError
from .photcal_check import PhotCalError, inspect_conversion_photcal
from .columns import (
    DEFAULT_LABEL_UCD,
    assign_photometry_column_semantics,
    column_main_role,
    find_columns_by_ucd,
    get_error_colnames,
    get_flux_colnames,
    get_label_colnames,
    get_mag_colnames,
    get_time_colnames,
    is_flux_column,
    is_mag_column,
)
from .lightcurve import VOLightCurve
from .model import CooSys, PhotCal, PhotDM, PhotometryFilter, TimeSys
from .votable_reader import extract_coosys, extract_photdm, extract_timesys
from .votable_writer import write_vo_lightcurve
from .interpretation import (
    apply_non_votable_heuristics,
    assign_label_column_ucd,
    is_magnitude_phot_column,
)
from .suggestions import (
    SUGGESTION_REASONS,
    suggested_calibration,
    suggested_time_system,
    suggestions_for_missing,
)
from .photdm import (
    AsinhZeroPoint,
    LinearFluxZeroPoint,
    MagnitudeSystem,
    PogsonZeroPoint,
    ZeroPoint,
)
from .time_reference import JD_TO_MJD, TIME_OFFSET_ABSOLUTE_JD_THRESHOLD
from .vo_unit_codec import (
    UnitReport,
    inspect_unit_string,
    to_display,
    to_internal,
)

__all__ = [
    "__version__",
    "VOLightCurve",
    "PhotCal",
    "PhotCalError",
    "inspect_conversion_photcal",
    "PhotDM",
    "PhotometryFilter",
    "CooSys",
    "TimeSys",
    "extract_photdm",
    "extract_timesys",
    "extract_coosys",
    "column_main_role",
    "assign_label_column_ucd",
    "DEFAULT_LABEL_UCD",
    "ZeroPoint",
    "PogsonZeroPoint",
    "AsinhZeroPoint",
    "LinearFluxZeroPoint",
    "MagnitudeSystem",
    "write_vo_lightcurve",
    "find_columns_by_ucd",
    "get_time_colnames",
    "get_mag_colnames",
    "get_flux_colnames",
    "get_error_colnames",
    "get_label_colnames",
    "is_mag_column",
    "is_flux_column",
    "is_magnitude_phot_column",
    "assign_photometry_column_semantics",
    "apply_non_votable_heuristics",
    "SUGGESTION_REASONS",
    "suggested_calibration",
    "suggested_time_system",
    "suggestions_for_missing",
    "UnitReport",
    "inspect_unit_string",
    "to_internal",
    "to_display",
    "JD_TO_MJD",
    "TIME_OFFSET_ABSOLUTE_JD_THRESHOLD",
    "LightcurveIOError",
    "read_dat_table",
    "read_lightcurve",
    "write_lightcurve",
    "assemble_volightcurve",
]
