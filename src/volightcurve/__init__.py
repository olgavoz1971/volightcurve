"""Virtual Observatory (VO) Lightcurve Processing Package.

This package provides a comprehensive data model and parser for lightcurves, supporting
standard VOTable files as well as heuristic ASCII text formats. It maps columns to
astronomical concepts like coordinate systems, time systems, and photometric calibrations.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("volightcurve")
except PackageNotFoundError:  # pragma: no cover - editable / source tree without metadata
    __version__ = "0.0.0+unknown"

from .io import (
    assemble_volightcurve,
    normalise_io_format,
    read_lightcurve,
    write_lightcurve,
)
from .io_dat import read_dat_table
from .io_errors import LightcurveIOError
from .io_keywords import (
    CALIBRATION_KEYWORDS,
    KEY_BAND,
    KEY_EPOCH,
    KEY_FILTER,
    KEY_FILTER_NAME,
    KEY_JD0,
    KEY_MAG0,
    KEY_MAG_SYS,
    KEY_PERIOD,
    KEY_ZP_FLUX,
    KEY_ZP_FLUX_UNIT,
    KEY_ZP_MAG,
    KEY_ZP_MAG_UNIT,
    WRITE_KEYWORD_ORDER,
    canonical_keyword,
    is_calibration_keyword,
    is_metadata_assignment_line,
    parse_keyword_assignment,
)
from .lightcurve import *
from .photdm import (
    AsinhZeroPoint,
    LinearFluxZeroPoint,
    MagnitudeSystem,
    PogsonZeroPoint,
    ZeroPoint,
)
from .time_reference import JD_TO_MJD, TIME_OFFSET_ABSOLUTE_JD_THRESHOLD
from .vo_unit_codec import (
    VO_DIMENSIONLESS_WIRE,
    to_display,
    to_internal,
    to_wire,
)

__all__ = [
    "__version__",
    "VOLightCurve",
    "PhotCal",
    "PhotDM",
    "PhotometryFilter",
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
    "print_col_ucd",
    "is_mag_column",
    "is_flux_column",
    "is_magnitude_phot_column",
    "assign_photometry_column_semantics",
    "apply_non_votable_heuristics",
    "VO_DIMENSIONLESS_WIRE",
    "to_internal",
    "to_wire",
    "to_display",
    "JD_TO_MJD",
    "TIME_OFFSET_ABSOLUTE_JD_THRESHOLD",
    "LightcurveIOError",
    "read_dat_table",
    "read_lightcurve",
    "write_lightcurve",
    "assemble_volightcurve",
    "normalise_io_format",
    "CALIBRATION_KEYWORDS",
    "KEY_BAND",
    "KEY_EPOCH",
    "KEY_FILTER",
    "KEY_FILTER_NAME",
    "KEY_JD0",
    "KEY_MAG0",
    "KEY_MAG_SYS",
    "KEY_PERIOD",
    "KEY_ZP_FLUX",
    "KEY_ZP_FLUX_UNIT",
    "KEY_ZP_MAG",
    "KEY_ZP_MAG_UNIT",
    "WRITE_KEYWORD_ORDER",
    "canonical_keyword",
    "is_calibration_keyword",
    "is_metadata_assignment_line",
    "parse_keyword_assignment",
]
