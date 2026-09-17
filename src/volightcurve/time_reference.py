"""TIMESYS resolution and time-origin conversions for VO lightcurves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

# Julian Date offset for Modified Julian Date: MJD = JD - JD_TO_MJD.
# Owned here so this package has no dependency on a host application layer.
JD_TO_MJD = 2400000.5

# Values at or above this threshold are treated as absolute Julian Date on ingest.
TIME_OFFSET_ABSOLUTE_JD_THRESHOLD = JD_TO_MJD

if TYPE_CHECKING:
    from volightcurve.lightcurve import TimeSys, VOLightCurve


@dataclass(frozen=True)
class TimesysMetadata:
    """TIMESYS metadata extracted from a GAVO-parsed VOTable tree.

    Attributes:
        registry (dict[str, TimeSys]): ``TIMESYS/@ID`` to parsed metadata.
        field_refs (dict[str, str]): TABLE ``FIELD/@name`` to ``TIMESYS/@ID``.
        param_refs (dict[str, str | None]): TABLE ``PARAM/@name`` to optional ref.
        default_timesys (TimeSys): First ``TIMESYS`` element in document order.
    """

    registry: dict[str, "TimeSys"]
    field_refs: dict[str, str]
    param_refs: dict[str, str | None]
    default_timesys: "TimeSys"


def _timesys_from_gavo_attrs(attrs: dict) -> "TimeSys":
    """Builds a ``TimeSys`` instance from GAVO ``TIMESYS`` element attributes.

    Args:
        attrs (dict): Attribute mapping from a GAVO stanxml node.

    Returns:
        TimeSys: Parsed time-system metadata.
    """
    from volightcurve.lightcurve import TimeSys

    try:
        timeorigin = float(attrs.get("timeorigin", 0.0) or 0.0)
    except (TypeError, ValueError):
        timeorigin = 0.0
    return TimeSys(
        refposition=attrs.get("refposition") or "HELIOCENTER",
        timeorigin=timeorigin,
        timescale=attrs.get("timescale") or "UTC",
    )


def extract_timesys_registry_from_gavo(tree) -> dict[str, "TimeSys"]:
    """Builds a ``TIMESYS/@ID`` registry from a GAVO VOTable tree.

    Args:
        tree: GAVO stanxml root returned by ``votparse.readRaw``.

    Returns:
        dict[str, TimeSys]: Mapping of TIMESYS identifier to parsed metadata.
    """
    registry: dict[str, TimeSys] = {}

    def find_ts(node, text, attrs, childIter):
        if node.name_ == "TIMESYS":
            ts_id = attrs.get("ID") or attrs.get("id") or "ts"
            registry[str(ts_id)] = _timesys_from_gavo_attrs(attrs)
        for child in childIter:
            if hasattr(child, "apply"):
                child.apply(find_ts)

    tree.apply(find_ts)
    return registry


def extract_field_timesys_refs_from_gavo(tree) -> dict[str, str]:
    """Maps TABLE field names to their ``TIMESYS`` reference identifiers.

    Only direct ``FIELD`` children of ``TABLE`` are considered so photcal GROUP
    parameters are excluded.

    Args:
        tree: GAVO stanxml root returned by ``votparse.readRaw``.

    Returns:
        dict[str, str]: Field name to TIMESYS ``ID`` (from ``FIELD/@ref``).
    """
    refs: dict[str, str] = {}

    def walk(node, text, attrs, childIter):
        if node.name_ == "TABLE":
            for child in childIter:
                if child.name_ != "FIELD":
                    continue
                name = getattr(child, "name", None)
                ref = getattr(child, "ref", None)
                if name and ref:
                    refs[str(name)] = str(ref)
            return
        for child in childIter:
            if hasattr(child, "apply"):
                child.apply(walk)

    tree.apply(walk)
    return refs


def extract_param_timesys_refs_from_gavo(tree) -> dict[str, str | None]:
    """Maps TABLE PARAM names to optional ``TIMESYS`` reference identifiers.

    Only direct ``PARAM`` children of ``TABLE`` are considered.

    Args:
        tree: GAVO stanxml root returned by ``votparse.readRaw``.

    Returns:
        dict[str, str or None]: Param name to TIMESYS ``ID`` when ``PARAM/@ref`` is set.
    """
    refs: dict[str, str | None] = {}

    def walk(node, text, attrs, childIter):
        if node.name_ == "TABLE":
            for child in childIter:
                if child.name_ != "PARAM":
                    continue
                name = getattr(child, "name", None)
                if not name:
                    continue
                ref = getattr(child, "ref", None)
                refs[str(name)] = str(ref) if ref else None
            return
        for child in childIter:
            if hasattr(child, "apply"):
                child.apply(walk)

    tree.apply(walk)
    return refs


def extract_timesys_metadata_from_gavo(tree) -> TimesysMetadata:
    """Extracts TIMESYS registry and TABLE cross-references from a GAVO tree.

    Args:
        tree: GAVO stanxml root returned by ``votparse.readRaw``.

    Returns:
        TimesysMetadata: Registry, field/param refs, and the first TIMESYS block.
    """
    from volightcurve.lightcurve import TimeSys

    registry = extract_timesys_registry_from_gavo(tree)
    field_refs = extract_field_timesys_refs_from_gavo(tree)
    param_refs = extract_param_timesys_refs_from_gavo(tree)
    if registry:
        default_timesys = next(iter(registry.values()))
    else:
        default_timesys = TimeSys()
    return TimesysMetadata(
        registry=registry,
        field_refs=field_refs,
        param_refs=param_refs,
        default_timesys=default_timesys,
    )


def resolve_observation_timesys(volc: VOLightCurve) -> "TimeSys":
    """Resolves the TIMESYS shared by observation-time columns (``ucd=time.epoch``).

    When a single time column references a TIMESYS, that system is returned. When
    several time columns are present they must all reference the same ``timeorigin``;
    otherwise an error is raised.

    Args:
        volc (VOLightCurve): Parsed lightcurve product.

    Returns:
        TimeSys: Shared time system for observation epochs.

    Raises:
        ValueError: When time columns disagree on ``timeorigin`` or none are found.
    """
    from volightcurve.lightcurve import TimeSys, get_time_colnames

    time_cols = get_time_colnames(volc.table)
    if not time_cols:
        for fallback in ("obs_time", "time", "jd", "mjd"):
            if fallback in volc.table.colnames:
                time_cols = [fallback]
                break
    if not time_cols:
        raise ValueError("No observation time column found for TIMESYS resolution.")

    registry = getattr(volc, "timesys_by_id", None) or {}
    field_refs = getattr(volc, "field_timesys_ref", None) or {}
    default_ts = volc.timesys or TimeSys()
    resolved: list[TimeSys] = []

    for col in time_cols:
        ref_id = field_refs.get(col)
        if ref_id and ref_id in registry:
            resolved.append(registry[ref_id])
        elif ref_id and ref_id not in registry:
            raise ValueError(
                f"Time column {col!r} references unknown TIMESYS {ref_id!r}."
            )
        else:
            resolved.append(default_ts)

    origins = {float(ts.timeorigin or 0.0) for ts in resolved}
    if len(origins) > 1:
        raise ValueError(
            "Multiple time columns reference different TIMESYS timeorigin values."
        )
    return resolved[0]


def resolve_timesys_for_table_param(volc: VOLightCurve, param_name: str) -> "TimeSys":
    """Resolves TIMESYS metadata for a TABLE PARAM holding a time epoch.

    Uses an explicit ``PARAM/@ref`` when present; otherwise assumes the same TIMESYS
    as the sole (or uniquely consistent) observation-time column.

    Args:
        volc (VOLightCurve): Parsed lightcurve product.
        param_name (str): TABLE PARAM name (e.g. ``epoch``).

    Returns:
        TimeSys: Time system governing the parameter value.

    Raises:
        ValueError: When references are missing or ambiguous.
    """
    param_refs = getattr(volc, "param_timesys_ref", None) or {}
    registry = getattr(volc, "timesys_by_id", None) or {}
    ref_id = param_refs.get(param_name)
    if ref_id:
        if ref_id not in registry:
            raise ValueError(
                f"PARAM {param_name!r} references unknown TIMESYS {ref_id!r}."
            )
        return registry[ref_id]
    return resolve_observation_timesys(volc)


def time_offset_to_absolute_jd(raw_value: float, timeorigin: float) -> float:
    """Converts a VOTable time offset to absolute Julian Date.

    Values above ``TIME_OFFSET_ABSOLUTE_JD_THRESHOLD`` are treated as already
    absolute Julian dates (legacy catalogues sometimes store full JD).

    Args:
        raw_value (float): Stored PARAM or column value.
        timeorigin (float): ``TIMESYS/@timeorigin`` for the linked system.

    Returns:
        float: Absolute Julian Date.
    """
    value = float(raw_value)
    if value >= TIME_OFFSET_ABSOLUTE_JD_THRESHOLD:
        return value
    return value + float(timeorigin or 0.0)


def absolute_jd_to_time_offset(absolute_jd: float, timeorigin: float) -> float:
    """Converts absolute Julian Date to a VOTable time offset for a TIMESYS.

    Args:
        absolute_jd (float): Absolute Julian Date in the host session.
        timeorigin (float): Target ``TIMESYS/@timeorigin`` for export.

    Returns:
        float: Offset value such that ``absolute_jd = offset + timeorigin``.
    """
    return float(absolute_jd) - float(timeorigin or 0.0)


def normalise_table_epoch_to_absolute_jd(volc: VOLightCurve, raw_epoch) -> float | None:
    """Normalises a folding ``epoch`` TABLE PARAM to absolute Julian Date.

    Args:
        volc (VOLightCurve): Parsed lightcurve with TIMESYS cross-reference metadata.
        raw_epoch: Raw ``epoch`` PARAM value from ``table.meta``.

    Returns:
        float or None: Absolute Julian Date for folding, or ``None`` when
            ``raw_epoch`` is absent.
    """
    if raw_epoch is None:
        return None
    timesys = resolve_timesys_for_table_param(volc, "epoch")
    return time_offset_to_absolute_jd(float(raw_epoch), timesys.timeorigin or 0.0)


def export_absolute_jd_as_time_offset(
    absolute_jd: float | None,
    *,
    timeorigin: float = JD_TO_MJD,
) -> float | None:
    """Serialises an absolute Julian Date using a target ``TIMESYS/@timeorigin``.

    Args:
        absolute_jd (float, optional): Absolute Julian Date from session metadata.
        timeorigin (float): Export ``TIMESYS/@timeorigin`` (default MJD standard).

    Returns:
        float or None: Offset suitable for ``PARAM name=\"epoch\"`` alongside
            ``obs_time`` written under the same TIMESYS.
    """
    if absolute_jd is None:
        return None
    return absolute_jd_to_time_offset(float(absolute_jd), timeorigin)
