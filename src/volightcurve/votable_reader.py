"""Readers of declared VOTable metadata (GAVO tree): PhotDM, TIMESYS, COOSYS, INFO."""

import io
import logging

import numpy as np

from volightcurve.vo_unit_codec import (
    to_internal,
)

from volightcurve.model import CooSys, PhotCal, PhotDM, PhotometryFilter

logger = logging.getLogger(__name__)


def _apply_gavo_votable_metadata(volc_instance, gavo_tree) -> None:
    """Populates TIMESYS, COOSYS, and PhotDM metadata from a GAVO VOTable tree.

    Args:
        volc_instance (VOLightCurve): Target instance to mutate in place.
        gavo_tree: GAVO stanxml root from ``_gavo_votable_metadata_tree``.
    """
    from volightcurve.time_reference import extract_timesys_metadata_from_gavo

    ts_meta = extract_timesys_metadata_from_gavo(gavo_tree)
    volc_instance.timesys_by_id = ts_meta.registry
    volc_instance.field_timesys_ref = ts_meta.field_refs
    volc_instance.param_timesys_ref = ts_meta.param_refs
    volc_instance.timesys = ts_meta.default_timesys
    volc_instance.coosys = extract_coosys(gavo_tree)
    volc_instance.photdms = extract_photdm(gavo_tree)


def _normalize_photdm_column_key(name: str) -> str:
    """Normalises a photometry column name for ``photdms`` lookup.

    Args:
        name (str): VOTable ``FIELD/@name`` or ``FIELDref/@ref`` value.

    Returns:
        str: Lowercase column key.
    """
    return str(name).strip().lower()


def _link_photdm_field_refs(tree, photcal_groups: dict[str, PhotDM], dm_map: dict[str, PhotDM]) -> None:
    """Links ``photcal`` GROUP instances to TABLE fields via ``FIELD/@ref``.

    Gaia and other archives attach ``FIELD/@ref`` to a ``photcal`` GROUP ``@ID``
    instead of embedding ``FIELDref`` inside the GROUP (UPJS/DaCHS style).

    Args:
        tree: GAVO stanxml root from ``votparse.parse``.
        photcal_groups (dict): ``photcal`` GROUP ``ID`` to ``PhotDM`` instances.
        dm_map (dict): Target mapping mutated in place (column name → ``PhotDM``).
    """

    def walk(node, text, attrs, childIter):
        if node.name_ == "TABLE":
            for child in childIter:
                if child.name_ != "FIELD":
                    continue
                field_name = getattr(child, "name", None)
                group_ref = getattr(child, "ref", None)
                if not field_name or not group_ref:
                    continue
                photdm = photcal_groups.get(str(group_ref))
                if photdm is None:
                    continue
                key = _normalize_photdm_column_key(field_name)
                dm_map.setdefault(key, photdm)
            return
        for child in childIter:
            if hasattr(child, "apply"):
                child.apply(walk)

    tree.apply(walk)


def _declared_float(value) -> float | None:
    """Reads a numeric PARAM value, treating the absent-value wire forms as ``None``.

    Files of earlier versions wrote ``NaN`` (double) or an empty string for an absent
    value; both are read as "not declared".

    Args:
        value: Raw PARAM value (number, string, masked value or ``None``).

    Returns:
        float or None: The declared number, or ``None`` when absent.
    """
    if value is None or np.ma.is_masked(value):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    number = float(value)
    return None if np.isnan(number) else number


def _declared_text(value) -> str | None:
    """Reads a text PARAM value, treating an empty string as absent.

    Args:
        value: Raw PARAM value (string, masked value or ``None``).

    Returns:
        str or None: The stripped text, or ``None`` when empty or absent.
    """
    if value is None or np.ma.is_masked(value):
        return None
    text = str(value).strip()
    return text or None


def extract_photdm(tree):
    """GAVO tree walker for IVOA PhotDM ``photcal`` GROUP metadata.

    Traverses a GAVO-parsed VOTable tree and resolves photometric calibration
    groups linked to photometry columns. Handles inline ``PARAM`` entries and
    ``PARAMref`` indirection (the pattern used by modern VO publishers such as
    DaCHS/UPJS TAP services), plus ``FIELD/@ref`` links to ``photcal`` GROUP
    ``@ID`` values (Gaia DR3 ARI epoch photometry).

    Args:
        tree: The GAVO VOTable tree node/element to parse.

    Returns:
        dict: A mapping of target column references (strings) to PhotDM instances.
    """
    id_map = {}

    def map_ids(node, text, attrs, childIter):
        node_id = getattr(node, "id", None) or getattr(node, "ID", None)
        if node_id:
            id_map[node_id] = node
        for child in childIter:
            if hasattr(child, "apply"):
                child.apply(map_ids)

    tree.apply(map_ids)

    dm_map = {}
    photcal_groups: dict[str, PhotDM] = {}

    UT_FLUX = "photDM:PhotCal.zeroPoint.flux.value"
    # PhotDM 1.1 attribute name; the older spelling ``...referenceMagnitude.value`` that
    # earlier files carry is still read.
    UT_MAG = "photDM:PhotCal.zeroPoint.referenceMagnitudeValue"
    UT_MAG_LEGACY = "photDM:PhotCal.zeroPoint.referenceMagnitude.value"
    UT_MAG_SYS = "photDM:PhotCal.magnitudeSystem.type"
    UT_FILTER = "photDM:PhotometryFilter.identifier"
    UT_FILTER_NAME = "photDM:PhotometryFilter.name"
    UT_FILTER_SPEC = "photDM:PhotometryFilter.spectralLocation.value"

    def process_group(node, text, attrs, childIter):
        if node.name_ == "GROUP" and getattr(node, "name", None) == "photcal":
            cal_params = {}
            filter_params = {
                "filter_id": None,
                "name": None,
                "spectral_location": None,
                "spectral_location_unit": None,
            }
            target_col = None
            group_id = getattr(node, "id", None) or getattr(node, "ID", None)
            for child in childIter:
                target_param = None
                role_utype = (getattr(child, "utype", None) or "").lower()

                if child.name_ == "PARAM":
                    target_param = child
                elif child.name_ == "PARAMref":
                    target_param = id_map.get(getattr(child, "ref", None))
                elif child.name_ == "FIELDref":
                    target_col = getattr(child, "ref", None)

                if target_param is not None:
                    ut = role_utype or (getattr(target_param, "utype", None) or "").lower()

                    if ut == UT_FLUX.lower():
                        value = _declared_float(target_param.value)
                        if value is not None:
                            cal_params["zp_flux"] = value
                            cal_params["zp_flux_unit"] = to_internal(
                                getattr(target_param, "unit", None)
                            )
                    elif ut in (UT_MAG.lower(), UT_MAG_LEGACY.lower()):
                        value = _declared_float(target_param.value)
                        if value is not None:
                            cal_params["zp_mag"] = value
                            cal_params["zp_mag_unit"] = target_param.unit
                    elif ut == UT_MAG_SYS.lower():
                        cal_params["mag_sys"] = _declared_text(target_param.value)
                    elif ut == UT_FILTER.lower():
                        filter_params["filter_id"] = _declared_text(target_param.value)
                    elif ut == UT_FILTER_NAME.lower():
                        filter_params["name"] = _declared_text(target_param.value)
                    elif (
                        not ut
                        and (getattr(target_param, "name", None) or "").lower()
                        in ("filter", "filtername")
                    ):
                        if not filter_params.get("name"):
                            filter_params["name"] = _declared_text(target_param.value)
                    elif ut == UT_FILTER_SPEC.lower():
                        value = _declared_float(target_param.value)
                        if value is not None:
                            filter_params["spectral_location"] = value
                            filter_params["spectral_location_unit"] = getattr(
                                target_param, "unit", None
                            )

            phot_filter = PhotometryFilter(**filter_params)
            photcal = PhotCal(**cal_params, photometry_filter=phot_filter)
            photdm = PhotDM(photcal=photcal, photometry_filter=phot_filter)
            if group_id:
                photcal_groups[str(group_id)] = photdm
            if target_col:
                dm_map[_normalize_photdm_column_key(target_col)] = photdm

        for child in childIter:
            if hasattr(child, "apply"):
                child.apply(process_group)

    tree.apply(process_group)
    _link_photdm_field_refs(tree, photcal_groups, dm_map)
    return dm_map


def _list_votable_table_ids(payload: bytes) -> list[str | int]:
    """Lists Astropy VOTable table identifiers for a byte payload.

    Args:
        payload (bytes): Raw VOTable content.

    Returns:
        list: Table ``ID``, ``name``, or zero-based index for each embedded table.
    """
    import astropy.io.votable as vot

    vot_file = vot.parse(io.BytesIO(payload))
    table_ids: list[str | int] = []
    for index, table in enumerate(vot_file.iter_tables()):
        table_ids.append(table.ID or table.name or index)
    return table_ids


def _read_votable_payload(file_path) -> bytes:
    """Reads a VOTable path or stream into bytes for a single-pass ingest.

    Args:
        file_path (str or file-like): Path or binary stream.

    Returns:
        bytes: Raw VOTable content.
    """
    if hasattr(file_path, "read"):
        if hasattr(file_path, "seek"):
            file_path.seek(0)
        return file_path.read()
    with open(file_path, "rb") as handle:
        return handle.read()


def _clean_text(value) -> str | None:
    """Returns stripped text, or ``None`` when the value is empty.

    Args:
        value (object): A description or INFO text.

    Returns:
        str or None: The stripped text, ``None`` for ``None`` or blank input.
    """
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _declared_infos(astro_tree, selected_table) -> list[dict]:
    """Lists the INFO elements that apply to the selected table, in file order.

    Collects the INFOs of the VOTABLE root, of the RESOURCE that holds the table and of
    the table itself. Nothing is dropped or merged, so repeated names (several
    ``creator`` entries) are all kept.

    Args:
        astro_tree (astropy.io.votable.tree.VOTableFile): Parsed VOTable.
        selected_table (astropy.io.votable.tree.TableElement): The table in use.

    Returns:
        list[dict]: One dict per INFO with ``name`` and ``value`` and, when declared,
            ``ID``, ``ucd``, ``utype``, ``unit``, ``xtype``, ``ref`` and ``content``.
    """
    holders = [astro_tree]
    for resource in astro_tree.resources:
        if any(table is selected_table for table in resource.tables):
            holders.append(resource)
    holders.append(selected_table)
    found: list[dict] = []
    for holder in holders:
        for info in getattr(holder, "infos", []) or []:
            entry = {"name": info.name, "value": info.value}
            for attribute in ("ID", "ucd", "utype", "unit", "xtype", "ref"):
                text = getattr(info, attribute, None)
                if text:
                    entry[attribute] = str(text)
            content = _clean_text(getattr(info, "content", None))
            if content:
                entry["content"] = content
            found.append(entry)
    return found


def _gavo_votable_metadata_tree(payload: bytes):
    """Builds a GAVO ``VOTABLE`` tree for metadata walkers without row decode.

    ``votparse.parse`` always yields ``tableparser.Rows`` at ``<DATA>``; that
    iterator must not be consumed (no ``list(rows)``), which avoids BINARY decode
    failures while preserving PARAMref resolution in photcal GROUPs.

    Args:
        payload (bytes): Raw VOTable bytes.

    Returns:
        GAVO ``V.VOTABLE`` root for ``extract_photdm`` / TIMESYS walkers.

    Raises:
        ImportError: When the GAVO VOTable stack is not installed.
        ValueError: When parsing completes without a ``VOTABLE`` root node.
    """
    try:
        from gavo.votable import tableparser, votparse
        from gavo.votable.model import VOTable as V
    except ImportError as exc:
        raise ImportError(
            "Reading a VOTable needs the GAVO VOTable stack. Install it with "
            "'pip install volightcurve[read]'. Writing a VOTable does not need it."
        ) from exc

    votable_node = None
    for element in votparse.parse(io.BytesIO(payload), {V.VOTABLE}):
        if isinstance(element, tableparser.Rows):
            continue
        if type(element).__name__ == "VOTABLE":
            votable_node = element
    if votable_node is None:
        raise ValueError("GAVO metadata parse did not yield a VOTABLE root.")
    return votable_node


def _gavo_votable_tree_from_source(file_path):
    """Parses VOTable metadata into a GAVO stanxml tree (no row materialisation).

    Args:
        file_path (str or file-like): Path or stream containing VOTable bytes.

    Returns:
        GAVO stanxml tree root for metadata walkers.
    """
    return _gavo_votable_metadata_tree(_read_votable_payload(file_path))


def extract_timesys(tree):
    """GAVO tree walker for the default ``TIMESYS`` metadata block.

    Args:
        tree: The GAVO VOTable tree node/element to parse.

    Returns:
        TimeSys: The first ``TIMESYS`` element in document order, or an empty default.
    """
    from volightcurve.time_reference import extract_timesys_metadata_from_gavo

    return extract_timesys_metadata_from_gavo(tree).default_timesys


def extract_coosys(tree):
    """GAVO tree walker for IVOA ``COOSYS`` metadata.

    Preserves absent ``@epoch`` attributes as ``None`` rather than inventing a
    default proper-motion epoch.

    Args:
        tree: The GAVO VOTable tree node/element to parse.

    Returns:
        CooSys or None: Populated coordinate-system metadata when present.
    """
    data = {"coosys_id": "system", "epoch": None, "system": None}

    def find_cs(node, text, attrs, childIter):
        if node.name_ == "COOSYS":
            data["coosys_id"] = attrs.get("ID", "system")
            data["system"] = attrs.get("system")
            epoch = attrs.get("epoch")
            if epoch is not None:
                try:
                    data["epoch"] = float(epoch)
                except (TypeError, ValueError):
                    data["epoch"] = epoch
        for child in childIter:
            if hasattr(child, "apply"):
                child.apply(find_cs)

    tree.apply(find_cs)
    if not data["system"]:
        return None
    return CooSys(
        coosys_id=data["coosys_id"],
        system=data["system"],
        epoch=data["epoch"],
    )
