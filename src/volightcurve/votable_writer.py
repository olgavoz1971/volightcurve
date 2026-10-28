"""VOTable writer: ``write_vo_lightcurve`` and its helpers."""

import io
import logging

import astropy.units as u
from astropy.table import Table

from volightcurve.vo_unit_codec import (
    rewrite_astropy_empty_unit_attributes,
    to_internal,
    to_wire,
)

from volightcurve.columns import DEFAULT_TIME_EPOCH_UCD, is_measurement_ucd
from volightcurve.model import PhotDM

logger = logging.getLogger(__name__)


def _unit_text_for_check(unit) -> str:
    """Returns the text of an Astropy unit as it would appear on the VOTable wire.

    Args:
        unit (astropy.units.UnitBase): Unit of a table column.

    Returns:
        str: ``1`` for dimensionless, the raw text for an unrecognised unit, the VOUnits
            form otherwise (the plain text when Astropy cannot express it in VOUnits).
    """
    if isinstance(unit, u.UnrecognizedUnit):
        return str(unit)
    if unit == u.dimensionless_unscaled:
        return "1"
    try:
        return unit.to_string("vounit")
    except (ValueError, u.UnitsError):
        return str(unit)


def _force_vounits_on_wire(vot_file) -> None:
    """Makes every FIELD and PARAM unit be written as VOUnits text.

    WORKAROUND for old Astropy. Up to the 6.x/7.x series (6.1.7 here) Astropy writes
    FIELD and PARAM units with its CDS formatter (``count/s`` becomes ``ct.s-1``), with no
    option to change that. Astropy 8.0 and later write VOUnits instead, but they need
    Python 3.11 or newer; the development machine has 3.11, the production server does
    not. Remove this function once the production server can run Astropy 8.0 or later.

    Each recognised unit is therefore replaced
    by an unrecognised-unit object that carries its VOUnits text, which Astropy
    writes as given. Unrecognised units already carry their own text and are kept.
    A unit that VOUnits cannot express is left to Astropy.
    Dimensionless units are left to ``rewrite_astropy_empty_unit_attributes``.

    Args:
        vot_file (astropy.io.votable.tree.VOTableFile): File about to be serialised.
    """
    for element in vot_file.iter_fields_and_params():
        unit = element.unit
        if unit is None or isinstance(unit, u.UnrecognizedUnit):
            continue
        if unit == u.dimensionless_unscaled:
            continue
        try:
            text = unit.to_string("vounit")
        except (ValueError, u.UnitsError):
            # VOUnits cannot express this unit (for example electron): Astropy writes
            # it as it did before; the unit check reports it on reading.
            continue
        element._unit = u.UnrecognizedUnit(text)


def _repair_votable_xml_char_param_ampersands(payload: bytes) -> bytes:
    """Corrects double-escaped ampersands in astropy VOTable XML char PARAM values.

    Astropy 6.x may serialise ``&`` in char PARAM ``value`` attributes as
    ``&amp;amp;`` instead of ``&amp;``. Archives and VO clients expect a single
    XML escape (e.g. bibcodes containing ``A&A``).

    Args:
        payload (bytes): Raw VOTable XML document bytes.

    Returns:
        bytes: XML with ``&amp;amp;`` normalised to ``&amp;``.
    """
    return payload.replace(b"&amp;amp;", b"&amp;")


def _photcal_params_from_photdm(photdm) -> dict:
    """Collects the declared photcal values of one ``PhotDM`` for writing.

    Args:
        photdm (PhotDM): Calibration of one or more columns.

    Returns:
        dict: Keys ``filter_identifier``, ``filter_name``, ``zero_point_flux``,
            ``zero_point_flux_unit`` (internal text or ``None``), ``zero_point_ref_mag``,
            ``zero_point_ref_mag_unit``, ``magnitude_system``, ``effective_wavelength``
            and ``effective_wavelength_unit``; ``None`` for everything not declared.
    """
    params = {
        "filter_identifier": None,
        "filter_name": None,
        "zero_point_flux": None,
        "zero_point_flux_unit": None,
        "zero_point_ref_mag": None,
        "zero_point_ref_mag_unit": "mag",
        "magnitude_system": None,
        "effective_wavelength": None,
        "effective_wavelength_unit": "m",
    }
    phot_filter = photdm.filter
    if phot_filter is not None:
        params["filter_identifier"] = phot_filter.filter_id or None
        params["filter_name"] = phot_filter.name
        spectral = phot_filter.spectral_location
        if spectral is not None:
            params["effective_wavelength"] = float(getattr(spectral, "value", spectral))
            unit = getattr(spectral, "unit", None)
            if unit is not None:
                params["effective_wavelength_unit"] = str(unit)
    photcal = photdm.photcal
    if photcal is not None:
        if photcal.zp_flux is not None:
            params["zero_point_flux"] = float(getattr(photcal.zp_flux, "value", photcal.zp_flux))
            params["zero_point_flux_unit"] = to_internal(photcal.zero_point._zp_flux_unit_text)
        if photcal.zp_mag is not None:
            params["zero_point_ref_mag"] = float(getattr(photcal.zp_mag, "value", photcal.zp_mag))
            params["zero_point_ref_mag_unit"] = photcal.zero_point._zp_mag_unit_text or "mag"
        params["magnitude_system"] = photcal.mag_sys or None
    return params


def _photcal_group_specs(photdms: dict, phot_field_ids: list[str], error_field_ids: list[str]) -> list:
    """Groups table columns by the ``PhotDM`` object that describes them.

    Columns mapped to the same object share one group. Photometry columns that are not
    mapped share one empty calibration. Error columns that are not mapped are left out.

    Args:
        photdms (dict): Column name to ``PhotDM`` (names compared case-insensitively).
        phot_field_ids (list[str]): Flux and magnitude columns in table order.
        error_field_ids (list[str]): Flux and magnitude error columns in table order.

    Returns:
        list: ``(group_id, params, columns)`` triples in table order.
    """
    lowered = {str(key).lower(): value for key, value in photdms.items()}
    groups: dict[int, tuple[object, list[str]]] = {}
    unmapped: list[str] = []
    for name in phot_field_ids + error_field_ids:
        photdm = lowered.get(name.lower())
        if photdm is None:
            if name in phot_field_ids:
                unmapped.append(name)
            continue
        groups.setdefault(id(photdm), (photdm, []))[1].append(name)
    specs = [
        (f"photcal_{columns[0]}", _photcal_params_from_photdm(photdm), columns)
        for photdm, columns in groups.values()
    ]
    if unmapped:
        empty = _photcal_params_from_photdm(PhotDM())
        specs.append((f"photcal_{unmapped[0]}", empty, unmapped))
    return specs


def _append_photcal_group(vot_file, res, tab, group_id: str, params: dict, columns: list[str]) -> None:
    """Adds one ``GROUP name="photcal"`` to the resource and links its columns.

    Only declared values become PARAMs. Every column gets ``FIELD/@ref`` and a
    ``FIELDref`` inside the group.

    Args:
        vot_file: Astropy ``VOTableFile`` being built.
        res: Resource that receives the group.
        tab: Table whose FIELDs are linked.
        group_id (str): ``GROUP/@ID``.
        params (dict): Output of ``_photcal_params_from_photdm``.
        columns (list[str]): Column names described by this calibration.
    """
    from astropy.io.votable.tree import Group, Param, FieldRef

    g = Group(vot_file, ID=group_id, name='photcal')
    # TimeSeries Note: utype is mandatory, ucd="phot" is recommended for the group.
    g.utype = "PhotDM:PhotCal"
    g.ucd = "phot"

    def add(name, value, datatype, utype, ucd, unit=None, arraysize=None, description=None):
        kwargs = {"name": name, "value": value, "datatype": datatype}
        if arraysize is not None:
            kwargs["arraysize"] = arraysize
        param = Param(vot_file, **kwargs)
        if unit is not None:
            # Keep the declared text; old Astropy (before 8.0, which needs Python 3.11)
            # would re-spell it in CDS on writing. See _force_vounits_on_wire.
            param._unit = u.UnrecognizedUnit(unit)
        param.utype = utype
        param.ucd = ucd
        if description:
            param.description = description
        g.entries.append(param)

    if params["filter_identifier"]:
        add("filterIdentifier", params["filter_identifier"], "char",
            "photDM:PhotometryFilter.identifier", "meta.id;instr.filter", arraysize="*")
    if params["zero_point_flux"] is not None:
        add("zeroPointFlux", float(params["zero_point_flux"]), "double",
            "photDM:PhotCal.zeroPoint.flux.value", "phot.flux;arith.zp",
            unit=to_wire(to_internal(params["zero_point_flux_unit"])))
    if params["zero_point_ref_mag"] is not None:
        add("zeroPointReferenceMagnitude", float(params["zero_point_ref_mag"]), "double",
            "photDM:PhotCal.zeroPoint.referenceMagnitudeValue", "phot.mag;arith.zp",
            unit=params["zero_point_ref_mag_unit"])
    if params["magnitude_system"]:
        add("magnitudeSystem", str(params["magnitude_system"]), "char",
            "photDM:PhotCal.magnitudeSystem.type", "meta.code", arraysize="*")
    if params["filter_name"] is not None:
        add("filterName", str(params["filter_name"]), "char",
            "photDM:PhotometryFilter.name", "meta.id;instr.filter", arraysize="*",
            description="Human-readable photometric filter name")
    if params["effective_wavelength"] is not None:
        add("effectiveWavelength", float(params["effective_wavelength"]), "double",
            "photDM:PhotometryFilter.spectralLocation.value", "em.wl.effective",
            unit=params["effective_wavelength_unit"])

    for name in columns:
        g.entries.append(
            FieldRef(
                vot_file,
                ref=name,
                utype="adhoc:location",
                config={"version_1_2_or_later": True},
            )
        )
        for f in tab.fields:
            if f.name == name:
                f.ref = group_id
    res.groups.append(g)


def write_vo_lightcurve(
    output_stream_or_path,
    table_data,
    table_name: str,
    filter_identifier: str | None = None,
    refposition: str | None = None,
    timescale: str | None = None,
    timeorigin: float | None = None,
    votable_description: str | None = None,
    creator: str | None = None,
    zero_point_flux: float | None = None,
    zero_point_flux_unit: str | None = None,
    zero_point_ref_mag: float | None = None,
    zero_point_ref_mag_unit: str = "mag",
    magnitude_system: str | None = None,
    effective_wavelength: float | None = None,
    effective_wavelength_unit: str = "m",
    table_description: str | None = None,
    ra: float | None = None,
    dec: float | None = None,
    filter_name: str | None = None,
    period: float | None = None,
    epoch: float | None = None,
    binary: bool = True,
    coosys_id: str = "system",
    coosys_system: str | None = None,
    coosys_epoch: float | str | None = None,
    publication_id: str | None = None,
    facility_name: str | None = None,
    instrument_name: str | None = None,
    photdms: dict | None = None,
    infos: list[dict] | None = None,
    table_params: list[dict] | None = None,
):
    """Writes a lightcurve to a compliant IVOA VOTable (v1.4) XML file/stream.

    Nothing is invented: a value that is not given is not written. The photometric
    calibration comes from ``photdms`` (column name to ``PhotDM``): one
    ``GROUP name="photcal"`` is written per distinct ``PhotDM`` object, referencing every
    column mapped to it (a flux and its error may share one). A ``PhotDM`` with nothing
    declared still gives a group, without value PARAMs. The flat keyword arguments
    (``filter_identifier``, ``zero_point_*``, ``magnitude_system``,
    ``effective_wavelength*``, ``filter_name``) are DEPRECATED: they describe one
    calibration for all photometry columns and are kept for one release. They cannot be
    combined with ``photdms``.

    Column names are preserved. UCDs already stored on columns are written
    through; this function does not rename fields or drop columns
    (``docs/io_contract.md`` §8). A TIME-role column with no UCD receives
    ``time.epoch``. A non-empty time UCD is not rewritten. ``FIELD/@ref``
    (and the epoch PARAM) point at the ``TIMESYS/@ID`` of the time system
    actually written on this resource, and are omitted if none exists.
    Photometry columns that already carry a photometry UCD are linked from
    the photcal GROUP.

    Args:
        output_stream_or_path (str or file-like object): Path or stream to write the output to.
        table_data (astropy.table.Table or pandas.DataFrame or VOLightCurve):
            The source lightcurve containing timing and photometry.
        table_name (str): Value for the `<TABLE name="...">` attribute (Obligatory).
        filter_identifier (str, optional): DEPRECATED (use ``photdms``). Value for the
            ``filterIdentifier`` PARAM. Omitted from the photcal GROUP when absent.
        refposition (str, optional): Time reference position (e.g. 'BARYCENTER',
            'HELIOCENTER'). ``None`` when unknown.
        timescale (str, optional): Time scale. ``None`` when unknown.
        timeorigin (float, optional): TIMESYS origin (``JD0`` sense) added to
            time-role columns to obtain absolute Julian Date. ``None`` when unknown;
            then no ``TIMESYS`` is written (a declared ``refposition`` or ``timescale``
            is dropped with a warning). When given and ``refposition`` or ``timescale``
            is ``None``, the IVOA vocabulary term ``UNKNOWN`` is written for it.
        votable_description (str, optional): High-level global description. Defaults to None.
        creator (str, optional): Pipeline or entity creator name. Defaults to None.
        zero_point_flux (float, optional): DEPRECATED (use ``photdms``). Zero point flux value.
        zero_point_flux_unit (str, optional): DEPRECATED. Internal unit of zeroPointFlux
            (``None`` = dimensionless; written as ``VO_DIMENSIONLESS_WIRE``).
        zero_point_ref_mag (float, optional): DEPRECATED. Reference magnitude zero point.
        zero_point_ref_mag_unit (str, optional): DEPRECATED. Unit of zeroPointReferenceMagnitude.
        magnitude_system (str, optional): DEPRECATED. Type of magnitude system; not written
            when ``None``.
        effective_wavelength (float, optional): DEPRECATED. Effective wavelength value.
        effective_wavelength_unit (str, optional): DEPRECATED. Unit of effectiveWavelength.
        table_description (str, optional): Table block description. Defaults to None.
        infos (list[dict], optional): INFO elements to write on the RESOURCE, each a dict
            with ``name`` and ``value`` and optionally ``ID``, ``ucd``, ``utype``, ``unit``,
            ``xtype``, ``ref`` and ``content``. Written as given, repeated names included.
        ra (float, optional): RA of target in degrees. Defaults to None.
        dec (float, optional): Dec of target in degrees. Defaults to None.
        filter_name (str, optional): DEPRECATED (use ``photdms``). Human-readable filter
            name written as ``photDM:PhotometryFilter.name`` in the photcal GROUP (and a
            legacy table PARAM ``filter``). Defaults to None.
        period (float, optional): Variability period in days. Defaults to None.
        epoch (float, optional): Reference time epoch in days. Defaults to None.
        binary (bool, optional): If True, encodes table data in BINARY format.
            If False, encodes in TABLEDATA XML format. Defaults to True.
        coosys_id (str, optional): ``COOSYS/@ID`` when writing coordinate metadata.
        coosys_system (str, optional): ``COOSYS/@system`` (e.g. ``ICRS``). When omitted,
            no ``<COOSYS>`` element is written unless supplied via a ``VOLightCurve``.
        coosys_epoch (float or str, optional): ``COOSYS/@epoch`` (proper-motion epoch).
        publication_id (str, optional): Publication bibcode written as TABLE ``bibcode`` PARAM.
        facility_name (str, optional): Observatory or facility name TABLE PARAM; not
            written when ``None`` or empty.
        instrument_name (str, optional): Instrument name TABLE PARAM; not written when
            ``None`` or empty.
        table_params (list[dict], optional): PARAM elements to write on the TABLE, each a dict
            with ``name`` and ``value`` and optionally ``datatype`` (default ``char``),
            ``arraysize`` (default ``*`` for ``char``), ``unit``, ``ucd``, ``utype``, ``ref`` (the
            ``TIMESYS`` ID, for a time value) and ``description``. Written as given and in
            this order, after the standard PARAMs of this function.
        photdms (dict, optional): Column name to ``PhotDM``. Columns that share one
            ``PhotDM`` object share one photcal GROUP. Photometry columns that are not
            listed are tied to one shared empty calibration; error columns that are not
            listed stay unlinked.

    Raises:
        ValueError: When ``photdms`` is combined with the deprecated flat calibration
            arguments, or when an entry of ``table_params`` has no ``name`` or ``value`` or
            repeats the name of a PARAM already written.
    """
    import astropy.io.votable as vot
    from astropy.io.votable.tree import Param, Info, TimeSys, CooSys as VOTableCooSys
    import pandas as pd

    # Extract astropy Table
    if isinstance(table_data, Table):
        t = table_data.copy()
    elif hasattr(table_data, 'table') and isinstance(table_data.table, Table):
        t = table_data.table.copy()
    elif isinstance(table_data, pd.DataFrame):
        t = Table.from_pandas(table_data)
    else:
        raise TypeError(
            "table_data must be an astropy Table, VOLightCurve, or pandas DataFrame."
        )

    t_out = t

    # Convert to VOTableFile structure
    vot_file = vot.from_table(t_out)
    vot_file.version = '1.4'
    if votable_description:
        vot_file.description = votable_description

    res = vot_file.resources[0]
    tab = res.tables[0]
    tab.name = table_name
    if table_description:
        tab.description = table_description

    for entry in infos or []:
        info = Info(name=entry["name"], value=entry["value"])
        for attribute in ("ID", "ucd", "utype", "unit", "xtype", "ref"):
            if entry.get(attribute):
                setattr(info, attribute, entry[attribute])
        if entry.get("content"):
            info.content = entry["content"]
        res.infos.append(info)

    # Add creator info if provided
    if creator:
        info = Info(name='creator', value=creator)
        info.ucd = 'meta.bib.author'
        info.content = 'Pipeline or contributing resource creator'
        res.infos.append(info)

    # TIMESYS is written only when the time origin is known (VOTable 1.4: the origin is
    # required unless the time is a calendar date). FIELD/@ref below uses its ID.
    timesys_id = None
    if timeorigin is None:
        if refposition is not None or timescale is not None:
            logger.warning(
                "No time origin given: TIMESYS is not written; refposition=%r and "
                "timescale=%r are dropped.",
                refposition,
                timescale,
            )
    else:
        ts = TimeSys(
            ID='ts',
            refposition=refposition or "UNKNOWN",
            timescale=timescale or "UNKNOWN",
            timeorigin=str(timeorigin),
            config={'version_1_4_or_later': True}
        )
        res.time_systems.append(ts)
        timesys_id = ts.ID

    if coosys_system is not None:
        cs = VOTableCooSys(
            ID=coosys_id or "system",
            system=coosys_system,
            epoch=str(coosys_epoch) if coosys_epoch is not None else None,
        )
        res.coordinate_systems.append(cs)

    flat_given = any(
        value is not None
        for value in (
            filter_identifier,
            zero_point_flux,
            zero_point_ref_mag,
            magnitude_system,
            effective_wavelength,
            filter_name,
        )
    )
    if photdms is not None and flat_given:
        raise ValueError(
            "photdms cannot be combined with the deprecated flat calibration arguments."
        )
    if photdms is None and not flat_given and hasattr(table_data, "photdms"):
        photdms = table_data.photdms

    # Preserve column UCDs. Link TIME-role fields to the TIMESYS we just wrote.
    phot_field_ids: list[str] = []
    error_field_ids: list[str] = []
    time_field_ids: list[str] = []
    for f in tab.fields:
        f.ID = f.name
        col = t_out[f.name]
        ucd = ""
        if col.info.meta:
            ucd = str(col.info.meta.get("ucd") or "")
        if DEFAULT_TIME_EPOCH_UCD in ucd:
            time_field_ids.append(f.name)
            f.ucd = ucd
            if timesys_id:
                f.ref = timesys_id
        elif ucd:
            f.ucd = ucd
        if ucd and ("phot.mag" in ucd or "phot.flux" in ucd) and is_measurement_ucd(ucd):
            if "stat.error" in ucd:
                error_field_ids.append(f.name)
            else:
                phot_field_ids.append(f.name)

    if not time_field_ids:
        raise ValueError(
            "The table has no column with a time UCD (time.epoch). The writer does not "
            "guess roles from column names; build the product with VOLightCurve.from_table "
            "or assign the UCDs first."
        )

    if photdms is None:
        # Deprecated flat route: one calibration for every photometry column.
        flat_params = {
            "filter_identifier": filter_identifier,
            "filter_name": filter_name,
            "zero_point_flux": zero_point_flux,
            "zero_point_flux_unit": zero_point_flux_unit,
            "zero_point_ref_mag": zero_point_ref_mag,
            "zero_point_ref_mag_unit": zero_point_ref_mag_unit,
            "magnitude_system": magnitude_system,
            "effective_wavelength": effective_wavelength,
            "effective_wavelength_unit": effective_wavelength_unit,
        }
        group_specs = []
        shared_columns = phot_field_ids + error_field_ids
        if shared_columns:
            group_specs.append((f"photcal_{shared_columns[0]}", flat_params, shared_columns))
        legacy_filter_name = filter_name
    else:
        group_specs = _photcal_group_specs(photdms, phot_field_ids, error_field_ids)
        legacy_filter_name = next(
            (params["filter_name"] for _, params, _ in group_specs if params["filter_name"]),
            None,
        )

    for group_id, params, columns in group_specs:
        _append_photcal_group(vot_file, res, tab, group_id, params, columns)

    # Table-level optional metadata PARAMs
    if ra is not None:
        p_ra = Param(vot_file, name='ra', value=float(ra), datatype='double')
        p_ra.ucd = 'pos.eq.ra'
        p_ra.description = 'RA of source object'
        tab.params.append(p_ra)

    if dec is not None:
        p_dec = Param(vot_file, name='dec', value=float(dec), datatype='double')
        p_dec.ucd = 'pos.eq.dec'
        p_dec.description = 'Dec of source object'
        tab.params.append(p_dec)

    if legacy_filter_name is not None:
        p_filt = Param(vot_file, name='filter', value=str(legacy_filter_name), datatype='char', arraysize='*')
        p_filt.ucd = 'meta.id;instr.filter'
        p_filt.description = 'Photometric filter name'
        tab.params.append(p_filt)

    if period is not None:
        p_per = Param(vot_file, name='period', value=float(period), datatype='double', unit='d')
        p_per.ucd = 'src.var;time.period'
        p_per.description = 'Period of the variable star'
        tab.params.append(p_per)

    if epoch is not None:
        p_ep = Param(vot_file, name='epoch', value=float(epoch), datatype='double', unit='d')
        p_ep.ucd = 'time.epoch'
        if timesys_id:
            p_ep.ref = timesys_id
        p_ep.description = 'Reference time'
        tab.params.append(p_ep)

    if publication_id:
        p_bib = Param(
            vot_file,
            name='bibcode',
            value=str(publication_id),
            datatype='char',
            arraysize='*',
        )
        p_bib.ucd = 'meta.bib.bibcode'
        p_bib.utype = 'ssa:Curation.Reference'
        p_bib.description = 'URL or bibcode of a publication describing this data.'
        tab.params.append(p_bib)

    for param_name, param_value in (
        ("facility_name", facility_name),
        ("instrument_name", instrument_name),
    ):
        if param_value is None or not str(param_value).strip():
            continue
        p_fac = Param(
            vot_file,
            name=param_name,
            value=str(param_value),
            datatype="char",
            arraysize="*",
        )
        p_fac.description = f"Observatory {param_name.replace('_', ' ')}"
        tab.params.append(p_fac)

    for entry in table_params or []:
        if not entry.get("name") or entry.get("value") is None:
            raise ValueError("Each table_params entry needs a 'name' and a 'value'.")
        if any(existing.name == entry["name"] for existing in tab.params):
            raise ValueError(f"The table already has a PARAM named {entry['name']!r}.")
        datatype = entry.get("datatype", "char")
        kwargs = {"name": entry["name"], "value": entry["value"], "datatype": datatype}
        if entry.get("unit"):
            kwargs["unit"] = entry["unit"]
        arraysize = entry.get("arraysize", "*" if datatype == "char" else None)
        if arraysize:
            kwargs["arraysize"] = arraysize
        extra = Param(vot_file, **kwargs)
        for attribute in ("ucd", "utype", "ref"):
            if entry.get(attribute):
                setattr(extra, attribute, entry[attribute])
        if entry.get("description"):
            extra.description = entry["description"]
        tab.params.append(extra)

    meta_src = getattr(t, 'meta', None) or {}
    optional_char_params = (
        ('sectors', 'meta.id;meta.dataset', 'TESS sector identifiers present in this lightcurve'),
        ('flux_origins', 'meta.code', 'Photometry extraction method (e.g. pdcsap, sap)'),
        ('authors', 'meta.bib.author', 'Pipeline author(s)'),
        ('title', 'meta.note', 'Display title for the lightcurve figure'),
        ('name', 'meta.id', 'Target identifier'),
        ('stitched', 'meta.code', 'True when sectors were stitched and flux calibration is relative'),
        ('cutout_source', 'meta.id;instr', 'Cutout data source: FFI or TPF'),
        ('mask_mode', 'meta.code', 'Aperture mask mode: handmade, threshold, or pipeline'),
    )
    for param_name, param_ucd, param_desc in optional_char_params:
        param_val = meta_src.get(param_name)
        if param_val is None:
            continue
        if isinstance(param_val, (list, tuple)):
            param_val = ','.join(str(v) for v in param_val)
        param_val = str(param_val).strip()
        if not param_val:
            continue
        p_extra = Param(vot_file, name=param_name, value=param_val, datatype='char', arraysize='*')
        p_extra.ucd = param_ucd
        p_extra.description = param_desc
        tab.params.append(p_extra)

    # Write to target path or stream. Astropy emits unit="---" for empty units;
    # always rewrite to VO_DIMENSIONLESS_WIRE (see vo_unit_codec).
    tabledata_format = 'binary' if binary else 'tabledata'
    needs_ampersand_repair = bool(publication_id and "&" in str(publication_id))
    buffer = io.BytesIO()
    # Workaround for Astropy before 8.0 (CDS unit text); see _force_vounits_on_wire.
    _force_vounits_on_wire(vot_file)
    vot_file.to_xml(buffer, tabledata_format=tabledata_format)
    payload = rewrite_astropy_empty_unit_attributes(buffer.getvalue())
    if needs_ampersand_repair:
        payload = _repair_votable_xml_char_param_ampersands(payload)
    if hasattr(output_stream_or_path, "write"):
        output_stream_or_path.write(payload)
    else:
        with open(output_stream_or_path, "wb") as handle:
            handle.write(payload)
