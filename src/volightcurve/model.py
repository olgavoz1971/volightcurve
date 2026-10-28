"""Data model classes: filter, calibration, PhotDM, coordinate and time systems."""

import logging

import astropy.units as u

from volightcurve.photdm import MagnitudeSystem, PogsonZeroPoint, ZeroPoint

logger = logging.getLogger(__name__)


class PhotometryFilter:
    """Represents a photometric filter with its identifier and physical spectral location.

    This class corresponds to IVOA photDM:PhotometryFilter data model component, which
    is used to describe a bandpass/filter in astronomical observations.

    Two different things are kept apart (PhotDM 1.1):

    * ``filter_id`` (``PhotometryFilter.identifier``) is the code that identifies the
      filter in a Filter Profile Service, for example ``TESS/TESS.Red`` or
      ``Palomar/ZTF.r`` (``Facility/Subcategory.Band``). It is not a label.
    * ``name`` (``PhotometryFilter.name``) is a short human-readable label, for example
      ``Red`` or ``V``, for file names, titles and plots.

    The core stores exactly what a file or a caller declares for each. Only the
    interpretation step guesses: a loose value with a slash becomes the identifier, a
    bare label becomes the name.
    """

    def __init__(
        self,
        filter_id=None,
        name: str | None = None,
        spectral_location=None,
        spectral_location_unit: str = None,
    ):
        """Initialises a PhotometryFilter instance.

        Args:
            filter_id (str, optional): Unique passport
                (``photDM:PhotometryFilter.identifier``), e.g. ``Generic/Bessell.I``.
            name (str, optional): Human-readable label
                (``photDM:PhotometryFilter.name``), e.g. ``I`` for plot axes.
            spectral_location (float or astropy.units.Quantity, optional): The physical
                spectral location (e.g., central wavelength) of the filter. Defaults to None.
            spectral_location_unit (str, optional): The physical unit of the spectral
                location (e.g., 'nm', 'AA', 'um') if spectral_location is not already
                an astropy.units.Quantity. Defaults to None.
        """
        self._filter_id = filter_id  # photDM:PhotometryFilter.identifier
        self._name = name  # photDM:PhotometryFilter.name

        # Build the physical quantity
        if spectral_location is None:
            self.spectral_location = None
        else:
            if isinstance(spectral_location, u.Quantity):
                self.spectral_location = spectral_location
            else:
                # Pair the value with the unit (defaulting to dimensionless if None)
                unit = u.dimensionless_unscaled
                if spectral_location_unit is not None:
                    try:
                        unit = u.Unit(spectral_location_unit)
                    except Exception as e:
                        logger.warning(f'Inappropriate spectral_location_unit {spectral_location_unit}: {type(e)}')

                self.spectral_location = spectral_location * unit
        # todo: blah-blah-blah

    @property
    def filter_id(self):
        """Gets or sets the filter identifier (Filter Profile Service code).

        Not the display label; see ``name``.

        Returns:
            str: The filter identifier, for example ``TESS/TESS.Red``.
        """
        return self._filter_id

    @filter_id.setter
    def filter_id(self, value):
        if not isinstance(value, str):
            raise ValueError("Filter ID must be a string.")
        self._filter_id = value.strip()

    @property
    def name(self):
        """Gets or sets the human-readable filter name (PhotDM ``name``).

        A short label for file names, titles and plots. It does not identify the
        filter; see ``filter_id``.

        Returns:
            str or None: Display label for plots and UI.
        """
        return self._name

    @name.setter
    def name(self, value):
        if value is None:
            self._name = None
            return
        if not isinstance(value, str):
            raise ValueError("Filter name must be a string.")
        stripped = value.strip()
        self._name = stripped or None

    def __repr__(self):
        return (
            f"<PhotometryFilter: filter_id={self.filter_id} name={self.name} "
            f"spectralLocation={self.spectral_location}>"
        )


class PhotCal:
    """photDM:PhotCal façade — filter calibration with nested ZeroPoint.

    Callers use ``mag_to_flux`` / ``flux_to_mag`` (and error helpers) on this
    object only. Scale-specific maths live on ``self.zero_point``
    (``PogsonZeroPoint`` today; Asinh/Linear stubs for later).

    Flat constructor args (``zp_flux``, ``zp_mag``, ``mag_sys``) remain supported
    and build a ``PogsonZeroPoint`` + ``MagnitudeSystem`` (compressed PhotDM).
    Omitted zero points stay missing. They are not filled with a usable pair.
    Conversion then raises :class:`PhotCalError`.
    """

    def __init__(
        self,
        zp_flux=None,
        zp_flux_unit=None,
        zp_mag=None,
        zp_mag_unit=None,
        mag_sys=None,
        *,
        zero_point: ZeroPoint | None = None,
        magnitude_system: MagnitudeSystem | None = None,
        photometry_filter: PhotometryFilter | None = None,
        identifier: str | None = None,
    ):
        """Initialises a PhotCal instance.

        Args:
            zp_flux: Zero-point flux (used when ``zero_point`` is omitted).
            zp_flux_unit: Unit for ``zp_flux`` when not a Quantity.
            zp_mag: Reference magnitude (used when ``zero_point`` is omitted).
            zp_mag_unit: Unit for ``zp_mag`` when not a Quantity.
            mag_sys: Magnitude system type string when ``magnitude_system`` omitted.
                ``None`` stays ``None`` (no system is assumed).
            zero_point: Nested PhotDM zero-point (Pogson/Asinh/Linear).
            magnitude_system: Nested ``MagnitudeSystem``.
            photometry_filter: Optional ``PhotometryFilter``.
            identifier: Optional PhotCal identifier.
        """
        if zero_point is not None:
            self.zero_point = zero_point
        else:
            self.zero_point = PogsonZeroPoint(
                zp_flux=zp_flux,
                zp_flux_unit=zp_flux_unit,
                zp_mag=zp_mag,
                zp_mag_unit=zp_mag_unit,
            )
        self.magnitude_system = magnitude_system or MagnitudeSystem(type=mag_sys or None)
        self.photometry_filter = photometry_filter
        self.identifier = identifier

    @property
    def zp_flux(self):
        """Gets or sets the zero-point flux.

        Returns:
            astropy.units.Quantity: The zero-point flux with its unit.
        """
        return self.zero_point.zp_flux

    @zp_flux.setter
    def zp_flux(self, value):
        self.zero_point.zp_flux = value

    @property
    def zp_mag(self):
        """Gets or sets the zero-point magnitude.

        Returns:
            astropy.units.Quantity: The zero-point magnitude with its unit.
        """
        return self.zero_point.zp_mag

    @zp_mag.setter
    def zp_mag(self, value):
        self.zero_point.zp_mag = value

    @property
    def mag_sys(self):
        """Gets or sets the magnitude system name.

        Returns:
            str: The magnitude system type.
        """
        return self.magnitude_system.type

    @mag_sys.setter
    def mag_sys(self, value):
        if not isinstance(value, str):
            raise ValueError("Magnitude system must be a string.")
        self.magnitude_system.type = value.strip()

    def conversion_problems(self, *, flux_unit=None, mag_unit=None) -> list[str]:
        """Lists why this calibration cannot convert magnitude and flux.

        Does not insert default zero points.

        Args:
            flux_unit: Flux column unit when a flux Quantity is in play.
            mag_unit: Magnitude unit when a magnitude Quantity is in play.

        Returns:
            list[str]: Empty when conversion may proceed.
        """
        from volightcurve.photcal_check import inspect_conversion_photcal

        zp = self.zero_point
        zp_flux = None if zp._zp_flux is None else zp._zp_flux.value
        zp_mag = None if zp._zp_mag is None else zp._zp_mag.value
        return inspect_conversion_photcal(
            zp_flux=zp_flux,
            zp_mag=zp_mag,
            zp_flux_unit=zp._zp_flux_unit_text,
            zp_mag_unit=zp._zp_mag_unit_text,
            flux_unit=flux_unit,
            mag_unit=mag_unit,
            zp_flux_unit_unparsed=bool(zp._zp_flux_unit_unparsed),
            zp_mag_unit_unparsed=bool(zp._zp_mag_unit_unparsed),
        )

    def _require_conversion(self, *, flux_unit=None, mag_unit=None) -> None:
        """Raises when this calibration cannot convert.

        Args:
            flux_unit: Flux column unit, when relevant.
            mag_unit: Magnitude unit, when relevant.

        Raises:
            PhotCalError: One or more specific reasons.
        """
        from volightcurve.photcal_check import PhotCalError

        problems = self.conversion_problems(flux_unit=flux_unit, mag_unit=mag_unit)
        if problems:
            raise PhotCalError(problems)

    def mag_to_flux(self, mag):
        """Converts magnitude to flux via the nested ``zero_point``.

        Args:
            mag (astropy.units.Quantity): Magnitude values.

        Returns:
            astropy.units.Quantity: Flux values.

        Raises:
            PhotCalError: When the zero points or units cannot support conversion.
        """
        self._require_conversion(mag_unit=getattr(mag, "unit", None))
        return self.zero_point.mag_to_flux(mag)

    def flux_to_mag(self, flux):
        """Converts flux to magnitude via the nested ``zero_point``.

        Args:
            flux (astropy.units.Quantity): Flux values.

        Returns:
            astropy.units.Quantity: Magnitude values.

        Raises:
            PhotCalError: When the zero points or units cannot support conversion.
        """
        self._require_conversion(flux_unit=getattr(flux, "unit", None))
        return self.zero_point.flux_to_mag(flux)

    def mag_err_to_flux_err(self, mag, mag_err):
        """Propagates magnitude uncertainties to flux via ``zero_point``.

        Args:
            mag (astropy.units.Quantity): Magnitude values.
            mag_err (astropy.units.Quantity): Magnitude uncertainties.

        Returns:
            astropy.units.Quantity: Flux uncertainties.

        Raises:
            PhotCalError: When the zero points or units cannot support conversion.
        """
        self._require_conversion(mag_unit=getattr(mag, "unit", None))
        return self.zero_point.mag_err_to_flux_err(mag, mag_err)

    def flux_err_to_mag_err(self, flux, flux_err):
        """Propagates flux uncertainties to magnitude via ``zero_point``.

        Args:
            flux (astropy.units.Quantity): Flux values.
            flux_err (astropy.units.Quantity): Flux uncertainties.

        Returns:
            astropy.units.Quantity: Magnitude uncertainties.

        Raises:
            PhotCalError: When the zero points or units cannot support conversion.
        """
        self._require_conversion(flux_unit=getattr(flux, "unit", None))
        return self.zero_point.flux_err_to_mag_err(flux, flux_err)

    def __repr__(self):
        return (
            f"<PhotCal zeroPoint={self.zero_point!r} "
            f"magnitudeSystem={self.magnitude_system!r}>"
        )


class PhotDM:
    """Photometry Data Model (photDM) hub: one calibration with its filter.

    Follows PhotDM 1.1: the filter belongs to the ``PhotCal`` (``PhotCal.photometryFilter``).
    ``PhotDM.filter`` is a shortcut to ``photcal.photometry_filter``. A filter given
    while no calibration exists is held until a calibration is assigned, then handed to
    it. Two different filters (one on the calibration, one on the hub) are refused.
    """

    def __init__(self, photcal: PhotCal = None, photometry_filter: PhotometryFilter = None):
        """Initialises a PhotDM instance.

        Args:
            photcal (PhotCal, optional): The photometric calibration. Defaults to None.
            photometry_filter (PhotometryFilter, optional): The filter. Stored on
                ``photcal`` when one is given. Defaults to None.

        Raises:
            ValueError: When ``photcal`` already holds a different filter object.
        """
        self._photcal = None
        self._held_filter = None
        if (
            photcal is not None
            and photometry_filter is not None
            and photcal.photometry_filter is not None
            and photcal.photometry_filter is not photometry_filter
        ):
            raise ValueError(
                "The calibration and the PhotDM hold different filters; "
                "set the filter on one of them only."
            )
        self.photcal = photcal
        if photometry_filter is not None:
            self.filter = photometry_filter

    @property
    def photcal(self):
        """Gets or sets the calibration.

        Setting a calibration that has no filter gives it the filter held so far.

        Returns:
            PhotCal or None: The calibration.

        Raises:
            ValueError: On assignment, when both the new calibration and this hub
                hold a filter and the two are different objects.
        """
        return self._photcal

    @photcal.setter
    def photcal(self, value):
        previous = self.filter
        if value is not None and previous is not None:
            if value.photometry_filter is None:
                value.photometry_filter = previous
            elif value.photometry_filter is not previous:
                raise ValueError(
                    "The calibration and the PhotDM hold different filters; "
                    "set the filter on one of them only."
                )
        self._photcal = value
        self._held_filter = None if value is not None else previous

    @property
    def filter(self):
        """Gets or sets the filter (``photcal.photometry_filter`` when a calibration exists).

        Returns:
            PhotometryFilter or None: The filter.
        """
        if self._photcal is not None:
            return self._photcal.photometry_filter
        return self._held_filter

    @filter.setter
    def filter(self, value):
        if self._photcal is not None:
            self._photcal.photometry_filter = value
        else:
            self._held_filter = value

    @property
    def filter_id(self):
        """Gets or sets the filter identifier shortcut.

        Returns:
            str or None: The identifier of the filter, or ``None`` when no filter is set
                or the filter has no identifier.
        """
        return self.filter.filter_id if self.filter else None

    @filter_id.setter
    def filter_id(self, value):
        self.filter.filter_id = value

    @property
    def filter_name(self):
        """Gets or sets the human-readable filter name on the nested filter.

        Returns:
            str or None: ``PhotometryFilter.name``, or None if no filter is set.
        """
        return self.filter.name if self.filter else None

    @filter_name.setter
    def filter_name(self, value):
        if self.filter is None:
            self.filter = PhotometryFilter(name=value)
        else:
            self.filter.name = value

    @property
    def mag0(self):
        """Gets or sets the zero-point magnitude shortcut.

        Returns:
            astropy.units.Quantity or None: The zero-point magnitude of the calibration,
                or ``None`` when no calibration is set or it declares none.
        """
        return self.photcal.zp_mag if self.photcal else None

    @mag0.setter
    def mag0(self, value):
        self.photcal.zp_mag = value

    def conversion_problems(self, *, flux_unit=None, mag_unit=None) -> list[str]:
        """Lists why this calibration cannot convert magnitude and flux.

        Args:
            flux_unit: Flux column unit, when relevant.
            mag_unit: Magnitude column unit, when relevant.

        Returns:
            list[str]: Empty when conversion may proceed; one sentence per problem.
        """
        if self.photcal is None:
            return ["No photometric calibration (photcal) is declared."]
        return self.photcal.conversion_problems(flux_unit=flux_unit, mag_unit=mag_unit)

    def __repr__(self):
        # f_id = self.filter_id
        # sys = self.photcal.mag_sys if self.photcal else "None"
        # return f"<PhotDM Filter={f_id} Sys={sys}>"
        return f"<PhotDM Filter={self.filter} photcal={self.photcal}>"


class CooSys:
    """Represents a coordinate system reference frame for spatial coordinates.

    Used to model VOTable ``<COOSYS>`` metadata (reference frame and epoch).
    """

    def __init__(self, *, epoch=None, system='ICRS', coosys_id='system'):
        """Initialises a CooSys instance.

        Args:
            epoch (float, int, or str, optional): Reference epoch of the coordinates
                (e.g. proper-motion epoch for Gaia). ``None`` when the source VOTable
                omits ``COOSYS/@epoch``.
            system (str, optional): Reference coordinate system/frame. Defaults to ``ICRS``.
            coosys_id (str, optional): VOTable ``COOSYS/@ID`` value. Defaults to ``system``.
        """
        self.epoch = epoch
        self.system = system
        self.coosys_id = coosys_id

    def __repr__(self):
        return f"<CooSys id={self.coosys_id} epoch={self.epoch}: system={self.system}>"


class TimeSys:
    """Represents a time system standard for astronomical timing metadata.

    Captures reference position, origin (such as JD0/MJD0 offset), and timescale.
    The values of a time column are offsets from ``timeorigin`` (absolute JD = value +
    ``timeorigin``); ``VOLightCurve.get_observation_times`` does the addition.
    """

    def __init__(self, refposition=None, timeorigin=None, timescale=None):
        """Initialises a TimeSys instance.

        Args:
            refposition (str, optional): Time reference position (e.g., 'HELIOCENTER',
                'BARYCENTER'). ``None`` when not declared.
            timeorigin (float, optional): The origin offset of the time scale, i.e.,
                the JD0 or MJD0 value. ``None`` when not declared (0.0 is a declared value).
            timescale (str, optional): The timing scale used (e.g., 'UTC', 'TDB', 'TCB').
                ``None`` when not declared.
        """
        self.refposition = refposition  # HELIOCENTER OR BARYCENTER ...
        self.timeorigin = timeorigin  # JD0, f.e. 2400000.5
        self.timescale = timescale  # UTC, TCB, TBD etc

    @property
    def jd0(self):
        """Gets the reference time origin (JD0 offset).

        Returns:
            float or None: The reference time origin value, ``None`` when not declared.
        """
        return self.timeorigin

    def __repr__(self):
        return f"<TimeSys: timescale={self.timescale} refposition={self.refposition} timeorigin={self.timeorigin}>"
