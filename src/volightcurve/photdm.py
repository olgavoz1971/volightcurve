"""PhotDM-aligned photometric calibration types (nested ZeroPoint model).

Structure follows IVOA PhotDM as reflected by the modelinstanceinvot client:
https://github.com/ivoa/modelinstanceinvot-code/blob/merge-syntax/python/client/photdm/photcal.py

Callers use ``PhotCal`` as the façade. Scale-specific magnitude↔flux maths
live on ``ZeroPoint`` subtypes (``PogsonZeroPoint``, stubs for Asinh/Linear).
"""

from __future__ import annotations

import logging
from typing import Any

import astropy.units as u
import numpy as np

from volightcurve.vo_unit_codec import to_internal

logger = logging.getLogger(__name__)


class MagnitudeSystem:
    """photDM:MagnitudeSystem -- catalogue magnitude system label.

    Args:
        type: System name (e.g. ``Vega``, ``AB``).
        reference_spectrum: Optional reference spectrum identifier.
    """

    def __init__(self, type: str = "Vega", reference_spectrum: str | None = None):
        self.type = type
        self.reference_spectrum = reference_spectrum

    def __repr__(self) -> str:
        return (
            f"<MagnitudeSystem type={self.type!r} "
            f"referenceSpectrum={self.reference_spectrum!r}>"
        )


class ZeroPoint:
    """Base photDM:ZeroPoint -- holds ZP flux/magnitude; subclasses own the scale maths.

    Args:
        zp_flux: Zero-point flux value or Quantity.
        zp_flux_unit: Unit string for ``zp_flux`` when not a Quantity.
        zp_mag: Reference magnitude value or Quantity.
        zp_mag_unit: Unit string for ``zp_mag`` when not a Quantity.
    """

    dmtype = "photdm:ZeroPoint"

    def __init__(
        self,
        zp_flux=None,
        zp_flux_unit=None,
        zp_mag=None,
        zp_mag_unit=None,
    ):
        internal_flux_unit = to_internal(zp_flux_unit)
        self._zp_flux_unit_text = internal_flux_unit
        self._zp_flux_unit_unparsed = False
        if zp_flux is None:
            self._zp_flux = None
        elif isinstance(zp_flux, u.Quantity):
            self._zp_flux = zp_flux
        else:
            unit = u.dimensionless_unscaled
            if internal_flux_unit is not None:
                try:
                    unit = u.Unit(internal_flux_unit)
                except Exception as e:
                    logger.warning(
                        "Invalid zp_flux_unit '%s' retained for photcal reconcile: %s",
                        internal_flux_unit,
                        e,
                    )
                    self._zp_flux_unit_unparsed = True
                    self._zp_flux_unit_text = str(internal_flux_unit).strip()
                    unit = u.dimensionless_unscaled
            self._zp_flux = zp_flux * unit

        self._zp_mag_unit_text = zp_mag_unit
        self._zp_mag_unit_unparsed = False
        if zp_mag is None:
            self._zp_mag = None
        elif isinstance(zp_mag, u.Quantity):
            self._zp_mag = zp_mag
        else:
            m_unit = u.mag
            if zp_mag_unit:
                try:
                    m_unit = u.Unit(zp_mag_unit)
                except Exception as e:
                    logger.warning(
                        "Invalid zp_mag_unit '%s' retained for photcal reconcile: %s",
                        zp_mag_unit,
                        e,
                    )
                    self._zp_mag_unit_unparsed = True
                    m_unit = u.mag
            self._zp_mag = zp_mag * m_unit

    @property
    def zp_flux(self):
        """Zero-point flux Quantity."""
        return self._zp_flux

    @zp_flux.setter
    def zp_flux(self, value):
        if not isinstance(value, (int, float, np.number)):
            raise ValueError("Zero point flux must be a number.")
        self._zp_flux = float(value)

    @property
    def zp_mag(self):
        """Reference magnitude Quantity."""
        return self._zp_mag

    @zp_mag.setter
    def zp_mag(self, value):
        if not isinstance(value, (int, float, np.number)):
            raise ValueError("Zero point magnitude must be a number.")
        self._zp_mag = float(value)

    def mag_to_flux(self, mag):
        """Converts magnitude to flux. Subclasses implement the scale."""
        raise NotImplementedError(f"{self.dmtype} does not implement mag_to_flux")

    def flux_to_mag(self, flux):
        """Converts flux to magnitude. Subclasses implement the scale."""
        raise NotImplementedError(f"{self.dmtype} does not implement flux_to_mag")

    def mag_err_to_flux_err(self, mag, mag_err):
        """Propagates magnitude uncertainty to flux. Subclasses implement."""
        raise NotImplementedError(f"{self.dmtype} does not implement mag_err_to_flux_err")

    def flux_err_to_mag_err(self, flux, flux_err):
        """Propagates flux uncertainty to magnitude. Subclasses implement."""
        raise NotImplementedError(f"{self.dmtype} does not implement flux_err_to_mag_err")

    @staticmethod
    def from_dmtype(dmtype: str, **kwargs: Any) -> ZeroPoint:
        """Builds a ZeroPoint subtype from a PhotDM ``dmtype`` string.

        Args:
            dmtype: e.g. ``photdm:PogsonZeroPoint``.
            **kwargs: Forwarded to the subtype constructor.

        Returns:
            ZeroPoint: Concrete zero-point instance.

        Raises:
            ValueError: If ``dmtype`` is unknown.
        """
        key = (dmtype or "").strip()
        if key in ("photdm:PogsonZeroPoint", "PogsonZeroPoint", "pogson", ""):
            return PogsonZeroPoint(**kwargs)
        if key in ("photdm:AsinhZeroPoint", "AsinhZeroPoint", "asinh"):
            return AsinhZeroPoint(**kwargs)
        if key in ("photdm:LinearFluxZeroPoint", "LinearFluxZeroPoint", "linear"):
            return LinearFluxZeroPoint(**kwargs)
        raise ValueError(f"ZeroPoint of type {dmtype!r} not supported")

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} zp_mag={self.zp_mag} zp_flux={self.zp_flux}>"
        )


class PogsonZeroPoint(ZeroPoint):
    """photDM:PogsonZeroPoint -- classic Pogson magnitude↔flux relation."""

    dmtype = "photdm:PogsonZeroPoint"

    def mag_to_flux(self, mag):
        """Converts magnitude to flux via Pogson: ``F = F0 * 10**(-0.4*(m-m0))``.

        Args:
            mag: Magnitude Quantity.

        Returns:
            Flux Quantity in the zero-point flux unit.

        Raises:
            astropy.units.UnitsError: On incompatible magnitude units.
        """
        if mag.unit is None or not mag.unit.is_equivalent(self._zp_mag.unit):
            raise u.UnitsError(
                f"magnitude column unit[{mag.unit}] and zero_point unit[{self._zp_mag.unit}] must match"
            )
        mag_zp = mag.to(self._zp_mag.unit)
        delta_mag = (mag_zp - self._zp_mag).to_value(self._zp_mag.unit)
        return self._zp_flux * 10 ** (-0.4 * delta_mag)

    def flux_to_mag(self, flux):
        """Converts flux to magnitude via Pogson: ``m = m0 - 2.5 log10(F/F0)``.

        Args:
            flux: Flux Quantity.

        Returns:
            Magnitude Quantity.

        Raises:
            astropy.units.UnitsError: On incompatible flux units.
        """
        if flux.unit is None or not flux.unit.is_equivalent(self._zp_flux.unit):
            raise u.UnitsError(
                f"flux column unit[{flux.unit}] and zero_point unit[{self._zp_flux.unit}] must match"
            )
        flux_zp = flux.to(self._zp_flux.unit)
        ratio = (flux_zp / self._zp_flux).to_value(u.dimensionless_unscaled)
        return self._zp_mag - 2.5 * np.log10(ratio) * self._zp_mag.unit

    def mag_err_to_flux_err(self, mag, mag_err):
        """Propagates σ_m to σ_F with ``|dF/dm| = 0.4 ln(10) F``.

        Args:
            mag: Magnitude Quantity.
            mag_err: Magnitude uncertainty Quantity.

        Returns:
            Flux uncertainty Quantity.
        """
        flux = self.mag_to_flux(mag)
        pogson_slope = 0.4 * np.log(10.0)
        if mag_err.unit is None or not mag_err.unit.is_equivalent(self._zp_mag.unit):
            raise u.UnitsError(
                f"magnitude error unit[{mag_err.unit}] and zero_point unit[{self._zp_mag.unit}] must match"
            )
        mag_err_zp = mag_err.to(self._zp_mag.unit)
        sigma_m = mag_err_zp.to_value(self._zp_mag.unit)
        return np.abs(flux * pogson_slope * sigma_m)

    def flux_err_to_mag_err(self, flux, flux_err):
        """Propagates σ_F to σ_m with ``|dm/dF| = 2.5 / (F ln 10)``.

        Args:
            flux: Flux Quantity.
            flux_err: Flux uncertainty Quantity.

        Returns:
            Magnitude uncertainty Quantity.
        """
        if flux.unit is None or not flux.unit.is_equivalent(self._zp_flux.unit):
            raise u.UnitsError(
                f"flux column unit[{flux.unit}] and zero_point unit[{self._zp_flux.unit}] must match"
            )
        flux_zp = flux.to(self._zp_flux.unit)
        if flux_err.unit is None or not flux_err.unit.is_equivalent(self._zp_flux.unit):
            raise u.UnitsError(
                f"flux error unit[{flux_err.unit}] and zero_point unit[{self._zp_flux.unit}] must match"
            )
        err_zp = flux_err.to(self._zp_flux.unit)
        ln10 = np.log(10.0)
        ratio = (err_zp / flux_zp).to_value(u.dimensionless_unscaled)
        return (2.5 / ln10) * np.abs(ratio) * self._zp_mag.unit


class AsinhZeroPoint(ZeroPoint):
    """photDM:AsinhZeroPoint stub -- softening parameter reserved for luptitudes.

    Conversion methods are not implemented yet (Phase 5 stub only).

    Args:
        softening_parameter: Asinh softening (luptitude) parameter.
        **kwargs: Forwarded to ``ZeroPoint``.
    """

    dmtype = "photdm:AsinhZeroPoint"

    def __init__(self, softening_parameter=None, **kwargs):
        super().__init__(**kwargs)
        self.softening_parameter = softening_parameter

    def mag_to_flux(self, mag):
        """Asinh mag→flux is not implemented yet."""
        raise NotImplementedError(
            "AsinhZeroPoint.mag_to_flux is not implemented yet "
            "(luptitude / asinh scale reserved)"
        )

    def flux_to_mag(self, flux):
        """Asinh flux→mag is not implemented yet."""
        raise NotImplementedError(
            "AsinhZeroPoint.flux_to_mag is not implemented yet "
            "(luptitude / asinh scale reserved)"
        )


class LinearFluxZeroPoint(ZeroPoint):
    """photDM:LinearFluxZeroPoint stub -- conversion not implemented yet."""

    dmtype = "photdm:LinearFluxZeroPoint"

    def mag_to_flux(self, mag):
        """Linear flux zero-point mag→flux is not implemented yet."""
        raise NotImplementedError("LinearFluxZeroPoint.mag_to_flux is not implemented yet")

    def flux_to_mag(self, flux):
        """Linear flux zero-point flux→mag is not implemented yet."""
        raise NotImplementedError("LinearFluxZeroPoint.flux_to_mag is not implemented yet")
