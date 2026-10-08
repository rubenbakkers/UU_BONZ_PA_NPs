"""Explicit, SI-anchored absorption assumptions for the wavelength scan.

The six Stark-energy arrays below reproduce Table 4(B), page 9 of Conrad et al.,
J. Phys. Chem. C (2025), DOI 10.1021/acs.jpcc.5c06209. They are the same numbers
as in the supplied wavelength script; no energy has been adjusted to fit Lee.
That table is not a measured absorption spectrum for the Lee nanoparticles.

Lee et al. (2021), SI Tables S6/S8, determine the effective cross sections at
1064 nm. This module preserves that anchor exactly. The following are explicit
EXPLORATORY assumptions, not parameters fitted in Lee:

* thermal populations within each starting multiplet at 300 K;
* equal band-integrated strengths and equal strengths to upper Stark states
  unless optional band weights / oscillator-strength matrices are supplied;
* narrow Gaussian standard deviation 8.5 cm^-1;
* a 50% integrated broad GSA component of width 1000 cm^-1, centred on actual
  H6 -> F4/H5 transition energies, rather than the old fictitious 943-nm band.

The broad component is a phenomenological phonon-sideband approximation. Its
red (photon-energy-deficit) wing has a thermal Boltzmann penalty; its blue wing
corresponds to disposing of excess photon energy. The penalty encodes a simple
thermal asymmetry, not a microscopic phonon spectrum, selection rules or a
measured sideband strength. The broad fraction refers to the symmetric kernel
before applying this penalty. Width/fraction/temperature need independent
constraints before any wavelength-detuning result is quantitative.

ESA consists of the explicit F4 -> H4/F3/F2 Stark transitions. No arbitrary
broad ESA band is added. Gaussian profiles are densities in WAVENUMBER, not
wavelength. Evaluating them at 1e7/lambda_nm needs no Jacobian: the returned
quantity is a cross section at a wavelength, not a probability density in nm.

Only 1064 nm is SI-calibrated. In particular, this module does not claim to
reproduce the 1450-nm threshold or slope. Stronger agreement requires fitting
measured wavelength-dependent absorption or Lee Fig. S9 data.
"""

from __future__ import annotations

from typing import Mapping, Optional

import numpy as np


H_PLANCK = 6.62607015e-34
C_LIGHT = 2.99792458e8
K_B_OVER_HC_CM1_K = 0.6950348004861274
REFERENCE_WAVELENGTH_NM = 1064.0

# Values deliberately preserved verbatim from the supplied script.
E_3H6_CM1 = np.array([
    0.0, 9.0, 54.0, 80.0, 122.0, 149.0, 187.0,
    223.0, 246.0, 296.0, 314.0, 405.0, 417.0,
])
E_3F4_CM1 = np.array([
    5634.0, 5661.0, 5689.0, 5717.0, 5730.0,
    5750.0, 5761.0, 5782.0, 5823.0,
])
E_3H5_CM1 = np.array([
    8207.0, 8218.0, 8267.0, 8288.0, 8317.0, 8348.0,
    8377.0, 8428.0, 8440.0, 8528.0, 8539.0,
])
E_3H4_CM1 = np.array([
    12546.0, 12566.0, 12588.0, 12613.0, 12642.0,
    12673.0, 12721.0, 12766.0, 12776.0,
])
E_3F3_CM1 = np.array([
    14279.0, 14317.0, 14341.0, 14361.0,
    14380.0, 14398.0, 14420.0,
])
E_3F2_CM1 = np.array([
    14936.0, 14951.0, 15018.0, 15061.0, 15084.0,
])

ENERGY_LEVELS = {
    "3H6": E_3H6_CM1,
    "3F4": E_3F4_CM1,
    "3H5": E_3H5_CM1,
    "3H4": E_3H4_CM1,
    "3F3": E_3F3_CM1,
    "3F2": E_3F2_CM1,
}

_CHANNELS = {
    "gsa_3f4": ("gsa", "3H6", "3F4"),
    "gsa_3h5": ("gsa", "3H6", "3H5"),
    "esa_3h4": ("esa", "3F4", "3H4"),
    "esa_3f3": ("esa", "3F4", "3F3"),
    "esa_3f2": ("esa", "3F4", "3F2"),
}


def _positive_finite(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and strictly positive.")
    return value


def boltzmann_populations(levels_cm1, temperature_k=300.0):
    """Normalized population of each resolved Stark state, within a multiplet."""
    temperature = _positive_finite(temperature_k, "temperature_k")
    levels = np.asarray(levels_cm1, dtype=float)
    if levels.ndim != 1 or not len(levels) or np.any(~np.isfinite(levels)):
        raise ValueError("Stark levels must be a nonempty finite 1D array.")
    weights = np.exp(-(levels - levels.min()) /
                     (K_B_OVER_HC_CM1_K * temperature))
    return weights / weights.sum()


def transition_energies(upper, lower):
    """All exact pair differences, flattened in upper-then-lower order."""
    delta = (np.asarray(upper)[:, None] - np.asarray(lower)[None, :]).ravel()
    if np.any(delta <= 0):
        raise ValueError("Every upper level must lie above every lower level.")
    return delta


class StarkAbsorptionModel:
    """Effective GSA/ESA spectra with exact anchoring at 1064 nm.

    ``rates(wavelength_nm, p)`` returns the two coefficients in
    s^-1 / (kW cm^-2), using ``p.sigma_gsa_m2`` and ``p.sigma_esa_m2`` as the
    1064-nm cross sections. ``diagnostics`` identifies all off-anchor values as
    exploratory; changing the spectrum cannot silently change the SI anchor.

    Optional ``band_weights`` keys are gsa_3f4, gsa_3h5, esa_3h4, esa_3f3 and
    esa_3f2. Optional ``oscillator_strengths`` uses the same keys, with arrays
    of shape (n_upper, n_lower). Its entries are relative line strengths and
    are multiplied by starting-state Boltzmann populations. They are not
    silently normalized; their relative integrated band strengths are retained.
    Without a matrix, each lower state couples equally to upper states with
    summed strength one. Default equal band strength is an assumption.
    """

    reference_wavelength_nm = REFERENCE_WAVELENGTH_NM
    reference = REFERENCE_WAVELENGTH_NM

    def __init__(self, *, temperature_k=300.0, narrow_sigma_cm1=8.5,
                 gsa_broad_sigma_cm1=1000.0, gsa_broad_fraction=0.5,
                 thermal_sideband_asymmetry=True,
                 band_weights: Optional[Mapping[str, float]] = None,
                 oscillator_strengths: Optional[Mapping[str, np.ndarray]] = None):
        self.temperature_k = _positive_finite(temperature_k, "temperature_k")
        self.narrow_sigma_cm1 = _positive_finite(narrow_sigma_cm1,
                                                "narrow_sigma_cm1")
        self.gsa_broad_sigma_cm1 = _positive_finite(gsa_broad_sigma_cm1,
                                                  "gsa_broad_sigma_cm1")
        self.gsa_broad_fraction = float(gsa_broad_fraction)
        if (not np.isfinite(self.gsa_broad_fraction) or
                not 0 <= self.gsa_broad_fraction <= 1):
            raise ValueError("gsa_broad_fraction must be between zero and one.")
        self.thermal_sideband_asymmetry = bool(thermal_sideband_asymmetry)
        self.band_weights = {name: 1.0 for name in _CHANNELS}
        supplied_weights = {} if band_weights is None else dict(band_weights)
        supplied_strengths = ({} if oscillator_strengths is None
                              else dict(oscillator_strengths))
        unknown = (supplied_weights.keys() | supplied_strengths.keys()) - _CHANNELS.keys()
        if unknown:
            raise ValueError(f"Unknown spectral channels: {sorted(unknown)}")
        self.band_weights.update(supplied_weights)
        self._transitions = {}
        for name, (role, lower_name, upper_name) in _CHANNELS.items():
            lower, upper = ENERGY_LEVELS[lower_name], ENERGY_LEVELS[upper_name]
            band_weight = float(self.band_weights[name])
            if not np.isfinite(band_weight) or band_weight < 0:
                raise ValueError(f"Band weight {name} must be finite and nonnegative.")
            self.band_weights[name] = band_weight
            shape = (len(upper), len(lower))
            oscillator = np.asarray(supplied_strengths.get(
                name, np.ones(shape) / len(upper)), dtype=float)
            if (oscillator.shape != shape or np.any(~np.isfinite(oscillator))
                    or np.any(oscillator < 0)):
                raise ValueError(f"Oscillator strengths for {name} must have shape {shape} "
                                 "and finite nonnegative entries.")
            weights = oscillator * boltzmann_populations(lower, self.temperature_k)[None, :]
            self._transitions[name] = (role, transition_energies(upper, lower),
                                       band_weight * weights.ravel())
        self._reference_density = {
            role: self._density(REFERENCE_WAVELENGTH_NM, role)
            for role in ("gsa", "esa")
        }
        for role, density in self._reference_density.items():
            if not np.isfinite(density) or density <= np.finfo(float).tiny:
                raise ValueError(f"The {role.upper()} spectrum has no usable 1064-nm "
                                 "absorption. Revise the explicit sideband/band assumptions.")

    def _density(self, wavelength_nm, role):
        wavelength = np.asarray(wavelength_nm, dtype=float)
        if np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0):
            raise ValueError("wavelength_nm must be finite and strictly positive.")
        energy = 1e7 / wavelength
        flat = energy.ravel()
        result = np.zeros(flat.shape, dtype=float)
        normalizer = np.sqrt(2 * np.pi)
        for channel_role, transitions, strengths in self._transitions.values():
            if channel_role != role or not np.any(strengths):
                continue
            for start in range(0, len(flat), 2048):
                delta = flat[start:start + 2048, None] - transitions[None, :]
                narrow = np.exp(-0.5 * (delta / self.narrow_sigma_cm1)**2)
                narrow /= normalizer * self.narrow_sigma_cm1
                profile = narrow
                if role == "gsa" and self.gsa_broad_fraction:
                    broad = np.exp(-0.5 * (delta / self.gsa_broad_sigma_cm1)**2)
                    broad /= normalizer * self.gsa_broad_sigma_cm1
                    if self.thermal_sideband_asymmetry:
                        deficit = np.maximum(-delta, 0.0)
                        broad *= np.exp(-deficit /
                                        (K_B_OVER_HC_CM1_K * self.temperature_k))
                    profile = ((1 - self.gsa_broad_fraction) * narrow
                               + self.gsa_broad_fraction * broad)
                result[start:start + 2048] += profile @ strengths
        result = result.reshape(energy.shape)
        return float(result) if result.ndim == 0 else result

    def factors(self, wavelength_nm, p=None):
        """Relative cross sections, each equal to one at exactly 1064 nm."""
        return tuple(self._density(wavelength_nm, role) / self._reference_density[role]
                     for role in ("gsa", "esa"))

    def cross_sections(self, wavelength_nm, p):
        """GSA and ESA cross sections in m^2, scalar or wavelength arrays."""
        gsa, esa = self.factors(wavelength_nm)
        sigma_g = _positive_finite(p.sigma_gsa_m2, "p.sigma_gsa_m2")
        sigma_e = _positive_finite(p.sigma_esa_m2, "p.sigma_esa_m2")
        return sigma_g * gsa, sigma_e * esa

    def rates(self, wavelength_nm, p):
        """Return (GSA, ESA) rate coefficients per kW cm^-2, not rates at I."""
        sigma_g, sigma_e = self.cross_sections(wavelength_nm, p)
        photon_flux_per_kwcm2 = 1e7 * np.asarray(wavelength_nm) * 1e-9 / (H_PLANCK * C_LIGHT)
        return sigma_g * photon_flux_per_kwcm2, sigma_e * photon_flux_per_kwcm2

    def diagnostics(self, wavelength_nm, p):
        """Scalar audit record; only the exact 1064-nm point is SI-anchored."""
        wavelength = _positive_finite(wavelength_nm, "wavelength_nm")
        fg, fe = self.factors(wavelength)
        sg, se = self.cross_sections(wavelength, p)
        wg, we = self.rates(wavelength, p)
        anchored = bool(np.isclose(wavelength, REFERENCE_WAVELENGTH_NM,
                                   rtol=0, atol=1e-9))
        si_values = (np.isclose(p.sigma_gsa_m2, 6.0e-29, rtol=1e-12, atol=0)
                     and np.isclose(p.sigma_esa_m2, 6.4e-25, rtol=1e-12, atol=0))
        status = ("SI Table S6/S8 anchor" if anchored and si_values
                  else "user-supplied 1064-nm anchor" if anchored
                  else "exploratory; not calibrated by Lee et al.")
        return {
            "wavelength_nm": wavelength,
            "reference_wavelength_nm": REFERENCE_WAVELENGTH_NM,
            "source_status": status,
            "is_si_calibrated": bool(anchored and si_values),
            "sigma_gsa_m2": float(sg), "sigma_esa_m2": float(se),
            "gsa_factor": float(fg), "esa_factor": float(fe),
            "w_gsa_per_kwcm2": float(wg), "w_esa_per_kwcm2": float(we),
            "esa_gsa_ratio": float(se / sg) if sg > 0 else float("inf"),
            "temperature_k": self.temperature_k,
            "narrow_sigma_cm1": self.narrow_sigma_cm1,
            "gsa_broad_sigma_cm1": self.gsa_broad_sigma_cm1,
            "gsa_broad_fraction": self.gsa_broad_fraction,
            "thermal_sideband_asymmetry": self.thermal_sideband_asymmetry,
            "band_weights": self.band_weights.copy(),
            "level_source": "Conrad et al. (2025), Table 4(B), DOI 10.1021/acs.jpcc.5c06209",
        }

    def get_coefficients(self, wavelength_nm, p):
        """Named coefficient/factor/provenance record; ``rates`` gives a pair."""
        return self.diagnostics(wavelength_nm, p)

