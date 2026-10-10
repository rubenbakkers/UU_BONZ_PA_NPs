# -*- coding: utf-8 -*-
"""
Tm3+ in NaYF4: broadened 3H6 -> 3H5 GSA, without GSA or ESA backgrounds.

NEW DEFAULT: ABSORPTION_MODEL="broad_gsa". Only the 3H6 -> 3H5
channel has Gaussian sigma GSA_H5_SIGMA_CM1=300 cm^-1. Its own
line tails provide the off-resonant GSA at 1064 nm. Both background
weights are zero and the additive background term is removed. ESA
and the other GSA channel retain their 8.5 cm^-1 Gaussian width;
all Stark energies and Boltzmann weights are retained. The SI anchor at
1064 nm is preserved. The broadened linewidth is a model choice,
not a measured linewidth from Lee et al.
The original stark, si_1064 and hybrid modes remain selectable.

ABSORPTIEMODEL (schakelaar ABSORPTION_MODEL, precies een tegelijk actief):
  "broad_gsa": brede H5-GSA-lijnen; GSA en ESA zonder achtergrond,
               met 1064-nm-anker. Dit is de standaard in dit bestand.
  "stark"   : alleen Stark-overgangen (sticks + Boltzmann + convolutie).
              Geen brede achtergrond, geen 1064-nm-anker. GSA bij 1064 nm en
              1390-1490 nm is dan vrijwel nul: verwacht "no avalanche".
  "si_1064" : alleen de Gauss-lineshape uit pa_model, met de SI-waarden
              als amplitudes. Deze route is op de Gauss-pieken genormeerd,
              niet op 1064 nm. Geen Stark-structuur.
  "hybrid"  : oorspronkelijke smalle Stark-lijnen met 1064-nm-anker,
              nu eveneens zonder achtergrond. Bij de standaard smalle
              GSA-lijnen is de referentie te klein; de kalibratiecontrole
              wijst deze instelling af. Gebruik standaard "broad_gsa".

Kinetics: Lee et al. SI Eqs S1-S6, Tables S4, S6, S7 (sample 3, 8% Tm).
In the anchored broad_gsa and hybrid models, at 1064 nm the effective cross sections
are exactly the SI Table S6 values:
  sigma_GSA = 6.0e-29 m^2, sigma_ESA = 6.4e-25 m^2, ratio = 10666.7.
Table S8 explicitly identifies 1064 nm as the ESA reference wavelength.
Shapes are normalized AT THIS REFERENCE, not at their global maximum;
adding another band therefore cannot silently rescale the reference rates.

Stark positions: Physical Chemistry.pdf, page 9, Table 4(B), ab initio values.
GSA channels: 3H6 -> 3F4/3H5. ESA channels: 3F4 -> 3H4/3F3/3F2.
The SI three-population approximation is retained, with rapid feeding from
3H5 to 3F4 and from 3F2/3F3 to 3H4. Emission, relaxation and energy transfer
remain in the rate equations, not in the pump absorption spectrum.

Within each absorbing multiplet, transitions are weighted by the Boltzmann
population of the LOWER Stark sublevel at TEMPERATURE_K (fast intra-
multiplet thermalization assumed), exactly as in the standalone Tm3+
Stark-level transition-spectrum script: for each channel the (lower,
upper) sublevel pairs are enumerated explicitly, p_i = Boltzmann
occupation of lower sublevel i (conditional on being in that multiplet),
and each pair contributes a stick of weight p_i * r_ij at wavenumber
DeltaE_ij, with r_ij defaulting to 1/(number of upper sublevels) when no
independent relative line strengths are available. BOLTZMANN_MODE switches
this weighting on ("multiplet") or off (equal weights).

LOCAL THERMAL EQUILIBRIUM. The three-population vector is [N0, N1, N2],
with N0 = total 3H6 population, N1 = total 3F4 population and N2 = total
3H4 population. At the fixed temperature of 300 K the Stark populations are
    N_H6,i(t) = N0(t) * p_H6,i(T),
    N_F4,i(t) = N1(t) * p_F4,i(T).
Ions fed into 3F4 by relaxation or cross-relaxation redistribute within
that multiplet before further excitation. This local-equilibrium assumption
requires intra-multiplet thermalization to be faster than further excitation
AND depletion (including decay and transfer); the same assumption applies
to 3H6. No thermalization time or additional decay rate is introduced.
Relaxation changes total N1, which is never separately normalized to one.
Only the conditional p_i sum to one. For a fixed temperature these p_i are
cached, and the already-weighted effective spectra enter the pump terms
R_gsa*N0 and R_esa*N1 without another Boltzmann factor. The "off" mode is an
equal-weight comparison, rather than the local-thermal-equilibrium model.

CONVOLUTION. For both GSA and ESA, zero-Lorentz convolution is evaluated
as the exact weighted Gaussian sum, with unit-integral kernels; this
avoids FFT round-off in the far wings without adding a background.
With nonzero Lorentz width, each channel is convolved explicitly
(on a uniform wavenumber grid, via the convolution theorem) with the line
shape:
    channel shape = sticks (*) Gauss(sigma) (*) Lorentz(HWHM)
Gauss = inhomogeneous broadening (STARK_SIGMA_CM1, quadrature-added to an
optional instrument resolution INSTRUMENT_SIGMA_CM1); Lorentz = homogeneous
broadening (LORENTZ_HWHM_CM1). Both together give a Voigt profile. With the
defaults (Lorentz HWHM = 0, no instrument term) the result equals the earlier
Gaussian sum. Each channel keeps unit integral over wavenumber.

In de "hybrid"-stand is een brede, zwakke effectieve GSA-bijdrage aanwezig:
pure smalle Stark-staarten modelleren de fononondersteunde GSA van Lee niet.
Widths, mixing fractions and relative band weights remain phenomenological,
NOT SI-fitted parameters. The GSA envelope center is corrected using
10^7/1064 - 1200 cm^-1, approximately 1220 nm, consistent with 3H6 -> 3H5.
The existing Stark energies are retained; they already place this band
near 1200 nm. This approximate envelope center is not a measured line.
Equal band weights are not derived from the SI Judd-Ofelt table
(which lacks the F2/F3 elements).

The solver checks population stationarity and extends integration as needed.
No intensity smoothing or clipping of large slopes is used to shape curves.
The 1450-nm band is exploratory: its strength is not independently fitted.
The default scan is 850-1180 nm; set INCLUDE_EXTRAPOLATED_WAVELENGTHS=True
for the longer-wave scan, whose slopes are not quantitative predictions.
"""

from __future__ import annotations

import os  # Toegevoegd voor het opslaan in de specifieke map
import contextlib
from functools import lru_cache
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

import pa_model
from pa_model import (
    emission_from_population,
    DEFAULT_LAMBDA0_ESA_NM,
    DEFAULT_SIGMA_LAMBDA_ESA_NM,
    parameters_for_sample,
    power_dependence,
    nonlinearity_slope,
    nonlinearity_slope_max_converged,
)

_LOG_PLOT_FLOOR = 1e-300
# Ondergrens van alleen de absorptiegrafiek; geen vloer in het spectrum.
ABSORPTION_PLOT_MIN_FACTOR = 1e-20

# A real avalanche has a sharp knee with s_max of order 10-100 (about 40 for
# the 8% Tm sample here). Without an avalanche the emission is a pure power
# law with s ~ 2-3. Anything in between is a safe cut-off.
MIN_AVALANCHE_SLOPE = 10.0

# The knee must lie at least this many grid points away from the edges of the
# intensity window; otherwise the transition is (partly) outside the window
# and a broken-line fit is not possible.
KNEE_EDGE_MARGIN = 3

# Adaptive intensity scan (see adaptive_knee_scan):
#   N_COARSE : sparse log-spaced points over the whole intensity window
#   N_KNEE   : dense points around the coarse knee (+/- 2 coarse cells)
#   N_POLISH : very dense points around the dense-scan knee (+/- 2 cells)
N_COARSE = 30
N_KNEE = 40
N_POLISH = 40

# Fast steady-state solver (see docstring, version 5).
# Longer wavelengths lack independent absorption calibration in this model.
# Keep all transitions in the spectra, but opt in explicitly to extrapolation.
INCLUDE_EXTRAPOLATED_WAVELENGTHS = False
T_STEADY_S = 20.0     # initial integration duration [s]
MAX_STEADY_TIME_S = 1280.0
STEADY_RELATIVE_TOL = 1e-6
STEADY_POPULATION_FLOOR = 1e-18

# Choose the widest near-linear window around EACH curve's steepest rise.
# These are explicit fitting criteria, not physical model parameters.
SLOPE_FRACTION = 0.25  # None: automatic; set e.g. 0.9 for a fixed-window comparison.
FIT_MAX_ERROR_DEX = 0.03  # at most about 7.2% vertical discrepancy
FIT_MIN_R2 = 0.999
FIT_MIN_PEAK_RATIO = 0.95  # fitted slope at least 95% of local peak
FIT_MIN_POINTS = 7
FIT_MIN_SPAN_DEX = 0.5  # at least a factor sqrt(10) in emission
LINE_EXTEND_DECADES = 0.0  # draw only where the line was verified


def _safe_for_log(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), _LOG_PLOT_FLOOR, None)


# ---------------------------------------------------------------------------
# Stark-level parameters from Table 4 [cm^-1]
# ---------------------------------------------------------------------------

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
    12546.0,
    12566.0,
    12588.0,
    12613.0,
    12642.0,
    12673.0,
    12721.0,
    12766.0,
    12776.0,
])

E_3F3_CM1 = np.array([
    14279.0,
    14317.0,
    14341.0,
    14361.0,
    14380.0,
    14398.0,
    14420.0,
])

E_3F2_CM1 = np.array([
    14936.0,
    14951.0,
    15018.0,
    15061.0,
    15084.0,
])

# ---------------------------------------------------------------------------
# ABSORPTIEMODEL: precies een van de vier tegelijk actief
#   "broad_gsa": GSA H5 sigma300, GSA/ESA zonder achtergrond, anker behouden
#   "stark"   : alleen Stark-sticks (geen achtergrond, geen 1064-nm-anker)
#   "si_1064" : Gauss-lineshape met SI-amplitudes (piekgenormeerd)
#   "hybrid"  : smalle Stark-lijnen + 1064-nm-anker; geen achtergrond
# ---------------------------------------------------------------------------
ABSORPTION_MODEL = "broad_gsa"  # "broad_gsa", "stark", "si_1064", "hybrid"

if ABSORPTION_MODEL not in ("stark", "si_1064", "hybrid", "broad_gsa"):
    raise ValueError("Unknown ABSORPTION_MODEL; use broad_gsa, stark, si_1064 or hybrid.")

_USE_STARK = ABSORPTION_MODEL in ("stark", "hybrid", "broad_gsa")   # Stark-spectra gebruiken?
ABSORPTION_KIND = "stark_gaussian" if _USE_STARK else "gaussian"
ABSORPTION_LABEL = {
    "stark": "Stark GSA + ESA (geen achtergrond)",
    "si_1064": "Gaussian met SI-amplitudes (piekgenormeerd)",
    "hybrid": "Stark zonder achtergrond + 1064-nm-anker",
    "broad_gsa": "Verbrede GSA + ESA zonder achtergrond + 1064-nm-anker",
}[ABSORPTION_MODEL]

# Explicit model parameters (not linewidths/strengths supplied by Table 4).
STARK_SIGMA_CM1 = 8.5  # ESA and unmodified GSA channels
GSA_H5_SIGMA_CM1 = 300.0  # Gaussian sigma of 3H6 -> 3H5 only, cm^-1
ESA_WEIGHT_3H4 = 1.0
ESA_WEIGHT_3F3 = 1.0
ESA_WEIGHT_3F2 = 1.0
GSA_WEIGHT_3F4 = 1.0
GSA_WEIGHT_3H5 = 1.0

# Beide aparte achtergronden zijn volledig verwijderd uit StarkSpectrum.
# Niet-nulgewichten worden afgewezen; deze waarden rapporteren de instelling.
ESA_BACKGROUND_WEIGHT = 0.0
GSA_BACKGROUND_WEIGHT = 0.0

# SI-anker (1064 nm): in "hybrid" en "broad_gsa". In "si_1064" gebruikt
# pa_model piekgenormeerde Gauss-factoren; in "stark" is er geen anker.
SI_REFERENCE_ANCHOR = ABSORPTION_MODEL in ("hybrid", "broad_gsa")
REFERENCE_WAVELENGTH_NM = 1064.0
# Historische namen: alleen normeringsgrid en Gauss-envelope in si_1064.
# Deze centra/breedtes voegen geen achtergrond toe aan de Stark-spectra.
ESA_BACKGROUND_LAMBDA0_NM = DEFAULT_LAMBDA0_ESA_NM
ESA_BACKGROUND_SIGMA_NM = DEFAULT_SIGMA_LAMBDA_ESA_NM
# 3H5 lies BELOW the photon energy at 1064 nm: subtract the mismatch.
# Lee et al., Methods and Fig. 1c: approximately 1200 cm^-1 mismatch.
GSA_ENERGY_MISMATCH_CM1 = 1200.0
GSA_BACKGROUND_LAMBDA0_NM = 1e7 / (
    1e7 / REFERENCE_WAVELENGTH_NM - GSA_ENERGY_MISMATCH_CM1
)  # 1219.7 nm; same corrected center in all absorption modes
GSA_BACKGROUND_SIGMA_NM = pa_model.DEFAULT_SIGMA_LAMBDA_GSA_NM

# Host and fixed intra-multiplet thermalization temperature. The supplied
# Stark energies, linewidths and rates are retained for this NaYF4 model.
HOST_MATERIAL = "NaYF4"
# Fast intra-multiplet thermalization temperature for the Boltzmann-weighted
# transition strengths below (population of the lower Stark sublevel within
# its multiplet), following the standalone Stark-transition-spectrum model.
TEMPERATURE_K = 300.0
K_B_CM1_PER_K = 0.6950348005  # Boltzmann constant, cm^-1/K

# Boltzmann weighting of the lower Stark sublevels:
#   "multiplet" -> p_i ~ exp[-(E_i - E_min)/(kB T)] within the lower multiplet
#   "off"       -> all lower sublevels equal weight
BOLTZMANN_MODE = "multiplet"

# Line broadening applied by explicit convolution of the stick spectrum.
# STARK_SIGMA_CM1 (above) is the Gaussian sigma. These two are extra:
LORENTZ_HWHM_CM1 = 0.0       # homogeneous (Lorentz) HWHM; 0 = pure Gauss
INSTRUMENT_SIGMA_CM1 = 0.0   # extra Gaussian (resolution), added in quadrature

# Numerical grid for the convolution (uniform in wavenumber).
CONV_GRID_STEP_CM1 = 0.5     # grid step [cm^-1]
CONV_GRID_PAD_CM1 = 8000.0   # margin around the sticks [cm^-1]


@lru_cache(maxsize=64)
def _cached_boltzmann_within(levels_tuple, temperature_k):
    """Immutable probabilities cached by energies and temperature."""
    levels = np.asarray(levels_tuple, dtype=float)
    p = np.exp(-(levels - levels.min()) / (K_B_CM1_PER_K * temperature_k))
    return tuple(p / p.sum())


def boltzmann_within(levels_cm1, temperature_k=TEMPERATURE_K):
    """Conditional occupation of the sublevels of ONE multiplet at T.

    p_i proportional to exp(-(E_i - E_min)/(kB*T)), normalized to sum 1.
    This is a population WITHIN the multiplet (given the ion is already
    there), not an absolute population of that multiplet. Each multiplet
    uses its OWN minimum, so the absolute energy offset of 3F4 cancels.
    Repeated calls at fixed energies and T reuse the cached probabilities;
    each call returns a fresh array so callers cannot modify the cache.
    """
    if not np.isfinite(temperature_k) or temperature_k <= 0:
        raise ValueError("TEMPERATURE_K must be finite and > 0 K.")
    levels = np.asarray(levels_cm1, dtype=float)
    if levels.ndim != 1 or levels.size == 0:
        raise ValueError("levels_cm1 must be a nonempty 1D array.")
    if not np.all(np.isfinite(levels)):
        raise ValueError("levels_cm1 must contain finite energies.")
    return np.asarray(_cached_boltzmann_within(tuple(levels), float(temperature_k)))


def level_populations(levels_cm1, temperature_k=TEMPERATURE_K):
    """Lower-sublevel weights according to BOLTZMANN_MODE (sum = 1)."""
    levels = np.asarray(levels_cm1, dtype=float)
    if BOLTZMANN_MODE == "multiplet":
        return boltzmann_within(levels, temperature_k)
    if BOLTZMANN_MODE == "off":
        if levels.ndim != 1 or levels.size == 0:
            raise ValueError("levels_cm1 must be a nonempty 1D array.")
        if not np.all(np.isfinite(levels)):
            raise ValueError("levels_cm1 must contain finite energies.")
        return np.full(levels.size, 1.0 / levels.size)
    raise ValueError("BOLTZMANN_MODE must be 'multiplet' or 'off'.")


# Two separately normalized physical Boltzmann distributions, computed once
# at the fixed temperature. These are conditional fractions, not N0 or N1.
P_3H6_BOLTZMANN = boltzmann_within(E_3H6_CM1)
P_3F4_BOLTZMANN = boltzmann_within(E_3F4_CM1)
P_3H6_BOLTZMANN.setflags(write=False)
P_3F4_BOLTZMANN.setflags(write=False)


def stark_populations(population_vector, temperature_k=TEMPERATURE_K):
    """Return (N_H6_i, N_F4_i) from [N0, N1, N2], without normalization.

    N0 = total 3H6, N1 = total 3F4, N2 = total 3H4, matching pa_model.
    For BOLTZMANN_MODE="multiplet": N_H6_i=N0*p_H6_i, N_F4_i=N1*p_F4_i.
    For "off", both use the same equal lower-level weights as the spectra.
    Relaxation/cross-relaxation change N1; rapid redistribution only changes
    how that total is partitioned. Temperature must remain fixed for this
    model; the cached probabilities need no integration-step update.

    Accepts a vector of shape (3,) or trajectories of shape (..., 3).
    The outputs have shape (..., 13) and (..., 9), respectively. For a
    solve_ivp result use stark_populations(sol.y.T); for the RK4 trajectory
    use stark_populations(y_arr). The input totals and units are preserved.
    """
    populations = np.asarray(population_vector, dtype=float)
    if populations.ndim == 0 or populations.shape[-1] != 3:
        raise ValueError("population_vector must have shape (..., 3): [N0, N1, N2].")
    if not np.all(np.isfinite(populations)):
        raise ValueError("population_vector must contain finite populations.")
    p_h6 = level_populations(E_3H6_CM1, temperature_k)
    p_f4 = level_populations(E_3F4_CM1, temperature_k)
    return (populations[..., 0, None] * p_h6,
            populations[..., 1, None] * p_f4)


def _broadening():
    """Validated total Gauss sigma and Lorentz HWHM in cm^-1."""
    values = (STARK_SIGMA_CM1, LORENTZ_HWHM_CM1, INSTRUMENT_SIGMA_CM1)
    if not all(np.isfinite(v) and v >= 0 for v in values):
        raise ValueError("Line widths must be finite and nonnegative.")
    sigma_tot = float(np.hypot(STARK_SIGMA_CM1, INSTRUMENT_SIGMA_CM1))
    hwhm = float(LORENTZ_HWHM_CM1)
    if sigma_tot < CONV_GRID_STEP_CM1 and hwhm < CONV_GRID_STEP_CM1:
        raise ValueError("Line width is smaller than CONV_GRID_STEP_CM1; "
                         "decrease the grid step.")
    return sigma_tot, hwhm


def convolve_sticks(sticks_cm1, weights, wavenumber_cm1, sigma_cm1, hwhm_cm1):
    """Convolve sticks with Gauss(sigma) and Lorentz(HWHM); evaluate at nu.

    Done on a uniform grid via the convolution theorem:
      Gauss (sigma):  FT = exp(-2 pi^2 sigma^2 f^2)
      Lorentz (HWHM): FT = exp(-2 pi HWHM |f|)
    Lorentz tails are not truncated (apart from periodic wrap-around at the
    grid edges, negligible thanks to CONV_GRID_PAD_CM1). Each stick is split
    linearly over two grid points. The integral of the result is
    sum(weights). Evaluation points outside the grid (more than
    CONV_GRID_PAD_CM1 away from any stick) return 0.
    """
    sticks = np.asarray(sticks_cm1, dtype=float)
    weights = np.asarray(weights, dtype=float)
    nu = np.asarray(wavenumber_cm1, dtype=float)
    lo = sticks.min() - CONV_GRID_PAD_CM1
    hi = sticks.max() + CONV_GRID_PAD_CM1
    n = int(np.ceil((hi - lo) / CONV_GRID_STEP_CM1)) + 2
    n = 1 << (n - 1).bit_length()  # power of 2: fast FFT
    grid = lo + CONV_GRID_STEP_CM1 * np.arange(n)

    binned = np.zeros(n)
    pos = (sticks - lo) / CONV_GRID_STEP_CM1
    i0 = np.floor(pos).astype(int)
    frac = pos - i0
    np.add.at(binned, i0, weights * (1.0 - frac))
    np.add.at(binned, i0 + 1, weights * frac)

    f = np.fft.rfftfreq(n, d=CONV_GRID_STEP_CM1)
    kernel = (np.exp(-2.0 * np.pi**2 * sigma_cm1**2 * f**2)
              * np.exp(-2.0 * np.pi * hwhm_cm1 * f))
    density = np.fft.irfft(np.fft.rfft(binned) * kernel, n) / CONV_GRID_STEP_CM1
    density = np.maximum(density, 0.0)  # numerical noise around zero
    return np.interp(nu, grid, density, left=0.0, right=0.0)


def _stark_gaussian_unnormalized(energy_cm1, levels_cm1, sigma_cm1, amplitudes=None):
    """Gaussian sum in cm^-1; chunk arrays to bound memory usage."""
    energy = np.asarray(energy_cm1, dtype=float)
    levels = np.asarray(levels_cm1, dtype=float)
    if not np.isfinite(sigma_cm1) or sigma_cm1 <= 0:
        raise ValueError("sigma_cm1 must be finite and > 0.")
    amps = np.ones_like(levels) if amplitudes is None else np.asarray(amplitudes)
    if levels.ndim != 1 or not len(levels) or amps.shape != levels.shape:
        raise ValueError("Levels and amplitudes must be matching nonempty vectors.")
    flat = energy.ravel()
    result = np.empty_like(flat)
    for start in range(0, flat.size, 2048):
        delta = flat[start:start + 2048, None] - levels
        result[start:start + 2048] = np.exp(-0.5 * (delta / sigma_cm1)**2) @ amps
    return float(result[0]) if energy.ndim == 0 else result.reshape(energy.shape)


A_3H6 = np.ones(len(E_3H6_CM1))

def S_3H6(energy_cm1, sigma_cm1=STARK_SIGMA_CM1, amplitudes=None):
    """Absolute 3H6 level positions; not transition photon energies."""
    return _stark_gaussian_unnormalized(
        energy_cm1, E_3H6_CM1, sigma_cm1,
        A_3H6 if amplitudes is None else amplitudes)


A_3F4 = np.ones(len(E_3F4_CM1))

def S_3F4(energy_cm1, sigma_cm1=STARK_SIGMA_CM1, amplitudes=None):
    """Absolute 3F4 level positions; not transition photon energies."""
    return _stark_gaussian_unnormalized(
        energy_cm1, E_3F4_CM1, sigma_cm1,
        A_3F4 if amplitudes is None else amplitudes)


A_3H5 = np.ones(len(E_3H5_CM1))

def S_3H5(energy_cm1, sigma_cm1=STARK_SIGMA_CM1, amplitudes=None):
    """Absolute 3H5 level positions; not transition photon energies."""
    return _stark_gaussian_unnormalized(
        energy_cm1, E_3H5_CM1, sigma_cm1,
        A_3H5 if amplitudes is None else amplitudes)


A_3H4 = np.ones(len(E_3H4_CM1))

def S_3H4(energy_cm1, sigma_cm1=STARK_SIGMA_CM1, amplitudes=None):
    """Absolute 3H4 level positions; not transition photon energies."""
    return _stark_gaussian_unnormalized(
        energy_cm1, E_3H4_CM1, sigma_cm1,
        A_3H4 if amplitudes is None else amplitudes)


A_3F3 = np.ones(len(E_3F3_CM1))

def S_3F3(energy_cm1, sigma_cm1=STARK_SIGMA_CM1, amplitudes=None):
    """Absolute 3F3 level positions; not transition photon energies."""
    return _stark_gaussian_unnormalized(
        energy_cm1, E_3F3_CM1, sigma_cm1,
        A_3F3 if amplitudes is None else amplitudes)


A_3F2 = np.ones(len(E_3F2_CM1))

def S_3F2(energy_cm1, sigma_cm1=STARK_SIGMA_CM1, amplitudes=None):
    """Absolute 3F2 level positions; not transition photon energies."""
    return _stark_gaussian_unnormalized(
        energy_cm1, E_3F2_CM1, sigma_cm1,
        A_3F2 if amplitudes is None else amplitudes)


def transition_energies(upper, lower):
    """All pair differences, with upper and lower arrays in cm^-1."""
    differences = (upper[:, None] - lower[None, :]).ravel()
    if np.any(differences <= 0):
        raise ValueError("Upper levels must lie above all lower levels.")
    return differences


def transition_gaps(lower_cm1, upper_cm1):
    """(n_lower, n_upper) matrix of upper-lower sublevel gaps, in cm^-1."""
    lower = np.asarray(lower_cm1, dtype=float)
    upper = np.asarray(upper_cm1, dtype=float)
    gaps = upper[None, :] - lower[:, None]
    if np.any(gaps <= 0):
        raise ValueError("Upper levels must lie above all lower levels.")
    return gaps


def transition_sticks(lower_cm1, upper_cm1, temperature_k=TEMPERATURE_K,
                      relative_lines=None):
    """Return (gap_ij, weight_ij) with weight_ij = p_lower_i * r_ij.

    The upper multiplet receives NO Boltzmann factor. Default r_ij=1/n_upper
    gives unit total weight per channel and row sums equal to p_lower_i.
    Custom relative strengths are retained as supplied, without rescaling.
    """
    gaps = transition_gaps(lower_cm1, upper_cm1)
    p = level_populations(lower_cm1, temperature_k)
    if relative_lines is None:
        relative = np.full(gaps.shape, 1.0 / gaps.shape[1])
    else:
        relative = np.asarray(relative_lines, dtype=float)
        if relative.shape != gaps.shape or not np.all(np.isfinite(relative)):
            raise ValueError("relative_lines has the wrong shape or contains non-finite values.")
        if np.any(relative < 0) or not np.any(relative > 0):
            raise ValueError("relative_lines must be nonnegative, with at least one nonzero entry.")
    return gaps, p[:, None] * relative


GSA_TRANSITIONS_3F4_CM1 = transition_energies(E_3F4_CM1, E_3H6_CM1)
GSA_TRANSITIONS_3H5_CM1 = transition_energies(E_3H5_CM1, E_3H6_CM1)
ESA_TRANSITIONS_3H4_CM1 = transition_energies(E_3H4_CM1, E_3F4_CM1)
ESA_TRANSITIONS_3F3_CM1 = transition_energies(E_3F3_CM1, E_3F4_CM1)
ESA_TRANSITIONS_3F2_CM1 = transition_energies(E_3F2_CM1, E_3F4_CM1)

# Inspection data for non-pump processes in Fig. 1c. Their rates remain in
# pa_model: energy gaps alone do not determine lifetimes or branching ratios.
EMISSION_3H4_3H6_CM1 = transition_energies(E_3H4_CM1, E_3H6_CM1)
RELAXATION_GAPS_CM1 = {
    "3F2 -> 3H4": transition_energies(E_3F2_CM1, E_3H4_CM1),
    "3F3 -> 3H4": transition_energies(E_3F3_CM1, E_3H4_CM1),
    "3H4 -> 3H5": transition_energies(E_3H4_CM1, E_3H5_CM1),
    "3H5 -> 3F4": transition_energies(E_3H5_CM1, E_3F4_CM1),
    "3F4 -> 3H6": GSA_TRANSITIONS_3F4_CM1,
}
# Donor release minus acceptor demand for 3H4+3H6 -> 3F4+3F4.
CR_MISMATCH_CM1 = (ESA_TRANSITIONS_3H4_CM1[:, None]
                   - GSA_TRANSITIONS_3F4_CM1[None, :]).ravel()

# GSA/ESA pump channels, now carried as (lower_levels, upper_levels, weight)
# rather than pre-flattened transition energies, so that each channel's
# Boltzmann-weighted Stark structure (see StarkSpectrum.channel_shape) can be
# computed from the individual lower/upper sublevel arrays.
GSA_CHANNELS = {
    "3H6 -> 3F4": (E_3H6_CM1, E_3F4_CM1, GSA_WEIGHT_3F4),
    "3H6 -> 3H5": (E_3H6_CM1, E_3H5_CM1, GSA_WEIGHT_3H5),
}
ESA_CHANNELS = {
    "3F4 -> 3H4": (E_3F4_CM1, E_3H4_CM1, ESA_WEIGHT_3H4),
    "3F4 -> 3F3": (E_3F4_CM1, E_3F3_CM1, ESA_WEIGHT_3F3),
    "3F4 -> 3F2": (E_3F4_CM1, E_3F2_CM1, ESA_WEIGHT_3F2),
}


class StarkSpectrum:
    """Globally peak-normalized Stark spectrum, without a separate background.

    Each channel's Stark structure is the stick spectrum of all (lower,
    upper) sublevel pairs, with weight p_i * r_ij (p_i = Boltzmann
    population of lower sublevel i at TEMPERATURE_K, conditional on being in
    that multiplet; r_ij = relative line strength, uniform 1/n_upper by
    default), convolved explicitly with Gauss(sigma) and Lorentz(HWHM)
    (see convolve_sticks and channel_shape).
    """

    def __init__(self, channels, background_weight, center_nm, width_nm,
                 channel_sigmas_cm1=None, analytic_gaussian=False):
        self.channels = channels
        self.analytic_gaussian = bool(analytic_gaussian)
        self.channel_sigmas_cm1 = dict(channel_sigmas_cm1 or {})
        if set(self.channel_sigmas_cm1) - set(channels):
            raise ValueError("Unknown channel in channel_sigmas_cm1.")
        for width in self.channel_sigmas_cm1.values():
            if not np.isfinite(width) or width <= 0:
                raise ValueError("Channel Gaussian sigmas must be positive and finite.")
        weights = np.array([w for _, _, w in channels.values()])
        if not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("Channel weights must be finite, nonnegative and nonzero in total.")
        if not np.isfinite(background_weight) or background_weight != 0.0:
            raise ValueError("GSA and ESA backgrounds are removed; background_weight must be 0.")
        if not np.isfinite(center_nm) or not np.isfinite(width_nm) or min(center_nm, width_nm) <= 0:
            raise ValueError("Normalization-grid center and width must be finite and positive.")
        if not np.isfinite(STARK_SIGMA_CM1) or STARK_SIGMA_CM1 <= 0:
            raise ValueError("STARK_SIGMA_CM1 must be finite and positive.")
        sigma_tot, hwhm = _broadening()
        self.weight_sum = weights.sum()
        self.background_weight = background_weight
        self.center_nm, self.width_nm = center_nm, width_nm
        transitions = np.concatenate([
            transition_gaps(lower, upper).ravel()
            for lower, upper, w in channels.values() if w > 0
        ])
        channel_widths = [np.hypot(self.channel_sigmas_cm1.get(name, STARK_SIGMA_CM1),
                                    INSTRUMENT_SIGMA_CM1)
                          for name, (_, _, weight) in channels.items() if weight > 0]
        minimum_sigma = min(channel_widths)
        maximum_sigma = max(channel_widths)
        margin = 10 * maximum_sigma + 20 * hwhm
        lo = max(1.0, transitions.min() - margin)
        hi = transitions.max() + margin
        # At least 20 samples per Gauss sigma; resolve even a width of 8.5 cm^-1.
        energy = np.linspace(lo, hi, max(20001, int((hi-lo)/minimum_sigma*20)+1))
        self.stark_norm = float(np.max(self.raw_stark(energy)))
        wavelength_nm = np.unique(np.concatenate([
            1e7 / energy,
            np.linspace(max(1.0, center_nm - 10*width_nm), center_nm + 10*width_nm, 20001)
        ]))
        self.norm = float(np.max(self.mixed(wavelength_nm * 1e-9)))
        if not np.isfinite(self.norm) or self.norm <= 0:
            raise ValueError("Invalid spectrum normalization.")

    def channel_shape(self, energy_cm1, lower_cm1, upper_cm1,
                      temperature_k=TEMPERATURE_K, relative_lines=None, sigma_cm1=None):
        """Sticks (p_i * r_ij at DeltaE_ij) convolved with Gauss and Lorentz.

        p_i = level_populations(lower_cm1, temperature_k): the weight of
        lower sublevel i (Boltzmann within the multiplet, or equal weights
        for BOLTZMANN_MODE="off"). r_ij defaults to 1/n_upper (equal weight
        among the reachable upper sublevels for each lower sublevel) when
        relative_lines is None. The convolution kernel has unit integral
        over wavenumber, so with DEFAULT relative strengths the channel's
        total stick weight is sum_ij p_i r_ij = 1 regardless of temperature,
        sublevel count or line widths. Custom strengths keep their supplied
        total weight. The upper multiplet is never Boltzmann-weighted.
        """
        energy = np.asarray(energy_cm1, dtype=float)
        if not np.all(np.isfinite(energy)):
            raise ValueError("energy_cm1 must be finite.")
        gaps, weights = transition_sticks(lower_cm1, upper_cm1,
                                         temperature_k, relative_lines)
        sigma_tot, hwhm = _broadening()
        if sigma_cm1 is not None:
            if not np.isfinite(sigma_cm1) or sigma_cm1 <= 0:
                raise ValueError("Channel sigma must be positive and finite.")
            sigma_tot = float(np.hypot(sigma_cm1, INSTRUMENT_SIGMA_CM1))
        if self.analytic_gaussian and hwhm == 0:
            # Exact Gaussian stick convolution; avoids FFT round-off in far tails.
            return (_stark_gaussian_unnormalized(
                energy, gaps.ravel(), sigma_tot, weights.ravel())
                / (np.sqrt(2.0*np.pi)*sigma_tot))
        shape = convolve_sticks(gaps.ravel(), weights.ravel(),
                                energy.ravel(), sigma_tot, hwhm)
        return shape.reshape(energy.shape)

    def raw_stark(self, energy):
        value = np.zeros_like(np.asarray(energy, dtype=float))
        for name, (lower, upper, weight) in self.channels.items():
            if weight:
                value = value + (weight/self.weight_sum) * self.channel_shape(
                    energy, lower, upper, sigma_cm1=self.channel_sigmas_cm1.get(name))
        return value

    def mixed(self, wavelength_m):
        wavelength = np.asarray(wavelength_m, dtype=float)
        if np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0):
            raise ValueError("Wavelength must be finite and positive.")
        stark = self.raw_stark(1e-2/wavelength) / self.stark_norm
        # Alleen lijnprofielen: geen additieve achtergrond of absorptievloer.
        return stark

    def factor(self, wavelength_m):
        value = self.mixed(wavelength_m) / self.norm
        return float(value) if np.ndim(value) == 0 else value


def _build_spectra():
    """Bouw GSA/ESA-spectra en referentiefactoren voor het actieve model."""
    channel_sigmas = ({"3H6 -> 3H5": GSA_H5_SIGMA_CM1}
                      if ABSORPTION_MODEL == "broad_gsa" else None)
    gsa = StarkSpectrum(GSA_CHANNELS, GSA_BACKGROUND_WEIGHT,
                        GSA_BACKGROUND_LAMBDA0_NM, GSA_BACKGROUND_SIGMA_NM,
                        channel_sigmas_cm1=channel_sigmas,
                        analytic_gaussian=True)
    esa = StarkSpectrum(ESA_CHANNELS, ESA_BACKGROUND_WEIGHT,
                        ESA_BACKGROUND_LAMBDA0_NM, ESA_BACKGROUND_SIGMA_NM,
                        analytic_gaussian=True)
    if SI_REFERENCE_ANCHOR:
        g_ref = gsa.factor(REFERENCE_WAVELENGTH_NM * 1e-9)
        e_ref = esa.factor(REFERENCE_WAVELENGTH_NM * 1e-9)
        if min(g_ref, e_ref) <= 1e-12:
            raise ValueError("Reference absorption is too small to calibrate reliably. "
                             "Increase the GSA line width or revise the spectral model.")
    else:
        g_ref = e_ref = 1.0   # geen anker: factoren worden niet gebruikt
    return gsa, esa, g_ref, e_ref


if _USE_STARK:
    GSA_SPECTRUM, ESA_SPECTRUM, _GSA_REFERENCE_FACTOR, _ESA_REFERENCE_FACTOR = _build_spectra()
else:   # "si_1064": geen Stark-spectra
    GSA_SPECTRUM = ESA_SPECTRUM = None
    _GSA_REFERENCE_FACTOR = _ESA_REFERENCE_FACTOR = 1.0


def _stark_resonance_factors(pump_wavelength_nm, p):
    """Relative cross sections; may exceed one away from the reference."""
    wavelength_m = pump_wavelength_nm * 1e-9
    gsa = GSA_SPECTRUM.factor(wavelength_m)
    esa = ESA_SPECTRUM.factor(wavelength_m)
    if SI_REFERENCE_ANCHOR:
        gsa /= _GSA_REFERENCE_FACTOR
        esa /= _ESA_REFERENCE_FACTOR
    return gsa, esa


def _stark_rate_coefficients(pump_wavelength_nm, p):
    """Select GSA/ESA explicitly, without guessing from resonance centers."""
    wavelength_m = pump_wavelength_nm * 1e-9
    gsa, esa = _stark_resonance_factors(pump_wavelength_nm, p)
    return (pa_model.cross_section_to_rate_coeff(p.sigma_gsa_m2, wavelength_m)*gsa,
            pa_model.cross_section_to_rate_coeff(p.sigma_esa_m2, wavelength_m)*esa)


def analytical_threshold_intensity(w_gsa, w_esa, w2, w3, s31, a32):
    """
    Volledige lineaire-stabiliteitsdrempel I_th, inclusief GSA in de
    Jacobiaan (niet alleen als infinitesimale seed). Volgt uit
    det(J(I)) = 0 met R_gsa = w_gsa*I, R_esa = w_esa*I, wat kwadratisch
    is in I:
        A*I^2 + B*I + C = 0
    A = w_esa*w_gsa
    B = w_gsa*(s31+w3) - w_esa*(a32+s31-w3)
    C = w2*(s31+w3)
    Neemt de kleinste positieve wortel (eerste keer dat det(J) van
    positief naar negatief gaat als I stijgt vanaf 0).
    """
    if w_esa <= 0:
        return float("inf")

    C = w2 * (s31 + w3)

    if w_gsa <= 0:
        # Lineair geval (A=0): terugvallen op de gecorrigeerde
        # GSA-vrije formule (let op: a32 + s31 - w3, niet a32 + 2*s31).
        denom = a32 + s31 - w3
        if denom <= 0:
            return float("inf")
        return C / (w_esa * denom)

    A = w_esa * w_gsa
    B = w_gsa * (s31 + w3) - w_esa * (a32 + s31 - w3)

    disc = B * B - 4.0 * A * C
    if disc < 0:
        return float("inf")  # geen reeel snijpunt: geen lineaire instabiliteit

    sqrt_disc = disc ** 0.5
    roots = [(-B - sqrt_disc) / (2.0 * A), (-B + sqrt_disc) / (2.0 * A)]
    positive_roots = [r for r in roots if r > 0]

    if not positive_roots:
        return float("inf")

    return min(positive_roots)

def _wavelength_cauchy_factor(wavelength_m, lambda0_m, gamma_m):
    return float(1.0/(1.0+((wavelength_m-lambda0_m)/gamma_m)**2))


@contextlib.contextmanager
def _lineshape(kind):
    """Use the same coefficients in fast and fallback solvers; restore on exit."""
    original_factor = pa_model.wavelength_gaussian_factor
    original_rates = pa_model.gsa_esa_rate_coefficients
    original_resonances = pa_model.resonance_factors
    try:
        if kind == "stark_gaussian":
            pa_model.gsa_esa_rate_coefficients = _stark_rate_coefficients
            pa_model.resonance_factors = _stark_resonance_factors
        elif kind == "cauchy":
            pa_model.wavelength_gaussian_factor = _wavelength_cauchy_factor
        elif kind != "gaussian":
            raise ValueError(f"Unknown lineshape kind: {kind!r}")
        yield
    finally:
        pa_model.wavelength_gaussian_factor = original_factor
        pa_model.gsa_esa_rate_coefficients = original_rates
        pa_model.resonance_factors = original_resonances


# ---------------------------------------------------------------------------
# Avalanche detection
# ---------------------------------------------------------------------------

def has_avalanche(
        I_values: np.ndarray,
        emission: np.ndarray,
        min_slope: float = MIN_AVALANCHE_SLOPE,
        edge_margin: int = KNEE_EDGE_MARGIN) -> Tuple[bool, float]:
    """
    Decide whether the emission curve contains a photon-avalanche knee
    inside the simulated intensity window.

    Criteria:
      1. The maximum log-log slope is >= min_slope (a pure power law from
         GSA/ESA-limited pumping has s ~ 2-3, an avalanche has s >> 10).
      2. The slope maximum is not within `edge_margin` points of either edge
         of the window (otherwise the knee is cut off and no threshold can
         be extracted from the data).

    Returns (avalanche_present, s_max).
    """

    emission = np.asarray(emission, dtype=float)

    if not np.all(np.isfinite(emission)) or np.all(emission <= 0.0):
        return False, float("nan")

    slope = nonlinearity_slope(I_values, emission)
    idx = int(np.argmax(slope))
    s_max = float(slope[idx])

    knee_is_interior = edge_margin <= idx < len(I_values) - edge_margin

    return bool(s_max >= min_slope and knee_is_interior), s_max


# ---------------------------------------------------------------------------
# Per-curve fit of the steep, nearly linear avalanche segment
# ---------------------------------------------------------------------------

def fit_avalanche_line(I_values, emission, fraction=SLOPE_FRACTION):
    """Fit the widest verified straight segment near the maximum log slope."""
    intensity = np.asarray(I_values, dtype=float)
    emission = np.asarray(emission, dtype=float)
    if intensity.ndim != 1 or emission.shape != intensity.shape:
        raise ValueError("Intensity and emission must be matching 1D arrays.")
    if (len(intensity) < 3 or np.any(~np.isfinite(intensity))
            or np.any(~np.isfinite(emission)) or np.any(intensity <= 0)
            or np.any(emission <= 0) or np.any(np.diff(intensity) <= 0)):
        return None
    if fraction is not None and not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1], or None for automatic fitting.")
    x, y = np.log10(intensity), np.log10(emission)
    slopes = np.diff(y)/np.diff(x)
    k = int(np.argmax(slopes))
    peak = float(slopes[k])
    if peak <= 0:
        return None
    candidates = np.linspace(0.20, 0.99, 80) if fraction is None else [fraction]
    seen, accepted = set(), []
    for f in candidates:
        lo = hi = k
        while lo > 0 and slopes[lo-1] >= f*peak:
            lo -= 1
        while hi < len(slopes)-1 and slopes[hi+1] >= f*peak:
            hi += 1
        if (lo, hi) in seen:
            continue
        seen.add((lo, hi))
        xw, yw = x[lo:hi+2], y[lo:hi+2]
        if len(xw) < (FIT_MIN_POINTS if fraction is None else 3):
            continue
        if fraction is None and (k-lo < 2 or hi-k < 2):
            continue  # resolve both sides of the maximum
        xu = np.linspace(xw[0], xw[-1], 200)
        yu = np.interp(xu, xw, yw)
        slope, intercept = np.polyfit(xu, yu, 1)
        residual = yu-(slope*xu+intercept)
        total = np.sum((yu-yu.mean())**2)
        r2 = 1-np.sum(residual**2)/total if total > 0 else 0.0
        max_error = max(np.max(np.abs(residual)),
                        np.max(np.abs(yw-(slope*xw+intercept))))
        span = float(yw[-1]-yw[0])
        fit = {
            "slope": float(slope), "intercept": float(intercept),
            "x_range": (float(xw[0]), float(xw[-1])),
            "y_range": (float(yw[0]), float(yw[-1])),
            "s_peak": peak, "n_points": len(xw),
            "fraction": float(f), "r_squared": float(r2),
            "max_error_dex": float(max_error), "span_dex": span,
            "peak_ratio": float(slope/peak),
            "method": "automatic" if fraction is None else "fixed",
        }
        if fraction is None and (r2 < FIT_MIN_R2 or max_error > FIT_MAX_ERROR_DEX
                or slope/peak < FIT_MIN_PEAK_RATIO or span < FIT_MIN_SPAN_DEX):
            continue
        accepted.append(fit)
    if not accepted:
        return None
    return max(accepted, key=lambda f: (f["x_range"][1]-f["x_range"][0],
                                        -f["max_error_dex"]))


# ---------------------------------------------------------------------------
# Threshold determination
# ---------------------------------------------------------------------------

def pa_threshold_cauchy(
        p8,
        I_values,
        pump_wavelength_nm,
        emission_coarse):

    s_max, I_at_smax, _n_iter = nonlinearity_slope_max_converged(
        I_values,
        p8,
        pump_wavelength_nm=pump_wavelength_nm,
        emission_coarse=emission_coarse,
        n_refine=25,
        max_iter=4,
        rel_tol=2e-2
    )

    return float(I_at_smax), float(s_max)


# ---------------------------------------------------------------------------
# Fast steady-state emission
# ---------------------------------------------------------------------------

def _make_rhs(I_pump: float, p, w_gsa: float, w_esa: float):
    """
    Right-hand side of the DREs (SI Eqns. S1-S3) for fixed pump intensity
    and FIXED rate coefficients. Identical equations to
    pa_model.pa_rate_equations, but on plain floats and without
    recomputing the wavelength dependence at every call.

    y=[N0(3H6), N1(3F4), N2(3H4)]. The effective GSA/ESA spectra already
    include p_lower_i: use R_gsa*N0 and R_esa*N1 exactly once. Relaxation
    a32*N2 and cross-relaxation 2*s31*N2*N0 feed the TOTAL N1. Its Stark
    populations are recovered by stark_populations(y), without feeding an
    extra thermal factor back into these three-population equations.
    """

    R_gsa = w_gsa * I_pump
    R_esa = w_esa * I_pump

    w2, w3 = p.w2, p.w3
    s31, q23 = p.s31, p.q23
    a32 = p.b32 * p.w3_r + p.w3_nr
    q_sum = p.q22 + 2.0 * p.q23

    def rhs(_t, y):

        N0, N1, N2 = y

        dN2 = (
            R_esa * N1
            - w3 * N2
            - s31 * N2 * N0
            + q23 * N1 * N1
        )

        dN1 = (
            R_gsa * N0
            - R_esa * N1
            - w2 * N1
            + a32 * N2
            + 2.0 * s31 * N2 * N0
            - q_sum * N1 * N1
        )

        return (-(dN1 + dN2), dN1, dN2)

    return rhs


def _check_fast_rhs(p, pump_wavelength_nm: float) -> None:
    """
    Safety net: the local right-hand side must reproduce
    pa_model.pa_rate_equations (with the currently active line shape).
    """

    w_gsa, w_esa = pa_model.gsa_esa_rate_coefficients(
        pump_wavelength_nm,
        p
    )

    y = np.array([0.7, 0.2, 0.1])
    I_test = 37.0

    mine = np.array(
        _make_rhs(I_test, p, w_gsa, w_esa)(0.0, y)
    )

    ref = pa_model.pa_rate_equations(
        y,
        I_test,
        p,
        pump_wavelength_nm
    )

    if not np.allclose(mine, ref, rtol=1e-10, atol=1e-12):
        raise RuntimeError(
            "Fast right-hand side differs from pa_model.pa_rate_equations: "
            f"{mine} vs {ref}. Check the local rate equations."
        )


def check_thermal_population_coupling(p=None):
    """Check normalized p_i, preserved N0/N1, default sticks and SI anchor.

    If parameters p are supplied, also check the actual pa_model RHS and
    the SI cross sections. This does not integrate or alter any populations.
    Alleen aanroepen in de "stark"- en "hybrid"-stand.
    """
    def require_close(actual, expected, description):
        if not np.allclose(actual, expected, rtol=1e-12, atol=0.0):
            raise RuntimeError(f"Thermal-population check failed: {description}.")

    checks = {}
    for name, levels, expected_count, stored_p in (
            ("3H6", E_3H6_CM1, 13, P_3H6_BOLTZMANN),
            ("3F4", E_3F4_CM1, 9, P_3F4_BOLTZMANN)):
        probabilities = boltzmann_within(levels)
        if (probabilities.shape != (expected_count,)
                or np.any(~np.isfinite(probabilities))
                or np.any(probabilities < 0)):
            raise RuntimeError(f"Invalid Boltzmann distribution for {name}.")
        require_close(probabilities.sum(), 1.0, f"sum p_{name} = 1")
        require_close(stored_p, probabilities, f"stored p_{name}")
        require_close(probabilities, boltzmann_within(levels - levels.min()),
                      f"{name} uses its own minimum energy")
        checks[f"boltzmann_sum_{name}"] = float(probabilities.sum())

    # Include N1=0 and non-unit totals to detect any accidental normalization
    # of N1 or the input population vector by the reconstruction helper.
    states = np.array([[1.0, 0.0, 0.0], [0.7, 0.2, 0.1],
                       [0.1, 0.75, 0.15], [0.0, 1.0, 0.0],
                       [7.0, 2.0, 1.0]])
    n_h6, n_f4 = stark_populations(states)
    require_close(n_h6.sum(axis=-1), states[:, 0], "sum N_H6_i = N0")
    require_close(n_f4.sum(axis=-1), states[:, 1], "sum N_F4_i = N1")
    checks["stark_population_totals_preserved"] = True

    stick_sums = {}
    for name, (lower, upper, _channel_weight) in {
            **GSA_CHANNELS, **ESA_CHANNELS}.items():
        _gaps, weights = transition_sticks(lower, upper)
        require_close(weights.sum(), 1.0, f"default stick sum for {name}")
        require_close(weights.sum(axis=1), level_populations(lower),
                      f"lower-only marginal weights for {name}")
        stick_sums[name] = float(weights.sum())
    checks["default_stick_sums"] = stick_sums

    if SI_REFERENCE_ANCHOR:
        factors = _stark_resonance_factors(REFERENCE_WAVELENGTH_NM, p)
        require_close(factors, (1.0, 1.0), "1064-nm spectral reference factors")
        if p is not None:
            sigma_gsa = p.sigma_gsa_m2 * factors[0]
            sigma_esa = p.sigma_esa_m2 * factors[1]
            require_close(sigma_gsa, 6.0e-29, "SI sigma_GSA at 1064 nm")
            require_close(sigma_esa, 6.4e-25, "SI sigma_ESA at 1064 nm")
            checks["sigma_gsa_1064_m2"] = float(sigma_gsa)
            checks["sigma_esa_1064_m2"] = float(sigma_esa)

    if p is not None:
        with _lineshape("stark_gaussian"):
            _check_fast_rhs(p, REFERENCE_WAVELENGTH_NM)
        checks["pa_model_rhs_matches"] = True
    return checks


def _stationary_population(I_pump, p, w_gsa, w_esa):
    """Integrate from ground state; verify residual and time stability."""
    if not _HAVE_SCIPY:
        raise RuntimeError("Stationary detuning scans require scipy (scipy.integrate.solve_ivp).")
    rhs = _make_rhs(float(I_pump), p, w_gsa, w_esa)
    seed_scale = w_gsa * float(I_pump) / max(p.w2 + p.w3, 1.0)
    pop_atol = max(1e-280, min(1e-16, seed_scale * 1e-10))
    atol = np.array([1e-12, pop_atol, pop_atol])
    y = np.array([1.0, 0.0, 0.0])
    elapsed, duration = 0.0, T_STEADY_S
    while elapsed < MAX_STEADY_TIME_S:
        duration = min(duration, MAX_STEADY_TIME_S - elapsed)
        previous = y.copy()
        sol = solve_ivp(rhs, (0.0, duration), y, method="LSODA",
                        rtol=1e-9, atol=atol)
        if not sol.success or not np.all(np.isfinite(sol.y[:, -1])):
            raise RuntimeError(f"Integration failed: {sol.message}")
        y = sol.y[:, -1]
        elapsed += duration
        if np.min(y) < -1e-12 or abs(y.sum()-1.0) > 1e-8:
            raise RuntimeError("Unphysical populations from integration.")
        scale = np.maximum(np.abs(y), STEADY_POPULATION_FLOOR)
        change = np.max(np.abs(y - previous)/scale)
        residual = np.max(np.abs(rhs(0.0, y))/scale) * T_STEADY_S
        if change < STEADY_RELATIVE_TOL and residual < STEADY_RELATIVE_TOL:
            return y, elapsed
        duration *= 2.0
    raise RuntimeError(f"Stationarity not reached at I={I_pump:g} kW/cm^2 "
                       f"after {elapsed:g} s; no threshold reported.")


def _emission_curve(I_values, p8, pump_wavelength_nm):
    """Stationarity-checked 800-nm emission with the active absorption model."""
    w_gsa, w_esa = pa_model.gsa_esa_rate_coefficients(pump_wavelength_nm, p8)
    emission = np.empty(len(I_values))
    for i, intensity in enumerate(I_values):
        y, _elapsed = _stationary_population(float(intensity), p8, w_gsa, w_esa)
        emission[i] = emission_from_population(max(float(y[2]), 0.0), p8)
    return emission


def adaptive_knee_scan(
        p8,
        pump_wavelength_nm: float,
        I_min: float,
        I_max: float,
        n_coarse: int = N_COARSE,
        n_knee: int = N_KNEE,
        n_polish: int = N_POLISH):
    
    I_coarse = np.logspace(
        np.log10(I_min),
        np.log10(I_max),
        n_coarse
    )

    E_coarse = _emission_curve(
        I_coarse,
        p8,
        pump_wavelength_nm
    )

    present, s_coarse = has_avalanche(
        I_coarse,
        E_coarse,
        edge_margin=1
    )

    if not present:
        return I_coarse, E_coarse, np.nan, s_coarse, False, None

    def _zoom(I_grid, E_grid, n_points):
        idx = int(np.argmax(nonlinearity_slope(I_grid, E_grid)))
        lo = I_grid[max(idx - 2, 0)]
        hi = I_grid[min(idx + 2, len(I_grid) - 1)]
        I_new = np.logspace(
            np.log10(lo),
            np.log10(hi),
            n_points
        )
        E_new = _emission_curve(
            I_new,
            p8,
            pump_wavelength_nm
        )
        return I_new, E_new

    I_knee_grid, E_knee_grid = _zoom(I_coarse, E_coarse, n_knee)
    I_pol, E_pol = _zoom(I_knee_grid, E_knee_grid, n_polish)

    slope_pol = nonlinearity_slope(I_pol, E_pol)
    k = int(np.argmax(slope_pol))

    Ith_star = float(I_pol[k])

    I_all = np.concatenate([I_coarse, I_knee_grid, I_pol])
    E_all = np.concatenate([E_coarse, E_knee_grid, E_pol])
    order = np.argsort(I_all)
    I_all = I_all[order]
    E_all = E_all[order]
    keep = np.concatenate(
        [[True], np.diff(np.log(I_all)) > 1e-9]
    )
    I_all = I_all[keep]
    E_all = E_all[keep]

    fit = fit_avalanche_line(I_all, E_all)
    s_max = fit["slope"] if fit is not None else float("nan")

    return I_all, E_all, Ith_star, s_max, True, fit


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_wavelength_detuning(
        curves: Dict[float, Tuple[np.ndarray, np.ndarray]],
        thresholds: Dict[float, float],
        lineshape_label: str,
        lineshape_kind: str,
        fit_lines: Optional[Dict[float, Optional[Dict[str, float]]]] = None,
        savepath: Optional[str] = None) -> None:
    
    ith_label = "Ith*"
    fig, ax = plt.subplots(figsize=(9.0, 7.0))
    colors = plt.cm.tab20(np.linspace(0, 1, len(curves)))
    I_lo = min(float(I.min()) for I, _E in curves.values())
    I_hi = max(float(I.max()) for I, _E in curves.values())

    for (lam, (I_lam, emission)), color in zip(curves.items(), colors):
        ith = thresholds[lam]
        if np.isfinite(ith):
            label = (f"{lam:.0f} nm ({ith_label} = {ith:.1f} " r"kW cm$^{-2}$)")
            linestyle = "o-"
        else:
            label = f"{lam:.0f} nm (no PA in range)"
            linestyle = "o--"

        ax.loglog(I_lam, _safe_for_log(emission), linestyle, markersize=3.5, color=color, label=label)

    if fit_lines:
        for fit in fit_lines.values():
            if fit is None or fit["slope"] <= 0.0:
                continue
            s = fit["slope"]
            b = fit["intercept"]
            y0 = fit["y_range"][0] - LINE_EXTEND_DECADES
            y1 = fit["y_range"][1] + LINE_EXTEND_DECADES
            x0 = (y0 - b) / s
            x1 = (y1 - b) / s
            ax.loglog([10.0 ** x0, 10.0 ** x1], [10.0 ** y0, 10.0 ** y1], "-", color="k", linewidth=1.3, zorder=5, label="_nolegend_")

    ax.set_xlabel(r"Excitation intensity (kW cm$^{-2}$)", fontsize=13)
    ax.set_ylabel("800-nm steady-state emission intensity (a.u.)", fontsize=13)
    ax.tick_params(axis="both", which="both", labelsize=11)
    ax.set_xlim(I_lo, I_hi)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.32, top=0.97)

    if savepath:
        fig.savefig(savepath, format="pdf", bbox_inches="tight")
        print(f"Figuur opgeslagen: {savepath}")


def plot_smax_vs_wavelength(
        fit_lines: Dict[float, Optional[Dict[str, float]]],
        savepath: Optional[str] = None) -> None:
    
    wavelengths = np.array(sorted(fit_lines.keys()), dtype=float)
    s_values = np.array([(float(fit_lines[lam]["slope"]) if fit_lines[lam] is not None else np.nan) for lam in wavelengths])

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(wavelengths, s_values, "o-", linewidth=1.6, markersize=5)
    ax.set_xlabel("Pump wavelength (nm)")
    ax.set_ylabel(r"Fitted slope $s_{\mathrm{fit}}$", fontsize=13)
    ax.set_title(r"Slope of the verified linear segment vs. pump wavelength")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    
    if savepath:
        fig.savefig(savepath, format="pdf", bbox_inches="tight")
        print(f"Figuur opgeslagen: {savepath}")


def plot_ith_vs_wavelength(
        thresholds_num: Dict[float, float],
        thresholds_ana: Dict[float, float],
        savepath: Optional[str] = None) -> None:
    
    wavelengths = np.array(sorted(thresholds_num.keys()), dtype=float)
    ith_num_values = np.array([float(thresholds_num[lam]) for lam in wavelengths])
    ith_ana_values = np.array([float(thresholds_ana[lam]) for lam in wavelengths])

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    
    # Numerieke curve (via modelering en knie-detectie)
    ax.plot(wavelengths, ith_num_values, "o-", linewidth=1.6, markersize=5, label=r"Numeriek ($I_{\mathrm{th}}^{*}$)")
    
    # Analytische curve (vergelijking 5)
    ax.plot(wavelengths, ith_ana_values, "s--", linewidth=1.6, markersize=5, label="Analytisch")

    ax.set_xlabel("Pump wavelength (nm)")
    ax.set_ylabel(r"$I_{\mathrm{th}}$ (kW cm$^{-2}$)")
    ax.set_title(r"Photon-avalanche threshold $I_{\mathrm{th}}$ vs. pump wavelength")
    
    # Logaritmische schaal om de grote variatie in analytische drempelwaarden te tonen
    ax.set_yscale("log")
    
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=10)
    
    fig.tight_layout()
    
    if savepath:
        fig.savefig(savepath, format="pdf", bbox_inches="tight")
        print(f"Figuur opgeslagen: {savepath}")


# ---------------------------------------------------------------------------
# Curve generation
# ---------------------------------------------------------------------------

def compute_detuning_curves(
        I_values: np.ndarray,
        p8,
        wavelengths_nm,
        kind: str,
        label: str):
    
    curves: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
    thresholds: Dict[float, float] = {}
    analytical_thresholds: Dict[float, float] = {}  # TOEVOEGING: voor de analytische data
    fit_lines: Dict[float, Optional[Dict[str, float]]] = {}

    print(f"\n=== Wavelength-detuning demonstration (8% Tm sample, {label}) ===")
    print(f"  ABSORPTION_MODEL = {ABSORPTION_MODEL!r}")

    if kind == "stark_gaussian":
        print(f"  Stark sigma = {STARK_SIGMA_CM1:g} cm^-1 (model parameter)")
        print(f"  Convolution: Gauss sigma_tot = "
              f"{np.hypot(STARK_SIGMA_CM1, INSTRUMENT_SIGMA_CM1):g} cm^-1 "
              f"(Stark {STARK_SIGMA_CM1:g} + instrument {INSTRUMENT_SIGMA_CM1:g} "
              f"in quadrature), Lorentz HWHM = {LORENTZ_HWHM_CM1:g} cm^-1")
        print(f"  Intra-multiplet thermalization: T = {TEMPERATURE_K:g} K, "
              f"BOLTZMANN_MODE = {BOLTZMANN_MODE!r}")
        for role, spectrum in (("GSA", GSA_SPECTRUM), ("ESA", ESA_SPECTRUM)):
            for name, (lower, upper, weight) in spectrum.channels.items():
                gaps = transition_gaps(lower, upper)
                wavelengths = 1e7 / gaps.ravel()
                p_lower = level_populations(lower)
                print(f"  {role} {name}: {lower.size} lower x {upper.size} upper sublevels, "
                      f"{wavelengths.min():.1f}-{wavelengths.max():.1f} nm, "
                      f"relative channel weight={weight/spectrum.weight_sum:.3f}, "
                      f"lower-level weight range="
                      f"[{p_lower.min():.3g}, {p_lower.max():.3g}]")
            print(f"  {role} background weight = {spectrum.background_weight:g}")
        print("  Three-population approximation; channel strengths are not fitted.")
        print("  Stationarity verified by time stability and rate-equation residual.")
        print(f"  SI absorption anchor at {REFERENCE_WAVELENGTH_NM:g} nm: {SI_REFERENCE_ANCHOR}")
        print("  Missing PA means no detected avalanche in the scanned intensity window.")

    with _lineshape(kind):

        if (_HAVE_SCIPY and kind in ("gaussian", "stark_gaussian")):
            _check_fast_rhs(p8, float(wavelengths_nm[0]))
            print("  Fast steady-state solver (LSODA); right-hand side "
                  "verified against pa_model.pa_rate_equations.")

        for lam in wavelengths_nm:
            
            # --- Analytische drempel: volledige kwadratische lineaire-
            # stabiliteitsdrempel, inclusief GSA in de Jacobiaan ---
            w_gsa, w_esa = pa_model.gsa_esa_rate_coefficients(lam, p8)
            a32 = p8.b32 * p8.w3_r + p8.w3_nr

            ith_ana = analytical_threshold_intensity(
                w_gsa, w_esa, p8.w2, p8.w3, p8.s31, a32
            )

            analytical_thresholds[lam] = ith_ana
            # ------------------------------------------

            if kind in ("gaussian", "stark_gaussian"):

                I_lam, emission_lam, Ith_lam, s_max, present, fit = (
                    adaptive_knee_scan(p8, lam, float(I_values[0]), float(I_values[-1]))
                )

                curves[lam] = (I_lam, emission_lam)
                thresholds[lam] = Ith_lam
                fit_lines[lam] = fit

                if not present:
                    print(f"  lambda_pump = {lam:6.1f} nm | Ith*(num) =    n/a   | Ith(ana) = {ith_ana:7.2f} kW/cm^2 | s_max = {s_max:6.2f} | E({I_lam[-1]:.0f} kW/cm^2) = {emission_lam[-1]:.3g}  <-- no avalanche in {I_lam[0]:.0f}-{I_lam[-1]:.0f} kW/cm^2")
                elif fit is None:
                    print(f"  lambda_pump = {lam:.1f} nm | Ith*(num) = {Ith_lam:.2f} | Ith(ana) = {ith_ana:7.2f} kW/cm^2 | PA detected, but no adequately resolved linear fit.")
                else:
                    print(f"  lambda_pump = {lam:6.1f} nm | Ith*(num) = {Ith_lam:7.2f} kW/cm^2 | Ith(ana) = {ith_ana:7.2f} kW/cm^2 | s_fit = {s_max:6.2f} (peak local slope {fit['s_peak']:6.2f}) | fit over {10.0 ** fit['x_range'][0]:.2f}-{10.0 ** fit['x_range'][1]:.2f} kW/cm^2 ({fit['n_points']} pts, fraction={fit['fraction']:.2f}, R2={fit['r_squared']:.5f}, max error={fit['max_error_dex']:.3f} dex) | E({I_lam[-1]:.0f} kW/cm^2) = {emission_lam[-1]:.3g}")

            else:
                emission_lam = power_dependence(I_values, p8, pump_wavelength_nm=lam)
                curves[lam] = (np.asarray(I_values), emission_lam)

                Ith_lam, s_max_converged = pa_threshold_cauchy(p8, I_values, lam, emission_coarse=emission_lam)

                thresholds[lam] = Ith_lam
                fit_lines[lam] = None

                print(f"  lambda_pump = {lam:6.1f} nm | Ith*(num) = {Ith_lam:7.2f} kW/cm^2 | Ith(ana) = {ith_ana:7.2f} kW/cm^2 | s_max = {s_max_converged:6.2f} | E({I_values[-1]:.0f} kW/cm^2) = {emission_lam[-1]:.3g}")

    return curves, thresholds, analytical_thresholds, fit_lines

# ---------------------------------------------------------------------------
# Rise time (95%): piek zoeken op een adaptieve intensiteitsgrid
# ---------------------------------------------------------------------------

RISE_FRACTION = 0.95
RISE_RTOL = 1e-8
RISE_WAVELENGTHS_NM = [1060.0, 1064.0, 1070.0, 1080.0, 1410, 1430, 1450.0, 1470]
RISE_SIGMA_VALUES_CM1 = [3.0, 5.0, 8.5, 12.0, 16.0, 24.0, 50, 100 ,150,200]
RISE_SIGMA_WAVELENGTHS_NM = [1064.0, 1450.0]

# Piekzoeker: grove log-grid + n_rounds keer verfijnen rond het maximum
RISE_N_COARSE, RISE_N_REFINE, RISE_N_ROUNDS = 24, 13, 2          # hoofdplot
RISE_N_COARSE_SWEEP, RISE_N_REFINE_SWEEP = 14, 9                 # sigma-sweep

RISE_YSCALE = "log"   # "linear" voor dezelfde stijl als pa_fig2b_rise_time_v2.png
RISE_XSCALE = "linear"

# Bij SI_REFERENCE_ANCHOR ("hybrid" en "broad_gsa") ligt de absorptie bij 1064 nm vast,
# onafhankelijk van sigma. Zet op False om de anker tijdens de sigma-sweep uit
# te zetten.
RISE_SIGMA_KEEP_ANCHOR = True


def rise_time_95_fast(I_pump, p, w_gsa, w_esa, fraction=RISE_FRACTION):
    """Tijd (s) tot N2 (=800-nm emissie) `fraction` van stationair bereikt.
    Eén integratie met dense_output; de 95%-doorgang wordt achteraf met brentq bepaald."""
    if not _HAVE_SCIPY:
        raise RuntimeError("Rise time vereist scipy.")
    I_pump = float(I_pump)
    rhs = _make_rhs(I_pump, p, w_gsa, w_esa)
    seed_scale = w_gsa * I_pump / max(p.w2 + p.w3, 1.0)
    pop_atol = max(1e-280, min(1e-16, seed_scale * 1e-10))
    atol = np.array([1e-12, pop_atol, pop_atol])

    y = np.array([1.0, 0.0, 0.0])
    t_off, duration = 0.0, T_STEADY_S
    pieces = []
    while t_off < MAX_STEADY_TIME_S:
        duration = min(duration, MAX_STEADY_TIME_S - t_off)
        previous = y.copy()
        sol = solve_ivp(rhs, (0.0, duration), y, method="LSODA",
                        rtol=RISE_RTOL, atol=atol, dense_output=True)
        if not sol.success or not np.all(np.isfinite(sol.y[:, -1])):
            raise RuntimeError(f"Integration failed: {sol.message}")
        pieces.append((t_off, sol))
        y = sol.y[:, -1]
        t_off += duration
        scale = np.maximum(np.abs(y), STEADY_POPULATION_FLOOR)
        change = np.max(np.abs(y - previous) / scale)
        residual = np.max(np.abs(rhs(0.0, y)) / scale) * T_STEADY_S
        if change < STEADY_RELATIVE_TOL and residual < STEADY_RELATIVE_TOL:
            break
        duration *= 2.0
    else:
        raise RuntimeError(f"Stationarity not reached at I={I_pump:g} kW/cm^2")

    target = fraction * y[2]
    if target <= 0:
        return float("nan")
    for off, sol in pieces:
        above = np.nonzero(sol.y[2] >= target)[0]
        if above.size:
            j = int(above[0])
            if j == 0:
                return float(off)
            root = brentq(lambda t: sol.sol(t)[2] - target,
                          sol.t[j - 1], sol.t[j], xtol=1e-12, rtol=1e-10)
            return float(off + root)
    return float("nan")


def _rise_ms(p8, w_gsa, w_esa, I_array):
    out = np.empty(len(I_array))
    for i, I in enumerate(I_array):
        try:
            out[i] = rise_time_95_fast(I, p8, w_gsa, w_esa) * 1e3
        except RuntimeError:
            out[i] = np.nan
    return out


def rise_peak_scan(p8, wavelength_nm, I_min, I_max,
                   n_coarse=RISE_N_COARSE, n_refine=RISE_N_REFINE,
                   n_rounds=RISE_N_ROUNDS):
    """Rise time (ms) vs intensiteit; zoekt de piek adaptief (actieve lineshape).
    Geeft (I, rise_ms, index_van_piek) terug."""
    w_gsa, w_esa = pa_model.gsa_esa_rate_coefficients(wavelength_nm, p8)
    I_all = np.logspace(np.log10(I_min), np.log10(I_max), n_coarse)
    rt_all = _rise_ms(p8, w_gsa, w_esa, I_all)
    if not np.any(np.isfinite(rt_all)):
        return I_all, rt_all, None
    for _ in range(n_rounds):
        k = int(np.nanargmax(rt_all))
        lo, hi = I_all[max(k - 1, 0)], I_all[min(k + 1, len(I_all) - 1)]
        I_new = np.logspace(np.log10(lo), np.log10(hi), n_refine)[1:-1]
        rt_new = _rise_ms(p8, w_gsa, w_esa, I_new)
        I_all = np.concatenate([I_all, I_new])
        rt_all = np.concatenate([rt_all, rt_new])
        order = np.argsort(I_all)
        I_all, rt_all = I_all[order], rt_all[order]
    return I_all, rt_all, int(np.nanargmax(rt_all))


@contextlib.contextmanager
def _stark_sigma(sigma_cm1, keep_anchor=True):
    """Tijdelijk STARK_SIGMA_CM1 wijzigen en spectra + referentiefactoren herbouwen.
    Voor alle Stark-modi. In broad_gsa blijft de H5-GSA sigma afzonderlijk
    op GSA_H5_SIGMA_CM1 staan; alleen de overige Stark-kanalen veranderen."""
    global STARK_SIGMA_CM1, GSA_SPECTRUM, ESA_SPECTRUM
    global _GSA_REFERENCE_FACTOR, _ESA_REFERENCE_FACTOR, SI_REFERENCE_ANCHOR
    old = (STARK_SIGMA_CM1, GSA_SPECTRUM, ESA_SPECTRUM,
           _GSA_REFERENCE_FACTOR, _ESA_REFERENCE_FACTOR, SI_REFERENCE_ANCHOR)
    try:
        STARK_SIGMA_CM1 = float(sigma_cm1)
        SI_REFERENCE_ANCHOR = bool(keep_anchor) and old[-1]
        GSA_SPECTRUM, ESA_SPECTRUM, _GSA_REFERENCE_FACTOR, _ESA_REFERENCE_FACTOR = _build_spectra()
        yield
    finally:
        (STARK_SIGMA_CM1, GSA_SPECTRUM, ESA_SPECTRUM,
         _GSA_REFERENCE_FACTOR, _ESA_REFERENCE_FACTOR, SI_REFERENCE_ANCHOR) = old


def compute_rise_vs_intensity(p8, I_min, I_max, wavelengths_nm=RISE_WAVELENGTHS_NM):
    results = {}
    with _lineshape(ABSORPTION_KIND):
        for lam in wavelengths_nm:
            I_grid, rt, k = rise_peak_scan(p8, lam, I_min, I_max)
            if k is None:
                print(f"  Rise time {lam:.0f} nm: geen geldige punten.")
                continue
            edge = " (LET OP: piek aan de rand van het venster)" if k in (0, len(I_grid) - 1) else ""
            print(f"  Rise time {lam:.0f} nm: piek {rt[k]:.2f} ms bij {I_grid[k]:.2f} kW/cm^2{edge}")
            results[lam] = (I_grid, rt, k)
    return results


def compute_rise_peak_vs_sigma(p8, wavelength_nm, sigmas, I_min, I_max):
    peaks = []
    for s in sigmas:
        with _stark_sigma(s, keep_anchor=RISE_SIGMA_KEEP_ANCHOR), _lineshape("stark_gaussian"):
            I_grid, rt, k = rise_peak_scan(p8, wavelength_nm, I_min, I_max,
                                           n_coarse=RISE_N_COARSE_SWEEP,
                                           n_refine=RISE_N_REFINE_SWEEP)
            if k is None:
                peaks.append(np.nan)
                print(f"  {wavelength_nm:.0f} nm, sigma={s:g}: geen geldige punten")
                continue
            peaks.append(float(rt[k]))
            print(f"  {wavelength_nm:.0f} nm, sigma={s:g} cm^-1: "
                  f"piek {rt[k]:.2f} ms bij {I_grid[k]:.2f} kW/cm^2")
    return np.array(peaks)


def plot_rise_vs_intensity(results, savepath=None):
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(results), 2)))
    for (lam, (I_grid, rt, k)), color in zip(results.items(), colors):
        ax.plot(I_grid, rt, "o-", markersize=3.5, color=color, label=f"{lam:.0f} nm")
        ax.plot(I_grid[k], rt[k], "*", markersize=12, color=color,
                markeredgecolor="k", zorder=6)
    ax.set_xscale(RISE_XSCALE)
    ax.set_yscale(RISE_YSCALE)
    ax.set_xlabel(r"Excitation intensity (kW cm$^{-2}$)", fontsize=13)
    ax.set_ylabel("800-nm rise time (ms)", fontsize=13)
    ax.set_title("Rise-time signature of photon avalanching")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=10)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, format="pdf", bbox_inches="tight")
        print(f"Figuur opgeslagen: {savepath}")


def plot_rise_peak_vs_sigma(sigmas, peaks_by_wavelength, savepath=None):
    n = len(peaks_by_wavelength)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.6), squeeze=False)
    for ax, (lam, peaks) in zip(axes[0], peaks_by_wavelength.items()):
        ax.plot(sigmas, peaks, "o-", linewidth=1.6, markersize=5)
        ax.set_xlabel(r"Stark Gaussian $\sigma$ (cm$^{-1}$)", fontsize=12)
        ax.set_ylabel("Peak 800-nm rise time (ms)", fontsize=12)
        ax.set_title(f"{lam:.0f} nm pump")
        ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, format="pdf", bbox_inches="tight")
        print(f"Figuur opgeslagen: {savepath}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def absorption_factors(wavelengths_nm, p):
    """The SAME dimensionless cross-section factors used by the pump rates.

    hybrid and broad_gsa: sigma(lambda)/sigma(1064 nm), hence f(1064)=1.
    stark and si_1064: preserve their existing global/peak normalization.
    No extra wavelength/photon-flux multiplier is included here.
    """
    wavelengths = np.asarray(wavelengths_nm, dtype=float)
    if wavelengths.ndim != 1 or not len(wavelengths):
        raise ValueError("wavelengths_nm must be a nonempty 1D array.")
    if np.any(~np.isfinite(wavelengths)) or np.any(wavelengths <= 0):
        raise ValueError("Wavelengths must be finite and positive.")
    if _USE_STARK:
        return tuple(np.asarray(v) for v in _stark_resonance_factors(wavelengths, p))
    values = np.array([pa_model.resonance_factors(float(lam), p) for lam in wavelengths])
    return values[:, 0], values[:, 1]


def plot_absorption_factors(p, save_dir, wavelength_min_nm=900.0,
                            wavelength_max_nm=1500.0, step_nm=0.05):
    """Plot/export f_GSA and f_ESA without running population simulations."""
    if wavelength_max_nm <= wavelength_min_nm or step_nm <= 0:
        raise ValueError("Invalid wavelength range or step.")
    n = int(np.ceil((wavelength_max_nm-wavelength_min_nm)/step_nm)) + 1
    wavelengths = np.linspace(wavelength_min_nm, wavelength_max_nm, n)
    f_gsa, f_esa = absorption_factors(wavelengths, p)
    if np.any(~np.isfinite(f_gsa)) or np.any(~np.isfinite(f_esa)):
        raise RuntimeError("Non-finite absorption factors.")

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(10.8, 6.4))
        fig.subplots_adjust(left=0.11, right=0.97, bottom=0.17, top=0.81)
        fig.suptitle("GSA- en ESA-versterkingsfactor", x=0.11, y=0.96,
                     ha="left", fontsize=20, fontweight="bold")
        if SI_REFERENCE_ANCHOR:
            subtitle = (("Verbrede GSA | " if ABSORPTION_MODEL == "broad_gsa"
                         else "Hybrid model | ") + r" $f_i(\lambda)=\sigma_i(\lambda)/\sigma_i(1064\,\mathrm{nm})$"
                        + " | T = 300 K")
        else:
            subtitle = f"{ABSORPTION_LABEL} | bestaande pieknormalisatie | T = 300 K"
        fig.text(0.11, 0.885, subtitle, fontsize=11, color="#526171")

        ax.plot(wavelengths, np.where(f_gsa > 0, f_gsa, np.nan),
                color="#ba4a18", linewidth=2.1, label=r"$f_{\mathrm{gsa}}$")
        ax.plot(wavelengths, np.where(f_esa > 0, f_esa, np.nan),
                color="#1468b1", linewidth=2.1, label=r"$f_{\mathrm{esa}}$")
        ax.set_yscale("log")
        ax.set_xlim(wavelength_min_nm, wavelength_max_nm)
        tick_step = 50 if wavelength_max_nm-wavelength_min_nm >= 300 else 5
        ax.set_xticks(np.arange(wavelength_min_nm, wavelength_max_nm+1, tick_step))
        ax.set_xlabel("Pompgolflengte (nm)", labelpad=10)
        ax.set_ylabel("Versterkingsfactor f (logaritmische schaal)", labelpad=10)
        positive = np.concatenate((f_gsa[f_gsa > 0], f_esa[f_esa > 0]))
        # Gaussstaarten kunnen onderlopen tot nul. Begrens alleen het zichtbare
        # bereik; de berekende/exporteerde factoren worden niet afgeknipt.
        lower = max(ABSORPTION_PLOT_MIN_FACTOR,
                    10.0**np.floor(np.log10(positive.min())))
        upper = max(lower*10.0, 10.0**np.ceil(np.log10(positive.max())))
        ax.set_ylim(lower, upper)
        ax.grid(axis="y", which="major", color="#dce3eb", linewidth=0.8)
        ax.grid(axis="x", which="major", color="#eef1f5", linewidth=0.6)
        ax.axvline(REFERENCE_WAVELENGTH_NM, color="#687684", linestyle="--",
                   linewidth=1, zorder=1)
        if SI_REFERENCE_ANCHOR:
            ax.axhline(1.0, color="#9ba6b1", linestyle=":", linewidth=1)
            ax.plot([REFERENCE_WAVELENGTH_NM], [1.0], "o", color="#465360", ms=5, zorder=5)
            ax.annotate("1064 nm: beide f = 1", xy=(REFERENCE_WAVELENGTH_NM, 1),
                        xytext=(wavelength_min_nm+0.03*(wavelength_max_nm-wavelength_min_nm),
                                80 if wavelength_max_nm-wavelength_min_nm >= 300
                                else max(f_gsa.max(), f_esa.max())*0.5),
                        color="#465360", fontsize=10,
                        arrowprops={"arrowstyle": "-", "color": "#687684"})
        ax.legend(loc="upper right", frameon=False, fontsize=13)
        if _USE_STARK and wavelength_min_nm < 1170 and wavelength_max_nm > 1290:
            peak = int(np.argmax(f_gsa))
            ax.annotate(r"GSA $^3H_6\rightarrow{}^3H_5$" + f"\npiek: {wavelengths[peak]:.1f} nm",
                        xy=(wavelengths[peak], f_gsa[peak]), xytext=(1320, f_gsa[peak]*0.65),
                        fontsize=10, color="#ba4a18",
                        arrowprops={"arrowstyle": "-", "color": "#ba4a18"})
        footer = ("GSA- en ESA-achtergrond: uit"
                  + f" | ESA sigma: {STARK_SIGMA_CM1:g} cm$^{{-1}}$"
                  + (f" | GSA H5 sigma: {GSA_H5_SIGMA_CM1:g} cm$^{{-1}}$"
                     if ABSORPTION_MODEL == "broad_gsa" else ""))
        fig.text(0.11, 0.035, footer, fontsize=10, color="#526171")

        os.makedirs(save_dir, exist_ok=True)
        basename = os.path.join(save_dir,
                               f"f_gsa_f_esa_{wavelength_min_nm:g}_{wavelength_max_nm:g}nm_{ABSORPTION_MODEL}")
        fig.savefig(basename + ".png", dpi=200, facecolor="white")
        fig.savefig(basename + ".svg", facecolor="white")
        np.savetxt(basename + ".csv", np.column_stack((wavelengths, f_gsa, f_esa)),
                   delimiter=",", header="wavelength_nm,f_gsa,f_esa", comments="", fmt="%.12g")
        print("GSA and ESA backgrounds: removed.")
        print(f"Factor plot and data saved: {basename}.png / .svg / .csv")
    return fig, wavelengths, f_gsa, f_esa


def main(spectra_only=False, save_dir=None):

    # Only I_values[0] and I_values[-1] matter for the adaptive
    # Gaussian/Stark line-shape scans: the points inside the window are chosen adaptively.
    I_values = np.logspace(
        0,
        np.log10(150.0),
        100
    )

    p8 = parameters_for_sample(3, lambda0_gsa_nm=GSA_BACKGROUND_LAMBDA0_NM)
    if save_dir is None:
        save_dir = os.path.dirname(os.path.abspath(__file__))
    plot_absorption_factors(p8, save_dir)
    plot_absorption_factors(p8, save_dir, 1040.0, 1100.0)
    if spectra_only:
        if _USE_STARK:
            check_thermal_population_coupling(p8)
        plt.show(block=False)
        return

    print(f"ABSORPTION_MODEL = {ABSORPTION_MODEL!r} ({ABSORPTION_LABEL})")
    if _USE_STARK:
        thermal_checks = check_thermal_population_coupling(p8)
        print(f"Host: {HOST_MATERIAL}; T = {TEMPERATURE_K:g} K; "
              f"population mapping [N0(3H6), N1(3F4), N2(3H4)].")
        print("Thermalization is assumed faster than excitation and depletion.")
        print("Thermal checks passed: both Boltzmann sums = 1; Stark populations "
              "sum to N0/N1; all five default stick sums = 1.")
        if "sigma_gsa_1064_m2" in thermal_checks:
            print(f"SI anchor: sigma_GSA = {thermal_checks['sigma_gsa_1064_m2']:.3e} m^2, "
                  f"sigma_ESA = {thermal_checks['sigma_esa_1064_m2']:.3e} m^2.")
        else:
            print("Geen SI-anker en geen brede achtergrond: pure Stark-absorptie.")
    else:
        print("Absorptiemodel: piekgenormeerde Gaussian met SI-amplitudes "
              f"(sigma_GSA = {p8.sigma_gsa_m2:.3e} m^2, "
              f"sigma_ESA = {p8.sigma_esa_m2:.3e} m^2); geen Stark-structuur.")

    wavelengths_nm = [
        1055,
        1060.0, 1064, 1065,
        1070.0, 1075,
        1080.0, 1085,
        1090.0, 1100, 1390,
        1400,1410,1420,1430, 1440,1450,1460,1470,1480,1490
    ]
    if INCLUDE_EXTRAPOLATED_WAVELENGTHS:
        wavelengths_nm += [1200.0, 1250.0, 1400.0, 1450.0, 1490.0, 1750.0, 1850.0]
        print("WARNING: extended scan is uncalibrated. Very large long-wave "
              "slopes can reflect an underestimated phonon-assisted GSA.")

    # ---------------------------------------------------------------
    # MAP INSTELLEN
    # ---------------------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nAlle grafieken worden opgeslagen in: {save_dir}\n")

    # Bestandsnaam-suffix zodat de drie modellen elkaar niet overschrijven
    sfx = f"_{ABSORPTION_MODEL}"

    # ---------------------------------------------------------------
    # Detuning-curves voor het gekozen absorptiemodel
    # ---------------------------------------------------------------

    (
        stark_curves,
        stark_thresholds,
        stark_ana_thresholds,
        stark_fit_lines
    ) = compute_detuning_curves(
        I_values,
        p8,
        wavelengths_nm,
        kind=ABSORPTION_KIND,
        label=ABSORPTION_LABEL
    )

    plot_wavelength_detuning(
        stark_curves,
        stark_thresholds,
        lineshape_label=ABSORPTION_LABEL,
        lineshape_kind=ABSORPTION_KIND,
        fit_lines=stark_fit_lines,
        savepath=os.path.join(save_dir, f"pa_wavelength_detuning{sfx}.pdf")
    )


    # ---------------------------------------------------------------
    # Separate summary plots versus wavelength
    # ---------------------------------------------------------------

    plot_smax_vs_wavelength(
        stark_fit_lines,
        savepath=os.path.join(save_dir, f"pa_smax_vs_wavelength{sfx}.pdf")
    )

    plot_ith_vs_wavelength(
        stark_thresholds,
        stark_ana_thresholds,  # Geef nu ook de analytische drempelwaarden mee
        savepath=os.path.join(save_dir, f"pa_ith_vs_wavelength{sfx}.pdf")
    )
    
    # ---------------------------------------------------------------
    # Rise-time plots
    # ---------------------------------------------------------------
    I_min, I_max = float(I_values[0]), float(I_values[-1])
    
    print("\n=== Rise time vs intensiteit ===")
    rise_results = compute_rise_vs_intensity(p8, I_min, I_max)
    if rise_results:
        plot_rise_vs_intensity(
            rise_results,
            savepath=os.path.join(save_dir, f"pa_rise_time_vs_intensity{sfx}.pdf")
        )
    else:
        print("  Geen geldige rise-time data: plot overgeslagen.")
    
    # De sigma-sweep bestaat alleen als er Stark-spectra zijn
    if _USE_STARK:
        print("\n=== Piek rise time vs Stark sigma ===")
        peaks_by_wl = {
            lam: compute_rise_peak_vs_sigma(p8, lam, RISE_SIGMA_VALUES_CM1, I_min, I_max)
            for lam in RISE_SIGMA_WAVELENGTHS_NM
        }
        plot_rise_peak_vs_sigma(
            RISE_SIGMA_VALUES_CM1, peaks_by_wl,
            savepath=os.path.join(save_dir, f"pa_rise_peak_vs_sigma{sfx}.pdf")
        )
    else:
        print("\nSigma-sweep overgeslagen (alleen zinvol met Stark-spectra).")

    # Display all figures together.
    # block=False prevents Spyder from remaining busy until the plot windows
    # are manually closed.
    plt.show(block=False)
    plt.pause(0.1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Corrected GSA/ESA wavelength model")
    parser.add_argument("--spectra-only", action="store_true",
                        help="Export the full and zoom factor plots; skip kinetics.")
    parser.add_argument("--output-dir", default=None, help="Directory for plots/data.")
    args = parser.parse_args()
    main(spectra_only=args.spectra_only, save_dir=args.output_dir)