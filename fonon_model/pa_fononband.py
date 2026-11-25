# -*- coding: utf-8 -*-
"""
Tm3+ in NaYF4: GSA/ESA with a phonon band, coupled to the existing PA kinetics.

Default ABSORPTION_MODEL="phonon_band": the same continuous, finite-T
Huang-Rhys kernel as in the supplied phonon script, for all five GSA/ESA
channels. The electronic Gaussian sigma is STARK_SIGMA_CM1=8.5 cm^-1; the
phonon sidebands replace the effective H5 width of 300 cm^-1.

    f_gsa(lambda) = sigma_gsa(lambda) / sigma_gsa(1064 nm)
    f_esa(lambda) = sigma_esa(lambda) / sigma_esa(1064 nm)
    w_i(lambda) = cross_section_to_rate_coeff(sigma_i_ref, lambda) * f_i(lambda)

Both factors are 1 at 1064 nm. They are applied exactly once in the pump
rates, for both the fast and the pa_model solver. The Boltzmann weights are
already contained in the spectra; the population terms remain R_gsa*N0 and
R_esa*N1. The rate equations, relaxation and transfer parameters are taken
from the supplied kinetics script.

Band settings, Huang-Rhys factors and electronic widths are listed below with
the model settings. The band shape and couplings are the uncalibrated
assumptions of the supplied phonon script, except the 3H6 -> 3H5 coupling: it
is calibrated so that the 1064-nm anchor (SI sigma_GSA) reproduces the
Judd-Ofelt integral of 3H6 -> 3H5 (see GSA_H5_HUANG_RHYS). By default the
phonon sidebands come from the four Gaussian bands that approximate the dotted
Yb/Tm Raman curve of Wilhelm et al. (2013, Fig. 3A) (250/306/364/408 cm^-1, the
same values as in the "1 fonon zijband" script). The bands are NOT sampled into
separate modes. Each band k is one Poisson process for phonon emission and one
for phonon absorption (Bose occupation at the band centre), and every phonon of
band k has a Gaussian energy N(w_k, s_k^2). The m-fold convolution of a
Gaussian is again a Gaussian, so m emitted and l absorbed phonons of band k
shift a line by N((m-l) w_k, (m+l) s_k^2), and the four bands are combined by
convolution (centres and variances add). With --band-profile gaussian a single
Gaussian band (centre 320 cm^-1) is used; with --phonon-model discrete three
sharp modes.
There is no absorption floor.

Channel weights follow from Judd-Ofelt: w_c ~ S_c * nu_c with
S_c = sum_t Omega_t [U(t)]^2 (Omega_t: SI Table S4 of Lee et al. 2021;
[U(t)]^2: SI Table S5 and Carnall et al. 1978, Appendix XIII Table 2).

The absorption spectrum is always saved twice, with a linear and with a
logarithmic y-axis (<name>_linear.png/.svg and <name>_log.png/.svg); both
figures show the same factors, which are written once to <name>.csv.

The original models "broad_gsa", "stark", "si_1064" and "hybrid" remain
available through ABSORPTION_MODEL. Only "broad_gsa" still uses
GSA_H5_SIGMA_CM1=300 cm^-1. A nonzero Lorentz width is explicitly rejected for
the phonon route: the supplied kernel uses Gaussian lines.

Usage (with this script in the same folder as your existing pa_model.py):
    python pa_fononband.py
    python pa_fononband.py --spectra-only --no-show
    python pa_fononband.py --data-only
    python pa_fononband.py --spectra-only --coupling-h5 0.08
    python pa_fononband.py --data-only --calibrate-h5             # refit S(3H6 -> 3H5)
    python pa_fononband.py --data-only --phonon-model discrete
    python pa_fononband.py --data-only --band-profile gaussian
    python pa_fononband.py --no-show --plots spectra ith          # only these
    python pa_fononband.py --plots risewl                         # rise time vs wavelength
    python pa_fononband.py --plots ith_full                       # analytical Ith 800-1600 nm

Plots are saved to disk and shown inline in an IPython console (VS Code
Interactive Window, Spyder); separate plot windows only with SHOW_WINDOWS.
    python pa_fononband.py --no-show --skip-plots sigma rise      # all except these
    python pa_fononband.py --no-show --scan-range 1050 1095 0.5   # other pump grid
    python pa_fononband.py --no-show --include-1450               # also 1390-1490 nm

The detuning scan covers the 1000-1100 nm measurement range in 1-nm steps
(SCAN_MIN_NM, SCAN_MAX_NM, SCAN_STEP_NM); the analytical threshold is also
evaluated every 0.1 nm. pa_detuning_summary_<model>.csv lists Ith, s and f per
pump wavelength for comparison with measurements.

Requirements: numpy; matplotlib for plots; scipy for the fast kinetics; the
existing pa_model.py for the population simulations. The default phonon
factors can be exported without pa_model (--data-only).
"""

from __future__ import annotations

import os  # Used for saving into the chosen output folder
import contextlib
import json
import logging
from pathlib import Path
from functools import lru_cache
from typing import Dict, Optional, Tuple

import numpy as np
try:
    import matplotlib
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    if exc.name != 'matplotlib':
        raise
    matplotlib = plt = None

try:
    from scipy.integrate import solve_ivp
    from scipy.optimize import brentq
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False

try:
    import pa_model
except ModuleNotFoundError as exc:
    # The existing pa_model imports matplotlib even when only its rates are
    # needed. Keep the numpy-only factor export available without that import.
    if exc.name != "pa_model" and not (exc.name == "matplotlib" and plt is None):
        raise
    pa_model = None
else:
    from pa_model import (
        emission_from_population,
        DEFAULT_LAMBDA0_ESA_NM,
        DEFAULT_SIGMA_LAMBDA_ESA_NM,
        parameters_for_sample,
        power_dependence,
        nonlinearity_slope,
        nonlinearity_slope_max_converged,
    )


def _require_pa_model():
    if pa_model is None:
        _require_matplotlib()
        raise RuntimeError(
            "Place your existing pa_model.py next to this script for the kinetics "
            "and the older absorption models. --data-only and --spectra-only "
            "work without this module with ABSORPTION_MODEL='phonon_band'.")


def _running_in_ipython():
    """True inside Spyder, Jupyter or the VS Code Interactive Window."""
    try:
        return get_ipython() is not None  # noqa: F821 (defined by IPython)
    except NameError:
        return False


def _require_matplotlib():
    if plt is None:
        raise RuntimeError("Plots require matplotlib; --data-only requires numpy only.")

_LOG_PLOT_FLOOR = 1e-300
# Lower limit of the log absorption plot only; no floor is added to the spectrum.
ABSORPTION_PLOT_MIN_FACTOR = 1e-20

# A real avalanche has a sharp knee with s_max of order 10-100 (local maximum
# about 70 for the 8% Tm sample at 1064 nm). Without an avalanche the
# emission is a pure power law with s ~ 2-3. Anything in between is a safe
# cut-off. The test uses the maximum local slope of the complete refined curve
# (all points of adaptive_knee_scan), not of the coarse grid alone.
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

# ---------------------------------------------------------------------------
# Which parts/plots a full run makes (True = on, False = off).
# Override on the command line with --plots ... (only these) or
# --skip-plots ... (all except these). Roughly in order of run time:
#   spectra   : absorption factors f_gsa/f_esa, always with a linear and a
#               log y-axis (fast, seconds)
#   detuning  : emission vs intensity per pump wavelength (minutes)
#   smax, ith : summaries of the detuning scan (they need the detuning
#               calculation; it also runs when 'detuning' itself is off,
#               but without the detuning figure)
#   ith_full  : analytical threshold only, 800-1600 nm (ITH_FULL_*; seconds)
#   rise      : 95% rise time vs intensity (seconds)
#   risewl    : longest rise time (at the threshold) vs pump wavelength
#               (about 0.5 s per wavelength; grid RISE_WL_*_NM)
#   sigma     : peak rise time vs electronic width (rebuilds the spectra
#               once for every width)
# ---------------------------------------------------------------------------
PLOTS = {
    "spectra": True,
    "detuning": True,
    "smax": True,
    "ith": True,
    "ith_full": True,
    "rise": True,
    "risewl": True,
    "sigma": True,
}

# ---------------------------------------------------------------------------
# Pump wavelengths of the detuning scan (detuning, smax, ith), in nm.
# Default: the measurement range 1000-1100 nm in 1-nm steps. The narrowest
# structure in the model (3F4 -> 3F2 ESA Stark lines, sigma 8.5 cm^-1) is
# about 2.3 nm FWHM near 1064 nm, so 1 nm samples it only coarsely (about two
# points per FWHM): interpolating the 1-nm scan deviates up to ~15% from the
# 0.1-nm analytical threshold. A finer step mainly costs time (about 0.15 s
# per wavelength without PA, 0.6 s with PA).
# The analytical threshold is also evaluated every ANALYTICAL_STEP_NM, which
# costs almost nothing and shows the structure between the scan points.
# Override on the command line with --scan-range MIN MAX STEP.
# ---------------------------------------------------------------------------
SCAN_MIN_NM, SCAN_MAX_NM, SCAN_STEP_NM = 1000.0, 1100.0, 1.0
ANALYTICAL_STEP_NM = 0.1
# Separate figure "ith_full": analytical threshold only, over a wide pump range.
ITH_FULL_MIN_NM, ITH_FULL_MAX_NM, ITH_FULL_STEP_NM = 800.0, 1600.0, 0.1
ITH_FULL_Y_MAX = 1e7  # kW/cm^2; cap of the y-axis (I_th diverges between the bands)
DETUNING_PLOT_STEP_NM = 5.0  # detuning figure: one curve per 5 nm (all are computed)
# Optional second range around the 3F4 -> 3H4 ESA resonance (Lee et al.: PA
# between 1400 and 1470 nm). Off by default; switch on with --include-1450.
INCLUDE_1450_BAND = False
BAND_1450_WAVELENGTHS_NM = [1390.0, 1400.0, 1410.0, 1420.0, 1430.0, 1440.0,
                            1450.0, 1460.0, 1470.0, 1480.0, 1490.0]


def _uniform_grid_nm(minimum, maximum, step):
    """Uniform wavelength grid including both ends; maximum-minimum must fit the step."""
    values = np.array([minimum, maximum, step], dtype=float)
    if not np.all(np.isfinite(values)) or not 0 < minimum < maximum or step <= 0:
        raise ValueError("Wavelength range needs finite 0 < min < max and a positive step.")
    count = int(round((maximum - minimum) / step))
    if not np.isclose(minimum + count * step, maximum, rtol=0, atol=1e-9):
        raise ValueError("Wavelength max minus min must be a multiple of the step.")
    return np.round(minimum + step * np.arange(count + 1), 9)


def detuning_wavelengths():
    """Pump wavelengths of the detuning scan, sorted and without duplicates."""
    wavelengths = list(_uniform_grid_nm(SCAN_MIN_NM, SCAN_MAX_NM, SCAN_STEP_NM))
    if INCLUDE_1450_BAND:
        wavelengths += BAND_1450_WAVELENGTHS_NM
    if INCLUDE_EXTRAPOLATED_WAVELENGTHS:
        wavelengths += [1200.0, 1250.0, 1400.0, 1450.0, 1490.0, 1750.0, 1850.0]
    return sorted({round(float(lam), 9) for lam in wavelengths})


def analytical_wavelength_ranges():
    """Ranges (min, max) on which the dense analytical threshold is evaluated."""
    ranges = [(SCAN_MIN_NM, SCAN_MAX_NM)]
    if INCLUDE_1450_BAND:
        ranges.append((min(BAND_1450_WAVELENGTHS_NM), max(BAND_1450_WAVELENGTHS_NM)))
    return ranges
T_STEADY_S = 20.0     # initial integration duration [s]
MAX_STEADY_TIME_S = 1280.0
STEADY_RELATIVE_TOL = 1e-6
STEADY_POPULATION_FLOOR = 1e-18

# Straight-line fit around EACH curve's steepest rise (explicit fitting
# criteria, not physical model parameters). SLOPE_FRACTION=None: the widest
# window that meets all FIT_* criteria ("verified"). SLOPE_FRACTION=f: the
# window where the local slope >= f * peak; the FIT_* criteria are then only
# reported (fit["verified"]). The slope depends strongly on the window: at
# 1064 nm the local maximum is ~70, f=0.25 gives ~42 (not verified) and a
# +/-0.10-decade window ~21 (Lee et al.: s = 22.2-22.6).
SLOPE_FRACTION = 0.25  # None: automatic; set e.g. 0.9 for a fixed-window comparison.
FIT_MAX_ERROR_DEX = 0.03  # at most about 7.2% vertical discrepancy
FIT_MIN_R2 = 0.999
FIT_MIN_PEAK_RATIO = 0.95  # fitted slope at least 95% of local peak
FIT_MIN_POINTS = 7
FIT_MIN_SPAN_DEX = 0.5  # at least a factor sqrt(10) in emission
LINE_EXTEND_DECADES = 0.0  # draw the line only over the fitted window


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
# ABSORPTION MODEL: exactly one model active at a time
#   "phonon_band": electronic Gaussian lines + phonon sidebands, 1064-nm anchor
#   "broad_gsa": GSA H5 sigma 300, GSA/ESA without background, anchor kept
#   "stark"   : Stark sticks only (no background, no 1064-nm anchor)
#   "si_1064" : Gaussian line shape (pa_model), SI cross sections at 1064 nm
#   "hybrid"  : narrow Stark lines + 1064-nm anchor; no background
# ---------------------------------------------------------------------------
ABSORPTION_MODEL = "phonon_band"  # "phonon_band", "broad_gsa", "stark", "si_1064", "hybrid"

if ABSORPTION_MODEL not in ("phonon_band", "stark", "si_1064", "hybrid", "broad_gsa"):
    raise ValueError("Unknown ABSORPTION_MODEL; use phonon_band, broad_gsa, stark, si_1064 or hybrid.")

_USE_STARK = ABSORPTION_MODEL in ("phonon_band", "stark", "hybrid", "broad_gsa")   # Use Stark spectra?
ABSORPTION_KIND = "stark_gaussian" if _USE_STARK else "gaussian"
ABSORPTION_LABEL = {
    "phonon_band": "GSA + ESA with phonon band + 1064-nm anchor",
    "stark": "Stark GSA + ESA (no background)",
    "si_1064": "Gaussian line shape, SI cross sections at 1064 nm",
    "hybrid": "Stark without background + 1064-nm anchor",
    "broad_gsa": "Broadened GSA + ESA without background + 1064-nm anchor",
}[ABSORPTION_MODEL]

# Explicit model parameters (not linewidths/strengths supplied by Table 4).
STARK_SIGMA_CM1 = 8.5  # ESA and unmodified GSA channels
GSA_H5_SIGMA_CM1 = 300.0  # Gaussian sigma of 3H6 -> 3H5 only, cm^-1

# Channel weights from Judd-Ofelt (replace the former weights 1.0).
#   S_c = sum_t Omega_t [U(t)]^2_c   (line strength, 1e-20 cm^2)
#   w_c = S_c * nu_c                 (nu_c: gap between the multiplet barycentres;
#                                     sigma_int ~ nu * S)
# The weights are normalized within a spectrum; because of the 1064-nm anchor
# only the ratios between channels of the same spectrum matter.
# Electric-dipole part only: the magnetic-dipole parts of 3H6 -> 3H5 and
# 3F4 -> 3F3 (Delta J = 1) are omitted. With the anchor this only matters for
# the ESA above about 1095 nm.
JO_OMEGA_1E20_CM2 = np.array([2.37, 3.05, 0.41])  # Omega_2,4,6: SI Table S4 (beta-NaGdF4:Tm3+)
JO_REDUCED_MATRIX_ELEMENTS = {  # [U(2)]^2, [U(4)]^2, [U(6)]^2
    "3H6 -> 3F4": (0.5395, 0.7261, 0.2421),  # SI Table S5
    "3H6 -> 3H5": (0.1074, 0.2314, 0.6385),  # SI Table S5
    "3F4 -> 3H4": (0.1275, 0.1311, 0.2113),  # SI Table S5
    "3F4 -> 3F3": (0.0025, 0.0005, 0.1688),  # Carnall et al. 1978, App. XIII Table 2 (LaF3)
    "3F4 -> 3F2": (0.3026, 0.0562, 0.0440),  # Carnall et al. 1978, App. XIII Table 2 (LaF3)
}
JO_USE_FREQUENCY_FACTOR = True  # False: w_c = S_c (without nu_c)
USE_JUDD_OFELT_WEIGHTS = True   # False: all channel weights 1.0 (old behaviour)
# Absolute Judd-Ofelt integral, only for the anchor check and the 3H6 -> 3H5
# calibration (the channel weights above need ratios only).
JO_REFRACTIVE_INDEX = 1.48  # beta-NaYF4 (assumption); local field (n^2+2)^2/(9n)
E_CHARGE_ESU, H_PLANCK_CGS, C_LIGHT_CGS = 4.80320471e-10, 6.62607015e-27, 2.99792458e10
MULTIPLET_J = {"3H6": 6, "3F4": 4, "3H5": 5, "3H4": 4, "3F3": 3, "3F2": 2}
# SI cross sections at 1064 nm (Lee et al. 2021, SI Table S6), m^2.
SI_SIGMA_GSA_1064_M2 = 6.0e-29
SI_SIGMA_ESA_1064_M2 = 6.4e-25


def judd_ofelt_line_strength(name):
    """Line strength S_c in 1e-20 cm^2."""
    return float(JO_OMEGA_1E20_CM2 @ np.asarray(JO_REDUCED_MATRIX_ELEMENTS[name], dtype=float))


def judd_ofelt_weight(name, lower_cm1, upper_cm1):
    """Relative channel weight S_c * nu_c (arbitrary scale)."""
    if not USE_JUDD_OFELT_WEIGHTS:
        return 1.0
    weight = judd_ofelt_line_strength(name)
    if JO_USE_FREQUENCY_FACTOR:
        weight *= float(np.mean(upper_cm1) - np.mean(lower_cm1))
    return weight


def judd_ofelt_integrated_cross_section(name, lower_cm1, upper_cm1):
    """Electric-dipole integral of sigma over wavenumber, in cm^2 cm^-1 = cm.

    int sigma dnu = 8 pi^3 e^2 nu / (3 h c (2J+1)) * (n^2+2)^2/(9n) * S_ED  (CGS)
    with nu the gap between the multiplet barycentres, J that of the lower
    multiplet and n = JO_REFRACTIVE_INDEX.
    """
    n = JO_REFRACTIVE_INDEX
    nu = float(np.mean(upper_cm1) - np.mean(lower_cm1))
    two_j_plus_1 = 2 * MULTIPLET_J[name.split(" -> ")[0]] + 1
    line_strength_cm2 = judd_ofelt_line_strength(name) * 1e-20
    return (8.0 * np.pi**3 * nu / (3.0 * H_PLANCK_CGS * C_LIGHT_CGS * two_j_plus_1)
            * (n**2 + 2.0)**2 / (9.0 * n) * E_CHARGE_ESU**2 * line_strength_cm2)


# Both separate backgrounds have been removed completely from StarkSpectrum.
# Nonzero weights are rejected; these values only report the setting.
ESA_BACKGROUND_WEIGHT = 0.0
GSA_BACKGROUND_WEIGHT = 0.0

# SI anchor (1064 nm): in "phonon_band", "hybrid" and "broad_gsa"; in "si_1064"
# pa_model normalizes its Gaussian factors at 1064 nm itself. "stark" has no anchor.
SI_REFERENCE_ANCHOR = ABSORPTION_MODEL in ("phonon_band", "hybrid", "broad_gsa", "si_1064")
REFERENCE_WAVELENGTH_NM = 1064.0
# Historical names: only the normalization grid and Gaussian envelope in si_1064.
# These centers/widths add no background to the Stark spectra.
ESA_BACKGROUND_LAMBDA0_NM = getattr(pa_model, "DEFAULT_LAMBDA0_ESA_NM", None)
ESA_BACKGROUND_SIGMA_NM = getattr(pa_model, "DEFAULT_SIGMA_LAMBDA_ESA_NM", None)
# 3H5 lies BELOW the photon energy at 1064 nm: subtract the mismatch.
# Lee et al., Methods and Fig. 1c: approximately 1200 cm^-1 mismatch.
GSA_ENERGY_MISMATCH_CM1 = 1200.0
GSA_BACKGROUND_LAMBDA0_NM = 1e7 / (
    1e7 / REFERENCE_WAVELENGTH_NM - GSA_ENERGY_MISMATCH_CM1
)  # 1219.7 nm; same corrected center in all absorption modes
GSA_BACKGROUND_SIGMA_NM = getattr(pa_model, "DEFAULT_SIGMA_LAMBDA_GSA_NM", None)

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


# ---------------------------------------------------------------------------
# Phonon sidebands: Gaussian phonon bands and their convolution
# ---------------------------------------------------------------------------
# PHONON_MODEL "band": the electron-phonon coupling is spread over Gaussian
# phonon bands. Profile "four_gaussian" (default): the four Gaussians that
# approximate the dotted Yb/Tm Raman curve of Wilhelm et al. 2013, Fig. 3A
# (https://doi.org/10.7150/thno.5113); shape proxy, not a measured phonon DOS.
# Same centres, sigmas and heights as in the "1 fonon zijband" script; sigma
# is a standard deviation, not a FWHM. Each band is used as a whole (no
# sampling into separate modes) and gets the share A_k*s_k / sum_j(A_j*s_j)
# of the total Huang-Rhys factor, i.e. its area under the Raman curve.
# Profile "gaussian": one Gaussian band (BAND_MEAN_CM1, BAND_SIGMA_CM1).
# PHONON_MODEL "discrete": sharp modes (sigma 0) at DISCRETE_ENERGIES_CM1.
PHONON_MODEL = "band"
BAND_PROFILE = "four_gaussian"  # "four_gaussian" or "gaussian"
FOUR_GAUSSIAN_CENTERS_CM1 = (250.0, 306.0, 364.0, 408.0)
FOUR_GAUSSIAN_SIGMAS_CM1 = (14.0, 24.0, 15.0, 24.0)
FOUR_GAUSSIAN_HEIGHTS = (0.68, 0.95, 0.66, 0.36)
BAND_MEAN_CM1 = 320.0    # Only for profile "gaussian": band centre.
BAND_SIGMA_CM1 = 45.0    # Only for profile "gaussian": band sigma (assumption).
DISCRETE_ENERGIES_CM1 = (250.0, 305.0, 365.0)
DISCRETE_FRACTIONS = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)

# Total Huang-Rhys factors per channel. Only 3H6 -> 3H5 is calibrated; the
# others are uncalibrated assumptions.
# 3H6 -> 3H5 sets the whole GSA tail at 1000-1100 nm. It is chosen so that the
# 1064-nm anchor (SI sigma_GSA) and Judd-Ofelt give the same integrated
# 3H6 -> 3H5 cross section (calibrate_gsa_h5_huang_rhys(): 0.1119 for the
# default four_gaussian band at 300 K). Electric-dipole part only, n = 1.48;
# a magnetic-dipole part would lower S. A factor 2 in the JO integral gives
# 0.087-0.144. Recalibrate with --calibrate-h5 after changing the phonon band.
GSA_H5_HUANG_RHYS = 0.112
GSA_F4_HUANG_RHYS = 0.05  # To be fitted separately for 3H6 -> 3F4.
ESA_H4_HUANG_RHYS = 0.05  # 3F4 -> 3H4.
ESA_F3_HUANG_RHYS = 0.05  # 3F4 -> 3F3.
ESA_F2_HUANG_RHYS = 0.05  # 3F4 -> 3F2.
# Step of the shift grid on which the band convolution is evaluated. This is a
# quadrature step, not a phonon-mode spacing: every Gaussian is evaluated
# analytically at the grid points, and the sum of a Gaussian of sigma s over a
# grid of step h equals its integral to a relative 2*exp(-2 pi^2 s^2 / h^2)
# (Poisson summation). phonon_grid_error_bound() checks this for every channel.
PHONON_GRID_STEP_CM1 = 1.0
PHONON_GRID_MAX_RELATIVE_ERROR = 1e-30  # Reject a grid step that is too coarse.
PHONON_TAIL_TOLERANCE = 1e-35  # Upper bound on the missing total kernel mass.
PHONON_MAX_EVENTS_PER_BAND = 256  # Safety limit; exceeding it is an error.
REFERENCE_MAX_RELATIVE_KERNEL_ERROR = 1e-8  # Reject an insufficiently resolved anchor.

# Filled by set_phonon_model(); all kernel functions read these globals.
# One entry per band: (centre_cm1, sigma_cm1, fraction of the Huang-Rhys factor).
PHONON_BANDS = ()
PHONON_DESCRIPTION = ""
PHONON_INFO = {}

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
GSA_WEIGHT_3F4 = judd_ofelt_weight("3H6 -> 3F4", E_3H6_CM1, E_3F4_CM1)
GSA_WEIGHT_3H5 = judd_ofelt_weight("3H6 -> 3H5", E_3H6_CM1, E_3H5_CM1)
ESA_WEIGHT_3H4 = judd_ofelt_weight("3F4 -> 3H4", E_3F4_CM1, E_3H4_CM1)
ESA_WEIGHT_3F3 = judd_ofelt_weight("3F4 -> 3F3", E_3F4_CM1, E_3F3_CM1)
ESA_WEIGHT_3F2 = judd_ofelt_weight("3F4 -> 3F2", E_3F4_CM1, E_3F2_CM1)
GSA_CHANNELS = {
    "3H6 -> 3F4": (E_3H6_CM1, E_3F4_CM1, GSA_WEIGHT_3F4),
    "3H6 -> 3H5": (E_3H6_CM1, E_3H5_CM1, GSA_WEIGHT_3H5),
}
ESA_CHANNELS = {
    "3F4 -> 3H4": (E_3F4_CM1, E_3H4_CM1, ESA_WEIGHT_3H4),
    "3F4 -> 3F3": (E_3F4_CM1, E_3F3_CM1, ESA_WEIGHT_3F3),
    "3F4 -> 3F2": (E_3F4_CM1, E_3F2_CM1, ESA_WEIGHT_3F2),
}


def channel_weight_report():
    """Line strengths and normalized channel shares per spectrum."""
    report = {}
    for spectrum, channels in (("GSA", GSA_CHANNELS), ("ESA", ESA_CHANNELS)):
        total = sum(w for _, _, w in channels.values())
        for name, (_, _, weight) in channels.items():
            report[name] = {"spectrum": spectrum,
                            "line_strength_1e-20_cm2": judd_ofelt_line_strength(name),
                            "weight": float(weight),
                            "share_in_spectrum": float(weight / total)}
    return report


def set_phonon_model(model=PHONON_MODEL, profile=BAND_PROFILE,
                     mean_cm1=BAND_MEAN_CM1, sigma_cm1=BAND_SIGMA_CM1):
    """Set PHONON_BANDS: (centre, sigma, fraction of S) for every phonon band.

    "band" + "four_gaussian": the four Raman Gaussians, each used as one band.
    Its share of the Huang-Rhys factor is its area A_k*s_k*sqrt(2 pi); the
    common factor sqrt(2 pi) cancels in the normalization.
    "band" + "gaussian": one Gaussian band at mean_cm1 with sigma_cm1.
    "discrete": sharp modes (sigma 0) at DISCRETE_ENERGIES_CM1.
    """
    global PHONON_BANDS, PHONON_DESCRIPTION, PHONON_INFO
    if model == "discrete":
        centers = np.asarray(DISCRETE_ENERGIES_CM1, dtype=float)
        sigmas = np.zeros_like(centers)
        fractions = np.asarray(DISCRETE_FRACTIONS, dtype=float)
    elif model == "band" and profile == "four_gaussian":
        centers = np.asarray(FOUR_GAUSSIAN_CENTERS_CM1, dtype=float)
        sigmas = np.asarray(FOUR_GAUSSIAN_SIGMAS_CM1, dtype=float)
        heights = np.asarray(FOUR_GAUSSIAN_HEIGHTS, dtype=float)
        if (centers.shape != sigmas.shape or centers.shape != heights.shape
                or np.any(sigmas <= 0) or np.any(heights < 0)):
            raise ValueError("Four-Gaussian band: matching centres, positive sigmas "
                             "and nonnegative heights required.")
        fractions = heights * sigmas  # area of each Raman Gaussian (without sqrt(2 pi))
    elif model == "band" and profile == "gaussian":
        if not np.isfinite(sigma_cm1) or sigma_cm1 <= 0:
            raise ValueError("band-sigma must be positive.")
        centers = np.array([mean_cm1], dtype=float)
        sigmas = np.array([sigma_cm1], dtype=float)
        fractions = np.ones(1)
    elif model == "band":
        raise ValueError("band profile must be 'four_gaussian' or 'gaussian'.")
    else:
        raise ValueError("phonon model must be 'band' or 'discrete'.")
    if (centers.ndim != 1 or not len(centers) or fractions.shape != centers.shape
            or not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sigmas))
            or not np.all(np.isfinite(fractions)) or np.any(sigmas < 0)
            or np.any(fractions < 0) or fractions.sum() <= 0):
        raise ValueError("Phonon bands need finite centres, sigmas >= 0 and "
                         "nonnegative fractions with a positive sum.")
    if np.any(centers <= 0) or np.any(centers <= 5 * sigmas):
        raise ValueError("Each band centre must be positive and exceed 5 sigma, so "
                         "that the Gaussian band does not reach zero phonon energy.")
    keep = fractions > 0  # Bands without weight contribute nothing.
    centers, sigmas, fractions = centers[keep], sigmas[keep], fractions[keep]
    fractions = fractions / fractions.sum()
    PHONON_BANDS = tuple((float(c), float(s), float(f))
                         for c, s, f in zip(centers, sigmas, fractions))
    mean = float(np.sum(fractions * centers))
    # Spread of the one-phonon energy: band widths plus spread of the centres.
    spread = float(np.sqrt(np.sum(fractions * (sigmas**2 + centers**2)) - mean**2))
    if model == "discrete":
        description = ("discrete " + "/".join(f"{c:g}" for c in centers)
                       + " cm⁻¹ (sharp modes)")
    else:
        description = (f"{len(centers)} Gaussian band(s) "
                       + "/".join(f"{c:g}" for c in centers)
                       + f" cm⁻¹, convolved (mean {mean:.1f} cm⁻¹)")
    PHONON_DESCRIPTION = description
    PHONON_INFO = {
        "phonon_model": model,
        "description": description,
        "profile": profile if model == "band" else "discrete",
        "number_of_bands": len(centers),
        "bands": [{"center_cm1": float(c), "sigma_cm1": float(s),
                   "fraction_of_huang_rhys_factor": float(f),
                   "bose_occupation_at_center": phonon_occupation(float(c), TEMPERATURE_K)}
                  for c, s, f in zip(centers, sigmas, fractions)],
        "weighted_mean_cm1": mean,
        "one_phonon_std_cm1": spread,
        "grid_step_cm1": float(PHONON_GRID_STEP_CM1),
        "method": ("Per band: Poisson emission and absorption (Bose occupation at the "
                   "band centre); m emitted and l absorbed phonons give the Gaussian "
                   "N((m-l)w_k, (m+l)s_k^2); bands combined by direct convolution."),
    }
    if model == "band" and profile == "four_gaussian":
        PHONON_INFO["four_gaussian"] = {
            "centers_cm1": list(FOUR_GAUSSIAN_CENTERS_CM1),
            "sigmas_cm1": list(FOUR_GAUSSIAN_SIGMAS_CM1),
            "heights": list(FOUR_GAUSSIAN_HEIGHTS),
            "source": "Wilhelm et al. 2013, fig. 3A (Yb/Tm), https://doi.org/10.7150/thno.5113",
        }
    return PHONON_INFO


def phonon_occupation(energy_cm1, temperature_k):
    """Bose occupation; T=0 is supported for the phonon kernel itself."""
    if (not np.isfinite(temperature_k) or temperature_k < 0
            or not np.isfinite(energy_cm1) or energy_cm1 <= 0):
        raise ValueError("Phonon energy must be > 0 and temperature >= 0.")
    if temperature_k == 0:
        return 0.0
    exponent = energy_cm1 / (K_B_CM1_PER_K * temperature_k)
    return 0.0 if exponent > 700 else float(1.0 / np.expm1(exponent))


set_phonon_model()  # Default: the four Raman Gaussians, each used as one band.


def _poisson_tail_bound(mean, order, probability):
    """Upper bound on P(N > order), avoiding cancellation of 1-CDF."""
    if mean == 0:
        return 0.0
    ratio = mean / (order + 2)
    if ratio >= 1:
        return float("inf")
    return probability * mean / (order + 1) / (1.0 - ratio)


def _poisson_pair(mean_emit, mean_absorb, tolerance):
    """Use the SAME event cutoff for both signs, preserving detailed balance."""
    if max(mean_emit, mean_absorb) > 100:
        raise ValueError("Phonon event mean exceeds the supported range (100).")
    emit = [float(np.exp(-mean_emit))]
    absorb = [float(np.exp(-mean_absorb))]
    for order in range(PHONON_MAX_EVENTS_PER_BAND + 1):
        missing = (_poisson_tail_bound(mean_emit, order, emit[-1])
                   + _poisson_tail_bound(mean_absorb, order, absorb[-1]))
        if missing <= tolerance:
            return np.asarray(emit), np.asarray(absorb), missing
        emit.append(emit[-1] * mean_emit / (order + 1))
        absorb.append(absorb[-1] * mean_absorb / (order + 1))
    raise ValueError("Phonon expansion did not converge; revise cutoff or parameters.")


def _band_shift_masses(center, sigma, strength, temperature_k, step_cm1, tolerance):
    """Shift distribution of ONE phonon band, as probability masses on a grid.

    m ~ Poisson(S_k (n_k + 1)) phonons are emitted and l ~ Poisson(S_k n_k)
    absorbed, with n_k the Bose occupation at the band centre. The m-fold
    convolution of the Gaussian band N(w_k, s_k^2) is N(m w_k, m s_k^2), so the
    pair (m, l) shifts a line by the Gaussian N((m - l) w_k, (m + l) s_k^2).
    (m, l) = (0, 0) is the exact zero-phonon delta at shift 0. Each Gaussian is
    evaluated analytically at the grid points and multiplied by the step (mass
    per grid point). A sharp mode (sigma 0) puts its masses exactly at
    (m - l) w_k, which must then be a multiple of the step.

    Returns (index_of_first_entry, masses, omitted_mass_bound, max_events).
    """
    n = phonon_occupation(center, temperature_k)
    emit, absorb, missing = _poisson_pair(strength * (n + 1), strength * n, tolerance)
    order = len(emit) - 1
    if sigma == 0:
        stride = center / step_cm1
        if not np.isclose(stride, round(stride), rtol=0, atol=1e-9):
            raise ValueError("Sharp phonon modes must be multiples of PHONON_GRID_STEP_CM1.")
        stride = int(round(stride))
        masses = np.zeros(2 * order * stride + 1)
        for m, p_emit in enumerate(emit):
            for l, p_absorb in enumerate(absorb):
                masses[(m - l + order) * stride] += p_emit * p_absorb
        return -order * stride, masses, missing, order
    # Gaussian band: the farthest and widest pairs set the extent of the grid;
    # beyond 40 sigma every Gaussian underflows to exactly zero in float64.
    half = int(np.ceil((order * center + 40.0 * sigma * np.sqrt(2 * order)) / step_cm1))
    x = step_cm1 * np.arange(-half, half + 1)
    masses = np.zeros(len(x))
    masses[half] = emit[0] * absorb[0]  # zero-phonon delta (Debye-Waller factor)
    for m, p_emit in enumerate(emit):
        for l, p_absorb in enumerate(absorb):
            weight = p_emit * p_absorb
            if (m == 0 and l == 0) or weight == 0.0:
                continue
            mean = (m - l) * center
            width = sigma * np.sqrt(m + l)
            lo = np.searchsorted(x, mean - 40.0 * width, side="left")
            hi = np.searchsorted(x, mean + 40.0 * width, side="right")
            z = (x[lo:hi] - mean) / width
            masses[lo:hi] += (weight * step_cm1 / (np.sqrt(2.0 * np.pi) * width)
                              * np.exp(-0.5 * z * z))
    return -half, masses, missing, order


@lru_cache(maxsize=32)
def _cached_phonon_kernel(bands, strengths, temperature_k, step_cm1, tolerance):
    if (len(bands) != len(strengths) or not bands
            or not np.all(np.isfinite(strengths)) or min(strengths) < 0
            or not np.isfinite(step_cm1) or step_cm1 <= 0
            or not np.isfinite(temperature_k) or temperature_k < 0
            or not np.isfinite(tolerance) or not 0 < tolerance < 1):
        raise ValueError("Invalid phonon kernel parameters.")
    kernel = np.array([1.0])
    first_index = 0
    missing_bound = 0.0
    orders = []
    for (center, sigma, _fraction), strength in zip(bands, strengths):
        start, masses, missing, order = _band_shift_masses(
            center, sigma, strength, temperature_k, step_cm1, tolerance / len(bands))
        nonzero = np.flatnonzero(masses)  # trim exact zeros (float underflow) only
        masses = masses[nonzero[0]:nonzero[-1] + 1]
        start += int(nonzero[0])
        # Convolution of the band distributions by direct summation of
        # nonnegative products (np.convolve, no FFT): far tails suffer no FFT
        # round-off. Shifts near the event cutoff miss higher-order terms and
        # lose relative precision, but only at masses far below the tolerance.
        kernel = np.convolve(kernel, masses)
        first_index += start
        missing_bound += missing  # Union bound for all omitted event tails.
        orders.append(order)
    kernel /= kernel.sum()  # Conditional normalization; correction bounded above.
    shifts = (first_index + np.arange(len(kernel))) * step_cm1
    positive = kernel > 0  # Remove exact zeros only; no amplitude threshold/floor.
    shifts, kernel = shifts[positive], kernel[positive]
    shifts.setflags(write=False)
    kernel.setflags(write=False)
    return shifts, kernel, float(missing_bound), tuple(orders)


def phonon_kernel(total_coupling, temperature_k=TEMPERATURE_K,
                  tolerance=PHONON_TAIL_TOLERANCE):
    """Distribution of the total phonon shift for a channel with Huang-Rhys S."""
    if not PHONON_BANDS or not np.isfinite(total_coupling) or total_coupling < 0:
        raise ValueError("Set the phonon bands first; the coupling must be >= 0.")
    strengths = tuple(float(total_coupling * fraction) for _, _, fraction in PHONON_BANDS)
    return _cached_phonon_kernel(PHONON_BANDS, strengths, float(temperature_k),
                                 float(PHONON_GRID_STEP_CM1), float(tolerance))


def no_phonon_event_weight(total_coupling, temperature_k=TEMPERATURE_K):
    """Debye-Waller factor exp(-sum_k S_k (2 n_k + 1)): no phonon event at all."""
    return float(np.exp(-sum(
        total_coupling * fraction * (2.0 * phonon_occupation(center, temperature_k) + 1.0)
        for center, _sigma, fraction in PHONON_BANDS)))


def phonon_grid_error_bound(electronic_sigma_cm1):
    """Relative quadrature error caused by the shift grid (Poisson summation).

    The grid only enters in convolutions of smooth Gaussians: band with band
    (np.convolve) and band with the electronic line (gaussian_density). For two
    Gaussians the integrand is a Gaussian of variance s1^2 s2^2 / (s1^2 + s2^2),
    and its sum over a grid of step h has relative error <= 2 exp(-2 pi^2 s^2/h^2).
    Within one band the m-fold convolutions are analytic (no grid). Sharp modes
    and the zero-phonon delta sit exactly on grid points (no error).
    """
    band_sigmas = [s for _, s, _ in PHONON_BANDS if s > 0]
    if not band_sigmas:
        return 0.0
    pairs = [(a, b) for i, a in enumerate(band_sigmas) for b in band_sigmas[i + 1:]]
    pairs += [(a, electronic_sigma_cm1) for a in band_sigmas]
    variance = min(a * a * b * b / (a * a + b * b) for a, b in pairs)
    return float(2.0 * np.exp(-2.0 * np.pi**2 * variance / PHONON_GRID_STEP_CM1**2))


@lru_cache(maxsize=64)
def _cached_broadened_sticks(lower, upper, temperature_k, coupling, tolerance,
                             bands, step_cm1, boltzmann_mode):
    """Every Stark stick combined with every phonon shift (key holds all inputs)."""
    gaps, line_weights = transition_sticks(np.asarray(lower), np.asarray(upper),
                                           temperature_k)
    shifts, probabilities, missing, orders = phonon_kernel(coupling, temperature_k, tolerance)
    weights = (line_weights.ravel()[:, None] * probabilities[None, :]).ravel()
    gap_index = gaps.ravel() / step_cm1
    if np.array_equal(gap_index, np.rint(gap_index)):
        # Gaps on the grid as well: merge coincident lines exactly by index.
        shift_index = np.rint(shifts / step_cm1).astype(np.int64)
        index = (np.rint(gap_index).astype(np.int64)[:, None]
                 + shift_index[None, :]).ravel()
        base = int(index.min())
        combined = np.bincount(index - base, weights=weights)
        unique = (base + np.arange(len(combined))) * step_cm1
    else:
        positions = (gaps.ravel()[:, None] + shifts[None, :]).ravel()
        unique, inverse = np.unique(positions, return_inverse=True)
        combined = np.bincount(inverse, weights=weights, minlength=len(unique))
    positive = combined > 0  # No binning of electronic energies; exact zeros only.
    unique, combined = unique[positive], combined[positive]
    unique.setflags(write=False)
    combined.setflags(write=False)
    return unique, combined, missing, orders


def broadened_transition_sticks(lower, upper, temperature_k, coupling, tolerance):
    """Stark sticks combined with all phonon shifts (cached; independent of sigma)."""
    return _cached_broadened_sticks(
        tuple(float(e) for e in np.ravel(lower)), tuple(float(e) for e in np.ravel(upper)),
        float(temperature_k), float(coupling), float(tolerance), PHONON_BANDS,
        float(PHONON_GRID_STEP_CM1), BOLTZMANN_MODE)


def gaussian_density(energy_cm1, sticks_cm1, weights, sigma_cm1):
    """Analytic unit-area Gaussian mixture; no FFT noise or numerical floor.

    A distance >40 sigma has exp(-800), which underflows to zero in float64.
    Restricting each sorted query chunk to that interval changes no nonzero
    float64 Gaussian contribution and keeps memory bounded.
    """
    energy = np.asarray(energy_cm1, dtype=float)
    if not np.all(np.isfinite(energy)) or not np.isfinite(sigma_cm1) or sigma_cm1 <= 0:
        raise ValueError("Energy must be finite and sigma must be positive.")
    flat = energy.ravel()
    order = np.argsort(flat)
    sorted_energy = flat[order]
    result = np.empty_like(flat)
    for start in range(0, len(flat), 256):
        values = sorted_energy[start:start + 256]
        left = np.searchsorted(sticks_cm1, values[0] - 40 * sigma_cm1, side="left")
        right = np.searchsorted(sticks_cm1, values[-1] + 40 * sigma_cm1, side="right")
        delta = (values[:, None] - sticks_cm1[None, left:right]) / sigma_cm1
        result[order[start:start + 256]] = np.sum(
            np.exp(-0.5 * delta**2) * weights[None, left:right], axis=1)
    result /= np.sqrt(2 * np.pi) * sigma_cm1
    return float(result[0]) if energy.ndim == 0 else result.reshape(energy.shape)


def merge_sticks(sticks, weights):
    unique, inverse = np.unique(sticks, return_inverse=True)
    return unique, np.bincount(inverse, weights=weights, minlength=len(unique))


class PhononSpectrum:
    """Spectrum with per-channel phonon coupling and width.

    Besides the full spectrum (with phonon sidebands), the zero-phonon line
    (purely electronic transitions) is stored per channel, so that it can be
    plotted with the same normalization (see electronic_only).

    normalize_at_nm=None: global peak normalization (dense grid over the full
    sideband support; slow, several seconds). With a wavelength, the spectrum
    is divided by its own value there instead. With the 1064-nm anchor both
    give the same anchored factors, because the anchor divides any constant
    normalization out again; the direct route skips the dense peak search.
    """
    def __init__(self, channels, temperature_k=TEMPERATURE_K,
                 channel_couplings=None, channel_sigmas_cm1=None,
                 tolerance=PHONON_TAIL_TOLERANCE, normalize_at_nm=None):
        self.channels = channels
        self.background_weight = 0.0  # Same diagnostics interface as StarkSpectrum.
        self.temperature_k = temperature_k
        self.channel_couplings = dict(channel_couplings or {})
        self.channel_sigmas_cm1 = dict(channel_sigmas_cm1 or {})
        if (set(self.channel_couplings) - set(channels)
                or set(self.channel_sigmas_cm1) - set(channels)):
            raise ValueError("Unknown channel in width/coupling settings.")
        self.weight_sum = sum(weight for _, _, weight in channels.values())
        if (not np.isfinite(self.weight_sum) or self.weight_sum <= 0
                or any(not np.isfinite(w) or w < 0 for _, _, w in channels.values())
                or not np.isfinite(INSTRUMENT_SIGMA_CM1) or INSTRUMENT_SIGMA_CM1 < 0):
            raise ValueError("Invalid channel weights or instrument width.")
        self.prepared = {}
        self.electronic = {}  # Zero-phonon lines: (sticks, weights, sigma, channel weight)
        self.diagnostics = {}
        for name, (lower, upper, weight) in channels.items():
            if weight == 0:
                continue
            coupling = self.channel_couplings.get(name, 0.0)
            base_sigma = self.channel_sigmas_cm1.get(name, STARK_SIGMA_CM1)
            if not np.isfinite(base_sigma) or base_sigma <= 0:
                raise ValueError("Electronic Gaussian sigma must be positive.")
            sigma = float(np.hypot(base_sigma, INSTRUMENT_SIGMA_CM1))
            sticks, amplitudes, missing, orders = broadened_transition_sticks(
                lower, upper, temperature_k, coupling, tolerance)
            channel_weight = weight / self.weight_sum
            self.prepared[name] = (sticks, amplitudes, sigma, channel_weight)
            shifts, kernel, _, _ = phonon_kernel(coupling, temperature_k, tolerance)
            no_event_weight = no_phonon_event_weight(coupling, temperature_k)
            grid_error = phonon_grid_error_bound(sigma) if coupling > 0 else 0.0
            if grid_error > PHONON_GRID_MAX_RELATIVE_ERROR:
                raise ValueError(
                    f"{name}: phonon grid too coarse for sigma {sigma:g} cm^-1 (relative "
                    f"quadrature error {grid_error:.3g}); lower PHONON_GRID_STEP_CM1.")
            self.diagnostics[name] = {
                "total_huang_rhys_S": float(coupling),
                "electronic_sigma_cm1": float(base_sigma),
                "kernel_mass": float(kernel.sum()),
                "omitted_kernel_mass_upper_bound": missing,
                "max_events_per_band_each_sign": list(orders),
                "no_phonon_event_weight": no_event_weight,
                "grid_quadrature_relative_error_bound": grid_error,
                "integrated_stick_weight": float(amplitudes.sum()),
            }
            # Purely electronic transitions: the bare Stark lines, scaled by the
            # weight of "no phonon event at all" in the full model.
            gaps, line_weights = transition_sticks(lower, upper, temperature_k)
            e_sticks, e_weights = merge_sticks(gaps.ravel(), line_weights.ravel())
            self.electronic[name] = (e_sticks, no_event_weight * e_weights,
                                     sigma, channel_weight)
        if normalize_at_nm is not None:
            # Value at the reference wavelength: factor(normalize_at_nm) = 1.
            self.stark_norm = float(self.raw_stark(1e7 / float(normalize_at_nm)))
        else:
            # Global peak normalization includes the full retained sideband support.
            lo = max(1.0, min(s[0] - 10 * sig for s, _, sig, _ in self.prepared.values()))
            hi = max(s[-1] + 10 * sig for s, _, sig, _ in self.prepared.values())
            min_sigma = min(sig for _, _, sig, _ in self.prepared.values())
            energy = np.linspace(lo, hi, max(20001, int((hi - lo) / min_sigma * 20) + 1))
            self.stark_norm = float(self.raw_stark(energy).max())
        self.norm = 1.0
        if not np.isfinite(self.stark_norm) or self.stark_norm <= 0:
            raise ValueError("Invalid spectrum normalization (zero or non-finite absorption "
                             "at the normalization point; check the phonon coupling).")

    def raw_stark(self, energy):
        result = np.zeros_like(np.asarray(energy, dtype=float))
        for sticks, weights, sigma, channel_weight in self.prepared.values():
            result += channel_weight * gaussian_density(energy, sticks, weights, sigma)
        return result

    def raw_electronic(self, energy):
        """Sum of the zero-phonon lines over all channels (same unit as raw_stark)."""
        result = np.zeros_like(np.asarray(energy, dtype=float))
        for sticks, weights, sigma, channel_weight in self.electronic.values():
            result += channel_weight * gaussian_density(energy, sticks, weights, sigma)
        return result

    def mixed(self, wavelength_m):
        wavelength = np.asarray(wavelength_m, dtype=float)
        if np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0):
            raise ValueError("Wavelength must be finite and positive.")
        return self.raw_stark(1e-2 / wavelength) / self.stark_norm

    def factor(self, wavelength_m):
        result = self.mixed(wavelength_m)
        return float(result) if np.ndim(result) == 0 else result

    def electronic_only(self, wavelength_m):
        """Purely electronic contribution (no phonon events), with the same
        peak normalization as factor(); adding the sidebands gives factor()."""
        wavelength = np.asarray(wavelength_m, dtype=float)
        if np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0):
            raise ValueError("Wavelength must be finite and positive.")
        result = self.raw_electronic(1e-2 / wavelength) / self.stark_norm
        return float(result) if np.ndim(result) == 0 else result

    def channel_components(self, name, energy):
        """No-phonon-event component and remaining vibronic contributions.

        The remainder includes phonon processes with zero NET energy shift;
        it must not be equated to simply everything outside the central line.
        """
        sticks, weights, sigma, channel_weight = self.electronic[name]
        no_event = channel_weight * gaussian_density(energy, sticks, weights, sigma)
        sticks, weights, sigma, channel_weight = self.prepared[name]
        total = channel_weight * gaussian_density(energy, sticks, weights, sigma)
        return no_event, np.maximum(total - no_event, 0.0)


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
        # Line profiles only: no additive background or absorption floor.
        return stark

    def factor(self, wavelength_m):
        value = self.mixed(wavelength_m) / self.norm
        return float(value) if np.ndim(value) == 0 else value


def _reference_error_bound(spectrum):
    """Relative upper bound on the 1064-nm error caused by omitted kernel mass."""
    raw_reference = float(spectrum.raw_stark(1e7 / REFERENCE_WAVELENGTH_NM))
    if not np.isfinite(raw_reference) or raw_reference <= 0:
        return float("inf")
    error_bound = 0.0
    for name, (_, _, sigma, channel_weight) in spectrum.prepared.items():
        missing = spectrum.diagnostics[name]["omitted_kernel_mass_upper_bound"]
        error_bound += channel_weight * 2 * missing / (np.sqrt(2 * np.pi) * sigma)
    return error_bound / raw_reference


def _build_phonon_spectra():
    if (not np.isfinite(LORENTZ_HWHM_CM1) or LORENTZ_HWHM_CM1 != 0.0):
        raise ValueError("phonon_band requires LORENTZ_HWHM_CM1=0 (electronic Gaussian kernel).")
    if GSA_BACKGROUND_WEIGHT != 0.0 or ESA_BACKGROUND_WEIGHT != 0.0:
        raise ValueError("phonon_band uses no GSA/ESA background.")
    # All five electronic channel widths follow STARK_SIGMA_CM1. The legacy
    # broad_gsa H5 width (300 cm^-1) must not broaden the phonon spectrum again.
    # With the anchor, normalize directly at 1064 nm: same factors as the global
    # peak normalization followed by the anchor, without the slow peak search.
    normalize_at = REFERENCE_WAVELENGTH_NM if SI_REFERENCE_ANCHOR else None
    gsa = PhononSpectrum(GSA_CHANNELS, TEMPERATURE_K,
                        {"3H6 -> 3H5": GSA_H5_HUANG_RHYS,
                         "3H6 -> 3F4": GSA_F4_HUANG_RHYS},
                        tolerance=PHONON_TAIL_TOLERANCE, normalize_at_nm=normalize_at)
    esa = PhononSpectrum(ESA_CHANNELS, TEMPERATURE_K,
                        {"3F4 -> 3H4": ESA_H4_HUANG_RHYS,
                         "3F4 -> 3F3": ESA_F3_HUANG_RHYS,
                         "3F4 -> 3F2": ESA_F2_HUANG_RHYS},
                        tolerance=PHONON_TAIL_TOLERANCE, normalize_at_nm=normalize_at)
    if SI_REFERENCE_ANCHOR:
        g_ref = gsa.factor(REFERENCE_WAVELENGTH_NM * 1e-9)
        e_ref = esa.factor(REFERENCE_WAVELENGTH_NM * 1e-9)
        if min(g_ref, e_ref) <= 0 or not np.all(np.isfinite([g_ref, e_ref])):
            raise ValueError("Reference absorption is zero or non-finite; check the phonon coupling or anchor.")
        for label, spectrum in (("GSA", gsa), ("ESA", esa)):
            relative_bound = _reference_error_bound(spectrum)
            if relative_bound > REFERENCE_MAX_RELATIVE_KERNEL_ERROR:
                raise ValueError(
                    f"{label}: 1064-nm anchor insufficiently resolved (relative error bound "
                    f"{relative_bound:.3g}); lower PHONON_TAIL_TOLERANCE or --kernel-tolerance.")
            spectrum.reference_relative_kernel_error_bound = float(relative_bound)
    else:
        g_ref = e_ref = 1.0
    return gsa, esa, g_ref, e_ref


def anchor_judd_ofelt_ratios(spectrum, sigma_ref_cm2):
    """Per channel: integral of sigma implied by the 1064-nm anchor / Judd-Ofelt.

    The anchor fixes sigma(1064 nm) = sigma_ref, so a channel's integrated
    cross section is sigma_ref * (channel weight * stick weight) / raw(1064).
    1 means the anchored spectrum agrees with the JO electric-dipole integral.
    """
    raw_reference = float(spectrum.raw_stark(1e7 / REFERENCE_WAVELENGTH_NM))
    ratios = {}
    for name, (_, amplitudes, _, channel_weight) in spectrum.prepared.items():
        lower, upper, _ = spectrum.channels[name]
        implied = sigma_ref_cm2 * channel_weight * float(np.sum(amplitudes)) / raw_reference
        ratios[name] = implied / judd_ofelt_integrated_cross_section(name, lower, upper)
    return ratios


def calibrate_gsa_h5_huang_rhys(target_ratio=1.0, bracket=(0.01, 1.0)):
    """Huang-Rhys factor of 3H6 -> 3H5 for which anchor_judd_ofelt_ratios = target_ratio.

    Uses the current phonon bands, temperature, electronic width and kernel
    settings. A larger S puts more sideband at 1064 nm and lowers the ratio
    monotonically, so the root in the bracket is unique. Each evaluation
    rebuilds the 3H5 spectrum (a few seconds).
    """
    if not _HAVE_SCIPY:
        raise RuntimeError("The 3H6 -> 3H5 calibration requires scipy (scipy.optimize.brentq).")
    name = "3H6 -> 3H5"
    channel = {name: GSA_CHANNELS[name]}

    def log_ratio(coupling):
        spectrum = PhononSpectrum(channel, TEMPERATURE_K, {name: coupling},
                                  tolerance=PHONON_TAIL_TOLERANCE,
                                  normalize_at_nm=REFERENCE_WAVELENGTH_NM)
        ratio = anchor_judd_ofelt_ratios(spectrum, SI_SIGMA_GSA_1064_M2 * 1e4)[name]
        return np.log(ratio / target_ratio)

    return float(brentq(log_ratio, *bracket, xtol=1e-3))


def _build_spectra():
    """Build the GSA/ESA spectra and reference factors for the active model."""
    if ABSORPTION_MODEL == "phonon_band":
        return _build_phonon_spectra()
    _require_pa_model()
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
        g_ref = e_ref = 1.0   # no anchor: the factors are not used
    return gsa, esa, g_ref, e_ref


if _USE_STARK:
    GSA_SPECTRUM, ESA_SPECTRUM, _GSA_REFERENCE_FACTOR, _ESA_REFERENCE_FACTOR = _build_spectra()
else:   # "si_1064": no Stark spectra
    GSA_SPECTRUM = ESA_SPECTRUM = None
    _GSA_REFERENCE_FACTOR = _ESA_REFERENCE_FACTOR = 1.0


def _stark_resonance_factors(pump_wavelength_nm, p=None):
    """Relative cross sections; may exceed one away from the reference."""
    wavelength_m = np.asarray(pump_wavelength_nm, dtype=float) * 1e-9
    gsa = GSA_SPECTRUM.factor(wavelength_m)
    esa = ESA_SPECTRUM.factor(wavelength_m)
    if SI_REFERENCE_ANCHOR:
        gsa /= _GSA_REFERENCE_FACTOR
        esa /= _ESA_REFERENCE_FACTOR
    return gsa, esa


def _stark_rate_coefficients(pump_wavelength_nm, p):
    """Select GSA/ESA explicitly, without guessing from resonance centers."""
    wavelength_m = np.asarray(pump_wavelength_nm, dtype=float) * 1e-9
    _require_pa_model()
    f_gsa, f_esa = _stark_resonance_factors(pump_wavelength_nm, p)
    # f already includes Stark structure AND phonons. Apply it once.
    return (pa_model.cross_section_to_rate_coeff(p.sigma_gsa_m2, wavelength_m)*f_gsa,
            pa_model.cross_section_to_rate_coeff(p.sigma_esa_m2, wavelength_m)*f_esa)


def analytical_threshold_intensity(w_gsa, w_esa, w2, w3, s31, a32):
    """
    Full linear-stability threshold I_th, including GSA in the
    Jacobian (not only as an infinitesimal seed). Follows from
    det(J(I)) = 0 with R_gsa = w_gsa*I, R_esa = w_esa*I, which is
    quadratic in I:
        A*I^2 + B*I + C = 0
    A = w_esa*w_gsa
    B = w_gsa*(s31+w3) - w_esa*(a32+s31-w3)
    C = w2*(s31+w3)
    Takes the smallest positive root (the first time det(J) goes from
    positive to negative as I increases from 0).
    """
    if w_esa <= 0:
        return float("inf")

    C = w2 * (s31 + w3)

    if w_gsa <= 0:
        # Linear case (A=0): fall back on the corrected GSA-free
        # formula (note: a32 + s31 - w3, not a32 + 2*s31).
        denom = a32 + s31 - w3
        if denom <= 0:
            return float("inf")
        return C / (w_esa * denom)

    A = w_esa * w_gsa
    B = w_gsa * (s31 + w3) - w_esa * (a32 + s31 - w3)

    disc = B * B - 4.0 * A * C
    if disc < 0:
        return float("inf")  # no real crossing: no linear instability

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
    _require_pa_model()
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
    """Fit a straight segment around the maximum log slope.

    fraction=None: the widest segment that meets all FIT_* criteria.
    fraction=f: the segment where the secant slope >= f * peak; the FIT_*
    criteria are not enforced, only reported in fit["verified"].
    """
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
        verified = (len(xw) >= FIT_MIN_POINTS and k-lo >= 2 and hi-k >= 2
                    and r2 >= FIT_MIN_R2 and max_error <= FIT_MAX_ERROR_DEX
                    and slope/peak >= FIT_MIN_PEAK_RATIO and span >= FIT_MIN_SPAN_DEX)
        fit = {
            "slope": float(slope), "intercept": float(intercept),
            "x_range": (float(xw[0]), float(xw[-1])),
            "y_range": (float(yw[0]), float(yw[-1])),
            "s_peak": peak, "n_points": len(xw),
            "fraction": float(f), "r_squared": float(r2),
            "max_error_dex": float(max_error), "span_dex": span,
            "peak_ratio": float(slope/peak),
            "method": "automatic" if fraction is None else "fixed",
            "verified": bool(verified),
        }
        if fraction is None and not verified:
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
    Called for every Stark-based model (phonon_band, stark, hybrid, broad_gsa).
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
            require_close(sigma_gsa, SI_SIGMA_GSA_1064_M2, "SI sigma_GSA at 1064 nm")
            require_close(sigma_esa, SI_SIGMA_ESA_1064_M2, "SI sigma_ESA at 1064 nm")
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
    """Coarse intensity scan, then two zooms around the steepest point.

    The coarse grid only locates the steepest point; every curve is refined.
    The avalanche criterion (maximum local log-log slope >= MIN_AVALANCHE_SLOPE,
    and that maximum not at the first or last point, where the knee would be
    cut off by the intensity window) is applied to the complete refined curve,
    i.e. to all points of the coarse grid and of both zooms together. A coarse
    grid alone underestimates the slope of a knee (9.2 instead of 11.6 at
    1093 nm) and would miss weak avalanches.

    Ith* is the intensity of the maximum local log-log slope. Lee et al.
    (SI Fig. S3) define the threshold as the intersection of straight fits
    below and above the knee (pa_model.pa_threshold_broken_line); at 1064 nm
    that gives about 23.0 instead of 24.2 kW/cm^2.

    Returns (I, E, Ith*, s, avalanche_present, fit). With an avalanche, I and E
    are the refined curve and s is the fitted slope s_fit. Without one, I and E
    are the coarse curve (for plotting) and s is the maximum local slope of the
    refined curve.
    """

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

    # The coarse grid decides nothing about the avalanche; it only locates
    # the steepest point for the zooms. Stop here only for invalid emission.
    emission_ok, s_coarse = has_avalanche(
        I_coarse,
        E_coarse,
        min_slope=-np.inf,
        edge_margin=0
    )

    if not emission_ok:
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

    # Avalanche criterion on all points of the refined curve. edge_margin=1:
    # a slope maximum at the first or last point means the knee is cut off.
    present, s_local = has_avalanche(
        I_all,
        E_all,
        edge_margin=1
    )

    if not present:
        # Plot the coarse curve only; the zoom points would pile up markers.
        return I_coarse, E_coarse, np.nan, s_local, False, None

    fit = fit_avalanche_line(I_all, E_all)
    s_max = fit["slope"] if fit is not None else float("nan")

    return I_all, E_all, Ith_star, s_max, True, fit


# ---------------------------------------------------------------------------
# Plotting: one common style for every figure (report style, no titles)
# ---------------------------------------------------------------------------
# - Font: Computer Modern (cmr10 + mathtext "cm"), the design of the Latin
#   Modern font (lmodern) of the LaTeX report. Real Latin Modern is an
#   OpenType/CFF font that matplotlib embeds incorrectly in PDF files.
# - Size: FIG_WIDTH_IN = text width of the report (A4, 2.5-cm margins), font
#   11 pt = \small of the 12-pt report (captions). Include the PDFs with
#   \includegraphics[width=\linewidth]{...} so the text keeps this size.
# - Closed frame, ticks inward on all four sides with minor ticks, no grid.
# - No axis margins: every curve starts and ends in the corner of the axes.
#   Linear y-axes with nonnegative data run from exactly 0 (so 0-0 sits in
#   the corner) to the next major tick; log y-axes span whole decades.
# SHOW_WINDOWS=False: never open separate plot windows. With the inline
# backend (VS Code Interactive Window, Spyder Plots pane) the figures still
# appear inline; in a plain `python` run they are only saved to disk.
SHOW_WINDOWS = False
FIG_WIDTH_IN = 16.0 / 2.54          # report text width: 21 cm - 2 x 2.5 cm
FIG_HEIGHT_IN = 0.62 * FIG_WIDTH_IN
COLOR_SIM = "#1468b1"       # simulation / numerical / ESA
COLOR_MODEL = "#ba4a18"     # analytical / model curve / GSA
COLOR_EXTRA = ("#2e9e5b", "#8a3ffc")
COLOR_REF = "#7a7a7a"       # reference lines (1064 nm, intensity window)
PLOT_STYLE = {
    "font.family": "serif",
    "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,   # cmr10 has no minus sign
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.titlesize": 11,
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.grid": False,
    "axes.xmargin": 0.0,
    "axes.ymargin": 0.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 4.0,
    "ytick.major.size": 4.0,
    "xtick.minor.size": 2.0,
    "ytick.minor.size": 2.0,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.6,
    "ytick.minor.width": 0.6,
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "lines.linewidth": 1.3,
    "lines.markersize": 3.5,
    "figure.figsize": (FIG_WIDTH_IN, FIG_HEIGHT_IN),
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "svg.fonttype": "path",
}


def _new_figure(width=FIG_WIDTH_IN, height=FIG_HEIGHT_IN, ncols=1):
    """Figure in the common style; returns (fig, ax) or (fig, axes) for ncols > 1."""
    _require_matplotlib()
    plt.rcParams.update(PLOT_STYLE)
    # Embedding cmr10 in PDFs makes fontTools warn about the font's (harmless)
    # old timestamps; keep the console clean.
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), squeeze=False)
    return fig, (axes[0, 0] if ncols == 1 else axes[0])


def _wavelength_colors(n):
    """Sequential colours for curves ordered by pump wavelength."""
    return plt.cm.viridis(np.linspace(0.0, 0.88, max(int(n), 1)))


def _corner_axes(ax, xlim=None, ylim=None, y_max=None):
    """Axis limits that end exactly at the data (see the style notes above).

    xlim/ylim override the automatic limits. y_max caps the automatic upper
    limit (e.g. for thresholds that diverge where there is no avalanche).
    """
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
        return
    lo, hi = ax.get_ylim()  # data limits: the style has no margins
    if y_max is not None:
        hi = min(hi, y_max)
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return
    if ax.get_yscale() == "log":
        ax.set_ylim(10.0 ** np.floor(np.log10(lo) + 1e-9),
                    10.0 ** np.ceil(np.log10(hi) - 1e-9))
        return
    if lo >= 0:
        lo = 0.0
    ticks = np.asarray(ax.yaxis.get_major_locator().tick_values(lo, hi))
    above = ticks[ticks >= hi - 1e-9 * abs(hi)]
    ax.set_ylim(lo, float(above.min()) if above.size else hi)


def _save_figure(fig, savepath):
    if savepath:
        fig.savefig(savepath)
        print(f"Figure saved: {savepath}")


def _inline_backend():
    """True for backends that draw into the console (no separate windows)."""
    if plt is None:
        return False
    backend = plt.get_backend().lower()
    return any(key in backend for key in ("inline", "ipympl", "widget", "nbagg"))


def _show_or_close(no_show=False):
    """Show figures inline (Interactive Window/Spyder), never as separate windows
    unless SHOW_WINDOWS is True; otherwise only close them (they are saved)."""
    if plt is None:
        return
    if no_show or not (_inline_backend() or SHOW_WINDOWS):
        plt.close("all")
        return
    # Inline: draws every figure into the console and closes it.
    # SHOW_WINDOWS in a plain run: blocks until the windows are closed.
    plt.show()


@contextlib.contextmanager
def _no_plot_windows():
    """Stop a GUI backend in interactive mode from opening every new figure."""
    if plt is None or SHOW_WINDOWS or _inline_backend():
        yield
        return
    was_interactive = plt.isinteractive()
    plt.ioff()
    try:
        yield
    finally:
        if was_interactive:
            plt.ion()


def plot_wavelength_detuning(
        curves: Dict[float, Tuple[np.ndarray, np.ndarray]],
        thresholds: Dict[float, float],
        lineshape_label: str,
        lineshape_kind: str,
        fit_lines: Optional[Dict[float, Optional[Dict[str, float]]]] = None,
        savepath: Optional[str] = None) -> None:

    # With a fine wavelength grid, show one curve every DETUNING_PLOT_STEP_NM
    # (all wavelengths are still in the smax/ith plots and the summary CSV).
    step = DETUNING_PLOT_STEP_NM
    if step and step > 0:
        shown = [lam for lam in curves
                 if np.isclose(lam / step, round(lam / step), rtol=0, atol=1e-6)]
        if shown:
            curves = {lam: curves[lam] for lam in shown}
            if fit_lines:
                fit_lines = {lam: fit_lines.get(lam) for lam in shown}
    fig, ax = _new_figure(FIG_WIDTH_IN, 0.7 * FIG_WIDTH_IN)
    # Colour = pump wavelength (colour bar); dashed = no avalanche in the window.
    lams = np.array(list(curves), dtype=float)
    norm = matplotlib.colors.Normalize(lams.min(), lams.max() if lams.max() > lams.min()
                                       else lams.min() + 1.0)
    cmap = matplotlib.colors.ListedColormap(plt.cm.viridis(np.linspace(0.0, 0.88, 256)))
    I_lo = min(float(I.min()) for I, _E in curves.values())
    I_hi = max(float(I.max()) for I, _E in curves.values())

    for lam, (I_lam, emission) in curves.items():
        linestyle = "-" if np.isfinite(thresholds[lam]) else "--"
        ax.loglog(I_lam, _safe_for_log(emission), linestyle, linewidth=1.0,
                  color=cmap(norm(lam)))

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
            style = "-" if fit.get("verified") else "--"  # dashed: fit criteria not met
            ax.loglog([10.0 ** x0, 10.0 ** x1], [10.0 ** y0, 10.0 ** y1], style,
                      color="k", linewidth=1.1, zorder=5)

    ax.set_xlabel(r"Excitation intensity (kW cm$^{-2}$)")
    ax.set_ylabel("800-nm emission (a.u.)")
    _corner_axes(ax, xlim=(I_lo, I_hi))
    handles = [
        matplotlib.lines.Line2D([], [], color="0.35", linestyle="-", label="Avalanche"),
        matplotlib.lines.Line2D([], [], color="0.35", linestyle="--",
                                label="No avalanche in window"),
        matplotlib.lines.Line2D([], [], color="k", linewidth=1.1,
                                label=r"Fitted slope near $I_{\mathrm{th}}^{*}$"),
    ]
    ax.legend(handles=handles, loc="lower right")
    colorbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                            pad=0.02, fraction=0.05)
    colorbar.set_label("Pump wavelength (nm)")
    colorbar.ax.tick_params(direction="in", which="both")
    _save_figure(fig, savepath)


def plot_smax_vs_wavelength(
        fit_lines: Dict[float, Optional[Dict[str, float]]],
        savepath: Optional[str] = None) -> None:

    wavelengths = np.array(sorted(fit_lines.keys()), dtype=float)
    s_values = np.array([(float(fit_lines[lam]["slope"]) if fit_lines[lam] is not None else np.nan) for lam in wavelengths])

    fig, ax = _new_figure()
    ax.plot(wavelengths, s_values, "o-", color=COLOR_SIM)
    ax.set_xlabel("Pump wavelength (nm)")
    ax.set_ylabel(r"Fitted slope $s_{\mathrm{fit}}$")
    _corner_axes(ax)
    fig.tight_layout()
    _save_figure(fig, savepath)


def plot_ith_vs_wavelength(
        thresholds_num: Dict[float, float],
        thresholds_ana: Dict[float, float],
        savepath: Optional[str] = None,
        analytical_dense: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        intensity_window_max: Optional[float] = None) -> None:
    """Numerical knees (markers) and the analytical threshold.

    analytical_dense=(wavelengths_nm, ith): the analytical threshold on a fine
    grid, drawn as a continuous line; otherwise it is drawn at the scan points.
    intensity_window_max: upper end of the simulated intensity window; a
    numerical threshold can only be found below it.
    """
    wavelengths = np.array(sorted(thresholds_num.keys()), dtype=float)
    ith_num_values = np.array([float(thresholds_num[lam]) for lam in wavelengths])
    ith_ana_values = np.array([float(thresholds_ana[lam]) for lam in wavelengths])

    fig, ax = _new_figure()

    # Analytical threshold (full quadratic linear-stability threshold)
    if analytical_dense is not None:
        dense_wl, dense_ith = (np.asarray(v, dtype=float) for v in analytical_dense)
        ax.plot(dense_wl, np.where(np.isfinite(dense_ith), dense_ith, np.nan), "-",
                color=COLOR_MODEL, label="Analytical")
    else:
        ax.plot(wavelengths, np.where(np.isfinite(ith_ana_values), ith_ana_values, np.nan),
                "s--", color=COLOR_MODEL, label="Analytical")

    # Numerical knee (from the simulation and knee detection)
    ax.plot(wavelengths, ith_num_values, "o", color=COLOR_SIM,
            label=r"Numerical ($I_{\mathrm{th}}^{*}$)", zorder=5)

    ax.set_xlabel("Pump wavelength (nm)")
    ax.set_ylabel(r"$I_{\mathrm{th}}$ (kW cm$^{-2}$)")

    # Logarithmic scale to show the large variation in the analytical thresholds
    ax.set_yscale("log")
    y_max = None
    if intensity_window_max is not None:
        ax.axhline(intensity_window_max, color=COLOR_REF, linestyle=":", linewidth=1,
                   label="Upper end of intensity window")
        y_max = 10.0 * intensity_window_max
    _corner_axes(ax, y_max=y_max)
    ax.legend(loc="best")
    fig.tight_layout()
    _save_figure(fig, savepath)


def plot_ith_full_range(wavelengths_nm, ith, savepath=None, csvpath=None,
                        intensity_window_max=None, y_max=ITH_FULL_Y_MAX):
    """Analytical threshold over a wide pump range (only the analytical formula).

    Gaps: no linear instability (I_th infinite). y_max caps the y-axis, where
    I_th diverges far from the absorption bands.
    """
    wl = np.asarray(wavelengths_nm, dtype=float)
    ith = np.asarray(ith, dtype=float)
    if csvpath:
        np.savetxt(csvpath, np.column_stack([wl, ith]), delimiter=",", fmt="%.8g",
                   comments="", header="wavelength_nm,ith_analytical_kw_cm2")
        print(f"Data saved: {csvpath}")
    shown = np.where(np.isfinite(ith) & (ith > 0), ith, np.nan)
    fig, ax = _new_figure()
    ax.plot(wl, shown, "-", color=COLOR_MODEL, label="Analytical")
    ax.set_yscale("log")
    if intensity_window_max is not None:
        ax.axhline(intensity_window_max, color=COLOR_REF, linestyle=":", linewidth=1,
                   label="Upper end of intensity window")
    ax.set_xlabel("Pump wavelength (nm)")
    ax.set_ylabel(r"$I_{\mathrm{th}}$ (kW cm$^{-2}$)")
    _corner_axes(ax, xlim=(wl.min(), wl.max()), y_max=y_max)
    ax.legend(loc="upper center")
    fig.tight_layout()
    _save_figure(fig, savepath)
    print("  Gaps: no avalanche (no linear instability). Pump channels in the model: "
          "GSA 3H6 -> 3F4, 3H5; ESA 3F4 -> 3H4, 3F3, 3F2 (no GSA 3H6 -> 3H4 near 790 nm).")
    finite = np.isfinite(ith)
    if np.any(finite):
        k = int(np.nanargmin(np.where(finite, ith, np.nan)))
        print(f"  Lowest analytical threshold: {ith[k]:.3g} kW/cm^2 at {wl[k]:g} nm; "
              f"avalanche possible at {100 * finite.mean():.0f}% of the grid.")


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
    analytical_thresholds: Dict[float, float] = {}  # analytical threshold per wavelength
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

            # --- Analytical threshold: full quadratic linear-stability
            # threshold, including GSA in the Jacobian ---
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
                    print(f"  lambda_pump = {lam:6.1f} nm | Ith*(num) = {Ith_lam:7.2f} kW/cm^2 | Ith(ana) = {ith_ana:7.2f} kW/cm^2 | s_fit = {s_max:6.2f} (peak local slope {fit['s_peak']:6.2f}) | fit over {10.0 ** fit['x_range'][0]:.2f}-{10.0 ** fit['x_range'][1]:.2f} kW/cm^2 ({fit['n_points']} pts, fraction={fit['fraction']:.2f}, R2={fit['r_squared']:.5f}, max error={fit['max_error_dex']:.3f} dex, verified={'yes' if fit['verified'] else 'no'}) | E({I_lam[-1]:.0f} kW/cm^2) = {emission_lam[-1]:.3g}")

            else:
                emission_lam = power_dependence(I_values, p8, pump_wavelength_nm=lam)
                curves[lam] = (np.asarray(I_values), emission_lam)

                Ith_lam, s_max_converged = pa_threshold_cauchy(p8, I_values, lam, emission_coarse=emission_lam)

                thresholds[lam] = Ith_lam
                fit_lines[lam] = None

                print(f"  lambda_pump = {lam:6.1f} nm | Ith*(num) = {Ith_lam:7.2f} kW/cm^2 | Ith(ana) = {ith_ana:7.2f} kW/cm^2 | s_max = {s_max_converged:6.2f} | E({I_values[-1]:.0f} kW/cm^2) = {emission_lam[-1]:.3g}")

    return curves, thresholds, analytical_thresholds, fit_lines


def analytical_threshold_curve(p8, kind, step_nm=ANALYTICAL_STEP_NM, ranges=None):
    """Analytical threshold on a dense grid over `ranges` (default:
    analytical_wavelength_ranges()).

    Uses exactly the pump-rate coefficients of the kinetics. Separate ranges
    are joined with a NaN, so that a plot shows a gap between them.
    Returns (wavelengths_nm, ith_kw_cm2); inf means no linear instability.
    """
    if not np.isfinite(step_nm) or step_nm <= 0:
        raise ValueError("ANALYTICAL_STEP_NM must be positive.")
    a32 = p8.b32 * p8.w3_r + p8.w3_nr
    wl_parts, ith_parts = [], []
    with _lineshape(kind):
        for lo, hi in (analytical_wavelength_ranges() if ranges is None else ranges):
            grid = np.linspace(lo, hi, int(np.ceil((hi - lo) / step_nm - 1e-9)) + 1)
            if kind == "stark_gaussian":
                w_gsa, w_esa = _stark_rate_coefficients(grid, p8)  # vectorized
            else:
                rates = np.array([pa_model.gsa_esa_rate_coefficients(float(lam), p8)
                                  for lam in grid])
                w_gsa, w_esa = rates[:, 0], rates[:, 1]
            ith = np.array([analytical_threshold_intensity(float(g), float(e), p8.w2,
                                                           p8.w3, p8.s31, a32)
                            for g, e in zip(np.atleast_1d(w_gsa), np.atleast_1d(w_esa))])
            if wl_parts:
                wl_parts.append([np.nan])
                ith_parts.append([np.nan])
            wl_parts.append(grid)
            ith_parts.append(ith)
    return np.concatenate(wl_parts), np.concatenate(ith_parts)


def save_detuning_summary(save_dir, sfx, p8, thresholds, analytical_thresholds,
                          fit_lines, analytical_dense=None):
    """CSV per scanned wavelength (for comparison with measurements).

    Columns: numerical knee Ith* (NaN: no PA in the intensity window), analytical
    threshold, fitted slope s_fit and peak local slope (NaN without a fit), and
    the absorption factors f_gsa, f_esa used in the pump rates.
    The dense analytical curve is written to a second CSV.
    """
    wavelengths = np.array(sorted(thresholds), dtype=float)
    f_gsa, f_esa = absorption_factors(wavelengths, p8)

    def fit_value(lam, key):
        fit = fit_lines.get(lam)
        return float(fit[key]) if fit is not None else np.nan

    data = np.column_stack((
        wavelengths,
        [thresholds[lam] for lam in wavelengths],
        [analytical_thresholds[lam] for lam in wavelengths],
        [fit_value(lam, "slope") for lam in wavelengths],
        [fit_value(lam, "s_peak") for lam in wavelengths],
        f_gsa, f_esa))
    path = os.path.join(save_dir, f"pa_detuning_summary{sfx}.csv")
    np.savetxt(path, data, delimiter=",", comments="", fmt="%.10g",
               header="wavelength_nm,ith_numerical_kw_cm2,ith_analytical_kw_cm2,"
                      "s_fit,s_peak_local,f_gsa,f_esa")
    print(f"Detuning summary saved: {path}")
    if analytical_dense is not None:
        dense_wl, dense_ith = analytical_dense
        keep = np.isfinite(dense_wl)
        dense_path = os.path.join(save_dir, f"pa_ith_analytical_dense{sfx}.csv")
        np.savetxt(dense_path, np.column_stack((dense_wl[keep], dense_ith[keep])),
                   delimiter=",", comments="", fmt="%.10g",
                   header="wavelength_nm,ith_analytical_kw_cm2")
        print(f"Dense analytical threshold saved: {dense_path}")

# ---------------------------------------------------------------------------
# Rise time (95%): peak search on an adaptive intensity grid
# ---------------------------------------------------------------------------

RISE_FRACTION = 0.95
RISE_RTOL = 1e-8
# Wavelengths in the 1000-1100 nm measurement range; the 1450-nm entries are
# added when INCLUDE_1450_BAND is on. At 1064 nm the sigma sweep is flat by
# construction (anchor); 1075 nm (lowest threshold) and 1090 nm (edge of the
# 3F4 -> 3F2 ESA band) show how strongly the line width matters.
RISE_WAVELENGTHS_NM = [1060.0, 1064.0, 1070.0, 1075.0, 1080.0, 1090.0]
RISE_WAVELENGTHS_1450_NM = [1410.0, 1430.0, 1450.0, 1470.0]
RISE_SIGMA_VALUES_CM1 = [3.0, 5.0, 8.5, 12.0, 16.0, 24.0, 50, 100 ,150,200]
RISE_SIGMA_WAVELENGTHS_NM = [1064.0, 1075.0, 1090.0]
RISE_SIGMA_WAVELENGTHS_1450_NM = [1450.0]
# Rise time vs pump wavelength ("risewl"): pump grid in nm. Wavelengths
# without an avalanche inside the intensity window are skipped.
RISE_WL_MIN_NM, RISE_WL_MAX_NM, RISE_WL_STEP_NM = 1050.0, 1100.0, 0.5
RISE_WL_REFERENCE_NM = 1064.0  # fixes the constant of the c/sqrt(R_GSA) curve

# Peak finder. The rise time peaks sharply just above the threshold (critical
# slowing down), so a coarse grid alone misses the maximum by tens of percent.
# Steps: coarse log grid + RISE_N_NEAR_ITH points within +/- RISE_NEAR_ITH_DEX
# of the analytical threshold + n_rounds grid refinements around the maximum,
# then a bounded Brent search for the maximum in log10(I) to RISE_PEAK_XTOL_DEX.
RISE_N_COARSE, RISE_N_REFINE, RISE_N_ROUNDS = 24, 13, 2          # main plot
RISE_N_COARSE_SWEEP, RISE_N_REFINE_SWEEP = 14, 9                 # sigma sweep
RISE_N_NEAR_ITH = 15
RISE_NEAR_ITH_DEX = 0.02      # +/- 4.7 % around the analytical threshold
RISE_PEAK_XTOL_DEX = 1e-5     # final peak position: relative 2e-5 in intensity

RISE_YSCALE = "log"   # "linear" for the same style as pa_fig2b_rise_time_v2.png
RISE_XSCALE = "linear"

# With SI_REFERENCE_ANCHOR ("phonon_band", "hybrid" and "broad_gsa") the absorption
# at 1064 nm is fixed, independent of sigma. Set to False to switch the anchor
# off during the sigma sweep.
RISE_SIGMA_KEEP_ANCHOR = True


def rise_time_95_fast(I_pump, p, w_gsa, w_esa, fraction=RISE_FRACTION):
    """Time (s) until N2 (= 800-nm emission) reaches `fraction` of its steady state.
    One integration with dense_output; the 95% crossing is found afterwards with brentq."""
    if not _HAVE_SCIPY:
        raise RuntimeError("Rise time requires scipy.")
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
    """Rise time (ms) vs intensity; finds the peak adaptively (active line shape).

    Coarse log grid, extra points around the analytical threshold, n_rounds
    grid refinements around the maximum and a final bounded Brent search in
    log10(I) (see RISE_* settings). All evaluated points are returned.
    Returns (I, rise_ms, peak_index); peak_index is None without valid points.
    """
    from scipy.optimize import minimize_scalar

    w_gsa, w_esa = pa_model.gsa_esa_rate_coefficients(wavelength_nm, p8)
    I_all = np.logspace(np.log10(I_min), np.log10(I_max), n_coarse)
    a32 = p8.b32 * p8.w3_r + p8.w3_nr
    ith = analytical_threshold_intensity(w_gsa, w_esa, p8.w2, p8.w3, p8.s31, a32)
    if np.isfinite(ith) and I_min < ith < I_max and RISE_N_NEAR_ITH > 0:
        near = ith * np.logspace(-RISE_NEAR_ITH_DEX, RISE_NEAR_ITH_DEX, RISE_N_NEAR_ITH)
        I_all = np.unique(np.concatenate([I_all, np.clip(near, I_min, I_max)]))
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
    k = int(np.nanargmax(rt_all))
    if 0 < k < len(I_all) - 1:
        # Polish: bounded Brent search between the neighbours of the grid maximum.
        evaluated = {}

        def negative_rise(log_intensity):
            value = float(_rise_ms(p8, w_gsa, w_esa, [10.0 ** log_intensity])[0])
            evaluated[log_intensity] = value
            return -value if np.isfinite(value) else 0.0

        minimize_scalar(negative_rise, method="bounded",
                        bounds=(np.log10(I_all[k - 1]), np.log10(I_all[k + 1])),
                        options={"xatol": RISE_PEAK_XTOL_DEX})
        if evaluated:
            I_all = np.concatenate([I_all, 10.0 ** np.array(list(evaluated))])
            rt_all = np.concatenate([rt_all, list(evaluated.values())])
            order = np.argsort(I_all)
            I_all, rt_all = I_all[order], rt_all[order]
    return I_all, rt_all, int(np.nanargmax(rt_all))


@contextlib.contextmanager
def _stark_sigma(sigma_cm1, keep_anchor=True):
    """Temporarily change STARK_SIGMA_CM1 and rebuild spectra + reference factors.
    In phonon_band all electronic channels follow the new sigma.
    The phonon band and Huang-Rhys factors stay the same.
    For all Stark modes. In broad_gsa the H5 GSA sigma stays separately at
    GSA_H5_SIGMA_CM1; only the other Stark channels change."""
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
                print(f"  Rise time {lam:.0f} nm: no valid points.")
                continue
            edge = " (NOTE: peak at the edge of the window)" if k in (0, len(I_grid) - 1) else ""
            print(f"  Rise time {lam:.0f} nm: peak {rt[k]:.2f} ms at {I_grid[k]:.2f} kW/cm^2{edge}")
            results[lam] = (I_grid, rt, k)
    return results


def compute_rise_peak_vs_sigma(p8, wavelengths_nm, sigmas, I_min, I_max):
    """Peak rise time vs electronic sigma for each wavelength.

    The spectra are rebuilt once per sigma and reused for all wavelengths.
    Returns {wavelength: array of peak rise times (ms), one per sigma}.
    """
    peaks = {lam: np.full(len(sigmas), np.nan) for lam in wavelengths_nm}
    for i, s in enumerate(sigmas):
        with _stark_sigma(s, keep_anchor=RISE_SIGMA_KEEP_ANCHOR), _lineshape("stark_gaussian"):
            for lam in wavelengths_nm:
                I_grid, rt, k = rise_peak_scan(p8, lam, I_min, I_max,
                                               n_coarse=RISE_N_COARSE_SWEEP,
                                               n_refine=RISE_N_REFINE_SWEEP)
                if k is None:
                    print(f"  {lam:.0f} nm, sigma={s:g}: no valid points")
                    continue
                peaks[lam][i] = float(rt[k])
                edge = (" (NOTE: peak at the edge of the window)"
                        if k in (0, len(I_grid) - 1) else "")
                print(f"  {lam:.0f} nm, sigma={s:g} cm^-1: "
                      f"peak {rt[k]:.2f} ms at {I_grid[k]:.3f} kW/cm^2{edge}")
    return peaks


def plot_rise_vs_intensity(results, savepath=None):
    fig, ax = _new_figure()
    colors = _wavelength_colors(len(results))
    for (lam, (I_grid, rt, k)), color in zip(results.items(), colors):
        ax.plot(I_grid, rt, "-", color=color, label=f"{lam:.0f} nm")
        ax.plot(I_grid[k], rt[k], "o", markersize=4, color=color, zorder=6)
    ax.set_xscale(RISE_XSCALE)
    ax.set_yscale(RISE_YSCALE)
    ax.set_xlabel(r"Excitation intensity (kW cm$^{-2}$)")
    ax.set_ylabel("800-nm rise time (ms)")
    _corner_axes(ax)
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save_figure(fig, savepath)


def plot_rise_peak_vs_sigma(sigmas, peaks_by_wavelength, savepath=None):
    n = len(peaks_by_wavelength)
    fig, axes = _new_figure(FIG_WIDTH_IN, 0.4 * FIG_WIDTH_IN, ncols=n)
    axes = np.atleast_1d(axes)
    colors = _wavelength_colors(n)
    for i, (ax, (lam, peaks), color) in enumerate(zip(axes, peaks_by_wavelength.items(),
                                                    colors)):
        ax.plot(sigmas, peaks, "o-", color=color)
        if i == 0:
            ax.set_ylabel("Longest rise time (ms)")
        if i == len(axes) // 2:
            ax.set_xlabel(r"Electronic Gaussian width $\sigma$ (cm$^{-1}$)")
        ax.text(0.95, 0.06, f"{lam:.0f} nm", transform=ax.transAxes, ha="right",
                va="bottom")
        _corner_axes(ax)
    fig.tight_layout()
    _save_figure(fig, savepath)


def compute_rise_vs_wavelength(p8, wavelengths_nm, I_min, I_max):
    """Longest rise time (peak of rise time vs intensity) per pump wavelength.

    Same peak search as the "rise" plot (rise_peak_scan). Wavelengths whose
    analytical threshold lies outside [I_min, I_max], or whose peak sits at
    the edge of the window, are skipped. Returns a dict of arrays:
    wavelength_nm, peak_rise_time_ms, intensity_at_peak_kw_cm2,
    ith_analytical_kw_cm2, r_gsa_at_threshold_per_s, f_gsa, f_esa.
    """
    a32 = p8.b32 * p8.w3_r + p8.w3_nr
    rows = []
    with _lineshape(ABSORPTION_KIND):
        for lam in wavelengths_nm:
            w_gsa, w_esa = pa_model.gsa_esa_rate_coefficients(lam, p8)
            ith = analytical_threshold_intensity(w_gsa, w_esa, p8.w2, p8.w3, p8.s31, a32)
            if not np.isfinite(ith) or not I_min < ith < I_max:
                continue  # no avalanche inside the intensity window
            I_grid, rt, k = rise_peak_scan(p8, lam, I_min, I_max)
            if k is None or k in (0, len(I_grid) - 1):
                continue  # peak cut off by the window
            rows.append((lam, rt[k], I_grid[k], ith, w_gsa * ith))
    keys = ("wavelength_nm", "peak_rise_time_ms", "intensity_at_peak_kw_cm2",
            "ith_analytical_kw_cm2", "r_gsa_at_threshold_per_s")
    data = {key: np.array([row[i] for row in rows], dtype=float)
            for i, key in enumerate(keys)}
    if rows:
        f_gsa, f_esa = absorption_factors(data["wavelength_nm"], p8)
        data["f_gsa"], data["f_esa"] = np.asarray(f_gsa, float), np.asarray(f_esa, float)
    return data


def plot_rise_vs_wavelength(data, savepath=None, csvpath=None,
                            reference_nm=RISE_WL_REFERENCE_NM):
    """Longest rise time vs pump wavelength, with c/sqrt(R_GSA at threshold).

    The stronger the ground-state absorption that seeds the avalanche, the
    less the system slows down at the threshold; c is fixed at reference_nm.
    """
    lam = data["wavelength_nm"]
    if not len(lam):
        print("  No wavelength with an avalanche inside the window: plot skipped.")
        return
    peak = data["peak_rise_time_ms"]
    r_gsa = data["r_gsa_at_threshold_per_s"]
    ref = int(np.argmin(np.abs(lam - reference_nm)))
    scaling = peak[ref] * np.sqrt(r_gsa[ref] / r_gsa)

    if csvpath:
        columns = list(data) + ["scaling_prediction_ms"]
        np.savetxt(csvpath, np.column_stack([data[c] for c in data] + [scaling]),
                   delimiter=",", fmt="%.6g", comments="", header=",".join(columns))
        print(f"Data saved: {csvpath}")

    fig, ax = _new_figure()
    ax.plot(lam, scaling, "-", color=COLOR_MODEL,
            label=r"$\propto 1/\sqrt{R_\mathrm{GSA}}$ at threshold "
                  f"(fixed at {lam[ref]:g} nm)")
    ax.plot(lam, peak, "o", color=COLOR_SIM, label="Simulation")
    ax.set_xlabel("Pump wavelength (nm)")
    ax.set_ylabel("Longest 800-nm rise time (ms)")
    _corner_axes(ax)
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save_figure(fig, savepath)

    deviation = np.abs(peak / scaling - 1.0)
    print(f"  {len(lam)} wavelengths with an avalanche: {lam.min():g}-{lam.max():g} nm; "
          f"longest {peak.max():.0f} ms at {lam[peak.argmax()]:g} nm")
    print(f"  Deviation from c/sqrt(R_GSA): median {100 * np.median(deviation):.1f}%, "
          f"max {100 * deviation.max():.1f}%")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def absorption_factors(wavelengths_nm, p=None):
    """The SAME dimensionless cross-section factors used by the pump rates.

    phonon_band, hybrid, broad_gsa and si_1064: sigma(lambda)/sigma(1064 nm),
    f(1064)=1 (si_1064 via pa_model.anchored_resonance_factors).
    stark: preserves its global peak normalization.
    No extra wavelength/photon-flux multiplier is included here.
    """
    wavelengths = np.asarray(wavelengths_nm, dtype=float)
    if wavelengths.ndim != 1 or not len(wavelengths):
        raise ValueError("wavelengths_nm must be a nonempty 1D array.")
    if np.any(~np.isfinite(wavelengths)) or np.any(wavelengths <= 0):
        raise ValueError("Wavelengths must be finite and positive.")
    if _USE_STARK:
        return tuple(np.asarray(v) for v in _stark_resonance_factors(wavelengths, p))
    _require_pa_model()
    values = np.array([pa_model.anchored_resonance_factors(float(lam), p)
                       for lam in wavelengths])
    return values[:, 0], values[:, 1]


def electronic_absorption_factors(wavelengths_nm):
    """No-phonon-event contribution with the full spectrum's normalization."""
    wavelengths = np.asarray(wavelengths_nm, dtype=float)
    if (wavelengths.ndim != 1 or not wavelengths.size
            or np.any(~np.isfinite(wavelengths)) or np.any(wavelengths <= 0)):
        raise ValueError("Provide a nonempty 1D array of positive finite wavelengths.")
    if ABSORPTION_MODEL != "phonon_band":
        raise ValueError("Electronic components are provided for phonon_band only.")
    gsa = np.asarray(GSA_SPECTRUM.electronic_only(wavelengths * 1e-9))
    esa = np.asarray(ESA_SPECTRUM.electronic_only(wavelengths * 1e-9))
    if SI_REFERENCE_ANCHOR:
        gsa = gsa / _GSA_REFERENCE_FACTOR
        esa = esa / _ESA_REFERENCE_FACTOR
    return gsa, esa


def _save_factor_data(basename, wavelengths, f_gsa, f_esa):
    data = [wavelengths, f_gsa, f_esa]
    header = "wavelength_nm,f_gsa,f_esa"
    if ABSORPTION_MODEL == "phonon_band":
        data.extend(electronic_absorption_factors(wavelengths))
        header += ",f_gsa_zero_phonon,f_esa_zero_phonon"
    np.savetxt(str(basename) + ".csv", np.column_stack(data), delimiter=",",
               comments="", header=header, fmt="%.12g")


def export_absorption_data(p, save_dir, wavelength_min_nm=900.0,
                           wavelength_max_nm=1500.0, step_nm=0.05):
    if (not np.all(np.isfinite([wavelength_min_nm, wavelength_max_nm, step_nm]))
            or wavelength_min_nm <= 0 or wavelength_max_nm <= wavelength_min_nm
            or step_nm <= 0):
        raise ValueError("Invalid wavelength range or step.")
    count = int(np.ceil((wavelength_max_nm-wavelength_min_nm)/step_nm)) + 1
    wavelengths = np.linspace(wavelength_min_nm, wavelength_max_nm, count)
    f_gsa, f_esa = absorption_factors(wavelengths, p)
    if np.any(~np.isfinite(f_gsa)) or np.any(~np.isfinite(f_esa)):
        raise RuntimeError("Non-finite absorption factors.")
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    basename = Path(save_dir) / f"f_gsa_f_esa_{wavelength_min_nm:g}_{wavelength_max_nm:g}nm_{ABSORPTION_MODEL}"
    _save_factor_data(basename, wavelengths, f_gsa, f_esa)
    print(f"Factor data saved: {basename}.csv")
    return wavelengths, f_gsa, f_esa


def export_phonon_settings(save_dir):
    if ABSORPTION_MODEL != "phonon_band":
        return
    folder = Path(save_dir)
    folder.mkdir(parents=True, exist_ok=True)
    bands = np.asarray(PHONON_BANDS, dtype=float).reshape(-1, 3)
    centers, sigmas, fractions = bands[:, 0], bands[:, 1], bands[:, 2]
    np.savetxt(folder / "phonon_band.csv", np.column_stack((
        centers, sigmas, fractions, GSA_H5_HUANG_RHYS * fractions,
        GSA_F4_HUANG_RHYS * fractions, ESA_H4_HUANG_RHYS * fractions,
        ESA_F3_HUANG_RHYS * fractions, ESA_F2_HUANG_RHYS * fractions)),
        delimiter=",", comments="", fmt="%.12g",
        header="band_center_cm1,band_sigma_cm1,fraction,S_k_GSA_3H5,S_k_GSA_3F4,"
               "S_k_ESA_3H4,S_k_ESA_3F3,S_k_ESA_3F2")
    metadata = {
        "absorption_model": ABSORPTION_MODEL,
        "temperature_K": TEMPERATURE_K,
        "phonon": PHONON_INFO,
        "channel_weights_judd_ofelt": channel_weight_report(),
        "judd_ofelt_enabled": USE_JUDD_OFELT_WEIGHTS,
        "judd_ofelt_omega_1e-20_cm2": JO_OMEGA_1E20_CM2.tolist(),
        "judd_ofelt_use_frequency_factor": JO_USE_FREQUENCY_FACTOR,
        "judd_ofelt_sources": ["Lee et al. 2021, SI Tables S4 and S5",
                               "Carnall, Crosswhite & Crosswhite 1978, ANL-78-XX-95, Appendix XIII Table 2"],
        "judd_ofelt_consistency": {
            "GSA": anchor_judd_ofelt_ratios(GSA_SPECTRUM, SI_SIGMA_GSA_1064_M2 * 1e4),
            "ESA": anchor_judd_ofelt_ratios(ESA_SPECTRUM, SI_SIGMA_ESA_1064_M2 * 1e4),
            "refractive_index": JO_REFRACTIVE_INDEX,
            "definition": ("integral of sigma implied by the 1064-nm SI anchor / Judd-Ofelt "
                           "electric-dipole integral; 1 = consistent"),
        },
        "reference_wavelength_nm": REFERENCE_WAVELENGTH_NM,
        "reference_anchor_enabled": SI_REFERENCE_ANCHOR,
        "reference_factors": [float(x) for x in
                              _stark_resonance_factors(REFERENCE_WAVELENGTH_NM)],
        "reference_relative_kernel_error_upper_bound": {
            "GSA": getattr(GSA_SPECTRUM, "reference_relative_kernel_error_bound", None),
            "ESA": getattr(ESA_SPECTRUM, "reference_relative_kernel_error_bound", None)},
        "GSA": GSA_SPECTRUM.diagnostics,
        "ESA": ESA_SPECTRUM.diagnostics,
        "factor_definition": "sigma(lambda)/sigma(1064 nm); applied once in pump rates",
        "electronic_components": "No phonon event; same full-spectrum normalization and anchor",
        "plot_y_scales": ["linear", "log"],
        "assumptions": "Band shape (four Gaussians approximating a Raman curve, each used as one band with the Bose occupation of its centre), the Huang-Rhys factors except 3H6 -> 3H5 (calibrated so that the 1064-nm anchor matches the Judd-Ofelt integral), electronic widths and relative Stark line strengths are uncalibrated assumptions; channel weights follow Judd-Ofelt.",
        "sources_from_supplied_script": ["https://doi.org/10.7150/thno.5113",
                                        "https://doi.org/10.1016/j.omx.2021.100102",
                                        "https://doi.org/10.1080/0026897021000046889"],
    }
    (folder / "phonon_settings.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def plot_absorption_factors(p, save_dir, wavelength_min_nm=900.0,
                            wavelength_max_nm=1500.0, step_nm=0.05):
    """Plot/export f_GSA and f_ESA without running population simulations.

    Always saves the spectrum twice, with a linear and with a logarithmic
    y-axis (<basename>_linear.png/.svg and <basename>_log.png/.svg). Both
    figures show the same factors, which are written once to <basename>.csv.
    Returns ([linear_fig, log_fig], wavelengths, f_gsa, f_esa).
    """
    _require_matplotlib()
    if wavelength_max_nm <= wavelength_min_nm or step_nm <= 0:
        raise ValueError("Invalid wavelength range or step.")
    n = int(np.ceil((wavelength_max_nm-wavelength_min_nm)/step_nm)) + 1
    wavelengths = np.linspace(wavelength_min_nm, wavelength_max_nm, n)
    f_gsa, f_esa = absorption_factors(wavelengths, p)
    if np.any(~np.isfinite(f_gsa)) or np.any(~np.isfinite(f_esa)):
        raise RuntimeError("Non-finite absorption factors.")
    # Evaluate every curve once; both figures reuse the same data.
    curves = [(f_gsa, COLOR_MODEL, "-", 1.3, r"$f_{\mathrm{GSA}}$"),
              (f_esa, COLOR_SIM, "-", 1.3, r"$f_{\mathrm{ESA}}$")]
    if ABSORPTION_MODEL == "phonon_band":
        g_el, e_el = electronic_absorption_factors(wavelengths)
        curves += [(g_el, COLOR_EXTRA[0], "--", 1.0, "GSA, zero-phonon only"),
                   (e_el, COLOR_EXTRA[1], "--", 1.0, "ESA, zero-phonon only")]
    span = wavelength_max_nm - wavelength_min_nm
    positive = np.concatenate((f_gsa[f_gsa > 0], f_esa[f_esa > 0]))

    os.makedirs(save_dir, exist_ok=True)
    basename = os.path.join(save_dir,
                           f"f_gsa_f_esa_{wavelength_min_nm:g}_{wavelength_max_nm:g}nm_{ABSORPTION_MODEL}")
    figures = []
    for scale_name, log_y in (("linear", False), ("log", True)):
        fig, ax = _new_figure()

        for values, color, style, width, label in curves:
            # Only the log axis needs zeros hidden; the linear axis shows them.
            shown = np.where(values > 0, values, np.nan) if log_y else values
            ax.plot(wavelengths, shown, color=color, linestyle=style,
                    linewidth=width, label=label)
        ax.set_yscale("log" if log_y else "linear")
        tick_step = 100 if span >= 300 else 10
        ax.set_xticks(np.arange(wavelength_min_nm, wavelength_max_nm+1, tick_step))
        ax.set_xlabel("Pump wavelength (nm)")
        ax.set_ylabel(r"Relative absorption factor $f$")
        if SI_REFERENCE_ANCHOR:
            ax.axhline(1.0, color=COLOR_REF, linestyle=":", linewidth=0.8)
            ax.plot([REFERENCE_WAVELENGTH_NM], [1.0], "o", color="k", ms=3.5, zorder=5)
            ax.annotate(r"1064 nm: $f = 1$", xy=(REFERENCE_WAVELENGTH_NM, 1),
                        xytext=(0.03, 0.80), textcoords="axes fraction", fontsize=9,
                        arrowprops={"arrowstyle": "-", "color": COLOR_REF,
                                    "linewidth": 0.7})
        if log_y:
            # Gaussian tails can underflow to zero. Only the visible range is
            # limited; the computed/exported factors are never clipped.
            lower = max(ABSORPTION_PLOT_MIN_FACTOR,
                        10.0**np.floor(np.log10(positive.min())))
            upper = max(lower*10.0, 10.0**np.ceil(np.log10(positive.max())))
            _corner_axes(ax, xlim=(wavelength_min_nm, wavelength_max_nm),
                         ylim=(lower, upper))
        else:
            # From exactly 0 to the next tick above the highest peak: after the
            # 1064-nm normalization the peaks can lie far above 1.
            _corner_axes(ax, xlim=(wavelength_min_nm, wavelength_max_nm))
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
        if _USE_STARK and wavelength_min_nm < 1170 and wavelength_max_nm > 1290:
            peak = int(np.argmax(f_gsa))
            ax.annotate(r"GSA $^3H_6\rightarrow{}^3H_5$" + f"\npeak: {wavelengths[peak]:.1f} nm",
                        xy=(wavelengths[peak], f_gsa[peak]), xytext=(1320, f_gsa[peak]*0.65),
                        fontsize=9,
                        arrowprops={"arrowstyle": "-", "color": COLOR_REF, "linewidth": 0.7})
        fig.tight_layout()
        for extension in ("pdf", "png", "svg"):
            fig.savefig(f"{basename}_{scale_name}.{extension}")
        figures.append(fig)
    _save_factor_data(basename, wavelengths, f_gsa, f_esa)
    print("GSA and ESA backgrounds: removed.")
    print(f"Factor plots saved: {basename}_linear.pdf/.png/.svg and "
          f"{basename}_log.pdf/.png/.svg; data: {basename}.csv")
    return figures, wavelengths, f_gsa, f_esa


def main(spectra_only=False, save_dir=None, data_only=False, no_show=False, plots=None):
    """plots: dict like PLOTS; None uses the setting at the top of the script."""
    plots = dict(PLOTS if plots is None else plots)
    unknown = set(plots) - set(PLOTS)
    if unknown:
        raise ValueError(f"Unknown plot names: {sorted(unknown)}; choose from {sorted(PLOTS)}.")
    need_detuning = plots.get("detuning") or plots.get("smax") or plots.get("ith")
    need_kinetics = (need_detuning or plots.get("ith_full") or plots.get("rise")
                     or plots.get("risewl") or plots.get("sigma"))
    import time
    t_start = time.perf_counter()

    def _done(label, t0):
        print(f"  [{label}: {time.perf_counter() - t0:.1f} s]")

    # Only I_values[0] and I_values[-1] matter for the adaptive
    # Gaussian/Stark line-shape scans: the points inside the window are chosen adaptively.
    I_values = np.logspace(
        0,
        np.log10(150.0),
        100
    )

    if not (spectra_only or data_only):
        if not need_kinetics:
            spectra_only = True  # Only spectra requested: skip the kinetics.
    if not (spectra_only or data_only) or ABSORPTION_MODEL != "phonon_band":
        _require_pa_model()
    if not data_only:
        _require_matplotlib()
    p8 = (parameters_for_sample(3, lambda0_gsa_nm=GSA_BACKGROUND_LAMBDA0_NM)
          if pa_model is not None else None)
    if save_dir is None:
        try:
            save_dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:  # e.g. pasted into an interactive window
            save_dir = os.getcwd()
    export_phonon_settings(save_dir)
    t0 = time.perf_counter()
    if data_only or not plots.get("spectra"):
        export_absorption_data(p8, save_dir)
        export_absorption_data(p8, save_dir, 1040.0, 1100.0)
    else:
        plot_absorption_factors(p8, save_dir)
        plot_absorption_factors(p8, save_dir, 1040.0, 1100.0)
    _done("spectra", t0)
    if spectra_only or data_only:
        if _USE_STARK:
            check_thermal_population_coupling(p8)
        _show_or_close(no_show or data_only)
        return

    print(f"ABSORPTION_MODEL = {ABSORPTION_MODEL!r} ({ABSORPTION_LABEL})")
    print("Parts: " + ", ".join(f"{k}={'on' if v else 'off'}" for k, v in plots.items()))
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
            print("No SI anchor and no broad background: pure Stark absorption.")
    else:
        print("Absorption model: Gaussian line shape normalized to the SI cross sections "
              f"at 1064 nm (sigma_GSA = {p8.sigma_gsa_m2:.3e} m^2, "
              f"sigma_ESA = {p8.sigma_esa_m2:.3e} m^2); no Stark structure.")

    wavelengths_nm = detuning_wavelengths()
    print(f"Detuning scan: {len(wavelengths_nm)} pump wavelengths, "
          f"{SCAN_MIN_NM:g}-{SCAN_MAX_NM:g} nm in {SCAN_STEP_NM:g}-nm steps"
          + (" + 1450-nm band" if INCLUDE_1450_BAND else "")
          + f"; analytical threshold every {ANALYTICAL_STEP_NM:g} nm.")
    if INCLUDE_EXTRAPOLATED_WAVELENGTHS:
        print("WARNING: extended scan is uncalibrated. Very large long-wave "
              "slopes can reflect an underestimated phonon-assisted GSA.")
    rise_wavelengths = RISE_WAVELENGTHS_NM + (RISE_WAVELENGTHS_1450_NM if INCLUDE_1450_BAND else [])
    sigma_wavelengths = (RISE_SIGMA_WAVELENGTHS_NM
                         + (RISE_SIGMA_WAVELENGTHS_1450_NM if INCLUDE_1450_BAND else []))

    # ---------------------------------------------------------------
    # OUTPUT FOLDER
    # ---------------------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nAll figures are saved in: {save_dir}\n")

    # File-name suffix so that the absorption models do not overwrite each other
    sfx = f"_{ABSORPTION_MODEL}"

    # ---------------------------------------------------------------
    # Detuning curves for the chosen absorption model
    # ---------------------------------------------------------------

    if need_detuning:
        t0 = time.perf_counter()
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

        if plots.get("detuning"):
            plot_wavelength_detuning(
                stark_curves,
                stark_thresholds,
                lineshape_label=ABSORPTION_LABEL,
                lineshape_kind=ABSORPTION_KIND,
                fit_lines=stark_fit_lines,
                savepath=os.path.join(save_dir, f"pa_wavelength_detuning{sfx}.pdf")
            )

        # -----------------------------------------------------------
        # Separate summary plots versus wavelength
        # -----------------------------------------------------------
        if plots.get("smax"):
            plot_smax_vs_wavelength(
                stark_fit_lines,
                savepath=os.path.join(save_dir, f"pa_smax_vs_wavelength{sfx}.pdf")
            )

        analytical_dense = analytical_threshold_curve(p8, ABSORPTION_KIND)
        if plots.get("ith"):
            plot_ith_vs_wavelength(
                stark_thresholds,
                stark_ana_thresholds,  # also pass the analytical thresholds
                savepath=os.path.join(save_dir, f"pa_ith_vs_wavelength{sfx}.pdf"),
                analytical_dense=analytical_dense,
                intensity_window_max=float(I_values[-1])
            )
        save_detuning_summary(save_dir, sfx, p8, stark_thresholds, stark_ana_thresholds,
                              stark_fit_lines, analytical_dense)
        _done("detuning/smax/ith", t0)
    else:
        print("\nDetuning scan skipped (detuning, smax and ith are off).")

    if plots.get("ith_full"):
        t0 = time.perf_counter()
        print(f"\n=== Analytical threshold {ITH_FULL_MIN_NM:g}-{ITH_FULL_MAX_NM:g} nm "
              f"(step {ITH_FULL_STEP_NM:g} nm) ===")
        wl_full, ith_full = analytical_threshold_curve(
            p8, ABSORPTION_KIND, step_nm=ITH_FULL_STEP_NM,
            ranges=[(ITH_FULL_MIN_NM, ITH_FULL_MAX_NM)])
        plot_ith_full_range(
            wl_full, ith_full,
            savepath=os.path.join(save_dir, f"pa_ith_analytical_full_range{sfx}.pdf"),
            csvpath=os.path.join(save_dir, f"pa_ith_analytical_full_range{sfx}.csv"),
            intensity_window_max=float(I_values[-1]))
        _done("ith_full", t0)

    # ---------------------------------------------------------------
    # Rise-time plots
    # ---------------------------------------------------------------
    I_min, I_max = float(I_values[0]), float(I_values[-1])

    if plots.get("rise"):
        t0 = time.perf_counter()
        print("\n=== Rise time vs intensity ===")
        rise_results = compute_rise_vs_intensity(p8, I_min, I_max, rise_wavelengths)
        if rise_results:
            plot_rise_vs_intensity(
                rise_results,
                savepath=os.path.join(save_dir, f"pa_rise_time_vs_intensity{sfx}.pdf")
            )
        else:
            print("  No valid rise-time data: plot skipped.")
        _done("rise", t0)
    else:
        print("\nRise-time scan skipped (rise is off).")

    if plots.get("risewl"):
        t0 = time.perf_counter()
        rise_wl_grid = _uniform_grid_nm(RISE_WL_MIN_NM, RISE_WL_MAX_NM, RISE_WL_STEP_NM)
        print(f"\n=== Longest rise time vs pump wavelength "
              f"({RISE_WL_MIN_NM:g}-{RISE_WL_MAX_NM:g} nm, step {RISE_WL_STEP_NM:g} nm) ===")
        rise_wl_data = compute_rise_vs_wavelength(p8, rise_wl_grid, I_min, I_max)
        plot_rise_vs_wavelength(
            rise_wl_data,
            savepath=os.path.join(save_dir, f"pa_rise_time_vs_wavelength{sfx}.pdf"),
            csvpath=os.path.join(save_dir, f"pa_rise_time_vs_wavelength{sfx}.csv"),
        )
        _done("risewl", t0)
    else:
        print("\nRise time vs wavelength skipped (risewl is off).")

    # The sigma sweep only exists when there are Stark spectra
    if not plots.get("sigma"):
        print("\nSigma sweep skipped (sigma is off).")
    elif _USE_STARK:
        t0 = time.perf_counter()
        print("\n=== Peak rise time vs Stark sigma ===")
        peaks_by_wl = compute_rise_peak_vs_sigma(p8, sigma_wavelengths,
                                                 RISE_SIGMA_VALUES_CM1, I_min, I_max)
        plot_rise_peak_vs_sigma(
            RISE_SIGMA_VALUES_CM1, peaks_by_wl,
            savepath=os.path.join(save_dir, f"pa_rise_peak_vs_sigma{sfx}.pdf")
        )
        _done("sigma", t0)
    else:
        print("\nSigma sweep skipped (only meaningful with Stark spectra).")

    print(f"\nTotal run time: {time.perf_counter() - t_start:.1f} s")

    # Inline (Interactive Window/Spyder): figures appear in the console.
    # Never separate windows unless SHOW_WINDOWS; all figures are saved anyway.
    _show_or_close(no_show)
    if not _inline_backend() and not SHOW_WINDOWS:
        print(f"Figures saved in {save_dir} (no plot windows: SHOW_WINDOWS = False).")


if __name__ == "__main__":
    import argparse
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="GSA/ESA phonon factors in the PA model")
    parser.add_argument("--spectra-only", action="store_true",
                        help="Export factor plots and data; skip kinetics.")
    parser.add_argument("--data-only", action="store_true",
                        help="Export factor CSVs/settings; skip plots and kinetics.")
    parser.add_argument("--no-show", action="store_true", help="Save plots without opening windows.")
    parser.add_argument("--output-dir", default=None, help="Directory for plots/data.")
    parser.add_argument("--phonon-model", choices=("band", "discrete"), default=PHONON_MODEL)
    parser.add_argument("--band-profile", choices=("four_gaussian", "gaussian"),
                        default=BAND_PROFILE)
    parser.add_argument("--band-mean", type=float, default=BAND_MEAN_CM1,
                        help="Centre of the single band (--band-profile gaussian).")
    parser.add_argument("--band-sigma", type=float, default=BAND_SIGMA_CM1,
                        help="Sigma of the single band (--band-profile gaussian).")
    parser.add_argument("--phonon-grid-step", type=float, default=PHONON_GRID_STEP_CM1,
                        help="Quadrature step of the phonon-shift grid in cm^-1.")
    h5_choice = parser.add_mutually_exclusive_group()
    h5_choice.add_argument("--coupling-h5", type=float, default=GSA_H5_HUANG_RHYS)
    h5_choice.add_argument("--calibrate-h5", action="store_true",
                           help="Fit the 3H6 -> 3H5 Huang-Rhys factor to Judd-Ofelt "
                                "(after changing the phonon band or temperature).")
    parser.add_argument("--coupling-f4", type=float, default=GSA_F4_HUANG_RHYS)
    parser.add_argument("--coupling-h4", type=float, default=ESA_H4_HUANG_RHYS)
    parser.add_argument("--coupling-f3", type=float, default=ESA_F3_HUANG_RHYS)
    parser.add_argument("--coupling-f2", type=float, default=ESA_F2_HUANG_RHYS)
    parser.add_argument("--kernel-tolerance", type=float, default=PHONON_TAIL_TOLERANCE)
    parser.add_argument("--scan-range", type=float, nargs=3, metavar=("MIN", "MAX", "STEP"),
                        default=None,
                        help="Pump wavelengths of the detuning scan in nm (default: "
                             f"{SCAN_MIN_NM:g} {SCAN_MAX_NM:g} {SCAN_STEP_NM:g}).")
    parser.add_argument("--include-1450", action="store_true",
                        help="Also scan the 1390-1490 nm ESA band (and its rise times).")
    plot_choice = parser.add_mutually_exclusive_group()
    plot_choice.add_argument("--plots", nargs="+", choices=tuple(PLOTS), metavar="NAME",
                             help="Make only these parts: " + ", ".join(PLOTS) + ".")
    plot_choice.add_argument("--skip-plots", nargs="+", choices=tuple(PLOTS), metavar="NAME",
                             help="Skip these parts; the rest follows PLOTS at the top.")
    # parse_known_args: Jupyter/IPython kernels (VS Code Interactive Window)
    # pass their own arguments such as "-f kernel.json"; ignore those.
    args, _unknown_args = parser.parse_known_args()
    if _unknown_args and not _running_in_ipython():
        parser.error("unrecognized arguments: " + " ".join(_unknown_args))
    if args.plots is not None:
        selected_plots = {name: name in args.plots for name in PLOTS}
    elif args.skip_plots is not None:
        selected_plots = {name: on and name not in args.skip_plots for name, on in PLOTS.items()}
    else:
        selected_plots = dict(PLOTS)
    if args.scan_range is not None:
        SCAN_MIN_NM, SCAN_MAX_NM, SCAN_STEP_NM = args.scan_range
        _uniform_grid_nm(SCAN_MIN_NM, SCAN_MAX_NM, SCAN_STEP_NM)  # validate early
    INCLUDE_1450_BAND = INCLUDE_1450_BAND or args.include_1450
    if ABSORPTION_MODEL == "phonon_band":
        GSA_H5_HUANG_RHYS, GSA_F4_HUANG_RHYS = args.coupling_h5, args.coupling_f4
        ESA_H4_HUANG_RHYS, ESA_F3_HUANG_RHYS, ESA_F2_HUANG_RHYS = (
            args.coupling_h4, args.coupling_f3, args.coupling_f2)
        PHONON_TAIL_TOLERANCE = args.kernel_tolerance
        PHONON_GRID_STEP_CM1 = args.phonon_grid_step
        set_phonon_model(args.phonon_model, args.band_profile, args.band_mean, args.band_sigma)
        if args.calibrate_h5:
            GSA_H5_HUANG_RHYS = calibrate_gsa_h5_huang_rhys()
            print(f"Calibrated 3H6 -> 3H5 Huang-Rhys factor: {GSA_H5_HUANG_RHYS:.4f}")
        GSA_SPECTRUM, ESA_SPECTRUM, _GSA_REFERENCE_FACTOR, _ESA_REFERENCE_FACTOR = _build_spectra()
        print(f"Phonons: {PHONON_DESCRIPTION}")
        for _name, _info in channel_weight_report().items():
            print(f"{_name}: S = {_info['line_strength_1e-20_cm2']:.4f} e-20 cm2, "
                  f"share in {_info['spectrum']} = {_info['share_in_spectrum']:.3f}")
        for _spectrum, _sigma_m2 in ((GSA_SPECTRUM, SI_SIGMA_GSA_1064_M2),
                                     (ESA_SPECTRUM, SI_SIGMA_ESA_1064_M2)):
            for _name, _ratio in anchor_judd_ofelt_ratios(_spectrum, _sigma_m2 * 1e4).items():
                print(f"{_name}: integral from 1064-nm anchor / Judd-Ofelt = {_ratio:.2f}")
    with _no_plot_windows():
        main(spectra_only=args.spectra_only, save_dir=args.output_dir,
             data_only=args.data_only, no_show=args.no_show, plots=selected_plots)