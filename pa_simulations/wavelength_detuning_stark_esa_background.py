# -*- coding: utf-8 -*-
"""
Stark-level Gaussian wavelength detuning
========================================

8% Tm sample with an ESA line shape constructed from the crystal-field
(Stark) levels in Table 4 of the supplied beta-NaYF4:Tm3+ paper.

The absolute manifolds are represented by

    S_J(E) = sum_i A_i exp[-(E-E_i)^2/(2 sigma^2)]

for 3H4 (9 levels), 3F3 (7 levels) and 3F2 (5 levels).

For the wavelength-dependent ESA used by the PA model, the relevant photon
energies are not the absolute upper-state energies but the transition energies
from the populated 3F4 intermediate manifold:

    Delta E_ij = E_i(upper) - E_j(3F4).

The effective ESA spectrum therefore contains all 3F4->3F2 and 3F4->3F3
Stark-to-Stark transitions. With 9 lower 3F4 levels this gives 9*5 + 9*7 =
108 Gaussian transition lines. Within each upper multiplet all transitions are
assigned equal amplitude because Table 4 gives level positions, not individual
transition oscillator strengths. The relative 3F2/3F3 manifold weights and the
Gaussian sigma are explicit model parameters below.

Only the ESA gets this Stark-derived line shape. The GSA retains the original
Gaussian from pa_model. The resulting ESA line shape is globally normalized to
1 before it multiplies the ESA rate coefficient.

The adaptive intensity scan, LSODA fast solver, avalanche detection and fitted
log-log slope are retained from the previous double-Gaussian script.
"""

from __future__ import annotations

import contextlib
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.integrate import solve_ivp
    _HAVE_SCIPY = True
except ImportError:  # fall back to pa_model's own RK4
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
USE_FAST_SOLVER = True
T_STEADY_S = 20.0     # integration time [s]; >> all relaxation times


# Line through the avalanche (see fit_avalanche_line, version 6).
#   SLOPE_FRACTION      : the fit window contains all neighbouring intervals
#                         around the steepest one whose local slope is at
#                         least this fraction of the peak local slope.
#                         Larger -> narrower window, s_max closer to the peak
#                         local slope; smaller -> wider window, lower s_max.
#   LINE_EXTEND_DECADES : the black line is drawn this many decades of
#                         EMISSION beyond the fit window (top and bottom).
SLOPE_FRACTION = 0.2
LINE_EXTEND_DECADES = 0.4


def _safe_for_log(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), _LOG_PLOT_FLOOR, None)


# ---------------------------------------------------------------------------
# Stark-level parameters from Table 4 [cm^-1]
# ---------------------------------------------------------------------------

# Intermediate manifold populated before ESA.
E_3F4_CM1 = np.array([
    5634.0,
    5661.0,
    5689.0,
    5717.0,
    5730.0,
    5750.0,
    5761.0,
    5782.0,
    5823.0,
])

# Upper / emitting manifolds requested from Table 4.
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
# Phenomenological broadening / relative multiplet weights
# ---------------------------------------------------------------------------

# Table 4 supplies the Stark-level energies but not a linewidth for these
# transitions. This sigma is therefore an explicit MODEL PARAMETER.
# Units are cm^-1 because the Stark energies are given in cm^-1.
STARK_SIGMA_CM1 = 8.5

# Relative total contributions of the two ESA upper manifolds.  They are set
# equal by default, analogous to the previous 50/50 double-Gaussian model.
# Individual Stark-to-Stark lines inside a manifold are equally weighted.
ESA_WEIGHT_3F2 = 0.5
ESA_WEIGHT_3F3 = 0.5

# IMPORTANT: the Stark transitions themselves occupy only roughly 1058-1183 nm.
# If they are used alone, the ESA factor becomes essentially zero at shorter
# wavelengths (850-1030 nm), which removes the avalanche and produces the
# nearly straight/flat curves seen in the broken version.
#
# Keep the broad ESA envelope that was already present in the previous
# double-Gaussian model and place the Stark fine structure on top of it.
# 0.0 -> pure Stark spectrum; 1.0 -> pure broad pa_model ESA Gaussian.
ESA_BACKGROUND_WEIGHT = 0.5
ESA_BACKGROUND_LAMBDA0_NM = DEFAULT_LAMBDA0_ESA_NM
ESA_BACKGROUND_SIGMA_NM = DEFAULT_SIGMA_LAMBDA_ESA_NM

if STARK_SIGMA_CM1 <= 0.0:
    raise ValueError("STARK_SIGMA_CM1 must be > 0.")

if ESA_WEIGHT_3F2 < 0.0 or ESA_WEIGHT_3F3 < 0.0:
    raise ValueError("ESA manifold weights must be non-negative.")

if ESA_WEIGHT_3F2 + ESA_WEIGHT_3F3 <= 0.0:
    raise ValueError("At least one ESA manifold weight must be > 0.")

if not (0.0 <= ESA_BACKGROUND_WEIGHT <= 1.0):
    raise ValueError("ESA_BACKGROUND_WEIGHT must lie between 0 and 1.")

# Normalize the relative multiplet weights to sum to one.
_WEIGHT_SUM = ESA_WEIGHT_3F2 + ESA_WEIGHT_3F3
ESA_WEIGHT_3F2 /= _WEIGHT_SUM
ESA_WEIGHT_3F3 /= _WEIGHT_SUM

# Explicit amplitudes for the absolute-manifold S_J(E) functions.  Table 4
# does not provide line-by-line oscillator strengths, so the default is A_i=1.
A_3H4 = np.ones(len(E_3H4_CM1), dtype=float)
A_3F3 = np.ones(len(E_3F3_CM1), dtype=float)
A_3F2 = np.ones(len(E_3F2_CM1), dtype=float)

# Keep the original pa_model Gaussian so the GSA is unchanged.
_ORIGINAL_GAUSSIAN_FACTOR = pa_model.wavelength_gaussian_factor
_ESA_LAMBDA0_M = DEFAULT_LAMBDA0_ESA_NM * 1e-9


# ---------------------------------------------------------------------------
# Absolute multiplet functions S_3H4(E), S_3F3(E), S_3F2(E)
# ---------------------------------------------------------------------------

def _stark_gaussian_unnormalized(
        energy_cm1: np.ndarray | float,
        levels_cm1: np.ndarray,
        sigma_cm1: float,
        amplitudes: Optional[np.ndarray] = None) -> np.ndarray | float:
    """
    Sum of Gaussian lines centred on crystal-field energies.

        S(E) = sum_i A_i exp[-(E-E_i)^2/(2 sigma^2)]

    Energy and sigma are both expressed in cm^-1.
    """

    energy = np.asarray(energy_cm1, dtype=float)
    levels = np.asarray(levels_cm1, dtype=float)

    if sigma_cm1 <= 0.0:
        raise ValueError("sigma_cm1 must be > 0.")

    if amplitudes is None:
        amps = np.ones(len(levels), dtype=float)
    else:
        amps = np.asarray(amplitudes, dtype=float)

    if amps.shape != levels.shape:
        raise ValueError(
            "amplitudes must have exactly the same length as levels_cm1."
        )

    delta = np.expand_dims(energy, axis=-1) - levels
    gaussians = np.exp(-0.5 * (delta / sigma_cm1) ** 2)
    result = np.sum(amps * gaussians, axis=-1)

    if energy.ndim == 0:
        return float(result)

    return result


def S_3H4(
        energy_cm1: np.ndarray | float,
        sigma_cm1: float = STARK_SIGMA_CM1,
        amplitudes: Optional[np.ndarray] = None) -> np.ndarray | float:
    """Gaussian sum for the nine 3H4 Stark levels from Table 4."""
    amps = A_3H4 if amplitudes is None else amplitudes
    return _stark_gaussian_unnormalized(
        energy_cm1, E_3H4_CM1, sigma_cm1, amps
    )


def S_3F3(
        energy_cm1: np.ndarray | float,
        sigma_cm1: float = STARK_SIGMA_CM1,
        amplitudes: Optional[np.ndarray] = None) -> np.ndarray | float:
    """Gaussian sum for the seven 3F3 Stark levels from Table 4."""
    amps = A_3F3 if amplitudes is None else amplitudes
    return _stark_gaussian_unnormalized(
        energy_cm1, E_3F3_CM1, sigma_cm1, amps
    )


def S_3F2(
        energy_cm1: np.ndarray | float,
        sigma_cm1: float = STARK_SIGMA_CM1,
        amplitudes: Optional[np.ndarray] = None) -> np.ndarray | float:
    """Gaussian sum for the five 3F2 Stark levels from Table 4."""
    amps = A_3F2 if amplitudes is None else amplitudes
    return _stark_gaussian_unnormalized(
        energy_cm1, E_3F2_CM1, sigma_cm1, amps
    )


# ---------------------------------------------------------------------------
# Effective ESA spectrum from 3F4 -> 3F2 and 3F4 -> 3F3 transitions
# ---------------------------------------------------------------------------

# Every upper Stark level can, in this simplified model, be reached from every
# lower 3F4 Stark level.  This yields 45 + 63 = 108 transition energies.
ESA_TRANSITIONS_3F2_CM1 = (
    E_3F2_CM1[:, None] - E_3F4_CM1[None, :]
).ravel()

ESA_TRANSITIONS_3F3_CM1 = (
    E_3F3_CM1[:, None] - E_3F4_CM1[None, :]
).ravel()

# Equal line strengths *within* each upper multiplet. Using 1/N here means
# ESA_WEIGHT_3F2 and ESA_WEIGHT_3F3 control the total multiplet contributions
# instead of the multiplet with more Stark-to-Stark pairs automatically
# receiving more weight.
A_ESA_3F2 = np.full(
    len(ESA_TRANSITIONS_3F2_CM1),
    1.0 / len(ESA_TRANSITIONS_3F2_CM1),
    dtype=float
)

A_ESA_3F3 = np.full(
    len(ESA_TRANSITIONS_3F3_CM1),
    1.0 / len(ESA_TRANSITIONS_3F3_CM1),
    dtype=float
)


def _esa_stark_unnormalized(
        energy_cm1: np.ndarray | float) -> np.ndarray | float:
    """
    Unnormalized effective ESA line shape in photon wavenumber [cm^-1].

    It is the weighted sum of all 3F4->3F2 and 3F4->3F3 Gaussian
    Stark-to-Stark transitions.
    """

    f_3f2 = _stark_gaussian_unnormalized(
        energy_cm1,
        ESA_TRANSITIONS_3F2_CM1,
        STARK_SIGMA_CM1,
        A_ESA_3F2
    )

    f_3f3 = _stark_gaussian_unnormalized(
        energy_cm1,
        ESA_TRANSITIONS_3F3_CM1,
        STARK_SIGMA_CM1,
        A_ESA_3F3
    )

    return ESA_WEIGHT_3F2 * f_3f2 + ESA_WEIGHT_3F3 * f_3f3


def _esa_stark_only_normalization() -> float:
    """GLOBAL maximum of the pure Stark-transition ESA spectrum."""

    all_transitions = np.concatenate([
        ESA_TRANSITIONS_3F2_CM1,
        ESA_TRANSITIONS_3F3_CM1,
    ])

    energy_min = float(np.min(all_transitions) - 10.0 * STARK_SIGMA_CM1)
    energy_max = float(np.max(all_transitions) + 10.0 * STARK_SIGMA_CM1)

    energy_grid = np.linspace(energy_min, energy_max, 100000)
    values = _esa_stark_unnormalized(energy_grid)
    normalization = float(np.max(values))

    if not np.isfinite(normalization) or normalization <= 0.0:
        raise ValueError("Pure Stark-ESA normalization is zero or non-finite.")

    return normalization


# Normalize the Stark fine-structure component first so that the mixing weight
# below has a transparent meaning (both the Stark component and the broad
# Gaussian have a peak of 1 before they are mixed).
_ESA_STARK_ONLY_NORM = _esa_stark_only_normalization()


def _esa_stark_plus_background_unnormalized(
        wavelength_m: np.ndarray | float) -> np.ndarray | float:
    """
    Effective ESA line shape = Stark fine structure + broad ESA envelope.

        f(lambda) = (1-bg) * f_Stark(lambda) + bg * f_broad(lambda)

    where both components are individually peak-normalized before mixing.

    The broad component is the same Gaussian ESA envelope used by pa_model
    (lambda0 = 1060 nm, sigma = 150 nm by default).  It prevents the ESA rate
    from collapsing to ~0 away from the narrow Stark-transition cluster.
    """

    wavelength = np.asarray(wavelength_m, dtype=float)

    if np.any(wavelength <= 0.0):
        raise ValueError("wavelength_m must be > 0.")

    # Photon wavenumber [cm^-1].
    energy_cm1 = 1.0e-2 / wavelength

    stark = _esa_stark_unnormalized(energy_cm1) / _ESA_STARK_ONLY_NORM

    lambda_bg_m = ESA_BACKGROUND_LAMBDA0_NM * 1e-9
    sigma_bg_m = ESA_BACKGROUND_SIGMA_NM * 1e-9
    broad = np.exp(
        -0.5 * ((wavelength - lambda_bg_m) / sigma_bg_m) ** 2
    )

    value = (
        (1.0 - ESA_BACKGROUND_WEIGHT) * stark
        + ESA_BACKGROUND_WEIGHT * broad
    )

    if wavelength.ndim == 0:
        return float(value)

    return value


def _esa_stark_background_normalization() -> float:
    """GLOBAL maximum of the combined Stark + broad ESA line shape."""

    # Cover the wavelength range used by the detuning scan with generous
    # margins so the true maximum is captured regardless of small parameter
    # changes.
    wavelength_grid_m = np.linspace(600.0e-9, 1600.0e-9, 200000)
    values = _esa_stark_plus_background_unnormalized(wavelength_grid_m)
    normalization = float(np.max(values))

    if not np.isfinite(normalization) or normalization <= 0.0:
        raise ValueError(
            "Combined Stark/background ESA normalization is zero or non-finite."
        )

    return normalization


_ESA_STARK_NORM = _esa_stark_background_normalization()


def _wavelength_stark_esa_factor(
        wavelength_m: float,
        lambda0_m: float,
        sigma_m: float) -> float:
    """
    Peak-normalized Stark-derived ESA factor with a broad background.

    lambda0_m and sigma_m are intentionally unused for the Stark component;
    the transition positions and Stark linewidth are defined above. The broad
    envelope uses ESA_BACKGROUND_LAMBDA0_NM / ESA_BACKGROUND_SIGMA_NM.
    """

    value = _esa_stark_plus_background_unnormalized(wavelength_m)
    return float(value / _ESA_STARK_NORM)


def _wavelength_stark_esa_only(
        wavelength_m: float,
        lambda0_m: float,
        sigma_m: float) -> float:
    """
    Apply the Stark-derived spectrum only to ESA. GSA keeps pa_model's
    original Gaussian line shape.
    """

    if abs(lambda0_m - _ESA_LAMBDA0_M) < 1e-12:
        return _wavelength_stark_esa_factor(
            wavelength_m,
            lambda0_m,
            sigma_m
        )

    return _ORIGINAL_GAUSSIAN_FACTOR(
        wavelength_m,
        lambda0_m,
        sigma_m
    )


# ---------------------------------------------------------------------------
# Existing Cauchy line shape
# ---------------------------------------------------------------------------

def _wavelength_cauchy_factor(
        wavelength_m: float,
        lambda0_m: float,
        gamma_m: float) -> float:
    """Cauchy/Lorentzian resonance line shape, peak-normalized to 1."""

    detuning = wavelength_m - lambda0_m

    return float(
        1.0 / (1.0 + (detuning / gamma_m) ** 2)
    )


# ---------------------------------------------------------------------------
# Select line shape
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _lineshape(kind: str):
    """
    Temporarily select the resonance line shape used inside pa_model.
    """

    original = pa_model.wavelength_gaussian_factor

    if kind == "gaussian":

        replacement = original

    elif kind == "stark_gaussian":

        replacement = _wavelength_stark_esa_only

    elif kind == "cauchy":

        replacement = _wavelength_cauchy_factor

    else:

        raise ValueError(
            f"Unknown lineshape kind: {kind!r}"
        )

    pa_model.wavelength_gaussian_factor = replacement

    try:
        yield

    finally:
        pa_model.wavelength_gaussian_factor = original


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
# Line through the avalanche (NEW in version 6)
# ---------------------------------------------------------------------------

def fit_avalanche_line(
        I_values: np.ndarray,
        emission: np.ndarray,
        fraction: float = SLOPE_FRACTION) -> Dict[str, float]:
    """
    Fit a straight line (in the log-log plot) through the steep rise of the
    photon avalanche. Its slope is s_max.

    Where the line is drawn:
      1. Local slope of every interval between neighbouring points,
         s_j = d log10(E) / d log10(I).
      2. Steepest interval k (peak local slope).
      3. Fit window = contiguous intervals around k with
         s_j >= fraction * s_k (walk left and right from k until the local
         slope drops below the threshold).
      4. Least-squares line through the curve in that window. The
         piecewise-linear log-log curve is resampled uniformly in log I
         first, so that regions with many grid points do not dominate.

    Returns a dict with
      slope, intercept : log10(E) = slope * log10(I) + intercept
      x_range, y_range : (start, end) of the fit window in log10(I) and
                         log10(E)
      s_peak           : peak local slope (for comparison)
      n_points         : number of grid points inside the window
    """

    I_values = np.asarray(I_values, dtype=float)
    emission = np.asarray(emission, dtype=float)

    # log10 needs positive values (E = 0 would give a spurious huge slope).
    good = (I_values > 0.0) & (emission > 0.0) & np.isfinite(emission)
    x = np.log10(I_values[good])
    y = np.log10(emission[good])

    slopes = np.diff(y) / np.diff(x)

    k = int(np.argmax(slopes))
    threshold = fraction * slopes[k]

    lo = k
    while lo > 0 and slopes[lo - 1] >= threshold:
        lo -= 1

    hi = k
    while hi < len(slopes) - 1 and slopes[hi + 1] >= threshold:
        hi += 1

    # Interval j lies between points j and j+1 -> points lo .. hi+1.
    x_win = x[lo:hi + 2]
    y_win = y[lo:hi + 2]

    x_uniform = np.linspace(x_win[0], x_win[-1], 200)
    y_uniform = np.interp(x_uniform, x_win, y_win)

    slope, intercept = np.polyfit(x_uniform, y_uniform, 1)

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "x_range": (float(x_win[0]), float(x_win[-1])),
        "y_range": (float(y_win[0]), float(y_win[-1])),
        "s_peak": float(slopes[k]),
        "n_points": int(len(x_win)),
    }


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
            f"{mine} vs {ref}. Set USE_FAST_SOLVER = False."
        )


def _emission_curve(
        I_values: np.ndarray,
        p8,
        pump_wavelength_nm: float) -> np.ndarray:
    """
    800-nm steady-state emission for every intensity in I_values.
    Must be called while the wanted line shape is active (_lineshape).
    """

    if not (USE_FAST_SOLVER and _HAVE_SCIPY):

        return power_dependence(
            I_values,
            p8,
            pump_wavelength_nm=pump_wavelength_nm
        )

    # Rate coefficients: once per wavelength (not once per RHS call).
    w_gsa, w_esa = pa_model.gsa_esa_rate_coefficients(
        pump_wavelength_nm,
        p8
    )

    emission = np.empty(len(I_values))

    for i, I_pump in enumerate(I_values):

        sol = solve_ivp(
            _make_rhs(float(I_pump), p8, w_gsa, w_esa),
            (0.0, T_STEADY_S),
            (1.0, 0.0, 0.0),
            method="LSODA",
            rtol=1e-9,
            atol=1e-16
        )

        n2 = max(float(sol.y[2, -1]), 0.0)

        emission[i] = emission_from_population(n2, p8)

    return emission


def adaptive_knee_scan(
        p8,
        pump_wavelength_nm: float,
        I_min: float,
        I_max: float,
        n_coarse: int = N_COARSE,
        n_knee: int = N_KNEE,
        n_polish: int = N_POLISH):
    """
    Emission curve with few points far from the knee and many points at the
    knee, plus Ith* (intensity of maximum slope) and s_max (slope of the
    line fitted through the avalanche, see fit_avalanche_line).

    1. Sparse scan (n_coarse log-spaced points) over the whole window. If it
       contains no avalanche (see has_avalanche) we stop here: no threshold.
    2. Dense scan (n_knee points) between the coarse points 2 cells below and
       2 cells above the coarse slope maximum.
    3. Very dense scan (n_polish points) between the dense points 2 cells
       below and 2 cells above the dense slope maximum. Ith* is read off this
       last scan.
    4. The three scans are merged and a line is fitted through the steep part
       of the merged curve (fit_avalanche_line); its slope is s_max.

    This needs ~110 steady-state solves per wavelength for an avalanche
    (30 for a curve without one).

    Returns (I_all, E_all, Ith_star, s_max, avalanche_present, fit);
    I_all/E_all are the sorted union of all scans, used for plotting.
    `fit` is the dict of fit_avalanche_line (None if there is no avalanche).
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

    # edge_margin = 1: on a sparse grid the knee may legitimately sit
    # one cell from the edge of the window.
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

    # Union of the three scans, sorted, near-duplicates removed.
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

    # s_max = slope of the line through the steep part of the merged curve.
    fit = fit_avalanche_line(I_all, E_all)
    s_max = fit["slope"]

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
    """
    curves[lam] = (I_values_for_this_wavelength, emission)
    fit_lines[lam] = dict from fit_avalanche_line, or None (no line drawn)
    """

    # Ith* = intensity of maximum slope (converged), used for all line shapes.
    ith_label = "Ith*"

    fig, ax = plt.subplots(figsize=(8.0, 5.8))

    colors = plt.cm.tab20(
        np.linspace(0, 1, len(curves))
    )

    I_lo = min(float(I.min()) for I, _E in curves.values())
    I_hi = max(float(I.max()) for I, _E in curves.values())

    for (lam, (I_lam, emission)), color in zip(curves.items(), colors):

        ith = thresholds[lam]

        if np.isfinite(ith):

            label = (
                f"{lam:.0f} nm "
                f"({ith_label} = {ith:.1f} "
                r"kW cm$^{-2}$)"
            )
            linestyle = "o-"

        else:

            # No avalanche in the simulated window: no threshold exists.
            label = f"{lam:.0f} nm (no PA in range)"
            linestyle = "o--"

        ax.loglog(
            I_lam,
            _safe_for_log(emission),
            linestyle,
            markersize=3.5,
            color=color,
            label=label
        )

    # Black lines through the avalanche (slope = s_max). Not in the legend:
    # labels starting with an underscore are ignored by legend().
    if fit_lines:

        for fit in fit_lines.values():

            if fit is None or fit["slope"] <= 0.0:
                continue

            s = fit["slope"]
            b = fit["intercept"]

            # Extend the line LINE_EXTEND_DECADES of emission beyond the
            # fit window (top and bottom) and convert to intensity.
            y0 = fit["y_range"][0] - LINE_EXTEND_DECADES
            y1 = fit["y_range"][1] + LINE_EXTEND_DECADES
            x0 = (y0 - b) / s
            x1 = (y1 - b) / s

            ax.loglog(
                [10.0 ** x0, 10.0 ** x1],
                [10.0 ** y0, 10.0 ** y1],
                "-",
                color="k",
                linewidth=1.3,
                zorder=5,
                label="_nolegend_"
            )

    ax.set_xlabel(
        r"Excitation intensity (kW cm$^{-2}$)"
    )

    ax.set_ylabel(
        "800-nm emission intensity (a.u.)"
    )

    ax.set_xlim(I_lo, I_hi)

    ax.grid(
        alpha=0.3,
        which="both"
    )

    ax.legend(
        fontsize=8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3
    )

    fig.tight_layout()

    if savepath:
        fig.savefig(
            savepath,
            format="pdf",
            bbox_inches="tight"
        )

    # Do not call plt.show() here. main() first constructs all three
    # figures and displays them together at the end.


# ---------------------------------------------------------------------------
# Summary plots: s_max and Ith versus wavelength
# ---------------------------------------------------------------------------

def plot_smax_vs_wavelength(
        fit_lines: Dict[float, Optional[Dict[str, float]]]) -> None:
    """
    Plot fitted photon-avalanche nonlinearity s_max versus pump wavelength.

    If no avalanche/fitted line exists at a wavelength, that point is NaN
    and is not connected to neighbouring valid points.
    """

    wavelengths = np.array(
        sorted(fit_lines.keys()),
        dtype=float
    )

    s_values = np.array([
        (
            float(fit_lines[lam]["slope"])
            if fit_lines[lam] is not None
            else np.nan
        )
        for lam in wavelengths
    ])

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    ax.plot(
        wavelengths,
        s_values,
        "o-",
        linewidth=1.6,
        markersize=5
    )

    ax.set_xlabel("Pump wavelength (nm)")
    ax.set_ylabel(r"$s_{\mathrm{max}}$")
    ax.set_title(
        r"Maximum nonlinearity $s_{\mathrm{max}}$ vs. pump wavelength"
    )

    ax.grid(
        alpha=0.3,
        which="both"
    )

    fig.tight_layout()


def plot_ith_vs_wavelength(
        thresholds: Dict[float, float]) -> None:
    """
    Plot photon-avalanche threshold Ith* versus pump wavelength.

    Ith* is the intensity at the maximum local log-log slope returned by
    adaptive_knee_scan. Wavelengths without a detected avalanche contain NaN.
    """

    wavelengths = np.array(
        sorted(thresholds.keys()),
        dtype=float
    )

    ith_values = np.array([
        float(thresholds[lam])
        for lam in wavelengths
    ])

    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    ax.plot(
        wavelengths,
        ith_values,
        "o-",
        linewidth=1.6,
        markersize=5
    )

    ax.set_xlabel("Pump wavelength (nm)")
    ax.set_ylabel(r"$I_{\mathrm{th}}^{*}$ (kW cm$^{-2}$)")
    ax.set_title(
        r"Photon-avalanche threshold $I_{\mathrm{th}}^{*}$ vs. pump wavelength"
    )

    ax.grid(
        alpha=0.3,
        which="both"
    )

    fig.tight_layout()


# ---------------------------------------------------------------------------
# Curve generation
# ---------------------------------------------------------------------------

def compute_detuning_curves(
        I_values: np.ndarray,
        p8,
        wavelengths_nm,
        kind: str,
        label: str):
    """
    For "gaussian" / "stark_gaussian" only the range I_values[0] ..
    I_values[-1] is used: the intensity points themselves are chosen
    adaptively per wavelength (adaptive_knee_scan). For "cauchy" the uniform
    grid I_values is used as before.

    Returns (curves, thresholds, fit_lines) with
    curves[lam] = (I_lam, emission_lam) and fit_lines[lam] = fit dict / None.
    """

    curves: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
    thresholds: Dict[float, float] = {}
    fit_lines: Dict[float, Optional[Dict[str, float]]] = {}

    print(
        f"\n=== Wavelength-detuning demonstration "
        f"(8% Tm sample, {label} line shape) ==="
    )

    if kind == "stark_gaussian":

        f2_wavelengths_nm = 1.0e7 / ESA_TRANSITIONS_3F2_CM1
        f3_wavelengths_nm = 1.0e7 / ESA_TRANSITIONS_3F3_CM1

        print(
            f"  Stark sigma = {STARK_SIGMA_CM1:.1f} cm^-1 "
            f"(model parameter; not supplied by Table 4)"
        )

        print(
            f"  3F4 -> 3F2: {len(ESA_TRANSITIONS_3F2_CM1)} transitions, "
            f"{np.min(f2_wavelengths_nm):.1f}-"
            f"{np.max(f2_wavelengths_nm):.1f} nm, "
            f"weight = {ESA_WEIGHT_3F2:.3f}"
        )

        print(
            f"  3F4 -> 3F3: {len(ESA_TRANSITIONS_3F3_CM1)} transitions, "
            f"{np.min(f3_wavelengths_nm):.1f}-"
            f"{np.max(f3_wavelengths_nm):.1f} nm, "
            f"weight = {ESA_WEIGHT_3F3:.3f}"
        )

        print(
            f"  Total ESA Stark transitions = "
            f"{len(ESA_TRANSITIONS_3F2_CM1) + len(ESA_TRANSITIONS_3F3_CM1)}"
        )

        print(
            f"  Line shape applied to: ESA only "
            f"(GSA keeps pa_model Gaussian)"
        )

        print(
            f"  Broad ESA background: center = "
            f"{ESA_BACKGROUND_LAMBDA0_NM:.1f} nm, sigma = "
            f"{ESA_BACKGROUND_SIGMA_NM:.1f} nm, weight = "
            f"{ESA_BACKGROUND_WEIGHT:.3f}"
        )

        print(
            f"  Combined global normalization = {_ESA_STARK_NORM:.6f}"
        )

        print(
            f"  s_max = slope of a line fitted through the avalanche "
            f"(window: local slope >= {SLOPE_FRACTION:.2f} x peak)"
        )

    with _lineshape(kind):

        if (
                USE_FAST_SOLVER
                and _HAVE_SCIPY
                and kind in ("gaussian", "stark_gaussian")):

            _check_fast_rhs(p8, float(wavelengths_nm[0]))

            print(
                "  Fast steady-state solver (LSODA); right-hand side "
                "verified against pa_model.pa_rate_equations."
            )

        for lam in wavelengths_nm:

            if kind in ("gaussian", "stark_gaussian"):

                I_lam, emission_lam, Ith_lam, s_max, present, fit = (
                    adaptive_knee_scan(
                        p8,
                        lam,
                        float(I_values[0]),
                        float(I_values[-1])
                    )
                )

                curves[lam] = (I_lam, emission_lam)
                thresholds[lam] = Ith_lam
                fit_lines[lam] = fit

                if not present:

                    print(
                        f"  lambda_pump = {lam:6.1f} nm | "
                        f"Ith* =     n/a         | "
                        f"s_max = {s_max:6.2f} | "
                        f"E({I_lam[-1]:.0f} kW/cm^2) = "
                        f"{emission_lam[-1]:.3g}"
                        f"  <-- no avalanche in "
                        f"{I_lam[0]:.0f}-{I_lam[-1]:.0f} kW/cm^2"
                    )

                else:

                    print(
                        f"  lambda_pump = {lam:6.1f} nm | "
                        f"Ith* = {Ith_lam:7.2f} kW/cm^2 | "
                        f"s_max (line) = {s_max:6.2f} "
                        f"(peak local slope {fit['s_peak']:6.2f}) | "
                        f"fit over "
                        f"{10.0 ** fit['x_range'][0]:.2f}-"
                        f"{10.0 ** fit['x_range'][1]:.2f} kW/cm^2 "
                        f"({fit['n_points']} pts) | "
                        f"E({I_lam[-1]:.0f} kW/cm^2) = "
                        f"{emission_lam[-1]:.3g}"
                    )

            else:

                emission_lam = power_dependence(
                    I_values,
                    p8,
                    pump_wavelength_nm=lam
                )

                curves[lam] = (np.asarray(I_values), emission_lam)

                Ith_lam, s_max_converged = (
                    pa_threshold_cauchy(
                        p8,
                        I_values,
                        lam,
                        emission_coarse=emission_lam
                    )
                )

                thresholds[lam] = Ith_lam
                fit_lines[lam] = None

                print(
                    f"  lambda_pump = {lam:6.1f} nm | "
                    f"Ith* = {Ith_lam:7.2f} kW/cm^2 | "
                    f"s_max = {s_max_converged:6.2f} | "
                    f"E({I_values[-1]:.0f} kW/cm^2) = "
                    f"{emission_lam[-1]:.3g}"
                )

    return curves, thresholds, fit_lines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    # Only I_values[0] and I_values[-1] matter for the adaptive
    # Gaussian/Stark line-shape scans: the points inside the window are chosen adaptively.
    I_values = np.logspace(
        np.log10(20.0),
        np.log10(200.0),
        100
    )

    p8 = parameters_for_sample(3)

    wavelengths_nm = [
        850.0,
        900.0,
        930.0,
        960.0,
        980.0,
        1000.0,
        1030.0,
        1040.0,
        1050.0,
        1060.0,
        1070.0,
        1080.0,
        1090.0,
        1130.0,
        1180.0
    ]

    # ---------------------------------------------------------------
    # Stark-derived Gaussian ESA
    # ---------------------------------------------------------------

    (
        stark_curves,
        stark_thresholds,
        stark_fit_lines
    ) = compute_detuning_curves(
        I_values,
        p8,
        wavelengths_nm,
        kind="stark_gaussian",
        label="Stark Gaussian + broad background"
    )

    plot_wavelength_detuning(
        stark_curves,
        stark_thresholds,
        lineshape_label="Stark Gaussian + broad background",
        lineshape_kind="stark_gaussian",
        fit_lines=stark_fit_lines,
        savepath="pa_wavelength_detuning_stark_gaussian.pdf"
    )


    # ---------------------------------------------------------------
    # Separate summary plots versus wavelength
    # ---------------------------------------------------------------

    plot_smax_vs_wavelength(
        stark_fit_lines
    )

    plot_ith_vs_wavelength(
        stark_thresholds
    )

    # Display all figures together.
    #
    # block=False prevents Spyder from remaining busy until the plot windows
    # are manually closed. The two new summary figures are not saved.
    plt.show(block=False)
    plt.pause(0.1)


if __name__ == "__main__":
    main()