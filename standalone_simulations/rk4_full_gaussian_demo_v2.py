# -*- coding: utf-8 -*-
"""
rk4_full_gaussian_demo_v2.py
=====================================

DRE/RK4 model for the Tm3+-doped NaYF4 photon-avalanche nanoparticles
reported by Lee et al., Nature 589, 230-235 (2021).

This version fixes several issues in the previous script and adds a new
wavelength-dependence feature. Summary of changes relative to
photon_avalanche_rk4_corrected.py:

CORRECTIONS
-----------
1. Q22 / Q23 denominator.
   SI Eqns. S5-S6 read  Q = a * c^3 / (c^2 + 4.32),  i.e. the constant
   4.32 is NOT squared. The previous script used (c**2 + 4.32**2), which
   is wrong. Fixed in `transfer_coefficients()`.

2. Threshold determination.
   The previous script used an arbitrary "first point where the local
   log-log slope reaches 15" criterion (`threshold_from_slope15`) for the
   Fig. 3b-style calculations, which is not how the paper defines I_th.
   SI Fig. S3 defines the threshold as the INTERSECTION of two straight
   lines fitted to the emission-vs-excitation curve on a log-log plot,
   one below and one above the knee. This script now uses a single,
   canonical broken-line-intersection routine, `pa_threshold_broken_line`,
   everywhere a threshold is needed (Fig. 2b, Fig. 3a, Fig. 3b analogues).

3. Steady-state populations.
   The previous script solved for the stationary populations with
   `scipy.optimize.root`, i.e. algebraically, not by integrating the
   DREs. This script now obtains the steady state exclusively by RK4
   time-integration of the DREs, always started from the physical
   initial condition (ground state: N0=1, N1=0, N2=0), continued until
   the state stops changing (see `steady_state_solution`). This matches
   how Lee et al. describe obtaining steady states in the SI Methods
   ("populations at steady state are derived by solving the integrated
   rate equation with the Runge-Kutta 4th order method").

NEW PHYSICS: WAVELENGTH-DEPENDENT GSA / ESA
---------------------------------------------
The GSA and ESA rate coefficients are no longer fixed at their 1064-nm
values. Each is now the product of:
  (a) the ordinary photon-energy scaling of rate = sigma * I / (h*nu),
      evaluated at the actual pump wavelength, and
  (b) an illustrative Gaussian resonance line shape centered on a
      transition wavelength lambda0, with width sigma_lambda:

        W(lambda) = W0 * exp( -(lambda - lambda0)^2 / (2 * sigma_lambda^2) )

  so that excitation further from resonance is more strongly
  suppressed (i.e., requires more phonons to bridge the mismatch, per
  the qualitative Energy Gap Law argument made in the paper's
  Supplementary Discussion).

  lambda0_gsa_nm, lambda0_esa_nm, sigma_lambda_gsa_nm and
  sigma_lambda_esa_nm are now explicit, tunable parameters of
  PAParameters. By default both resonance centers are placed at
  1064 nm (the wavelength at which sigma_G_M2 / sigma_ESA_M2 were
  originally quoted/fit), so pumping exactly at 1064 nm reproduces the
  previous (unweighted) behaviour exactly; detuning away from 1064 nm
  illustrates the falloff.

Reference:
Lee et al., "Giant nonlinear optical responses from photon-avalanching
nanoparticles", Nature 589, 230-235 (2021), Supplementary Information.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. Physical constants / pump conversion
# ---------------------------------------------------------------------------

H_PLANCK = 6.62607015e-34       # J s
C_LIGHT = 2.99792458e8          # m s^-1
LAMBDA_PUMP_DEFAULT_NM = 1064.0  # nm, reference wavelength of the SI fits
NM_TO_M = 1.0e-9


def cross_section_to_rate_coeff(sigma_m2: float,
                                wavelength_m: float) -> float:
    """Return absorption rate coefficient [s^-1 / (kW cm^-2)] at wavelength_m.

    rate = sigma * I / (h*nu), with I expressed in kW/cm^2 (1 kW cm^-2 =
    1e7 W m^-2). This is the ordinary photon-flux scaling: for the same
    intensity (power per area), longer-wavelength photons carry less
    energy each, so more photons arrive per second, changing the rate.
    """
    photon_energy = H_PLANCK * C_LIGHT / wavelength_m
    intensity_si = 1.0e7  # 1 kW cm^-2 = 1e7 W m^-2
    return sigma_m2 * intensity_si / photon_energy


def wavelength_gaussian_factor(wavelength_m: float,
                               lambda0_m: float,
                               sigma_lambda_m: float) -> float:
    """Illustrative Gaussian resonance line shape, peak-normalized to 1.

    W(lambda)/W(lambda0) = exp(-(lambda-lambda0)^2 / (2 sigma_lambda^2))
    """
    detuning = wavelength_m - lambda0_m
    return float(np.exp(-(detuning ** 2) / (2.0 * sigma_lambda_m ** 2)))


def rate_coefficient(sigma_m2: float,
                     wavelength_m: float,
                     lambda0_m: float,
                     sigma_lambda_m: float) -> float:
    """Effective absorption rate coefficient [s^-1/(kW cm^-2)] at wavelength_m,
    combining the photon-energy scaling with the Gaussian detuning factor.
    Reduces exactly to `cross_section_to_rate_coeff` when wavelength_m ==
    lambda0_m (Gaussian factor = 1).
    """
    base = cross_section_to_rate_coeff(sigma_m2, wavelength_m)
    return base * wavelength_gaussian_factor(wavelength_m, lambda0_m, sigma_lambda_m)


# Reference rate coefficients at 1064 nm, purely for printed diagnostics
# (they are NOT used directly in the rate equations anymore -- those call
# `rate_coefficient` at the actual pump wavelength every time).
SIGMA_G_M2 = 6.0e-4 * 1e-25
SIGMA_ESA_M2 = 6.4e-25
RATE_GSA_PER_KWCM2_AT_1064 = cross_section_to_rate_coeff(SIGMA_G_M2, 1064e-9)
RATE_ESA_PER_KWCM2_AT_1064 = cross_section_to_rate_coeff(SIGMA_ESA_M2, 1064e-9)


# ---------------------------------------------------------------------------
# 2. SI parameters from Lee et al.
# ---------------------------------------------------------------------------

W2_R = 162.60
W3_R = 636.01
B32 = 0.144

A_CR_4 = 49.7
A_CR_HIGH = 160.0
A_UC_4 = 6.67
a_UC_HIGH = 25.6
A_INV_4 = 2.35
A_INV_HIGH = 9.00

# Illustrative default resonance parameters for the new wavelength model.
# Both centers default to 1064 nm (the wavelength at which sigma_G_M2 and
# sigma_ESA_M2 were quoted), so pumping at 1064 nm is unaffected by the new
# feature unless the user explicitly changes lambda0/sigma_lambda.
DEFAULT_LAMBDA0_GSA_NM = 1064.0
DEFAULT_LAMBDA0_ESA_NM = 1064.0
DEFAULT_SIGMA_LAMBDA_GSA_NM = 50.0   # illustrative linewidth, not a fitted value
DEFAULT_SIGMA_LAMBDA_ESA_NM = 80.0   # illustrative linewidth, not a fitted value


class SISample(NamedTuple):
    concentration_percent: float
    w2_nonradiative: Optional[float]
    w3_nonradiative: Optional[float]
    threshold_measured_kWcm2: Optional[float]


SI_SAMPLES: Dict[int, SISample] = {
    1: SISample(1.0, None, None, None),
    2: SISample(4.0, 56.9, 103.58, None),
    3: SISample(8.0, 512.0, 1030.0, 23.3),
    4: SISample(8.0, 40.7, 87.3, 6.4),
    5: SISample(8.0, 0.0, 0.0, 4.9),
    6: SISample(20.0, 976.0, 1957.0, 32.8),
    7: SISample(20.0, 585.0, 1176.0, 21.7),
    8: SISample(100.0, 862.0, 1730.0, 29.6),
}


# ---------------------------------------------------------------------------
# 3. Model parameters
# ---------------------------------------------------------------------------

@dataclass
class PAParameters:
    sigma_gsa_m2: float
    sigma_esa_m2: float
    lambda0_gsa_nm: float
    lambda0_esa_nm: float
    sigma_lambda_gsa_nm: float
    sigma_lambda_esa_nm: float
    w2: float
    w3: float
    w2_r: float
    w3_r: float
    w2_nr: float
    w3_nr: float
    b32: float
    s31: float
    q22: float
    q23: float
    concentration_percent: float


def transfer_coefficients(c_percent: float) -> Tuple[float, float, float]:
    """Return (s31, Q22, Q23) using the SI concentration relations.

    CORRECTED: SI Eqns. S5-S6 give
        Q22  = a_u   * c^3 / (c^2 + 4.32)
        Q23  = a_inv * c^3 / (c^2 + 4.32)
    i.e. the additive constant 4.32 is NOT squared (the previous script
    used (c**2 + 4.32**2), which is wrong).

    For c<4% the paper does not provide fitted coefficients; for the 1%
    sample we therefore use the c=4% coefficients only as an explicit
    extrapolation, and label it in output as such.
    """
    if c_percent > 4.0:
        a_cr = A_CR_HIGH
        a_uc = a_UC_HIGH
        a_inv = A_INV_HIGH
    else:
        a_cr = A_CR_4
        a_uc = A_UC_4
        a_inv = A_INV_4

    s31 = a_cr * c_percent ** 2
    q22 = a_uc * c_percent ** 3 / (c_percent ** 2 + 4.32)
    q23 = a_inv * c_percent ** 3 / (c_percent ** 2 + 4.32)
    return s31, q22, q23


def build_parameters(c_percent: float,
                     w2_nr: Optional[float] = None,
                     w3_nr: Optional[float] = None,
                     lambda0_gsa_nm: float = DEFAULT_LAMBDA0_GSA_NM,
                     lambda0_esa_nm: float = DEFAULT_LAMBDA0_ESA_NM,
                     sigma_lambda_gsa_nm: float = DEFAULT_SIGMA_LAMBDA_GSA_NM,
                     sigma_lambda_esa_nm: float = DEFAULT_SIGMA_LAMBDA_ESA_NM,
                     ) -> PAParameters:
    if w2_nr is None:
        w2_nr = 0.0
    if w3_nr is None:
        w3_nr = 0.0

    s31, q22, q23 = transfer_coefficients(c_percent)

    return PAParameters(
        sigma_gsa_m2=SIGMA_G_M2,
        sigma_esa_m2=SIGMA_ESA_M2,
        lambda0_gsa_nm=lambda0_gsa_nm,
        lambda0_esa_nm=lambda0_esa_nm,
        sigma_lambda_gsa_nm=sigma_lambda_gsa_nm,
        sigma_lambda_esa_nm=sigma_lambda_esa_nm,
        w2=W2_R + w2_nr,
        w3=W3_R + w3_nr,
        w2_r=W2_R,
        w3_r=W3_R,
        w2_nr=w2_nr,
        w3_nr=w3_nr,
        b32=B32,
        s31=s31,
        q22=q22,
        q23=q23,
        concentration_percent=c_percent,
    )


def parameters_for_sample(sample_id: int, **wavelength_kwargs) -> PAParameters:
    sample = SI_SAMPLES[sample_id]
    return build_parameters(
        sample.concentration_percent,
        w2_nr=sample.w2_nonradiative,
        w3_nr=sample.w3_nonradiative,
        **wavelength_kwargs,
    )


DEFAULT_PARAMS = parameters_for_sample(3)


# ---------------------------------------------------------------------------
# 4. DREs from SI Eqns. S1-S3 (with wavelength-dependent GSA/ESA rates)
# ---------------------------------------------------------------------------

def pa_rate_equations(y: np.ndarray,
                      I_pump: float,
                      p: PAParameters,
                      pump_wavelength_nm: float = LAMBDA_PUMP_DEFAULT_NM) -> np.ndarray:
    """Return d[N0,N1,N2]/dt for the full SI DRE model at a given pump I and
    pump wavelength. N0/N1/N2 correspond to the paper's n1 (3H6), n2 (3F4),
    n3 (3H4).
    """
    N0, N1, N2 = y

    wavelength_m = pump_wavelength_nm * NM_TO_M
    R_GSA = rate_coefficient(
        p.sigma_gsa_m2, wavelength_m,
        p.lambda0_gsa_nm * NM_TO_M, p.sigma_lambda_gsa_nm * NM_TO_M,
    ) * I_pump
    R_ESA = rate_coefficient(
        p.sigma_esa_m2, wavelength_m,
        p.lambda0_esa_nm * NM_TO_M, p.sigma_lambda_esa_nm * NM_TO_M,
    ) * I_pump

    # Paper n3 equation (emitting 3H4)
    dN2 = (
        R_ESA * N1
        - p.w3 * N2
        - p.s31 * N2 * N0
        + p.q23 * N1 ** 2
    )

    # Paper n2 equation (3F4 intermediate)
    dN1 = (
        R_GSA * N0
        - R_ESA * N1
        - p.w2 * N1
        + (p.b32 * p.w3_r + p.w3_nr) * N2
        + 2.0 * p.s31 * N2 * N0
        - (p.q22 + 2.0 * p.q23) * N1 ** 2
    )

    # Enforce N0 + N1 + N2 = 1 exactly by construction.
    dN0 = -(dN1 + dN2)

    return np.array([dN0, dN1, dN2], dtype=float)


# ---------------------------------------------------------------------------
# 5. RK4 integration
# ---------------------------------------------------------------------------

def rk4_step(f: Callable[[np.ndarray], np.ndarray],
             y: np.ndarray,
             dt: float) -> np.ndarray:
    k1 = f(y)
    k2 = f(y + 0.5 * dt * k1)
    k3 = f(y + 0.5 * dt * k2)
    k4 = f(y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def estimate_stable_dt(I_pump: float,
                       p: PAParameters,
                       pump_wavelength_nm: float = LAMBDA_PUMP_DEFAULT_NM,
                       safety: float = 0.35) -> float:
    """Conservative RK4 step estimate based on the fastest rate in the
    system (GSA, ESA, W2, W3, s31, Q22, Q23). The Q-terms are population
    dependent, so this is intentionally conservative (evaluated as if
    n1 = 1).
    """
    wavelength_m = pump_wavelength_nm * NM_TO_M
    r_gsa = rate_coefficient(p.sigma_gsa_m2, wavelength_m,
                             p.lambda0_gsa_nm * NM_TO_M, p.sigma_lambda_gsa_nm * NM_TO_M) * I_pump
    r_esa = rate_coefficient(p.sigma_esa_m2, wavelength_m,
                             p.lambda0_esa_nm * NM_TO_M, p.sigma_lambda_esa_nm * NM_TO_M) * I_pump

    max_rate = max(
        r_gsa, r_esa, p.w2, p.w3, p.s31, p.q22, p.q23, 1.0,
    )
    return safety / max_rate


# Hard cap on the number of RK4 steps taken in a single integration call.
# This bounds worst-case run time for very stiff parameter sets (e.g. very
# high Tm3+ concentration, where s31 becomes very large) while still
# allowing the early-exit convergence check to terminate much sooner in the
# vast majority of cases.
MAX_RK4_STEPS = 300_000


def rk4_integrate(f: Callable[[np.ndarray], np.ndarray],
                  y0: np.ndarray,
                  dt: float,
                  t_final: float,
                  store_every: int = 50,
                  steady_state_tol: Optional[float] = None,
                  derivative_tol: Optional[float] = None,
                  check_every: int = 500,
                  max_steps: int = MAX_RK4_STEPS) -> Tuple[np.ndarray, np.ndarray]:
    n_steps = min(max(1, int(np.ceil(t_final / dt))), max_steps)
    y = y0.copy()
    t = 0.0

    t_list: List[float] = [0.0]
    y_list: List[np.ndarray] = [y.copy()]

    prev_n2 = y[2]

    for i in range(1, n_steps + 1):
        y = rk4_step(f, y, dt)
        t += dt

        # Small numerical clipping avoids tiny overshoots becoming negative.
        y = np.clip(y, 0.0, None)
        total = y.sum()
        if total > 0.0:
            y /= total

        if i % store_every == 0:
            t_list.append(t)
            y_list.append(y.copy())

        if i % check_every == 0:
            d = f(y)
            population_ok = steady_state_tol is not None and abs(y[2] - prev_n2) < steady_state_tol
            derivative_ok = derivative_tol is not None and np.max(np.abs(d)) < derivative_tol

            if population_ok or derivative_ok:
                break

            prev_n2 = y[2]

    if t_list[-1] != t:
        t_list.append(t)
        y_list.append(y.copy())

    return np.asarray(t_list), np.asarray(y_list)


# ---------------------------------------------------------------------------
# 6. Emission and steady-state functions (steady state via RK4, not root-finding)
# ---------------------------------------------------------------------------

def emission_from_population(n2: float, p: PAParameters) -> float:
    """800-nm radiative emission rate, proportional to detected emission."""
    return (1.0 - p.b32) * p.w3_r * n2


def simulate_time_evolution(I_pump: float,
                            p: PAParameters = DEFAULT_PARAMS,
                            pump_wavelength_nm: float = LAMBDA_PUMP_DEFAULT_NM,
                            dt: Optional[float] = None,
                            t_final: float = 2.0,
                            store_every: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    if dt is None:
        dt = estimate_stable_dt(I_pump, p, pump_wavelength_nm)
    y0 = np.array([1.0, 0.0, 0.0])
    f = lambda y: pa_rate_equations(y, I_pump, p, pump_wavelength_nm)
    return rk4_integrate(f, y0, dt, t_final, store_every=store_every)


def steady_state_solution(I_pump: float,
                          p: PAParameters = DEFAULT_PARAMS,
                          pump_wavelength_nm: float = LAMBDA_PUMP_DEFAULT_NM,
                          dt: Optional[float] = None,
                          derivative_tol: float = 1e-5,
                          max_total_steps: int = 30_000,
                          chunk_steps: int = 10_000,
                          warn_if_unconverged: bool = False) -> np.ndarray:
    """Stationary [N0, N1, N2] obtained purely by RK4 time-integration.

    CORRECTED APPROACH: the previous script solved the stationary DREs
    algebraically with scipy.optimize.root. Lee et al.'s SI Methods state
    that "populations at steady state are derived by solving the
    integrated rate equation with the Runge-Kutta 4th order method" --
    i.e. steady state should come from time integration, not root-finding.

    The integration ALWAYS starts from the physical ground state
    (N0=1, N1=0, N2=0) -- never from a guessed root -- and proceeds in
    chunks of `chunk_steps` RK4 steps, continuing forward in time (not
    restarting) until either the time-derivative of the state falls
    below `derivative_tol`, or `max_total_steps` is reached.

    Note on stiffness: because this is an explicit, fixed-step RK4
    integrator, the step size dt is set by the fastest rate in the
    system (see `estimate_stable_dt`), which for very high Tm3+
    concentrations includes a very large cross-relaxation rate s31.
    For such stiff cases, reaching the true steady state can require a
    very large number of small steps; `max_total_steps` bounds the
    wall-clock cost of a single call at the price of returning a
    not-fully-converged (but still physically reasonable) estimate. Pass
    a larger `max_total_steps` for higher precision at higher Tm3+
    content.
    """
    if dt is None:
        dt = estimate_stable_dt(I_pump, p, pump_wavelength_nm)

    f = lambda y: pa_rate_equations(y, I_pump, p, pump_wavelength_nm)
    y = np.array([1.0, 0.0, 0.0])  # always start from the ground state

    steps_done = 0
    converged = False
    while steps_done < max_total_steps:
        this_chunk = min(chunk_steps, max_total_steps - steps_done)
        t_arr, y_arr = rk4_integrate(
            f, y, dt, t_final=this_chunk * dt,
            store_every=10 ** 9,   # only initial/final state needed
            steady_state_tol=1e-10,
            derivative_tol=derivative_tol,
            check_every=200,
            max_steps=this_chunk,
        )
        y = y_arr[-1]
        steps_done += this_chunk
        d = f(y)
        if np.max(np.abs(d)) < derivative_tol:
            converged = True
            break

    if warn_if_unconverged and not converged:
        print(f"[steady_state_solution] warning: not fully converged after "
              f"{steps_done} RK4 steps (I={I_pump:g} kW/cm^2, "
              f"c={p.concentration_percent:g}%); max|dy/dt|={np.max(np.abs(d)):.2e}. "
              f"Increase max_total_steps for a tighter result.")

    return y


def steady_state_population(I_pump: float,
                            p: PAParameters = DEFAULT_PARAMS,
                            pump_wavelength_nm: float = LAMBDA_PUMP_DEFAULT_NM) -> float:
    """Return stationary N2 using RK4 time-integration from the ground state."""
    return float(steady_state_solution(I_pump, p, pump_wavelength_nm)[2])


def steady_state_emission(I_pump: float,
                          p: PAParameters = DEFAULT_PARAMS,
                          pump_wavelength_nm: float = LAMBDA_PUMP_DEFAULT_NM) -> float:
    n2_ss = steady_state_population(I_pump, p, pump_wavelength_nm)
    return emission_from_population(n2_ss, p)


def power_dependence(I_values: np.ndarray,
                     p: PAParameters = DEFAULT_PARAMS,
                     pump_wavelength_nm: float = LAMBDA_PUMP_DEFAULT_NM) -> np.ndarray:
    return np.array([steady_state_emission(float(I), p, pump_wavelength_nm) for I in I_values])


# ---------------------------------------------------------------------------
# 7. Nonlinearity, threshold (broken-line intersection) and rise time
# ---------------------------------------------------------------------------

def nonlinearity_slope(I_values: np.ndarray,
                       emission: np.ndarray) -> np.ndarray:
    log_I = np.log10(I_values)
    log_E = np.log10(np.clip(emission, 1e-300, None))
    return np.gradient(log_E, log_I)


def pa_threshold_broken_line(I_values: np.ndarray,
                             emission: np.ndarray,
                             pre_points: int = 8,
                             post_points: int = 6) -> Tuple[float, float, float]:
    """Canonical Ith determination: intersection of two straight lines
    fitted to the emission-vs-excitation curve on log-log axes, one
    below and one above the knee -- exactly as described for SI Fig. S3
    ("Threshold values are calculated from changes in slope ... Dotted
    black lines are linear fits of data points below and above
    threshold, whose intersection is the photon avalanche threshold").

    This single routine replaces the previous script's redundant/
    inconsistent `broken_line_threshold`, `threshold_from_curve`, and
    the non-canonical `threshold_from_slope15` criterion.

    Returns (I_threshold, slope_below, slope_above).
    """
    if len(I_values) != len(emission) or len(I_values) < 8:
        raise ValueError("Need at least 8 pump-intensity points.")

    x = np.log10(np.asarray(I_values, dtype=float))
    y = np.log10(np.clip(np.asarray(emission, dtype=float), 1e-300, None))
    slope = np.gradient(y, x)
    peak = int(np.argmax(slope))

    low_start = max(0, peak - pre_points)
    low_end = peak
    high_start = peak + 1
    high_end = min(len(x), high_start + post_points)

    if low_end - low_start < 3 or high_end - high_start < 3:
        raise ValueError("Not enough points on both sides of the knee.")

    m_low, b_low = np.polyfit(x[low_start:low_end], y[low_start:low_end], 1)
    m_high, b_high = np.polyfit(x[high_start:high_end], y[high_start:high_end], 1)

    if np.isclose(m_low, m_high):
        # No resolvable knee; report the slope maximum only as a diagnostic.
        return float(I_values[peak]), float(m_low), float(m_high)

    x_threshold = (b_high - b_low) / (m_low - m_high)
    threshold = 10.0 ** x_threshold

    # Keep the result in the physically relevant neighbourhood of the knee.
    lower = I_values[max(0, peak - 4)]
    upper = I_values[min(len(I_values) - 1, peak + 4)]
    threshold = float(np.clip(threshold, lower, upper))

    return threshold, float(m_low), float(m_high)


def pa_threshold_refined(p: PAParameters,
                         I_coarse: np.ndarray,
                         emission_coarse: Optional[np.ndarray] = None,
                         pump_wavelength_nm: float = LAMBDA_PUMP_DEFAULT_NM,
                         zoom_factor: float = 5.0,
                         n_refine: int = 20,
                         pre_points: int = 7,
                         post_points: int = 4) -> Tuple[float, float, float]:
    """Locate Ith robustly for a simulated (near-ideal, very steep) curve.

    A real experimental "S" curve (SI Fig. S3) is broad enough that many
    data points sit on each straight-line segment flanking the knee. A
    noise-free simulated PA curve can be far steeper, so a coarse pump
    grid often has only one or two points anywhere near the transition,
    with the rest either deep in the (slope~2-7) energy-looping regime or
    already deep in saturation. Fitting straight lines directly to a
    coarse grid then produces a biased, much-too-high "threshold".

    This helper first locates the coarse knee (maximum log-log slope),
    then re-evaluates the emission on a zoomed-in, densely log-spaced
    grid spanning [I_knee/zoom_factor, I_knee*zoom_factor], and applies
    `pa_threshold_broken_line` to that refined curve, which is now well
    resolved on both sides of the transition.
    """
    if emission_coarse is None:
        emission_coarse = power_dependence(I_coarse, p, pump_wavelength_nm)

    slope_coarse = nonlinearity_slope(I_coarse, emission_coarse)
    knee_idx = int(np.argmax(slope_coarse))
    I_knee = float(I_coarse[knee_idx])

    I_refined = np.logspace(
        np.log10(max(I_knee / zoom_factor, I_coarse[0])),
        np.log10(min(I_knee * zoom_factor, I_coarse[-1])),
        n_refine,
    )
    emission_refined = power_dependence(I_refined, p, pump_wavelength_nm)
    return pa_threshold_broken_line(I_refined, emission_refined,
                                    pre_points=pre_points, post_points=post_points)


def rise_time_95(I_pump: float,
                  p: PAParameters = DEFAULT_PARAMS,
                  pump_wavelength_nm: float = LAMBDA_PUMP_DEFAULT_NM,
                  t_final: float = 2.0) -> float:
    """95% rise time of the 800-nm luminescence (SI Fig. S7 definition)."""
    final_n2 = steady_state_population(I_pump, p, pump_wavelength_nm)
    final_value = emission_from_population(final_n2, p)

    if final_value <= 0.0:
        return np.nan

    target = 0.95 * final_value
    t, y = simulate_time_evolution(I_pump, p, pump_wavelength_nm, t_final=t_final, store_every=5)
    emission = emission_from_population(y[:, 2], p)
    indices = np.flatnonzero(emission >= target)
    if len(indices) == 0:
        return np.nan

    idx = int(indices[0])
    if idx == 0:
        return float(t[0])

    e0, e1 = emission[idx - 1], emission[idx]
    t0, t1 = t[idx - 1], t[idx]
    if np.isclose(e1, e0):
        return float(t1)
    return float(t0 + (target - e0) * (t1 - t0) / (e1 - e0))


def delta_av(threshold: float,
             p: PAParameters,
             pump_wavelength_nm: float = LAMBDA_PUMP_DEFAULT_NM) -> float:
    """Emission enhancement Delta_av = E(2 Ith)/E(Ith), cf. Table S9."""
    e1 = steady_state_emission(threshold, p, pump_wavelength_nm)
    e2 = steady_state_emission(2.0 * threshold, p, pump_wavelength_nm)
    if e1 <= 0.0:
        return np.nan
    return e2 / e1


# ---------------------------------------------------------------------------
# 8. Plotting
# ---------------------------------------------------------------------------

def plot_fig2a_style(I_values: np.ndarray,
                     curves: Dict[str, Tuple[np.ndarray, float]],
                     savepath: Optional[str] = None) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    for label, (emission, slope) in curves.items():
        ax.loglog(I_values, emission, "o-", markersize=3.5,
                  label=f"{label},  s = {slope:.1f}")
    ax.set_xlabel(r"Excitation intensity (kW cm$^{-2}$)")
    ax.set_ylabel("800-nm emission intensity (a.u.)")
    ax.set_title("Photon-avalanche power dependence")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200)


def plot_wavelength_detuning(I_values: np.ndarray,
                             curves: Dict[float, np.ndarray],
                             thresholds: Dict[float, float],
                             savepath: Optional[str] = None) -> None:
    """New: emission vs excitation intensity at several pump wavelengths,
    illustrating the effect of detuning from resonance on both the curve
    and the extracted threshold.
    """
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for lam, emission in curves.items():
        ax.loglog(I_values, emission, "o-", markersize=3.5,
                  label=f"{lam:.0f} nm (Ith = {thresholds[lam]:.1f} kW cm$^{{-2}}$)")
    ax.set_xlabel(r"Excitation intensity (kW cm$^{-2}$)")
    ax.set_ylabel("800-nm emission intensity (a.u.)")
    ax.set_title("Effect of pump-wavelength detuning on PA response")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200)


def plot_fig2b_rise_time(I_values: np.ndarray,
                         rise_times_ms: np.ndarray,
                         threshold: float,
                         savepath: Optional[str] = None) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(I_values, rise_times_ms, "o-")
    ax.axvline(threshold, linestyle="--", linewidth=1,
              label=fr"$I_{{th}}$ = {threshold:.1f} kW cm$^{{-2}}$")
    ax.set_xlabel(r"Excitation intensity (kW cm$^{-2}$)")
    ax.set_ylabel("800-nm rise time (ms)")
    ax.set_title("Rise-time signature of photon avalanching")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200)


def plot_fig3a_style(I_values: np.ndarray,
                     sample_curves: Dict[int, np.ndarray],
                     savepath: Optional[str] = None) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 8.0), sharex=True)
    top_ids = [3, 4, 5]
    bottom_ids = [3, 7, 8]

    for sid in top_ids:
        p = parameters_for_sample(sid)
        slope = np.max(nonlinearity_slope(I_values, sample_curves[sid]))
        axes[0].loglog(I_values, sample_curves[sid], "o-", markersize=3.5,
                       label=f"Sample {sid}: {p.concentration_percent:.0f}% Tm, s_max={slope:.1f}")

    for sid in bottom_ids:
        p = parameters_for_sample(sid)
        slope = np.max(nonlinearity_slope(I_values, sample_curves[sid]))
        axes[1].loglog(I_values, sample_curves[sid], "o-", markersize=3.5,
                       label=f"Sample {sid}: {p.concentration_percent:.0f}% Tm, s_max={slope:.1f}")

    axes[0].set_title("8% Tm3+: effect of core/shell architecture")
    axes[1].set_title("Effect of Tm3+ concentration")
    axes[1].set_xlabel(r"Excitation intensity (kW cm$^{-2}$)")
    axes[0].set_ylabel("800-nm emission (a.u.)")
    axes[1].set_ylabel("800-nm emission (a.u.)")
    for ax in axes:
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200)


def fig3b_model_curve(concentration_percent: float,
                      W2_values: np.ndarray,
                      I_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Continuous DRE-model Ith-vs-W2 curve for Fig. 3b, using the
    canonical broken-line threshold everywhere (replacing the previous
    slope>=15 criterion).
    """
    thresholds = []
    for W2_total in np.asarray(W2_values, dtype=float):
        W2_nr = max(0.0, W2_total - W2_R)
        W3_nr = 2.0 * W2_nr
        p = build_parameters(concentration_percent, w2_nr=W2_nr, w3_nr=W3_nr)
        try:
            Ith, _, _ = pa_threshold_refined(p, I_grid)
        except ValueError:
            Ith = np.nan
        thresholds.append(Ith)
    return np.asarray(W2_values, dtype=float), np.asarray(thresholds, dtype=float)


def plot_fig3b_style(model_results: List[Tuple[int, float, float, Optional[float]]],
                     savepath: Optional[str] = None) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    # NOTE: kept deliberately coarse (esp. for the 100% Tm branch, which is
    # numerically stiff -- see `steady_state_solution` docstring) so the
    # demonstration script finishes in a reasonable time. Increase these
    # for a publication-quality reproduction of Fig. 3b.
    W2_grid = np.linspace(W2_R, 1500.0, 3)
    I_grid = np.logspace(np.log10(1.0), np.log10(60.0), 8)

    groups = {8.0: [3, 4, 5], 20.0: [6, 7], 100.0: [8]}

    for concentration, ids in groups.items():
        x_model, y_model = fig3b_model_curve(concentration, W2_grid, I_grid)
        ax.plot(x_model, y_model, linestyle="--", linewidth=1.8,
                label=f"DRE model, {concentration:.0f}% Tm")

    for concentration, ids in groups.items():
        xs, ys = [], []
        for sid, w2, model_thr, measured_thr in model_results:
            if sid in ids and measured_thr is not None:
                xs.append(w2)
                ys.append(measured_thr)
        if xs:
            ax.scatter(xs, ys, s=62, label=f"Experiment, {concentration:.0f}% Tm")

    ax.set_xlabel(r"Relaxation rate, $W_2$ (s$^{-1}$)")
    ax.set_ylabel(r"PA threshold, $I_{th}$ (kW cm$^{-2}$)")
    ax.set_title(r"PA threshold intensity versus $W_2$")
    ax.set_xlim(0, 1500)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200)


# ---------------------------------------------------------------------------
# 9. Main / demonstration
# ---------------------------------------------------------------------------

def main() -> None:
    # Moderate-density pump grid: enough for slope/threshold estimation
    # while keeping the RK4 steady-state calculations fast. Increase for
    # smoother/more accurate curves (at the cost of runtime, especially
    # for the high-Tm3+-content samples -- see `steady_state_solution`).
    I_values = np.logspace(np.log10(1.0), np.log10(150.0), 14)

    print("\n=== Corrected Lee et al. DRE/RK4 model (v2) ===")
    print(f"GSA coefficient @1064nm = {RATE_GSA_PER_KWCM2_AT_1064:.6g} s^-1/(kW cm^-2)")
    print(f"ESA coefficient @1064nm = {RATE_ESA_PER_KWCM2_AT_1064:.6g} s^-1/(kW cm^-2)")
    print(f"ESA/GSA ratio           = {RATE_ESA_PER_KWCM2_AT_1064 / RATE_GSA_PER_KWCM2_AT_1064:.1f}")

    # ---- Sanity check on the Q22/Q23 fix ----
    c_test = 8.0
    s31_new, q22_new, q23_new = transfer_coefficients(c_test)
    q22_old_buggy = a_UC_HIGH * c_test ** 3 / (c_test ** 2 + 4.32 ** 2)
    print(f"\nQ22 fix check at c={c_test:.0f}%: corrected={q22_new:.4f}, "
          f"previous(buggy, squared-4.32)={q22_old_buggy:.4f}")

    # ---------------- Figure 2a style (all at 1064 nm) ----------------
    fig2a_curves: Dict[str, Tuple[np.ndarray, float]] = {}
    for sid, label in [(1, "1% Tm, 1064 nm"), (2, "4% Tm, 1064 nm"), (3, "8% Tm, 1064 nm")]:
        p = parameters_for_sample(sid)
        emission = power_dependence(I_values, p)
        slope = np.max(nonlinearity_slope(I_values, emission))
        fig2a_curves[label] = (emission, slope)
        print(f"Sample {sid}: c={p.concentration_percent:g}% | W2={p.w2:.2f} | "
              f"s31={p.s31:.2f} | Q22={p.q22:.3g} | Q23={p.q23:.3g} | s_max={slope:.2f}")

    plot_fig2a_style(I_values, fig2a_curves, "pa_fig2a_style_v2.png")

    # ---------------- Figure 2b style: threshold + rise time (8% sample) ----------------
    p8 = parameters_for_sample(3)
    threshold8, slope_low8, slope_high8 = pa_threshold_refined(p8, I_values)
    print(f"\n8% sample 3 model threshold (broken-line) = {threshold8:.2f} kW/cm^2")
    print(f"Broken-line slopes: below={slope_low8:.2f}, above={slope_high8:.2f}")
    print(f"Measured threshold (Table S3)              = {SI_SAMPLES[3].threshold_measured_kWcm2:.2f} kW/cm^2")

    rise_I = np.array([0.60, 0.75, 0.90, 1.00, 1.10, 1.25, 1.50]) * threshold8
    rise_times = np.array([rise_time_95(float(I), p8, t_final=1.5) for I in rise_I])
    plot_fig2b_rise_time(rise_I, rise_times * 1e3, threshold8, "pa_fig2b_rise_time_v2.png")

    # ---------------- Figure 3a style ----------------
    sample_ids_fig3 = [3, 4, 5, 6, 7, 8]
    sample_curves: Dict[int, np.ndarray] = {}
    model_results: List[Tuple[int, float, float, Optional[float]]] = []

    for sid in sample_ids_fig3:
        p = parameters_for_sample(sid)
        emission = power_dependence(I_values, p)
        sample_curves[sid] = emission
        try:
            threshold, _, _ = pa_threshold_refined(p, I_values, emission_coarse=emission)
        except ValueError:
            threshold = np.nan
        measured = SI_SAMPLES[sid].threshold_measured_kWcm2
        model_results.append((sid, p.w2, threshold, measured))
        print(
            f"Sample {sid}: c={p.concentration_percent:5.1f}% | W2={p.w2:7.2f} s^-1 | "
            f"model Ith={threshold:7.2f} | measured Ith="
            f"{measured if measured is not None else np.nan:7.2f}"
        )

    plot_fig3a_style(I_values, sample_curves, "pa_fig3a_style_v2.png")
    plot_fig3b_style(model_results, "pa_fig3b_style_v2.png")

    # ---------------- Delta_av vs Table S9 ----------------
    print("\n=== Delta_av = E(2 Ith) / E(Ith), model values (cf. Table S9) ===")
    for sid in sample_ids_fig3:
        p = parameters_for_sample(sid)
        threshold_model = next(r[2] for r in model_results if r[0] == sid)
        if np.isnan(threshold_model):
            print(f"Sample {sid}: threshold not resolved, skipping Delta_av")
            continue
        value = delta_av(threshold_model, p)
        print(f"Sample {sid} ({p.concentration_percent:.0f}% Tm): Delta_av(model) = {value:.3g}")

    # ---------------- NEW: wavelength-detuning demonstration ----------------
    print("\n=== Wavelength-detuning demonstration (8% Tm sample, sigma_lambda "
          f"GSA={DEFAULT_SIGMA_LAMBDA_GSA_NM:.0f} nm, ESA={DEFAULT_SIGMA_LAMBDA_ESA_NM:.0f} nm) ===")
    wavelengths_nm = [1064.0, 1090.0, 1130.0, 1180.0]
    detune_curves: Dict[float, np.ndarray] = {}
    detune_thresholds: Dict[float, float] = {}
    for lam in wavelengths_nm:
        emission_lam = power_dependence(I_values, p8, pump_wavelength_nm=lam)
        detune_curves[lam] = emission_lam
        try:
            Ith_lam, _, _ = pa_threshold_refined(p8, I_values, emission_coarse=emission_lam,
                                                 pump_wavelength_nm=lam)
        except ValueError:
            Ith_lam = np.nan
        detune_thresholds[lam] = Ith_lam
        max_slope = np.max(nonlinearity_slope(I_values, emission_lam))
        print(f"  lambda_pump = {lam:6.1f} nm | Ith = {Ith_lam:7.2f} kW/cm^2 | "
              f"s_max = {max_slope:6.2f} | E(150 kW/cm^2) = {emission_lam[-1]:.3g}")

    plot_wavelength_detuning(I_values, detune_curves, detune_thresholds,
                             "pa_wavelength_detuning_v2.png")

    plt.plot("all")


if __name__ == "__main__":
    main()
