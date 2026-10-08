# -*- coding: utf-8 -*-
"""
photon_avalanche_rk4_corrected_v2.py
=====================================

DRE/RK4 model for the Tm3+-doped NaYF4 photon-avalanche nanoparticles
reported by Lee et al., Nature 589, 230-235 (2021).

This version fixes several issues in the previous script, adds a
wavelength-dependence feature for GSA/ESA, and audits/repairs every plot.

CORRECTIONS (carried over from the previous revision)
-------------------------------------------------------
2. Threshold determination: canonical broken-line log-log intersection
   (`pa_threshold_broken_line`, refined via `pa_threshold_refined`),
   matching SI Fig. S3, used everywhere Ith is needed.
3. Steady-state populations obtained purely by RK4 time-integration from
   the physical ground state (N0=1, N1=N2=0), never by root-finding (see
   `steady_state_solution`), matching the SI Methods description.

WAVELENGTH-DEPENDENT GSA / ESA
-------------------------------
Three physically distinct ingredients are combined, and are kept
conceptually separate throughout the code and comments below:

  (a) Photon-energy (photon-flux) scaling: rate = sigma * I / (h*nu).
      For fixed intensity I (power per area), a longer-wavelength photon
      carries less energy, so more photons per second arrive -- this is
      ordinary photon counting, not a resonance effect. Implemented in
      `cross_section_to_rate_coeff`, unchanged from before.

  (b) Resonance profile: how efficiently an absorption event actually
      happens once a photon is available, as a function of how close the
      photon energy is to the relevant electronic transition. Modelled
      here with an explicitly *illustrative, phenomenological* Gaussian,

          f(lambda) = exp( -(lambda - lambda0)^2 / (2 sigma_lambda^2) ),

      peak-normalized to 1 exactly at lambda0. GSA and ESA get their own,
      independent (lambda0, sigma_lambda) pair -- they are never forced
      to share a resonance center or width (see `PAParameters` /
      `build_parameters`). This is implemented in
      `wavelength_gaussian_factor` and combined with (a) in
      `rate_coefficient`.

  (c) Phonon-assisted / off-resonant absorption: the physical reason a
      real transition has *any* nonzero rate away from perfect resonance
      at all -- lattice phonons make up the energy mismatch between the
      photon and the electronic transition. This is the physical
      mechanism the Gaussian in (b) stands in for; the paper's own
      Supplementary Discussion invokes exactly this (Energy Gap Law) to
      explain why GSA is weak at 1064 nm. As the paper's Methods put it:
      off-resonant excitation requires phonon-assisted energy matching,
      reducing the effective transition probability. We do NOT attempt a
      quantitative multiphonon calculation here -- the Gaussian is a
      qualitative stand-in for this falloff, nothing more.

  IMPORTANT PHYSICAL NUANCE (see item 8 of the request this revision
  responds to): sigma_lambda_gsa_nm and sigma_lambda_esa_nm are
  illustrative model parameters, NOT measured Tm3+ transition
  linewidths. No experimental GSA/ESA linewidth is quoted anywhere in
  Lee et al. or its SI, and none is invented here. Their only role is to
  make the qualitative statement "an off-resonant pump induces a weaker
  upward transition" concrete enough to simulate.

  lambda0_gsa_nm and lambda0_esa_nm, by contrast, ARE grounded in the
  paper wherever the paper gives us the information to do so:

  - ESA (3F4 -> 3F2,3): the Methods state this absorption peak is
    "close to the 1,064-nm excitation wavelength" -- so lambda0_esa_nm
    defaults to 1064.0 nm (treated as resonant).
  - GSA (3H6 -> 3H5): the Methods state a specific, quoted mismatch of
    "~1,200 cm^-1" between the 1064-nm pump and this transition. Given
    lambda0 = hc/DeltaE, that lets us *compute* (not guess) an implied
    GSA resonance wavelength:

        E_pump(1064 nm)   = 1e7 / 1064            [cm^-1]
        E_GSA_transition  = E_pump(1064 nm) + 1200 [cm^-1]  (phonon-assisted
                                                              absorption: the
                                                              photon supplies
                                                              less energy than
                                                              the transition
                                                              needs, and
                                                              lattice phonons
                                                              supply the rest)
        lambda0_gsa_nm    = 1e7 / E_GSA_transition  ~= 943.5 nm

    This value -- NOT 1064 nm -- is used as the GSA resonance default, so
    GSA and ESA genuinely have different resonance centers by default
    (per explicit request: do not silently default them to the same
    value). One consequence, reported as a printed sanity check in
    main(), is that the *effective* GSA rate at the paper's actual
    1064-nm operating point is now somewhat smaller than the raw fitted
    cross-section would give in isolation (since 1064 nm sits ~120 nm off
    the derived GSA resonance) -- which only reinforces the paper's own
    R2/R1 > 10^4 avalanche criterion, since GSA becomes even weaker
    relative to ESA. No transition energy is invented beyond what the
    paper explicitly states.

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

# Illustrative default resonance parameters for the wavelength model.
#
# lambda0_esa_nm: the paper's Methods state the ESA absorption peak is
# "close to the 1,064-nm excitation wavelength" -> treated as resonant.
#
# lambda0_gsa_nm: DERIVED (not guessed) from the paper's explicitly stated
# ~1200 cm^-1 mismatch between the 1064-nm pump and the GSA (3H6-3H5)
# transition (see module docstring for the full derivation):
#     E_pump(1064 nm) + 1200 cm^-1  ->  lambda0_gsa ~= 943.5 nm
# GSA and ESA are therefore NOT given the same default resonance center.
#
# sigma_lambda_gsa_nm / sigma_lambda_esa_nm: phenomenological linewidths.
# No experimental GSA/ESA linewidth is reported anywhere in Lee et al. or
# its SI -- these values are illustrative model parameters only, chosen
# to give a visible but not overwhelming detuning penalty over the
# ~50-150 nm range explored in this script's demonstrations. They must
# NOT be read as measured Tm3+ transition linewidths.
_E_PUMP_1064_CM1 = 1.0e7 / 1064.0          # ~9398.5 cm^-1
_GSA_MISMATCH_CM1 = 1200.0                 # stated explicitly in the paper's Methods
_E_GSA_TRANSITION_CM1 = _E_PUMP_1064_CM1 + _GSA_MISMATCH_CM1

DEFAULT_LAMBDA0_GSA_NM = 1.0e7 / _E_GSA_TRANSITION_CM1   # ~= 943.5 nm
DEFAULT_LAMBDA0_ESA_NM = 1064.0
DEFAULT_SIGMA_LAMBDA_GSA_NM = 120.0   # illustrative linewidth, not a fitted/measured value
DEFAULT_SIGMA_LAMBDA_ESA_NM = 150.0   # illustrative linewidth, not a fitted/measured value


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
    q22 = a_uc * c_percent ** 3 / (c_percent ** 2 + 4.3**2)
    q23 = a_inv * c_percent ** 3 / (c_percent ** 2 + 4.3**2)
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

def resonance_factors(pump_wavelength_nm: float, p: PAParameters) -> Tuple[float, float]:
    """Return (f_GSA, f_ESA): the dimensionless Gaussian resonance factors
    (component (b) in the module docstring) at the given pump wavelength,
    each in [0, 1] and peak-normalized to 1 at its own lambda0. Used both
    by `pa_rate_equations` (via `gsa_esa_rates`) and directly for the
    diagnostic/plotting functions below, so the exact same numbers appear
    in the rate equations and in the plots.
    """
    wavelength_m = pump_wavelength_nm * NM_TO_M
    f_gsa = wavelength_gaussian_factor(wavelength_m, p.lambda0_gsa_nm * NM_TO_M,
                                       p.sigma_lambda_gsa_nm * NM_TO_M)
    f_esa = wavelength_gaussian_factor(wavelength_m, p.lambda0_esa_nm * NM_TO_M,
                                       p.sigma_lambda_esa_nm * NM_TO_M)
    return f_gsa, f_esa


def gsa_esa_rate_coefficients(pump_wavelength_nm: float, p: PAParameters) -> Tuple[float, float]:
    """Return (W_GSA, W_ESA): the full rate coefficients [s^-1/(kW cm^-2)]
    at the given pump wavelength, i.e. photon-energy scaling (a) times the
    resonance factor (b). This is the single place where (a) and (b) are
    combined; `pa_rate_equations`, `estimate_stable_dt`, and every
    diagnostic/plotting function below all go through this function (or
    the equivalent `rate_coefficient` calls it wraps), so there is exactly
    one implementation of "the GSA/ESA rate at wavelength lambda".
    """
    wavelength_m = pump_wavelength_nm * NM_TO_M
    w_gsa = rate_coefficient(p.sigma_gsa_m2, wavelength_m,
                             p.lambda0_gsa_nm * NM_TO_M, p.sigma_lambda_gsa_nm * NM_TO_M)
    w_esa = rate_coefficient(p.sigma_esa_m2, wavelength_m,
                             p.lambda0_esa_nm * NM_TO_M, p.sigma_lambda_esa_nm * NM_TO_M)
    return w_gsa, w_esa


def pa_rate_equations(y: np.ndarray,
                      I_pump: float,
                      p: PAParameters,
                      pump_wavelength_nm: float = LAMBDA_PUMP_DEFAULT_NM) -> np.ndarray:
    """Return d[N0,N1,N2]/dt for the full SI DRE model at a given pump I and
    pump wavelength. N0/N1/N2 correspond to the paper's n1 (3H6), n2 (3F4),
    n3 (3H4).
    """
    N0, N1, N2 = y

    W_gsa, W_esa = gsa_esa_rate_coefficients(pump_wavelength_nm, p)
    R_GSA = W_gsa * I_pump
    R_ESA = W_esa * I_pump

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
    w_gsa, w_esa = gsa_esa_rate_coefficients(pump_wavelength_nm, p)
    r_gsa = w_gsa * I_pump
    r_esa = w_esa * I_pump

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
                            pump_wavelength_nm: float = LAMBDA_PUMP_DEFAULT_NM,
                            max_total_steps: int = 30_000) -> float:
    return float(steady_state_solution(I_pump, p, pump_wavelength_nm,
                                       max_total_steps=max_total_steps)[2])


def steady_state_emission(I_pump: float,
                          p: PAParameters = DEFAULT_PARAMS,
                          pump_wavelength_nm: float = LAMBDA_PUMP_DEFAULT_NM,
                          max_total_steps: int = 30_000) -> float:
    n2_ss = steady_state_population(I_pump, p, pump_wavelength_nm, max_total_steps)
    return emission_from_population(n2_ss, p)


def power_dependence(I_values: np.ndarray,
                     p: PAParameters = DEFAULT_PARAMS,
                     pump_wavelength_nm: float = LAMBDA_PUMP_DEFAULT_NM,
                     max_total_steps: int = 30_000) -> np.ndarray:
    return np.array([steady_state_emission(float(I), p, pump_wavelength_nm, max_total_steps)
                     for I in I_values])

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
                  t_final: float = 5.0,
                  max_t_final: float = 200.0) -> float:
    """95% rise time of the 800-nm luminescence (SI Fig. S7 definition).

    Near the PA threshold, critical slowing down means the 95%-of-steady-
    state point can lie well beyond a short simulation window. Rather than
    silently returning NaN (which corrupts peak-finding on the rise-time
    curve), t_final is doubled until the target is reached or max_t_final
    is hit.
    """
    final_n2 = steady_state_population(I_pump, p, pump_wavelength_nm)
    final_value = emission_from_population(final_n2, p)

    if final_value <= 0.0:
        return np.nan

    target = 0.95 * final_value

    current_t_final = t_final
    while current_t_final <= max_t_final:
        t, y = simulate_time_evolution(I_pump, p, pump_wavelength_nm,
                                        t_final=current_t_final, store_every=1)
        emission = emission_from_population(y[:, 2], p)
        indices = np.flatnonzero(emission >= target)

        if len(indices) > 0:
            idx = int(indices[0])
            if idx == 0:
                return float(t[0])
            e0, e1 = emission[idx - 1], emission[idx]
            t0, t1 = t[idx - 1], t[idx]
            if np.isclose(e1, e0):
                return float(t1)
            return float(t0 + (target - e0) * (t1 - t0) / (e1 - e0))

        current_t_final *= 2.0  # window too short -- extend and retry

    return np.nan  # genuinely didn't converge even at max_t_final


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
# 7b. Wavelength scan (new) and physical sanity checks (new)
# ---------------------------------------------------------------------------

@dataclass
class WavelengthScanResult:
    wavelengths_nm: np.ndarray
    f_gsa: np.ndarray          # dimensionless resonance factor, GSA
    f_esa: np.ndarray          # dimensionless resonance factor, ESA
    w_gsa: np.ndarray          # [s^-1/(kW cm^-2)] full GSA rate coefficient
    w_esa: np.ndarray          # [s^-1/(kW cm^-2)] full ESA rate coefficient
    populations: Dict[float, np.ndarray]   # I_pump -> array of [N0,N1,N2] steady states
    emission: Dict[float, np.ndarray]      # I_pump -> emission array (same order as wavelengths_nm)


def wavelength_dependence(p: PAParameters,
                          I_pump_values: Tuple[float, ...] = (10.0, 30.0, 60.0),
                          wavelengths_nm: Optional[np.ndarray] = None) -> WavelengthScanResult:
    """Scan pump wavelength at one or more fixed pump intensities.

    For every wavelength in `wavelengths_nm`, this computes:
      1. the GSA and ESA resonance factors f(lambda) (component (b));
      2. the corresponding full rate coefficients W_GSA(lambda),
         W_ESA(lambda) (photon-energy scaling x resonance factor);
      3. for each pump intensity in `I_pump_values`, the RK4 steady-state
         populations and the resulting 800-nm emission.

    Default range 950-1250 nm brackets both the derived GSA resonance
    (~943.5 nm, just below the low end -- included so the plot shows the
    approach to resonance) and the paper's actual 1064 nm operating point.
    """
    if wavelengths_nm is None:
        wavelengths_nm = np.linspace(950.0, 1250.0, 41)
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    if np.any(wavelengths_nm <= 0):
        raise ValueError("wavelengths_nm must be strictly positive.")

    f_gsa = np.empty_like(wavelengths_nm)
    f_esa = np.empty_like(wavelengths_nm)
    w_gsa = np.empty_like(wavelengths_nm)
    w_esa = np.empty_like(wavelengths_nm)

    for i, lam in enumerate(wavelengths_nm):
        f_gsa[i], f_esa[i] = resonance_factors(float(lam), p)
        w_gsa[i], w_esa[i] = gsa_esa_rate_coefficients(float(lam), p)

    populations: Dict[float, np.ndarray] = {}
    emission: Dict[float, np.ndarray] = {}
    for I_pump in I_pump_values:
        pops = np.empty((len(wavelengths_nm), 3))
        emis = np.empty(len(wavelengths_nm))
        for i, lam in enumerate(wavelengths_nm):
            y = steady_state_solution(I_pump, p, pump_wavelength_nm=float(lam))
            pops[i] = y
            emis[i] = emission_from_population(y[2], p)
        populations[I_pump] = pops
        emission[I_pump] = emis

    return WavelengthScanResult(wavelengths_nm, f_gsa, f_esa, w_gsa, w_esa, populations, emission)


def sanity_check_wavelength_model(p: PAParameters,
                                  I_pump_probe: float = 30.0,
                                  detune_probe_nm: float = 300.0) -> None:
    """Print a short set of physical sanity checks on the wavelength model
    (per the explicit checklist requested): the Gaussian factor should be
    ~1 exactly at resonance and ~0 far away; GSA/ESA rates must never
    increase with detuning; the same intensity at a strongly detuned
    wavelength must give a lower rate (and, downstream, lower steady-state
    emission) than at resonance; and the rate exactly at lambda0 must
    equal the plain photon-energy-scaled rate (Gaussian factor = 1,
    i.e. no double-counting of the resonance factor at its own center).
    """
    print("\n--- Wavelength-model sanity checks "
          f"(lambda0_gsa={p.lambda0_gsa_nm:.1f} nm, lambda0_esa={p.lambda0_esa_nm:.1f} nm, "
          f"sigma_gsa={p.sigma_lambda_gsa_nm:.0f} nm, sigma_esa={p.sigma_lambda_esa_nm:.0f} nm) ---")

    # 1. Gaussian factor ~= 1 exactly at resonance.
    f_gsa_res, f_esa_res = resonance_factors(p.lambda0_gsa_nm, p)[0], resonance_factors(p.lambda0_esa_nm, p)[1]
    print(f"  f_GSA(lambda0_gsa)  = {f_gsa_res:.6f}  (expect 1.000000)")
    print(f"  f_ESA(lambda0_esa)  = {f_esa_res:.6f}  (expect 1.000000)")

    # 2. Gaussian factor -> 0 far from resonance.
    f_gsa_far, _ = resonance_factors(p.lambda0_gsa_nm + detune_probe_nm, p)
    _, f_esa_far = resonance_factors(p.lambda0_esa_nm + detune_probe_nm, p)
    print(f"  f_GSA(lambda0_gsa + {detune_probe_nm:.0f} nm) = {f_gsa_far:.3e}  (expect << 1)")
    print(f"  f_ESA(lambda0_esa + {detune_probe_nm:.0f} nm) = {f_esa_far:.3e}  (expect << 1)")

    # 3. Rate at lambda0 must equal the plain photon-energy-scaled rate
    #    (no extra suppression right at the resonance center).
    w_gsa_res, _ = gsa_esa_rate_coefficients(p.lambda0_gsa_nm, p)
    w_gsa_plain = cross_section_to_rate_coeff(p.sigma_gsa_m2, p.lambda0_gsa_nm * NM_TO_M)
    _, w_esa_res = gsa_esa_rate_coefficients(p.lambda0_esa_nm, p)
    w_esa_plain = cross_section_to_rate_coeff(p.sigma_esa_m2, p.lambda0_esa_nm * NM_TO_M)
    print(f"  W_GSA(lambda0_gsa) = {w_gsa_res:.6e}  vs. plain photon-scaled = {w_gsa_plain:.6e}  "
          f"(match: {np.isclose(w_gsa_res, w_gsa_plain)})")
    print(f"  W_ESA(lambda0_esa) = {w_esa_res:.6e}  vs. plain photon-scaled = {w_esa_plain:.6e}  "
          f"(match: {np.isclose(w_esa_res, w_esa_plain)})")

    # 4. Rates must be monotonically non-increasing with |detuning|, and a
    #    detuned pump must give a strictly lower rate at fixed intensity.
    detunings = np.array([0.0, 50.0, 100.0, 200.0, 400.0])
    w_gsa_scan = np.array([gsa_esa_rate_coefficients(p.lambda0_gsa_nm + d, p)[0] for d in detunings])
    monotone_ok = bool(np.all(np.diff(w_gsa_scan) <= 1e-30))
    print(f"  W_GSA(lambda0_gsa + d) for d={list(detunings)} nm:")
    print(f"    {np.array2string(w_gsa_scan, precision=4, floatmode='fixed')}")
    print(f"    monotonically non-increasing with detuning: {monotone_ok}")

    # 5. At fixed intensity, steady-state emission at resonance vs. far
    #    detuned should be higher at resonance (weaker pump response off
    #    resonance).
    e_resonant = steady_state_emission(I_pump_probe, p, pump_wavelength_nm=p.lambda0_esa_nm)
    e_detuned = steady_state_emission(I_pump_probe, p,
                                      pump_wavelength_nm=p.lambda0_esa_nm + detune_probe_nm)
    print(f"  E(I={I_pump_probe:g} kW/cm^2) at ESA resonance ({p.lambda0_esa_nm:.0f} nm) = {e_resonant:.4g}")
    print(f"  E(I={I_pump_probe:g} kW/cm^2) detuned by +{detune_probe_nm:.0f} nm       = {e_detuned:.4g}")
    print(f"    emission decreases under strong detuning: {bool(e_detuned <= e_resonant)}")


# ---------------------------------------------------------------------------
# 8. Plotting
# ---------------------------------------------------------------------------

# Small positive floor used only to keep zero/negative values off log axes
# in plots (never used inside the physics -- the DRE integration itself is
# unaffected). Guards against "log-axis problems" (zero values silently
# dropped, or a spurious "no positive values" warning) when the pump grid
# or steady state happens to give an exact zero.
_LOG_PLOT_FLOOR = 1e-300


def _safe_for_log(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), _LOG_PLOT_FLOOR, None)


def plot_fig2a_style(I_values: np.ndarray,
                     curves: Dict[str, Tuple[np.ndarray, float]],
                     savepath: Optional[str] = None) -> None:
    # x = pump intensity (kW/cm^2), y = 800 nm emission (a.u.); both
    # log-scaled because the curves span several decades in each axis and
    # the PA threshold/knee is only visible on a log-log plot (matches SI
    # Fig. S3 style). I_values is a strictly positive logspace grid, and
    # emission is floored above zero before plotting (see _safe_for_log).
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    for label, (emission, slope) in curves.items():
        ax.loglog(I_values, _safe_for_log(emission), "o-", markersize=3.5,
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
    """Emission vs. PUMP INTENSITY, one curve per PUMP WAVELENGTH.

    x = pump intensity (kW/cm^2, log axis -- same reasoning as
        `plot_fig2a_style`); y = 800-nm emission (a.u., log axis, floored
        above zero). Each curve is a different `pump_wavelength_nm` passed
        through `power_dependence` -> `steady_state_emission` ->
        `pa_rate_equations`, so the curves differ *because* GSA and ESA
        are evaluated at different wavelengths for each curve (verified
        below in `main()` via the printed s_max/Ith/emission-at-150-kW
        diagnostics, which are NOT all identical across wavelengths).
    """
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    for lam, emission in curves.items():
        ax.loglog(I_values, _safe_for_log(emission), "o-", markersize=3.5,
                  label=f"{lam:.0f} nm (Ith = {thresholds[lam]:.1f} kW cm$^{{-2}}$)")
    ax.set_xlabel(r"Excitation intensity (kW cm$^{-2}$)")
    ax.set_ylabel("800-nm emission intensity (a.u.)")
    ax.set_title("Effect of pump-wavelength detuning on PA response\n"
                 "(x = pump intensity; each curve = one pump wavelength)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200)


def plot_resonance_factor_vs_wavelength(wavelengths_nm: np.ndarray,
                                        factor: np.ndarray,
                                        label: str,
                                        lambda0_nm: float,
                                        sigma_lambda_nm: float,
                                        color: str,
                                        savepath: Optional[str] = None) -> None:
    """NEW (request item 4D / 6, plots 1 and 2): resonance factor f(lambda)
    vs. pump wavelength for a single transition (GSA or ESA).

    x = pump wavelength [nm], LINEAR axis (a ~300 nm scan does not need a
        log axis, and the Gaussian shape is only readable on a linear
        wavelength axis).
    y = dimensionless resonance factor f(lambda) in [0, 1], LINEAR axis
        (this is a bounded ratio, not a quantity spanning decades).
    """
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(wavelengths_nm, factor, "-", color=color, linewidth=2.0)
    ax.axvline(lambda0_nm, linestyle="--", color="gray", linewidth=1,
              label=fr"$\lambda_0$ = {lambda0_nm:.1f} nm")
    ax.axvspan(lambda0_nm - sigma_lambda_nm, lambda0_nm + sigma_lambda_nm,
              color=color, alpha=0.08, label=fr"$\pm\,\sigma_\lambda$ = {sigma_lambda_nm:.0f} nm")
    ax.set_xlabel("Pump wavelength (nm)")
    ax.set_ylabel(f"{label} resonance factor  f(" + r"$\lambda$)  (dimensionless)")
    ax.set_title(f"{label} resonance profile (illustrative, phenomenological)")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200)


def plot_emission_vs_wavelength(wavelengths_nm: np.ndarray,
                                emission_by_intensity: Dict[float, np.ndarray],
                                savepath: Optional[str] = None) -> None:
    """NEW (request item 4D / 6, plot 3): steady-state 800-nm emission vs.
    PUMP WAVELENGTH, one curve per fixed pump intensity.

    x = pump wavelength [nm], LINEAR axis.
    y = steady-state 800-nm emission (a.u.). A log y-axis is used because,
        near/above the PA threshold, emission can change by many orders of
        magnitude as the wavelength scan crosses in and out of resonance
        (an avalanche "turns on" only within some wavelength window around
        resonance at a given fixed intensity); emission is floored above
        zero before plotting (see _safe_for_log).
    """
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for I_pump, emission in emission_by_intensity.items():
        ax.semilogy(wavelengths_nm, _safe_for_log(emission), "o-", markersize=3.5,
                   label=fr"$I_p$ = {I_pump:.0f} kW cm$^{{-2}}$")
    ax.set_xlabel("Pump wavelength (nm)")
    ax.set_ylabel("Steady-state 800-nm emission (a.u.)")
    ax.set_title("PA emission vs. pump wavelength at fixed excitation intensity")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200)


def plot_fig2b_rise_time(I_values, rise_times_ms, peak_I, savepath=None):
    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    ax.plot(I_values, rise_times_ms, "o-")

    ax.axvline(
        peak_I,
        linestyle="--",
        linewidth=1,
        label=fr"$I_{{peak}}$ = {peak_I:.1f} kW cm$^{{-2}}$"
    )

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
    # x = pump intensity (kW/cm^2, log), y = 800 nm emission (a.u., log,
    # floored above zero) -- same reasoning as plot_fig2a_style.
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 8.0), sharex=True)
    top_ids = [3, 4, 5]
    bottom_ids = [3, 7, 8]

    for sid in top_ids:
        p = parameters_for_sample(sid)
        slope = np.max(nonlinearity_slope(I_values, sample_curves[sid]))
        axes[0].loglog(I_values, _safe_for_log(sample_curves[sid]), "o-", markersize=3.5,
                       label=f"Sample {sid}: {p.concentration_percent:.0f}% Tm, s_max={slope:.1f}")

    for sid in bottom_ids:
        p = parameters_for_sample(sid)
        slope = np.max(nonlinearity_slope(I_values, sample_curves[sid]))
        axes[1].loglog(I_values, _safe_for_log(sample_curves[sid]), "o-", markersize=3.5,
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
    # Stiffness (and thus the steps needed to actually reach steady state)
    # scales with s31 ~ concentration^2-3; give higher-concentration
    # branches a proportionally larger step budget instead of silently
    # returning unconverged (near-zero) populations.
    max_steps = int(30_000 * max(1.0, (concentration_percent / 8.0) ** 2))

    # Threshold rises with W2 and with concentration-driven quenching;
    # make sure the sampled intensity window can actually reach it.
    I_hi = I_grid[-1] * max(1.0, concentration_percent / 20.0)
    I_grid_c = np.logspace(np.log10(I_grid[0]), np.log10(I_hi), len(I_grid))

    thresholds = []
    for W2_total in np.asarray(W2_values, dtype=float):
        W2_nr = max(0.0, W2_total - W2_R)
        W3_nr = 2.0 * W2_nr
        p = build_parameters(concentration_percent, w2_nr=W2_nr, w3_nr=W3_nr)
        try:
            emission_c = power_dependence(I_grid_c, p, max_total_steps=max_steps)
            Ith, _, _ = pa_threshold_refined(p, I_grid_c, emission_coarse=emission_c)
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
    I_values = np.logspace(np.log10(1.0), np.log10(150.0), 100)

    print("\n=== Corrected Lee et al. DRE/RK4 model (v2) ===")
    print(f"GSA coefficient @1064nm (photon-energy scaling only, no resonance factor) "
          f"= {RATE_GSA_PER_KWCM2_AT_1064:.6g} s^-1/(kW cm^-2)")
    print(f"ESA coefficient @1064nm (photon-energy scaling only, no resonance factor) "
          f"= {RATE_ESA_PER_KWCM2_AT_1064:.6g} s^-1/(kW cm^-2)")
    print(f"ESA/GSA ratio (no resonance factor, as in the original SI fit) "
          f"= {RATE_ESA_PER_KWCM2_AT_1064 / RATE_GSA_PER_KWCM2_AT_1064:.1f}")
    print(f"lambda0_gsa (derived from stated ~1200 cm^-1 mismatch) = "
          f"{DEFAULT_LAMBDA0_GSA_NM:.2f} nm")
    print(f"lambda0_esa (paper: ESA peak 'close to' 1064 nm)       = "
          f"{DEFAULT_LAMBDA0_ESA_NM:.2f} nm")

    # ---- Sanity check on the Q22/Q23 fix ----
    c_test = 8.0
    s31_new, q22_new, q23_new = transfer_coefficients(c_test)
    q22_old_buggy = a_UC_HIGH * c_test ** 3 / (c_test ** 2 + 4.32 ** 2)
    print(f"\nQ22 fix check at c={c_test:.0f}%: corrected={q22_new:.4f}, "
          f"previous(buggy, squared-4.32)={q22_old_buggy:.4f}")

    p8 = parameters_for_sample(3)  # 8% Tm sample, used below for several demos

    # ---- Physical sanity checks on the wavelength model (request item 7) ----
    sanity_check_wavelength_model(p8)

    # ---- Effective ESA/GSA ratio AT the paper's actual 1064 nm pump,
    #      now including the resonance factor (GSA sits ~120 nm off its
    #      derived resonance at 1064 nm, so this ratio increases relative
    #      to the "no resonance factor" number above -- see module
    #      docstring for why this is a consistent, not a broken, result). --
    w_gsa_1064, w_esa_1064 = gsa_esa_rate_coefficients(1064.0, p8)
    print(f"\nEffective ESA/GSA ratio AT 1064 nm WITH the resonance factor "
          f"= {w_esa_1064 / w_gsa_1064:.1f}  (still >> 1e4, satisfies the paper's PA criterion)")

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
    threshold8, slope_low8, slope_high8 = pa_threshold_refined(p8, I_values)
    print(f"\n8% sample 3 model threshold (broken-line) = {threshold8:.2f} kW/cm^2")
    print(f"Broken-line slopes: below={slope_low8:.2f}, above={slope_high8:.2f}")
    print(f"Measured threshold (Table S3)              = {SI_SAMPLES[3].threshold_measured_kWcm2:.2f} kW/cm^2")

    rise_I = np.array([
    0.50, 0.65, 0.80, 0.95,
    1.10, 1.25, 1.40, 1.55,
    1.70, 1.85, 2.00, 2.10,
    2.20, 2.30, 2.50, 2.75,
    3.00, 3.50, 4.00, 5.00
    ]) * threshold8
    rise_times = np.array([rise_time_95(float(I), p8, t_final=5.0, max_t_final=200.0) for I in rise_I])

    peak_idx = np.nanargmax(rise_times)
    rise_peak_I = rise_I[peak_idx]
    plot_fig2b_rise_time(rise_I, rise_times * 1e3, rise_peak_I, "pa_fig2b_rise_time_v2.png")

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

    # ---------------- NEW: dedicated wavelength scan (request item 6) ----------------
    # 950-1250 nm brackets the derived GSA resonance (~943.5 nm, just below
    # the scan's low end) and the paper's actual 1064 nm operating point.
    print("\n=== Wavelength scan, 950-1250 nm (8% Tm sample) ===")
    scan_wavelengths = np.linspace(950.0, 1250.0, 31)
    scan = wavelength_dependence(p8, I_pump_values=(20.0, 60.0), wavelengths_nm=scan_wavelengths)

    print(f"  f_GSA peak = {scan.f_gsa.max():.4f} at "
          f"{scan.wavelengths_nm[np.argmax(scan.f_gsa)]:.1f} nm "
          f"(lambda0_gsa = {p8.lambda0_gsa_nm:.1f} nm)")
    print(f"  f_ESA peak = {scan.f_esa.max():.4f} at "
          f"{scan.wavelengths_nm[np.argmax(scan.f_esa)]:.1f} nm "
          f"(lambda0_esa = {p8.lambda0_esa_nm:.1f} nm)")
    for I_pump in (20.0, 60.0):
        emis = scan.emission[I_pump]
        print(f"  I_pump={I_pump:5.1f} kW/cm^2: emission ranges from "
              f"{emis.min():.3g} to {emis.max():.3g} across the scan "
              f"(peak at {scan.wavelengths_nm[np.argmax(emis)]:.1f} nm)")

    # Plot 1: GSA resonance factor vs wavelength.
    plot_resonance_factor_vs_wavelength(
        scan.wavelengths_nm, scan.f_gsa, "GSA", p8.lambda0_gsa_nm, p8.sigma_lambda_gsa_nm,
        color="tab:blue", savepath="pa_resonance_gsa_v2.png",
    )
    # Plot 2: ESA resonance factor vs wavelength.
    plot_resonance_factor_vs_wavelength(
        scan.wavelengths_nm, scan.f_esa, "ESA", p8.lambda0_esa_nm, p8.sigma_lambda_esa_nm,
        color="tab:orange", savepath="pa_resonance_esa_v2.png",
    )
    # Plot 3: steady-state emission vs wavelength at fixed pump intensities.
    plot_emission_vs_wavelength(scan.wavelengths_nm, scan.emission,
                                savepath="pa_emission_vs_wavelength_v2.png")

    plt.plot("all")


if __name__ == "__main__":
    main()