# -*- coding: utf-8 -*-
"""
Tm3+ photon-avalanche wavelength detuning: SI-anchored absorption model.
Inclusief analytische berekening van Ith (Vergelijking 5) vs numerieke Ith*.
"""

from __future__ import annotations

from pathlib import Path
import os  
import contextlib
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.integrate import solve_ivp
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
MIN_AVALANCHE_SLOPE = 10.0
KNEE_EDGE_MARGIN = 3

N_COARSE = 30
N_KNEE = 40
N_POLISH = 40

INCLUDE_EXTRAPOLATED_WAVELENGTHS = False
T_STEADY_S = 20.0     
MAX_STEADY_TIME_S = 1280.0
STEADY_RELATIVE_TOL = 1e-6
STEADY_POPULATION_FLOOR = 1e-18

SLOPE_FRACTION = 0.25  
FIT_MAX_ERROR_DEX = 0.03  
FIT_MIN_R2 = 0.999
FIT_MIN_PEAK_RATIO = 0.95  
FIT_MIN_POINTS = 7
FIT_MIN_SPAN_DEX = 0.5  
LINE_EXTEND_DECADES = 0.0  


def _safe_for_log(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), _LOG_PLOT_FLOOR, None)


# ---------------------------------------------------------------------------
# Stark-level parameters from Table 4 [cm^-1]
# ---------------------------------------------------------------------------
E_3H6_CM1 = np.array([0.0, 9.0, 54.0, 80.0, 122.0, 149.0, 187.0, 223.0, 246.0, 296.0, 314.0, 405.0, 417.0])
E_3F4_CM1 = np.array([5634.0, 5661.0, 5689.0, 5717.0, 5730.0, 5750.0, 5761.0, 5782.0, 5823.0])
E_3H5_CM1 = np.array([8207.0, 8218.0, 8267.0, 8288.0, 8317.0, 8348.0, 8377.0, 8428.0, 8440.0, 8528.0, 8539.0])
E_3H4_CM1 = np.array([12546.0, 12566.0, 12588.0, 12613.0, 12642.0, 12673.0, 12721.0, 12766.0, 12776.0])
E_3F3_CM1 = np.array([14279.0, 14317.0, 14341.0, 14361.0, 14380.0, 14398.0, 14420.0])
E_3F2_CM1 = np.array([14936.0, 14951.0, 15018.0, 15061.0, 15084.0])

STARK_SIGMA_CM1 = 8.5
ESA_WEIGHT_3H4 = 1.0
ESA_WEIGHT_3F3 = 1.0
ESA_WEIGHT_3F2 = 1.0
GSA_WEIGHT_3F4 = 1.0
GSA_WEIGHT_3H5 = 1.0
ESA_BACKGROUND_WEIGHT = 0.00001
GSA_BACKGROUND_WEIGHT = 0.00001
SI_REFERENCE_ANCHOR = True
REFERENCE_WAVELENGTH_NM = 1064.0
ESA_BACKGROUND_LAMBDA0_NM = DEFAULT_LAMBDA0_ESA_NM
ESA_BACKGROUND_SIGMA_NM = DEFAULT_SIGMA_LAMBDA_ESA_NM
GSA_BACKGROUND_LAMBDA0_NM = pa_model.DEFAULT_LAMBDA0_GSA_NM
GSA_BACKGROUND_SIGMA_NM = pa_model.DEFAULT_SIGMA_LAMBDA_GSA_NM


def _stark_gaussian_unnormalized(energy_cm1, levels_cm1, sigma_cm1, amplitudes=None):
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


def transition_energies(upper, lower):
    differences = (upper[:, None] - lower[None, :]).ravel()
    if np.any(differences <= 0):
        raise ValueError("Upper levels must lie above all lower levels.")
    return differences

GSA_TRANSITIONS_3F4_CM1 = transition_energies(E_3F4_CM1, E_3H6_CM1)
GSA_TRANSITIONS_3H5_CM1 = transition_energies(E_3H5_CM1, E_3H6_CM1)
ESA_TRANSITIONS_3H4_CM1 = transition_energies(E_3H4_CM1, E_3F4_CM1)
ESA_TRANSITIONS_3F3_CM1 = transition_energies(E_3F3_CM1, E_3F4_CM1)
ESA_TRANSITIONS_3F2_CM1 = transition_energies(E_3F2_CM1, E_3F4_CM1)

GSA_CHANNELS = {
    "3H6 -> 3F4": (GSA_TRANSITIONS_3F4_CM1, GSA_WEIGHT_3F4),
    "3H6 -> 3H5": (GSA_TRANSITIONS_3H5_CM1, GSA_WEIGHT_3H5),
}
ESA_CHANNELS = {
    "3F4 -> 3H4": (ESA_TRANSITIONS_3H4_CM1, ESA_WEIGHT_3H4),
    "3F4 -> 3F3": (ESA_TRANSITIONS_3F3_CM1, ESA_WEIGHT_3F3),
    "3F4 -> 3F2": (ESA_TRANSITIONS_3F2_CM1, ESA_WEIGHT_3F2),
}


class StarkSpectrum:
    def __init__(self, channels, background_weight, center_nm, width_nm):
        self.channels = channels
        weights = np.array([w for _, w in channels.values()])
        self.weight_sum = weights.sum()
        self.background_weight = background_weight
        self.center_nm, self.width_nm = center_nm, width_nm
        transitions = np.concatenate([e for e, w in channels.values() if w > 0])
        lo = max(1.0, transitions.min() - 10 * STARK_SIGMA_CM1)
        hi = transitions.max() + 10 * STARK_SIGMA_CM1
        energy = np.linspace(lo, hi, max(20001, int((hi-lo)/STARK_SIGMA_CM1*20)+1))
        self.stark_norm = float(np.max(self.raw_stark(energy)))
        wavelength_nm = np.unique(np.concatenate([
            1e7 / energy,
            np.linspace(max(1.0, center_nm - 10*width_nm), center_nm + 10*width_nm, 20001)
        ]))
        self.norm = float(np.max(self.mixed(wavelength_nm * 1e-9)))

    def raw_stark(self, energy):
        value = np.zeros_like(np.asarray(energy, dtype=float))
        for transitions, weight in self.channels.values():
            if weight:
                amps = np.full(len(transitions), 1.0/len(transitions))
                value = value + (weight/self.weight_sum) * _stark_gaussian_unnormalized(
                    energy, transitions, STARK_SIGMA_CM1, amps)
        return value

    def mixed(self, wavelength_m):
        wavelength = np.asarray(wavelength_m, dtype=float)
        stark = self.raw_stark(1e-2/wavelength) / self.stark_norm
        broad = np.exp(-0.5*((wavelength*1e9-self.center_nm)/self.width_nm)**2)
        return (1-self.background_weight)*stark + self.background_weight*broad

    def factor(self, wavelength_m):
        value = self.mixed(wavelength_m) / self.norm
        return float(value) if np.ndim(value) == 0 else value


GSA_SPECTRUM = StarkSpectrum(GSA_CHANNELS, GSA_BACKGROUND_WEIGHT, GSA_BACKGROUND_LAMBDA0_NM, GSA_BACKGROUND_SIGMA_NM)
ESA_SPECTRUM = StarkSpectrum(ESA_CHANNELS, ESA_BACKGROUND_WEIGHT, ESA_BACKGROUND_LAMBDA0_NM, ESA_BACKGROUND_SIGMA_NM)

_GSA_REFERENCE_FACTOR = GSA_SPECTRUM.factor(REFERENCE_WAVELENGTH_NM * 1e-9)
_ESA_REFERENCE_FACTOR = ESA_SPECTRUM.factor(REFERENCE_WAVELENGTH_NM * 1e-9)


def _stark_resonance_factors(pump_wavelength_nm, p):
    wavelength_m = pump_wavelength_nm * 1e-9
    gsa = GSA_SPECTRUM.factor(wavelength_m)
    esa = ESA_SPECTRUM.factor(wavelength_m)
    if SI_REFERENCE_ANCHOR:
        gsa /= _GSA_REFERENCE_FACTOR
        esa /= _ESA_REFERENCE_FACTOR
    return gsa, esa


def _stark_rate_coefficients(pump_wavelength_nm, p):
    wavelength_m = pump_wavelength_nm * 1e-9
    gsa, esa = _stark_resonance_factors(pump_wavelength_nm, p)
    return (pa_model.cross_section_to_rate_coeff(p.sigma_gsa_m2, wavelength_m)*gsa,
            pa_model.cross_section_to_rate_coeff(p.sigma_esa_m2, wavelength_m)*esa)


def _wavelength_cauchy_factor(wavelength_m, lambda0_m, gamma_m):
    return float(1.0/(1.0+((wavelength_m-lambda0_m)/gamma_m)**2))


@contextlib.contextmanager
def _lineshape(kind):
    original_factor = pa_model.wavelength_gaussian_factor
    original_rates = pa_model.gsa_esa_rate_coefficients
    original_resonances = pa_model.resonance_factors
    try:
        if kind == "stark_gaussian":
            pa_model.gsa_esa_rate_coefficients = _stark_rate_coefficients
            pa_model.resonance_factors = _stark_resonance_factors
        elif kind == "cauchy":
            pa_model.wavelength_gaussian_factor = _wavelength_cauchy_factor
        yield
    finally:
        pa_model.wavelength_gaussian_factor = original_factor
        pa_model.gsa_esa_rate_coefficients = original_rates
        pa_model.resonance_factors = original_resonances


def has_avalanche(I_values: np.ndarray, emission: np.ndarray, min_slope: float = MIN_AVALANCHE_SLOPE, edge_margin: int = KNEE_EDGE_MARGIN) -> Tuple[bool, float]:
    emission = np.asarray(emission, dtype=float)
    if not np.all(np.isfinite(emission)) or np.all(emission <= 0.0):
        return False, float("nan")
    slope = nonlinearity_slope(I_values, emission)
    idx = int(np.argmax(slope))
    s_max = float(slope[idx])
    knee_is_interior = edge_margin <= idx < len(I_values) - edge_margin
    return bool(s_max >= min_slope and knee_is_interior), s_max


def fit_avalanche_line(I_values, emission, fraction=SLOPE_FRACTION):
    intensity = np.asarray(I_values, dtype=float)
    emission = np.asarray(emission, dtype=float)
    if (len(intensity) < 3 or np.any(~np.isfinite(intensity)) or np.any(~np.isfinite(emission)) or np.any(intensity <= 0) or np.any(emission <= 0) or np.any(np.diff(intensity) <= 0)):
        return None
    x, y = np.log10(intensity), np.log10(emission)
    slopes = np.diff(y)/np.diff(x)
    k = int(np.argmax(slopes))
    peak = float(slopes[k])
    if peak <= 0: return None
    candidates = np.linspace(0.20, 0.99, 80) if fraction is None else [fraction]
    seen, accepted = set(), []
    for f in candidates:
        lo = hi = k
        while lo > 0 and slopes[lo-1] >= f*peak: lo -= 1
        while hi < len(slopes)-1 and slopes[hi+1] >= f*peak: hi += 1
        if (lo, hi) in seen: continue
        seen.add((lo, hi))
        xw, yw = x[lo:hi+2], y[lo:hi+2]
        if len(xw) < (FIT_MIN_POINTS if fraction is None else 3): continue
        if fraction is None and (k-lo < 2 or hi-k < 2): continue
        xu = np.linspace(xw[0], xw[-1], 200)
        yu = np.interp(xu, xw, yw)
        slope, intercept = np.polyfit(xu, yu, 1)
        residual = yu-(slope*xu+intercept)
        total = np.sum((yu-yu.mean())**2)
        r2 = 1-np.sum(residual**2)/total if total > 0 else 0.0
        max_error = max(np.max(np.abs(residual)), np.max(np.abs(yw-(slope*xw+intercept))))
        span = float(yw[-1]-yw[0])
        fit = {
            "slope": float(slope), "intercept": float(intercept),
            "x_range": (float(xw[0]), float(xw[-1])), "y_range": (float(yw[0]), float(yw[-1])),
            "s_peak": peak, "n_points": len(xw), "fraction": float(f), "r_squared": float(r2),
            "max_error_dex": float(max_error), "span_dex": span, "peak_ratio": float(slope/peak),
            "method": "automatic" if fraction is None else "fixed",
        }
        if fraction is None and (r2 < FIT_MIN_R2 or max_error > FIT_MAX_ERROR_DEX or slope/peak < FIT_MIN_PEAK_RATIO or span < FIT_MIN_SPAN_DEX):
            continue
        accepted.append(fit)
    if not accepted: return None
    return max(accepted, key=lambda f: (f["x_range"][1]-f["x_range"][0], -f["max_error_dex"]))


def pa_threshold_cauchy(p8, I_values, pump_wavelength_nm, emission_coarse):
    s_max, I_at_smax, _n_iter = nonlinearity_slope_max_converged(
        I_values, p8, pump_wavelength_nm=pump_wavelength_nm,
        emission_coarse=emission_coarse, n_refine=25, max_iter=4, rel_tol=2e-2
    )
    return float(I_at_smax), float(s_max)


def _make_rhs(I_pump: float, p, w_gsa: float, w_esa: float):
    R_gsa = w_gsa * I_pump
    R_esa = w_esa * I_pump
    w2, w3 = p.w2, p.w3
    s31, q23 = p.s31, p.q23
    a32 = p.b32 * p.w3_r + p.w3_nr
    q_sum = p.q22 + 2.0 * p.q23

    def rhs(_t, y):
        N0, N1, N2 = y
        dN2 = R_esa * N1 - w3 * N2 - s31 * N2 * N0 + q23 * N1 * N1
        dN1 = R_gsa * N0 - R_esa * N1 - w2 * N1 + a32 * N2 + 2.0 * s31 * N2 * N0 - q_sum * N1 * N1
        return (-(dN1 + dN2), dN1, dN2)
    return rhs


def _check_fast_rhs(p, pump_wavelength_nm: float) -> None:
    w_gsa, w_esa = pa_model.gsa_esa_rate_coefficients(pump_wavelength_nm, p)
    y = np.array([0.7, 0.2, 0.1])
    I_test = 37.0
    mine = np.array(_make_rhs(I_test, p, w_gsa, w_esa)(0.0, y))
    ref = pa_model.pa_rate_equations(y, I_test, p, pump_wavelength_nm)
    if not np.allclose(mine, ref, rtol=1e-10, atol=1e-12):
        raise RuntimeError("Fast right-hand side differs from pa_model.pa_rate_equations.")


def _stationary_population(I_pump, p, w_gsa, w_esa):
    rhs = _make_rhs(float(I_pump), p, w_gsa, w_esa)
    seed_scale = w_gsa * float(I_pump) / max(p.w2 + p.w3, 1.0)
    pop_atol = max(1e-280, min(1e-16, seed_scale * 1e-10))
    atol = np.array([1e-12, pop_atol, pop_atol])
    y = np.array([1.0, 0.0, 0.0])
    elapsed, duration = 0.0, T_STEADY_S
    while elapsed < MAX_STEADY_TIME_S:
        duration = min(duration, MAX_STEADY_TIME_S - elapsed)
        previous = y.copy()
        sol = solve_ivp(rhs, (0.0, duration), y, method="LSODA", rtol=1e-9, atol=atol)
        y = sol.y[:, -1]
        elapsed += duration
        scale = np.maximum(np.abs(y), STEADY_POPULATION_FLOOR)
        change = np.max(np.abs(y - previous)/scale)
        residual = np.max(np.abs(rhs(0.0, y))/scale) * T_STEADY_S
        if change < STEADY_RELATIVE_TOL and residual < STEADY_RELATIVE_TOL:
            return y, elapsed
        duration *= 2.0
    raise RuntimeError(f"Stationarity not reached at I={I_pump:g} kW/cm^2 after {elapsed:g} s.")


def _emission_curve(I_values, p8, pump_wavelength_nm):
    w_gsa, w_esa = pa_model.gsa_esa_rate_coefficients(pump_wavelength_nm, p8)
    emission = np.empty(len(I_values))
    for i, intensity in enumerate(I_values):
        y, _elapsed = _stationary_population(float(intensity), p8, w_gsa, w_esa)
        emission[i] = emission_from_population(max(float(y[2]), 0.0), p8)
    return emission


def adaptive_knee_scan(p8, pump_wavelength_nm: float, I_min: float, I_max: float, n_coarse: int = N_COARSE, n_knee: int = N_KNEE, n_polish: int = N_POLISH):
    I_coarse = np.logspace(np.log10(I_min), np.log10(I_max), n_coarse)
    E_coarse = _emission_curve(I_coarse, p8, pump_wavelength_nm)
    present, s_coarse = has_avalanche(I_coarse, E_coarse, edge_margin=1)
    if not present:
        return I_coarse, E_coarse, np.nan, s_coarse, False, None

    def _zoom(I_grid, E_grid, n_points):
        idx = int(np.argmax(nonlinearity_slope(I_grid, E_grid)))
        lo = I_grid[max(idx - 2, 0)]
        hi = I_grid[min(idx + 2, len(I_grid) - 1)]
        I_new = np.logspace(np.log10(lo), np.log10(hi), n_points)
        E_new = _emission_curve(I_new, p8, pump_wavelength_nm)
        return I_new, E_new

    I_knee_grid, E_knee_grid = _zoom(I_coarse, E_coarse, n_knee)
    I_pol, E_pol = _zoom(I_knee_grid, E_knee_grid, n_polish)
    slope_pol = nonlinearity_slope(I_pol, E_pol)
    k = int(np.argmax(slope_pol))
    Ith_star = float(I_pol[k])

    I_all = np.concatenate([I_coarse, I_knee_grid, I_pol])
    E_all = np.concatenate([E_coarse, E_knee_grid, E_pol])
    order = np.argsort(I_all)
    I_all, E_all = I_all[order], E_all[order]
    keep = np.concatenate([[True], np.diff(np.log(I_all)) > 1e-9])
    I_all, E_all = I_all[keep], E_all[keep]

    fit = fit_avalanche_line(I_all, E_all)
    s_max = fit["slope"] if fit is not None else float("nan")

    return I_all, E_all, Ith_star, s_max, True, fit


def plot_wavelength_detuning(curves: Dict[float, Tuple[np.ndarray, np.ndarray]], thresholds: Dict[float, float], lineshape_label: str, lineshape_kind: str, fit_lines: Optional[Dict[float, Optional[Dict[str, float]]]] = None, savepath: Optional[str] = None) -> None:
    ith_label = "Ith*"
    fig, ax = plt.subplots(figsize=(9.0, 7.0))
    colors = plt.cm.tab20(np.linspace(0, 1, len(curves)))
    I_lo = min(float(I.min()) for I, _E in curves.values())
    I_hi = max(float(I.max()) for I, _E in curves.values())

    for (lam, (I_lam, emission)), color in zip(curves.items(), colors):
        ith = thresholds[lam]
        if np.isfinite(ith):
            label = f"{lam:.0f} nm ({ith_label} = {ith:.1f} " + r"kW cm$^{-2}$)"
            linestyle = "o-"
        else:
            label = f"{lam:.0f} nm (no PA in range)"
            linestyle = "o--"

        ax.loglog(I_lam, _safe_for_log(emission), linestyle, markersize=3.5, color=color, label=label)

    if fit_lines:
        for fit in fit_lines.values():
            if fit is None or fit["slope"] <= 0.0: continue
            s, b = fit["slope"], fit["intercept"]
            y0, y1 = fit["y_range"][0] - LINE_EXTEND_DECADES, fit["y_range"][1] + LINE_EXTEND_DECADES
            x0, x1 = (y0 - b) / s, (y1 - b) / s
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


def plot_smax_vs_wavelength(fit_lines: Dict[float, Optional[Dict[str, float]]], savepath: Optional[str] = None) -> None:
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


def plot_ith_vs_wavelength(thresholds_num: Dict[float, float], thresholds_ana: Dict[float, float], savepath: Optional[str] = None) -> None:
    wavelengths = np.array(sorted(thresholds_num.keys()), dtype=float)
    ith_num_values = np.array([float(thresholds_num[lam]) for lam in wavelengths])
    ith_ana_values = np.array([float(thresholds_ana[lam]) for lam in wavelengths])

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    
    # Numerieke curve (via modelering en knie-detectie)
    ax.plot(wavelengths, ith_num_values, "o-", linewidth=1.6, markersize=5, label="Numeriek ($I_{\mathrm{th}}^{*}$)")
    
    # Analytische curve (vergelijking 5)
    ax.plot(wavelengths, ith_ana_values, "s--", linewidth=1.6, markersize=5, label="Analytisch (Vergelijking 5)")

    ax.set_xlabel("Pump wavelength (nm)")
    ax.set_ylabel(r"$I_{\mathrm{th}}$ (kW cm$^{-2}$)")
    ax.set_title(r"Photon-avalanche threshold $I_{\mathrm{th}}$ vs. pump wavelength")
    
    # Logaritmische schaal toepassen
    ax.set_yscale("log")
    
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=10)
    
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, format="pdf", bbox_inches="tight")
        print(f"Figuur opgeslagen: {savepath}")


def compute_detuning_curves(I_values: np.ndarray, p8, wavelengths_nm, kind: str, label: str):
    curves: Dict[float, Tuple[np.ndarray, np.ndarray]] = {}
    thresholds: Dict[float, float] = {}
    analytical_thresholds: Dict[float, float] = {}  # Analytische opslag
    fit_lines: Dict[float, Optional[Dict[str, float]]] = {}

    print(f"\n=== Wavelength-detuning demonstration (8% Tm sample, {label} line shape) ===")

    with _lineshape(kind):
        if _HAVE_SCIPY and kind in ("gaussian", "stark_gaussian"):
            _check_fast_rhs(p8, float(wavelengths_nm[0]))

        for lam in wavelengths_nm:
            
            # -------------------------------------------------------------
            # Analytische berekening (volgens Vergelijking 5)
            # -------------------------------------------------------------
            w_gsa, w_esa = pa_model.gsa_esa_rate_coefficients(lam, p8)
            a32 = p8.b32 * p8.w3_r + p8.w3_nr
            
            # Voorkom deling door nul als w_esa 0 is
            if w_esa > 0:
                ith_ana = (p8.w2 / w_esa) * (p8.w3 + p8.s31) / (a32 + 2.0 * p8.s31)
            else:
                ith_ana = float("inf")
            
            analytical_thresholds[lam] = ith_ana
            # -------------------------------------------------------------

            if kind in ("gaussian", "stark_gaussian"):
                I_lam, emission_lam, Ith_lam, s_max, present, fit = adaptive_knee_scan(
                    p8, lam, float(I_values[0]), float(I_values[-1]))

                curves[lam] = (I_lam, emission_lam)
                thresholds[lam] = Ith_lam
                fit_lines[lam] = fit

                if not present:
                    print(f"  lambda_pump = {lam:6.1f} nm | Ith*(num) =    n/a   | Ith(ana) = {ith_ana:7.2f} kW/cm^2 | s_max = {s_max:6.2f} | E({I_lam[-1]:.0f} kW/cm^2) = {emission_lam[-1]:.3g}  <-- no avalanche")
                elif fit is None:
                    print(f"  lambda_pump = {lam:.1f} nm | Ith*(num) = {Ith_lam:.2f} | Ith(ana) = {ith_ana:7.2f} kW/cm^2 | PA detected, but no adequately resolved linear fit.")
                else:
                    print(f"  lambda_pump = {lam:6.1f} nm | Ith*(num) = {Ith_lam:7.2f} kW/cm^2 | Ith(ana) = {ith_ana:7.2f} kW/cm^2 | s_fit = {s_max:6.2f} (peak local slope {fit['s_peak']:6.2f}) | E({I_lam[-1]:.0f} kW/cm^2) = {emission_lam[-1]:.3g}")

            else:
                emission_lam = power_dependence(I_values, p8, pump_wavelength_nm=lam)
                curves[lam] = (np.asarray(I_values), emission_lam)
                Ith_lam, s_max_converged = pa_threshold_cauchy(p8, I_values, lam, emission_coarse=emission_lam)

                thresholds[lam] = Ith_lam
                fit_lines[lam] = None

                print(f"  lambda_pump = {lam:6.1f} nm | Ith*(num) = {Ith_lam:7.2f} kW/cm^2 | Ith(ana) = {ith_ana:7.2f} kW/cm^2 | s_max = {s_max_converged:6.2f} | E({I_values[-1]:.0f} kW/cm^2) = {emission_lam[-1]:.3g}")

    return curves, thresholds, analytical_thresholds, fit_lines


def main():
    I_values = np.logspace(np.log10(10.0), np.log10(1000.0), 100)
    p8 = parameters_for_sample(3)

    wavelengths_nm = [850.0, 900.0, 930.0, 960.0, 980.0, 1000.0, 1030.0, 1040.0, 
                      1050.0, 1060.0, 1070.0, 1080.0, 1090.0, 1130.0, 1180.0]

    save_dir = str(Path(__file__).resolve().parent / "results")
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nAlle grafieken worden opgeslagen in: {save_dir}\n")

    (
        stark_curves,
        stark_thresholds,
        stark_ana_thresholds,
        stark_fit_lines
    ) = compute_detuning_curves(
        I_values, p8, wavelengths_nm, kind="stark_gaussian", label="Stark GSA + ESA"
    )

    plot_wavelength_detuning(
        stark_curves, stark_thresholds,
        lineshape_label="Stark GSA + ESA", lineshape_kind="stark_gaussian",
        fit_lines=stark_fit_lines, savepath=os.path.join(save_dir, "pa_wavelength_detuning.pdf")
    )

    plot_smax_vs_wavelength(
        stark_fit_lines, savepath=os.path.join(save_dir, "pa_smax_vs_wavelength.pdf")
    )

    # Deze functie is uitgebreid om zowel de numerieke als de analytische waarden te tonen
    plot_ith_vs_wavelength(
        stark_thresholds, stark_ana_thresholds, 
        savepath=os.path.join(save_dir, "pa_ith_vs_wavelength.pdf")
    )

    plt.show(block=False)
    plt.pause(0.1)


if __name__ == "__main__":
    main()