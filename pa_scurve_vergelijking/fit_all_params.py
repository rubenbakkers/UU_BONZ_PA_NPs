# -*- coding: utf-8 -*-
"""Fit of the measured S-curves with ALL model parameters free, not only k.

make_figures.py fits only the intensity scale k (one common factor on sigma_GSA
and sigma_ESA) and keeps Lee's values for everything else; scan_shape.py only
tries a coarse grid of a few parameter sets. Here all shape parameters of the
rate equations are fitted at the same time (scipy.optimize.least_squares):

  counts = A * M(k * I_reported; theta) + b_FW   (M: emission summed over the spot)

  per curve : k (intensity scale), A and one background per filter-wheel
              setting (A and b_FW are linear and solved exactly at every step)
  theta     : multipliers on Lee SI sample 5 (8 % Tm, no non-radiative loss):
                sigma_gsa  GSA cross section (the GSA/ESA ratio)
                w2         3F4 decay rate
                s31        cross-relaxation 3H4 + 3H6 -> 2 x 3F4
                q22, q23   energy-transfer upconversion terms
                b32        3H4 -> 3F4 branching ratio (sets a32)
  fixed     : w3 (3H4 decay) and sigma_ESA. The steady state does not change
              when all rates and k are scaled by the same factor, and k scales
              sigma_GSA and sigma_ESA together, so one rate and one cross
              section must stay fixed; otherwise the fit is degenerate. The
              fitted multipliers are therefore relative to w3 and sigma_ESA.
              (w3 is in PARAMS, but not in FREE_PARAMS: it may only be freed
              when another rate, e.g. s31/Q22/Q23 through a known Tm
              concentration, stays fixed; see fit_fixed_concentration.py.)

Three fits, over all measurements in measurements.py:
  "k only"   : theta = Lee (as make_figures.py), for comparison;
  "joint"    : per sample (field "sample" in measurements.py), the same
               particles in all its curves: theta shared, k, A and b_FW per
               curve (k differs per laser: calibration). Only for samples
               with two or more measurements;
  "separate" : each curve with its own theta.
Multipliers are fitted in log10 within the bounds below, from Lee's values and
from N_STARTS random starting points. The 1-sigma uncertainty comes from the
Jacobian. A multiplier with an uncertainty larger than a factor 10, or at a
bound, is not determined by these data (open marker in fig. 5).

The steady state uses pa_fit_lib.emission_curve_fast (algebraic, ~250x faster
than the LSODA routine and identical to ~1e-13); the best parameters are
checked against LSODA at the end.

Output: results/fig4_all_params_fit.png, results/fig5_all_params_values.png,
results/all_params_fit.json. Run time a few minutes.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # make the scripts in this folder importable
import json
import time
from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from pa_fit_lib import (out_path, pm, lee_params, rates, analytical_ith, emission_curve_fast,
                        check_fast_solver, ensemble_curve, load, local_slope_binned, measurements,
                        describe, format_parameters, MODEL_INFO)
from fit_scurves import XB, WB, REL_SCATTER, linear_fit
from plotting import BLUE, ORANGE, GREY, INK, PALETTE, apply_style, fw_markers, scurve_axes

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
# "conc" scales the Tm concentration (8 % x multiplier); it sets s31, Q22 and Q23 through
# Lee's relations (pa_model.transfer_coefficients), on top of their own multipliers.
PARAMS = ("conc", "sigma_gsa", "w2", "w3", "s31", "q22", "q23", "b32")
FREE_PARAMS = ("sigma_gsa", "w2", "s31", "q22", "q23", "b32")   # e.g. ("sigma_gsa", "w2") to free only these;
                                                                 # never w3 or conc together with s31, q22 and q23
LABELS = {"conc": r"Tm concentration $c$", "sigma_gsa": r"$\sigma_{GSA}$ (GSA/ESA)",
          "w2": r"$w_2$ ($^3$F$_4$ decay)", "w3": r"$w_3$ ($^3$H$_4$ decay)",
          "s31": r"$s_{31}$ (cross-relaxation)", "q22": r"$Q_{22}$ (ETU)",
          "q23": r"$Q_{23}$ (ETU)", "b32": r"$b_{32}$ (branching)"}
P0 = pm.build_parameters(8.0, 0.0, 0.0, lambda0_gsa_nm=lee_params(3).lambda0_gsa_nm)  # Lee SI sample 5
BOUNDS_LOG10 = {name: (-2.0, 2.0) for name in PARAMS}         # factor 0.01 - 100
BOUNDS_LOG10["b32"] = (-2.0, float(np.log10(0.95 / P0.b32)))   # keep b32 < 0.95
# A fitted concentration stays above 4 %: there Lee's relations switch to the
# coefficients of the 4 % sample (a jump), and below 4 % they are an extrapolation.
CONC_FIT_RANGE_PERCENT = (4.01, 100.0)
BOUNDS_LOG10["conc"] = tuple(float(np.log10(c / P0.concentration_percent)) for c in CONC_FIT_RANGE_PERCENT)
LOG10_K_BOUNDS = (-2.0, 3.0)
N_STARTS = 8                  # random starting points (log10 multipliers in [-1, 1]) besides Lee
START_RANGE_LOG10 = {"conc": BOUNDS_LOG10["conc"]}   # other ranges for the random starts (default [-1, 1])
SEED = 1
DETERMINED_LOG10 = 0.5        # "determined": 1-sigma and spread over equivalent starts < factor 3
I_FIT = np.logspace(-4, 5, 450)   # single-NP model grid [kW/cm^2]; wide because k is free

apply_style(10)

DATA = measurements()
data = {n: load(m["file"]) for n, m in DATA.items()}
NAMES = list(DATA)
TITLES = {n: m["label"] for n, m in DATA.items()}
# "joint" fits: the measurements of one sample share the particle parameters.
GROUPS = {}
for _name, _m in DATA.items():
    GROUPS.setdefault(_m["sample"], []).append(_name)
JOINT = {sample: names for sample, names in GROUPS.items() if len(names) >= 2}
SIG = {n: np.sqrt(d["noise_cps"] ** 2 + (REL_SCATTER * np.clip(d["net_cps"], 0, None)) ** 2)
       for n, d in data.items()}
# GSA/ESA rate coefficients of Lee's parameters; they scale linearly with the
# cross sections, so sigma_gsa only multiplies w_gsa.
BASE_RATES = {n: rates(DATA[n]["wavelength_nm"], P0, DATA[n]["fwhm_nm"]) for n in NAMES}


def shared_label(sample):
    """'shared by both curves' (+ 'of <sample>' when there are several joint fits)."""
    n = len(JOINT[sample])
    text = "shared by both curves" if n == 2 else f"shared by all {n} curves"
    return text + (f" of {sample}" if len(JOINT) > 1 and sample else "")


# ---------------------------------------------------------------------------
# Model for given multipliers
# ---------------------------------------------------------------------------

def full_log_m(free_log_m, free, fixed=None):
    """log10 multipliers of all PARAMS from the free ones; the others are 0 (= Lee)
    or the value in `fixed` (dict name -> log10 multiplier)."""
    out = dict.fromkeys(PARAMS, 0.0)
    out.update(fixed or {})
    out.update(zip(free, free_log_m))
    return np.array([out[p] for p in PARAMS])


def model_params(log_m, name):
    """(PAParameters, w_gsa, w_esa) for log10 multipliers log_m (order PARAMS)."""
    m = dict(zip(PARAMS, 10.0 ** np.asarray(log_m, dtype=float)))
    c = P0.concentration_percent * m["conc"]
    s31, q22, q23 = (P0.s31, P0.q22, P0.q23) if m["conc"] == 1.0 else pm.transfer_coefficients(c)
    p = replace(P0, concentration_percent=c, w2=P0.w2 * m["w2"], w2_r=P0.w2_r * m["w2"],
                w3=P0.w3 * m["w3"], w3_r=P0.w3_r * m["w3"], s31=s31 * m["s31"],
                q22=q22 * m["q22"], q23=q23 * m["q23"], b32=P0.b32 * m["b32"])
    wg, we = BASE_RATES[name]
    return p, wg * m["sigma_gsa"], we


def effective_params(log_m, name):
    """As model_params, but with the sigma_gsa multiplier also in p.sigma_gsa_m2
    (model_params applies it to w_gsa only); for printing."""
    p, wg, we = model_params(log_m, name)
    m_gsa = 10.0 ** dict(zip(PARAMS, np.asarray(log_m, dtype=float)))["sigma_gsa"]
    return replace(p, sigma_gsa_m2=P0.sigma_gsa_m2 * m_gsa), wg, we


_cache = {}


def single_np_curve(name, log_m):
    key = (name, tuple(float(v) for v in log_m))
    if key not in _cache:
        if len(_cache) > 4000:
            _cache.clear()
        p, wg, we = model_params(log_m, name)
        _cache[key] = emission_curve_fast(I_FIT, p, wg, we)
    return _cache[key]


def curve_model(name, log_k, log_m, I=None):
    """Spot-summed model M at k * I (default: the measured intensities)."""
    I = data[name]["I_kWcm2"] if I is None else I
    return ensemble_curve(10.0 ** log_k * np.asarray(I), I_FIT, single_np_curve(name, log_m), XB, WB)


def curve_fit_given(name, log_k, log_m):
    """Linear part (A, b_FW) for fixed k and theta; returns (coef, model, chi2)."""
    d = data[name]
    M = curve_model(name, log_k, log_m)
    scale = M.max() if M.max() > 0 else 1.0           # well-conditioned least squares
    coef, model, chi2 = linear_fit(M / scale, d["net_cps"], SIG[name], d["fw6"])
    coef = np.array(coef, dtype=float)
    coef[0] /= scale
    return coef, model, chi2


def best_log_k(name, log_m):
    """Starting value for k: grid scan with theta fixed (only A > 0)."""
    grid = np.linspace(LOG10_K_BOUNDS[0], LOG10_K_BOUNDS[1], 251)
    best = (np.inf, 0.0)
    for lk in grid:
        coef, _model, chi2 = curve_fit_given(name, lk, log_m)
        if coef[0] > 0 and chi2 < best[0]:
            best = (chi2, lk)
    return best[1]


# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

def _start_value(p, u):
    """Random start of parameter p from u in [-1, 1]: u itself, or mapped onto START_RANGE_LOG10[p]."""
    if p not in START_RANGE_LOG10:
        return u
    a, b = START_RANGE_LOG10[p]
    return a + (b - a) * (u + 1.0) / 2.0


def run_fit(names, free, n_starts=N_STARTS, seed=SEED, label="", fixed=None):
    """least_squares over x = [log10 k per curve] + [log10 multiplier per free param].
    `fixed`: log10 multipliers of parameters that are not free and not Lee's (e.g. a known
    concentration: {"conc": log10(c / 8 %)})."""
    names, free = list(names), list(free)
    n_k = len(names)
    lo = np.array([LOG10_K_BOUNDS[0]] * n_k + [BOUNDS_LOG10[p][0] for p in free])
    hi = np.array([LOG10_K_BOUNDS[1]] * n_k + [BOUNDS_LOG10[p][1] for p in free])

    def resid(x):
        log_m = full_log_m(x[n_k:], free, fixed)
        out = []
        for i, name in enumerate(names):
            _coef, model, _chi2 = curve_fit_given(name, x[i], log_m)
            out.append((data[name]["net_cps"] - model) / SIG[name])
        return np.concatenate(out)

    rng = np.random.default_rng(seed)
    starts = [np.zeros(len(free))]
    if free:
        starts += [np.array([_start_value(p, u) for p, u in zip(free, rng.uniform(-1.0, 1.0, len(free)))])
                   for _ in range(n_starts)]
    runs = []
    for s, theta0 in enumerate(starts):
        theta0 = np.clip(theta0, lo[n_k:] + 1e-6, hi[n_k:] - 1e-6)
        log_m0 = full_log_m(theta0, free, fixed)
        x0 = np.concatenate([[best_log_k(n, log_m0) for n in names], theta0])
        x0 = np.clip(x0, lo + 1e-6, hi - 1e-6)
        t0 = time.perf_counter()
        res = least_squares(resid, x0, bounds=(lo, hi), method="trf", diff_step=1e-3,
                            x_scale=1.0, max_nfev=300)
        runs.append(res)
        print(f"  {label} start {s}: chi2 = {2 * res.cost:8.2f}  ({res.nfev} evaluations, "
              f"{time.perf_counter() - t0:.0f} s)", flush=True)
    best = min(runs, key=lambda r: r.cost)
    return summarize(best, names, free, runs, fixed)


def summarize(res, names, free, runs, fixed=None):
    fixed = dict(fixed or {})
    n_k = len(names)
    x = res.x
    all_chi2 = [2.0 * r.cost for r in runs]
    log_m = full_log_m(x[n_k:], free, fixed)
    n_points = sum(len(data[n]) for n in names)
    n_linear = sum(1 + len(np.unique(data[n]["fw6"])) for n in names)
    n_par = len(x) + n_linear
    chi2 = 2.0 * res.cost
    dof = n_points - n_par
    # 1-sigma from the Jacobian (scaled by chi2_red when that is > 1)
    J = res.jac
    cov = np.linalg.pinv(J.T @ J) * max(1.0, chi2 / dof)
    err = np.minimum(np.sqrt(np.clip(np.diag(cov), 0, None)), 99.0)  # log10; 99 = unbounded
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = cov / np.outer(err, err)
    lo = [LOG10_K_BOUNDS[0]] * n_k + [BOUNDS_LOG10[p][0] for p in free]
    hi = [LOG10_K_BOUNDS[1]] * n_k + [BOUNDS_LOG10[p][1] for p in free]
    at_bound = [(xi - l < 0.02) or (h - xi < 0.02) for xi, l, h in zip(x, lo, hi)]
    # The Jacobian only sees the curvature at the minimum. Starting points that
    # end (almost) as well, within delta chi2 <= max(1, chi2_red), show how far
    # a parameter can move along a flat valley; their spread is a second check.
    tol = max(1.0, chi2 / dof)
    equivalent = np.array([r.x for r in runs if 2.0 * r.cost <= chi2 + tol])
    spread = equivalent.max(axis=0) - equivalent.min(axis=0)

    out = dict(curves=names, free_params=free, fixed_log10=fixed, chi2=chi2, n_points=n_points, n_params=n_par,
               chi2_red=chi2 / dof, BIC=chi2 + n_par * np.log(n_points),
               chi2_of_all_starts=sorted(all_chi2), n_equivalent_starts=len(equivalent),
               per_curve={}, multipliers={},
               correlation_matrix=dict(order=[f"k_{n}" for n in names] + free,
                                       values=np.round(corr, 3).tolist()))
    for j, p in enumerate(free):
        i = n_k + j
        out["multipliers"][p] = dict(
            value=10.0 ** x[i], log10=x[i], log10_err=err[i], factor_err=10.0 ** err[i],
            range_equivalent_starts=[10.0 ** equivalent[:, i].min(), 10.0 ** equivalent[:, i].max()],
            at_bound=at_bound[i],
            determined=bool(err[i] < DETERMINED_LOG10 and spread[i] < DETERMINED_LOG10
                            and not at_bound[i]))
    for p in PARAMS:
        if p not in free:
            v = fixed.get(p, 0.0)
            out["multipliers"][p] = dict(value=10.0 ** v, log10=v, fixed=True)
    for i, name in enumerate(names):
        coef, model, c2 = curve_fit_given(name, x[i], log_m)
        p, wg, we = model_params(log_m, name)
        ith = analytical_ith(p, wg, we)
        n_lin = 1 + len(np.unique(data[name]["fw6"]))
        fws = np.unique(data[name]["fw6"])
        out["per_curve"][name] = dict(
            k=10.0 ** x[i], k_factor_err=10.0 ** err[i],
            k_range_equivalent_starts=[10.0 ** equivalent[:, i].min(), 10.0 ** equivalent[:, i].max()],
            A=coef[0],
            background_per_FW={int(f): float(b) for f, b in zip(fws, coef[1:])},
            chi2=c2, chi2_red=c2 / (len(data[name]) - n_lin - 1 - len(free) / len(names)),
            Ith_model_kWcm2=ith, Ith_on_reported_scale=ith / 10.0 ** x[i],
            parameters=dict(concentration_percent=p.concentration_percent,
                            sigma_gsa_m2=P0.sigma_gsa_m2 * 10.0 ** log_m[PARAMS.index("sigma_gsa")],
                            sigma_esa_m2=P0.sigma_esa_m2,
                            w2=p.w2, w3=p.w3, s31=p.s31, q22=p.q22, q23=p.q23, b32=p.b32,
                            a32=p.b32 * p.w3_r + p.w3_nr),
            max_rel_diff_fast_vs_lsoda=check_fast_solver(np.logspace(-2, 3, 50), p, wg, we))
    out["_x"] = x.tolist()
    out["_log_m"] = log_m.tolist()
    return out


def print_fit(title, fit):
    print(f"\n=== {title}: chi2 = {fit['chi2']:.1f}, chi2_red = {fit['chi2_red']:.3f}, "
          f"BIC = {fit['BIC']:.1f} ({fit['n_params']} parameters, {fit['n_points']} points) ===")
    for name, c in fit["per_curve"].items():
        k_lo, k_hi = c["k_range_equivalent_starts"]
        print(f"  {name:12s} k = {c['k']:6.2f} (equivalent starts {k_lo:.3g}-{k_hi:.3g})  "
              f"chi2_red = {c['chi2_red']:.3f}  "
              f"Ith(model) = {c['Ith_model_kWcm2']:.3g}  Ith/k = {c['Ith_on_reported_scale']:.3f} kW/cm^2  "
              f"(fast vs LSODA: {c['max_rel_diff_fast_vs_lsoda']:.1e})")
    for p in PARAMS:
        m = fit["multipliers"][p]
        conc = (f"  -> c = {m['value'] * P0.concentration_percent:.3g} %" if p == "conc" else "")
        if m.get("fixed"):
            if m["log10"] != 0.0:
                print(f"  {p:10s} x {m['value']:8.3g}  (fixed){conc}")
            continue
        flag = "" if m["determined"] else ("  <- at bound" if m["at_bound"] else "  <- not determined")
        r_lo, r_hi = m["range_equivalent_starts"]
        print(f"  {p:10s} x {m['value']:8.3g}  (1 sigma x/÷ {m['factor_err']:.3g}; "
              f"equivalent starts {r_lo:.3g}-{r_hi:.3g}){flag}{conc}")
    if len(fit["chi2_of_all_starts"]) > 1:
        print("  chi2 of all starts: " + ", ".join(f"{c:.1f}" for c in fit["chi2_of_all_starts"])
              + f"  ({fit['n_equivalent_starts']} within delta chi2 <= max(1, chi2_red))")
    # The parameters of this best fit in absolute units.
    log_m = np.array(fit["_log_m"])
    p, _wg, _we = effective_params(log_m, fit["curves"][0])
    print("  parameters of this fit" + ("" if fit["free_params"] or fit.get("fixed_log10")
                                        else " (Lee SI sample 5)") + ":")
    print(format_parameters(p))
    for name in fit["curves"]:
        _p, wg, we = effective_params(log_m, name)
        c = fit["per_curve"][name]
        print(f"    {name}: w_GSA = {wg:.4g}, w_ESA = {we:.4g} s^-1 per kW cm^-2; k = {c['k']:.4g}, "
              f"A = {c['A']:.4g}, background "
              + ", ".join(f"FW6={f}: {b:.4g}" for f, b in c["background_per_FW"].items()) + " cps")


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_fits(fits, what="all parameters", fname="fig4_all_params_fit.png",
              title="S-curves fitted with all model parameters free"):
    """Fig 4: data with the three fits, and the local slope (one column per measurement).
    `what` names the fitted parameters in the legend: one text, or a dict sample -> text
    (fit_concentration.py reuses this function)."""
    fig, panels = scurve_axes(len(NAMES))
    for name, (ax, ax2) in zip(NAMES, panels):
        d = data[name]
        Ir = d["I_kWcm2"]
        fws = np.unique(d["fw6"])
        sample = DATA[name]["sample"]
        joint = fits["joint"].get(sample)
        what_here = what[sample] if isinstance(what, dict) else what
        styles = [("k only, Lee parameters", fits["k_only"], dict(color=GREY, ls=":", lw=2.0))]
        if joint is not None:
            styles.append((f"{what_here}, {shared_label(sample)}", joint, dict(color=BLUE)))
        styles.append((f"{what_here}, per curve", fits["separate"][name],
                       dict(color=ORANGE, ls="--", lw=1.6)))
        bg = (joint or fits["separate"][name])["per_curve"][name]["background_per_FW"]
        ycorr = d["net_cps"] - np.array([bg[int(f)] for f in d["fw6"]])
        marks = fw_markers(fws)
        for f in fws:
            m = d["fw6"] == f
            good = m & (ycorr > 3 * d["noise_cps"])
            weak = m & (ycorr > 0) & ~good
            ax.plot(Ir[good], ycorr[good], marks[f], ms=5, mfc="none", mec=INK, mew=1,
                    label=f"measured FW6={int(f)}")
            ax.plot(Ir[weak], ycorr[weak], marks[f], ms=4, mfc="none", mec="#c8c8c8", mew=1)
        xs, sl = local_slope_binned(Ir, ycorr, d["noise_cps"], width_dex=0.07, min_snr=4)
        ax2.plot(xs, sl, "o", ms=4, mfc="none", mec=INK, mew=1, label="measured (local fit, 0.07 dex)")

        x_lo, x_hi = Ir.min() / 1.15, Ir.max() * 1.15        # the model lines run to the plot edges
        Iplot = np.logspace(np.log10(x_lo), np.log10(x_hi), 400)
        for label, fit, style in styles:
            i = fit["curves"].index(name)
            log_k, log_m = fit["_x"][i], np.array(fit["_log_m"])
            c = fit["per_curve"][name]
            M = curve_model(name, log_k, log_m, Iplot)
            ax.plot(Iplot, c["A"] * M, label=f"{label} ($\\chi^2_r$={c['chi2_red']:.2f})", **style)
            slope = np.gradient(np.log(np.clip(M, 1e-300, None)), np.log(Iplot))
            ax2.plot(Iplot, slope, label=label, **style)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_ylim(0.8, max(ycorr.max(), 30) * 2.5)
        ax.set_title(TITLES[name], color=INK, loc="left", fontsize=11)
        ax.set_ylabel("emission − background [cps]")
        ax.legend(fontsize=7.5, loc="upper left")
        ax2.axhline(1, color="#dddddd", lw=1, zorder=0)
        ax2.set_ylim(0, 35)
        ax2.set_xscale("log")
        ax2.set_xlim(x_lo, x_hi)                               # shared with the S-curve panel
        ax2.set_xlabel("reported intensity [kW cm$^{-2}$]")
        ax2.set_ylabel("slope d log E / d log I")
        ax2.legend(fontsize=7.5, loc="upper left")
    where = describe(DATA)
    fig.suptitle((f"{where}: " if where else "") + f"{title} "
                 + ("(background of the shared fit subtracted)" if JOINT
                    else "(background of the per-curve fit subtracted)"),
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path(fname), dpi=170)


def plot_values(fits, free_params=None, fname="fig5_all_params_values.png",
                fixed_text="$w_3$ and $\\sigma_{ESA}$"):
    """Fig 5: fitted multipliers relative to Lee SI sample 5 (open = not determined)."""
    rows = ([(shared_label(s), fits["joint"][s]) for s in JOINT]
            + [(f"only {TITLES[n]}", fits["separate"][n]) for n in NAMES])
    n_rows = len(rows)
    step = min(0.22, 0.8 / n_rows)                     # vertical offset between the rows
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    free = [p for p in PARAMS if p in (free_params or FREE_PARAMS)]
    any_given = False
    for r, (label, fit) in enumerate(rows):
        color = PALETTE[r % len(PALETTE)]
        for j, p in enumerate(free):
            m = fit["multipliers"][p]
            y = j + (r - (n_rows - 1) / 2) * step
            if m.get("fixed"):                         # not fitted in this row: the given value
                ax.plot(m["value"], y, "x", ms=6, mew=1.5, color=color)
                any_given = True
                continue
            err = m["log10_err"]
            lo, hi = 10 ** (m["log10"] - err), 10 ** (m["log10"] + err)
            ax.plot(m["range_equivalent_starts"], [y, y], color=color, lw=7, alpha=0.25,
                    solid_capstyle="butt")
            ax.plot([max(lo, 1e-3), min(hi, 1e3)], [y, y], color=color, lw=1.5)
            ax.plot(m["value"], y, "o", ms=6, color=color, mfc=color if m["determined"] else "white",
                    label=label if not ax.get_legend_handles_labels()[1].count(label) else None)
    ax.axvline(1.0, color="#dddddd", lw=1, zorder=0)
    ax.axvspan(1e-3, 1e-2, color="#f2f2f2", zorder=0)
    ax.axvspan(1e2, 1e3, color="#f2f2f2", zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 1e3)
    ax.set_yticks(range(len(free)))
    ax.set_yticklabels([LABELS[p] for p in free])
    ax.invert_yaxis()
    ax.set_xlabel("fitted value / Lee SI sample 5")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.16), ncol=min(3, n_rows))
    ax.set_title(f"Fitted parameters, relative to {fixed_text} (fixed)\n"
                 "thin bar: 1σ; thick bar: range of equally good starting points\n"
                 "open marker: not determined (> factor 3 or at a bound)"
                 + ("; ×: given, not fitted" if any_given else ""),
                 loc="left", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(out_path(fname), dpi=170)


if __name__ == "__main__":
    t_start = time.perf_counter()
    free = [p for p in PARAMS if p in FREE_PARAMS]
    print("Measurements: " + ", ".join(f"{n} ({TITLES[n]})" for n in NAMES) + "; joint fits: "
          + ("; ".join(f"{s or 'all'}: {', '.join(v)}" for s, v in JOINT.items())
             or "none (no sample with two or more measurements)"), flush=True)
    fits = {"k_only": run_fit(NAMES, [], label="k only")}
    print_fit("k only (Lee SI sample 5)", fits["k_only"])
    fits["joint"] = {}
    for sample, names in JOINT.items():
        fits["joint"][sample] = run_fit(names, free, label=f"joint {sample}".strip())
        print_fit(f"all parameters, {shared_label(sample)}", fits["joint"][sample])
    fits["separate"] = {}
    for name in NAMES:
        fits["separate"][name] = run_fit([name], free, label=name)
        print_fit(f"all parameters, only {name}", fits["separate"][name])
    plot_fits(fits)
    plot_values(fits)
    fits["model_info"] = MODEL_INFO
    fits["settings"] = dict(free_params=free, bounds_log10=BOUNDS_LOG10, log10_k_bounds=LOG10_K_BOUNDS,
                            n_starts=N_STARTS, rel_scatter=REL_SCATTER, joint_groups=JOINT,
                            reference="Lee SI sample 5 (8% Tm, w2_nr = w3_nr = 0); w3 and sigma_ESA fixed")
    json.dump(fits, open(out_path("all_params_fit.json"), "w"), indent=1, default=float)
    print(f"\nSaved results/fig4_all_params_fit.png, fig5_all_params_values.png, all_params_fit.json "
          f"({time.perf_counter() - t_start:.0f} s)")
