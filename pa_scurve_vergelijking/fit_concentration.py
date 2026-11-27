# -*- coding: utf-8 -*-
"""Fit of the measured S-curves with the Tm concentration of each sample given, or fitted.

The script asks for the Tm concentration of every sample in measurements.py
(field "sample"). In the model the concentration sets the energy-transfer rates
(s31, Q22, Q23; pa_model.transfer_coefficients), so

  concentration given   : s31, Q22, Q23 follow from it (Lee's relations);
  left blank / "none"   : the concentration is fitted as well (between 4 and
                          100 %; at 4 % Lee's relations jump to the coefficients
                          of the 4 % sample, below that they are extrapolated).

Everything that does not follow from the concentration is fitted, with the
machinery of fit_all_params.py (same model, weights, bounds, starting points
and "determined" criterion):

  sigma_gsa  GSA cross section (the GSA/ESA ratio; sigma_ESA stays fixed)
  w2         3F4 decay rate
  w3         3H4 decay rate (with s31, Q22 and Q23 tied to the concentration
             the fit is not degenerate, unlike in fit_all_params.py)
  b32        3H4 -> 3F4 branching ratio
  per curve: k (intensity scale), A and one background per filter-wheel setting

Fits, as in fit_all_params.py: "k only" (Lee SI sample 5, 8 % Tm, as reference),
"joint" (the curves of one sample share the particle parameters) and "separate"
(each curve on its own, with the concentration of its sample). At the end the
chi2 and BIC are compared with fit_all_params.py (results/all_params_fit.json,
if present), where s31, Q22 and Q23 are free.

Without a console (no input possible) every concentration is fitted.

Output: results/fig6_concentration_fit.png, results/fig7_concentration_values.png,
results/concentration_fit.json. Run time a few minutes per sample.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # make the scripts in this folder importable
import json
import time

import numpy as np

import fit_all_params as fa
from pa_fit_lib import out_path, pm, MODEL_INFO

FREE = ("sigma_gsa", "w2", "w3", "b32")         # everything that does not follow from the concentration
UNKNOWN = {"", "none", "no", "nee", "geen", "?", "-", "x", "unknown", "onbekend", "fit", "auto", "n/a", "na"}
C_REF = fa.P0.concentration_percent              # 8 %: the concentration of the reference (Lee SI sample 5)


def ask_concentration(sample, names):
    """Tm concentration (%) typed by the user, or None (blank, 'none', ...: fit it)."""
    who = f"sample {sample}" if sample else "the measurements without a sample name"
    prompt = (f"Tm concentration of {who} ({', '.join(names)}) in %, "
              "or leave blank to fit it: ")
    while True:
        try:
            answer = input(prompt)
        except EOFError:                          # no console: fit it
            print(f"\n(no input possible: the concentration of {who} is fitted)")
            return None
        text = answer.strip().lower().rstrip("%").strip().replace(",", ".")
        if text in UNKNOWN:
            return None
        try:
            c = float(text)
        except ValueError:
            print(f"  '{answer}' is not a number: type e.g. 8 or 8.5, or leave blank to fit it.")
            continue
        if not 0.0 < c <= 100.0:
            print("  The concentration must be above 0 and at most 100 %.")
            continue
        if c < 4.0:
            print("  Note: below 4 % Lee et al. give no relations; the model uses the coefficients "
                  "of the 4 % sample (an extrapolation).")
        return c


def physical(fit):
    """Concentration, lifetimes and branching ratio of a fit, next to Lee's values."""
    p, _wg, _we = fa.effective_params(np.array(fit["_log_m"]), fit["curves"][0])
    m = fit["multipliers"]
    out = dict(concentration_percent=p.concentration_percent, concentration_fitted=not m["conc"].get("fixed"),
               sigma_gsa_multiplier=m["sigma_gsa"]["value"], sigma_gsa_m2=p.sigma_gsa_m2,
               s31=p.s31, q22=p.q22, q23=p.q23,
               tau2_ms=1e3 / p.w2, tau2_lee_ms=1e3 / fa.P0.w2,
               tau3_ms=1e3 / p.w3, tau3_lee_ms=1e3 / fa.P0.w3,
               b32=p.b32, b32_lee=fa.P0.b32)
    if out["concentration_fitted"]:
        c = m["conc"]
        out.update(concentration_factor_err=c["factor_err"], concentration_determined=c["determined"],
                   concentration_at_bound=c["at_bound"],
                   concentration_range_equivalent_starts=[v * C_REF for v in c["range_equivalent_starts"]])
    return out


if __name__ == "__main__":
    print("Measurements: " + ", ".join(f"{n} ({fa.TITLES[n]})" for n in fa.NAMES), flush=True)
    given = {sample: ask_concentration(sample, names) for sample, names in fa.GROUPS.items()}
    t_start = time.perf_counter()
    free_of, fixed_of, what = {}, {}, {}
    for sample, c in given.items():
        if c is None:
            free_of[sample], fixed_of[sample] = ("conc",) + FREE, None
            what[sample] = "Tm conc. fitted too"
        else:
            free_of[sample], fixed_of[sample] = FREE, {"conc": float(np.log10(c / C_REF))}
            what[sample] = f"{c:g}% Tm, rest fitted"
        tc = pm.transfer_coefficients(c) if c is not None else None
        print(f"  {sample or '(no sample name)'}: "
              + (f"Tm concentration {c:g} % -> s31 = {tc[0]:.4g}, Q22 = {tc[1]:.4g}, Q23 = {tc[2]:.4g} "
                 f"(Lee et al.)" if c is not None else
                 f"Tm concentration fitted ({fa.CONC_FIT_RANGE_PERCENT[0]:g}-{fa.CONC_FIT_RANGE_PERCENT[1]:g} %)")
              + f"; free: {', '.join(free_of[sample])}", flush=True)

    fits = {"k_only": fa.run_fit(fa.NAMES, [], label="k only")}
    fa.print_fit(f"k only (Lee SI sample 5, {C_REF:g} % Tm)", fits["k_only"])
    fits["joint"] = {}
    for sample, names in fa.JOINT.items():
        fits["joint"][sample] = fa.run_fit(names, free_of[sample], label=f"joint {sample}".strip(),
                                           fixed=fixed_of[sample])
        fa.print_fit(f"{what[sample]}, {fa.shared_label(sample)}", fits["joint"][sample])
    fits["separate"] = {}
    for name in fa.NAMES:
        sample = fa.DATA[name]["sample"]
        fits["separate"][name] = fa.run_fit([name], free_of[sample], label=name, fixed=fixed_of[sample])
        fa.print_fit(f"{what[sample]}, only {name}", fits["separate"][name])

    fitted_any = any(c is None for c in given.values())
    fa.plot_fits(fits, what=what, fname="fig6_concentration_fit.png",
                 title="S-curves fitted with the Tm concentration " + (
                     "given per sample" if not fitted_any else
                     "fitted" if all(c is None for c in given.values()) else "given or fitted"))
    fa.plot_values(fits, free_params=("conc",) + FREE, fname="fig7_concentration_values.png",
                   fixed_text="$\\sigma_{ESA}$; $s_{31}$, $Q_{22}$, $Q_{23}$ follow from $c$")

    # ---- comparison: how good are these fits next to the others? ----
    rows = [(f"k only, Lee {C_REF:g} %", fits["k_only"])]
    rows += [(f"{what[s]}, joint {s}", f) for s, f in fits["joint"].items()]
    rows += [(f"{what[fa.DATA[n]['sample']]}, only {n}", f) for n, f in fits["separate"].items()]
    all_free_file = out_path("all_params_fit.json")
    if os.path.exists(all_free_file):
        prev = json.load(open(all_free_file))
        rows += [(f"all parameters free, joint {s}", f) for s, f in prev.get("joint", {}).items()]
        rows += [(f"all parameters free, only {n}", f) for n, f in prev.get("separate", {}).items()]
    print("\n=== Comparison (lower BIC = better; a difference > 6 is strong evidence) ===")
    for label, f in rows:
        print(f"  {label:50s} chi2 = {f['chi2']:7.1f}  chi2_red = {f['chi2_red']:.3f}  "
              f"{f['n_params']:2d} parameters  BIC = {f['BIC']:7.1f}")

    print("\n=== Best values ===")
    best = {}
    for label, fit in ([(f"joint {s}", f) for s, f in fits["joint"].items()]
                       + [(f"only {n}", f) for n, f in fits["separate"].items()]):
        ph = physical(fit)
        best[label] = ph
        mult = fit["multipliers"]
        flag = lambda q: "" if mult[q].get("fixed") or mult[q]["determined"] else (
            " (at the bound of the fit)" if mult[q]["at_bound"] else " (not determined)")
        if ph["concentration_fitted"]:
            lo, hi = ph["concentration_range_equivalent_starts"]
            conc = (f"c = {ph['concentration_percent']:.3g} % fitted (1 sigma x/÷ "
                    f"{ph['concentration_factor_err']:.2g}; equally good starts {lo:.3g}-{hi:.3g} %)"
                    f"{flag('conc')}")
        else:
            conc = f"c = {ph['concentration_percent']:g} % (given)"
        print(f"  {label}:\n    {conc}: s31 = {ph['s31']:.4g}, Q22 = {ph['q22']:.4g}, Q23 = {ph['q23']:.4g}\n"
              f"    sigma_GSA x {ph['sigma_gsa_multiplier']:.3g} (= {ph['sigma_gsa_m2'] * 1e4:.3g} cm^2)"
              f"{flag('sigma_gsa')}\n"
              f"    tau(3F4) = {ph['tau2_ms']:.3g} ms (Lee {ph['tau2_lee_ms']:.3g} ms){flag('w2')}\n"
              f"    tau(3H4) = {ph['tau3_ms']:.3g} ms (Lee {ph['tau3_lee_ms']:.3g} ms){flag('w3')}\n"
              f"    b32 = {ph['b32']:.3g} (Lee {ph['b32_lee']:.3g}){flag('b32')}\n"
              + "".join(f"    {n}: k = {c['k']:.3g}, I_th/k = {c['Ith_on_reported_scale']:.3g} kW/cm^2, "
                        f"chi2_red = {c['chi2_red']:.3f}\n" for n, c in fit["per_curve"].items()), end="")

    fits["best_values"] = best
    fits["model_info"] = MODEL_INFO
    fits["settings"] = dict(concentration_given_percent=given, free_params={s: list(v) for s, v in free_of.items()},
                            conc_fit_range_percent=fa.CONC_FIT_RANGE_PERCENT,
                            bounds_log10={q: fa.BOUNDS_LOG10[q] for q in ("conc",) + FREE},
                            log10_k_bounds=fa.LOG10_K_BOUNDS, n_starts=fa.N_STARTS,
                            joint_groups=fa.JOINT, sigma_esa_m2=fa.P0.sigma_esa_m2)
    json.dump(fits, open(out_path("concentration_fit.json"), "w"), indent=1, default=float)
    print(f"\nSaved results/fig6_concentration_fit.png, fig7_concentration_values.png, concentration_fit.json "
          f"({time.perf_counter() - t_start:.0f} s)")
