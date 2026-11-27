# -*- coding: utf-8 -*-
"""Figures + summary: the measured S-curves (all measurements in measurements.py)
vs. the pa_fononband model.

Fig. 1: per measurement the S-curve with the fits and the local slope.
Fig. 2: threshold vs. wavelength, relative to REFERENCE (measurements.py).
results/summary.json: fit results per measurement; the other scripts read it.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # make the scripts in this folder importable
import json
import numpy as np
import matplotlib.pyplot as plt
from pa_fit_lib import (out_path, pm, pf, lee_params, rates, analytical_ith, emission_curve,
                        ensemble_curve, load, local_slope_binned, measurements, reference_name,
                        describe, format_parameters, MODEL_INFO)
from fit_scurves import I_GRID, XB, WB, fit_k
from plotting import BLUE, ORANGE, GREY, INK, INK2, FW_MARKERS, apply_style, fw_markers, scurve_axes

apply_style(10)
DATA = measurements()
REF = reference_name(DATA)
data = {n: load(m["file"]) for n, m in DATA.items()}
LAM0 = lee_params(3).lambda0_gsa_nm
P = pm.build_parameters(8.0, 0.0, 0.0, lambda0_gsa_nm=LAM0)   # = Lee SI sample 5
print("Measurements: " + ", ".join(f"{n} ({m['label']})" for n, m in DATA.items())
      + f"; reference: {REF}", flush=True)


def dlog(I, y):
    return np.gradient(np.log(np.clip(y, 1e-300, None)), np.log(I))


def knee_sweep(d, I_knee):
    """Filter-wheel setting of the sweep through the knee: of the sweeps whose
    intensity range contains I_knee, the one with the most points within a
    factor 2 of it (if none contains I_knee: the sweep that comes closest)."""
    best = None
    for f in np.unique(d["fw6"]):
        I = d["I_kWcm2"][d["fw6"] == f]
        contains = bool(I.min() <= I_knee <= I.max())
        near = int(np.sum(np.abs(np.log(I / I_knee)) <= np.log(2.0)))
        gap = 0.0 if contains else float(min(abs(np.log(I.min() / I_knee)),
                                             abs(np.log(I.max() / I_knee))))
        score = (contains, near, -gap)
        if best is None or score > best[0]:
            best = (score, int(f))
    return best[1]


summary = {}
report = {}               # per measurement: the parameters of the fit (printed at the end)
fig, panels = scurve_axes(len(DATA))
for (name, cfg), (ax, ax2) in zip(DATA.items(), panels):
    d = data[name]
    wg, we = rates(cfg["wavelength_nm"], P, cfg["fwhm_nm"])
    ith = analytical_ith(P, wg, we)
    E = emission_curve(I_GRID, P, wg, we)
    S = ensemble_curve(I_GRID, I_GRID, E, XB, WB)
    r_ens = fit_k(S, d)
    r_np = fit_k(E, d)
    k, A, b = r_ens["k"], r_ens["A"], r_ens["b"]
    fws = np.unique(d["fw6"])
    bmap = dict(zip(fws, b))
    ycorr = d["net_cps"] - np.array([bmap[f] for f in d["fw6"]])
    if cfg["knee_fw"] is not None:
        knee_fw = int(cfg["knee_fw"])
        if knee_fw not in fws:
            raise ValueError(f"measurements.py, '{name}': knee_fw={knee_fw}, but the file only has "
                             f"fw6 = {', '.join(str(int(f)) for f in fws)}.")
    else:
        knee_fw = knee_sweep(d, ith / k)

    # model slopes (in reported-intensity units)
    sl_ens, sl_np = dlog(I_GRID, S), dlog(I_GRID, E)
    summary[name] = dict(
        label=cfg["label"], sample=cfg["sample"], date=cfg["date"],
        wavelength_nm=cfg["wavelength_nm"], fwhm_nm=cfg["fwhm_nm"],
        model="Lee SI sample 5 (8% Tm, w2_nr=w3_nr=0), phonon-band GSA/ESA",
        Ith_model_kWcm2=ith, k_ensemble=k, Ith_on_reported_scale=ith / k,
        chi2_red_ensemble=r_ens["chi2_red"], chi2_red_single_np=r_np["chi2_red"],
        k_single_np=r_np["k"], background_per_FW={int(f): float(v) for f, v in bmap.items()},
        knee_fw=knee_fw,
        Smax_model_single_np=float(sl_np.max()), Smax_model_ensemble=float(sl_ens.max()))

    Ir = d["I_kWcm2"]
    marks = fw_markers(fws)
    for f in fws:
        m = (d["fw6"] == f)
        good = m & (ycorr > 3 * d["noise_cps"])
        weak = m & (ycorr > 0) & ~good
        ax.plot(Ir[good], ycorr[good], marks[f], ms=5, mfc="none", mec=INK, mew=1,
                label=f"measured FW6={int(f)}")
        ax.plot(Ir[weak], ycorr[weak], marks[f], ms=4, mfc="none", mec="#c8c8c8", mew=1)
    x_lo, x_hi = Ir.min() / 1.15, Ir.max() * 1.15        # the model lines run to the plot edges
    Iplot = np.logspace(np.log10(x_lo), np.log10(x_hi), 400)
    li = np.log(I_GRID)
    interp = lambda Y, kk: np.exp(np.interp(np.log(kk * Iplot), li, np.log(np.clip(Y, 1e-300, None))))
    ax.plot(Iplot, A * interp(S, k), color=BLUE,
            label=f"model, ensemble in spot (k={k:.1f}, $\\chi^2_r$={r_ens['chi2_red']:.2f})")
    ax.plot(Iplot, r_np["A"] * interp(E, r_np["k"]), color=ORANGE, ls="--", lw=1.6,
            label=f"model, 1 NP (k={r_np['k']:.1f}, $\\chi^2_r$={r_np['chi2_red']:.2f})")
    ax.plot(Iplot, A * interp(S, 1.0), color=GREY, ls=":", lw=1.8,
            label="model without scale factor (k=1)")
    ax.axvline(ith / k, color=BLUE, lw=0.8, alpha=0.5)
    ax.text(ith / k * 0.96, 1.05, f"$I_{{th}}$(model)/k = {ith / k:.2f}", color=INK2, fontsize=8,
            va="bottom", ha="right")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(0.8, max(ycorr.max(), 30) * 2.5)
    ax.set_title(cfg["label"], color=INK, loc="left", fontsize=11)
    ax.set_ylabel("emission − background [cps]")
    ax.legend(fontsize=7.5, loc="upper left")

    xs, sl = local_slope_binned(Ir, ycorr, d["noise_cps"], width_dex=0.07, min_snr=4)
    ax2.plot(xs, sl, "o", ms=4, mfc="none", mec=INK, mew=1, label="measured (local fit, 0.07 dex)")
    ax2.plot(Iplot, np.interp(np.log(k * Iplot), li, sl_ens), color=BLUE, label="model, ensemble")
    ax2.plot(Iplot, np.interp(np.log(r_np["k"] * Iplot), li, sl_np), color=ORANGE, ls="--", lw=1.6,
             label="model, 1 NP")
    ax2.axhline(1, color="#dddddd", lw=1, zorder=0)
    ax2.set_ylim(0, 35)
    ax2.set_xscale("log")
    ax2.set_xlim(x_lo, x_hi)                               # shared with the S-curve panel
    ax2.set_xlabel("reported intensity [kW cm$^{-2}$]")
    ax2.set_ylabel("slope d log E / d log I")
    ax2.legend(fontsize=7.5, loc="upper left")
    summary[name]["Smax_data_binned"] = float(np.nanmax(sl)) if len(sl) else None
    print(f"{name}: k = {k:.2f}, I_th/k = {ith / k:.3f} kW/cm2, chi2_red = {r_ens['chi2_red']:.2f}, "
          f"knee in sweep FW6={knee_fw}", flush=True)
    bg_np = dict(zip(fws, r_np["b"]))
    report[name] = (
        f"  {name} ({cfg['label']}):\n"
        f"    pump {cfg['wavelength_nm']:g} nm (FWHM {cfg['fwhm_nm']:g} nm): w_GSA = {wg:.4g}, "
        f"w_ESA = {we:.4g} s^-1 per kW cm^-2; I_th(model) = {ith:.4g} kW/cm^2\n"
        f"    best fit, ensemble in spot: k = {k:.4g} (I_th/k = {ith / k:.4g} kW/cm^2), A = {A:.4g}, "
        "background " + ", ".join(f"FW6={int(f)}: {v:.4g}" for f, v in bmap.items())
        + f" cps, chi2_red = {r_ens['chi2_red']:.3f}\n"
        f"    best fit, single NP:        k = {r_np['k']:.4g} (I_th/k = {ith / r_np['k']:.4g} kW/cm^2), "
        f"A = {r_np['A']:.4g}, background " + ", ".join(f"FW6={int(f)}: {v:.4g}" for f, v in bg_np.items())
        + f" cps, chi2_red = {r_np['chi2_red']:.3f}")

where = describe(DATA)
fig.suptitle((f"{where}: " if where else "") + "measured S-curves vs. pa_fononband model "
             "(Lee SI sample 5: 8% Tm, no non-radiative loss)",
             fontsize=10, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(out_path("fig1_scurves_vs_model.png"), dpi=170)

# ---------------- Fig 2: threshold vs wavelength (relative to the reference) -----
lam_ref = DATA[REF]["wavelength_nm"]
lam_all = [m["wavelength_nm"] for m in DATA.values()]
lam = np.arange(min(1050.0, min(lam_all) - 5.0), max(1080.0, max(lam_all) + 5.0) + 0.01, 0.1)
fig2, ax = plt.subplots(figsize=(6.6, 4.0))
for sig, style in ((8.5, dict(color=BLUE)), (25.0, dict(color=BLUE, ls="--", lw=1.5, alpha=0.7))):
    with pf._stark_sigma(sig, keep_anchor=True):
        ith_l = np.array([analytical_ith(P, *rates(l, P)) for l in lam])
        ith_ref = analytical_ith(P, *rates(lam_ref, P))
    ax.plot(lam, ith_l / ith_ref, label=f"model, Stark width $\\sigma$ = {sig:g} cm$^{{-1}}$", **style)
ith_rep_ref = summary[REF]["Ith_on_reported_scale"]
open_markers = iter(FW_MARKERS[1:] * 10)
ratios = []
for name, cfg in DATA.items():
    ratio = summary[name]["Ith_on_reported_scale"] / ith_rep_ref
    summary[name]["Ith_ratio_to_reference"] = ratio
    ratios.append(ratio)
    if name == REF:
        ax.plot([cfg["wavelength_nm"]], [ratio], "o", ms=8, mfc=INK, mec="white",
                label=f"{cfg['label']} (reference)")
        continue
    ax.plot([cfg["wavelength_nm"]], [ratio], next(open_markers), ms=8, mfc="white", mec=INK, mew=1.5,
            label=f"{cfg['label']}: {ratio:.2f}")
    # the same data analysed as if measured at the reference wavelength
    if cfg["wavelength_nm"] != lam_ref:
        wg0, we0 = rates(lam_ref, P, cfg["fwhm_nm"])
        E0 = emission_curve(I_GRID, P, wg0, we0)
        r0 = fit_k(ensemble_curve(I_GRID, I_GRID, E0, XB, WB), data[name])
        ith0 = analytical_ith(P, wg0, we0)
        summary[name]["if_at_reference_wavelength"] = dict(
            wavelength_nm=lam_ref, k_ensemble=r0["k"], Ith_on_reported_scale=ith0 / r0["k"],
            chi2_red=r0["chi2_red"])
        report[name] += (f"\n    same data analysed as if at {lam_ref:g} nm: w_GSA = {wg0:.4g}, "
                         f"w_ESA = {we0:.4g}; k = {r0['k']:.4g}, A = {r0['A']:.4g}, "
                         f"I_th/k = {ith0 / r0['k']:.4g} kW/cm^2, chi2_red = {r0['chi2_red']:.3f}")
summary["reference"] = REF
summary["model_info"] = MODEL_INFO
ax.axhline(1, color="#dddddd", lw=1, zorder=0)
ax.set_xlim(lam[0], lam[-1])                               # the model lines run to the plot edges
ax.set_xlabel("pump wavelength [nm]")
ax.set_ylabel(f"$I_{{th}}(\\lambda)$ / $I_{{th}}$({lam_ref:g} nm)")
ax.set_ylim(0, max(2.5, 1.15 * max(ratios)))
ax.legend(fontsize=7.5, loc="upper right")
ax.set_title(f"Threshold relative to {DATA[REF]['label']}: model vs. measurement",
             loc="left", fontsize=10, color=INK)
fig2.tight_layout()
fig2.savefig(out_path("fig2_threshold_vs_wavelength.png"), dpi=170)

json.dump(summary, open(out_path("summary.json"), "w"), indent=1, default=float)
print(json.dumps(summary, indent=1, default=float))

print("\n=== Parameters used ===")
print("Model, the same for every measurement (Lee SI sample 5: 8% Tm, no non-radiative loss):")
print(format_parameters(P))
print("Fitted per measurement: counts = A * M(k * I_reported) + background per filter-wheel setting")
print("\n".join(report.values()))
