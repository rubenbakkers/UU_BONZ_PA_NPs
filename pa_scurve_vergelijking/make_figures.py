# -*- coding: utf-8 -*-
"""Figures + summary: measured BdJ1 S-curves vs pa_fononband model."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts in deze map vindbaar
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pa_fit_lib import (data_path, out_path, trapz, pm, pf, lee_params, rates, analytical_ith, emission_curve,
                        ensemble_curve, load, local_slope_binned, MODEL_INFO)
from fit_scurves import DATA, I_GRID, XB, WB, fit_k

BLUE, ORANGE, GREY, INK, INK2 = "#257de0", "#f47230", "#9a9a9a", "#191919", "#666666"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False,
                     "lines.linewidth": 2})

data = {n: load(v["file"]) for n, v in DATA.items()}
LAM0 = lee_params(3).lambda0_gsa_nm
P = pm.build_parameters(8.0, 0.0, 0.0, lambda0_gsa_nm=LAM0)   # = Lee SI sample 5
TITLES = {"cobolt_1060": "Cobolt, 1060 nm", "superk_1064": "SuperK, 1064 nm"}
MARK = {1: "o", 2: "s", 4: "o", 5: "s"}


def dlog(I, y):
    return np.gradient(np.log(np.clip(y, 1e-300, None)), np.log(I))


summary = {}
fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), sharex="col",
                         gridspec_kw=dict(height_ratios=[1.6, 1]))
for col, (name, cfg) in enumerate(DATA.items()):
    d = data[name]
    wg, we = rates(cfg["lam"], P)
    ith = analytical_ith(P, wg, we)
    E = emission_curve(I_GRID, P, wg, we)
    S = ensemble_curve(I_GRID, I_GRID, E, XB, WB)
    r_ens = fit_k(S, d)
    r_np = fit_k(E, d)
    k, A, b = r_ens["k"], r_ens["A"], r_ens["b"]
    fws = np.unique(d["fw6"])
    bmap = dict(zip(fws, b))
    ycorr = d["net_cps"] - np.array([bmap[f] for f in d["fw6"]])

    # model slopes (in reported-intensity units)
    sl_ens, sl_np = dlog(I_GRID, S), dlog(I_GRID, E)
    summary[name] = dict(
        wavelength_nm=cfg["lam"], model="Lee SI sample 5 (8% Tm, w2_nr=w3_nr=0), phonon-band GSA/ESA",
        Ith_model_kWcm2=ith, k_ensemble=k, Ith_on_reported_scale=ith / k,
        chi2_red_ensemble=r_ens["chi2_red"], chi2_red_single_np=r_np["chi2_red"],
        k_single_np=r_np["k"], background_per_FW={int(f): float(v) for f, v in bmap.items()},
        Smax_model_single_np=float(sl_np.max()), Smax_model_ensemble=float(sl_ens.max()))

    ax = axes[0, col]
    Ir = d["I_kWcm2"]
    for f in fws:
        m = (d["fw6"] == f)
        good = m & (ycorr > 3 * d["noise_cps"])
        weak = m & (ycorr > 0) & ~good
        ax.plot(Ir[good], ycorr[good], MARK[f], ms=5, mfc="none", mec=INK, mew=1,
                label=f"meting FW6={int(f)}")
        ax.plot(Ir[weak], ycorr[weak], MARK[f], ms=4, mfc="none", mec="#c8c8c8", mew=1)
    Iplot = np.logspace(np.log10(Ir.min()), np.log10(Ir.max() * 1.05), 400)
    li, le = np.log(I_GRID), np.log
    interp = lambda Y, kk: np.exp(np.interp(np.log(kk * Iplot), li, np.log(np.clip(Y, 1e-300, None))))
    ax.plot(Iplot, A * interp(S, k), color=BLUE,
            label=f"model, ensemble in spot (k={k:.1f}, $\\chi^2_r$={r_ens['chi2_red']:.2f})")
    ax.plot(Iplot, r_np["A"] * interp(E, r_np["k"]), color=ORANGE, ls="--", lw=1.6,
            label=f"model, 1 NP (k={r_np['k']:.1f}, $\\chi^2_r$={r_np['chi2_red']:.2f})")
    ax.plot(Iplot, A * interp(S, 1.0), color=GREY, ls=":", lw=1.8,
            label="model zonder schaalfactor (k=1)")
    ax.axvline(ith / k, color=BLUE, lw=0.8, alpha=0.5)
    ax.text(ith / k * 0.96, 1.05, f"$I_{{th}}$(model)/k = {ith / k:.2f}", color=INK2, fontsize=8,
            va="bottom", ha="right")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_ylim(0.8, max(ycorr.max(), 30) * 2.5)
    ax.set_title(TITLES[name], color=INK, loc="left", fontsize=11)
    ax.set_ylabel("emissie − achtergrond [cps]")
    ax.legend(fontsize=7.5, loc="upper left")

    ax2 = axes[1, col]
    xs, sl = local_slope_binned(Ir, ycorr, d["noise_cps"], width_dex=0.07, min_snr=4)
    ax2.plot(xs, sl, "o", ms=4, mfc="none", mec=INK, mew=1, label="meting (lokale fit, 0.07 dex)")
    ax2.plot(Iplot, np.interp(np.log(k * Iplot), li, sl_ens), color=BLUE, label="model, ensemble")
    ax2.plot(Iplot, np.interp(np.log(r_np["k"] * Iplot), li, sl_np), color=ORANGE, ls="--", lw=1.6,
             label="model, 1 NP")
    ax2.axhline(1, color="#dddddd", lw=1, zorder=0)
    ax2.set_ylim(0, 35)
    ax2.set_xscale("log")
    ax2.set_xlabel("gerapporteerde intensiteit [kW cm$^{-2}$]")
    ax2.set_ylabel("helling d log E / d log I")
    ax2.legend(fontsize=7.5, loc="upper left")
    summary[name]["Smax_data_binned"] = float(np.nanmax(sl))

fig.suptitle("BdJ1, 24-11-2026: gemeten S-curves vs. pa_fononband-model (Lee SI sample 5: 8% Tm, geen niet-radiatief verlies)",
             fontsize=10, color=INK, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(out_path("fig1_scurves_vs_model.png"), dpi=170)

# ---------------- Fig 2: threshold vs wavelength (relative to 1060 nm) -----
lam = np.arange(1050.0, 1080.01, 0.1)
fig2, ax = plt.subplots(figsize=(6.6, 4.0))
for sig, style in ((8.5, dict(color=BLUE)), (25.0, dict(color=BLUE, ls="--", lw=1.5, alpha=0.7))):
    with pf._stark_sigma(sig, keep_anchor=True):
        ith_l = np.array([analytical_ith(P, *rates(l, P)) for l in lam])
        ith_1060 = analytical_ith(P, *rates(1060.0, P))
    ax.plot(lam, ith_l / ith_1060, label=f"model, Stark-breedte $\\sigma$ = {sig:g} cm$^{{-1}}$", **style)
ratio = summary["superk_1064"]["Ith_on_reported_scale"] / summary["cobolt_1060"]["Ith_on_reported_scale"]
ax.plot([1060], [1.0], "o", ms=8, mfc=INK, mec="white", label="Cobolt (referentie)")
ax.plot([1064], [ratio], "s", ms=8, mfc="white", mec=INK, mew=1.5,
        label=f"SuperK, als 1064 nm: {ratio:.2f}")
# SuperK data analysed as if at 1060 nm
wg60, we60 = rates(1060.0, P)
E60 = emission_curve(I_GRID, P, wg60, we60)
r60 = fit_k(ensemble_curve(I_GRID, I_GRID, E60, XB, WB), data["superk_1064"])
ith60 = analytical_ith(P, wg60, we60)
summary["superk_if_1060nm"] = dict(k_ensemble=r60["k"], Ith_on_reported_scale=ith60 / r60["k"],
                                   chi2_red=r60["chi2_red"])
summary["ratio_Ith_superk_over_cobolt_reported"] = ratio
summary["model_info"] = MODEL_INFO
ax.axhline(1, color="#dddddd", lw=1, zorder=0)
ax.set_xlabel("pompgolflengte [nm]")
ax.set_ylabel("$I_{th}(\\lambda)$ / $I_{th}$(1060 nm)")
ax.set_ylim(0, 2.5)
ax.legend(fontsize=7.5, loc="upper right")
ax.set_title("Drempel t.o.v. 1060 nm: model vs. meting", loc="left", fontsize=10, color=INK)
fig2.tight_layout()
fig2.savefig(out_path("fig2_threshold_vs_wavelength.png"), dpi=170)

json.dump(summary, open(out_path("summary.json"), "w"), indent=1, default=float)
print(json.dumps(summary, indent=1, default=float))
