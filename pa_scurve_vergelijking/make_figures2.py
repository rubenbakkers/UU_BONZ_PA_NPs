# -*- coding: utf-8 -*-
"""Fig 3: other parts of pa_fononband vs. the BdJ1 data (24-11-2026)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts in deze map vindbaar
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pa_fit_lib import data_path, out_path, trapz, pm, pf, lee_params, rates, emission_curve, ensemble_curve, load
from fit_scurves import I_GRID, XB, WB

BLUE, ORANGE, AQUA, GREY, INK, INK2 = "#257de0", "#f47230", "#00b894", "#9a9a9a", "#191919", "#666666"
plt.rcParams.update({"font.size": 9.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2})
summ = json.load(open(out_path("summary.json")))
rise = json.load(open(out_path("rise_variants.json")))
prot = json.load(open(out_path("protocol_sim.json")))
var = json.load(open(out_path("variants_fit.json")))
P = pm.build_parameters(8.0, 0.0, 0.0, lambda0_gsa_nm=lee_params(3).lambda0_gsa_nm)

fig, ax = plt.subplots(2, 2, figsize=(10.5, 7.6))

# (a) emission spectrum vs 3H4 -> 3H6 Stark lines; fit from emission_spec.py
from emission_spec import load_spectrum, model_spectrum, fit_emission
em_path = out_path("emission_spec.json")
em = json.load(open(em_path)) if os.path.exists(em_path) else fit_emission()
lam, y = load_spectrum()
lg = np.linspace(775, 865, 600)
a = ax[0, 0]
y_smooth = np.convolve(y, np.ones(5) / 5, mode="same")
a.plot(lam, y_smooth, color=INK, lw=1.2, label="gemeten (SuperK, 5-px gemiddeld)")
curves_a = []
for s, ls, lab in ((0.0, "--", "model 3H4→3H6 (Stark-niveaus script, op 0 nm)"),
                   (em["shift_nm"], "-", f"model, {em['shift_nm']:.1f} nm verschoven (fit, "
                                         f"σ = {em['sigma_cm1']:.0f} cm$^{{-1}}$)")):
    m = model_spectrum(lg, s, em["sigma_cm1"], em["amplitude"], em["background"])
    curves_a.append(m)
    a.plot(lg, m, color=BLUE, ls=ls, lw=1.6, label=lab)
top = max(y_smooth.max(), max(c.max() for c in curves_a))
pk_model, pk_data = em["model_peak_nm_unshifted"], em["data_peak_nm"]
a.axvline(pk_model, color=GREY, lw=0.8); a.axvline(pk_data, color=INK2, lw=0.8)
a.text(pk_model - 0.9, 1.1 * top, f"model\n{pk_model:.0f} nm", ha="right", fontsize=8, color=INK2)
a.text(pk_data + 1.0, 1.1 * top, f"meting\n{pk_data:.0f} nm", ha="left", fontsize=8, color=INK2)
a.set_xlim(775, 865); a.set_ylim(-0.1 * top, 1.85 * top)
a.set_xlabel("golflengte [nm]"); a.set_ylabel("cts s$^{-1}$ pixel$^{-1}$")
a.set_title("a  Emissiespectrum vs. Stark-niveaus", loc="left", fontsize=10, color=INK)
a.legend(fontsize=7.5, loc="upper left")

# (b) rise time vs I/Ith for the three explanations of the factor k
b = ax[0, 1]
r = np.array(rise["ratio"])
for (name, col, ls) in (("kalibratie (I_echt = k*I_gerapp)", BLUE, "-"),
                        ("sigma_ESA x k", ORANGE, "--"),
                        ("w2 / k (langere 3F4-levensduur)", AQUA, "-.")):
    lab = {"kalibratie (I_echt = k*I_gerapp)": "factor k = kalibratie",
           "sigma_ESA x k": "factor k = σ$_{ESA}$ × k",
           "w2 / k (langere 3F4-levensduur)": "factor k = w$_2$ / k (3F4 langer)"}[name]
    b.plot(r, rise[name]["t95"], color=col, ls=ls, marker="o", ms=3.5, label=lab)
b.axhline(2.0, color=GREY, lw=1, ls=":"); b.text(1.6, 2.25, "meettijd per punt (2 s)", ha="left", fontsize=8, color=INK2)
b.set_yscale("log"); b.set_xlim(0.45, 3.05)
b.set_xlabel("$I / I_{th}$"); b.set_ylabel("95%-stijgtijd [s]")
b.set_title("b  Stijgtijd (rise-deel van het script)", loc="left", fontsize=10, color=INK)
b.legend(fontsize=7.5, loc="upper right", bbox_to_anchor=(1.0, 0.86))

# (c) 2-s protocol: under-reading vs steady state near the knee
c = ax[1, 0]
c.plot(prot["cobolt_FW5_A"]["I"], 100 * (np.array(prot["cobolt_FW5_A"]["dyn"]) / np.array(prot["cobolt_FW5_A"]["stat"]) - 1),
       color=BLUE, label="Cobolt FW6=5, k = kalibratie")
c.plot(prot["superk_FW1_A"]["I"], 100 * (np.array(prot["superk_FW1_A"]["dyn"]) / np.array(prot["superk_FW1_A"]["stat"]) - 1),
       color=BLUE, ls=":", label="SuperK FW6=1, k = kalibratie")
c.plot(var["sigma_esa_cobolt_1060"]["I"], 100 * np.array(var["sigma_esa_cobolt_1060"]["ratio"]),
       color=ORANGE, label="Cobolt FW6=5, k = σ$_{ESA}$ × k")
c.plot(var["sigma_esa_superk_1064"]["I"], 100 * np.array(var["sigma_esa_superk_1064"]["ratio"]),
       color=ORANGE, ls=":", label="SuperK FW6=1, k = σ$_{ESA}$ × k")
c.axhline(0, color="#dddddd", lw=1, zorder=0)
c.set_xscale("log"); c.set_xlim(0.6, 3.2); c.set_ylim(-45, 5)
c.set_xlabel("gerapporteerde intensiteit [kW cm$^{-2}$]")
c.set_ylabel("2-s gemiddelde t.o.v. steady state [%]")
c.set_title("c  Meetprotocol: oplopende sweep, 2 s per punt", loc="left", fontsize=10, color=INK)
c.legend(fontsize=7.5, loc="lower right")

# (d) Cobolt FW6=4 vs FW6=5 near the knee, common background
dd = load(str(data_path("cobolt_1060_scurve.csv")))
k = summ["cobolt_1060"]["k_ensemble"]; b5 = summ["cobolt_1060"]["background_per_FW"]["5"]
wg, we = rates(1060.0, P)
S = ensemble_curve(I_GRID, I_GRID, emission_curve(I_GRID, P, wg, we), XB, WB)
# amplitude from the FW6=5-only part of the global fit (A of the joint fit)
from fit_scurves import fit_k
A = fit_k(S, dd)["A"]
dax = ax[1, 1]
for fw, mk, lab in ((4, "o", "FW6=4 (sweep 1)"), (5, "s", "FW6=5 (sweep 2)")):
    m = (dd["fw6"] == fw) & (dd["I_kWcm2"] > 0.85) & (dd["I_kWcm2"] < 2.3)
    dax.plot(dd["I_kWcm2"][m], dd["net_cps"][m] - b5, mk, ms=5, mfc="none", mec=INK, mew=1, label=lab)
Ip = np.logspace(np.log10(0.85), np.log10(2.3), 300)
Mp = A * np.exp(np.interp(np.log(k * Ip), np.log(I_GRID), np.log(S)))
dax.plot(Ip, Mp, color=BLUE, label="model (steady state)")
dax.plot(Ip, Mp + 30, color=BLUE, ls="--", lw=1.3, label="model + 30 cps")
pB = prot["cobolt_FW4_B"]; IB = np.array(pB["I"]); rB = np.array(pB["dyn"]) / np.array(pB["stat"])
MB = A * np.exp(np.interp(np.log(k * IB), np.log(I_GRID), np.log(S))) * rB
mB = IB < 2.3
dax.plot(IB[mB], MB[mB], "x", color=ORANGE, ms=6, mew=1.5, label="model FW6=4, start na volle intensiteit")
dax.set_xscale("log"); dax.set_yscale("log"); dax.set_ylim(2, 2000)
dax.set_xlabel("gerapporteerde intensiteit [kW cm$^{-2}$]"); dax.set_ylabel("emissie − 1 cps [cps]")
dax.set_title("d  Cobolt: verschil tussen de twee sweeps", loc="left", fontsize=10, color=INK)
dax.legend(fontsize=7.5, loc="upper left")

fig.tight_layout()
fig.savefig(out_path("fig3_overige_scriptdelen.png"), dpi=170)
print("A =", A)
