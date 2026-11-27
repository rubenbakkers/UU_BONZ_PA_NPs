# -*- coding: utf-8 -*-
"""Fig 3: other parts of pa_fononband vs. the measurements in measurements.py.

a  emission spectrum: the first measurement with a spectrum_file
b  rise time: REFERENCE (measurements.py)
c  2-s protocol: the sweep through the knee of every measurement
d  two sweeps compared: the first measurement with compare_sweeps
Run make_figures.py, rise_variants.py, protocol_sim.py, variants_fit.py and
emission_spec.py first (run_all.py does this in the right order).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # make the scripts in this folder importable
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pa_fit_lib import (out_path, pm, lee_params, rates, emission_curve, ensemble_curve, load,
                        measurements, reference_name, read_summary)
from fit_scurves import I_GRID, XB, WB, fit_k
from emission_spec import load_spectrum, model_spectrum, fit_emission
from plotting import BLUE, ORANGE, AQUA, GREY, INK, INK2, LINESTYLES, apply_style

MODEL_OFFSET_CPS = 30.0     # fig. 3d: dashed line = model + this constant offset

apply_style(9.5)
DATA = measurements()
REF = reference_name(DATA)
summ = read_summary(DATA)


def read_results(file, keys, script):
    """results/<file>, checked to contain the given keys."""
    path = out_path(file)
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run {script} first.")
    res = json.load(open(path))
    missing = [k for k in keys if k not in res]
    if missing:
        raise KeyError(f"{file} has no results for {missing}: run {script} again "
                       "(after changing measurements.py).")
    return res


rise = read_results("rise_variants.json", [REF], "rise_variants.py")
prot = read_results("protocol_sim.json", [f"{n}_FW{summ[n]['knee_fw']}_A" for n in DATA], "protocol_sim.py")
var = read_results("variants_fit.json", [f"sigma_esa_{n}" for n in DATA], "variants_fit.py")
P = pm.build_parameters(8.0, 0.0, 0.0, lambda0_gsa_nm=lee_params(3).lambda0_gsa_nm)

fig, ax = plt.subplots(2, 2, figsize=(10.5, 7.6))


def empty_panel(a, title, text):
    a.set_title(title, loc="left", fontsize=10, color=INK)
    a.text(0.5, 0.5, text, transform=a.transAxes, ha="center", va="center", fontsize=8.5, color=INK2)
    a.set_axis_off()


# (a) emission spectrum vs 3H4 -> 3H6 Stark lines; fit from emission_spec.py
a = ax[0, 0]
with_spectrum = [n for n, m in DATA.items() if m["spectrum_file"]]
if with_spectrum:
    sm = DATA[with_spectrum[0]]
    em_path = out_path("emission_spec.json")
    em_all = json.load(open(em_path)) if os.path.exists(em_path) else {}
    em = em_all.get(sm["name"]) if isinstance(em_all.get(sm["name"]), dict) else fit_emission(sm)
    lam, y = load_spectrum(sm)
    lg = np.linspace(775, 865, 600)
    y_smooth = np.convolve(y, np.ones(5) / 5, mode="same")
    a.plot(lam, y_smooth, color=INK, lw=1.2, label=f"measured ({sm['label']}, 5-px average)")
    curves_a = []
    for s, ls, lab in ((0.0, "--", "model 3H4→3H6 (Stark levels of the script, at 0 nm)"),
                       (em["shift_nm"], "-", f"model, shifted by {em['shift_nm']:.1f} nm (fit, "
                                             f"σ = {em['sigma_cm1']:.0f} cm$^{{-1}}$)")):
        m = model_spectrum(lg, s, em["sigma_cm1"], em["amplitude"], em["background"])
        curves_a.append(m)
        a.plot(lg, m, color=BLUE, ls=ls, lw=1.6, label=lab)
    top = max(y_smooth.max(), max(c.max() for c in curves_a))
    pk_model, pk_data = em["model_peak_nm_unshifted"], em["data_peak_nm"]
    a.axvline(pk_model, color=GREY, lw=0.8); a.axvline(pk_data, color=INK2, lw=0.8)
    a.text(pk_model - 0.9, 1.1 * top, f"model peak\n{pk_model:.0f} nm", ha="right", fontsize=8, color=INK2)
    a.text(pk_data + 1.0, 1.1 * top, f"measured peak\n{pk_data:.0f} nm", ha="left", fontsize=8, color=INK2)
    a.set_xlim(775, 865); a.set_ylim(-0.1 * top, 1.85 * top)
    a.set_xlabel("wavelength [nm]"); a.set_ylabel("cts s$^{-1}$ pixel$^{-1}$")
    a.set_title("a  Emission spectrum vs. Stark levels", loc="left", fontsize=10, color=INK)
    a.legend(fontsize=7.5, loc="upper left")
else:
    empty_panel(a, "a  Emission spectrum vs. Stark levels",
                "no measurement in measurements.py\nhas a spectrum_file")

# (b) rise time vs I/Ith for the three explanations of the factor k (reference measurement)
b = ax[0, 1]
r = np.array(rise["ratio"])
for (name, col, ls, lab) in (("calibration (I_true = k*I_reported)", BLUE, "-", "factor k = calibration"),
                             ("sigma_ESA x k", ORANGE, "--", "factor k = σ$_{ESA}$ × k"),
                             ("w2 / k (longer 3F4 lifetime)", AQUA, "-.", "factor k = w$_2$ / k (longer 3F4 lifetime)")):
    b.plot(r, rise[REF][name]["t95"], color=col, ls=ls, marker="o", ms=3.5, label=lab)
b.axhline(2.0, color=GREY, lw=1, ls=":"); b.text(1.6, 2.25, "dwell time per point (2 s)", ha="left", fontsize=8, color=INK2)
b.set_yscale("log"); b.set_xlim(0.45, 3.05)
b.set_xlabel("$I / I_{th}$"); b.set_ylabel("95% rise time [s]")
b.set_title(f"b  Rise time (rise part of the script), {DATA[REF]['label']}", loc="left", fontsize=10, color=INK)
b.legend(fontsize=7.5, loc="upper right", bbox_to_anchor=(1.0, 0.86))

# (c) 2-s protocol: under-reading vs steady state near the knee (sweep through the knee)
c = ax[1, 0]
knees = [summ[n]["Ith_on_reported_scale"] for n in DATA]
x_lo, x_hi = min(knees) / 1.5, max(knees) * 3.4
lowest = 0.0
# colour = explanation of k, line style = measurement (sweep through the knee)
handles = [Line2D([], [], color=BLUE, label="k = calibration"),
           Line2D([], [], color=ORANGE, label="k = σ$_{ESA}$ × k")]
for i, (name, cfg) in enumerate(DATA.items()):
    fw = summ[name]["knee_fw"]
    ls = LINESTYLES[i % len(LINESTYLES)]
    pr = prot[f"{name}_FW{fw}_A"]
    v = var[f"sigma_esa_{name}"]
    for color, I, pct in ((BLUE, np.array(pr["I"]), 100 * (np.array(pr["dyn"]) / np.array(pr["stat"]) - 1)),
                          (ORANGE, np.array(v["I"]), 100 * np.array(v["ratio"]))):
        c.plot(I, pct, color=color, ls=ls)
        shown = (I >= x_lo) & (I <= x_hi)
        if shown.any():
            lowest = min(lowest, float(pct[shown].min()))
    handles.append(Line2D([], [], color=INK2, ls=ls, lw=1.6, label=f"{cfg['label']}, FW6={fw}"))
c.axhline(0, color="#dddddd", lw=1, zorder=0)
c.set_xscale("log"); c.set_xlim(x_lo, x_hi); c.set_ylim(min(-10.0, 5 * np.floor(1.1 * lowest / 5)), 5)
c.set_xlabel("reported intensity [kW cm$^{-2}$]")
c.set_ylabel("2-s average relative to steady state [%]")
c.set_title("c  Measurement protocol: upward sweep, 2 s per point", loc="left", fontsize=10, color=INK)
c.legend(handles=handles, fontsize=7.5, loc="lower right")

# (d) two sweeps of one measurement near the knee, common background
dax = ax[1, 1]
with_compare = [n for n, m in DATA.items() if m["compare_sweeps"]]
if with_compare:
    name = with_compare[0]
    cfg = DATA[name]
    fw_a, fw_b = (int(f) for f in cfg["compare_sweeps"])
    dd = load(cfg["file"])
    k = summ[name]["k_ensemble"]
    bg = summ[name]["background_per_FW"][str(fw_b)]
    wg, we = rates(cfg["wavelength_nm"], P, cfg["fwhm_nm"])
    S = ensemble_curve(I_GRID, I_GRID, emission_curve(I_GRID, P, wg, we), XB, WB)
    A = fit_k(S, dd)["A"]                # amplitude of the fit of the whole curve
    if cfg["compare_window_kWcm2"]:
        I_lo, I_hi = cfg["compare_window_kWcm2"]
    else:                                # where both sweeps have points
        Ia, Ib = dd["I_kWcm2"][dd["fw6"] == fw_a], dd["I_kWcm2"][dd["fw6"] == fw_b]
        I_lo, I_hi = max(Ia.min(), Ib.min()), min(Ia.max(), Ib.max())
        if I_lo >= I_hi:
            I_lo, I_hi = min(Ia.min(), Ib.min()), max(Ia.max(), Ib.max())
    shown = []
    for i, (fw, mk) in enumerate(((fw_a, "o"), (fw_b, "s"))):
        m = (dd["fw6"] == fw) & (dd["I_kWcm2"] > I_lo) & (dd["I_kWcm2"] < I_hi)
        dax.plot(dd["I_kWcm2"][m], dd["net_cps"][m] - bg, mk, ms=5, mfc="none", mec=INK, mew=1,
                 label=f"FW6={fw} (sweep {i + 1})")
        shown.append(dd["net_cps"][m] - bg)
    Ip = np.logspace(np.log10(I_lo), np.log10(I_hi), 300)
    Mp = A * np.exp(np.interp(np.log(k * Ip), np.log(I_GRID), np.log(S)))
    shown = np.concatenate(shown + [Mp])
    shown = shown[shown > 0]
    dax.plot(Ip, Mp, color=BLUE, label="model (steady state)")
    dax.plot(Ip, Mp + MODEL_OFFSET_CPS, color=BLUE, ls="--", lw=1.3, label=f"model + {MODEL_OFFSET_CPS:g} cps")
    key_B = f"{name}_FW{fw_a}_B"
    if key_B in prot:
        pB = prot[key_B]; IB = np.array(pB["I"]); rB = np.array(pB["dyn"]) / np.array(pB["stat"])
        MB = A * np.exp(np.interp(np.log(k * IB), np.log(I_GRID), np.log(S))) * rB
        mB = (IB > I_lo) & (IB < I_hi)
        dax.plot(IB[mB], MB[mB], "x", color=ORANGE, ms=6, mew=1.5,
                 label=f"model FW6={fw_a}, start after full intensity")
    dax.set_xscale("log"); dax.set_yscale("log")
    dax.set_ylim(max(shown.min() / 2.0, 0.5), shown.max() * 2.0)
    dax.set_xlabel("reported intensity [kW cm$^{-2}$]")
    dax.set_ylabel(f"emission − FW6={fw_b} background [cps]")
    dax.set_title(f"d  {cfg['label']}: difference between the two sweeps", loc="left", fontsize=10, color=INK)
    dax.legend(fontsize=7.5, loc="upper left")
    print("A =", A)
else:
    empty_panel(dax, "d  Difference between two sweeps",
                "no measurement in measurements.py\nhas compare_sweeps")

fig.tight_layout()
fig.savefig(out_path("fig3_other_script_parts.png"), dpi=170)
