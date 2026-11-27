# -*- coding: utf-8 -*-
"""Simulate the actual measurement protocol (07_emission.py): settings measured
one after another, 2 s dwell each, counts averaged over the dwell, no settling
time assumed. Compare with the steady-state S-curve used in the fit.

Per measurement in measurements.py:
Scenario A: each sweep starts in steady state at its first setting; simulated
            for the sweep through the knee (knee_fw, from summary.json) and for
            both sweeps of compare_sweeps.
Scenario B: the first sweep of compare_sweeps (e.g. Cobolt FW6=4) starts right
            after exposure at the highest intensity of the measurement (e.g.
            sample alignment at full power), i.e. in the avalanched state.
Order within a sweep: ascending reported intensity (order of the saved files).
Output keys: "<measurement>_FW<fw6>_<scenario>", e.g. "cobolt_1060_FW5_A".
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # make the scripts in this folder importable
import json
import numpy as np
from scipy.integrate import solve_ivp
from pa_fit_lib import (out_path, trapz, pm, pf, lee_params, rates, load, spot_weights,
                        measurements, read_summary)

DWELL = 2.0
P = pm.build_parameters(8.0, 0.0, 0.0, lambda0_gsa_nm=lee_params(3).lambda0_gsa_nm)
DATA = measurements()
summ = read_summary(DATA)

# coarse spot distribution (relative intensity x, area weight)
xb_f, wb_f = spot_weights(nbins=400)
edges = np.logspace(-3, 0, 41)
idx = np.digitize(xb_f, edges) - 1
XB = np.array([np.average(xb_f[idx == i], weights=wb_f[idx == i]) if np.any((idx == i) & (wb_f > 0)) else np.nan
               for i in range(40)])
WB = np.array([wb_f[idx == i].sum() for i in range(40)])
ok = np.isfinite(XB) & (WB > 0)
XB, WB = XB[ok], WB[ok]


def run_sequence(I_model_seq, wg, we, y0):
    """Mean emission over each dwell for one NP experiencing I_model_seq."""
    y = np.array(y0, float)
    out = np.empty(len(I_model_seq))
    for j, I in enumerate(I_model_seq):
        rhs = pf._make_rhs(float(I), P, wg, we)
        sol = solve_ivp(rhs, (0, DWELL), y, method="LSODA", rtol=1e-8, atol=1e-16,
                        dense_output=True)
        t = np.linspace(0, DWELL, 201)
        n2 = sol.sol(t)[2]
        out[j] = pm.emission_from_population(trapz(n2, t) / DWELL, P)
        y = sol.y[:, -1]
    return out


def steady(I, wg, we):
    return pf._stationary_population(float(I), P, wg, we)[0]


def ensemble_protocol(I_rep_seq, k, wg, we, start_I_rep):
    tot = np.zeros(len(I_rep_seq))
    stat = np.zeros(len(I_rep_seq))
    for x, w in zip(XB, WB):
        y0 = steady(start_I_rep * k * x, wg, we)
        tot += w * run_sequence(np.asarray(I_rep_seq) * k * x, wg, we, y0)
        stat += w * np.array([pm.emission_from_population(steady(I * k * x, wg, we)[2], P)
                              for I in I_rep_seq])
    return tot, stat


res = {}
for name, cfg in DATA.items():
    d = load(cfg["file"])
    k = summ[name]["k_ensemble"]
    wg, we = rates(cfg["wavelength_nm"], P, cfg["fwhm_nm"])
    compare = [int(f) for f in (cfg["compare_sweeps"] or ())]
    sweeps_A = sorted(set(compare) | {int(summ[name]["knee_fw"])})
    runs = [(fw, "A") for fw in sweeps_A] + ([(compare[0], "B")] if compare else [])
    for fw, scen in runs:
        m = d["fw6"] == fw
        if not m.any():
            raise ValueError(f"{name}: no points with fw6={fw} (knee_fw / compare_sweeps in "
                             "measurements.py).")
        I_seq = d["I_kWcm2"][m]
        start = I_seq[0] if scen == "A" else d["I_kWcm2"].max()
        dyn, stat = ensemble_protocol(I_seq, k, wg, we, start)
        res[f"{name}_FW{fw}_{scen}"] = dict(I=I_seq.tolist(), dyn=dyn.tolist(), stat=stat.tolist())
        rel = dyn / stat - 1
        j = int(np.argmin(rel))
        print(f"{name} FW6={fw} scenario {scen}: 2-s average vs. steady state at most "
              f"{rel[j] * 100:+.1f}% (at I = {I_seq[j]:.3f} kW/cm2)", flush=True)
json.dump(res, open(out_path("protocol_sim.json"), "w"))
