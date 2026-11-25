# -*- coding: utf-8 -*-
"""Simulate the actual measurement protocol (07_emission.py): settings measured
one after another, 2 s dwell each, counts averaged over the dwell, no settling
time assumed. Compare with the steady-state S-curve used in the fit.

Scenario A: each sweep starts in steady state at its first setting.
Scenario B (Cobolt FW6=4): the sweep starts right after exposure at the highest
            intensity (e.g. sample alignment at full power), i.e. in the
            avalanched state.
Order within a sweep: ascending reported intensity (order of the saved files).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts in deze map vindbaar
import json
import numpy as np
from scipy.integrate import solve_ivp
from pa_fit_lib import data_path, out_path, trapz, pm, pf, lee_params, rates, load, spot_weights

DWELL = 2.0
P = pm.build_parameters(8.0, 0.0, 0.0, lambda0_gsa_nm=lee_params(3).lambda0_gsa_nm)
summ = json.load(open(out_path("summary.json")))

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
# ---- Cobolt (1060 nm) ----
d = load(str(data_path("cobolt_1060_scurve.csv")))
k, A = summ["cobolt_1060"]["k_ensemble"], None
wg, we = rates(1060.0, P)
for fw in (4, 5):
    m = d["fw6"] == fw
    I_seq = d["I_kWcm2"][m]
    for scen, start in (("A", I_seq[0]), ("B", d["I_kWcm2"].max())):
        if fw == 5 and scen == "B":
            continue
        dyn, stat = ensemble_protocol(I_seq, k, wg, we, start)
        res[f"cobolt_FW{fw}_{scen}"] = dict(I=I_seq.tolist(), dyn=dyn.tolist(), stat=stat.tolist())
        print("cobolt", fw, scen, "max dyn/stat-1 near knee:",
              np.round((dyn / stat - 1)[:12], 3))
# ---- SuperK (1064 nm), knee only in FW6=1 ----
d2 = load(str(data_path("superk_1064_scurve.csv")))
k2 = summ["superk_1064"]["k_ensemble"]
wg2, we2 = rates(1064.0, P)
m = d2["fw6"] == 1
I_seq = d2["I_kWcm2"][m]
dyn, stat = ensemble_protocol(I_seq, k2, wg2, we2, I_seq[0])
res["superk_FW1_A"] = dict(I=I_seq.tolist(), dyn=dyn.tolist(), stat=stat.tolist())
sel = I_seq > 0.95
print("superk FW1 A dyn/stat-1 (I>0.95):", np.round((dyn / stat - 1)[sel][:14], 3))
json.dump(res, open(out_path("protocol_sim.json"), "w"))
