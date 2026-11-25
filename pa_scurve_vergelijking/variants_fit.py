"""Refit the S-curves with the two 'physical' alternatives to an intensity
calibration error (sigma_ESA x k, or w2 / k) and rerun the 2-s protocol."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts in deze map vindbaar
import numpy as np, json
from dataclasses import replace
from scipy.integrate import solve_ivp
from pa_fit_lib import data_path, out_path, trapz, pm, pf, lee_params, rates, analytical_ith, emission_curve, ensemble_curve, load, spot_weights
from fit_scurves import DATA, I_GRID, XB, WB, fit_k

P0 = pm.build_parameters(8.0, 0.0, 0.0, lambda0_gsa_nm=lee_params(3).lambda0_gsa_nm)
summ = json.load(open(out_path("summary.json")))
data = {n: load(v["file"]) for n, v in DATA.items()}
# coarse spot bins for the time-dependent runs
xb_f, wb_f = spot_weights(nbins=400)
edges = np.logspace(-3, 0, 31); idx = np.digitize(xb_f, edges) - 1
cx, cw = [], []
for i in range(30):
    m = (idx == i) & (wb_f > 0)
    if m.any(): cx.append(np.average(xb_f[m], weights=wb_f[m])); cw.append(wb_f[m].sum())
cx, cw = np.array(cx), np.array(cw)

def protocol_ratio(P, I_seq, kfit, wg, we):
    dyn = np.zeros(len(I_seq)); stat = np.zeros(len(I_seq))
    for x, w in zip(cx, cw):
        I_model = np.asarray(I_seq) * kfit * x
        y = pf._stationary_population(float(I_model[0]), P, wg, we)[0]
        for j, I in enumerate(I_model):
            sol = solve_ivp(pf._make_rhs(float(I), P, wg, we), (0, 2.0), y, method="LSODA",
                            rtol=1e-8, atol=1e-16, dense_output=True)
            t = np.linspace(0, 2.0, 201)
            dyn[j] += w * trapz(sol.sol(t)[2], t) / 2.0
            stat[j] += w * pf._stationary_population(float(I), P, wg, we)[0][2]
            y = sol.y[:, -1]
    return dyn / stat - 1

out = {}
for vname in ("sigma_esa", "w2"):
    for name, cfg in DATA.items():
        kk = summ[name]["k_ensemble"]
        P = (replace(P0, sigma_esa_m2=P0.sigma_esa_m2 * kk) if vname == "sigma_esa"
             else replace(P0, w2=P0.w2 / kk, w2_r=P0.w2_r / kk))
        wg, we = rates(cfg["lam"], P)
        E = emission_curve(I_GRID, P, wg, we)
        r = fit_k(ensemble_curve(I_GRID, I_GRID, E, XB, WB), data[name])
        d = data[name]; fw = 5 if name.startswith("cobolt") else 1
        I_seq = d["I_kWcm2"][d["fw6"] == fw]
        rr = protocol_ratio(P, I_seq, r["k"], wg, we)
        out[f"{vname}_{name}"] = dict(k_refit=r["k"], chi2_red=r["chi2_red"], Ith_model=analytical_ith(P, wg, we),
                                      min_protocol_ratio=float(rr.min()), I=I_seq.tolist(), ratio=rr.tolist())
        print(f"{vname:9s} {name}: k_refit={r['k']:.2f} chi2r={r['chi2_red']:.2f} "
              f"worst 2-s under-reading={rr.min()*100:.0f}% at I={I_seq[np.argmin(rr)]:.3f}", flush=True)
json.dump(out, open(out_path("variants_fit.json"), "w"))
