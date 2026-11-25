import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts in deze map vindbaar
import numpy as np, json
from dataclasses import replace
from pa_fit_lib import data_path, out_path, trapz, pm, pf, lee_params, rates, analytical_ith
P = pm.build_parameters(8.0, 0.0, 0.0, lambda0_gsa_nm=lee_params(3).lambda0_gsa_nm)
k = json.load(open(out_path("summary.json")))["cobolt_1060"]["k_ensemble"]
variants = {
 "kalibratie (I_echt = k*I_gerapp)": (P, 1.0),
 "sigma_ESA x k": (replace(P, sigma_esa_m2=P.sigma_esa_m2*k), None),
 "w2 / k (langere 3F4-levensduur)": (replace(P, w2=P.w2/k, w2_r=P.w2_r/k), None),
}
r = np.array([0.5,0.7,0.85,0.9,0.95,0.98,1.0,1.02,1.05,1.1,1.2,1.5,2.0,3.0])
out = {"ratio": r.tolist()}
for name,(p,_) in variants.items():
    wg,we = rates(1060.,p); ith = analytical_ith(p,wg,we)
    t = np.array([pf.rise_time_95_fast(x*ith,p,wg,we) for x in r])
    out[name] = dict(ith_model=ith, t95=t.tolist())
    print(f"{name:38s} Ith={ith:.3f}  t95 peak={t.max():.2f}s  at 0.7Ith={t[1]:.3f}s at 2Ith={t[-2]:.3f}s")
json.dump(out, open(out_path("rise_variants.json"), "w"), indent=1)
