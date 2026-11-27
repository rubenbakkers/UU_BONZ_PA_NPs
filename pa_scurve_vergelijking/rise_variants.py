"""95% rise time vs I/Ith for the three explanations of the factor k: an
intensity calibration error, a larger sigma_ESA, or a slower 3F4 decay (w2 / k).
One set per measurement (its own wavelength and k from summary.json)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # make the scripts in this folder importable
import numpy as np, json
from dataclasses import replace
from pa_fit_lib import out_path, pm, pf, lee_params, rates, analytical_ith, measurements, read_summary
P = pm.build_parameters(8.0, 0.0, 0.0, lambda0_gsa_nm=lee_params(3).lambda0_gsa_nm)
DATA = measurements()
summ = read_summary(DATA)
r = np.array([0.5,0.7,0.85,0.9,0.95,0.98,1.0,1.02,1.05,1.1,1.2,1.5,2.0,3.0])
out = {"ratio": r.tolist()}
for mname, cfg in DATA.items():
    k = summ[mname]["k_ensemble"]
    variants = {
     "calibration (I_true = k*I_reported)": P,
     "sigma_ESA x k": replace(P, sigma_esa_m2=P.sigma_esa_m2*k),
     "w2 / k (longer 3F4 lifetime)": replace(P, w2=P.w2/k, w2_r=P.w2_r/k),
    }
    out[mname] = {}
    for name, p in variants.items():
        wg,we = rates(cfg["wavelength_nm"], p, cfg["fwhm_nm"]); ith = analytical_ith(p,wg,we)
        t = np.array([pf.rise_time_95_fast(x*ith,p,wg,we) for x in r])
        out[mname][name] = dict(ith_model=ith, t95=t.tolist())
        print(f"{mname:14s} {name:38s} Ith={ith:.3f}  t95 peak={t.max():.2f}s  "
              f"at 0.7Ith={t[1]:.3f}s at 2Ith={t[-2]:.3f}s")
json.dump(out, open(out_path("rise_variants.json"), "w"), indent=1)
