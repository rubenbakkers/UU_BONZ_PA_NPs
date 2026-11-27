# -*- coding: utf-8 -*-
"""Vary the particle parameters (different NPs than Lee et al.) and refit.

Free per curve: k (intensity scale), A, b_FW. Scanned (shared): GSA/ESA ratio
(sigma_GSA multiplier), Tm concentration c (-> s31, Q22, Q23) and the
non-radiative rates (w2_nr, w3_nr scaled together along Lee's 8 % series).
All measurements in measurements.py are fitted; the joint chi2 is their sum.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # make the scripts in this folder importable
import itertools, json
import numpy as np
from dataclasses import replace
from pa_fit_lib import (out_path, lee_params, rates, analytical_ith, emission_curve, ensemble_curve, load,
                        measurements, format_parameters)
from fit_scurves import I_GRID, XB, WB, fit_k
import pa_model as pm

DATA = measurements()
data = {n: load(m["file"]) for n, m in DATA.items()}

GSA_SCALE = [0.1, 0.3, 1.0, 3.0, 10.0]
CONC = [4.0, 8.0, 12.0, 20.0]
NR = [(0.0, 0.0), (40.7, 87.3), (150.0, 300.0), (512.0, 1030.0)]


def params(gs, c, nr):
    p = pm.build_parameters(c, w2_nr=nr[0], w3_nr=nr[1],
                            lambda0_gsa_nm=lee_params(3).lambda0_gsa_nm)
    return replace(p, sigma_gsa_m2=p.sigma_gsa_m2 * gs)


rows = []
for gs, c, nr in itertools.product(GSA_SCALE, CONC, NR):
    p = params(gs, c, nr)
    row = dict(gsa_scale=gs, c=c, w2_nr=nr[0], w3_nr=nr[1], chi2=0.0, n=0)
    for name, cfg in DATA.items():
        wg, we = rates(cfg["wavelength_nm"], p, cfg["fwhm_nm"])
        E = emission_curve(I_GRID, p, wg, we)
        S = ensemble_curve(I_GRID, I_GRID, E, XB, WB)
        r = fit_k(S, data[name])
        ith = analytical_ith(p, wg, we)
        row[f"k_{name}"] = r["k"]
        row[f"ith_rep_{name}"] = ith / r["k"]
        row[f"chi2red_{name}"] = r["chi2_red"]
        row["chi2"] += r["chi2"]
        row["n"] += len(data[name])
    rows.append(row)
    print(f"gs={gs:5.1f} c={c:4.0f} nr={nr}  chi2red: "
          + " / ".join(f"{row[f'chi2red_{n}']:.2f}" for n in DATA)
          + "  k: " + " / ".join(f"{row[f'k_{n}']:.2f}" for n in DATA), flush=True)

json.dump(rows, open(out_path("scan_shape.json"), "w"), indent=1)
best = sorted(rows, key=lambda r: r["chi2"])[:8]
print("\nBEST (joint chi2):")
for r in best:
    print(r)
b0 = best[0]
print(f"\nParameters of the best combination (sigma_GSA x {b0['gsa_scale']:g}, c = {b0['c']:g} %, "
      f"w2_nr = {b0['w2_nr']:g}, w3_nr = {b0['w3_nr']:g} s^-1; joint chi2 = {b0['chi2']:.1f}):")
print(format_parameters(params(b0["gsa_scale"], b0["c"], (b0["w2_nr"], b0["w3_nr"]))))
for name in DATA:
    print(f"    {name}: k = {b0[f'k_{name}']:.4g}, I_th/k = {b0[f'ith_rep_{name}']:.4g} kW/cm^2, "
          f"chi2_red = {b0[f'chi2red_{name}']:.3f}")
