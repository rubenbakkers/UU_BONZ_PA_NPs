# -*- coding: utf-8 -*-
"""Fit pa_fononband model S-curves to the measurements in measurements.py.

model counts = A * M(k * I_reported) + b_FW
  M : single-NP emission, or emission summed over the measured spot
  k : intensity scale (absorbs calibration error AND differences in
      w2 / sigma_ESA between these NPs and Lee's 8% samples)
  A, b_FW : linear (weighted least squares), one background per filter-wheel setting
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # make the scripts in this folder importable

import json
import numpy as np
from pa_fit_lib import (out_path, lee_params, rates, analytical_ith, emission_curve,
                        spot_weights, ensemble_curve, load, measurements)

REL_SCATTER = 0.10          # multiplicative scatter between repeats (~10 %)
I_GRID = np.logspace(-2, 3, 260)   # model intensity grid [kW/cm^2]
XB, WB = spot_weights()


def model_curve(p, lam, fwhm=0.0, ensemble=False):
    wg, we = rates(lam, p, fwhm)
    E = emission_curve(I_GRID, p, wg, we)
    if ensemble:
        E = ensemble_curve(I_GRID, I_GRID, E, XB, WB)
    return E, analytical_ith(p, wg, we)


def linear_fit(Mvals, y, sig, fw):
    """Weighted LSQ for A and one background per FW setting."""
    levels = np.unique(fw)
    X = np.column_stack([Mvals] + [(fw == l).astype(float) for l in levels])
    W = 1.0 / sig
    coef, *_ = np.linalg.lstsq(X * W[:, None], y * W, rcond=None)
    model = X @ coef
    chi2 = np.sum(((y - model) / sig) ** 2)
    return coef, model, chi2


def fit_k(E, d, k_grid=np.logspace(-1.5, 2.5, 401)):
    I, y, noise, fw = d["I_kWcm2"], d["net_cps"], d["noise_cps"], d["fw6"]
    sig = np.sqrt(noise ** 2 + (REL_SCATTER * np.clip(y, 0, None)) ** 2)
    logI, logE = np.log(I_GRID), np.log(np.clip(E, 1e-300, None))
    best = None
    for k in k_grid:
        M = np.exp(np.interp(np.log(k * I), logI, logE))
        coef, model, chi2 = linear_fit(M, y, sig, fw)
        if coef[0] <= 0:
            continue
        if best is None or chi2 < best[0]:
            best = (chi2, k, coef, model)
    if best is None:
        raise RuntimeError("fit_k: no intensity scale k in the grid gives a positive amplitude A; "
                           "does this measurement contain an S-curve?")
    # refine
    chi2, k0, _, _ = best
    for k in k0 * np.logspace(-0.02, 0.02, 81):
        M = np.exp(np.interp(np.log(k * I), logI, logE))
        coef, model, c2 = linear_fit(M, y, sig, fw)
        if coef[0] > 0 and c2 < best[0]:
            best = (c2, k, coef, model)
    chi2, k, coef, model = best
    return dict(chi2=chi2, chi2_red=chi2 / (len(y) - 2 - len(np.unique(fw))),
                k=k, A=coef[0], b=coef[1:].tolist(), model=model)


if __name__ == "__main__":
    DATA = measurements()
    data = {n: load(m["file"]) for n, m in DATA.items()}
    results = {}
    for sid in (3, 4, 5):
        p = lee_params(sid)
        for ens in (False, True):
            for name, cfg in DATA.items():
                E, ith = model_curve(p, cfg["wavelength_nm"], cfg["fwhm_nm"], ens)
                r = fit_k(E, data[name])
                key = f"S{sid}_{'ens' if ens else 'np'}_{name}"
                results[key] = dict(ith_model=ith, k=r["k"], ith_reported=ith / r["k"],
                                    chi2_red=r["chi2_red"])
                print(f"{key:28s} Ith_model={ith:7.2f}  k={r['k']:6.2f}  "
                      f"Ith(reported scale)={ith / r['k']:.3f}  chi2_red={r['chi2_red']:.2f}")
    json.dump(results, open(out_path("results_lee_params.json"), "w"), indent=1)
