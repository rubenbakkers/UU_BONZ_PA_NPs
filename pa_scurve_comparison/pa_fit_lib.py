# -*- coding: utf-8 -*-
"""Helpers to compare measured S-curves (BdJ1, 24-11-2026) with the
pa_fononband / pa_model rate equations (Lee et al. 2021 SI DREs).
Steady states come from pa_fononband._stationary_population (LSODA from
the ground state), with the phonon-band GSA/ESA factors of pa_fononband.
"""
from __future__ import annotations

import sys
from dataclasses import replace

import numpy as np

# numpy < 2.0 has no np.trapezoid
trapz = getattr(np, 'trapezoid', None) or np.trapz

from pathlib import Path
import os

# All paths relative to this folder, so it does not matter from which folder
# (Spyder, terminal) the scripts are run.
try:
    HERE = Path(__file__).resolve().parent
except NameError:                      # e.g. pasted into a console
    HERE = Path.cwd()
DATA_DIR = HERE / "data"
OUT_DIR = HERE / "results"
OUT_DIR.mkdir(exist_ok=True)
# Model version (folder with pa_model.py and pa_fononband.py), in this order:
#   1. the environment variable PA_MODEL_DIR;
#   2. ../Script      (local BONZ folder: the current version, e.g. S(3H6->3H5) = 0.112);
#   3. ../fonon_model (GitHub repo UU_BONZ_PA_NPs: the same model code);
#   4. model/         (old bundled copy of 7-11-2026, S(3H6->3H5) = 0.05).
def _has_model(folder):
    return (folder / "pa_model.py").exists() and (folder / "pa_fononband.py").exists()


_CANDIDATES = [HERE.parent / "Script", HERE.parent / "fonon_model", HERE / "model"]
if os.environ.get("PA_MODEL_DIR"):
    MODEL_DIR = Path(os.environ["PA_MODEL_DIR"])
else:
    MODEL_DIR = next((c for c in _CANDIDATES if _has_model(c)), None)
    if MODEL_DIR is None:
        raise ImportError("No pa_model.py + pa_fononband.py found in "
                          + ", ".join(str(c) for c in _CANDIDATES)
                          + ". Set PA_MODEL_DIR to the folder with the model.")
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))
if str(HERE) not in sys.path:
    sys.path.insert(1, str(HERE))

import pa_model as pm          # noqa: E402
import pa_fononband as pf      # noqa: E402

# pa_model and pa_fononband must come from the same folder (otherwise they
# were imported earlier from another folder, e.g. in the same Spyder console).
for _module in (pm, pf):
    if Path(_module.__file__).resolve().parent != MODEL_DIR.resolve():
        raise ImportError(f"{_module.__name__} comes from {Path(_module.__file__).parent}, "
                          f"not from {MODEL_DIR}. Restart the console (kernel) and run again.")
MODEL_INFO = {
    "model_dir": str(MODEL_DIR),
    "absorption_model": pf.ABSORPTION_MODEL,
    "huang_rhys_3H6_3H5": pf.GSA_H5_HUANG_RHYS,
    "stark_sigma_cm1": pf.STARK_SIGMA_CM1,
    "phonon_bands": pf.PHONON_DESCRIPTION,
}
print(f"[pa_fit_lib] model from {MODEL_DIR} | S(3H6->3H5) = {pf.GSA_H5_HUANG_RHYS:g} | "
      f"sigma = {pf.STARK_SIGMA_CM1:g} cm^-1", flush=True)


# ---------------------------------------------------------------------------
# Parameters and GSA/ESA rate coefficients
# ---------------------------------------------------------------------------

def lee_params(sample_id=3):
    return pm.parameters_for_sample(sample_id,
                                    lambda0_gsa_nm=pf.GSA_BACKGROUND_LAMBDA0_NM)


def rates(wavelength_nm, p, fwhm_nm=0.0, n_lines=41):
    """(w_gsa, w_esa) in s^-1/(kW cm^-2) from the phonon-band spectra.

    fwhm_nm > 0 averages over a Gaussian laser spectrum (e.g. SuperK
    after its filter); 0 = single-frequency laser.
    """
    with pf._lineshape("stark_gaussian"):
        if fwhm_nm <= 0:
            return pm.gsa_esa_rate_coefficients(float(wavelength_nm), p)
        sig = fwhm_nm / 2.3548
        lam = wavelength_nm + np.linspace(-3 * sig, 3 * sig, n_lines)
        wt = np.exp(-0.5 * ((lam - wavelength_nm) / sig) ** 2)
        wt /= wt.sum()
        wg = sum(w * pm.gsa_esa_rate_coefficients(float(l), p)[0] for l, w in zip(lam, wt))
        we = sum(w * pm.gsa_esa_rate_coefficients(float(l), p)[1] for l, w in zip(lam, wt))
        return wg, we


def analytical_ith(p, wg, we):
    a32 = p.b32 * p.w3_r + p.w3_nr
    return pf.analytical_threshold_intensity(wg, we, p.w2, p.w3, p.s31, a32)


# ---------------------------------------------------------------------------
# Steady state: Ruben's own LSODA routine (integration from the ground state,
# i.e. the branch reached on an upward intensity sweep)
# ---------------------------------------------------------------------------

def steady_state(I, p, wg, we):
    return pf._stationary_population(float(I), p, wg, we)[0]


def emission_curve(I_values, p, wg, we):
    """800-nm emission (radiative 3H4 -> 3H6 rate) per ion, single NP."""
    out = np.empty(len(I_values))
    for j, I in enumerate(I_values):
        out[j] = pm.emission_from_population(max(float(steady_state(I, p, wg, we)[2]), 0.0), p)
    return out


# ---------------------------------------------------------------------------
# Spatial averaging over the measured spot (ensemble of NPs in the focus)
# ---------------------------------------------------------------------------

def spot_weights(core_fwhm=(0.7016, 1.1721), halo_fwhm=1.6988, halo_power=0.7525,
                 n=401, extent_um=4.0, nbins=400):
    """Area distribution of relative intensity x = I(r)/I_peak.

    Returns (x_bins, area_um2_per_bin). Profile = elliptic Gaussian core +
    round Gaussian halo (fit from 04_spot.py), normalised to peak 1.
    """
    xs = np.linspace(-extent_um, extent_um, n)
    X, Y = np.meshgrid(xs, xs)
    k = 4 * np.log(2)
    core = np.exp(-k * (X / core_fwhm[0]) ** 2 - k * (Y / core_fwhm[1]) ** 2)
    core /= core.sum()
    halo = np.exp(-k * (X ** 2 + Y ** 2) / halo_fwhm ** 2)
    halo /= halo.sum()
    g = (1 - halo_power) * core + halo_power * halo
    g /= g.max()
    dA = (xs[1] - xs[0]) ** 2
    edges = np.logspace(-4, 0, nbins + 1)
    hist, _ = np.histogram(g.ravel(), bins=edges)
    centres = np.sqrt(edges[1:] * edges[:-1])
    return centres, hist * dA


def ensemble_curve(I_peak, single_I_grid, single_E, xb, wb):
    """Sum of single-NP emission over the spot (uniform NP density)."""
    logI = np.log(single_I_grid)
    logE = np.log(np.clip(single_E, 1e-300, None))
    out = np.empty(len(I_peak))
    for j, I0 in enumerate(I_peak):
        Iloc = np.clip(I0 * xb, single_I_grid[0], single_I_grid[-1])
        E = np.exp(np.interp(np.log(Iloc), logI, logE))
        out[j] = np.sum(E * wb)
    return out


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def data_path(name):
    path = DATA_DIR / name
    if not path.exists():
        # The measurement data are not in the (public) GitHub repo; see README.md.
        raise FileNotFoundError(f"Measurement file missing: {path}. Put the CSV files from the "
                                "Drive (BdJ1, 24-11-2026) in the data/ folder (see README.md).")
    return path


def out_path(name):
    return OUT_DIR / name


def load(path):
    p = Path(path)
    if not p.is_absolute():
        p = DATA_DIR / p.name
    d = np.genfromtxt(p, delimiter=",", names=True)
    o = np.argsort(d["I_kWcm2"])
    return d[o]


def local_slope_binned(I, y, noise, width_dex=0.06, min_snr=5.0):
    """Log-log slope from a moving linear fit (window width_dex in log I),
    using only points with y > min_snr * noise."""
    ok = y > min_snr * noise
    x = np.log10(I[ok]); ly = np.log10(y[ok])
    xc, sl = [], []
    for c in np.arange(x.min() + width_dex / 2, x.max() - width_dex / 2, width_dex / 4):
        m = np.abs(x - c) <= width_dex / 2
        if m.sum() >= 4:
            a = np.polyfit(x[m], ly[m], 1)[0]
            xc.append(10 ** c); sl.append(a)
    return np.array(xc), np.array(sl)
