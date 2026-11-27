# -*- coding: utf-8 -*-
"""Helpers to compare measured S-curves (listed in measurements.py) with the
pa_fononband / pa_model rate equations (Lee et al. 2021 SI DREs).
Steady states come from pa_fononband._stationary_population (LSODA from
the ground state), with the phonon-band GSA/ESA factors of pa_fononband.
"""
from __future__ import annotations

import dataclasses
import json
import runpy
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
# This folder first, then the model folder: ../Script also holds older copies of
# fit_scurves.py, emission_spec.py etc., which must not shadow the ones here.
for _folder in (str(MODEL_DIR), str(HERE)):
    while _folder in sys.path:
        sys.path.remove(_folder)
    sys.path.insert(0, _folder)

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


_PARAM_LINES = (
    ("Tm concentration", ("concentration_percent",), "{concentration_percent:g} %"),
    ("cross sections", ("sigma_gsa_m2", "sigma_esa_m2"),
     "sigma_GSA = {sigma_gsa_m2:.4g} m^2, sigma_ESA = {sigma_esa_m2:.4g} m^2"),
    ("3F4 decay", ("w2", "w2_r", "w2_nr"),
     "w2 = {w2:.4g} s^-1 (radiative {w2_r:.4g}, non-radiative {w2_nr:.4g})"),
    ("3H4 decay", ("w3", "w3_r", "w3_nr"),
     "w3 = {w3:.4g} s^-1 (radiative {w3_r:.4g}, non-radiative {w3_nr:.4g})"),
    ("3H4 -> 3F4 branching", ("b32",), "b32 = {b32:.4g} (a32 = b32 w3_r + w3_nr = {a32:.4g} s^-1)"),
    ("energy transfer", ("s31", "q22", "q23"),
     "s31 = {s31:.4g}, Q22 = {q22:.4g}, Q23 = {q23:.4g} s^-1"),
    ("GSA/ESA line", ("lambda0_gsa_nm", "sigma_lambda_gsa_nm", "lambda0_esa_nm", "sigma_lambda_esa_nm"),
     "lambda0_gsa = {lambda0_gsa_nm:g} nm (sigma {sigma_lambda_gsa_nm:g} nm), "
     "lambda0_esa = {lambda0_esa_nm:g} nm (sigma {sigma_lambda_esa_nm:g} nm)"),
)


def format_parameters(p, indent="    "):
    """All model parameters of p (pa_model.PAParameters) as readable lines, for printing."""
    values = {f.name: getattr(p, f.name) for f in dataclasses.fields(p)}
    values["a32"] = p.b32 * p.w3_r + p.w3_nr
    lines, used = [], {"a32"}
    for label, names, fmt in _PARAM_LINES:
        if not all(n in values for n in names):
            continue
        try:
            text = fmt.format(**values)
        except (TypeError, ValueError):
            text = ", ".join(f"{n} = {values[n]}" for n in names)
        lines.append(f"{indent}{label:<21s}: {text}")
        used.update(names)
    other = [n for n in values if n not in used]      # fields added to the model later
    if other:
        lines.append(f"{indent}{'other':<21s}: " + ", ".join(f"{n} = {values[n]}" for n in other))
    return "\n".join(lines)


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
# Fast algebraic steady state (same DREs; for fits with many parameters)
# ---------------------------------------------------------------------------
# x = N1 (3F4), z = N2 (3H4), N0 = 1 - x - z. Steady state of the DREs:
#   F1 = Re x - w3 z - s31 z N0 + q23 x^2                          = 0
#   F2 = Rg N0 - Re x - w2 x + a32 z + 2 s31 z N0 - (q22+2q23) x^2 = 0
# 2 F1 + F2 removes the cross-relaxation term and is LINEAR in z:
#   z = [Rg (1-x) + (Re - w2) x - q22 x^2] / (2 w3 - a32 + Rg)
# Substituted into F1 this gives a quartic in x. Its physical roots
# (0 <= x, z and x + z <= 1) are the steady states; each is polished with
# Newton on (F1, F2). With exactly one physical root this IS the state that
# LSODA reaches from the ground state. With more than one (bistability), the
# intensity falls back to the LSODA routine above (upward-sweep branch).
# emission_curve_fast is ~500x faster than emission_curve; check_fast_solver
# compares both.

def _fast_state(I, p, wg, we):
    Rg, Re = wg * I, we * I
    w2, w3, s31, q22, q23 = p.w2, p.w3, p.s31, p.q22, p.q23
    a32 = p.b32 * p.w3_r + p.w3_nr
    qs = q22 + 2.0 * q23
    D = 2.0 * w3 - a32 + Rg
    P = np.array([Rg, Re - w2 - Rg, -q22])            # z * D, ascending powers of x
    pp = np.polynomial.polynomial
    # D^2 * F1 = D^2 (Re x + q23 x^2) - w3 D P - s31 P (D (1 - x) - P)
    quartic = pp.polysub(
        pp.polysub(np.array([0.0, D * D * Re, D * D * q23]), w3 * D * P),
        s31 * pp.polymul(P, pp.polysub(D * np.array([1.0, -1.0]), P)))
    quartic = np.trim_zeros(np.asarray(quartic, dtype=float), "b")
    roots = pp.polyroots(quartic)
    states = []
    for r in roots:
        if abs(r.imag) > 1e-7 * max(1.0, abs(r.real)):
            continue
        x = float(r.real)
        z = float(pp.polyval(x, P) / D)
        if x < -1e-12 or z < -1e-12 or x + z > 1.0 + 1e-12:
            continue
        x, z = min(max(x, 0.0), 1.0), max(z, 0.0)
        for _ in range(4):                             # Newton polish on (F1, F2)
            n0 = 1.0 - x - z
            F1 = Re * x - w3 * z - s31 * z * n0 + q23 * x * x
            F2 = Rg * n0 - Re * x - w2 * x + a32 * z + 2.0 * s31 * z * n0 - qs * x * x
            J11 = Re + s31 * z + 2.0 * q23 * x
            J12 = -w3 - s31 * (n0 - z)
            J21 = -Rg - Re - w2 - 2.0 * s31 * z - 2.0 * qs * x
            J22 = -Rg + a32 + 2.0 * s31 * (n0 - z)
            det = J11 * J22 - J12 * J21
            if det == 0.0:
                break
            dx = (F1 * J22 - F2 * J12) / det
            dz = (J11 * F2 - J21 * F1) / det
            x, z = x - dx, z - dz
            if abs(dx) <= 1e-15 * max(x, 1e-300) and abs(dz) <= 1e-15 * max(z, 1e-300):
                break
        if any(abs(x - s[0]) <= 1e-9 * max(x, 1e-30) and abs(z - s[1]) <= 1e-9 * max(z, 1e-30)
               for s in states):
            continue                                   # double root
        states.append((x, max(z, 0.0)))
    return states


def emission_curve_fast(I_values, p, wg, we, stats=None):
    """Same result as emission_curve, algebraically (LSODA only if bistable)."""
    out = np.empty(len(I_values))
    n_fallback = 0
    for j, I in enumerate(I_values):
        states = _fast_state(float(I), p, wg, we)
        if len(states) == 1:
            n2 = states[0][1]
        else:
            n_fallback += 1
            n2 = max(float(steady_state(I, p, wg, we)[2]), 0.0)
        out[j] = pm.emission_from_population(n2, p)
    if stats is not None:
        stats["lsoda_fallbacks"] = stats.get("lsoda_fallbacks", 0) + n_fallback
    return out


def check_fast_solver(I_values, p, wg, we):
    """Max relative difference between emission_curve_fast and emission_curve
    (only where the emission is above 1e-12 of its maximum)."""
    fast = emission_curve_fast(I_values, p, wg, we)
    ref = emission_curve(I_values, p, wg, we)
    m = ref > 1e-12 * ref.max()
    return float(np.max(np.abs(fast[m] / ref[m] - 1.0)))


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
        raise FileNotFoundError(f"Measurement file missing: {path}. Put it in the data/ folder "
                                "or correct the file name in measurements.py (see README.md).")
    return path


def out_path(name):
    return OUT_DIR / name


SCURVE_COLUMNS = ("fw6", "I_kWcm2", "net_cps", "noise_cps")


def load(path):
    """S-curve CSV sorted by intensity (rows without numbers are skipped)."""
    p = Path(path)
    if not p.is_absolute():
        p = DATA_DIR / p.name
    d = np.genfromtxt(p, delimiter=",", names=True)
    missing = [c for c in SCURVE_COLUMNS if c not in (d.dtype.names or ())]
    if missing:
        raise ValueError(f"{p.name}: column(s) {missing} missing (found {d.dtype.names}). "
                         f"An S-curve file needs the columns {', '.join(SCURVE_COLUMNS)}.")
    ok = np.all([np.isfinite(d[c]) for c in SCURVE_COLUMNS], axis=0)
    if not ok.all():
        print(f"NOTE: {p.name}: {np.sum(~ok)} row(s) without numbers skipped.")
    d = d[ok]
    o = np.argsort(d["I_kWcm2"])
    return d[o]


def local_slope_binned(I, y, noise, width_dex=0.06, min_snr=5.0):
    """Log-log slope from a moving linear fit (window width_dex in log I),
    using only points with y > min_snr * noise."""
    ok = y > min_snr * noise
    if ok.sum() < 4:                   # too few points above the noise for a slope
        return np.array([]), np.array([])
    x = np.log10(I[ok]); ly = np.log10(y[ok])
    xc, sl = [], []
    for c in np.arange(x.min() + width_dex / 2, x.max() - width_dex / 2, width_dex / 4):
        m = np.abs(x - c) <= width_dex / 2
        if m.sum() >= 4:
            a = np.polyfit(x[m], ly[m], 1)[0]
            xc.append(10 ** c); sl.append(a)
    return np.array(xc), np.array(sl)


# ---------------------------------------------------------------------------
# Measurements (the list in measurements.py)
# ---------------------------------------------------------------------------
MEASUREMENTS_FILE = HERE / "measurements.py"
MEASUREMENT_DEFAULTS = dict(file=None, label=None, wavelength_nm=None, fwhm_nm=0.0, sample="",
                            date="", knee_fw=None, compare_sweeps=None,
                            compare_window_kWcm2=None, spectrum_file=None,
                            spectrum_noise_cts=180.0)
_REQUIRED_FIELDS = ("file", "label", "wavelength_nm")
_SUMMARY_KEYS = ("model_info", "reference")      # other top-level keys of summary.json


def _measurements_file():
    """(MEASUREMENTS, ACTIVE, REFERENCE) from measurements.py. The file is read again
    on every call, so an edit counts without restarting the console (kernel)."""
    cfg = runpy.run_path(str(MEASUREMENTS_FILE))
    return cfg["MEASUREMENTS"], cfg.get("ACTIVE"), cfg.get("REFERENCE")


def measurements():
    """Active measurements as {name: settings}, in the order of measurements.py,
    with the defaults filled in and 'file' as the full path (checked to exist)."""
    table, active, _ref = _measurements_file()
    names = list(table) if active is None else list(active)
    unknown = [n for n in names if n not in table]
    if unknown:
        raise KeyError(f"ACTIVE in measurements.py names {unknown}, which are not in "
                       f"MEASUREMENTS ({', '.join(table)}).")
    if not names:
        raise ValueError("No measurements selected: MEASUREMENTS or ACTIVE in measurements.py is empty.")
    out = {}
    for name in names:
        entry = table[name]
        where = f"measurements.py, '{name}'"
        if name in _SUMMARY_KEYS:
            raise ValueError(f"{where}: this name is reserved; choose another one.")
        wrong = sorted(set(entry) - set(MEASUREMENT_DEFAULTS))
        if wrong:
            raise ValueError(f"{where}: unknown field(s) {wrong}. Allowed: "
                             + ", ".join(MEASUREMENT_DEFAULTS) + ".")
        missing = [f for f in _REQUIRED_FIELDS if entry.get(f) is None]
        if missing:
            raise ValueError(f"{where}: required field(s) {missing} missing.")
        m = dict(MEASUREMENT_DEFAULTS, **entry)
        m["name"] = name
        m["file"] = str(data_path(entry["file"]))
        m["wavelength_nm"] = float(m["wavelength_nm"])
        m["fwhm_nm"] = float(m["fwhm_nm"] or 0.0)
        if m["compare_sweeps"] is not None and len(m["compare_sweeps"]) != 2:
            raise ValueError(f"{where}: compare_sweeps must be two fw6 settings, e.g. (4, 5).")
        out[name] = m
    return out


def reference_name(meas):
    """REFERENCE from measurements.py, or the first active measurement."""
    ref = _measurements_file()[2]
    if ref in meas:
        return ref
    first = next(iter(meas))
    if ref is not None:
        print(f"NOTE: REFERENCE '{ref}' is not among the active measurements; using '{first}'.")
    return first


def describe(meas):
    """'BdJ1, 24-11-2026' for figure titles (each sample/date combination once)."""
    parts = []
    for m in meas.values():
        text = ", ".join(str(v) for v in (m["sample"], m["date"]) if v)
        if text and text not in parts:
            parts.append(text)
    return "; ".join(parts)


def read_summary(meas):
    """results/summary.json (from make_figures.py), checked to hold every active measurement."""
    path = out_path("summary.json")
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing: run make_figures.py first.")
    summary = json.loads(path.read_text())
    missing = [n for n in meas if n not in summary]
    if missing:
        raise KeyError(f"summary.json has no results for {missing}: run make_figures.py "
                       "again after adding measurements to measurements.py.")
    return summary
