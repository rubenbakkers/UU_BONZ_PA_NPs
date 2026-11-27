"""Emission spectra (e.g. SuperK, 785-862 nm) vs. the 3H4 -> 3H6 Stark lines of the model.

Fits every measurement in measurements.py that has a spectrum_file.
Model: sticks at E(3H4, i) - E(3H6, j), Boltzmann-weighted over 3H4, equal
line strengths, each a Gaussian (sigma in cm^-1). Fitted: shift (nm), sigma,
amplitude and background. The results go to results/emission_spec.json (one
entry per measurement); make_figures2.py (fig. 3a) reads them from there.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # make the scripts in this folder importable
import json
import numpy as np
from scipy.optimize import least_squares
from pa_fit_lib import data_path, out_path, pf, measurements

E4, E6 = pf.E_3H4_CM1, pf.E_3H6_CM1
NU = (E4[:, None] - E6[None, :]).ravel()               # sticks [cm^-1]
W = np.repeat(pf.boltzmann_within(E4), len(E6))        # Boltzmann over 3H4
START_SHIFTS_NM = (0.0, 10.0, 20.0)


def spectrum_measurements(meas=None):
    """The measurements (from measurements.py) that have an emission spectrum."""
    meas = measurements() if meas is None else meas
    return {n: m for n, m in meas.items() if m["spectrum_file"]}


def load_spectrum(m):
    """(wavelength_nm, cts_per_s_per_pixel) of measurement m (an entry of measurements())."""
    path = data_path(m["spectrum_file"])
    d = np.genfromtxt(str(path), delimiter=",", names=True)
    missing = [c for c in ("wavelength_nm", "cts_per_s_per_pixel") if c not in (d.dtype.names or ())]
    if missing:
        raise ValueError(f"{path.name}: column(s) {missing} missing (found {d.dtype.names}).")
    return d["wavelength_nm"], d["cts_per_s_per_pixel"]


def model_spectrum(lam_nm, shift_nm, sig_cm1, A, b):
    """Counts per pixel: Gaussian sum in wavenumber, converted to per wavelength
    (constant d lambda per pixel); shift_nm > 0 moves the model to the red."""
    nu_obs = 1e7 / (np.asarray(lam_nm, float) - shift_nm)
    s = np.sum(W[None, :] * np.exp(-0.5 * ((nu_obs[:, None] - NU[None, :]) / sig_cm1) ** 2),
               axis=1)
    return A * s * nu_obs ** 2 / 1e7 + b


def fit_emission(m):
    lam, y = load_spectrum(m)
    noise = m["spectrum_noise_cts"]                    # noise per pixel (estimate)

    def residual(x):
        return (model_spectrum(lam, *x) - y) / noise

    best = None
    for s0 in START_SHIFTS_NM:
        r = least_squares(residual, [s0, 60.0, 50.0, 0.0],
                          bounds=([-40, 5, 0, -500], [40, 400, 1e6, 500]))
        chi2_red = float(np.sum(r.fun ** 2) / (len(y) - 4))
        print(f"start shift {s0:4.1f} nm: shift={r.x[0]:.1f} nm sigma={r.x[1]:.0f} cm-1 "
              f"chi2r={chi2_red:.2f}")
        if best is None or chi2_red < best[1]:
            best = (r, chi2_red)
    (shift, sigma, A, b), chi2_red = best[0].x, best[1]

    # Peak positions: model without and with shift, and the (smoothed) measurement.
    grid = np.linspace(760.0, 880.0, 12001)
    peak_unshifted = float(grid[np.argmax(model_spectrum(grid, 0.0, sigma, A, b))])
    peak_shifted = float(grid[np.argmax(model_spectrum(grid, shift, sigma, A, b))])
    ys = np.convolve(y, np.ones(9) / 9, mode="same")
    i = int(np.argmax(ys))
    above = np.where(ys > ys[i] / 2)[0]
    yc = np.clip(y, 0, None)
    return dict(
        label=m["label"], spectrum_file=m["spectrum_file"], noise_cts=noise,
        shift_nm=float(shift), sigma_cm1=float(sigma), amplitude=float(A),
        background=float(b), chi2_red=chi2_red,
        shift_cm1_at_peak=float(1e7 / peak_unshifted - 1e7 / lam[i]),
        model_peak_nm_unshifted=peak_unshifted, model_peak_nm_shifted=peak_shifted,
        data_peak_nm=float(lam[i]), data_fwhm_range_nm=[float(lam[above[0]]), float(lam[above[-1]])],
        data_centroid_nm=float(np.sum(lam * yc) / np.sum(yc)),
        stick_range_nm=[float(1e7 / NU.max()), float(1e7 / NU.min())],
        boltzmann_mean_nm=float(np.sum(W * 1e7 / NU) / W.sum()),
    )


if __name__ == "__main__":
    spectra = spectrum_measurements()
    if not spectra:
        print("No measurement in measurements.py has a spectrum_file; nothing to fit.")
    results = {}
    for name, m in spectra.items():
        print(f"--- {name} ({m['label']}): {m['spectrum_file']}")
        res = fit_emission(m)
        results[name] = res
        print(f"sticks {res['stick_range_nm'][0]:.1f}-{res['stick_range_nm'][1]:.1f} nm, "
              f"Boltzmann mean {res['boltzmann_mean_nm']:.1f} nm")
        print(f"best fit: shift {res['shift_nm']:.1f} nm, sigma {res['sigma_cm1']:.0f} cm-1, "
              f"amplitude {res['amplitude']:.4g}, background {res['background']:.4g} cts/s/pixel, "
              f"chi2r {res['chi2_red']:.2f}")
        print(f"model peak {res['model_peak_nm_unshifted']:.1f} nm (unshifted), measured peak "
              f"{res['data_peak_nm']:.1f} nm (9-px average), measured centroid "
              f"{res['data_centroid_nm']:.1f} nm")
    json.dump(results, open(out_path("emission_spec.json"), "w"), indent=1)
