# -*- coding: utf-8 -*-
"""Calibrate '1 fonon zijband.py' against a measured room-temperature absorption spectrum.

Input: two PerkinElmer .asc files from the Lambda 950 (or any two-column file with a
'#DATA' line): the particles in solvent measured against solvent, and the solvent
reference scan (solvent against solvent). The reference scan is only used to find the
wavelengths where the cuvette is (nearly) opaque, so that those points are not fitted.

Every band is fitted with the model's OWN line shape: Table 4 Stark levels with a
Boltzmann-populated 3H6, equal upper-level weights, a Gaussian electronic width and the
four-Gaussian multiphonon kernel. A local straight baseline is removed under each band.

  1. 3H6 -> 3H5 (1110-1300 nm): the Huang-Rhys coupling S is set by the blue
     (one-phonon) wing at 1115-1165 nm; then sigma and a rigid shift are fitted.
  2. 3H6 -> 3F4: stretch of the 3F4 manifold about its lowest level (sigma, S fixed),
     on the points where the solvent transmits.
  3. 3H6 -> 3H4: stretch + shift.     4. 3H6 -> 3F2 + 3F3: 3F2 shift, 3F3 stretch + shift.

The result (JSON) holds the numbers for ABSORPTION_CALIBRATION in '1 fonon zijband.py';
copy them there by hand. A figure shows every band fit next to the original model.

Usage:
    python calibrate_from_absorption.py SAMPLE.asc REFERENCE.asc
    python calibrate_from_absorption.py SAMPLE.asc REFERENCE.asc --exclude 1700 1780 --output-dir out
In a C-H-free solvent (C2Cl4, CCl4) no region is excluded and the full 3F4 band is used.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.optimize import brentq, least_squares

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "1 fonon zijband.py"
FIT_MAX_PHONONS = 3  # enough near the bands; higher orders are < 1e-4 of the band area

# Band windows and straight-baseline anchor windows [nm] (NaYF4:Tm3+ at room temperature).
BANDS = {
    "3F2+3F3": dict(win=(640, 720), left=(636, 644), right=(722, 730), fit=(645, 715)),
    "3H4": dict(win=(735, 845), left=(728, 735), right=(840, 852), fit=(755, 835)),
    "3H5": dict(win=(1105, 1320), left=(1098, 1106), right=(1318, 1330), fit=(1110, 1300)),
    "3F4": dict(win=(1445, 1905), left=(1440, 1448), right=(1898, 1904), fit=(1600, 1895)),
}
H5_CORE_NM = (1180, 1260)   # band core: sigma and shift for a trial coupling
H5_WING_NM = (1115, 1165)   # blue wing: one-phonon sideband sets the coupling
OPACITY_THRESHOLD = 40.0    # reference-scan noise / quiet-region noise above which a point is excluded
OPACITY_QUIET_NM = (1500, 1660)
OPACITY_MERGE_NM = 15.0     # merge excluded stretches closer than this
OPACITY_PAD_NM = 3.0        # widen each excluded stretch by this on both sides


def load_model():
    spec = importlib.util.spec_from_file_location("one_phonon_sideband_model", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_asc(path):
    """Wavelength (nm, ascending) and absorbance from a PerkinElmer .asc file."""
    lines = Path(path).read_text(errors="replace").splitlines()
    start = lines.index("#DATA") + 1 if "#DATA" in lines else 0
    rows = []
    for line in lines[start:]:
        parts = line.replace(",", " ").split()
        if len(parts) >= 2:
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    data = np.array(rows)
    order = np.argsort(data[:, 0])
    return data[order, 0], data[order, 1]


def opaque_ranges(wl, reference, extra=()):
    """Wavelength ranges [nm] where the reference scan is too noisy to trust the sample."""
    pad = 3
    window = sliding_window_view(np.pad(reference, pad, mode="edge"), 2 * pad + 1)
    noise = 1.4826 * np.median(np.abs(window - np.median(window, axis=1)[:, None]), axis=1)
    quiet = (wl >= OPACITY_QUIET_NM[0]) & (wl <= OPACITY_QUIET_NM[1])
    ratio = noise / np.median(noise[quiet])
    ratio = np.median(sliding_window_view(np.pad(ratio, 4, mode="edge"), 9), axis=1)
    idx = np.flatnonzero(ratio > OPACITY_THRESHOLD)
    ranges = []
    if idx.size:
        for group in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            lo, hi = wl[group[0]] - OPACITY_PAD_NM, wl[group[-1]] + OPACITY_PAD_NM
            if ranges and lo - ranges[-1][1] < OPACITY_MERGE_NM:
                ranges[-1][1] = hi
            else:
                ranges.append([lo, hi])
    ranges += [list(r) for r in extra]
    return [(float(lo), float(hi)) for lo, hi in sorted(ranges)]


class Spectrum:
    def __init__(self, sample, reference, exclude=()):
        self.wl, self.a = read_asc(sample)
        wl_r, self.a_ref = read_asc(reference)
        if not np.allclose(self.wl, wl_r):
            self.a_ref = np.interp(self.wl, wl_r, self.a_ref)
        self.nu = 1e7 / self.wl
        self.excluded = opaque_ranges(self.wl, self.a_ref, exclude)
        self.usable = np.ones_like(self.wl, dtype=bool)
        for lo, hi in self.excluded:
            self.usable &= ~((self.wl >= lo) & (self.wl <= hi))

    def net(self, band):
        """Absorbance minus a straight line through the medians of the two anchor windows."""
        b = BANDS[band]
        pts = []
        for lo, hi in (b["left"], b["right"]):
            m = (self.wl >= lo) & (self.wl <= hi)
            pts.append((self.wl[m].mean(), np.median(self.a[m])))
        (x0, y0), (x1, y1) = pts
        return self.a - (y0 + (y1 - y0) * (self.wl - x0) / (x1 - x0))

    def select(self, lo, hi, usable_only=True):
        m = (self.wl >= lo) & (self.wl <= hi)
        return m & self.usable if usable_only else m

    def integral(self, y, mask):
        order = np.argsort(self.nu[mask])
        return float(np.trapezoid(y[mask][order], self.nu[mask][order]))


class LineShape:
    """Area-normalized band shape (per cm^-1) built from the model's own functions."""

    def __init__(self, model, temperature_k=None):
        self.m = model
        self.t = model.TEMPERATURE_K if temperature_k is None else temperature_k

    @lru_cache(maxsize=256)
    def _kernel(self, coupling):
        kernel, _ = self.m.multiphonon_kernel(coupling, self.t, self.m.band_mask(), FIT_MAX_PHONONS)
        return kernel

    def __call__(self, lower, upper, nu, sigma, coupling):
        centers, strengths = self.m.transition_sticks(np.asarray(lower, float), np.asarray(upper, float), self.t)
        components = []
        for term in self._kernel(round(float(coupling), 12)):
            width = math.sqrt(float(sigma) ** 2 + term["variance_cm2"])
            components.append((centers + term["shift_cm1"], strengths * term["probability"], width,
                               term["order"] == 0))
        return self.m.gaussian_mixture(np.asarray(nu, float),
                                       self.m.merge_gaussian_components(components), 0.0)


def transform(levels, stretch=1.0, shift=0.0):
    lowest = levels.min()
    return lowest + shift + stretch * (levels - lowest)


def rms(v):
    return float(np.sqrt(np.mean(np.square(v))))


def calibrate(spec, model):
    shape = LineShape(model)
    L = model.STARK_LEVELS_TABLE4_CM1
    h6 = L["3H6"]
    out = {}

    # ---- 1. 3H5: coupling from the blue wing, then sigma and shift ---------------------
    net = spec.net("3H5")
    core = spec.select(*H5_CORE_NM)
    wing = spec.select(*H5_WING_NM)

    def core_fit(coupling):
        f = least_squares(lambda p: p[0] * shape(h6, L["3H5"] + p[2], spec.nu[core], p[1], coupling) - net[core],
                          x0=[70.0, 28.0, -15.0], bounds=([1, 5, -150], [1e4, 150, 150]), diff_step=1e-3)
        return f.x

    def wing_ratio(coupling):
        area, sigma, shift = core_fit(coupling)
        model_wing = area * shape(h6, L["3H5"] + shift, spec.nu, sigma, coupling)
        return spec.integral(model_wing, wing) / spec.integral(net, wing)

    coupling = brentq(lambda s: wing_ratio(s) - 1.0, 0.02, 0.4, xtol=1e-3)
    fit_range = spec.select(*BANDS["3H5"]["fit"])
    x, y = spec.nu[fit_range], net[fit_range]
    f = least_squares(lambda p: p[0] * shape(h6, L["3H5"] + p[2], x, p[1], coupling) - y,
                      x0=[70.0, 28.0, -15.0], bounds=([1, 5, -150], [1e4, 150, 150]), diff_step=1e-3)
    area_h5, sigma, shift_h5 = f.x
    out["3H5"] = dict(coupling=coupling, sigma_cm1=sigma, shift_cm1=shift_h5, stretch=1.0,
                      area_fit=area_h5, area_measured=spec.integral(net, spec.select(*BANDS["3H5"]["win"], False)),
                      rms=rms(f.fun), wing_ratio_at_coupling_0p05=wing_ratio(0.05),
                      curve=(area_h5, L["3H5"] + shift_h5))

    # ---- 2. 3F4: stretch about the lowest level -------------------------------------------
    net = spec.net("3F4")
    m = spec.select(*BANDS["3F4"]["fit"])
    x, y = spec.nu[m], net[m]
    f = least_squares(lambda p: p[0] * shape(h6, transform(L["3F4"], p[1]), x, sigma, coupling) - y,
                      x0=[80.0, 1.5], bounds=([0, 0.5], [1e4, 5.0]), diff_step=1e-3)
    g = least_squares(lambda p: p[0] * shape(h6, L["3F4"] + p[1], x, sigma, coupling) - y,
                      x0=[80.0, 100.0], bounds=([0, -400], [1e4, 600]), diff_step=1e-3)
    h = least_squares(lambda p: p[0] * shape(h6, L["3F4"], x, sigma, coupling) - y,
                      x0=[80.0], bounds=([0], [1e4]), diff_step=1e-3)
    levels = transform(L["3F4"], f.x[1])
    out["3F4"] = dict(stretch=f.x[1], shift_cm1=0.0, area_fit=f.x[0], rms=rms(f.fun),
                      levels_cm1=(levels.min(), levels.max()),
                      rigid_shift_alternative=dict(shift_cm1=g.x[1], rms=rms(g.fun)),
                      unchanged_levels_rms=rms(h.fun),
                      area_measured_with_gaps=spec.integral(net, spec.select(*BANDS["3F4"]["win"], False)),
                      fitted_points=int(m.sum()), curve=(f.x[0], levels))

    # ---- 3. 3H4: stretch + shift ------------------------------------------------------------
    net = spec.net("3H4")
    m = spec.select(*BANDS["3H4"]["fit"])
    x, y = spec.nu[m], net[m]
    f = least_squares(lambda p: p[0] * shape(h6, transform(L["3H4"], p[1], p[2]), x, sigma, coupling) - y,
                      x0=[100.0, 1.5, 0.0], bounds=([1, 0.3, -400], [1e4, 6, 400]), diff_step=1e-3)
    levels = transform(L["3H4"], f.x[1], f.x[2])
    out["3H4"] = dict(stretch=f.x[1], shift_cm1=f.x[2], area_fit=f.x[0], rms=rms(f.fun),
                      levels_cm1=(levels.min(), levels.max()), curve=(f.x[0], levels))

    # ---- 4. 3F2 + 3F3: 3F2 shift, 3F3 stretch + shift -------------------------------------
    net = spec.net("3F2+3F3")
    m = spec.select(*BANDS["3F2+3F3"]["fit"])
    x, y = spec.nu[m], net[m]

    def two(p, xx):
        a2, a3, d2, k3, d3 = p
        return (a2 * shape(h6, L["3F2"] + d2, xx, sigma, coupling)
                + a3 * shape(h6, transform(L["3F3"], k3, d3), xx, sigma, coupling))

    f = least_squares(lambda p: two(p, x) - y, x0=[30.0, 150.0, 40.0, 1.5, 100.0],
                      bounds=([0, 0, -400, 0.3, -400], [1e4, 1e4, 400, 6, 400]), diff_step=1e-3)
    a2, a3, d2, k3, d3 = f.x
    levels3 = transform(L["3F3"], k3, d3)
    out["3F2"] = dict(stretch=1.0, shift_cm1=d2, area_fit=a2, levels_cm1=(L["3F2"].min() + d2, L["3F2"].max() + d2),
                      curve=(a2, L["3F2"] + d2))
    out["3F3"] = dict(stretch=k3, shift_cm1=d3, area_fit=a3, levels_cm1=(levels3.min(), levels3.max()),
                      curve=(a3, levels3))
    out["3F2+3F3"] = dict(rms=rms(f.fun))
    out["_shape"] = shape
    return out


def original_fit(spec, model, band, uppers):
    """Original model (Table 4 levels, sigma 8.5, coupling 0.05): only the areas are fitted."""
    shape = LineShape(model)
    h6 = model.STARK_LEVELS_TABLE4_CM1["3H6"]
    net = spec.net(band)
    m = spec.select(*BANDS[band]["fit"])
    basis = [shape(h6, U, spec.nu[m], model.ELECTRONIC_SIGMA_CM1, model.DEFAULT_COUPLING) for U in uppers]
    scales = np.linalg.lstsq(np.column_stack(basis), net[m], rcond=None)[0]
    return lambda nu: sum(s * shape(h6, U, nu, model.ELECTRONIC_SIGMA_CM1, model.DEFAULT_COUPLING)
                          for s, U in zip(scales, uppers))


def make_figure(spec, model, res, path):
    import matplotlib.pyplot as plt

    ink, grey, blue, shade = "#1a1a19", "#8a8984", "#2a78d6", "#ecebe7"
    L = model.STARK_LEVELS_TABLE4_CM1
    shape = res["_shape"]
    sigma, coupling = res["3H5"]["sigma_cm1"], res["3H5"]["coupling"]
    h6 = L["3H6"]
    panels = [("3F2+3F3", "³H₆ → ³F₂ + ³F₃", (640, 720), [L["3F2"], L["3F3"]], ["3F2", "3F3"]),
              ("3H4", "³H₆ → ³H₄", (740, 850), [L["3H4"]], ["3H4"]),
              ("3H5", "³H₆ → ³H₅", (1100, 1320), [L["3H5"]], ["3H5"]),
              ("3F4", "³H₆ → ³F₄", (1500, 1920), [L["3F4"]], ["3F4"])]
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))
        for ax, (band, title, (lo, hi), uppers, names) in zip(axes.ravel(), panels):
            wl_grid = np.linspace(lo, hi, 1200)
            nu_grid = 1e7 / wl_grid
            net = spec.net(band)
            shown = (spec.wl >= lo) & (spec.wl <= hi)
            for xlo, xhi in spec.excluded:
                if xhi > lo and xlo < hi:
                    ax.axvspan(max(xlo, lo), min(xhi, hi), color=shade, lw=0, zorder=0)
                    ax.text(0.5 * (max(xlo, lo) + min(xhi, hi)), 0.97, "solvent\nopaque", ha="center",
                            va="top", fontsize=8, color="#52514e", transform=ax.get_xaxis_transform())
            meas = np.where(spec.usable, net, np.nan)
            ax.plot(spec.wl[shown], meas[shown], color=ink, lw=1.6, label="measured (net)", zorder=3)
            gap = shown & ~spec.usable
            if gap.any():  # distorted data where the solvent is opaque: shown, not fitted
                ax.plot(spec.wl[gap], net[gap], color=ink, lw=0.9, ls=":", alpha=0.6, zorder=2)
            calibrated = sum(res[n]["curve"][0] * shape(h6, res[n]["curve"][1], nu_grid, sigma, coupling)
                             for n in names)
            original = original_fit(spec, model, band, uppers)(nu_grid)
            top = 1.3 * max(np.nanmax(meas[shown]), calibrated.max())
            ax.plot(wl_grid, np.minimum(original, top), color=grey, lw=1.0, ls="--",
                    label="original model (σ 8.5 cm⁻¹, S 0.05)", zorder=2)
            ax.plot(wl_grid, calibrated, color=blue, lw=2.0, label="calibrated model", zorder=4)
            clipped = original.max() > top
            ax.set(xlim=(lo, hi), ylim=(-0.05 * top, top), xlabel="Wavelength (nm)",
                   ylabel="Net absorbance")
            ax.set_title(title + ("   (original clipped)" if clipped else ""), loc="left", fontsize=10.5)
            ax.grid(alpha=0.18)
        axes[0, 0].legend(loc="upper left", fontsize=8.5, frameon=False)
        fig.suptitle("Room-temperature absorption bands vs the model line shape "
                     "(³H₄, ³F₃, ³F₂ enter the model as ESA end levels)", x=0.01, ha="left", fontsize=11.5)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sample", type=Path, help="Particles in solvent, measured against solvent (.asc)")
    parser.add_argument("reference", type=Path, help="Solvent against solvent reference scan (.asc)")
    parser.add_argument("--exclude", type=float, nargs=2, action="append", default=[], metavar=("MIN", "MAX"),
                        help="Extra wavelength range [nm] to leave out of the fits (repeatable).")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results" / "absorption_calibration")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    model = load_model()
    spec = Spectrum(args.sample, args.reference, args.exclude)
    res = calibrate(spec, model)
    h5, f4 = res["3H5"], res["3F4"]

    # Model channel-weight ratio 3F4 : 3H5 for comparison with the fitted areas.
    configuration = model.model_configuration(True)
    weights = {k: w for k, (_, _, w) in configuration["gsa_channels"].items()}
    weights_ed = {k: model.channel_weight(k, lo, up, False) for k, (lo, up, _) in configuration["gsa_channels"].items()}

    suggestion = {
        "electronic_sigma_gsa_cm1": round(h5["sigma_cm1"], 1),
        "electronic_sigma_esa_cm1": round(h5["sigma_cm1"], 1),
        "coupling": round(h5["coupling"], 2),
        "stark_corrections": {name: {"stretch": round(float(res[name]["stretch"]), 2),
                                     "shift_cm1": round(float(res[name]["shift_cm1"]), 1)}
                              for name in ("3F4", "3H5", "3H4", "3F3", "3F2")},
    }
    report = {
        "sample": str(args.sample), "reference": str(args.reference),
        "excluded_ranges_nm": spec.excluded,
        "suggested_ABSORPTION_CALIBRATION": suggestion,
        "3H5": {k: v for k, v in h5.items() if k != "curve"},
        "3F4": {k: v for k, v in f4.items() if k != "curve"},
        "3H4": {k: v for k, v in res["3H4"].items() if k != "curve"},
        "3F3": {k: v for k, v in res["3F3"].items() if k != "curve"},
        "3F2": {k: v for k, v in res["3F2"].items() if k != "curve"},
        "3F2+3F3_rms": res["3F2+3F3"]["rms"],
        "area_ratio_3F4_over_3H5": {
            "fitted": f4["area_fit"] / h5["area_fit"],
            "model_JO_ED_plus_MD": weights["3H6 -> 3F4"] / weights["3H6 -> 3H5"],
            "model_JO_ED_only": weights_ed["3H6 -> 3F4"] / weights_ed["3H6 -> 3H5"]},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "absorption_calibration.json").write_text(
        json.dumps(report, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)) + "\n")

    print(f"Excluded (solvent opaque): {', '.join(f'{lo:.0f}-{hi:.0f} nm' for lo, hi in spec.excluded) or 'none'}")
    print(f"3H5: coupling S = {h5['coupling']:.3f} (wing at S = 0.05: {h5['wing_ratio_at_coupling_0p05']:.2f} of measured); "
          f"sigma = {h5['sigma_cm1']:.1f} cm-1; shift = {h5['shift_cm1']:+.1f} cm-1; rms {h5['rms']:.4f}")
    print(f"3F4: stretch = {f4['stretch']:.2f} -> {f4['levels_cm1'][0]:.0f}-{f4['levels_cm1'][1]:.0f} cm-1; rms {f4['rms']:.4f} "
          f"(rigid shift {f4['rigid_shift_alternative']['shift_cm1']:+.0f}: rms {f4['rigid_shift_alternative']['rms']:.4f}; "
          f"unchanged: rms {f4['unchanged_levels_rms']:.4f}; {f4['fitted_points']} points)")
    for name in ("3H4", "3F3", "3F2"):
        r = res[name]
        print(f"{name}: stretch {r['stretch']:.2f}, shift {r['shift_cm1']:+.1f} cm-1 -> "
              f"{r['levels_cm1'][0]:.0f}-{r['levels_cm1'][1]:.0f} cm-1")
    ratio = report["area_ratio_3F4_over_3H5"]
    print(f"Area 3F4/3H5: fitted {ratio['fitted']:.2f}; model {ratio['model_JO_ED_plus_MD']:.2f} (ED+MD), "
          f"{ratio['model_JO_ED_only']:.2f} (ED only)")
    print("Suggested ABSORPTION_CALIBRATION:", json.dumps(suggestion))
    if not args.no_plot:
        make_figure(spec, model, res, args.output_dir / "absorption_calibration_fits.png")
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
