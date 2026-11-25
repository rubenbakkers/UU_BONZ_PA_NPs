# -*- coding: utf-8 -*-
"""Slope s of the measured S-curves, with two methods.

Only points with emission > MIN_SNR * noise are used (the log of noise has no
meaning).

Method 1, as in Lee et al. (2021): they define the nonlinearity s through
I_em ~ I_pump^s, the slope of the log-log curve. The threshold follows from SI
Fig. S3: straight lines (in log-log) through the points below and above the
threshold; their intersection is I_th, the slope of the steep line is s.
  - The points (sorted by intensity) are split into 2 or 3 contiguous
    segments, each with its own weighted straight line in log10(E) vs
    log10(I). The breakpoints are chosen such that the total chi^2 is
    minimal; the BIC chooses between 2 and 3 segments (3 only when the data
    also show saturation).
  - s = slope of the steepest segment. I_th = intersection of that segment
    with the one before it (only if there are enough significant points
    below the knee).

Method 2, as in pa_fononband (the black lines in its detuning figure): the
function pa_fononband.fit_avalanche_line itself, with SLOPE_FRACTION. It takes
the secant slopes between neighbouring points, finds the steepest one
(s_peak) and fits a straight line through the contiguous window where the
secant slope is >= SLOPE_FRACTION * s_peak. Secant slopes between single noisy
measurement points are meaningless, so the measurement is first averaged in
bins of BIN_DEX decades of intensity (weighted mean of log E per bin).

Uncertainty of both: bootstrap (resample the points, redo everything).
Output: results/fig_s_fit_lee.pdf/.png and results/s_fit_lee.json.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker

try:
    HERE = Path(__file__).resolve().parent
except NameError:                      # e.g. pasted into a console
    HERE = Path.cwd()
DATA_DIR, OUT_DIR = HERE / "data", HERE / "results"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
DATASETS = {
    "cobolt_1060": dict(file="cobolt_1060_scurve.csv", label="Cobolt, 1060 nm"),
    "superk_1064": dict(file="superk_1064_scurve.csv", label="SuperK, 1064 nm"),
}
MIN_SNR = 3.0          # only points with (net - background) > MIN_SNR * noise
REL_SCATTER = 0.10     # extra relative scatter between repeats (as in fit_scurves.py)
MIN_POINTS = 5         # minimum number of points per straight segment
N_SEGMENTS = (2, 3)    # numbers of straight segments that are tried (BIC chooses)
# Background per filter-wheel setting (FW6) still contained in net_cps:
#   "summary": from results/summary.json (fit of make_figures.py; model-dependent)
#   "none":    subtract nothing
BACKGROUND = "summary"
N_BOOTSTRAP = 1000
RNG_SEED = 1
# Method 2 (pa_fononband): bin width in decades of intensity; the slope fraction
# is taken from pa_fononband.SLOPE_FRACTION (0.25) so that both use the same window.
BIN_DEX = 0.05
N_BOOTSTRAP_PF = 300

# Same figure style as pa_fononband (LaTeX font, no titles).
STYLE = {
    "font.family": "serif", "font.serif": ["cmr10", "DejaVu Serif"],
    "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
    "font.size": 11, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 9, "axes.linewidth": 0.8, "axes.grid": False,
    "axes.xmargin": 0.0, "axes.ymargin": 0.0,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "legend.frameon": False, "lines.linewidth": 1.3,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "savefig.dpi": 300,
    "pdf.fonttype": 42,
}
FIG_WIDTH_IN = 16.0 / 2.54
COLOR_STEEP, COLOR_OTHER, COLOR_DATA = "#ba4a18", "#7a7a7a", "#1468b1"
COLOR_PF = "black"         # method 2, as the fitted lines in pa_fononband


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_dataset(name):
    cfg = DATASETS[name]
    path = DATA_DIR / cfg["file"]
    if not path.exists():
        raise FileNotFoundError(f"Measurement file missing: {path}. Put the CSV files from the "
                                "Drive (BdJ1, 24-11-2026) in the data/ folder (see README.md).")
    d = np.genfromtxt(path, delimiter=",", names=True)
    d = d[np.argsort(d["I_kWcm2"])]
    background = np.zeros(len(d))
    if BACKGROUND == "summary":
        path = OUT_DIR / "summary.json"
        if path.exists():
            per_fw = json.loads(path.read_text())[name]["background_per_FW"]
            background = np.array([per_fw.get(str(int(f)), 0.0) for f in d["fw6"]])
        else:
            print(f"  NOTE: {path} is missing (run make_figures.py first); "
                  "no background subtracted.")
    elif BACKGROUND != "none":
        raise ValueError("BACKGROUND must be 'summary' or 'none'.")
    y = d["net_cps"] - background
    sigma = np.sqrt(d["noise_cps"] ** 2 + (REL_SCATTER * np.clip(y, 0, None)) ** 2)
    return dict(I=d["I_kWcm2"], y=y, noise=d["noise_cps"], sigma=sigma, fw=d["fw6"],
                background=background)


# ---------------------------------------------------------------------------
# Piecewise linear fit (independent straight lines, as in Lee Fig. S3)
# ---------------------------------------------------------------------------
def _segment_tables(x, y, w):
    """chi^2, slope, intercept and var(slope) of the weighted straight line
    through the points a..b-1, for all 0 <= a < b <= n ((n+1) x (n+1) matrices)."""
    prefix = [np.concatenate([[0.0], np.cumsum(v)]) for v in (w, w * x, w * y, w * x * x,
                                                              w * x * y, w * y * y)]
    S0, Sx, Sy, Sxx, Sxy, Syy = (p[None, :] - p[:, None] for p in prefix)  # [a, b]
    n = len(x)
    idx = np.arange(n + 1)
    count = idx[None, :] - idx[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        Dxx = Sxx - Sx ** 2 / S0
        Dxy = Sxy - Sx * Sy / S0
        Dyy = Syy - Sy ** 2 / S0
        slope = Dxy / Dxx
        intercept = (Sy - slope * Sx) / S0
        chi2 = Dyy - Dxy * slope
        var_slope = 1.0 / Dxx
    bad = (count < MIN_POINTS) | ~np.isfinite(chi2) | (Dxx <= 1e-12)
    chi2 = np.where(bad, np.inf, np.maximum(chi2, 0.0))
    return chi2, slope, intercept, var_slope, count


def fit_segments(x, y, sigma_log, n_segments):
    """Best split into n_segments contiguous segments.
    Returns (total chi2, list of (a, b, slope, intercept, sd_slope))."""
    w = 1.0 / sigma_log ** 2
    chi2, slope, intercept, var_slope, count = _segment_tables(x, y, w)
    n = len(x)
    if n_segments == 2:
        total = chi2[0, :] + chi2[:, n]
        i = int(np.argmin(total))
        cuts = [0, i, n]
        best = total[i]
    elif n_segments == 3:
        total = chi2[0, :, None] + chi2 + chi2[None, :, n]   # [i, j]: 0..i, i..j, j..n
        i, j = np.unravel_index(int(np.argmin(total)), total.shape)
        cuts = [0, int(i), int(j), n]
        best = total[i, j]
    else:
        raise ValueError("Only 2 or 3 segments.")
    if not np.isfinite(best):
        return np.inf, []
    segments = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        dof = max(count[a, b] - 2, 1)
        scale = max(chi2[a, b] / dof, 1.0)          # inflate the sd when chi2_red > 1
        segments.append((a, b, float(slope[a, b]), float(intercept[a, b]),
                         float(np.sqrt(var_slope[a, b] * scale))))
    return float(best), segments


def analyse(x, y, sigma_log, n_segments=None):
    """Choose the number of segments (BIC), determine s and (if possible) I_th."""
    n = len(x)
    options = {}
    for k in ([n_segments] if n_segments else N_SEGMENTS):
        chi2, segs = fit_segments(x, y, sigma_log, k)
        if segs:
            n_par = 2 * k + (k - 1)                      # lines + breakpoints
            options[k] = (chi2 + n_par * np.log(n), chi2, segs)
    if not options:
        return None
    k = min(options, key=lambda key: options[key][0])
    _, chi2, segs = options[k]
    steep = int(np.argmax([s[2] for s in segs]))
    result = dict(n_segments=k, chi2=chi2, segments=segs, steep_index=steep,
                  s=segs[steep][2], s_sd=segs[steep][4], log_ith=None)
    if steep > 0:
        _, b_prev, m1, c1, _ = segs[steep - 1]
        a_steep, _, m2, c2, _ = segs[steep]
        if m2 > m1:
            x_cross = (c1 - c2) / (m2 - m1)
            # The intersection must lie between the two segments (with a small margin).
            if x[segs[steep - 1][0]] - 0.05 <= x_cross <= x[segs[steep][1] - 1] + 0.05:
                result["log_ith"] = float(x_cross)
    return result


# ---------------------------------------------------------------------------
# Method 2: pa_fononband.fit_avalanche_line on log-binned data
# ---------------------------------------------------------------------------
def bin_log(x, y, sigma_log, width=BIN_DEX):
    """Weighted mean of log10 E per bin of `width` decades in log10 I.
    Returns (log10 I, log10 E) of the bins, strictly increasing in I."""
    edges = np.arange(x.min(), x.max() + width, width)
    idx = np.clip(np.digitize(x, edges) - 1, 0, len(edges) - 1)
    xb, yb = [], []
    for i in np.unique(idx):
        m = idx == i
        w = 1.0 / sigma_log[m] ** 2
        xb.append(np.average(x[m], weights=w))
        yb.append(np.average(y[m], weights=w))
    return np.array(xb), np.array(yb)


def fit_pa_fononband(x, y, sigma_log, pf):
    """s_fit with exactly the algorithm of pa_fononband (fit_avalanche_line)."""
    xb, yb = bin_log(x, y, sigma_log)
    if len(xb) < 3:
        return None, xb, yb
    return pf.fit_avalanche_line(10 ** xb, 10 ** yb, fraction=pf.SLOPE_FRACTION), xb, yb


def bootstrap_pf(x, y, sigma_log, pf, rng):
    values = []
    n = len(x)
    for _ in range(N_BOOTSTRAP_PF):
        pick = np.sort(rng.integers(0, n, n))
        fit, _, _ = fit_pa_fononband(x[pick], y[pick], sigma_log[pick], pf)
        if fit is not None:
            values.append(fit["slope"])
    return np.array(values)


def bootstrap(x, y, sigma_log, n_segments, rng):
    s_values, ith_values = [], []
    n = len(x)
    for _ in range(N_BOOTSTRAP):
        pick = np.sort(rng.integers(0, n, n))
        r = analyse(x[pick], y[pick], sigma_log[pick], n_segments)
        if r is None:
            continue
        s_values.append(r["s"])
        ith_values.append(np.nan if r["log_ith"] is None else 10 ** r["log_ith"])
    return np.array(s_values), np.array(ith_values)


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------
def _decade_limits(values):
    v = values[np.isfinite(values) & (values > 0)]
    return 10.0 ** np.floor(np.log10(v.min())), 10.0 ** np.ceil(np.log10(v.max()))


def main():
    rng = np.random.default_rng(RNG_SEED)
    from pa_fit_lib import pf          # the model's own fit_avalanche_line (method 2)
    plt.rcParams.update(STYLE)
    import logging
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(FIG_WIDTH_IN, 0.45 * FIG_WIDTH_IN))
    summary = {}
    print(f"Background: {BACKGROUND}; points with signal > {MIN_SNR:g} x noise; "
          f"min. {MIN_POINTS} points per segment; bootstrap {N_BOOTSTRAP}x\n")

    for ax, name in zip(np.atleast_1d(axes), DATASETS):
        d = load_dataset(name)
        use = d["y"] > MIN_SNR * d["noise"]
        x, y = np.log10(d["I"][use]), np.log10(d["y"][use])
        sigma_log = d["sigma"][use] / (d["y"][use] * np.log(10.0))
        r = analyse(x, y, sigma_log)
        s_boot, ith_boot = bootstrap(x, y, sigma_log, r["n_segments"], rng)
        s_lo, s_med, s_hi = np.percentile(s_boot, [16, 50, 84])
        ith = None if r["log_ith"] is None else 10 ** r["log_ith"]
        ith_ok = ith_boot[np.isfinite(ith_boot)]
        a, b = r["segments"][r["steep_index"]][:2]
        I_steep = (10 ** x[a], 10 ** x[b - 1])
        pf_fit, xb, yb = fit_pa_fononband(x, y, sigma_log, pf)
        pf_boot = bootstrap_pf(x, y, sigma_log, pf, rng)
        pf_lo, pf_med, pf_hi = (np.percentile(pf_boot, [16, 50, 84]) if pf_boot.size
                                else (np.nan,) * 3)

        summary[name] = dict(
            pa_fononband_method=(None if pf_fit is None else dict(
                s_fit=pf_fit["slope"], s_peak_secant=pf_fit["s_peak"],
                window_kWcm2=[10 ** pf_fit["x_range"][0], 10 ** pf_fit["x_range"][1]],
                n_bins=pf_fit["n_points"], fraction=pf_fit["fraction"],
                r_squared=pf_fit["r_squared"], s_bootstrap_median=pf_med,
                s_bootstrap_16_84=[pf_lo, pf_hi], bin_dex=BIN_DEX)),
            s=r["s"], s_sd_fit=r["s_sd"], s_bootstrap_median=s_med,
            s_bootstrap_16_84=[s_lo, s_hi],
            steep_range_kWcm2=list(I_steep), n_points_steep=int(b - a),
            n_segments=r["n_segments"],
            segment_slopes=[seg[2] for seg in r["segments"]],
            ith_kWcm2=ith,
            ith_bootstrap_16_84=(list(np.percentile(ith_ok, [16, 84]))
                                 if ith_ok.size > 0.5 * len(ith_boot) else None),
            n_points_used=int(use.sum()), n_points_total=len(use),
            background=BACKGROUND, min_snr=MIN_SNR)

        print(f"{DATASETS[name]['label']}: {use.sum()} of {len(use)} points above "
              f"{MIN_SNR:g} x noise; {r['n_segments']} straight segments, slopes "
              + ", ".join(f"{seg[2]:.2f}" for seg in r["segments"]))
        print(f"  s = {r['s']:.2f} +/- {r['s_sd']:.2f} (fit)   bootstrap: {s_med:.2f} "
              f"[{s_lo:.2f}, {s_hi:.2f}]   steep segment {I_steep[0]:.3g}-{I_steep[1]:.3g} "
              f"kW/cm2 ({b - a} points)")
        if ith is not None:
            print(f"  I_th (intersection, as in Lee Fig. S3) = {ith:.3g} kW/cm2 (reported scale)")
        else:
            print("  I_th as in Lee cannot be determined: below the knee there are too few "
                  "points above the noise for a separate straight line.")
        if pf_fit is not None:
            print(f"  pa_fononband method ({BIN_DEX:g}-dex bins, secant >= "
                  f"{pf.SLOPE_FRACTION:g} x peak): s_fit = {pf_fit['slope']:.2f}, bootstrap "
                  f"{pf_med:.2f} [{pf_lo:.2f}, {pf_hi:.2f}]; s_peak = {pf_fit['s_peak']:.1f}; window "
                  f"{10 ** pf_fit['x_range'][0]:.3g}-{10 ** pf_fit['x_range'][1]:.3g} kW/cm2 "
                  f"({pf_fit['n_points']} bins)")
        print()

        # ---- figure ----
        shown = d["y"] > 0
        weak = shown & ~use
        ax.plot(d["I"][use], d["y"][use], "o", ms=3.2, mfc="none", mec=COLOR_DATA,
                mew=0.8, label="Measurement")
        ax.plot(d["I"][weak], d["y"][weak], "o", ms=3.0, mfc="none", mec="0.75", mew=0.7,
                label=rf"Below ${MIN_SNR:g}\sigma$ (not fitted)")
        for k, (a_k, b_k, m, c, _) in enumerate(r["segments"]):
            xs = np.array([x[a_k], x[b_k - 1]])
            if k == r["steep_index"] - 1 and r["log_ith"] is not None:
                xs[1] = r["log_ith"]                   # extend to the intersection
            if k == r["steep_index"] and r["log_ith"] is not None:
                xs[0] = r["log_ith"]
            steep = k == r["steep_index"]
            ax.plot(10 ** xs, 10 ** (m * xs + c), "-" if steep else "--",
                    color=COLOR_STEEP if steep else COLOR_OTHER, lw=1.4 if steep else 1.0,
                    zorder=5)
        # Method 2: binned data and the fitted line, as in the pa_fononband figure.
        ax.plot(10 ** xb, 10 ** yb, "s", ms=2.8, color="0.25", zorder=6)
        if pf_fit is not None:
            xs = np.array(pf_fit["x_range"])
            ax.plot(10 ** xs, 10 ** (pf_fit["slope"] * xs + pf_fit["intercept"]), "-",
                    color=COLOR_PF, lw=1.2, zorder=7)
        # Both slopes under the panel label (upper left is empty: the curve rises at the right).
        ax.text(0.04, 0.83, rf"$s_\mathrm{{Lee}} = {r['s']:.1f}$", transform=ax.transAxes,
                va="top", color=COLOR_STEEP)
        if pf_fit is not None:
            ax.text(0.04, 0.72, rf"$s_\mathrm{{fit}} = {pf_fit['slope']:.1f}$",
                    transform=ax.transAxes, va="top", color=COLOR_PF)
        if ith is not None:
            ax.axvline(ith, color=COLOR_OTHER, lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        # Less than two decades: labels at 1-2-5 instead of overlapping minor ticks.
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xlim(d["I"].min(), d["I"].max())
        ax.set_ylim(*_decade_limits(d["y"][shown]))
        ax.set_xlabel(r"Excitation intensity (kW cm$^{-2}$)")
        ax.text(0.04, 0.95, DATASETS[name]["label"], transform=ax.transAxes, va="top")
    np.atleast_1d(axes)[0].set_ylabel("800-nm emission (counts s$^{-1}$)")
    from matplotlib.lines import Line2D
    handles = np.atleast_1d(axes)[0].get_legend_handles_labels()[0] + [
        Line2D([], [], ls="none", marker="s", ms=2.8, color="0.25",
               label=f"Binned ({BIN_DEX:g} dex)"),
        Line2D([], [], color=COLOR_STEEP, lw=1.4, label="Lee et al. (segments)"),
        Line2D([], [], color=COLOR_PF, lw=1.2,
               label=rf"pa$\_$fononband method (secant $\geq {pf.SLOPE_FRACTION:g}\,s_\mathrm{{peak}}$)"),
    ]
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 0.0))
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig_s_fit_lee.{ext}")
    (OUT_DIR / "s_fit_lee.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"Saved: {OUT_DIR / 'fig_s_fit_lee.pdf'} (+ .png) and "
          f"{OUT_DIR / 's_fit_lee.json'}")
    if "inline" in plt.get_backend().lower():
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
