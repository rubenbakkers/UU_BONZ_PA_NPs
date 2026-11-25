# -*- coding: utf-8 -*-
"""Helling s van de gemeten S-curves, bepaald zoals Lee et al. (2021).

Lee et al. definieren de niet-lineariteit s via I_em ~ I_pomp^s: de helling van
de log-log-curve. De drempel volgt uit SI Fig. S3: rechte lijnen (in log-log)
door de punten onder en boven de drempel; het snijpunt is I_th, de helling van
de steile lijn is s.

Dit script doet dat automatisch:
  1. Alleen punten met emissie > MIN_SNR * ruis tellen mee (log van ruis
     heeft geen betekenis).
  2. De punten (gesorteerd op intensiteit) worden verdeeld in 2 of 3
     aaneengesloten stukken; per stuk een eigen gewogen rechte lijn in
     log10(E) vs log10(I). De knikpunten worden zo gekozen dat de totale chi^2
     minimaal is; tussen 2 en 3 stukken kiest het BIC (3 stukken alleen als de
     data ook verzadiging laten zien).
  3. s = helling van het steilste stuk. I_th = snijpunt van dat stuk met het
     stuk ervoor (alleen als er onder de knik genoeg significante punten zijn).
  4. Onzekerheid: bootstrap (punten opnieuw trekken, alles opnieuw fitten).

Los van het model: alleen numpy en matplotlib. Draait in ~10 s.
Uitvoer: results/fig_s_fit_lee.pdf/.png en results/s_fit_lee.json.
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
except NameError:                      # bv. geplakt in een console
    HERE = Path.cwd()
DATA_DIR, OUT_DIR = HERE / "data", HERE / "results"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Instellingen
# ---------------------------------------------------------------------------
DATASETS = {
    "cobolt_1060": dict(file="cobolt_1060_scurve.csv", label="Cobolt, 1060 nm"),
    "superk_1064": dict(file="superk_1064_scurve.csv", label="SuperK, 1064 nm"),
}
MIN_SNR = 3.0          # alleen punten met (net - achtergrond) > MIN_SNR * ruis
REL_SCATTER = 0.10     # extra relatieve spreiding tussen herhalingen (zoals fit_scurves.py)
MIN_POINTS = 5         # minimaal aantal punten per recht stuk
N_SEGMENTS = (2, 3)    # geprobeerde aantallen rechte stukken (BIC kiest)
# Achtergrond per filterwielstand (FW6) die nog in net_cps zit:
#   "summary": uit results/summary.json (fit van make_figures.py; modelafhankelijk)
#   "geen":    niets aftrekken
BACKGROUND = "summary"
N_BOOTSTRAP = 1000
RNG_SEED = 1

# Zelfde figuurstijl als pa_fononband (LaTeX-lettertype, geen titels).
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


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_dataset(name):
    cfg = DATASETS[name]
    path = DATA_DIR / cfg["file"]
    if not path.exists():
        raise FileNotFoundError(f"Meetbestand ontbreekt: {path}. Zet de CSV-bestanden van de "
                                "Drive (BdJ1, 24-11-2026) in de map data/ (zie README.md).")
    d = np.genfromtxt(path, delimiter=",", names=True)
    d = d[np.argsort(d["I_kWcm2"])]
    background = np.zeros(len(d))
    if BACKGROUND == "summary":
        path = OUT_DIR / "summary.json"
        if path.exists():
            per_fw = json.loads(path.read_text())[name]["background_per_FW"]
            background = np.array([per_fw.get(str(int(f)), 0.0) for f in d["fw6"]])
        else:
            print(f"  LET OP: {path} ontbreekt (eerst make_figures.py draaien); "
                  "geen achtergrond afgetrokken.")
    elif BACKGROUND != "geen":
        raise ValueError("BACKGROUND moet 'summary' of 'geen' zijn.")
    y = d["net_cps"] - background
    sigma = np.sqrt(d["noise_cps"] ** 2 + (REL_SCATTER * np.clip(y, 0, None)) ** 2)
    return dict(I=d["I_kWcm2"], y=y, noise=d["noise_cps"], sigma=sigma, fw=d["fw6"],
                background=background)


# ---------------------------------------------------------------------------
# Stuksgewijs lineaire fit (onafhankelijke rechte lijnen, zoals Lee Fig. S3)
# ---------------------------------------------------------------------------
def _segment_tables(x, y, w):
    """chi^2, helling, afsnijding en var(helling) van de gewogen rechte lijn door
    de punten a..b-1, voor alle 0 <= a < b <= n (matrices (n+1) x (n+1))."""
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
    """Beste verdeling in n_segments aaneengesloten stukken.
    Geeft (chi2_totaal, lijst van (a, b, helling, afsnijding, sd_helling))."""
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
        raise ValueError("Alleen 2 of 3 stukken.")
    if not np.isfinite(best):
        return np.inf, []
    segments = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        dof = max(count[a, b] - 2, 1)
        scale = max(chi2[a, b] / dof, 1.0)          # sd opschalen als chi2_red > 1
        segments.append((a, b, float(slope[a, b]), float(intercept[a, b]),
                         float(np.sqrt(var_slope[a, b] * scale))))
    return float(best), segments


def analyse(x, y, sigma_log, n_segments=None):
    """Kies het aantal stukken (BIC), bepaal s en (indien mogelijk) I_th."""
    n = len(x)
    options = {}
    for k in ([n_segments] if n_segments else N_SEGMENTS):
        chi2, segs = fit_segments(x, y, sigma_log, k)
        if segs:
            n_par = 2 * k + (k - 1)                      # lijnen + knikpunten
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
            # Snijpunt moet tussen de twee stukken liggen (met een kleine marge).
            if x[segs[steep - 1][0]] - 0.05 <= x_cross <= x[segs[steep][1] - 1] + 0.05:
                result["log_ith"] = float(x_cross)
    return result


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
# Hoofdprogramma
# ---------------------------------------------------------------------------
def _decade_limits(values):
    v = values[np.isfinite(values) & (values > 0)]
    return 10.0 ** np.floor(np.log10(v.min())), 10.0 ** np.ceil(np.log10(v.max()))


def main():
    rng = np.random.default_rng(RNG_SEED)
    plt.rcParams.update(STYLE)
    import logging
    logging.getLogger("fontTools").setLevel(logging.ERROR)
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(FIG_WIDTH_IN, 0.45 * FIG_WIDTH_IN))
    summary = {}
    print(f"Achtergrond: {BACKGROUND}; punten met signaal > {MIN_SNR:g} x ruis; "
          f"min. {MIN_POINTS} punten per stuk; bootstrap {N_BOOTSTRAP}x\n")

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

        summary[name] = dict(
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

        print(f"{DATASETS[name]['label']}: {use.sum()} van {len(use)} punten boven "
              f"{MIN_SNR:g} x ruis; {r['n_segments']} rechte stukken, hellingen "
              + ", ".join(f"{seg[2]:.2f}" for seg in r["segments"]))
        print(f"  s = {r['s']:.2f} +/- {r['s_sd']:.2f} (fit)   bootstrap: {s_med:.2f} "
              f"[{s_lo:.2f}, {s_hi:.2f}]   steile stuk {I_steep[0]:.3g}-{I_steep[1]:.3g} "
              f"kW/cm2 ({b - a} punten)")
        if ith is not None:
            print(f"  I_th (snijpunt, zoals Lee Fig. S3) = {ith:.3g} kW/cm2 (gerapporteerde schaal)")
        else:
            print("  I_th volgens Lee niet te bepalen: onder de knik te weinig punten "
                  "boven de ruis voor een eigen rechte lijn.")
        print()

        # ---- figuur ----
        shown = d["y"] > 0
        weak = shown & ~use
        ax.plot(d["I"][use], d["y"][use], "o", ms=3.2, mfc="none", mec=COLOR_DATA,
                mew=0.8, label="Measurement")
        ax.plot(d["I"][weak], d["y"][weak], "o", ms=3.0, mfc="none", mec="0.75", mew=0.7,
                label=rf"Below ${MIN_SNR:g}\sigma$ (not fitted)")
        for k, (a_k, b_k, m, c, _) in enumerate(r["segments"]):
            xs = np.array([x[a_k], x[b_k - 1]])
            if k == r["steep_index"] - 1 and r["log_ith"] is not None:
                xs[1] = r["log_ith"]                   # doortrekken tot het snijpunt
            if k == r["steep_index"] and r["log_ith"] is not None:
                xs[0] = r["log_ith"]
            steep = k == r["steep_index"]
            ax.plot(10 ** xs, 10 ** (m * xs + c), "-" if steep else "--",
                    color=COLOR_STEEP if steep else COLOR_OTHER, lw=1.4 if steep else 1.0,
                    zorder=5)
        xm = 0.5 * (x[a] + x[b - 1])
        ym = r["s"] * xm + r["segments"][r["steep_index"]][3]
        ax.annotate(rf"$s = {r['s']:.1f}$", xy=(10 ** xm, 10 ** ym), xytext=(-8, 6),
                    textcoords="offset points", ha="right", color=COLOR_STEEP)
        if ith is not None:
            ax.axvline(ith, color=COLOR_OTHER, lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        # Minder dan twee decaden: labels op 1-2-5 i.p.v. overlappende kleine ticks.
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_xlim(d["I"].min(), d["I"].max())
        ax.set_ylim(*_decade_limits(d["y"][shown]))
        ax.set_xlabel(r"Excitation intensity (kW cm$^{-2}$)")
        ax.text(0.04, 0.95, DATASETS[name]["label"], transform=ax.transAxes, va="top")
    np.atleast_1d(axes)[0].set_ylabel("800-nm emission (counts s$^{-1}$)")
    np.atleast_1d(axes)[0].legend(loc="lower right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig_s_fit_lee.{ext}")
    (OUT_DIR / "s_fit_lee.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"Opgeslagen: {OUT_DIR / 'fig_s_fit_lee.pdf'} (+ .png) en "
          f"{OUT_DIR / 's_fit_lee.json'}")
    if "inline" in plt.get_backend().lower():
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
