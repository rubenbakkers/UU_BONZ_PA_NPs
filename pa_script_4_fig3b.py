# -*- coding: utf-8 -*-
"""
pa_script_4_fig3b.py
======================

Genereert alleen de Figure-3b-stijl plot (PA-drempelintensiteit vs. W2) uit
het oorspronkelijke script "photon_avalanche_rk4_corrected_v2.py".
Fysica/model komt uit pa_model.py (ongewijzigd overgenomen).

Let op: dit is de rekenkundig zwaarste plot van het origineel (dichte W2- en
pompgrids, met tot miljoenen RK4-stappen per punt bij hoge Tm3+-concentratie
via `max_steps` in `fig3b_model_curve`). Dat is ongewijzigd overgenomen
gedrag uit het origineel, dus deze berekening kan enkele minuten duren --
precies zoals in het oorspronkelijke script.

Output: pa_fig3b_style_v2.png
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from pa_model import (
    W2_R,
    SI_SAMPLES,
    parameters_for_sample,
    power_dependence,
    pa_threshold_refined,
    build_parameters,
)


def fig3b_model_curve(concentration_percent: float,
                      W2_values: np.ndarray,
                      I_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the DRE threshold as a function of total W2.

    Figure 3b is a threshold-vs-W2 plot.  The previous implementation used
    only three W2 points and only eight pump-intensity points.  That is not
    sufficient for the broken-line threshold extraction: at large W2 the
    avalanche knee can fall between the coarse intensity samples, causing
    `pa_threshold_refined` to raise ValueError and the model line to terminate
    prematurely.

    Here the threshold is evaluated on a dense logarithmic pump grid.  The
    W2 dependence is then fitted with a straight line, as in the linear
    threshold-vs-W2 relationship shown in Fig. 3b.
    """
    W2_values = np.asarray(W2_values, dtype=float)
    I_grid = np.asarray(I_grid, dtype=float)

    # The numerical stiffness increases strongly with concentration because
    # s31 ~ c^2.  Keep a larger RK4 step budget for the high-Tm branches.
    max_steps = int(60_000 * max(1.0, (concentration_percent / 8.0) ** 2))

    # Use a dense pump grid.  It must extend well above the largest expected
    # threshold so the high-W2 end of the branch is always resolved.
    I_min = max(float(I_grid[0]), 1e-3)
    I_max = max(float(I_grid[-1]), 150.0)
    I_dense = np.logspace(np.log10(I_min), np.log10(I_max), 80)

    thresholds = np.full(W2_values.shape, np.nan, dtype=float)

    for j, W2_total in enumerate(W2_values):
        W2_nr = max(0.0, W2_total - W2_R)

        # The measured SI samples have W3_NR approximately 2*W2_NR.
        # This is the relation used to interpolate/extrapolate the W2 scan.
        W3_nr = 2.0 * W2_nr

        p = build_parameters(
            concentration_percent,
            w2_nr=W2_nr,
            w3_nr=W3_nr,
        )

        emission = power_dependence(
            I_dense,
            p,
            max_total_steps=max_steps,
        )

        try:
            Ith, _, _ = pa_threshold_refined(
                p,
                I_dense,
                emission_coarse=emission,
                zoom_factor=4.0,
                n_refine=40,
                pre_points=10,
                post_points=8,
            )
            thresholds[j] = Ith
        except ValueError:
            # If the threshold cannot be resolved for a particular parameter
            # set, leave it as NaN rather than fabricating an extrapolation.
            thresholds[j] = np.nan

    return W2_values, thresholds


def _linear_model_fit(x: np.ndarray,
                      y: np.ndarray) -> Tuple[float, float]:
    """Return slope and intercept for a finite-data linear fit."""
    mask = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(mask) < 2:
        raise ValueError("Need at least two finite points for a linear fit.")
    slope, intercept = np.polyfit(x[mask], y[mask], 1)
    return float(slope), float(intercept)


def plot_fig3b_style(model_results: List[Tuple[int, float, float, Optional[float]]],
                     savepath: Optional[str] = None) -> None:
    """Plot the corrected Fig. 3b-style threshold-vs-W2 model.

    The experimental points are the measured thresholds from Table S3.
    The dashed DRE branches are *linear fits to densely sampled DRE
    threshold calculations*, rather than a line through only three arbitrary
    W2 points.
    """
    fig, ax = plt.subplots(figsize=(7.2, 5.6))

    # Dense W2 grid gives an actual fit rather than a three-point polyline.
    W2_grid = np.linspace(W2_R, 1500.0, 18)

    # Dense pump grid used internally by fig3b_model_curve; these values set
    # the requested minimum/maximum range only.
    I_grid = np.logspace(np.log10(1.0), np.log10(150.0), 80)

    groups = {
        8.0: [3, 4, 5],
        20.0: [6, 7],
        100.0: [8],
    }

    for concentration, ids in groups.items():
        x_model, y_model = fig3b_model_curve(
            concentration,
            W2_grid,
            I_grid,
        )

        mask = np.isfinite(y_model)
        if np.count_nonzero(mask) >= 2:
            slope, intercept = _linear_model_fit(
                x_model[mask],
                y_model[mask],
            )

            x_fit = np.linspace(
                x_model[mask].min(),
                x_model[mask].max(),
                200,
            )
            y_fit = slope * x_fit + intercept

            ax.plot(
                x_fit,
                y_fit,
                linestyle="--",
                linewidth=1.8,
                label=(
                    f"DRE model, {concentration:.0f}% Tm "
                    f"(slope={slope:.3g})"
                ),
            )

            print(
                f"Fig. 3b DRE fit, {concentration:.0f}% Tm: "
                f"Ith = {slope:.6g} W2 + {intercept:.6g}"
            )

    # Experimental threshold values from Table S3.
    for concentration, ids in groups.items():
        xs, ys = [], []
        for sid, w2, model_thr, measured_thr in model_results:
            if sid in ids and measured_thr is not None:
                xs.append(w2)
                ys.append(measured_thr)

        if xs:
            ax.scatter(
                xs,
                ys,
                s=62,
                label=f"Experiment, {concentration:.0f}% Tm",
            )

    ax.set_xlabel(r"Relaxation rate, $W_2$ (s$^{-1}$)")
    ax.set_ylabel(r"PA threshold, $I_{th}$ (kW cm$^{-2}$)")
    ax.set_title(r"PA threshold intensity versus $W_2$")
    ax.set_xlim(0, 1500)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=200)

    plt.close(fig)


def main() -> None:
    I_values = np.logspace(np.log10(1.0), np.log10(150.0), 100)

    sample_ids_fig3 = [3, 4, 5, 6, 7, 8]
    model_results: List[Tuple[int, float, float, Optional[float]]] = []

    for sid in sample_ids_fig3:
        p = parameters_for_sample(sid)
        emission = power_dependence(I_values, p)
        try:
            threshold, _, _ = pa_threshold_refined(p, I_values, emission_coarse=emission)
        except ValueError:
            threshold = np.nan
        measured = SI_SAMPLES[sid].threshold_measured_kWcm2
        model_results.append((sid, p.w2, threshold, measured))
        print(
            f"Sample {sid}: c={p.concentration_percent:5.1f}% | W2={p.w2:7.2f} s^-1 | "
            f"model Ith={threshold:7.2f} | measured Ith="
            f"{measured if measured is not None else np.nan:7.2f}"
        )

    plot_fig3b_style(model_results, "pa_fig3b_style_v2.png")
    plt.close("all")


if __name__ == "__main__":
    main()
