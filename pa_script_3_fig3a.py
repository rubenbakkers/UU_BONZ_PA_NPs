# -*- coding: utf-8 -*-
"""
pa_script_3_fig3a.py
======================

Genereert alleen de Figure-3a-stijl plot (800-nm emissie vs. pompintensiteit,
boven: effect van core/shell architectuur bij 8% Tm; onder: effect van
Tm3+-concentratie) uit het oorspronkelijke script
"photon_avalanche_rk4_corrected_v2.py". Fysica/model komt uit pa_model.py
(ongewijzigd overgenomen).

Dit is 1 figuur (1 .png) met 2 subplots -- net als in het origineel.

Output: pa_fig3a_style_v2.png
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

from pa_model import (
    parameters_for_sample,
    power_dependence,
    nonlinearity_slope,
)

_LOG_PLOT_FLOOR = 1e-300


def _safe_for_log(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), _LOG_PLOT_FLOOR, None)


def plot_fig3a_style(I_values: np.ndarray,
                     sample_curves: Dict[int, np.ndarray],
                     savepath: Optional[str] = None) -> None:
    # x = pump intensity (kW/cm^2, log), y = 800 nm emission (a.u., log,
    # floored above zero) -- same reasoning as plot_fig2a_style.
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 8.0), sharex=True)
    top_ids = [3, 4, 5]
    bottom_ids = [3, 7, 8]

    for sid in top_ids:
        p = parameters_for_sample(sid)
        slope = np.max(nonlinearity_slope(I_values, sample_curves[sid]))
        axes[0].loglog(I_values, _safe_for_log(sample_curves[sid]), "o-", markersize=3.5,
                       label=f"Sample {sid}: {p.concentration_percent:.0f}% Tm, s_max={slope:.1f}")

    for sid in bottom_ids:
        p = parameters_for_sample(sid)
        slope = np.max(nonlinearity_slope(I_values, sample_curves[sid]))
        axes[1].loglog(I_values, _safe_for_log(sample_curves[sid]), "o-", markersize=3.5,
                       label=f"Sample {sid}: {p.concentration_percent:.0f}% Tm, s_max={slope:.1f}")

    axes[0].set_title("8% Tm3+: effect of core/shell architecture")
    axes[1].set_title("Effect of Tm3+ concentration")
    axes[1].set_xlabel(r"Excitation intensity (kW cm$^{-2}$)")
    axes[0].set_ylabel("800-nm emission (a.u.)")
    axes[1].set_ylabel("800-nm emission (a.u.)")
    for ax in axes:
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200)


def main() -> None:
    I_values = np.logspace(np.log10(1.0), np.log10(150.0), 100)

    sample_ids_fig3 = [3, 4, 5, 6, 7, 8]
    sample_curves: Dict[int, np.ndarray] = {}

    for sid in sample_ids_fig3:
        p = parameters_for_sample(sid)
        emission = power_dependence(I_values, p)
        sample_curves[sid] = emission
        print(f"Sample {sid}: c={p.concentration_percent:5.1f}% | W2={p.w2:7.2f} s^-1")

    plot_fig3a_style(I_values, sample_curves, "pa_fig3a_style_v2.png")


if __name__ == "__main__":
    main()
