# -*- coding: utf-8 -*-
"""
figure2a_power_dependence.py
=====================

Genereert de Figure-2a-stijl plot (800-nm emissie vs. pompintensiteit
voor de 1%, 4% en 8% Tm-samples, allemaal bij 1064 nm).

Voor iedere sample kan afzonderlijk het aantal steps/punten in de
intensiteitsgrid worden ingesteld.

Output: pa_fig2a_style_v2.png
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

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


def plot_fig2a_style(
    I_values: np.ndarray,
    curves: Dict[str, Tuple[np.ndarray, float]],
    savepath: Optional[str] = None
) -> None:

    fig, ax = plt.subplots(figsize=(7.0, 5.5))

    for label, (emission, slope) in curves.items():
        ax.loglog(
            I_values,
            _safe_for_log(emission),
            "o-",
            markersize=3.5,
            label=f"{label},  s = {slope:.1f}"
        )

    ax.set_xlabel(r"Excitation intensity (kW cm$^{-2}$)")
    ax.set_ylabel("800-nm emission intensity (a.u.)")
    ax.set_title("Photon-avalanche power dependence")
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=200)

    plt.show()


def main() -> None:

    # ============================================================
    # INSTELLINGEN
    # ============================================================

    # Bereik van de excitatie-intensiteit
    I_min = 1.0
    I_max = 150.0

    # Aantal steps/punten PER plot/sample
    steps_1pct = 100
    steps_4pct = 1000
    steps_8pct = 5000

    # ============================================================
    # SAMPLE-INSTELLINGEN
    # ============================================================

    samples = [
        (1, "1% Tm, 1064 nm", steps_1pct),
        (2, "4% Tm, 1064 nm", steps_4pct),
        (3, "8% Tm, 1064 nm", steps_8pct),
    ]

    fig2a_curves: Dict[str, Tuple[np.ndarray, float]] = {}

    # ============================================================
    # BEREKENING
    # ============================================================

    # Omdat iedere sample een eigen aantal steps kan hebben,
    # wordt I_values ook per sample aangemaakt.
    #
    # De plotfunctie hieronder verwacht één I_values-array voor
    # alle curves. Daarom slaan we ook de intensiteitsgrid per
    # sample afzonderlijk op.

    all_I_values = []
    all_emissions = []

    for sid, label, steps in samples:

        # Eigen intensiteitsgrid voor deze sample
        I_values = np.logspace(
            np.log10(I_min),
            np.log10(I_max),
            steps
        )

        p = parameters_for_sample(sid)

        emission = power_dependence(I_values, p)

        slope = np.max(
            nonlinearity_slope(I_values, emission)
        )

        all_I_values.append(I_values)
        all_emissions.append((label, emission, slope))

        print(
            f"Sample {sid}: "
            f"c={p.concentration_percent:g}% | "
            f"steps={steps} | "
            f"W2={p.w2:.2f} | "
            f"s31={p.s31:.2f} | "
            f"Q22={p.q22:.3g} | "
            f"Q23={p.q23:.3g} | "
            f"s_max={slope:.2f}"
        )

    # ============================================================
    # PLOT
    # ============================================================

    fig, ax = plt.subplots(figsize=(7.0, 5.5))

    for I_values, (label, emission, slope) in zip(
        all_I_values,
        all_emissions
    ):

        ax.loglog(
            I_values,
            _safe_for_log(emission),
            "o-",
            markersize=3.5,
            label=f"{label},  s = {slope:.1f}"
        )

    ax.set_xlabel(r"Excitation intensity (kW cm$^{-2}$)")
    ax.set_ylabel("800-nm emission intensity (a.u.)")
    ax.set_title("Photon-avalanche power dependence")
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        "pa_fig2a_style_v2.png",
        dpi=200
    )

    plt.show()


if __name__ == "__main__":
    main()
