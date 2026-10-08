# -*- coding: utf-8 -*-
"""
pa_script_2_fig2b_risetime.py
===============================

Genereert alleen de Figure-2b-stijl plot (800-nm stijgtijd vs. pompintensiteit
voor het 8% Tm-sample) uit het oorspronkelijke script
"photon_avalanche_rk4_corrected_v2.py". Fysica/model komt uit pa_model.py
(ongewijzigd overgenomen).

Output: pa_fig2b_rise_time_v2.png
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from pa_model import (
    parameters_for_sample,
    power_dependence,
    pa_threshold_refined,
    rise_time_95,
)


def plot_fig2b_rise_time(I_values, rise_times_ms, peak_I, savepath=None):
    fig, ax = plt.subplots(figsize=(7.0, 4.8))

    ax.plot(I_values, rise_times_ms, "o-")

    ax.axvline(
        peak_I,
        linestyle="--",
        linewidth=1,
        label=fr"$I_{{peak}}$ = {peak_I:.1f} kW cm$^{{-2}}$"
    )

    ax.set_xlabel(r"Excitation intensity (kW cm$^{-2}$)")
    ax.set_ylabel("800-nm rise time (ms)")
    ax.set_title("Rise-time signature of photon avalanching")
    ax.grid(alpha=0.3, which="both")
    ax.legend()

    # Zorg dat matplotlib daadwerkelijk de y-as als tijd weergeeft
    ax.ticklabel_format(axis="y", style="plain")

    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=200)

    plt.show()


def main() -> None:
    I_values = np.logspace(np.log10(1.0), np.log10(150.0), 100)
    p8 = parameters_for_sample(3)  # 8% Tm sample

    threshold8, slope_low8, slope_high8 = pa_threshold_refined(p8, I_values)
    print(f"8% sample 3 model threshold (broken-line) = {threshold8:.2f} kW/cm^2")
    print(f"Broken-line slopes: below={slope_low8:.2f}, above={slope_high8:.2f}")

    rise_I = np.unique(np.concatenate([
    np.linspace(1, 1.7, 5),
    np.linspace(1.7, 2.8, 30),
    np.linspace(2.9, 4.0, 10)
    ])) * threshold8
    rise_times = np.array([rise_time_95(float(I), p8, t_final=5.0, max_t_final=200.0) for I in rise_I])

    peak_idx = np.nanargmax(rise_times)
    rise_peak_I = rise_I[peak_idx]
    plot_fig2b_rise_time(rise_I, rise_times * 1e3, rise_peak_I, "pa_fig2b_rise_time_v2.png")


if __name__ == "__main__":
    main()
