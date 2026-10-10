# -*- coding: utf-8 -*-
"""
figure2a_power_dependence.py
=====================

Figure 2a-style plot:
800-nm emission vs. pump intensity for 1%, 4% and 8% Tm samples
at 1064 nm.

The nonlinear exponent s is obtained from a linear regression in
log-log space over a window of +/- 0.10 decade around the knee.

The same regression that is used to calculate s is now also plotted
as a straight line over the corresponding fit window.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from pa_model import (
    parameters_for_sample,
    power_dependence,
    nonlinearity_slope_windowed_fit,
)

_LOG_PLOT_FLOOR = 1e-300


def _safe_for_log(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), _LOG_PLOT_FLOOR, None)


def plot_fig2a_style(
    I_values: np.ndarray,
    curves: Dict[str, Tuple[np.ndarray, float, float, float]],
    savepath: Optional[str] = None,
) -> None:
    """
    Plot the emission curves and the linear-regression fit used to
    determine s.

    curves[label] contains:
        emission
        slope
        I_lo
        I_hi
    """

    fig, ax = plt.subplots(figsize=(7.0, 5.5))

    for label, (emission, slope, I_lo, I_hi) in curves.items():

        # ------------------------------------------------------------
        # Original emission curve
        # ------------------------------------------------------------
        ax.loglog(
            I_values,
            _safe_for_log(emission),
            "o-",
            markersize=3.5,
            label=f"{label},  s = {slope:.1f}",
        )

        # ------------------------------------------------------------
        # Select the points that were used in the s fit
        # ------------------------------------------------------------
        mask = (I_values >= I_lo) & (I_values <= I_hi)

        I_fit = I_values[mask]
        emission_fit = _safe_for_log(emission[mask])

        if len(I_fit) >= 2:

            # --------------------------------------------------------
            # Reproduce the log-log linear regression:
            #
            # log10(I_emission) = s * log10(I_pump) + intercept
            # --------------------------------------------------------
            log_I = np.log10(I_fit)
            log_emission = np.log10(emission_fit)

            fit_slope, fit_intercept = np.polyfit(
                log_I,
                log_emission,
                1,
            )

            # --------------------------------------------------------
            # Generate a smooth line over exactly the fit window
            # --------------------------------------------------------
            I_line = np.logspace(
                np.log10(I_lo),
                np.log10(I_hi),
                100,
            )

            emission_line = 10 ** (
                fit_slope * np.log10(I_line)
                + fit_intercept
            )

            # --------------------------------------------------------
            # Plot regression line
            # --------------------------------------------------------
            ax.loglog(
                I_line,
                emission_line,
                "--",
                linewidth=2.2,
                label=f"{label} fit,  s = {fit_slope:.1f}",
            )

            # --------------------------------------------------------
            # Highlight the actual points used for the fit
            # --------------------------------------------------------
            ax.loglog(
                I_fit,
                emission_fit,
                "o",
                markersize=6,
                markerfacecolor="none",
                markeredgewidth=1.5,
            )

    ax.set_xlabel(r"Excitation intensity (kW cm$^{-2}$)")
    ax.set_ylabel("800-nm emission intensity (a.u.)")
    ax.set_title("Photon-avalanche power dependence")

    ax.grid(alpha=0.3, which="both")
    ax.legend()

    fig.tight_layout()

    if savepath:
        fig.savefig(
            savepath,
            dpi=200,
            bbox_inches="tight",
        )

    plt.show()


def main() -> None:

    I_values = np.logspace(
        np.log10(1.0),
        np.log10(150.0),
        100,
    )

    # ------------------------------------------------------------
    # Store:
    # emission, s, I_lo, I_hi
    # ------------------------------------------------------------
    fig2a_curves: Dict[
        str,
        Tuple[np.ndarray, float, float, float]
    ] = {}

    for sid, label in [
        (1, "1% Tm, 1064 nm"),
        (2, "4% Tm, 1064 nm"),
        (3, "8% Tm, 1064 nm"),
    ]:

        p = parameters_for_sample(sid)

        # Calculate emission curve
        emission = power_dependence(
            I_values,
            p,
        )

        # --------------------------------------------------------
        # Determine s using the same windowed-fit method as before
        # --------------------------------------------------------
        slope, I_knee, I_lo, I_hi, n_pts = (
            nonlinearity_slope_windowed_fit(
                I_values,
                p,
                emission_coarse=emission,
                half_decade=0.10,
            )
        )

        # Store fit information for plotting
        fig2a_curves[label] = (
            emission,
            slope,
            I_lo,
            I_hi,
        )

        print(
            f"Sample {sid}: "
            f"c={p.concentration_percent:g}% | "
            f"W2={p.w2:.2f} | "
            f"s31={p.s31:.2f} | "
            f"Q22={p.q22:.3g} | "
            f"Q23={p.q23:.3g} | "
            f"s(fit)={slope:.2f} "
            f"(venster I=[{I_lo:.2f},{I_hi:.2f}] "
            f"kW/cm^2 rond knee bij "
            f"{I_knee:.2f} kW/cm^2, "
            f"n={n_pts} punten)"
        )

    # ------------------------------------------------------------
    # Plot curves + regression lines
    # ------------------------------------------------------------
    plot_fig2a_style(
        I_values,
        fig2a_curves,
        "pa_fig2a_style_v2.png",
    )


if __name__ == "__main__":
    main()