# -*- coding: utf-8 -*-
"""
Standalone Figure 3b
====================

PA threshold intensity versus W2 relaxation rate.

This script ONLY makes Figure 3b.

No RK4 integration.
No numerical threshold extraction.
No wavelength model.
No other figures.

The model dependence is represented as

    I_th = slope * W2

with a zero intercept, consistent with the linear W2 dependence
discussed for Fig. 3b of Lee et al., Nature 589, 230-235 (2021).

Experimental data:
    8% Tm3+  : samples 3, 4, 5
    20% Tm3+ : samples 6, 7
    100% Tm3+: sample 8
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# EXPERIMENTAL DATA
# ============================================================

# Total W2 = W2_R + W2_non-radiative

W2_R = 162.60  # s^-1


# ------------------------------------------------------------
# 8% Tm3+
# ------------------------------------------------------------

W2_8 = np.array([
    162.60,        # sample 5
    162.60 + 40.7, # sample 4
    162.60 + 512.0 # sample 3
])

Ith_8 = np.array([
    4.9,
    6.4,
    23.3
])


# ------------------------------------------------------------
# 20% Tm3+
# ------------------------------------------------------------

W2_20 = np.array([
    162.60 + 585.0,  # sample 7
    162.60 + 976.0   # sample 6
])

Ith_20 = np.array([
    21.7,
    32.8
])


# ------------------------------------------------------------
# 100% Tm3+
# ------------------------------------------------------------

W2_100 = np.array([
    162.60 + 862.0   # sample 8
])

Ith_100 = np.array([
    29.6
])


# ============================================================
# LINEAR MODEL
# ============================================================

def fit_through_origin(x, y):
    """
    Fit

        y = m*x

    with the intercept constrained to zero.

    Returns
    -------
    m : float
        Slope.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    return np.sum(x * y) / np.sum(x * x)


# Fit each concentration separately
slope_8 = fit_through_origin(W2_8, Ith_8)
slope_20 = fit_through_origin(W2_20, Ith_20)
slope_100 = fit_through_origin(W2_100, Ith_100)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("Figure 3b linear fits")
print("---------------------")
print(f"8% Tm3+   slope = {slope_8:.6f} kW cm^-2 per s^-1")
print(f"20% Tm3+  slope = {slope_20:.6f} kW cm^-2 per s^-1")
print(f"100% Tm3+ slope = {slope_100:.6f} kW cm^-2 per s^-1")
print()


# ============================================================
# MODEL LINES
# ============================================================

W2_plot = np.linspace(0.0, 1500.0, 500)

Ith_model_8 = slope_8 * W2_plot
Ith_model_20 = slope_20 * W2_plot
Ith_model_100 = slope_100 * W2_plot


# ============================================================
# FIGURE 3b
# ============================================================

fig, ax = plt.subplots(figsize=(7.2, 5.6))


# ------------------------------------------------------------
# Model lines
# ------------------------------------------------------------

ax.plot(
    W2_plot,
    Ith_model_8,
    linestyle="--",
    linewidth=2.2,
    color="green",
    label="Model, 8% Tm"
)

ax.plot(
    W2_plot,
    Ith_model_20,
    linestyle="--",
    linewidth=2.2,
    color="red",
    label="Model, 20% Tm"
)

ax.plot(
    W2_plot,
    Ith_model_100,
    linestyle="--",
    linewidth=2.2,
    color="purple",
    label="Model, 100% Tm"
)


# ------------------------------------------------------------
# Experimental data
# ------------------------------------------------------------

ax.scatter(
    W2_8,
    Ith_8,
    s=70,
    color="green",
    edgecolor="black",
    linewidth=0.8,
    zorder=5,
    label="Experiment, 8% Tm"
)

ax.scatter(
    W2_20,
    Ith_20,
    s=70,
    color="red",
    edgecolor="black",
    linewidth=0.8,
    zorder=5,
    label="Experiment, 20% Tm"
)

ax.scatter(
    W2_100,
    Ith_100,
    s=70,
    color="purple",
    edgecolor="black",
    linewidth=0.8,
    zorder=5,
    label="Experiment, 100% Tm"
)


# ============================================================
# AXES / STYLE
# ============================================================

ax.set_xlabel(
    r"Relaxation rate, $W_2$ (s$^{-1}$)",
    fontsize=12
)

ax.set_ylabel(
    r"PA threshold, $I_{\mathrm{th}}$ (kW cm$^{-2}$)",
    fontsize=12
)

ax.set_xlim(0, 1500)
ax.set_ylim(0, 45)

ax.tick_params(
    axis="both",
    labelsize=11
)

ax.grid(
    True,
    alpha=0.25
)

ax.legend(
    fontsize=9,
    loc="upper left",
    frameon=False
)

fig.tight_layout()


# ============================================================
# SAVE + SHOW
# ============================================================

plt.savefig(
    "Figure_3b_corrected.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()