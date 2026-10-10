# -*- coding: utf-8 -*-
"""
wavelength_detuning_gaussian_cauchy.py
====================================

Genereert de golflengte-detuning-demonstratieplot (800-nm emissie vs.
pompintensiteit, 1 curve per pompgolflengte, voor het 8% Tm-sample) uit het
oorspronkelijke script "rk4_full_gaussian_demo_v2.py". Fysica/model
komt uit pa_model.py (ongewijzigd overgenomen: DRE's, RK4-integrator en
threshold-berekening blijven exact hetzelfde).

Deze versie maakt TWEE aparte plots, die dezelfde wavelength-dependent
response vergelijken voor twee verschillende spectrale lijnvormen:

  1. pa_wavelength_detuning_gaussian.png
     -- de bestaande Gaussian resonance line shape uit pa_model.py
        (`wavelength_gaussian_factor`), ongewijzigd.

  2. pa_wavelength_detuning_cauchy.png
     -- dezelfde berekening, maar met een Cauchy/Lorentzian line shape
        f(Delta_lambda) = 1 / (1 + (Delta_lambda / gamma)^2)
        met dezelfde resonantiegolflengtes (lambda0_gsa, lambda0_esa) en
        dezelfde breedteparameters (sigma_lambda_gsa/esa, hier als gamma
        gebruikt) als uitgangspunt.

Voor beide plots identiek: 8% Tm-sample, dezelfde pompintensiteiten,
dezelfde geteste pompgolflengtes en hetzelfde DRE/RK4-model. Alleen de
spectrale lijnvorm die de resonantiefactor f(lambda) bepaalt, verandert.

Threshold-bepaling (Ith): voor Gaussian ongewijzigd via `pa_threshold_refined`
(broken-line fit). Voor Cauchy bleek diezelfde broken-line fit onbetrouwbaar
(zie de uitgebreide toelichting bij `pa_threshold_cauchy` verderop) -- Cauchy's
staarten maken de avalanche-knik veel scherper, waardoor er geen goed
gedefinieerd "net-na-de-knik"-lijnstuk meer over is om de tweede rechte lijn
doorheen te fitten. Voor Cauchy wordt daarom de al in pa_model.py aanwezige,
raster-onafhankelijke `nonlinearity_slope_max_converged` gebruikt (geconvergeerde
locatie van de maximale log-log helling), consistent voor alle vier golflengtes.
Dit geeft, in tegenstelling tot de broken-line fit, een stabiel en monotoon
stijgend Ith* met toenemende detuning. In de Cauchy-plot heet dit "Ith*"
(met sterretje) om het onderscheid met de Gaussian-definitie expliciet te
houden.

Implementatie: de Gaussian-vorm zit in pa_model.py als de losse functie
`wavelength_gaussian_factor`, die door `rate_coefficient` (en dus door de
DRE's, de RK4-integratie en de threshold-bepaling) wordt aangeroepen. Om de
Cauchy-variant te testen zonder pa_model.py's rate equations of andere
fysica aan te passen, wordt in dit script *uitsluitend* die ene
resonantiefactor-functie tijdelijk in de pa_model-module vervangen (monkey
patch via een contextmanager) en na gebruik weer teruggezet. Er wordt verder
niets aan pa_model.py gewijzigd.
"""

#lambda0_esa is nu 1060 ipv 1064 om duidelijker te maken dat de afstand van gsa
# ook invloed heeft op I_th en de slope waarde (symmetrischer met 1030 en 1090nm)


from __future__ import annotations

import contextlib
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

import pa_model
from pa_model import (
    DEFAULT_SIGMA_LAMBDA_GSA_NM,
    DEFAULT_SIGMA_LAMBDA_ESA_NM,
    parameters_for_sample,
    power_dependence,
    nonlinearity_slope,
    pa_threshold_refined,
    nonlinearity_slope_max_converged,
)

_LOG_PLOT_FLOOR = 1e-300


def _safe_for_log(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), _LOG_PLOT_FLOOR, None)


# ---------------------------------------------------------------------------
# Spectral line shapes
# ---------------------------------------------------------------------------

# The Gaussian line shape already lives in pa_model.wavelength_gaussian_factor
# and is used as-is for Plot 1 -- no changes here.

def _wavelength_cauchy_factor(wavelength_m: float,
                              lambda0_m: float,
                              gamma_m: float) -> float:
    """Cauchy/Lorentzian resonance line shape, peak-normalized to 1.

        f(Delta_lambda) = 1 / (1 + (Delta_lambda / gamma)^2)

    Same call signature as `pa_model.wavelength_gaussian_factor`, so it can
    be swapped in as a drop-in replacement. Uses the same lambda0 and the
    same width parameter (here interpreted as gamma, the Cauchy scale/HWHM)
    as the Gaussian factor it replaces -- see module docstring.
    """
    detuning = wavelength_m - lambda0_m
    return float(1.0 / (1.0 + (detuning / gamma_m) ** 2))


@contextlib.contextmanager
def _lineshape(kind: str):
    """Temporarily select the resonance line shape used inside pa_model.

    This monkey-patches `pa_model.wavelength_gaussian_factor` -- the single
    function that `pa_model.rate_coefficient` (and therefore
    `gsa_esa_rate_coefficients` and the DRE's `pa_rate_equations`) calls to
    obtain the dimensionless resonance factor f(lambda). Swapping only this
    one function changes nothing about the rate equations, the RK4
    integrator, or the threshold calculation -- it only changes which
    formula is used to turn a detuning into a resonance factor. The
    original function is restored on exit, regardless of kind.
    """
    original = pa_model.wavelength_gaussian_factor
    if kind == "gaussian":
        replacement = original
    elif kind == "cauchy":
        replacement = _wavelength_cauchy_factor
    else:
        raise ValueError(f"Unknown lineshape kind: {kind!r}")

    pa_model.wavelength_gaussian_factor = replacement
    try:
        yield
    finally:
        pa_model.wavelength_gaussian_factor = original


# ---------------------------------------------------------------------------
# Threshold determination
# ---------------------------------------------------------------------------
#
# Gaussian pathway: unchanged, uses `pa_threshold_refined` (broken-line fit,
# same as before) -- this already gives stable, physically sensible results.
#
# Cauchy pathway: `pa_threshold_refined` turns out to be unreliable here.
# Cauchy's much heavier tails make the avalanche transition far sharper than
# for Gaussian (log-log slope at the knee ~90-115 for Cauchy vs. ~45-55 for
# Gaussian in this model). A dense linear scan directly across the knee
# confirms the transition is a genuine, continuous rise -- just extremely
# narrow (order ~1 kW/cm^2 wide, sitting on a baseline intensity of order
# ~25 kW/cm^2). At that sharpness, the broken-line method's "point right
# after the knee" is no longer inside the steep rising part of the curve --
# it already sits in the saturation roll-off, so the fitted "after" slope
# comes out smaller than the "before" slope (an inverted, unphysical fit).
# This was verified to persist under both grid refinement and much narrower
# zoom windows, so it is a structural property of how sharp the Cauchy
# knee is here, not a one-off sampling artifact.
#
# For such a near-step-function transition, the physically meaningful
# threshold is simply the pump intensity at which the transition happens,
# i.e. the (grid-independent) location of maximum log-log slope. This is
# exactly what `nonlinearity_slope_max_converged` computes: it is already
# part of pa_model.py, unmodified, and used elsewhere in that module for the
# same purpose (finding s_max robustly by zooming in until convergence). It
# is used here, uniformly for all four Cauchy pump wavelengths, so the four
# points being compared in the Cauchy plot are all defined the same way.
# Verified to give a stable, monotonically increasing threshold with
# increasing detuning (as physically expected), unlike the broken-line fit.

def pa_threshold_cauchy(p8, I_values, pump_wavelength_nm, emission_coarse):
    """Robust Ith for the Cauchy pathway: converged knee location.

    See module-level note above for why this differs from the Gaussian
    pathway's `pa_threshold_refined`. Returns (Ith, s_max) so the console
    log can still report the achieved slope at threshold.
    """
    s_max, I_at_smax, _n_iter = nonlinearity_slope_max_converged(
        I_values, p8, pump_wavelength_nm=pump_wavelength_nm,
        emission_coarse=emission_coarse,
        n_refine=25, max_iter=4, rel_tol=2e-2)
    return float(I_at_smax), float(s_max)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Formula strings shown directly on the plot, so the line shape used is
# visible on the figure itself (not only in the title/filename).
_LINESHAPE_FORMULA = {
    "gaussian": r"$f(\Delta\lambda) = \exp\left(-\dfrac{\Delta\lambda^2}{2\sigma^2}\right)$",
    "cauchy": r"$f(\Delta\lambda) = \dfrac{1}{1 + (\Delta\lambda/\gamma)^2}$",
}


def plot_wavelength_detuning(I_values: np.ndarray,
                             curves: Dict[float, np.ndarray],
                             thresholds: Dict[float, float],
                             lineshape_label: str,
                             lineshape_kind: str,
                             savepath: Optional[str] = None) -> None:
    """Emission vs. PUMP INTENSITY, one curve per PUMP WAVELENGTH."""
    ith_label = "Ith" if lineshape_kind == "gaussian" else "Ith*"
    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    colors = plt.cm.tab20(np.linspace(0, 1, len(curves)))

    for (lam, emission), color in zip(curves.items(), colors):
        ax.loglog(
            I_values,
            _safe_for_log(emission),
            "o-",
            markersize=3.5,
            color=color,
            label=f"{lam:.0f} nm ({ith_label} = {thresholds[lam]:.1f} kW cm$^{{-2}}$)"
        )
    ax.set_xlabel(r"Excitation intensity (kW cm$^{-2}$)")
    ax.set_ylabel("800-nm emission intensity (a.u.)")
    ax.set_xlim(I_values.min(), I_values.max())
    # ax.set_title("Effect of pump-wavelength detuning on PA response\n"
    #              f"({lineshape_label} resonance line shape)", fontsize=11)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.12),
    ncol=3
    )

    # Explicit on-figure label of the spectral line shape used, with formula.
    # formula = _LINESHAPE_FORMULA[lineshape_kind]
    # box_text = f"Line shape: {lineshape_label}\n{formula}"
    # if lineshape_kind == "cauchy":
        # box_text += ("\n\nIth* = converged knee location\n(max. log-log slope), not the\n"
                     # "broken-line fit -- see script docs")
    # ax.text(0.03, 0.97, box_text,
            # transform=ax.transAxes, ha="left", va="top", fontsize=9,
            # bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
            #          edgecolor="0.6", alpha=0.9))

    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, format="pdf",bbox_inches="tight")


# ---------------------------------------------------------------------------
# Curve generation (identical for both line shapes; only f(lambda) differs)
# ---------------------------------------------------------------------------

def compute_detuning_curves(I_values: np.ndarray,
                            p8,
                            wavelengths_nm,
                            kind: str,
                            label: str):
    """Compute emission curves + thresholds for one line-shape kind.

    Everything here (sample, intensities, tested wavelengths, DRE/RK4 model,
    threshold routine) is identical between the Gaussian and Cauchy runs;
    only the resonance factor f(lambda) used internally differs, via the
    `_lineshape` context manager.
    """
    curves: Dict[float, np.ndarray] = {}
    thresholds: Dict[float, float] = {}

    print(f"\n=== Wavelength-detuning demonstration (8% Tm sample, "
          f"{label} line shape, "
          f"sigma_lambda GSA={DEFAULT_SIGMA_LAMBDA_GSA_NM:.0f} nm, "
          f"ESA={DEFAULT_SIGMA_LAMBDA_ESA_NM:.0f} nm) ===")

    with _lineshape(kind):
        for lam in wavelengths_nm:
            emission_lam = power_dependence(I_values, p8, pump_wavelength_nm=lam)
            curves[lam] = emission_lam
            max_slope = np.max(nonlinearity_slope(I_values, emission_lam))

            if kind == "gaussian":
                # Unchanged: same broken-line threshold fit as before.
                try:
                    Ith_lam, slope_below, slope_above = pa_threshold_refined(
                        p8, I_values, emission_coarse=emission_lam,
                        pump_wavelength_nm=lam)
                except ValueError:
                    Ith_lam, slope_below, slope_above = np.nan, np.nan, np.nan
                thresholds[lam] = Ith_lam
                note = ("" if np.isnan(Ith_lam) or slope_above > slope_below else
                        "  <-- WARNING: slope_after <= slope_before, broken-line "
                        "knee fit likely unreliable here")
                print(f"  lambda_pump = {lam:6.1f} nm | Ith = {Ith_lam:7.2f} kW/cm^2 | "
                      f"s_max = {max_slope:6.2f} | E(150 kW/cm^2) = {emission_lam[-1]:.3g}"
                      + note)
            else:  # kind == "cauchy"
                # Robust threshold: converged knee location (see note above
                # `pa_threshold_cauchy`). Uses the same DRE/RK4 model and the
                # same I_values -- only the extraction method differs.
                Ith_lam, s_max_converged = pa_threshold_cauchy(
                    p8, I_values, lam, emission_coarse=emission_lam)
                thresholds[lam] = Ith_lam
                print(f"  lambda_pump = {lam:6.1f} nm | Ith = {Ith_lam:7.2f} kW/cm^2 "
                      f"(converged knee location) | s_max = {s_max_converged:6.2f} | "
                      f"E(150 kW/cm^2) = {emission_lam[-1]:.3g}")

    return curves, thresholds


def main() -> None:
    I_values = np.logspace(np.log10(10.0), np.log10(150.0), 100)
    p8 = parameters_for_sample(3)  # 8% Tm sample
    wavelengths_nm = [
    850.0,
    900.0,
    930.0,
    960.0,
    980.0,
    1000.0,
    1030.0, 1040, 1050,
    1060.0, 1070,1080,
    1090.0,
    1130.0,
    1180.0
    ]      

    # --- Plot 1: existing Gaussian line shape, unchanged implementation ---
    gauss_curves, gauss_thresholds = compute_detuning_curves(
        I_values, p8, wavelengths_nm, kind="gaussian", label="Gaussian")
    plot_wavelength_detuning(I_values, gauss_curves, gauss_thresholds,
                             lineshape_label="Gaussian",
                             lineshape_kind="gaussian",
                             savepath="pa_wavelength_detuning_gaussian.pdf")

    # # --- Plot 2: Cauchy/Lorentzian line shape, same lambda0/width params ---
    # cauchy_curves, cauchy_thresholds = compute_detuning_curves(
    #     I_values, p8, wavelengths_nm, kind="cauchy", label="Cauchy/Lorentzian")
    # plot_wavelength_detuning(I_values, cauchy_curves, cauchy_thresholds,
    #                          lineshape_label="Cauchy/Lorentzian",
    #                          lineshape_kind="cauchy",
    #                          savepath="pa_wavelength_detuning_cauchy.png")


if __name__ == "__main__":
    main()