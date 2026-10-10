# -*- coding: utf-8 -*-
"""
esa_resonance_scan.py
==============================

Genereert alleen de ESA-resonantieprofiel-plot (resonantiefactor f(lambda)
vs. pompgolflengte, voor de ESA-overgang van het 8% Tm-sample) uit het
oorspronkelijke script "rk4_full_gaussian_demo_v2.py". Fysica/model
komt uit pa_model.py (ongewijzigd overgenomen).

Output: pa_resonance_esa_v2.png
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from pa_model import (
    parameters_for_sample,
    resonance_factors,
)


def plot_resonance_factor_vs_wavelength(wavelengths_nm: np.ndarray,
                                        factor: np.ndarray,
                                        label: str,
                                        lambda0_nm: float,
                                        sigma_lambda_nm: float,
                                        color: str,
                                        savepath: Optional[str] = None) -> None:
    """Resonance factor f(lambda) vs. pump wavelength for a single
    transition (here: GSA)."""
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(wavelengths_nm, factor, "-", color=color, linewidth=2.0)
    ax.axvline(lambda0_nm, linestyle="--", color="gray", linewidth=1,
              label=fr"$\lambda_0$ = {lambda0_nm:.1f} nm")
    ax.axvspan(lambda0_nm - sigma_lambda_nm, lambda0_nm + sigma_lambda_nm,
              color=color, alpha=0.08, label=fr"$\pm\,\sigma_\lambda$ = {sigma_lambda_nm:.0f} nm")
    ax.set_xlabel("Pump wavelength (nm)")
    ax.set_ylabel(f"{label} resonance factor  f(" + r"$\lambda$)  (dimensionless)")
    ax.set_title(f"{label} resonance profile (illustrative, phenomenological)")
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200)


def main() -> None:
    p8 = parameters_for_sample(3)  # 8% Tm sample

    # 950-1250 nm brackets the derived GSA resonance (~943.5 nm, just below
    # the scan's low end) and the paper's actual 1064 nm operating point.
    # (Only the resonance factor f_ESA(lambda) is needed for this plot --
    # identical to the corresponding slice of `wavelength_dependence` in the
    # original script, computed directly here to avoid the unrelated
    # steady-state/emission work that function also does for other plots.)
    scan_wavelengths = np.linspace(950.0, 1250.0, 31)
    f_esa = np.empty_like(scan_wavelengths)
    for i, lam in enumerate(scan_wavelengths):
        _, f_esa[i] = resonance_factors(float(lam), p8)

    print(f"f_ESA peak = {f_esa.max():.4f} at "
          f"{scan_wavelengths[np.argmax(f_esa)]:.1f} nm "
          f"(lambda0_esa = {p8.lambda0_esa_nm:.1f} nm)")

    plot_resonance_factor_vs_wavelength(
        scan_wavelengths, f_esa, "ESA", p8.lambda0_esa_nm, p8.sigma_lambda_esa_nm,
        color="tab:orange", savepath="pa_resonance_esa_v2.png",
    )


if __name__ == "__main__":
    main()
