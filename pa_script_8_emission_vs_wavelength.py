# -*- coding: utf-8 -*-
"""
pa_script_8_emission_vs_wavelength.py
=======================================

Genereert alleen de plot van steady-state 800-nm emissie vs. pompgolflengte
(bij vaste pompintensiteiten 20 en 60 kW/cm^2, voor het 8% Tm-sample) uit het
oorspronkelijke script "photon_avalanche_rk4_corrected_v2.py". Fysica/model
komt uit pa_model.py (ongewijzigd overgenomen).

Output: pa_emission_vs_wavelength_v2.png
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt

from pa_model import (
    parameters_for_sample,
    wavelength_dependence,
)

_LOG_PLOT_FLOOR = 1e-300


def _safe_for_log(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), _LOG_PLOT_FLOOR, None)


def plot_emission_vs_wavelength(wavelengths_nm: np.ndarray,
                                emission_by_intensity: Dict[float, np.ndarray],
                                savepath: Optional[str] = None) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for I_pump, emission in emission_by_intensity.items():
        ax.semilogy(wavelengths_nm, _safe_for_log(emission), "o-", markersize=3.5,
                   label=fr"$I_p$ = {I_pump:.0f} kW cm$^{{-2}}$")
    ax.set_xlabel("Pump wavelength (nm)")
    ax.set_ylabel("Steady-state 800-nm emission (a.u.)")
    ax.set_title("PA emission vs. pump wavelength at fixed excitation intensity")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=200)


def main() -> None:
    p8 = parameters_for_sample(3)  # 8% Tm sample

    # 950-1250 nm brackets the derived GSA resonance (~943.5 nm, just below
    # the scan's low end) and the paper's actual 1064 nm operating point.
    # This plot needs the full steady-state population/emission scan (unlike
    # the resonance-factor-only plots), so `wavelength_dependence` is used
    # here as in the original.
    scan_wavelengths = np.linspace(950.0, 1250.0, 31)
    scan = wavelength_dependence(p8, I_pump_values=(20.0, 60.0), wavelengths_nm=scan_wavelengths)

    for I_pump in (20.0, 60.0):
        emis = scan.emission[I_pump]
        print(f"I_pump={I_pump:5.1f} kW/cm^2: emission ranges from "
              f"{emis.min():.3g} to {emis.max():.3g} across the scan "
              f"(peak at {scan.wavelengths_nm[np.argmax(emis)]:.1f} nm)")

    plot_emission_vs_wavelength(scan.wavelengths_nm, scan.emission,
                                savepath="pa_emission_vs_wavelength_v2.png")


if __name__ == "__main__":
    main()
