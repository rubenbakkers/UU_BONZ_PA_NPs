# -*- coding: utf-8 -*-
"""
Gaussian Stark-functies voor de zes Tm3+-multiplets in Lee et al., figuur 1c.

Bron: Physical Chemistry.pdf, PDF-pagina 9, Tabel 4(B), kolom 'this work'.
Dit zijn berekende (ab initio) energieniveaus in beta-NaYF4, geen gemeten
lijnsterktes. E=0 is het laagste Starkniveau van 3H6.

S_J(E) = sum_i A_i exp[-(E-E_i)^2/(2 sigma^2)]

De amplitudes A_i volgen een Boltzmann-bezetting (zie BOLTZMANN_MODE):
A_i = exp[-(E_i - E_ref)/(k_B T)], zodat lagere niveaus zwaarder wegen.
sigma=8.5 cm^-1 is een modelparameter.
Elke multipletcurve wordt afzonderlijk op maximum 1 genormaliseerd.
De curves tonen niveauposities, geen absorptie- of emissiespectra:
voor een overgang geldt Delta E = E_boven - E_onder. Daarom is de
extra golflengte-as slechts de equivalente golflengte t.o.v. E=0.
Bij E=0 is die golflengte oneindig; daar wordt geen nm-tick geplaatst.

Het script toont het plot non-blocking (Spyder) en slaat de figuur op als PDF.
"""

from __future__ import annotations

from pathlib import Path
import os  # Toegevoegd voor het aanmaken van mappen en bestandspaden
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Instellingen
# ---------------------------------------------------------------------------

SIGMA_CM1 = 8.5

# True  -> iedere multipletcurve afzonderlijk genormaliseerd op max = 1.
# False -> originele som S(E) wordt geplot.
NORMALIZE_EACH = True

# Laat met dunne verticale lijnen zien waar de Starkniveaus liggen.
# Bij Boltzmann-bezetting is de streeplengte evenredig met de bezetting.
SHOW_STARK_LEVELS = True

# Boltzmann-bezetting van de Starkniveaus:
#   "off"       -> alle A_i = 1 (zoals voorheen)
#   "multiplet" -> bezetting binnen elk multiplet t.o.v. het laagste niveau
#                  van dat multiplet (thermalisatie per multiplet)
#   "global"    -> bezetting van alle niveaus t.o.v. E = 0 (3H6, laagste niveau)
#                  (thermisch evenwicht over alle multipletten)
BOLTZMANN_MODE = "multiplet"
TEMPERATURE_K = 300.0

K_B_CM1_PER_K = 0.6950348  # Boltzmann-constante in cm^-1 per K


# ---------------------------------------------------------------------------
# Stark-level energies uit Tabel 4 [cm^-1]
# ---------------------------------------------------------------------------

E_3H6_CM1 = np.array([
    0.0, 9.0, 54.0, 80.0, 122.0, 149.0, 187.0,
    223.0, 246.0, 296.0, 314.0, 405.0, 417.0,
])

E_3F4_CM1 = np.array([
    5634.0, 5661.0, 5689.0, 5717.0, 5730.0,
    5750.0, 5761.0, 5782.0, 5823.0,
])

E_3H5_CM1 = np.array([
    8207.0, 8218.0, 8267.0, 8288.0, 8317.0, 8348.0,
    8377.0, 8428.0, 8440.0, 8528.0, 8539.0,
])

E_3H4_CM1 = np.array([
    12546.0,
    12566.0,
    12588.0,
    12613.0,
    12642.0,
    12673.0,
    12721.0,
    12766.0,
    12776.0,
])

E_3F3_CM1 = np.array([
    14279.0,
    14317.0,
    14341.0,
    14361.0,
    14380.0,
    14398.0,
    14420.0,
])

E_3F2_CM1 = np.array([
    14936.0,
    14951.0,
    15018.0,
    15061.0,
    15084.0,
])


# ---------------------------------------------------------------------------
# Boltzmann-bezetting
# ---------------------------------------------------------------------------

def boltzmann_weights(levels_cm1: np.ndarray,
                      temperature_k: float,
                      reference_cm1: float | None = None) -> np.ndarray:
    """
    Relatieve Boltzmann-bezetting p_i = exp[-(E_i - E_ref) / (k_B T)].

    Elk Starkniveau telt hier als niet-gedegenereerd (Tm3+ is een
    niet-Kramers-ion, dus g_i = 1). E_ref = laagste niveau van de set
    indien reference_cm1 None is.
    """
    levels = np.asarray(levels_cm1, dtype=float)
    if temperature_k <= 0.0:
        raise ValueError("temperature_k moet groter zijn dan 0.")
    if reference_cm1 is None:
        reference_cm1 = levels.min()
    kT = K_B_CM1_PER_K * temperature_k
    return np.exp(-(levels - reference_cm1) / kT)


def stark_amplitudes(levels_cm1: np.ndarray) -> np.ndarray:
    """Amplitudes A_i volgens BOLTZMANN_MODE."""
    if BOLTZMANN_MODE == "off":
        return np.ones(len(levels_cm1), dtype=float)
    if BOLTZMANN_MODE == "multiplet":
        return boltzmann_weights(levels_cm1, TEMPERATURE_K)
    if BOLTZMANN_MODE == "global":
        return boltzmann_weights(levels_cm1, TEMPERATURE_K,
                                 reference_cm1=E_3H6_CM1.min())
    raise ValueError("BOLTZMANN_MODE moet 'off', 'multiplet' of 'global' zijn.")


A_3H6 = stark_amplitudes(E_3H6_CM1)
A_3F4 = stark_amplitudes(E_3F4_CM1)
A_3H5 = stark_amplitudes(E_3H5_CM1)
A_3H4 = stark_amplitudes(E_3H4_CM1)
A_3F3 = stark_amplitudes(E_3F3_CM1)
A_3F2 = stark_amplitudes(E_3F2_CM1)


# ---------------------------------------------------------------------------
# Gaussian functies
# ---------------------------------------------------------------------------

def stark_gaussian_sum(
        energy_cm1: np.ndarray | float,
        levels_cm1: np.ndarray,
        sigma_cm1: float,
        amplitudes: Optional[np.ndarray] = None
        ) -> np.ndarray | float:
    """
    Som van Gaussian lijnen rond de Starkniveaus:

        S(E) = sum_i A_i exp[-(E-E_i)^2 / (2 sigma^2)]

    Parameters
    ----------
    energy_cm1
        Energie / golfgetal [cm^-1].

    levels_cm1
        Stark-energieniveaus E_i [cm^-1].

    sigma_cm1
        Standaardafwijking van iedere Gaussian [cm^-1].

    amplitudes
        Relatieve amplitudes A_i. Indien None worden alle amplitudes 1.
    """

    energy = np.asarray(energy_cm1, dtype=float)
    levels = np.asarray(levels_cm1, dtype=float)

    if sigma_cm1 <= 0.0:
        raise ValueError("sigma_cm1 moet groter zijn dan 0.")

    if amplitudes is None:
        amplitudes = np.ones(len(levels), dtype=float)
    else:
        amplitudes = np.asarray(amplitudes, dtype=float)

    if amplitudes.shape != levels.shape:
        raise ValueError(
            "amplitudes moet evenveel elementen bevatten als levels_cm1."
        )

    delta = np.expand_dims(energy, axis=-1) - levels

    gaussians = np.exp(
        -0.5 * (delta / sigma_cm1) ** 2
    )

    result = np.sum(
        amplitudes * gaussians,
        axis=-1
    )

    if energy.ndim == 0:
        return float(result)

    return result


def S_3H6(energy_cm1, sigma_cm1=SIGMA_CM1):
    """Gaussian som van de 13 Starkniveaus van 3H6."""
    return stark_gaussian_sum(energy_cm1, E_3H6_CM1, sigma_cm1, A_3H6)


def S_3F4(energy_cm1, sigma_cm1=SIGMA_CM1):
    """Gaussian som van de 9 Starkniveaus van 3F4."""
    return stark_gaussian_sum(energy_cm1, E_3F4_CM1, sigma_cm1, A_3F4)


def S_3H5(energy_cm1, sigma_cm1=SIGMA_CM1):
    """Gaussian som van de 11 Starkniveaus van 3H5."""
    return stark_gaussian_sum(energy_cm1, E_3H5_CM1, sigma_cm1, A_3H5)


def S_3H4(
        energy_cm1: np.ndarray | float,
        sigma_cm1: float = SIGMA_CM1
        ) -> np.ndarray | float:
    """Gaussian som van de 9 Starkniveaus van 3H4."""
    return stark_gaussian_sum(
        energy_cm1,
        E_3H4_CM1,
        sigma_cm1,
        A_3H4
    )


def S_3F3(
        energy_cm1: np.ndarray | float,
        sigma_cm1: float = SIGMA_CM1
        ) -> np.ndarray | float:
    """Gaussian som van de 7 Starkniveaus van 3F3."""
    return stark_gaussian_sum(
        energy_cm1,
        E_3F3_CM1,
        sigma_cm1,
        A_3F3
    )


def S_3F2(
        energy_cm1: np.ndarray | float,
        sigma_cm1: float = SIGMA_CM1
        ) -> np.ndarray | float:
    """Gaussian som van de 5 Starkniveaus van 3F2."""
    return stark_gaussian_sum(
        energy_cm1,
        E_3F2_CM1,
        sigma_cm1,
        A_3F2
    )


def normalize_to_one(values: np.ndarray) -> np.ndarray:
    """Normaliseer een curve zodat het globale maximum gelijk is aan 1."""
    values = np.asarray(values, dtype=float)
    maximum = np.max(values)

    if maximum <= 0.0:
        raise ValueError("Curve kan niet worden genormaliseerd: max <= 0.")

    return values / maximum


# ---------------------------------------------------------------------------
# Conversie tussen golfgetal en golflengte
# ---------------------------------------------------------------------------

def cm1_to_nm(wavenumber_cm1):
    """
    Converteer golfgetal [cm^-1] naar golflengte [nm]:

        lambda[nm] = 1e7 / nu_tilde[cm^-1]
    """
    wavenumber_cm1 = np.asarray(wavenumber_cm1, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        wavelength_nm = 1.0e7 / wavenumber_cm1

    return wavelength_nm


def nm_to_cm1(wavelength_nm):
    """
    Inverse conversie van golflengte [nm] naar golfgetal [cm^-1].
    """
    wavelength_nm = np.asarray(wavelength_nm, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        wavenumber_cm1 = 1.0e7 / wavelength_nm

    return wavenumber_cm1


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def main() -> None:
    # Een kleine negatieve marge maakt ook de Gaussian rond E=0 zichtbaar.
    # Circa 0.5 cm^-1 per stap: ruim voldoende voor sigma=8.5 cm^-1.
    energy_cm1 = np.linspace(-100.0, 15500.0, 31201)
    multiplets = [
        (r"$^3H_6$", E_3H6_CM1, A_3H6, S_3H6),
        (r"$^3F_4$", E_3F4_CM1, A_3F4, S_3F4),
        (r"$^3H_5$", E_3H5_CM1, A_3H5, S_3H5),
        (r"$^3H_4$", E_3H4_CM1, A_3H4, S_3H4),
        (r"$^3F_3$", E_3F3_CM1, A_3F3, S_3F3),
        (r"$^3F_2$", E_3F2_CM1, A_3F2, S_3F2),
    ]
    fig, ax = plt.subplots(figsize=(12.0, 5.5))
    for label, levels, amps, spectrum in multiplets:
        values = spectrum(energy_cm1)
        if NORMALIZE_EACH:
            values = normalize_to_one(values)
        line, = ax.plot(energy_cm1, values, linewidth=1.5,
                        label=f"{label} ({len(levels)} levels)")
        if SHOW_STARK_LEVELS:
            for e, a in zip(levels, amps):
                # streeplengte evenredig met relatieve bezetting
                ax.axvline(e, ymin=0.0, ymax=0.02 + 0.10 * a,
                           color=line.get_color(), linewidth=1.0, alpha=0.75)

    ax.set_xlim(energy_cm1[0], energy_cm1[-1])
    ax.set_xlabel(r"Energy relative to lowest $^3H_6$ level (cm$^{-1}$)",
                  labelpad=8)
    ax_nm = ax.twiny()
    ax_nm.set_xlim(ax.get_xlim())
    ax_nm.xaxis.set_ticks_position("bottom")
    ax_nm.xaxis.set_label_position("bottom")
    ax_nm.spines["bottom"].set_position(("outward", 48))
    ax_nm.spines["bottom"].set_color("blue")
    for side in ("top", "left", "right"):
        ax_nm.spines[side].set_visible(False)
    wavelength_ticks_nm = np.array([10000, 3000, 1800, 1200, 1000, 800, 700, 650])
    ax_nm.set_xticks(nm_to_cm1(wavelength_ticks_nm))
    ax_nm.set_xticklabels([f"{lam:.0f}" for lam in wavelength_ticks_nm])
    ax_nm.set_xlabel("Equivalent wavelength relative to E = 0 (nm)",
                     color="blue", labelpad=7)
    ax_nm.tick_params(axis="x", colors="blue", direction="out", pad=3)

    if NORMALIZE_EACH:
        ax.set_ylabel("Normalized Gaussian sum")
        ax.set_ylim(0.0, 1.08)
    else:
        ax.set_ylabel(r"$S(E)$ (a.u.)")
        ax.set_ylim(bottom=0.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))

    if BOLTZMANN_MODE != "off":
        ax.set_title(f"Boltzmann ({BOLTZMANN_MODE}), T = {TEMPERATURE_K:.0f} K")

    fig.subplots_adjust(bottom=0.30, top=0.92, right=0.79)

    # -----------------------------------------------------------------------
    # PDF Opslaan
    # -----------------------------------------------------------------------

    # Specificeer de map en bestandsnaam.
    # (r"..." voorkomt dat Python de \ als ontsnappingsteken ziet)
    save_dir = str(Path(__file__).resolve().parent / "results")
    save_path = os.path.join(save_dir, "Gaussian_Stark_functies.pdf")

    # Zorg dat de map bestaat voordat we opslaan
    os.makedirs(save_dir, exist_ok=True)

    # Sla de afbeelding op. bbox_inches="tight" voorkomt dat bijv. de legenda wordt afgesneden
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    print(f"Figuur succesvol opgeslagen in: {save_path}")

    # -----------------------------------------------------------------------

    plt.show(block=False)
    plt.pause(0.1)


if __name__ == "__main__":
    main()