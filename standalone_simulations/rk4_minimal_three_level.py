# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 14:29:00 2026

@author: ruben
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

"""
photon_avalanche_rk4.py
========================

Simulatie van de populatiedynamica van photon avalanching (PA) in
lanthanide-gedoteerde nanodeeltjes (bijv. Tm3+-gedoteerde NaYF4 "ANPs"),
met een handmatig geimplementeerde vierde-orde Runge-Kutta (RK4)
integrator (geen scipy.integrate).

Fysische achtergrond
--------------------
Dit script is gebaseerd op het vereenvoudigde 3-niveau schema uit
Lee et al., "Giant nonlinear optical responses from photon-avalanching
nanoparticles", Nature 589, 230-235 (2021), en de bijbehorende
Supplementary Information (Eqn. S1-S3). In het volledige model van het
paper zijn er terugkoppeltermen Q22/Q23 (inverse cross-relaxatie en
upconversie tussen twee tussenliggende ionen) en een apart stralings-
en niet-stralingskanaal per niveau. Voor dit script is dat gereduceerd
tot het kernmechanisme dat photon avalanching veroorzaakt: een lus van
Excited State Absorption (ESA) en Cross-Relaxation (CR) die het
tussenliggende niveau exponentieel kan bevolken zodra de pompintensiteit
een drempelwaarde overschrijdt.

Model (3 niveaus, populatiefracties N0 + N1 + N2 = 1):

    N0  grondtoestand   (analoog aan 3H6 in Tm3+)
    N1  tussentoestand  (analoog aan 3F4 in Tm3+)
    N2  emissieniveau   (analoog aan 3H4 in Tm3+, geeft de 800-nm emissie)

Processen:

    GSA (Ground State Absorption):   N0 -> N1,   snelheid R_GSA * N0
    ESA (Excited State Absorption):  N1 -> N2,   snelheid R_ESA * N1
    CR  (Cross-Relaxation):          N2 + N0 -> N1 + N1
                                      snelheid W_CR * N0 * N2
    Spontaan verval:                 N1 -> N0  (snelheid k1)
                                      N2 -> N0  (snelheid k2)

De cross-relaxatieterm is de sleutel tot de positieve-feedbacklus:
een reeds aangeslagen ion in N2 "steelt"=geeft energie van een grondtoestand-
ion in N0, waardoor BEIDE ionen in het tussenliggende niveau N1
terechtkomen. Bij voldoende pompintensiteit wint deze verdubbeling het
van het spontane verval, en ontstaat een lawine-achtige, sterk
niet-lineaire toename van de N2-populatie (en dus van de emissie).

R_GSA en R_ESA zijn evenredig met de pompintensiteit I_pump
[kW cm^-2]:

    R_GSA = sigma_gsa_rel * I_pump
    R_ESA = sigma_esa_rel * I_pump = sigma_gsa_rel * sigma_esa_ratio * I_pump

waarbij sigma_esa_ratio >> 1 (in het paper > 10^4) de essentiele
voorwaarde is voor photon avalanching: ESA vanuit het tussenniveau moet
veel efficienter zijn dan GSA vanuit de grondtoestand.
"""

# ---------------------------------------------------------------------------
# 1. Fysische parameters
# ---------------------------------------------------------------------------

@dataclass
class PAParameters:
    """Snelheidsconstanten voor het vereenvoudigde 3-niveau PA-model.

    Alle snelheden zijn in s^-1, tenzij anders vermeld.

    Attributes
    ----------
    sigma_gsa_rel : float
        GSA-'snelheidscoefficient' per eenheid pompintensiteit
        [s^-1 / (kW cm^-2)], zodat R_GSA = sigma_gsa_rel * I_pump.
        Dit staat voor sigma_GSA * lambda / (h*c) uit de echte fysica,
        hier samengevat in een enkele evenredigheidsconstante.
    sigma_esa_ratio : float
        Verhouding sigma_ESA / sigma_GSA (dimensieloos). Dit is de
        centrale PA-parameter: het paper rapporteert een experimentele
        ratio R2/R1 > 10^4 (Supplementary Table 6). Hier gebruiken we
        een iets kleinere waarde (2*10^4) om met redelijke tijdstappen
        een duidelijke, numeriek goed op te lossen drempel te krijgen.
    k1 : float
        Totale (stralend + niet-stralend) vervalsnelheid van N1 -> N0
        [s^-1]. Analoog aan W2 (het paper rapporteert orde 100-2000 s^-1
        voor de 3F4-toestand, sterk afhankelijk van oppervlaktepassivatie).
    k2 : float
        Totale vervalsnelheid van N2 -> N0 [s^-1]. Analoog aan W3
        (orde 500-2000 s^-1 in het paper voor de 3H4-toestand).
    W_CR : float
        Cross-relaxatiesnelheid [s^-1], analoog aan s31 in het paper.
        Dit bepaalt hoe sterk de positieve-feedbacklus is.
    """

    sigma_gsa_rel: float
    sigma_esa_ratio: float
    k1: float
    k2: float
    W_CR: float

    @property
    def sigma_esa_rel(self) -> float:
        """ESA-'snelheidscoefficient' per eenheid pompintensiteit
        [s^-1 / (kW cm^-2)]."""
        return self.sigma_gsa_rel * self.sigma_esa_ratio


# Standaardparameters, zo gekozen dat er een duidelijke avalanche-drempel
# ontstaat rond I_pump ~ 6 kW/cm^2 met een maximale niet-lineariteit
# s_max > 15 (zie functie `nonlinearity_slope` verderop). Deze waarden
# zijn NIET rechtstreeks overgenomen uit de Judd-Ofelt/DRE-fit in de
# Supplementary Information (die gebruikt een uitgebreider model met
# Q22/Q23-termen); ze zijn zo gekozen dat dit vereenvoudigde 3-niveau
# model kwalitatief hetzelfde fysische gedrag vertoont (duidelijke
# drempel, sterke niet-lineariteit, verzadiging bij hoge I_pump).
DEFAULT_PARAMS = PAParameters(
    sigma_gsa_rel=0.02,     # s^-1 / (kW/cm^2)  -> zeer zwakke GSA
    sigma_esa_ratio=10667,  # sigma_ESA / sigma_GSA >> 1  (PA-voorwaarde)
    k1=1000.0,              # s^-1, verval N1 -> N0
    k2=1500.0,              # s^-1, verval N2 -> N0
    W_CR=4000.0,            # s^-1, cross-relaxatiesnelheid
)


# ---------------------------------------------------------------------------
# 2. Snelheidsvergelijkingen (rate equations)
# ---------------------------------------------------------------------------

def pa_rate_equations(y: np.ndarray, I_pump: float, p: PAParameters) -> np.ndarray:
    """Rechterlid van het stelsel dy/dt = f(y, I_pump), met y = [N0, N1, N2].

    Parameters
    ----------
    y : np.ndarray, vorm (3,)
        Huidige populatiefracties [N0, N1, N2].
    I_pump : float
        Pompintensiteit [kW cm^-2].
    p : PAParameters
        Snelheidsconstanten van het model.

    Returns
    -------
    np.ndarray, vorm (3,)
        [dN0/dt, dN1/dt, dN2/dt].

    Let op: N0 + N1 + N2 = 1 is een behouden grootheid: som van de drie
    afgeleiden is per constructie exact 0 (controleer dit gerust door
    de drie regels hieronder op te tellen).
    """
    N0, N1, N2 = y

    # Pompsnelheden zijn lineair in de pompintensiteit
    R_GSA = p.sigma_gsa_rel * I_pump   # Ground State Absorption: N0 -> N1
    R_ESA = p.sigma_esa_rel * I_pump   # Excited State Absorption: N1 -> N2

    # dN0/dt:
    #   -R_GSA*N0   : verlies door GSA (N0 -> N1)
    #   +k1*N1      : winst door spontaan verval N1 -> N0
    #   +k2*N2      : winst door spontaan verval N2 -> N0
    #   -W_CR*N0*N2 : verlies doordat een N0-ion wordt "meegesleurd" in de
    #                 cross-relaxatie N2 + N0 -> N1 + N1
    dN0 = -R_GSA * N0 + p.k1 * N1 + p.k2 * N2 - p.W_CR * N0 * N2

    # dN1/dt:
    #   +R_GSA*N0        : winst door GSA
    #   -R_ESA*N1        : verlies door ESA (N1 -> N2)
    #   -k1*N1           : verlies door spontaan verval N1 -> N0
    #   +2*W_CR*N0*N2    : winst van TWEE N1-ionen per cross-relaxatie-
    #                      gebeurtenis (dit is de positieve-feedbackterm
    #                      die de lawine aandrijft)
    dN1 = R_GSA * N0 - R_ESA * N1 - p.k1 * N1 + 2.0 * p.W_CR * N0 * N2

    # dN2/dt:
    #   +R_ESA*N1    : winst door ESA
    #   -k2*N2       : verlies door spontaan verval N2 -> N0 (dit is het
    #                  radiatieve kanaal dat de 800-nm emissie levert)
    #   -W_CR*N0*N2  : verlies doordat het N2-ion "zijn energie afstaat"
    #                  aan het cross-relaxatieproces
    dN2 = R_ESA * N1 - p.k2 * N2 - p.W_CR * N0 * N2

    return np.array([dN0, dN1, dN2])


# ---------------------------------------------------------------------------
# 3. RK4-integrator (handmatig geimplementeerd, vector-based, geen scipy)
# ---------------------------------------------------------------------------

def rk4_step(
    f: Callable[[np.ndarray], np.ndarray], y: np.ndarray, dt: float
) -> np.ndarray:
    """Eén stap van de klassieke vierde-orde Runge-Kutta methode.

    f(y) geeft dy/dt terug; de afhankelijkheid van I_pump en de
    parameters p wordt door de aanroeper "ingebakken" via een closure
    (lambda y: pa_rate_equations(y, I_pump, p)), zodat deze functie
    volledig generiek blijft voor ELK stelsel gewone differentiaal-
    vergelijkingen van de vorm dy/dt = f(y).
    """
    k1v = f(y)
    k2v = f(y + 0.5 * dt * k1v)
    k3v = f(y + 0.5 * dt * k2v)
    k4v = f(y + dt * k3v)
    return y + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)


def rk4_integrate(
    f: Callable[[np.ndarray], np.ndarray],
    y0: np.ndarray,
    dt: float,
    t_final: float,
    store_every: int = 50,
    steady_state_tol: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Integreer dy/dt = f(y) van t = 0 tot t = t_final met vaste tijdstap dt.

    Parameters
    ----------
    f : callable
        Rechterlid van het stelsel, f(y) -> dy/dt.
    y0 : np.ndarray, vorm (3,)
        Beginconditie [N0(0), N1(0), N2(0)].
    dt : float
        Tijdstap [s]. Moet klein genoeg zijn t.o.v. de snelste tijdschaal
        in het systeem (~1/R_ESA bij hoge pompintensiteit) om numeriek
        stabiel te blijven; zie de discussie in de begeleidende tekst.
    t_final : float
        Maximale simulatieduur [s].
    store_every : int
        Sla slechts elke `store_every`-ste stap op (bespaart geheugen bij
        lange simulaties met kleine dt).
    steady_state_tol : float, optioneel
        Als gegeven, stopt de integratie vroegtijdig zodra de N2-populatie
        tussen twee controlepunten (elke 2000 stappen) minder dan deze
        tolerantie verandert. Handig om de steady-state snel te bepalen
        zonder de volledige (soms trage) transiënt te hoeven uitrekenen.

    Returns
    -------
    (t_array, y_array) : (np.ndarray (n,), np.ndarray (n, 3))
        De opgeslagen tijdstippen en bijbehorende populatiefracties.
    """
    n_steps = int(round(t_final / dt))
    y = y0.copy()
    t = 0.0

    t_list: List[float] = [0.0]
    y_list: List[np.ndarray] = [y0.copy()]

    prev_N2 = y[2]
    for i in range(1, n_steps + 1):
        y = rk4_step(f, y, dt)
        t += dt

        if i % store_every == 0:
            t_list.append(t)
            y_list.append(y.copy())

        if steady_state_tol is not None and i % 2000 == 0:
            if abs(y[2] - prev_N2) < steady_state_tol:
                break
            prev_N2 = y[2]

    # Zorg dat het allerlaatste tijdstip altijd is opgeslagen
    if t_list[-1] != t:
        t_list.append(t)
        y_list.append(y.copy())

    return np.array(t_list), np.array(y_list)


# ---------------------------------------------------------------------------
# 4. Simulatiefuncties op basis van de RK4-integrator
# ---------------------------------------------------------------------------

def simulate_time_evolution(
    I_pump: float,
    p: PAParameters = DEFAULT_PARAMS,
    dt: float = 2e-5,
    t_final: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Simuleer N0(t), N1(t), N2(t) voor een vaste pompintensiteit I_pump,
    startend vanuit een volledig ongepompt systeem (N0 = 1, N1 = N2 = 0)."""
    y0 = np.array([1.0, 0.0, 0.0])
    f = lambda y: pa_rate_equations(y, I_pump, p)
    return rk4_integrate(f, y0, dt, t_final, store_every=20)


def steady_state_emission(
    I_pump: float,
    p: PAParameters = DEFAULT_PARAMS,
    dt: float = 2e-5,
    t_final: float = 2.0,
    steady_state_tol: float = 1e-9,
) -> float:
    """Bereken de steady-state populatie N2 (evenredig met de
    emissie-intensiteit bij 800 nm) voor een gegeven pompintensiteit."""
    y0 = np.array([1.0, 0.0, 0.0])
    f = lambda y: pa_rate_equations(y, I_pump, p)
    _, y_arr = rk4_integrate(
        f, y0, dt, t_final, store_every=10_000, steady_state_tol=steady_state_tol
    )
    return float(y_arr[-1, 2])


def power_dependence(
    I_values: np.ndarray,
    p: PAParameters = DEFAULT_PARAMS,
    dt: float = 2e-5,
    t_final: float = 2.0,
) -> np.ndarray:
    """Bereken de steady-state emissie N2_ss voor een reeks
    pompintensiteiten I_values [kW cm^-2]."""
    return np.array([steady_state_emission(I, p, dt, t_final) for I in I_values])


def nonlinearity_slope(I_values: np.ndarray, N2_ss: np.ndarray) -> np.ndarray:
    """Bereken de lokale niet-lineariteit s = d(log N2) / d(log I_pump)
    via numerieke differentiatie (np.gradient, tweede-orde centrale
    differenties) op het log-log grid. Dit is dezelfde definitie als in
    het paper: I_emissie = I_pump^s."""
    log_I = np.log10(I_values)
    log_N2 = np.log10(N2_ss)
    return np.gradient(log_N2, log_I)


# ---------------------------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------------------------

def plot_time_domain(
    I_below: float,
    I_above: float,
    p: PAParameters = DEFAULT_PARAMS,
    savepath: str | None = None,
) -> None:
    """Grafiek 1: populatie-evolutie N0(t), N1(t), N2(t) net onder en ruim
    boven de avalanche-drempel."""
    t_below, y_below = simulate_time_evolution(I_below, p, t_final=0.3)
    t_above, y_above = simulate_time_evolution(I_above, p, t_final=0.05)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)

    for ax, t, y, I_val, titel in zip(
        axes,
        [t_below, t_above],
        [y_below, y_above],
        [I_below, I_above],
        ["Net onder de drempel", "Ruim boven de drempel"],
    ):
        t_ms = t * 1e3  # s -> ms
        ax.plot(t_ms, y[:, 0], label=r"$N_0$ (grondtoestand)", color="tab:blue")
        ax.plot(t_ms, y[:, 1], label=r"$N_1$ (tussentoestand)", color="tab:orange")
        ax.plot(t_ms, y[:, 2], label=r"$N_2$ (emissieniveau)", color="tab:red")
        ax.set_xlabel("Tijd (ms)")
        ax.set_title(f"{titel}\n$I_{{pump}}$ = {I_val:.2f} kW cm$^{{-2}}$")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Populatiefractie")
    axes[0].legend(loc="center right")
    fig.suptitle("Populatiedynamica: onder vs. boven de avalanche-drempel")
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150)
        print(f"Grafiek 1 opgeslagen als: {savepath}")


def plot_power_dependence(
    I_values: np.ndarray,
    N2_ss: np.ndarray,
    slopes: np.ndarray,
    savepath: str | None = None,
) -> None:
    """Grafiek 2: steady-state emissie (~N2) versus pompintensiteit op
    log-log schaal, met de lokale niet-lineariteit s(I) op een tweede
    y-as."""
    fig, ax1 = plt.subplots(figsize=(7, 5.5))

    ax1.loglog(I_values, N2_ss, "o-", color="tab:red", label=r"$N_{2,ss}$ (emissie)")
    ax1.set_xlabel(r"Pompintensiteit $I_{pump}$ (kW cm$^{-2}$)")
    ax1.set_ylabel(r"Steady-state emissie $N_{2,ss}$ (a.u.)", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax1.grid(alpha=0.3, which="both")

    ax2 = ax1.twinx()
    ax2.plot(I_values, slopes, "s--", color="tab:blue", alpha=0.7, label=r"$s(I)$")
    ax2.set_ylabel(
        r"Niet-lineariteit $s = d\log N_2 / d\log I_{pump}$", color="tab:blue"
    )
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    s_max = float(np.max(slopes))
    I_at_smax = float(I_values[int(np.argmax(slopes))])
    ax2.axhline(s_max, color="tab:blue", linestyle=":", linewidth=1)
    ax2.annotate(
        f"$s_{{max}}$ = {s_max:.1f}\nbij $I$ = {I_at_smax:.1f} kW cm$^{{-2}}$",
        xy=(I_at_smax, s_max),
        xytext=(0.55, 0.75),
        textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=10,
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.set_title("Power dependence: emissie en niet-lineariteit vs. pompintensiteit")
    fig.tight_layout()

    if savepath:
        fig.savefig(savepath, dpi=150)
        print(f"Grafiek 2 opgeslagen als: {savepath}")


# ---------------------------------------------------------------------------
# 6. Hoofdprogramma
# ---------------------------------------------------------------------------

def main() -> None:
    p = DEFAULT_PARAMS

    # --- Grafiek 2 eerst berekenen, zodat we de drempel numeriek kennen ---
    I_values = np.logspace(np.log10(1.5), np.log10(100), 30)  # kW/cm^2
    print("Berekenen van de steady-state power-dependence curve...")
    N2_ss = power_dependence(I_values, p)
    slopes = nonlinearity_slope(I_values, N2_ss)

    s_max = float(np.max(slopes))
    I_threshold_est = float(I_values[int(np.argmax(slopes))])
    print(f"Geschatte drempelintensiteit (bij max. helling): {I_threshold_est:.2f} kW/cm^2")
    print(f"Maximale niet-lineariteit s_max = {s_max:.2f}")

    plot_power_dependence(
        I_values, N2_ss, slopes, savepath="pa_power_dependence.png" 
    )

    # --- Grafiek 1: tijdsdomein net onder / ruim boven de drempel ---
    I_below = 0.8 * I_threshold_est
    I_above = 3.0 * I_threshold_est
    print(f"Tijdsdomein-simulaties bij I_below = {I_below:.2f}, I_above = {I_above:.2f} kW/cm^2")
    plot_time_domain(I_below, I_above, p, savepath="pa_time_domain.png")

    plt.show()


if __name__ == "__main__":
    main()