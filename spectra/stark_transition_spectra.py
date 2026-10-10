# -*- coding: utf-8 -*-
"""Tm3+: van Starkniveaus naar spectra van overgangen.

Uitgangspunt: de zes energie-arrays uit het aangeleverde script, ongewijzigd.
De genoemde bron van die arrays (Physical Chemistry.pdf, Tabel 4B) is niet
meegeleverd en is hier niet opnieuw geverifieerd.

Lee et al., Nature 589 (2021), doi:10.1038/s41586-020-03092-9:
  PA fig. 1c / Methods: 1064 nm ESA voornamelijk 3F4 -> 3F2;
  1450 nm ESA: 3F4 -> 3H4; 1064 nm GSA is fononondersteund.
  SI Tabellen S6/S8: sigma_ESA(1064) = 6.4e-25 m2 = 6.4e-21 cm2;
                    sigma_GSA(1064) = 6.0e-29 m2 = 6.0e-25 cm2.
Dit zijn parameters van hun fit, geen gemeten Starklijnsterktes.

Opbouw van een bandspectrum (per overgang lower -> upper):

  1. Stick-spectrum: voor elk paar Starkniveaus (i, j) een deltalijn op
     DeltaE_ij = E_upper,j - E_lower,i met gewicht w_ij = p_i * r_ij.
       p_i  = Boltzmann-bezetting van beginniveau i, uitsluitend wanneer het
              beginmultiplet 3H6 of 3F4 is en BOLTZMANN_MODE = "multiplet";
              voor andere beginmultiplets of modus "off": gelijk gewicht.
              Het eindmultiplet krijgt geen Boltzmannweging.
       r_ij = relatieve lijnsterkte (standaard 1/aantal eindniveaus).
  2. Convolutie van de sticks met de lijnvorm (expliciet, via de
     convolutiestelling op een uniform golfgetalraster):
        spectrum = sticks (*) Gauss(sigma) (*) Lorentz(HWHM)
     Gauss = inhomogene verbreding (+ eventueel instrumentresolutie, die in
     kwadratuur bij sigma wordt opgeteld); Lorentz = homogene verbreding.
     Beide tegelijk geeft een Voigt-profiel. Met LORENTZ_HWHM_CM1 = 0 is het
     resultaat een puur Gaussiaanse som, zoals in de vorige versie.

Modelaannamen:
  * snelle thermalisatie BINNEN 3H6 en 3F4 wanneer ze het beginmultiplet zijn,
    standaard T = 300 K; andere beginmultiplets krijgen gelijke gewichten;
  * niet-ontaarde Starkniveaus zoals afzonderlijk opgesomd in de arrays;
  * lijnbreedtes (sigma = 150 cm^-1, Lorentz standaard 0) zijn niet
    gemeten/gefit;
  * bij ontbrekende relatieve lijnsterktes: gelijke gewichten als VORMAANNAME;
  * de effectieve ESA-fitwaarde wordt aan de dominante 3F4 -> 3F2 band
    toegewezen. De aldus berekende curve is een model, geen gemeten spectrum.

Onbekende absolute bandsterktes worden NaN, nooit automatisch 1 of 0.
Een energiematch is een kandidaat, geen bewijs van een toegestane/sterke lijn.
Fononzijbanden, andere multiplets, gestimuleerde emissie en populatiedynamiek
worden niet berekend. Daarom is dit geen volledig photon-avalanchemodel.

Gebruik: pas PUMP_NM aan; voer uit met Python/Spyder (numpy + matplotlib).
Met matplotlib wordt PNG naast dit script opgeslagen; plt.show is non-blocking.
Zonder matplotlib blijft de getalsmatige uitvoer beschikbaar.

Er worden ook twee multipletfiguren zoals het referentieplaatje opgeslagen:
  * tm_multiplets_3H6.png: 3H6 op nul, Boltzmann uitsluitend binnen 3H6;
  * tm_multiplets_3F4.png: 3F4 op nul, Boltzmann uitsluitend binnen 3F4.
Hogere multiplets krijgen in deze figuren gelijke gewichten. Elke curve wordt
op haar eigen maximum genormaliseerd; de stickhoogten tonen de gewichten.
Deze multipletfiguren zijn energieverdelingen, geen overgangsspectra.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np

PUMP_NM = 1064
TEMPERATURE_K = 300.0

# Boltzmann-bezetting uitsluitend binnen het onderste/beginmultiplet:
#   "multiplet" -> p_i ~ exp[-(E_i - E_min)/(k_B T)] voor 3H6 en 3F4;
#                  andere beginmultiplets krijgen gelijke gewichten.
#   "off"       -> alle beginniveaus gelijk gewicht
# Het eindmultiplet wordt nooit Boltzmanngewogen: bij 3H6 -> 3F4 dus alleen 3H6.
BOLTZMANN_MODE = "multiplet"
BOLTZMANN_MULTIPLETS = ("3H6", "3F4")

# Standaard lijnverbreding (cm^-1), per band te overschrijven.
LINE_WIDTH_CM1 = 8.5        # Gauss-sigma (inhomogeen)
LORENTZ_HWHM_CM1 = 0.0        # Lorentz-HWHM (homogeen); 0 = puur Gauss
INSTRUMENT_SIGMA_CM1 = 0.0    # extra Gauss-convolutie (resolutie), optioneel

# Aparte verbreding voor de gladde multipletfiguren zoals het referentiebeeld.
# Dit verandert LINE_WIDTH_CM1 van de overgangsspectra niet.
MULTIPLET_PLOT_SIGMA_CM1 = 150.0

# Numeriek raster voor de convolutie.
GRID_STEP_CM1 = 1.0           # rasterstap [cm^-1]
GRID_PAD_CM1 = 8000.0         # marge rond sticks/evaluatiepunten [cm^-1]

K_B_CM1_PER_K = 0.6950348005
H_J_S = 6.62607015e-34
C_M_S = 299792458.0

LEVELS = {
    "3H6": np.array([0., 9., 54., 80., 122., 149., 187., 223., 246., 296., 314., 405., 417.]),
    "3F4": np.array([5634., 5661., 5689., 5717., 5730., 5750., 5761., 5782., 5823.]),
    "3H5": np.array([8207., 8218., 8267., 8288., 8317., 8348., 8377., 8428., 8440., 8528., 8539.]),
    "3H4": np.array([12546., 12566., 12588., 12613., 12642., 12673., 12721., 12766., 12776.]),
    "3F3": np.array([14279., 14317., 14341., 14361., 14380., 14398., 14420.]),
    "3F2": np.array([14936., 14951., 15018., 15061., 15084.]),
}


@dataclass
class Band:
    lower: str
    upper: str
    width_cm1: float = LINE_WIDTH_CM1              # Gauss-sigma
    lorentz_hwhm_cm1: float = LORENTZ_HWHM_CM1     # Lorentz-HWHM
    instrument_sigma_cm1: float = INSTRUMENT_SIGMA_CM1
    # Een anker geldt alleen voor DIT multipletpaar en deze referentietemperatuur.
    reference_nm: float | None = None
    reference_cm2: float | None = None
    reference_temperature_k: float = 300.0
    # Relatieve lijnsterktes: matrix (aantal beginlevels, aantal eindlevels).
    # None = uitsluitend voor de lijnvorm gelijke gewichten aannemen.
    relative_lines: np.ndarray | None = None
    # Alternatief voor een anker: schaalfactor in cm voor de vormfunctie.
    # Gelijk aan integraal sigma(nu_tilde) dnu_tilde uitsluitend wanneer
    # band_shape integraal 1 heeft. Eenheid cm2 * cm^-1 = cm.
    strength_cm: float | None = None


# Alle 15 paren zijn energetische kandidaten, niet automatisch sterke lijnen.
BANDS = {(a, b): Band(a, b) for a, b in combinations(LEVELS, 2)}
BANDS[("3F4", "3F2")].reference_nm = 1064.0
BANDS[("3F4", "3F2")].reference_cm2 = 6.4e-21

# Voorbeeld wanneer je LATER een onafhankelijke 1450-nm-kruisdoorsnede hebt:
# BANDS[("3F4", "3H4")].reference_nm = 1450.0
# BANDS[("3F4", "3H4")].reference_cm2 = ...  # eigen waarde in cm2
# Een nieuw anker op dezelfde band vervangt het oude; fit bij meerdere punten
# gezamenlijk de sterkte, lijnbreedte en eventueel relatieve lijnsterktes.

GSA_1064_CM2 = 6.0e-25  # Los datapunt; GEEN universele GSA-spectrale functie.


# ---------------------------------------------------------------------------
# Bezetting
# ---------------------------------------------------------------------------

def boltzmann_within(levels_cm1, temperature_k):
    """Conditionele bezetting: gegeven dat het ion in dit multiplet zit."""
    if not np.isfinite(temperature_k) or temperature_k <= 0:
        raise ValueError("Temperatuur moet eindig en groter dan 0 K zijn.")
    levels = np.asarray(levels_cm1, dtype=float)
    p = np.exp(-(levels - levels.min()) / (K_B_CM1_PER_K * temperature_k))
    return p / p.sum()


def level_populations(multiplet_name, temperature_k):
    """Begingewichten: Boltzmann alleen voor 3H6/3F4; anders gelijk gewicht."""
    if BOLTZMANN_MODE not in ("multiplet", "off"):
        raise ValueError("BOLTZMANN_MODE moet 'multiplet' of 'off' zijn.")
    levels = np.asarray(LEVELS[multiplet_name], dtype=float)
    if BOLTZMANN_MODE == "multiplet" and multiplet_name in BOLTZMANN_MULTIPLETS:
        return boltzmann_within(levels, temperature_k)
    return np.full(len(levels), 1.0 / len(levels))


# ---------------------------------------------------------------------------
# Stick-spectrum en convolutie
# ---------------------------------------------------------------------------

def transition_gaps(band):
    return LEVELS[band.upper][None, :] - LEVELS[band.lower][:, None]


def _relative_lines(band, shape):
    if band.relative_lines is None:
        return np.full(shape, 1.0 / shape[1])
    relative = np.asarray(band.relative_lines, dtype=float)
    if relative.shape != shape or not np.all(np.isfinite(relative)):
        raise ValueError("Lijnsterktematrix heeft onjuiste vorm of waarden.")
    if np.any(relative < 0) or not np.any(relative > 0):
        raise ValueError("Lijnsterktes moeten niet-negatief zijn, met minstens een > 0.")
    return relative


def stick_spectrum(band, temperature_k=TEMPERATURE_K):
    """Sticks (golfgetal DeltaE_ij [cm^-1], gewicht w_ij = p_i * r_ij).

    Geeft twee 1D-arrays terug. De som van de gewichten is de integraal van
    de band: 1 bij relative_lines=None, anders sum(p_i * r_ij).
    """
    gaps = transition_gaps(band)
    p = level_populations(band.lower, temperature_k)
    weights = p[:, None] * _relative_lines(band, gaps.shape)
    return gaps.ravel(), weights.ravel()


def _check_widths(band):
    values = (band.width_cm1, band.lorentz_hwhm_cm1, band.instrument_sigma_cm1)
    if not all(np.isfinite(v) and v >= 0 for v in values):
        raise ValueError("Lijnbreedtes moeten eindig en niet-negatief zijn.")
    sigma_tot = float(np.hypot(band.width_cm1, band.instrument_sigma_cm1))
    if sigma_tot < GRID_STEP_CM1 and band.lorentz_hwhm_cm1 < GRID_STEP_CM1:
        raise ValueError(
            "Lijnbreedte is kleiner dan de rasterstap; verklein GRID_STEP_CM1."
        )
    return sigma_tot, float(band.lorentz_hwhm_cm1)


def convolve_sticks(sticks_cm1, weights, wavenumber_cm1, sigma_cm1, hwhm_cm1):
    """Convolueer sticks met Gauss(sigma) en Lorentz(HWHM); evalueer op nu.

    De convolutie gebeurt op een uniform raster via de convolutiestelling:
      Gauss (sigma):  FT = exp(-2 pi^2 sigma^2 f^2)
      Lorentz (HWHM): FT = exp(-2 pi HWHM |f|)
    Hierdoor is er geen afkapping van de Lorentz-staarten (afgezien van
    periodieke omslag aan de rastergrenzen; die is verwaarloosbaar klein
    door GRID_PAD_CM1). Sticks worden lineair over twee rasterpunten
    verdeeld. De integraal van het resultaat is sum(weights).
    """
    nu = np.asarray(wavenumber_cm1, dtype=float)
    lo = min(sticks_cm1.min(), nu.min()) - GRID_PAD_CM1
    hi = max(sticks_cm1.max(), nu.max()) + GRID_PAD_CM1
    n = int(np.ceil((hi - lo) / GRID_STEP_CM1)) + 2
    n = 1 << (n - 1).bit_length()  # macht van 2: snelle FFT
    grid = lo + GRID_STEP_CM1 * np.arange(n)

    binned = np.zeros(n)
    pos = (sticks_cm1 - lo) / GRID_STEP_CM1
    i0 = np.floor(pos).astype(int)
    frac = pos - i0
    np.add.at(binned, i0, weights * (1.0 - frac))
    np.add.at(binned, i0 + 1, weights * frac)

    f = np.fft.rfftfreq(n, d=GRID_STEP_CM1)
    kernel = np.exp(-2.0 * np.pi**2 * sigma_cm1**2 * f**2) \
        * np.exp(-2.0 * np.pi * hwhm_cm1 * f)
    density = np.fft.irfft(np.fft.rfft(binned) * kernel, n) / GRID_STEP_CM1
    density = np.maximum(density, 0.0)  # numerieke ruis rond nul
    return np.interp(nu, grid, density)


def band_shape(wavenumber_cm1, band, temperature_k=TEMPERATURE_K):
    """Gebogen bandvorm: sticks (Boltzmann x r_ij) geconvolueerd met de lijnvorm.

    Integraal over golfgetal = sum(p_i * r_ij): 1 bij relative_lines=None,
    ongeacht het aantal lijnen. Dat bepaalt slechts de VORM van de band.
    Bij een eigen r_ij-matrix worden de gewichten exact gebruikt;
    strength_cm is dan de bijbehorende schaalfactor.
    """
    nu = np.asarray(wavenumber_cm1, dtype=float)
    if not np.all(np.isfinite(nu)):
        raise ValueError("Golfgetallen moeten eindig zijn.")
    sigma_tot, hwhm = _check_widths(band)
    sticks, weights = stick_spectrum(band, temperature_k)
    return convolve_sticks(sticks, weights, nu, sigma_tot, hwhm)


# ---------------------------------------------------------------------------
# Kruisdoorsneden en pompsnelheid
# ---------------------------------------------------------------------------

def cross_section_nm(wavelength_nm, band, temperature_k=TEMPERATURE_K):
    """Kruisdoorsnede in cm2 per ion in het beginmultiplet; onbekend -> NaN."""
    lam = np.asarray(wavelength_nm, dtype=float)
    if not np.all(np.isfinite(lam)) or np.any(lam <= 0):
        raise ValueError("Golflengtes moeten eindig en positief zijn.")
    shape = band_shape(1e7 / lam, band, temperature_k)
    has_anchor = band.reference_nm is not None or band.reference_cm2 is not None
    if band.strength_cm is not None:
        if has_anchor:
            raise ValueError("Kies een bandsterkte OF een anker, niet beide.")
        if not np.isfinite(band.strength_cm) or band.strength_cm < 0:
            raise ValueError("Bandsterkte moet eindig en niet-negatief zijn.")
        return band.strength_cm * shape
    if not has_anchor:
        return np.full_like(shape, np.nan)
    if band.reference_nm is None or band.reference_cm2 is None:
        raise ValueError("Een anker vereist zowel golflengte als kruisdoorsnede.")
    if not np.isfinite(band.reference_nm) or band.reference_nm <= 0:
        raise ValueError("Ankergolflengte moet eindig en positief zijn.")
    if not np.isfinite(band.reference_cm2) or band.reference_cm2 <= 0:
        raise ValueError("Ankerkruisdoorsnede moet eindig en positief zijn.")
    ref_nu = 1e7 / band.reference_nm
    ref_shape = float(band_shape(ref_nu, band, band.reference_temperature_k))
    gaps = transition_gaps(band)
    sigma_tot, hwhm = _check_widths(band)
    margin = 4 * sigma_tot + 10 * hwhm
    peak_grid = np.linspace(gaps.min() - margin, gaps.max() + margin, 2001)
    peak = np.max(band_shape(peak_grid, band, band.reference_temperature_k))
    # Numerieke guardrail, GEEN fysische grens voor een overgang.
    if ref_shape < 1e-4 * peak:
        raise ValueError("Anker ligt te ver in een lijnstaart; gebruik een passend spectraal model.")
    return band.reference_cm2 * shape / ref_shape


def pump_rate_s1(wavelength_nm, irradiance_w_cm2, band, temperature_k=TEMPERATURE_K):
    """W_ab = sigma_ab * fotonenflux, per ion in a. Onbekend blijft NaN.

    Een rate equation krijgt +n_a*W_ab in dn_b/dt en -n_a*W_ab in dn_a/dt.
    Totale multipletpopulaties n_a volgen NIET uit level_populations().
    """
    if not np.isfinite(irradiance_w_cm2) or irradiance_w_cm2 < 0:
        raise ValueError("Intensiteit moet eindig en niet-negatief zijn.")
    sigma = cross_section_nm(wavelength_nm, band, temperature_k)
    photon_energy_j = H_J_S * C_M_S / (np.asarray(wavelength_nm) * 1e-9)
    return sigma * irradiance_w_cm2 / photon_energy_j


def nearest_candidates(wavelength_nm, count=5):
    """Rangschik naar kleinste energiemismatch, NIET naar absorptiesterkte."""
    if not np.isfinite(wavelength_nm) or wavelength_nm <= 0:
        raise ValueError("Golflengte moet eindig en positief zijn.")
    rows = []
    for band in BANDS.values():
        gaps = transition_gaps(band)
        detuning = float(np.min(np.abs(gaps - 1e7 / wavelength_nm)))
        sigma = float(cross_section_nm(wavelength_nm, band))
        rows.append((detuning, band.lower, band.upper,
                     1e7 / gaps.max(), 1e7 / gaps.min(), sigma))
    return sorted(rows)[:count]


# ---------------------------------------------------------------------------
# Multipletfiguren met 3H6 of 3F4 als energie-referentie
# ---------------------------------------------------------------------------

def multiplet_plot_data(reference_multiplet, temperature_k=TEMPERATURE_K,
                        sigma_cm1=MULTIPLET_PLOT_SIGMA_CM1):
    """Relatieve energieen, gewichten en piekgenormaliseerde Gaussische sommen.

    Alleen het referentiemultiplet krijgt Boltzmanngewichten (modus multiplet).
    Alle andere getoonde multiplets krijgen gelijke gewichten. Multiplets die
    onder de gekozen referentie liggen worden in deze figuur niet getoond.
    """
    if reference_multiplet not in ("3H6", "3F4"):
        raise ValueError("Kies 3H6 of 3F4 als referentiemultiplet.")
    if BOLTZMANN_MODE not in ("multiplet", "off"):
        raise ValueError("BOLTZMANN_MODE moet 'multiplet' of 'off' zijn.")
    if not np.isfinite(sigma_cm1) or sigma_cm1 <= 0:
        raise ValueError("Plotbreedte moet eindig en groter dan nul zijn.")
    reference_energy = float(LEVELS[reference_multiplet].min())
    names = [name for name, levels in LEVELS.items()
             if levels.min() >= reference_energy]
    highest_energy = max(float(LEVELS[name].max()) for name in names) - reference_energy
    energy_grid = np.linspace(-120.0, highest_energy + 3.0 * sigma_cm1, 32001)
    distributions = {}
    for name in names:
        energies = LEVELS[name] - reference_energy
        if name == reference_multiplet and BOLTZMANN_MODE == "multiplet":
            weights = boltzmann_within(LEVELS[name], temperature_k)
        else:
            weights = np.full(len(energies), 1.0 / len(energies))
        curve = np.exp(-0.5 * ((energy_grid[:, None] - energies[None, :])
                              / sigma_cm1)**2) @ weights
        distributions[name] = {
            "energies_cm1": energies,
            "weights": weights,
            "curve": curve / curve.max(),
        }
    return energy_grid, distributions


def plot_multiplets(reference_multiplet, temperature_k=TEMPERATURE_K,
                   sigma_cm1=MULTIPLET_PLOT_SIGMA_CM1, output_directory=None):
    """Maak en bewaar een multipletfiguur in de stijl van het referentiebeeld."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    energy_grid, distributions = multiplet_plot_data(
        reference_multiplet, temperature_k, sigma_cm1)
    colors = {
        "3H6": "#1f77b4", "3F4": "#ff7f0e", "3H5": "#2ca02c",
        "3H4": "#d62728", "3F3": "#9467bd", "3F2": "#8c564b",
    }
    labels = {
        "3H6": r"$^3H_6$", "3F4": r"$^3F_4$", "3H5": r"$^3H_5$",
        "3H4": r"$^3H_4$", "3F3": r"$^3F_3$", "3F2": r"$^3F_2$",
    }
    with plt.rc_context({"font.size": 13, "axes.titlesize": 17,
                         "axes.labelsize": 13, "legend.fontsize": 12,
                         "xtick.labelsize": 12, "ytick.labelsize": 12}):
        fig, ax = plt.subplots(figsize=(14.5, 7.5))
        fig.subplots_adjust(left=0.075, right=0.795, bottom=0.34, top=0.89)
        for name, data in distributions.items():
            ax.plot(energy_grid, data["curve"], color=colors[name], lw=1.8,
                    label=f"{labels[name]} ({len(data['weights'])} niveaus)")
            ax.vlines(data["energies_cm1"], 0.0, data["weights"],
                      color=colors[name], lw=1.0, alpha=0.9)
        ax.set_xlim(float(energy_grid[0]), float(energy_grid[-1]))
        ax.set_ylim(0.0, 1.08)
        ax.xaxis.set_major_locator(MultipleLocator(2000 if reference_multiplet == "3H6" else 1000))
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
        ax.grid(alpha=0.3)
        ax.set_ylabel("Genormaliseerde Gaussische som")
        ax.set_xlabel(f"Energie t.o.v. laagste {labels[reference_multiplet]}-niveau "
                      r"(cm$^{-1}$)", labelpad=10)
        title_weighting = f"Boltzmann alleen in {labels[reference_multiplet]}" if BOLTZMANN_MODE == "multiplet" else "Gelijke gewichten"
        ax.set_title(f"{title_weighting}, T = {temperature_k:g} K", pad=12)
        ax.legend(loc="center left", bbox_to_anchor=(1.015, 0.5), frameon=True)

        # Een tweede as met expliciete tickposities voorkomt de singulariteit
        # van lambda = 1e7 / E bij E = 0. Beide assen gebruiken dezelfde posities.
        wavelengths_nm = np.array([10000., 3000., 1800., 1200., 1000., 800., 700., 650.])
        if reference_multiplet == "3F4":
            wavelengths_nm = np.array([10000., 5000., 3000., 2000., 1500., 1200., 1100.])
        positions = 1e7 / wavelengths_nm
        visible = (positions > 0) & (positions >= energy_grid[0]) & (positions <= energy_grid[-1])
        wavelength_axis = ax.twiny()
        wavelength_axis.set_xlim(ax.get_xlim())
        wavelength_axis.xaxis.set_ticks_position("bottom")
        wavelength_axis.xaxis.set_label_position("bottom")
        wavelength_axis.spines["bottom"].set_position(("outward", 76))
        for side in ("top", "left", "right"):
            wavelength_axis.spines[side].set_visible(False)
        wavelength_axis.spines["bottom"].set_color("blue")
        wavelength_axis.tick_params(axis="x", colors="blue")
        wavelength_axis.set_xticks(positions[visible],
                                  [f"{value:g}" for value in wavelengths_nm[visible]])
        wavelength_axis.set_xlabel("Equivalente golflengte t.o.v. E = 0 (nm)",
                                   color="blue", labelpad=10)
        fig.text(0.075, 0.045,
                 f"Gaussbreedte: sigma = {sigma_cm1:g} cm-1. "
                 "Elke curve heeft een eigen maximum van 1; staafjes tonen de niveaugewichten.\n"
                 "Hogere multiplets: gelijke gewichten. Multipletverdelingen, geen overgangsspectra.",
                 fontsize=10, color="0.3", va="bottom")
        output_dir = Path(output_directory) if output_directory is not None else Path(__file__).resolve().parent
        output_path = output_dir / f"tm_multiplets_{reference_multiplet}.png"
        fig.savefig(output_path, dpi=180, facecolor="white")
        print(f"Multipletfiguur: {output_path}")
    return fig, output_path


def main():
    print(f"\nPomp: {PUMP_NM:g} nm; T={TEMPERATURE_K:g} K; "
          f"Boltzmann: {BOLTZMANN_MODE}")
    print("Boltzmannweging alleen bij beginmultiplet 3H6 of 3F4; "
          "eindmultiplet krijgt geen Boltzmannweging.")
    print(f"Lijnvorm: Gauss sigma={LINE_WIDTH_CM1:g} cm^-1, "
          f"Lorentz HWHM={LORENTZ_HWHM_CM1:g} cm^-1, "
          f"instrument sigma={INSTRUMENT_SIGMA_CM1:g} cm^-1")
    print("Energetische kandidaten binnen de zes opgegeven multiplets:")
    print("Overgang      kale Starkband (nm)   mismatch (cm^-1)   sigma (cm2)")
    for delta, a, b, lo, hi, sigma in nearest_candidates(PUMP_NM):
        value = "onbekend" if np.isnan(sigma) else f"{sigma:.3e} (model)"
        print(f"{a}->{b:3s}     {lo:7.1f}-{hi:7.1f}       {delta:7.1f}       {value}")
    print("\nEnergiematch zegt niets over de onbekende bandsterkte of beginpopulatie.")
    print("1064-nm GSA: los fononondersteund fitpunt, sigma=6.0e-25 cm2.")
    print("Dat GSA-punt wordt niet gebruikt om een elektronische band te schalen.")

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        print("Geen figuur gemaakt: matplotlib is niet beschikbaar in deze Python-omgeving.")
        return

    for reference_multiplet in ("3H6", "3F4"):
        plot_multiplets(reference_multiplet)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    selected = [band for band in BANDS.values() if band.lower in ("3H6", "3F4")]
    for row, band in enumerate(selected):
        gaps = transition_gaps(band)
        axes[0].plot([1e7 / gaps.max(), 1e7 / gaps.min()], [row, row], lw=7,
                     color="#237a9a" if band.lower == "3H6" else "#c46c28")
    axes[0].set_yticks(range(len(selected)), [f"{b.lower} -> {b.upper}" for b in selected])
    axes[0].invert_yaxis()
    axes[0].set_xscale("log")
    axes[0].set_xticks([650, 800, 1064, 1450, 2000, 3000, 4500],
                       ["650", "800", "1064", "1450", "2000", "3000", "4500"])
    # Verberg de automatische minor-labels van de log-as (bijv. 6x10^3, 4x10^3).
    axes[0].xaxis.set_minor_formatter(plt.NullFormatter())
    axes[0].tick_params(axis="x", labelrotation=45)
    axes[0].set_xlim(600, 4800)
    for wavelength in (1064, 1450):
        axes[0].axvline(wavelength, color="0.45", ls="--", lw=1)
    axes[0].set(title="Kandidaat-overgangen: energieposities",
                xlabel="Golflengte (nm); zonder lijnverbreding")
    axes[0].text(0.02, -0.28, "Lengte = spreiding van energieverschillen; geen sterkte.",
                 transform=axes[0].transAxes, fontsize=9)

    lam = np.linspace(980, 1200, 1401)
    band = BANDS[("3F4", "3F2")]
    axes[1].plot(lam, cross_section_nm(lam, band) / 1e-21, color="#c46c28",
                 label="Sticks (*) lijnvorm, geschaald aan 1064 nm")
    axes[1].scatter([1064], [6.4], color="black", zorder=3, label="Lee: ESA-fitpunt")
    axes[1].set(title="3F4 -> 3F2: voorwaardelijke schatting",
                xlabel="Golflengte (nm)", ylabel=r"Kruisdoorsnede ($10^{-21}$ cm$^2$)")
    axes[1].set_ylim(bottom=0)
    axes[1].legend(fontsize=9)

    # Stick-spectrum (Boltzmann x r_ij) op een tweede as, onder de curve.
    sticks, weights = stick_spectrum(band)
    sticks_nm = 1e7 / sticks
    in_range = (sticks_nm >= lam[0]) & (sticks_nm <= lam[-1])
    ax_sticks = axes[1].twinx()
    ax_sticks.vlines(sticks_nm[in_range], 0.0, weights[in_range] / weights.max(),
                     color="0.45", lw=1.0, alpha=0.7)
    ax_sticks.set_ylim(0.0, 4.0)
    ax_sticks.set_yticks([])
    axes[1].text(0.02, -0.28,
                 f"T = {TEMPERATURE_K:g} K; Gauss sigma = {band.width_cm1:g} cm-1, "
                 f"Lorentz HWHM = {band.lorentz_hwhm_cm1:g} cm-1 (aannames).\n"
                 "Grijze lijnen: sticks (Boltzmann x r_ij); andere golflengtes zijn geen meetdata.",
                 transform=axes[1].transAxes, fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.2)
    output_path = Path(__file__).resolve().with_name("tm_overgangsspectra.png")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"\nFiguur: {output_path}")
    plt.show(block=False)
    plt.pause(0.1)


if __name__ == "__main__":
    main()
