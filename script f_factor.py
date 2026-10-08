# -*- coding: utf-8 -*-
"""Zelfstandig script voor de berekende f_GSA/f_ESA-grafiek van 900-2000 nm.

De spectrale functies en Stark-energieniveaus zijn dezelfde als in het
aangepaste photon-avalanche-script. Dit bestand heeft geen pa_model nodig.

Instellingen van de getoonde grafiek:
  - 3H6 -> 3H5: Gaussian sigma = 300 cm^-1; geen aparte GSA-achtergrond.
  - 3H6 -> 3F4 en alle ESA-kanalen: Gaussian sigma = 8.5 cm^-1.
  - Zowel de GSA- als de ESA-achtergrond is volledig verwijderd.
  - Boltzmannweging van de onderste Stark-niveaus bij 300 K.
  - f_i(lambda) = S_i(lambda)/S_i(1064 nm), dus beide f(1064 nm) = 1.
  - Dit zijn factoren voor de absorptiedoorsnede, zonder fotonfluxfactor.

De gekozen lijnbreedtes en bandgewichten zijn modelkeuzes.
De Stark-energieniveaus komen uit de oorspronkelijke aangeleverde code.
Alle zero-Lorentz-profielen worden als exacte Gaussian stickconvolutie
berekend. Er wordt geen numerieke vloer bij de spectra opgeteld.

Gebruik:
    python gsa_esa_900_2000nm.py
    python gsa_esa_900_2000nm.py --data-only
    python gsa_esa_900_2000nm.py --no-show --output-dir "mijn map"

Benodigd: numpy; voor de grafiek ook matplotlib.
Uitvoer: CSV plus PNG/SVG naast dit script, tenzij --output-dir is opgegeven.
"""

from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import argparse
import numpy as np

# Aanpasbare instellingen.
WAVELENGTH_MIN_NM = 900.0
WAVELENGTH_MAX_NM = 2000.0
WAVELENGTH_STEP_NM = 0.05
REFERENCE_WAVELENGTH_NM = 1064.0
TEMPERATURE_K = 300.0
STARK_SIGMA_CM1 = 8.5
GSA_H5_SIGMA_CM1 = 300
GSA_BACKGROUND_WEIGHT = 0.0
ESA_BACKGROUND_WEIGHT = 0.0
ESA_BACKGROUND_LAMBDA0_NM = 1060.0
ESA_BACKGROUND_SIGMA_NM = 150.0
PLOT_MIN_FACTOR = 1e-20  # Alleen y-aslimiet; verandert de berekende data niet.

# Overige instellingen, gelijk aan de versie waarmee de grafiek is berekend.
ABSORPTION_MODEL = "broad_gsa"
SI_REFERENCE_ANCHOR = True
BOLTZMANN_MODE = "multiplet"
K_B_CM1_PER_K = 0.6950348005
LORENTZ_HWHM_CM1 = 0.0
INSTRUMENT_SIGMA_CM1 = 0.0
CONV_GRID_STEP_CM1 = 0.5
CONV_GRID_PAD_CM1 = 8000.0
GSA_WEIGHT_3F4 = GSA_WEIGHT_3H5 = 1.0
ESA_WEIGHT_3H4 = ESA_WEIGHT_3F3 = ESA_WEIGHT_3F2 = 1.0

# Behouden constructorinstellingen; beide achtergronden hebben gewicht nul.
GSA_BACKGROUND_LAMBDA0_NM = 1e7 / (1e7 / REFERENCE_WAVELENGTH_NM - 1200.0)
GSA_BACKGROUND_SIGMA_NM = 120.0

# Stark-energieniveaus [cm^-1], ongewijzigd uit het oorspronkelijke script.
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

@lru_cache(maxsize=64)
def _cached_boltzmann_within(levels_tuple, temperature_k):
    """Immutable probabilities cached by energies and temperature."""
    levels = np.asarray(levels_tuple, dtype=float)
    p = np.exp(-(levels - levels.min()) / (K_B_CM1_PER_K * temperature_k))
    return tuple(p / p.sum())

def boltzmann_within(levels_cm1, temperature_k=TEMPERATURE_K):
    """Conditional occupation of the sublevels of ONE multiplet at T.

    p_i proportional to exp(-(E_i - E_min)/(kB*T)), normalized to sum 1.
    This is a population WITHIN the multiplet (given the ion is already
    there), not an absolute population of that multiplet. Each multiplet
    uses its OWN minimum, so the absolute energy offset of 3F4 cancels.
    Repeated calls at fixed energies and T reuse the cached probabilities;
    each call returns a fresh array so callers cannot modify the cache.
    """
    if not np.isfinite(temperature_k) or temperature_k <= 0:
        raise ValueError("TEMPERATURE_K must be finite and > 0 K.")
    levels = np.asarray(levels_cm1, dtype=float)
    if levels.ndim != 1 or levels.size == 0:
        raise ValueError("levels_cm1 must be a nonempty 1D array.")
    if not np.all(np.isfinite(levels)):
        raise ValueError("levels_cm1 must contain finite energies.")
    return np.asarray(_cached_boltzmann_within(tuple(levels), float(temperature_k)))

def level_populations(levels_cm1, temperature_k=TEMPERATURE_K):
    """Lower-sublevel weights according to BOLTZMANN_MODE (sum = 1)."""
    levels = np.asarray(levels_cm1, dtype=float)
    if BOLTZMANN_MODE == "multiplet":
        return boltzmann_within(levels, temperature_k)
    if BOLTZMANN_MODE == "off":
        if levels.ndim != 1 or levels.size == 0:
            raise ValueError("levels_cm1 must be a nonempty 1D array.")
        if not np.all(np.isfinite(levels)):
            raise ValueError("levels_cm1 must contain finite energies.")
        return np.full(levels.size, 1.0 / levels.size)
    raise ValueError("BOLTZMANN_MODE must be 'multiplet' or 'off'.")

def _broadening():
    """Validated total Gauss sigma and Lorentz HWHM in cm^-1."""
    values = (STARK_SIGMA_CM1, LORENTZ_HWHM_CM1, INSTRUMENT_SIGMA_CM1)
    if not all(np.isfinite(v) and v >= 0 for v in values):
        raise ValueError("Line widths must be finite and nonnegative.")
    sigma_tot = float(np.hypot(STARK_SIGMA_CM1, INSTRUMENT_SIGMA_CM1))
    hwhm = float(LORENTZ_HWHM_CM1)
    if sigma_tot < CONV_GRID_STEP_CM1 and hwhm < CONV_GRID_STEP_CM1:
        raise ValueError("Line width is smaller than CONV_GRID_STEP_CM1; "
                         "decrease the grid step.")
    return sigma_tot, hwhm

def convolve_sticks(sticks_cm1, weights, wavenumber_cm1, sigma_cm1, hwhm_cm1):
    """Convolve sticks with Gauss(sigma) and Lorentz(HWHM); evaluate at nu.

    Done on a uniform grid via the convolution theorem:
      Gauss (sigma):  FT = exp(-2 pi^2 sigma^2 f^2)
      Lorentz (HWHM): FT = exp(-2 pi HWHM |f|)
    Lorentz tails are not truncated (apart from periodic wrap-around at the
    grid edges, negligible thanks to CONV_GRID_PAD_CM1). Each stick is split
    linearly over two grid points. The integral of the result is
    sum(weights). Evaluation points outside the grid (more than
    CONV_GRID_PAD_CM1 away from any stick) return 0.
    """
    sticks = np.asarray(sticks_cm1, dtype=float)
    weights = np.asarray(weights, dtype=float)
    nu = np.asarray(wavenumber_cm1, dtype=float)
    lo = sticks.min() - CONV_GRID_PAD_CM1
    hi = sticks.max() + CONV_GRID_PAD_CM1
    n = int(np.ceil((hi - lo) / CONV_GRID_STEP_CM1)) + 2
    n = 1 << (n - 1).bit_length()  # power of 2: fast FFT
    grid = lo + CONV_GRID_STEP_CM1 * np.arange(n)

    binned = np.zeros(n)
    pos = (sticks - lo) / CONV_GRID_STEP_CM1
    i0 = np.floor(pos).astype(int)
    frac = pos - i0
    np.add.at(binned, i0, weights * (1.0 - frac))
    np.add.at(binned, i0 + 1, weights * frac)

    f = np.fft.rfftfreq(n, d=CONV_GRID_STEP_CM1)
    kernel = (np.exp(-2.0 * np.pi**2 * sigma_cm1**2 * f**2)
              * np.exp(-2.0 * np.pi * hwhm_cm1 * f))
    density = np.fft.irfft(np.fft.rfft(binned) * kernel, n) / CONV_GRID_STEP_CM1
    density = np.maximum(density, 0.0)  # numerical noise around zero
    return np.interp(nu, grid, density, left=0.0, right=0.0)

def _stark_gaussian_unnormalized(energy_cm1, levels_cm1, sigma_cm1, amplitudes=None):
    """Gaussian sum in cm^-1; chunk arrays to bound memory usage."""
    energy = np.asarray(energy_cm1, dtype=float)
    levels = np.asarray(levels_cm1, dtype=float)
    if not np.isfinite(sigma_cm1) or sigma_cm1 <= 0:
        raise ValueError("sigma_cm1 must be finite and > 0.")
    amps = np.ones_like(levels) if amplitudes is None else np.asarray(amplitudes)
    if levels.ndim != 1 or not len(levels) or amps.shape != levels.shape:
        raise ValueError("Levels and amplitudes must be matching nonempty vectors.")
    flat = energy.ravel()
    result = np.empty_like(flat)
    for start in range(0, flat.size, 2048):
        delta = flat[start:start + 2048, None] - levels
        result[start:start + 2048] = np.exp(-0.5 * (delta / sigma_cm1)**2) @ amps
    return float(result[0]) if energy.ndim == 0 else result.reshape(energy.shape)

def transition_gaps(lower_cm1, upper_cm1):
    """(n_lower, n_upper) matrix of upper-lower sublevel gaps, in cm^-1."""
    lower = np.asarray(lower_cm1, dtype=float)
    upper = np.asarray(upper_cm1, dtype=float)
    gaps = upper[None, :] - lower[:, None]
    if np.any(gaps <= 0):
        raise ValueError("Upper levels must lie above all lower levels.")
    return gaps

def transition_sticks(lower_cm1, upper_cm1, temperature_k=TEMPERATURE_K,
                      relative_lines=None):
    """Return (gap_ij, weight_ij) with weight_ij = p_lower_i * r_ij.

    The upper multiplet receives NO Boltzmann factor. Default r_ij=1/n_upper
    gives unit total weight per channel and row sums equal to p_lower_i.
    Custom relative strengths are retained as supplied, without rescaling.
    """
    gaps = transition_gaps(lower_cm1, upper_cm1)
    p = level_populations(lower_cm1, temperature_k)
    if relative_lines is None:
        relative = np.full(gaps.shape, 1.0 / gaps.shape[1])
    else:
        relative = np.asarray(relative_lines, dtype=float)
        if relative.shape != gaps.shape or not np.all(np.isfinite(relative)):
            raise ValueError("relative_lines has the wrong shape or contains non-finite values.")
        if np.any(relative < 0) or not np.any(relative > 0):
            raise ValueError("relative_lines must be nonnegative, with at least one nonzero entry.")
    return gaps, p[:, None] * relative

GSA_CHANNELS = {
    "3H6 -> 3F4": (E_3H6_CM1, E_3F4_CM1, GSA_WEIGHT_3F4),
    "3H6 -> 3H5": (E_3H6_CM1, E_3H5_CM1, GSA_WEIGHT_3H5),
}

ESA_CHANNELS = {
    "3F4 -> 3H4": (E_3F4_CM1, E_3H4_CM1, ESA_WEIGHT_3H4),
    "3F4 -> 3F3": (E_3F4_CM1, E_3F3_CM1, ESA_WEIGHT_3F3),
    "3F4 -> 3F2": (E_3F4_CM1, E_3F2_CM1, ESA_WEIGHT_3F2),
}

class StarkSpectrum:
    """Globally peak-normalized mixture, independent of the scan window.

    Each channel's Stark structure is the stick spectrum of all (lower,
    upper) sublevel pairs, with weight p_i * r_ij (p_i = Boltzmann
    population of lower sublevel i at TEMPERATURE_K, conditional on being in
    that multiplet; r_ij = relative line strength, uniform 1/n_upper by
    default), convolved explicitly with Gauss(sigma) and Lorentz(HWHM)
    (see convolve_sticks and channel_shape).
    """

    def __init__(self, channels, background_weight, center_nm, width_nm,
                 channel_sigmas_cm1=None, analytic_gaussian=False):
        self.channels = channels
        self.analytic_gaussian = bool(analytic_gaussian)
        self.channel_sigmas_cm1 = dict(channel_sigmas_cm1 or {})
        if set(self.channel_sigmas_cm1) - set(channels):
            raise ValueError("Unknown channel in channel_sigmas_cm1.")
        for width in self.channel_sigmas_cm1.values():
            if not np.isfinite(width) or width <= 0:
                raise ValueError("Channel Gaussian sigmas must be positive and finite.")
        weights = np.array([w for _, _, w in channels.values()])
        if not np.all(np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
            raise ValueError("Channel weights must be finite, nonnegative and nonzero in total.")
        if not np.isfinite(background_weight) or not 0 <= background_weight <= 1:
            raise ValueError("Background weight must be between 0 and 1.")
        if background_weight != 0:
            raise ValueError("Dit zelfstandige script bevat geen achtergrond; gebruik gewicht 0.")
        if not np.isfinite(center_nm) or not np.isfinite(width_nm) or min(center_nm, width_nm) <= 0:
            raise ValueError("Background center and width must be finite and positive.")
        if not np.isfinite(STARK_SIGMA_CM1) or STARK_SIGMA_CM1 <= 0:
            raise ValueError("STARK_SIGMA_CM1 must be finite and positive.")
        sigma_tot, hwhm = _broadening()
        self.weight_sum = weights.sum()
        self.background_weight = background_weight
        self.center_nm, self.width_nm = center_nm, width_nm
        transitions = np.concatenate([
            transition_gaps(lower, upper).ravel()
            for lower, upper, w in channels.values() if w > 0
        ])
        channel_widths = [np.hypot(self.channel_sigmas_cm1.get(name, STARK_SIGMA_CM1),
                                    INSTRUMENT_SIGMA_CM1)
                          for name, (_, _, weight) in channels.items() if weight > 0]
        minimum_sigma = min(channel_widths)
        maximum_sigma = max(channel_widths)
        margin = 10 * maximum_sigma + 20 * hwhm
        lo = max(1.0, transitions.min() - margin)
        hi = transitions.max() + margin
        # At least 20 samples per Gauss sigma; resolve even a width of 8.5 cm^-1.
        energy = np.linspace(lo, hi, max(20001, int((hi-lo)/minimum_sigma*20)+1))
        self.stark_norm = float(np.max(self.raw_stark(energy)))
        wavelength_nm = np.unique(np.concatenate([
            1e7 / energy,
            np.linspace(max(1.0, center_nm - 10*width_nm), center_nm + 10*width_nm, 20001)
        ]))
        self.norm = float(np.max(self.mixed(wavelength_nm * 1e-9)))
        if not np.isfinite(self.norm) or self.norm <= 0:
            raise ValueError("Invalid spectrum normalization.")

    def channel_shape(self, energy_cm1, lower_cm1, upper_cm1,
                      temperature_k=TEMPERATURE_K, relative_lines=None, sigma_cm1=None):
        """Sticks (p_i * r_ij at DeltaE_ij) convolved with Gauss and Lorentz.

        p_i = level_populations(lower_cm1, temperature_k): the weight of
        lower sublevel i (Boltzmann within the multiplet, or equal weights
        for BOLTZMANN_MODE="off"). r_ij defaults to 1/n_upper (equal weight
        among the reachable upper sublevels for each lower sublevel) when
        relative_lines is None. The convolution kernel has unit integral
        over wavenumber, so with DEFAULT relative strengths the channel's
        total stick weight is sum_ij p_i r_ij = 1 regardless of temperature,
        sublevel count or line widths. Custom strengths keep their supplied
        total weight. The upper multiplet is never Boltzmann-weighted.
        """
        energy = np.asarray(energy_cm1, dtype=float)
        if not np.all(np.isfinite(energy)):
            raise ValueError("energy_cm1 must be finite.")
        gaps, weights = transition_sticks(lower_cm1, upper_cm1,
                                         temperature_k, relative_lines)
        sigma_tot, hwhm = _broadening()
        if sigma_cm1 is not None:
            if not np.isfinite(sigma_cm1) or sigma_cm1 <= 0:
                raise ValueError("Channel sigma must be positive and finite.")
            sigma_tot = float(np.hypot(sigma_cm1, INSTRUMENT_SIGMA_CM1))
        if self.analytic_gaussian and hwhm == 0:
            # Exact Gaussian stick convolution; avoids FFT round-off in far tails.
            return (_stark_gaussian_unnormalized(
                energy, gaps.ravel(), sigma_tot, weights.ravel())
                / (np.sqrt(2.0*np.pi)*sigma_tot))
        shape = convolve_sticks(gaps.ravel(), weights.ravel(),
                                energy.ravel(), sigma_tot, hwhm)
        return shape.reshape(energy.shape)

    def raw_stark(self, energy):
        value = np.zeros_like(np.asarray(energy, dtype=float))
        for name, (lower, upper, weight) in self.channels.items():
            if weight:
                value = value + (weight/self.weight_sum) * self.channel_shape(
                    energy, lower, upper, sigma_cm1=self.channel_sigmas_cm1.get(name))
        return value

    def mixed(self, wavelength_m):
        wavelength = np.asarray(wavelength_m, dtype=float)
        if np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0):
            raise ValueError("Wavelength must be finite and positive.")
        stark = self.raw_stark(1e-2/wavelength) / self.stark_norm
        # Geen achtergrondterm: uitsluitend de som van Stark-overgangen.
        return stark

    def factor(self, wavelength_m):
        value = self.mixed(wavelength_m) / self.norm
        return float(value) if np.ndim(value) == 0 else value

def _build_spectra():
    """Bouw GSA/ESA-spectra en referentiefactoren voor het actieve model."""
    channel_sigmas = ({"3H6 -> 3H5": GSA_H5_SIGMA_CM1}
                      if ABSORPTION_MODEL == "broad_gsa" else None)
    gsa = StarkSpectrum(GSA_CHANNELS, GSA_BACKGROUND_WEIGHT,
                        GSA_BACKGROUND_LAMBDA0_NM, GSA_BACKGROUND_SIGMA_NM,
                        channel_sigmas_cm1=channel_sigmas,
                        analytic_gaussian=(ABSORPTION_MODEL == "broad_gsa"))
    esa = StarkSpectrum(ESA_CHANNELS, ESA_BACKGROUND_WEIGHT,
                        ESA_BACKGROUND_LAMBDA0_NM, ESA_BACKGROUND_SIGMA_NM,
                        analytic_gaussian=True)
    if SI_REFERENCE_ANCHOR:
        g_ref = gsa.factor(REFERENCE_WAVELENGTH_NM * 1e-9)
        e_ref = esa.factor(REFERENCE_WAVELENGTH_NM * 1e-9)
        if min(g_ref, e_ref) <= 1e-12:
            raise ValueError("Reference absorption is too small to calibrate reliably. "
                             "Increase the GSA line width or revise the spectral model.")
    else:
        g_ref = e_ref = 1.0   # geen anker: factoren worden niet gebruikt
    return gsa, esa, g_ref, e_ref

def _stark_resonance_factors(pump_wavelength_nm, p):
    """Relative cross sections; may exceed one away from the reference."""
    wavelength_m = pump_wavelength_nm * 1e-9
    gsa = GSA_SPECTRUM.factor(wavelength_m)
    esa = ESA_SPECTRUM.factor(wavelength_m)
    if SI_REFERENCE_ANCHOR:
        gsa /= _GSA_REFERENCE_FACTOR
        esa /= _ESA_REFERENCE_FACTOR
    return gsa, esa


# Bouw spectra en referentiewaarden met exact dezelfde berekeningsfuncties.
GSA_SPECTRUM, ESA_SPECTRUM, _GSA_REFERENCE_FACTOR, _ESA_REFERENCE_FACTOR = _build_spectra()


def compute_factors(wavelengths_nm):
    """Return (f_gsa, f_esa) voor een 1D-array van golflengten in nm."""
    wavelengths = np.asarray(wavelengths_nm, dtype=float)
    if wavelengths.ndim != 1 or not len(wavelengths):
        raise ValueError("Geef een niet-lege 1D-array met golflengten.")
    if np.any(~np.isfinite(wavelengths)) or np.any(wavelengths <= 0):
        raise ValueError("Golflengten moeten positief en eindig zijn.")
    return tuple(np.asarray(value) for value in _stark_resonance_factors(wavelengths, None))


def plot_factors(wavelengths, f_gsa, f_esa, basename, show=True):
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(10.8, 6.4))
        fig.subplots_adjust(left=0.11, right=0.97, bottom=0.17, top=0.81)
        fig.suptitle("GSA en ESA zonder achtergrond", x=0.11, y=0.96,
                     ha="left", fontsize=20, fontweight="bold")
        fig.text(0.11, 0.885,
                 r"$f_i(\lambda)=\sigma_i(\lambda)/\sigma_i(1064\,\mathrm{nm})$ | beide f(1064 nm) = 1",
                 fontsize=11, color="#526171")
        ax.plot(wavelengths, np.where(f_gsa > 0, f_gsa, np.nan), color="#ba4a18",
                linewidth=2.1, label=r"$f_{\mathrm{gsa}}$")
        ax.plot(wavelengths, np.where(f_esa > 0, f_esa, np.nan), color="#1468b1",
                linewidth=2.1, label=r"$f_{\mathrm{esa}}$")
        ax.set_yscale("log")
        ax.set_xlim(wavelengths[0], wavelengths[-1])
        span = wavelengths[-1]-wavelengths[0]
        tick_step = 100 if span > 800 else (50 if span >= 300 else 5)
        ax.set_xticks(np.arange(wavelengths[0], wavelengths[-1]+1, tick_step))
        positive = np.concatenate((f_gsa[f_gsa > 0], f_esa[f_esa > 0]))
        ax.set_ylim(PLOT_MIN_FACTOR,
                    10.0**np.ceil(np.log10(positive.max())))
        ax.set_xlabel("Pompgolflengte (nm)", labelpad=10)
        ax.set_ylabel("Versterkingsfactor f (logaritmische schaal)", labelpad=10)
        ax.grid(axis="y", which="major", color="#dce3eb", linewidth=0.8)
        ax.grid(axis="x", which="major", color="#eef1f5", linewidth=0.6)
        ax.axvline(REFERENCE_WAVELENGTH_NM, color="#687684", linestyle="--", linewidth=1)
        ax.axhline(1.0, color="#9ba6b1", linestyle=":", linewidth=1)
        ax.plot([REFERENCE_WAVELENGTH_NM], [1], "o", color="#465360", ms=5, zorder=5)
        ax.annotate("1064 nm: beide f = 1", xy=(REFERENCE_WAVELENGTH_NM, 1),
                    xytext=(wavelengths[0]+0.03*span, 50), fontsize=10, color="#465360",
                    arrowprops={"arrowstyle": "-", "color": "#687684"})
        ax.legend(loc="upper right", frameon=False, fontsize=13)
        fig.text(0.11, 0.035,
                 f"GSA H5 sigma: {GSA_H5_SIGMA_CM1:g} cm$^{{-1}}$ | beide achtergronden: 0"
                 + f" | ESA sigma: {STARK_SIGMA_CM1:g} cm$^{{-1}}$ | T = {TEMPERATURE_K:g} K",
                 fontsize=10, color="#526171")
        fig.savefig(str(basename)+".png", dpi=200, facecolor="white")
        fig.savefig(str(basename)+".svg", facecolor="white")
        if show:
            plt.show()
        else:
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="GSA/ESA-factoren van 900-2000 nm")
    parser.add_argument("--data-only", action="store_true", help="Alleen CSV berekenen; geen matplotlib nodig.")
    parser.add_argument("--no-show", action="store_true", help="Grafiek opslaan zonder venster te openen.")
    parser.add_argument("--output-dir", default=None, help="Uitvoermap; standaard naast dit script.")
    args = parser.parse_args()
    if WAVELENGTH_MAX_NM <= WAVELENGTH_MIN_NM or WAVELENGTH_STEP_NM <= 0:
        raise ValueError("Ongeldig golflengtebereik of stap.")
    n = int(np.ceil((WAVELENGTH_MAX_NM-WAVELENGTH_MIN_NM)/WAVELENGTH_STEP_NM))+1
    wavelengths = np.linspace(WAVELENGTH_MIN_NM, WAVELENGTH_MAX_NM, n)
    f_gsa, f_esa = compute_factors(wavelengths)
    if np.any(~np.isfinite(f_gsa)) or np.any(~np.isfinite(f_esa)):
        raise RuntimeError("Niet-eindige factor gevonden.")
    reference = _stark_resonance_factors(REFERENCE_WAVELENGTH_NM, None)
    np.testing.assert_allclose(reference, (1.0, 1.0), rtol=1e-12, atol=0)
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)
    basename = output_dir / f"f_gsa_f_esa_{WAVELENGTH_MIN_NM:g}_{WAVELENGTH_MAX_NM:g}nm_zonder_achtergrond"
    np.savetxt(str(basename)+".csv", np.column_stack((wavelengths, f_gsa, f_esa)),
               delimiter=",", header="wavelength_nm,f_gsa,f_esa", comments="", fmt="%.12g")
    print(f"f_gsa(1064 nm) = {reference[0]:.12g}; f_esa(1064 nm) = {reference[1]:.12g}")
    print(f"CSV opgeslagen: {basename}.csv")
    if not args.data_only:
        plot_factors(wavelengths, f_gsa, f_esa, basename, show=not args.no_show)
        print(f"Grafiek opgeslagen: {basename}.png / .svg")


if __name__ == "__main__":
    main()
