# -*- coding: utf-8 -*-
"""GSA/ESA voor Tm3+:NaYF4, met expliciete GSA-fononzijdenbanden.

Model: Boltzmanngewogen Stark-lijnen, een elektronische Gauss-kern en een
genormaliseerde, eindige-T Huang-Rhys-kernel met drie onafhankelijke modi.
Voor optische absorptie betekent een POSITIEVE energieverschuiving creatie
van roosterfononen; een NEGATIEVE verschuiving verbruikt roosterfononen.

Literatuur voor het mechanisme en de effectieve fononenergieen:
  https://doi.org/10.1016/j.omx.2021.100102   (NaYF4:Tm, 250/305/365 cm^-1)
Literatuur voor het vereenvoudigde vibronische model:
  https://doi.org/10.1080/0026897021000046889 (Liu et al., 2003)

BELANGRIJK: de koppelingen S_H5 = S_F4 = 0.05 en gelijke verdeling over de
modi zijn ongekalibreerde VOORBEELDWAARDEN. Ook de elektronische breedte
8.5 cm^-1 en de gelijke kanaal-/lijnsterktes zijn modelaannames, geen fits.
Dit is een discrete drie-modebenadering, geen volledige fonon-DOS of een
voorspelling van een gemeten NaYF4-spectrum. De oorspronkelijke 300 cm^-1
Gauss voor H5 is vervangen; die wordt niet bij de zijbanden opgeteld.
ESA blijft het oorspronkelijke elektronische model zonder zijbanden.
Normalisatie op 1064 nm bepaalt geen absolute GSA/ESA-verhouding.

Gebruik:
  python gsa_esa_fononzijbanden.py --no-show
  python gsa_esa_fononzijbanden.py --data-only
  python gsa_esa_fononzijbanden.py --temperature 350 --coupling-h5 0.08
  python gsa_esa_fononzijbanden.py --coupling-f4 0 --data-only

Benodigd: numpy; voor PNG/SVG-grafieken ook matplotlib.
CSV, instellingen/diagnostiek (JSON) en grafieken komen naast dit script,
of in de map opgegeven met --output-dir. Er wordt geen vloer toegevoegd.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import json
from pathlib import Path
import numpy as np

# Aanpasbare spectrale instellingen.
WAVELENGTH_MIN_NM = 900.0
WAVELENGTH_MAX_NM = 2000.0
WAVELENGTH_STEP_NM = 0.05
REFERENCE_WAVELENGTH_NM = 1064.0
TEMPERATURE_K = 300.0
STARK_SIGMA_CM1 = 8.5
GSA_H5_SIGMA_CM1 = 8.5  # Elektronische kern; oude effectieve 300 vervalt.
INSTRUMENT_SIGMA_CM1 = 0.0
PLOT_MIN_FACTOR = 1e-20  # Alleen de y-as, nooit een datavloer.
K_B_CM1_PER_K = 0.6950348005
BOLTZMANN_MODE = "multiplet"
SI_REFERENCE_ANCHOR = True

# Effectieve NaYF4-modi. Posities zijn literatuurwaarden; gewichten zijn dat niet.
PHONON_ENERGIES_CM1 = (250.0, 305.0, 365.0)
PHONON_MODE_FRACTIONS = (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)
GSA_H5_HUANG_RHYS = 0.05  # Te fitten totale S voor 3H6 -> 3H5.
GSA_F4_HUANG_RHYS = 0.05  # Afzonderlijk te fitten voor 3H6 -> 3F4.
PHONON_SHIFT_GRID_STEP_CM1 = 5.0  # Modi moeten exact op dit rooster liggen.
PHONON_TAIL_TOLERANCE = 1e-35  # Bovengrens ontbrekende totale kernelmassa.
PHONON_MAX_EVENTS_PER_MODE = 256  # Veiligheidslimiet; overschrijding is een fout.
REFERENCE_MAX_RELATIVE_KERNEL_ERROR = 1e-8  # Weiger onvoldoende opgelost anker.

# Stark-niveaus en kanaalgewichten uit het aangeleverde script.
E_3H6_CM1 = np.array([0., 9., 54., 80., 122., 149., 187., 223., 246.,
                       296., 314., 405., 417.])
E_3F4_CM1 = np.array([5634., 5661., 5689., 5717., 5730., 5750., 5761.,
                       5782., 5823.])
E_3H5_CM1 = np.array([8207., 8218., 8267., 8288., 8317., 8348., 8377.,
                       8428., 8440., 8528., 8539.])
E_3H4_CM1 = np.array([12546., 12566., 12588., 12613., 12642., 12673.,
                       12721., 12766., 12776.])
E_3F3_CM1 = np.array([14279., 14317., 14341., 14361., 14380., 14398., 14420.])
E_3F2_CM1 = np.array([14936., 14951., 15018., 15061., 15084.])
GSA_WEIGHT_3F4 = GSA_WEIGHT_3H5 = 1.0
ESA_WEIGHT_3H4 = ESA_WEIGHT_3F3 = ESA_WEIGHT_3F2 = 1.0
GSA_CHANNELS = {
    "3H6 -> 3F4": (E_3H6_CM1, E_3F4_CM1, GSA_WEIGHT_3F4),
    "3H6 -> 3H5": (E_3H6_CM1, E_3H5_CM1, GSA_WEIGHT_3H5),
}
ESA_CHANNELS = {
    "3F4 -> 3H4": (E_3F4_CM1, E_3H4_CM1, ESA_WEIGHT_3H4),
    "3F4 -> 3F3": (E_3F4_CM1, E_3F3_CM1, ESA_WEIGHT_3F3),
    "3F4 -> 3F2": (E_3F4_CM1, E_3F2_CM1, ESA_WEIGHT_3F2),
}


def boltzmann_within(levels_cm1, temperature_k=TEMPERATURE_K):
    """Conditional populations WITHIN one multiplet, normalized to one."""
    levels = np.asarray(levels_cm1, dtype=float)
    if levels.ndim != 1 or not levels.size or not np.all(np.isfinite(levels)):
        raise ValueError("levels_cm1 must be a finite, nonempty 1D vector.")
    if not np.isfinite(temperature_k) or temperature_k <= 0:
        raise ValueError("Temperature must be finite and positive.")
    weights = np.exp(-(levels - levels.min()) / (K_B_CM1_PER_K * temperature_k))
    return weights / weights.sum()


def level_populations(levels_cm1, temperature_k=TEMPERATURE_K):
    if BOLTZMANN_MODE == "multiplet":
        return boltzmann_within(levels_cm1, temperature_k)
    if BOLTZMANN_MODE == "off":
        levels = np.asarray(levels_cm1, dtype=float)
        if levels.ndim != 1 or not levels.size or not np.all(np.isfinite(levels)):
            raise ValueError("levels_cm1 must be a finite, nonempty 1D vector.")
        return np.full(levels.size, 1.0 / levels.size)
    raise ValueError("BOLTZMANN_MODE must be 'multiplet' or 'off'.")


def transition_gaps(lower_cm1, upper_cm1):
    gaps = np.asarray(upper_cm1)[None, :] - np.asarray(lower_cm1)[:, None]
    if not np.all(np.isfinite(gaps)) or np.any(gaps <= 0):
        raise ValueError("All electronic transition gaps must be positive and finite.")
    return gaps


def transition_sticks(lower_cm1, upper_cm1, temperature_k=TEMPERATURE_K,
                      relative_lines=None):
    gaps = transition_gaps(lower_cm1, upper_cm1)
    populations = level_populations(lower_cm1, temperature_k)
    if relative_lines is None:
        relative = np.full(gaps.shape, 1.0 / gaps.shape[1])
    else:
        relative = np.asarray(relative_lines, dtype=float)
        if (relative.shape != gaps.shape or not np.all(np.isfinite(relative))
                or np.any(relative < 0) or not np.any(relative > 0)):
            raise ValueError("Invalid relative line strengths.")
    return gaps, populations[:, None] * relative


def phonon_occupation(energy_cm1, temperature_k):
    """Bose occupation; T=0 is supported for the phonon kernel itself."""
    if (not np.isfinite(temperature_k) or temperature_k < 0
            or not np.isfinite(energy_cm1) or energy_cm1 <= 0):
        raise ValueError("Phonon energy must be > 0 and temperature >= 0.")
    if temperature_k == 0:
        return 0.0
    exponent = energy_cm1 / (K_B_CM1_PER_K * temperature_k)
    return 0.0 if exponent > 700 else float(1.0 / np.expm1(exponent))


def _poisson_tail_bound(mean, order, probability):
    """Upper bound on P(N > order), avoiding cancellation of 1-CDF."""
    if mean == 0:
        return 0.0
    ratio = mean / (order + 2)
    if ratio >= 1:
        return float("inf")
    return probability * mean / (order + 1) / (1.0 - ratio)


def _poisson_pair(mean_emit, mean_absorb, tolerance):
    """Use the SAME event cutoff for both signs, preserving detailed balance."""
    if max(mean_emit, mean_absorb) > 100:
        raise ValueError("Phonon event mean exceeds the supported range (100).")
    emit = [float(np.exp(-mean_emit))]
    absorb = [float(np.exp(-mean_absorb))]
    for order in range(PHONON_MAX_EVENTS_PER_MODE + 1):
        missing = (_poisson_tail_bound(mean_emit, order, emit[-1])
                   + _poisson_tail_bound(mean_absorb, order, absorb[-1]))
        if missing <= tolerance:
            return np.asarray(emit), np.asarray(absorb), missing
        emit.append(emit[-1] * mean_emit / (order + 1))
        absorb.append(absorb[-1] * mean_absorb / (order + 1))
    raise ValueError("Phonon expansion did not converge; revise cutoff or parameters.")


@lru_cache(maxsize=32)
def _cached_phonon_kernel(energies, strengths, temperature_k, step_cm1, tolerance):
    if (len(energies) != len(strengths) or not energies
            or not np.all(np.isfinite(energies)) or min(energies) <= 0
            or not np.all(np.isfinite(strengths)) or min(strengths) < 0
            or not np.isfinite(step_cm1) or step_cm1 <= 0
            or not np.isfinite(temperature_k) or temperature_k < 0
            or not np.isfinite(tolerance) or not 0 < tolerance < 1):
        raise ValueError("Invalid phonon kernel parameters.")
    lattice_positions = np.asarray(energies) / step_cm1
    if not np.allclose(lattice_positions, np.rint(lattice_positions), rtol=0, atol=1e-9):
        raise ValueError("Phonon energies must be exact multiples of the shift-grid step.")
    kernel = np.array([1.0])
    min_index = 0
    missing_bound = 0.0
    orders = []
    for energy, strength, stride in zip(energies, strengths, np.rint(lattice_positions).astype(int)):
        n = phonon_occupation(energy, temperature_k)
        emit, absorb, missing = _poisson_pair(strength * (n + 1), strength * n,
                                             tolerance / len(energies))
        # Emitted minus absorbed phonons: q=-N,...,+N (Skellam distribution).
        mode = np.convolve(emit, absorb[::-1])
        order = len(emit) - 1
        expanded = np.zeros(len(kernel) + 2 * order * stride)
        for position, probability in enumerate(mode):
            start = position * stride
            expanded[start:start + len(kernel)] += probability * kernel
        kernel = expanded
        min_index -= order * stride
        missing_bound += missing  # Union bound for all omitted event tails.
        orders.append(order)
    kernel /= kernel.sum()  # Conditional normalization; correction bounded above.
    shifts = (min_index + np.arange(len(kernel))) * step_cm1
    positive = kernel > 0  # Remove exact zeros only; no amplitude threshold/floor.
    shifts, kernel = shifts[positive], kernel[positive]
    shifts.setflags(write=False)
    kernel.setflags(write=False)
    return shifts, kernel, float(missing_bound), tuple(orders)


def phonon_kernel(total_coupling, temperature_k=TEMPERATURE_K,
                  tolerance=PHONON_TAIL_TOLERANCE):
    fractions = np.asarray(PHONON_MODE_FRACTIONS, dtype=float)
    if (fractions.shape != (len(PHONON_ENERGIES_CM1),)
            or not np.all(np.isfinite(fractions)) or np.any(fractions < 0)
            or not np.isclose(fractions.sum(), 1.0)
            or not np.isfinite(total_coupling) or total_coupling < 0):
        raise ValueError("Mode fractions must sum to one and coupling must be >= 0.")
    return _cached_phonon_kernel(tuple(PHONON_ENERGIES_CM1),
                                 tuple(total_coupling * fractions),
                                 float(temperature_k), float(PHONON_SHIFT_GRID_STEP_CM1),
                                 float(tolerance))


def broadened_transition_sticks(lower, upper, temperature_k, coupling, tolerance):
    gaps, line_weights = transition_sticks(lower, upper, temperature_k)
    shifts, probabilities, missing, orders = phonon_kernel(coupling, temperature_k, tolerance)
    positions = (gaps.ravel()[:, None] + shifts[None, :]).ravel()
    weights = (line_weights.ravel()[:, None] * probabilities[None, :]).ravel()
    # Exact merging of coincident lines; no binning of electronic energies.
    unique, inverse = np.unique(positions, return_inverse=True)
    combined = np.bincount(inverse, weights=weights, minlength=len(unique))
    positive = combined > 0
    return unique[positive], combined[positive], missing, orders


def gaussian_density(energy_cm1, sticks_cm1, weights, sigma_cm1):
    """Analytic unit-area Gaussian mixture; no FFT noise or numerical floor.

    A distance >40 sigma has exp(-800), which underflows to zero in float64.
    Restricting each sorted query chunk to that interval changes no nonzero
    float64 Gaussian contribution and keeps memory bounded.
    """
    energy = np.asarray(energy_cm1, dtype=float)
    if not np.all(np.isfinite(energy)) or not np.isfinite(sigma_cm1) or sigma_cm1 <= 0:
        raise ValueError("Energy must be finite and sigma must be positive.")
    flat = energy.ravel()
    order = np.argsort(flat)
    sorted_energy = flat[order]
    result = np.empty_like(flat)
    for start in range(0, len(flat), 256):
        values = sorted_energy[start:start + 256]
        left = np.searchsorted(sticks_cm1, values[0] - 40 * sigma_cm1, side="left")
        right = np.searchsorted(sticks_cm1, values[-1] + 40 * sigma_cm1, side="right")
        delta = (values[:, None] - sticks_cm1[None, left:right]) / sigma_cm1
        result[order[start:start + 256]] = np.sum(
            np.exp(-0.5 * delta**2) * weights[None, left:right], axis=1)
    result /= np.sqrt(2 * np.pi) * sigma_cm1
    return float(result[0]) if energy.ndim == 0 else result.reshape(energy.shape)


class StarkSpectrum:
    """Peak-normalized spectrum with per-channel phonon coupling and width."""
    def __init__(self, channels, temperature_k=TEMPERATURE_K,
                 channel_couplings=None, channel_sigmas_cm1=None,
                 tolerance=PHONON_TAIL_TOLERANCE):
        self.channels = channels
        self.temperature_k = temperature_k
        self.channel_couplings = dict(channel_couplings or {})
        self.channel_sigmas_cm1 = dict(channel_sigmas_cm1 or {})
        if (set(self.channel_couplings) - set(channels)
                or set(self.channel_sigmas_cm1) - set(channels)):
            raise ValueError("Unknown channel in width/coupling settings.")
        self.weight_sum = sum(weight for _, _, weight in channels.values())
        if (not np.isfinite(self.weight_sum) or self.weight_sum <= 0
                or any(not np.isfinite(w) or w < 0 for _, _, w in channels.values())
                or not np.isfinite(INSTRUMENT_SIGMA_CM1) or INSTRUMENT_SIGMA_CM1 < 0):
            raise ValueError("Invalid channel weights or instrument width.")
        self.prepared = {}
        self.diagnostics = {}
        for name, (lower, upper, weight) in channels.items():
            if weight == 0:
                continue
            coupling = self.channel_couplings.get(name, 0.0)
            base_sigma = self.channel_sigmas_cm1.get(name, STARK_SIGMA_CM1)
            if not np.isfinite(base_sigma) or base_sigma <= 0:
                raise ValueError("Electronic Gaussian sigma must be positive.")
            sigma = float(np.hypot(base_sigma, INSTRUMENT_SIGMA_CM1))
            sticks, amplitudes, missing, orders = broadened_transition_sticks(
                lower, upper, temperature_k, coupling, tolerance)
            self.prepared[name] = (sticks, amplitudes, sigma, weight / self.weight_sum)
            shifts, kernel, _, _ = phonon_kernel(coupling, temperature_k, tolerance)
            self.diagnostics[name] = {
                "total_huang_rhys_S": float(coupling),
                "electronic_sigma_cm1": float(base_sigma),
                "kernel_mass": float(kernel.sum()),
                "omitted_kernel_mass_upper_bound": missing,
                "max_events_per_mode_each_sign": list(orders),
                "no_phonon_event_weight": float(np.exp(-sum(
                    coupling * fraction * (2 * phonon_occupation(e, temperature_k) + 1)
                    for e, fraction in zip(PHONON_ENERGIES_CM1, PHONON_MODE_FRACTIONS)))),
                "net_zero_energy_shift_weight": float(kernel[shifts == 0].sum()),
                "integrated_stick_weight": float(amplitudes.sum()),
            }
        # Global peak normalization includes the full retained sideband support.
        lo = max(1.0, min(s[0] - 10 * sig for s, _, sig, _ in self.prepared.values()))
        hi = max(s[-1] + 10 * sig for s, _, sig, _ in self.prepared.values())
        min_sigma = min(sig for _, _, sig, _ in self.prepared.values())
        energy = np.linspace(lo, hi, max(20001, int((hi - lo) / min_sigma * 20) + 1))
        self.stark_norm = float(self.raw_stark(energy).max())
        self.norm = 1.0
        if not np.isfinite(self.stark_norm) or self.stark_norm <= 0:
            raise ValueError("Invalid spectrum normalization.")

    def raw_stark(self, energy):
        result = np.zeros_like(np.asarray(energy, dtype=float))
        for sticks, weights, sigma, channel_weight in self.prepared.values():
            result += channel_weight * gaussian_density(energy, sticks, weights, sigma)
        return result

    def mixed(self, wavelength_m):
        wavelength = np.asarray(wavelength_m, dtype=float)
        if np.any(~np.isfinite(wavelength)) or np.any(wavelength <= 0):
            raise ValueError("Wavelength must be finite and positive.")
        return self.raw_stark(1e-2 / wavelength) / self.stark_norm

    def factor(self, wavelength_m):
        result = self.mixed(wavelength_m)
        return float(result) if np.ndim(result) == 0 else result

    def channel_components(self, name, energy):
        """No-phonon-event component and remaining vibronic contributions.

        The remainder includes phonon processes with zero NET energy shift;
        it must not be equated to simply everything outside the central line.
        """
        lower, upper, _ = self.channels[name]
        sticks, weights, sigma, channel_weight = self.prepared[name]
        gaps, original_weights = transition_sticks(lower, upper, self.temperature_k)
        no_event_weight = self.diagnostics[name]["no_phonon_event_weight"]
        no_event = channel_weight * no_event_weight * gaussian_density(
            energy, *merge_sticks(gaps.ravel(), original_weights.ravel()), sigma)
        total = channel_weight * gaussian_density(energy, sticks, weights, sigma)
        return no_event, np.maximum(total - no_event, 0.0)


def merge_sticks(sticks, weights):
    unique, inverse = np.unique(sticks, return_inverse=True)
    return unique, np.bincount(inverse, weights=weights, minlength=len(unique))


def _build_spectra(temperature_k=TEMPERATURE_K, coupling_h5=GSA_H5_HUANG_RHYS,
                   coupling_f4=GSA_F4_HUANG_RHYS, tolerance=PHONON_TAIL_TOLERANCE):
    gsa = StarkSpectrum(GSA_CHANNELS, temperature_k,
                        {"3H6 -> 3H5": coupling_h5, "3H6 -> 3F4": coupling_f4},
                        {"3H6 -> 3H5": GSA_H5_SIGMA_CM1}, tolerance)
    esa = StarkSpectrum(ESA_CHANNELS, temperature_k, tolerance=tolerance)
    g_ref = gsa.factor(REFERENCE_WAVELENGTH_NM * 1e-9) if SI_REFERENCE_ANCHOR else 1.0
    e_ref = esa.factor(REFERENCE_WAVELENGTH_NM * 1e-9) if SI_REFERENCE_ANCHOR else 1.0
    if min(g_ref, e_ref) <= 0 or not np.all(np.isfinite([g_ref, e_ref])):
        raise ValueError("Reference absorption is zero/non-finite; revise coupling or reference. "
                         "A purely narrow electronic H5 band cannot be anchored at 1064 nm.")
    # A tiny integrated omission can dominate an extremely weak reference.
    # Bound the density error using the maximum of a unit-area Gaussian.
    if SI_REFERENCE_ANCHOR:
        raw_reference = float(gsa.raw_stark(1e7 / REFERENCE_WAVELENGTH_NM))
        error_bound = 0.0
        for name, (_, _, sigma, channel_weight) in gsa.prepared.items():
            missing = gsa.diagnostics[name]["omitted_kernel_mass_upper_bound"]
            error_bound += channel_weight * 2 * missing / (np.sqrt(2 * np.pi) * sigma)
        relative_bound = error_bound / raw_reference
        if relative_bound > REFERENCE_MAX_RELATIVE_KERNEL_ERROR:
            raise ValueError("GSA reference is not resolved accurately by this phonon cutoff. "
                             "Reduce --kernel-tolerance or revise the reference/model. "
                             f"Relative error bound: {relative_bound:.3g}.")
        gsa.reference_relative_kernel_error_bound = float(relative_bound)
    return gsa, esa, g_ref, e_ref


GSA_SPECTRUM = ESA_SPECTRUM = None
_GSA_REFERENCE_FACTOR = _ESA_REFERENCE_FACTOR = None


def configure_spectra(temperature_k=TEMPERATURE_K, coupling_h5=GSA_H5_HUANG_RHYS,
                       coupling_f4=GSA_F4_HUANG_RHYS, tolerance=PHONON_TAIL_TOLERANCE):
    global GSA_SPECTRUM, ESA_SPECTRUM, _GSA_REFERENCE_FACTOR, _ESA_REFERENCE_FACTOR
    (GSA_SPECTRUM, ESA_SPECTRUM,
     _GSA_REFERENCE_FACTOR, _ESA_REFERENCE_FACTOR) = _build_spectra(
        temperature_k, coupling_h5, coupling_f4, tolerance)


def _stark_resonance_factors(pump_wavelength_nm, p=None):
    if GSA_SPECTRUM is None:
        configure_spectra()
    wavelength_m = np.asarray(pump_wavelength_nm, dtype=float) * 1e-9
    gsa = GSA_SPECTRUM.factor(wavelength_m)
    esa = ESA_SPECTRUM.factor(wavelength_m)
    if SI_REFERENCE_ANCHOR:
        gsa /= _GSA_REFERENCE_FACTOR
        esa /= _ESA_REFERENCE_FACTOR
    return gsa, esa


def compute_factors(wavelengths_nm):
    wavelengths = np.asarray(wavelengths_nm, dtype=float)
    if (wavelengths.ndim != 1 or not wavelengths.size
            or np.any(~np.isfinite(wavelengths)) or np.any(wavelengths <= 0)):
        raise ValueError("Provide a nonempty 1D array of positive finite wavelengths.")
    return tuple(np.asarray(x) for x in _stark_resonance_factors(wavelengths))


def plot_factors(wavelengths, f_gsa, f_esa, basename, show=True):
    import matplotlib.pyplot as plt
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(11, 6.5))
        fig.subplots_adjust(left=.11, right=.97, bottom=.19, top=.80)
        fig.suptitle("GSA met fononzijdenbanden | Tm³⁺:NaYF₄", x=.11, y=.96,
                     ha="left", fontsize=19, fontweight="bold")
        subtitle = ("Relatieve doorsneden, elk genormaliseerd op 1064 nm"
                    if SI_REFERENCE_ANCHOR else "Relatieve doorsneden, elk genormaliseerd op eigen piek")
        fig.text(.11, .885, subtitle,
                 color="#526171")
        ax.plot(wavelengths, np.where(f_gsa > 0, f_gsa, np.nan), color="#ba4a18",
                lw=2, label=r"$f_{\mathrm{GSA}}$ met fononen")
        ax.plot(wavelengths, np.where(f_esa > 0, f_esa, np.nan), color="#1468b1",
                lw=1.8, label=r"$f_{\mathrm{ESA}}$")
        ax.set_yscale("log")
        ax.set_xlim(wavelengths[0], wavelengths[-1])
        ax.set_xticks(np.arange(900, 2001, 100))
        positive = np.concatenate((f_gsa[f_gsa > 0], f_esa[f_esa > 0]))
        ax.set_ylim(PLOT_MIN_FACTOR, 10 ** np.ceil(np.log10(positive.max())))
        ax.set_xlabel("Pompgolflengte (nm)")
        ax.set_ylabel("Relatieve absorptiedoorsnede f")
        ax.grid(which="major", color="#dce3eb", lw=.7)
        ax.axvline(REFERENCE_WAVELENGTH_NM, color="#687684", ls="--", lw=1)
        ax.axhline(1., color="#9ba6b1", ls=":", lw=1)
        if SI_REFERENCE_ANCHOR:
            ax.plot([REFERENCE_WAVELENGTH_NM], [1.], "o", color="#465360", ms=4)
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2,
                  frameon=False, fontsize=10, borderaxespad=0)
        h5 = GSA_SPECTRUM.channel_couplings["3H6 -> 3H5"]
        f4 = GSA_SPECTRUM.channel_couplings["3H6 -> 3F4"]
        fig.text(.11, .075, f"Fononen: {PHONON_ENERGIES_CM1} cm⁻¹ | "
                 f"S(H5)={h5:g}, S(F4)={f4:g} | T={GSA_SPECTRUM.temperature_k:g} K",
                 fontsize=9, color="#526171")
        fig.text(.11, .035, "Koppelingen en kernbreedtes zijn ongekalibreerde modelaannames.",
                 fontsize=9, color="#526171")
        fig.savefig(str(basename) + ".png", dpi=200, facecolor="white")
        fig.savefig(str(basename) + ".svg", facecolor="white")
        
        # Display all figures together.
        # block=False prevents Spyder from remaining busy until the plot windows
        # are manually closed.
        plt.show(block=False)
        plt.pause(0.1)


def main():
    parser = argparse.ArgumentParser(description="GSA met NaYF4-fononzijdenbanden")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE_K)
    parser.add_argument("--coupling-h5", type=float, default=GSA_H5_HUANG_RHYS)
    parser.add_argument("--coupling-f4", type=float, default=GSA_F4_HUANG_RHYS)
    parser.add_argument("--kernel-tolerance", type=float, default=PHONON_TAIL_TOLERANCE)
    parser.add_argument("--data-only", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    configure_spectra(args.temperature, args.coupling_h5, args.coupling_f4,
                      args.kernel_tolerance)
    if WAVELENGTH_MAX_NM <= WAVELENGTH_MIN_NM or WAVELENGTH_STEP_NM <= 0:
        raise ValueError("Invalid wavelength range/step.")
    count = int(np.ceil((WAVELENGTH_MAX_NM - WAVELENGTH_MIN_NM) / WAVELENGTH_STEP_NM)) + 1
    wavelengths = np.linspace(WAVELENGTH_MIN_NM, WAVELENGTH_MAX_NM, count)
    f_gsa, f_esa = compute_factors(wavelengths)
    if np.any(~np.isfinite(f_gsa)) or np.any(~np.isfinite(f_esa)):
        raise RuntimeError("Non-finite spectrum factors.")
    reference = _stark_resonance_factors(REFERENCE_WAVELENGTH_NM)
    if SI_REFERENCE_ANCHOR:
        np.testing.assert_allclose(reference, (1., 1.), rtol=1e-12, atol=0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    basename = args.output_dir / "f_gsa_f_esa_fononzijbanden"
    np.savetxt(str(basename) + ".csv", np.column_stack((wavelengths, f_gsa, f_esa)),
               delimiter=",", header="wavelength_nm,f_gsa,f_esa", comments="", fmt="%.12g")
    metadata = {
        "model": "Independent harmonic modes, finite-T Huang-Rhys (Poisson/Skellam)",
        "temperature_K": args.temperature,
        "phonon_energies_cm1": list(PHONON_ENERGIES_CM1),
        "mode_fractions_assumed": list(PHONON_MODE_FRACTIONS),
        "reference_wavelength_nm": REFERENCE_WAVELENGTH_NM,
        "reference_anchor_enabled": SI_REFERENCE_ANCHOR,
        "reference_peak_normalized_gsa": float(GSA_SPECTRUM.factor(REFERENCE_WAVELENGTH_NM * 1e-9)),
        "reference_peak_normalized_esa": float(ESA_SPECTRUM.factor(REFERENCE_WAVELENGTH_NM * 1e-9)),
        "reference_relative_kernel_error_upper_bound": getattr(
            GSA_SPECTRUM, "reference_relative_kernel_error_bound", None),
        "GSA": GSA_SPECTRUM.diagnostics,
        "ESA": ESA_SPECTRUM.diagnostics,
        "assumptions": "Couplings, mode fractions, electronic widths and line/channel strengths "
                       "are uncalibrated; no phonon DOS or absolute cross-section fit.",
        "sources": ["https://doi.org/10.1016/j.omx.2021.100102",
                    "https://doi.org/10.1080/0026897021000046889"],
    }
    Path(str(basename) + "_instellingen.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"f_GSA(1064 nm)={reference[0]:.12g}; f_ESA(1064 nm)={reference[1]:.12g}")
    for name, info in GSA_SPECTRUM.diagnostics.items():
        print(f"{name}: S={info['total_huang_rhys_S']:g}, "
              f"omitted kernel mass <= {info['omitted_kernel_mass_upper_bound']:.3g}")
    print(f"Data saved: {basename}.csv")
    if not args.data_only:
        plot_factors(wavelengths, f_gsa, f_esa, basename, show=not args.no_show)
        print(f"Plot saved: {basename}.png / .svg")


if __name__ == "__main__":
    main()
