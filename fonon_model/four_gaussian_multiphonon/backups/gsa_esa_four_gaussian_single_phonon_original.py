# -*- coding: utf-8 -*-
"""Standalone GSA/ESA spectra with four Gaussian, single-phonon bands.

No sums/differences of multiple phonons and no normalization at 1064 nm.
Each complete GSA/ESA spectrum is normalized to its own global maximum.
Plots show these factors directly, without equalizing local peak heights.
The electronic Stark levels/weights follow the supplied electronic script.
All electronic widths are 8.5 cm^-1; the effective H5 width of 300 cm^-1
is replaced by explicit single-phonon sidebands, not applied a second time.

The four Gaussian components approximate the dotted Yb/Tm Raman curve in
Wilhelm et al., Fig. 3A: https://doi.org/10.7150/thno.5113 . They are a
phenomenological coupling proxy, not a measured phonon DOS or coupling.
See FOUR_GAUSSIAN_PHONONS.md for the full theory and limitations.

Usage (NumPy; Matplotlib only for plots):
    python gsa_esa_four_gaussian_phonons.py --no-show
    python gsa_esa_four_gaussian_phonons.py --coupling 0 --data-only
    python gsa_esa_four_gaussian_phonons.py --phonon-bands 1 3 --no-show
    python gsa_esa_four_gaussian_phonons.py --no-phonons --no-show
    python gsa_esa_four_gaussian_phonons.py --log-y --no-show
    python gsa_esa_four_gaussian_phonons.py --plot-normalization global --no-show
    python gsa_esa_four_gaussian_phonons.py --output-dir results/four_gaussians
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# Editable model assumptions. Sigma is a standard deviation, not FWHM.
PHONON_CENTERS_CM1 = np.array([253.0, 306.0, 364.0, 408.0])
PHONON_SIGMAS_CM1 = np.array([14.0, 24.0, 15.0, 24.0])
PHONON_PEAK_HEIGHTS = np.array([0.68, 0.95, 0.66, 0.36])
PHONON_BANDS_ENABLED = np.array([True, True, True, True])
TEMPERATURE_K = 300.0
ELECTRONIC_SIGMA_CM1 = 8.5
DEFAULT_COUPLING = 0.05
# Relative channel couplings: set any entry to 0 to disable its sidebands.
CHANNEL_COUPLING_MULTIPLIERS = {
    "3H6 -> 3F4": 1.0, "3H6 -> 3H5": 1.0,
    "3F4 -> 3H4": 1.0, "3F4 -> 3F3": 1.0, "3F4 -> 3F2": 1.0,
}
WAVELENGTH_MIN_NM, WAVELENGTH_MAX_NM, WAVELENGTH_STEP_NM = 900.0, 2000.0, 0.05
K_B_CM1_PER_K = 0.6950348005
PLOT_MIN_FACTOR = 1e-20  # Plot limit only; never added to the data.

# Stark levels [cm^-1], unchanged from the supplied script.
E_3H6_CM1 = np.array([0, 9, 54, 80, 122, 149, 187, 223, 246, 296, 314, 405, 417], dtype=float)
E_3F4_CM1 = np.array([5634, 5661, 5689, 5717, 5730, 5750, 5761, 5782, 5823], dtype=float)
E_3H5_CM1 = np.array([8207, 8218, 8267, 8288, 8317, 8348, 8377, 8428, 8440, 8528, 8539], dtype=float)
E_3H4_CM1 = np.array([12546, 12566, 12588, 12613, 12642, 12673, 12721, 12766, 12776], dtype=float)
E_3F3_CM1 = np.array([14279, 14317, 14341, 14361, 14380, 14398, 14420], dtype=float)
E_3F2_CM1 = np.array([14936, 14951, 15018, 15061, 15084], dtype=float)
GSA_CHANNELS = {
    "3H6 -> 3F4": (E_3H6_CM1, E_3F4_CM1, 1.0),
    "3H6 -> 3H5": (E_3H6_CM1, E_3H5_CM1, 1.0),
}
ESA_CHANNELS = {
    "3F4 -> 3H4": (E_3F4_CM1, E_3H4_CM1, 1.0),
    "3F4 -> 3F3": (E_3F4_CM1, E_3F3_CM1, 1.0),
    "3F4 -> 3F2": (E_3F4_CM1, E_3F2_CM1, 1.0),
}


def validate_phonons():
    arrays = (PHONON_CENTERS_CM1, PHONON_SIGMAS_CM1, PHONON_PEAK_HEIGHTS)
    if any(np.shape(a) != (4,) or not np.all(np.isfinite(a)) for a in arrays):
        raise ValueError("Exactly four finite centers, widths and heights are required.")
    if np.any(PHONON_CENTERS_CM1 <= 0) or np.any(PHONON_SIGMAS_CM1 <= 0):
        raise ValueError("Phonon centers and Gaussian sigmas must be positive.")
    if np.any(PHONON_PEAK_HEIGHTS < 0) or not np.any(PHONON_PEAK_HEIGHTS > 0):
        raise ValueError("Phonon heights must be nonnegative and nonzero in total.")
    if not np.isfinite(ELECTRONIC_SIGMA_CM1) or ELECTRONIC_SIGMA_CM1 <= 0:
        raise ValueError("Electronic Gaussian sigma must be positive and finite.")


def phonon_area_weights():
    """Unequal Gaussian widths require area weights: w_k ~ height_k*sigma_k."""
    validate_phonons()
    areas = PHONON_PEAK_HEIGHTS * PHONON_SIGMAS_CM1
    return areas / areas.sum()


def band_mask(enabled_bands=None):
    """Four on/off switches; disabling a band never reallocates its coupling."""
    enabled = PHONON_BANDS_ENABLED if enabled_bands is None else enabled_bands
    values = np.asarray(enabled)
    if values.shape != (4,) or not np.all(np.isin(values, [False, True])):
        raise ValueError("Supply four boolean phonon-band switches.")
    return values.astype(bool, copy=True)


def single_phonon_probabilities(coupling, temperature_k, enabled_bands=None):
    """Return P0, P(create one mode k), P(consume one mode k).

    This is a normalized first-order model, NOT a multiphonon Poisson
    expansion: P0=1/(1+mu), P+_k=S*w_k*(n_k+1)/(1+mu),
    P-_k=S*w_k*n_k/(1+mu). Occupations are evaluated at the band centers.
    """
    if not np.isfinite(coupling) or coupling < 0:
        raise ValueError("Coupling must be finite and nonnegative.")
    if not np.isfinite(temperature_k) or temperature_k < 0:
        raise ValueError("Temperature must be finite and nonnegative.")
    weights = phonon_area_weights() * band_mask(enabled_bands)
    if temperature_k == 0:
        occupation = np.zeros(4)
    else:
        x = PHONON_CENTERS_CM1 / (K_B_CM1_PER_K * temperature_k)
        exp_minus_x = np.exp(-x)
        occupation = exp_minus_x / (-np.expm1(-x))
    plus = coupling * weights * (occupation + 1.0)
    minus = coupling * weights * occupation
    denominator = 1.0 + plus.sum() + minus.sum()
    return float(1.0 / denominator), plus / denominator, minus / denominator


def transition_sticks(lower, upper, temperature_k):
    """Conditional lower-multiplet occupation; no upper Boltzmann factor."""
    if temperature_k == 0:
        p = (lower == lower.min()).astype(float)
    else:
        p = np.exp(-(lower - lower.min()) / (K_B_CM1_PER_K * temperature_k))
    p /= p.sum()
    gaps = upper[None, :] - lower[:, None]
    strengths = np.broadcast_to(p[:, None] / len(upper), gaps.shape)
    return gaps.ravel(), strengths.ravel()


def gaussian_sum(energy, centers, strengths, sigma):
    """Exact positive Gaussian sum, with unit area in wavenumber per stick."""
    values = np.asarray(energy, dtype=float)
    result = np.empty(values.size, dtype=float)
    flat = values.ravel()
    for start in range(0, flat.size, 512):
        delta = (flat[start:start + 512, None] - centers) / sigma
        result[start:start + 512] = np.sum(
            np.exp(-0.5 * delta**2) * strengths, axis=1) / (np.sqrt(2*np.pi) * sigma)
    return result.reshape(values.shape)


class SinglePhononSpectrum:
    """Electronic lines plus eight independent single-phonon sidebands.

    Components hold (shifted centers, weighted strengths, sigma, is_zero).
    There are no sidebands at sums/differences of phonon energies.
    """

    def __init__(self, channels, coupling=DEFAULT_COUPLING, temperature_k=TEMPERATURE_K,
                 enabled_bands=None):
        self.enabled_bands = band_mask(enabled_bands)
        single_phonon_probabilities(coupling, temperature_k, self.enabled_bands)
        self.components = []
        self.channel_probabilities = {}
        channel_weights = np.array([w for _, _, w in channels.values()], dtype=float)
        if (not np.all(np.isfinite(channel_weights)) or np.any(channel_weights < 0)
                or channel_weights.sum() <= 0):
            raise ValueError("Channel weights must be finite, nonnegative and nonzero in total.")
        for name, (lower, upper, weight) in channels.items():
            multiplier = CHANNEL_COUPLING_MULTIPLIERS[name]
            p0, plus, minus = single_phonon_probabilities(
                coupling * multiplier, temperature_k, self.enabled_bands)
            self.channel_probabilities[name] = {
                "zero_phonon": p0, "create_one": plus.tolist(), "consume_one": minus.tolist()}
            if weight == 0:
                continue
            centers, strengths = transition_sticks(lower, upper, temperature_k)
            strengths = strengths * weight / channel_weights.sum()
            self.components.append((centers, strengths * p0, ELECTRONIC_SIGMA_CM1, True))
            for mode, (omega, sigma_ph) in enumerate(zip(PHONON_CENTERS_CM1, PHONON_SIGMAS_CM1)):
                sigma = float(np.hypot(ELECTRONIC_SIGMA_CM1, sigma_ph))
                # In absorption, creation needs MORE photon energy; consumption LESS.
                for sign, probability in ((1, plus[mode]), (-1, minus[mode])):
                    if probability > 0:
                        self.components.append((centers + sign*omega, strengths * probability, sigma, False))
        # Fixed energy grid over all lines and sidebands, independent of output scan.
        lo = max(1.0, min(c.min() - 10*s for c, _, s, _ in self.components))
        hi = max(c.max() + 10*s for c, _, s, _ in self.components)
        grid = np.linspace(lo, hi, max(20001, int((hi-lo)/ELECTRONIC_SIGMA_CM1*20)+1))
        self.peak = float(self.raw(grid).max())
        if not np.isfinite(self.peak) or self.peak <= 0:
            raise ValueError("Invalid spectrum maximum.")

    def raw(self, energy_cm1, zero_only=False):
        energy = np.asarray(energy_cm1, dtype=float)
        if not np.all(np.isfinite(energy)):
            raise ValueError("Energies must be finite.")
        result = np.zeros_like(energy)
        for centers, strengths, sigma, is_zero in self.components:
            if not zero_only or is_zero:
                result += gaussian_sum(energy, centers, strengths, sigma)
        return result

    def factor(self, wavelengths_nm, zero_only=False):
        wavelengths = np.asarray(wavelengths_nm, dtype=float)
        if np.any(~np.isfinite(wavelengths)) or np.any(wavelengths <= 0):
            raise ValueError("Wavelengths must be finite and positive.")
        result = self.raw(1e7 / wavelengths, zero_only=zero_only) / self.peak
        return float(result) if result.ndim == 0 else result


def export_profile(output_dir, enabled_bands=None):
    grid = np.linspace(150, 500, 1401)
    components = PHONON_PEAK_HEIGHTS * np.exp(
        -0.5*((grid[:, None] - PHONON_CENTERS_CM1)/PHONON_SIGMAS_CM1)**2)
    # Keep all four component columns; the sum column contains active bands only.
    total = (components * band_mask(enabled_bands)).sum(axis=1)
    data = np.column_stack((grid, components, total))
    np.savetxt(output_dir / "phonon_profile_four_gaussians.csv", data, delimiter=",",
               header="wavenumber_cm1,gaussian_1,gaussian_2,gaussian_3,gaussian_4,sum", comments="", fmt="%.12g")
    return grid, components, total


def plot_results(output_dir, wavelengths, gsa, esa, profile, normalization="global", log_y=False):
    """Plot the model factors directly, with one constant scale per spectrum."""
    if normalization != "global":
        raise ValueError("Only global normalization is supported; local peak equalization was removed.")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(10.5, 5.8))
        for spectrum, color, label in ((gsa, "#ba4a18", "GSA"), (esa, "#1468b1", "ESA")):
            full = spectrum.factor(wavelengths)
            zero = spectrum.factor(wavelengths, zero_only=True)
            full_plot = np.where(full > 0, full, np.nan) if log_y else full
            zero_plot = np.where(zero > 0, zero, np.nan) if log_y else zero
            ax.plot(wavelengths, full_plot, color=color, label=label)
            ax.plot(wavelengths, zero_plot, color=color,
                    linestyle="--", alpha=0.65, label=f"{label}, zonder fononevent")
        ax.set(xlabel="Pompgolflengte (nm)", ylabel="Relatieve absorptiefactor f",
               xlim=(wavelengths[0], wavelengths[-1]),
               ylim=(PLOT_MIN_FACTOR if log_y else 0.0, 1.15),
               yscale="log" if log_y else "linear")
        ax.axvline(1064, color="#7d858c", linestyle=":", linewidth=1, label="1064 nm (alleen markering)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=9, ncol=2)
        fig.tight_layout()
        for suffix in ("png", "svg"):
            fig.savefig(output_dir / f"f_gsa_f_esa_four_gaussian_single_phonon.{suffix}", dpi=180)

        profile_fig, profile_ax = plt.subplots(figsize=(9, 4.8))
        grid, components, total = profile
        enabled = gsa.enabled_bands
        # Scale against the full reference profile, also when every band is off.
        scale = components.sum(axis=1).max()
        for k in range(4):
            style = {} if enabled[k] else {"color": "#bfc4c9", "alpha": 0.55}
            status = "" if enabled[k] else " (uit)"
            profile_ax.plot(grid, components[:, k]/scale, linestyle="--", **style,
                            label=f"Gaussian {k+1}: {PHONON_CENTERS_CM1[k]:g} cm⁻¹{status}")
        profile_ax.plot(grid, total/scale, color="#303844", linewidth=2, label="Som van actieve banden")
        profile_ax.set(xlabel="Ramanverschuiving (cm⁻¹)", ylabel="Relatieve profielintensiteit",
                       xlim=(200, 450), ylim=(0, 1.08))
        profile_ax.legend(fontsize=9)
        profile_ax.grid(alpha=0.2)
        profile_fig.tight_layout()
        for suffix in ("png", "svg"):
            profile_fig.savefig(output_dir / f"phonon_profile_four_gaussians.{suffix}", dpi=180)
        return fig, profile_fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coupling", type=float, default=DEFAULT_COUPLING, help="Dimensionless one-phonon strength; 0 disables sidebands.")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE_K, help="Temperature in K, including 0 K.")
    switches = parser.add_mutually_exclusive_group()
    switches.add_argument("--no-phonons", action="store_true", help="Disable all four phonon bands.")
    switches.add_argument("--phonon-bands", type=int, nargs="+", choices=(1, 2, 3, 4),
                          help="Enable only these bands, for example: --phonon-bands 1 3")
    parser.add_argument("--plot-normalization", choices=("global",), default="global",
                        help="Show the model factors directly (default); no local peak equalization.")
    parser.add_argument("--log-y", action="store_true", help="Use a logarithmic spectrum axis; default is linear.")
    parser.add_argument("--data-only", action="store_true", help="CSV/JSON only; no Matplotlib needed.")
    parser.add_argument("--no-show", action="store_true", help="Save plots without opening a window.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "results" / "four_gaussians")
    args = parser.parse_args()
    if (not np.all(np.isfinite([WAVELENGTH_MIN_NM, WAVELENGTH_MAX_NM, WAVELENGTH_STEP_NM]))
            or WAVELENGTH_MIN_NM <= 0 or WAVELENGTH_MAX_NM <= WAVELENGTH_MIN_NM or WAVELENGTH_STEP_NM <= 0):
        raise ValueError("Invalid wavelength range or step.")
    enabled = (np.zeros(4, dtype=bool) if args.no_phonons else
               np.isin(np.arange(1, 5), args.phonon_bands) if args.phonon_bands is not None else
               band_mask())
    gsa = SinglePhononSpectrum(GSA_CHANNELS, args.coupling, args.temperature, enabled)
    esa = SinglePhononSpectrum(ESA_CHANNELS, args.coupling, args.temperature, enabled)
    n = int(np.ceil((WAVELENGTH_MAX_NM-WAVELENGTH_MIN_NM)/WAVELENGTH_STEP_NM)) + 1
    wavelengths = np.linspace(WAVELENGTH_MIN_NM, WAVELENGTH_MAX_NM, n)
    factors = np.column_stack((wavelengths, gsa.factor(wavelengths), esa.factor(wavelengths)))
    if np.any(~np.isfinite(factors)) or np.any(factors[:, 1:] < 0):
        raise RuntimeError("Invalid spectral factors.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output_dir / "f_gsa_f_esa_four_gaussian_single_phonon.csv", factors,
               delimiter=",", header="wavelength_nm,f_gsa,f_esa", comments="", fmt="%.12g")
    profile = export_profile(args.output_dir, enabled)
    settings = {
        "model": "normalized_zero_or_one_phonon", "maximum_phonons_per_absorption": 1,
        "normalization": "separate_global_GSA_and_ESA_peaks; no 1064 nm anchor",
        "temperature_k": args.temperature, "coupling": args.coupling,
        "channel_coupling_multipliers": CHANNEL_COUPLING_MULTIPLIERS,
        "phonon_bands_enabled": enabled.tolist(),
        "active_phonon_bands": (np.flatnonzero(enabled) + 1).tolist(),
        "plot_normalization": args.plot_normalization,
        "plot_y_scale": "log" if args.log_y else "linear",
        "phonon_centers_cm1": PHONON_CENTERS_CM1.tolist(),
        "phonon_sigmas_cm1": PHONON_SIGMAS_CM1.tolist(),
        "phonon_peak_heights": PHONON_PEAK_HEIGHTS.tolist(),
        "phonon_area_weights": phonon_area_weights().tolist(),
        "electronic_sigma_cm1": ELECTRONIC_SIGMA_CM1,
        "channel_probabilities": gsa.channel_probabilities | esa.channel_probabilities,
        "spectral_peaks_per_cm1": {"gsa": gsa.peak, "esa": esa.peak},
        "factors_at_1064nm": {"gsa": gsa.factor(1064), "esa": esa.factor(1064)},
        "profile_source": "Wilhelm et al. 2013 Fig.3A https://doi.org/10.7150/thno.5113",
        "assumptions": ["Four Gaussians approximate a manually traced Raman curve, not raw data.",
                        "Raman intensities are a coupling-shape proxy; no Raman Bose correction.",
                        "Bose occupation evaluated at each Gaussian center; finite-band approximation.",
                        "No phonon combinations; 1064-nm GSA requiring several phonons is absent.",
                        "Full Gaussian tails retained; no artificial absorption floor."],
    }
    (args.output_dir / "four_gaussian_single_phonon_settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"f_GSA(1064 nm) = {gsa.factor(1064):.8g}; f_ESA(1064 nm) = {esa.factor(1064):.8g}")
    print(f"Results: {args.output_dir.resolve()}")
    print(f"Active phonon bands: {(np.flatnonzero(enabled) + 1).tolist()}")
    if not args.data_only:
        figures = plot_results(args.output_dir, wavelengths, gsa, esa, profile,
                               args.plot_normalization, log_y=args.log_y)
        import matplotlib.pyplot as plt
        if args.no_show:
            for figure in figures:
                plt.close(figure)
        else:
            plt.show()


if __name__ == "__main__":
    main()
