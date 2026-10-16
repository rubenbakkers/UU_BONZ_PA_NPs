# -*- coding: utf-8 -*-
"""Standalone GSA/ESA spectra with up to 1, 2, 3 or 4 phonons per transition.

Sums/differences include repeated and different phonon bands.
Choose DEFAULT_MAX_PHONONS below or use --max-phonons 1, 2, 3 or 4.
The cutoff includes every order from zero to the chosen maximum.
At maximum 1 the normalized model reproduces the supplied original.
No normalization at 1064 nm.
Each complete GSA/ESA spectrum is normalized to its own global maximum.
Plots show these factors directly, without equalizing local peak heights.
The electronic Stark levels/weights follow the supplied electronic script.
All electronic widths are 8.5 cm^-1; the effective H5 width of 300 cm^-1
is replaced by explicit phonon sidebands, not applied a second time.

The four Gaussian components approximate the dotted Yb/Tm Raman curve in
Wilhelm et al., Fig. 3A: https://doi.org/10.7150/thno.5113 . They are a
phenomenological coupling proxy, not a measured phonon DOS or coupling.
See README.md for the full theory and limitations.

Usage (NumPy; Matplotlib only for plots):
    python gsa_esa_four_gaussian_multiphonon.py --max-phonons 4 --no-show
    python gsa_esa_four_gaussian_multiphonon.py --max-phonons 2 --data-only
    python gsa_esa_four_gaussian_multiphonon.py --max-phonons 1 --no-show
    python gsa_esa_four_gaussian_multiphonon.py --phonon-bands 1 3 --no-show
    python gsa_esa_four_gaussian_multiphonon.py --no-phonons --no-show
    python gsa_esa_four_gaussian_multiphonon.py --log-y --no-show
"""

from __future__ import annotations

import argparse
import json
import math
from itertools import combinations_with_replacement
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
DEFAULT_MAX_PHONONS = 4  # Choose 1, 2, 3 or 4; includes all lower orders.
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


def phonon_event_means(coupling, temperature_k, enabled_bands=None):
    """Independent creation/consumption means, with Bose factors at band centers."""
    if not np.isfinite(coupling) or coupling < 0:
        raise ValueError("Coupling must be finite and nonnegative.")
    if not np.isfinite(temperature_k) or temperature_k < 0:
        raise ValueError("Temperature must be finite and nonnegative.")
    weights = phonon_area_weights() * band_mask(enabled_bands)
    if temperature_k == 0:
        occupation = np.zeros(4)
    else:
        with np.errstate(over="ignore", divide="ignore"):
            x = PHONON_CENTERS_CM1 / (K_B_CM1_PER_K * temperature_k)
        exp_minus_x = np.exp(-x)
        occupation = exp_minus_x / (-np.expm1(-x))
    with np.errstate(over="ignore", invalid="ignore"):
        plus = coupling * weights * (occupation + 1.0)
        minus = coupling * weights * occupation
    if not np.all(np.isfinite(np.r_[plus, minus])):
        raise ValueError("Phonon event means overflow; reduce coupling or temperature.")
    return plus, minus


def multiphonon_kernel(coupling, temperature_k, enabled_bands=None,
                      max_phonons=DEFAULT_MAX_PHONONS):
    """All count combinations with total created + consumed <= max_phonons.

    Independent Poisson weights are conditioned on this total-event cutoff.
    For counts c_a the weight is product(mu_a**c_a/c_a!)/D_N, where
    D_N=sum(mu_total**q/q!, q=0..N). Each unordered count vector occurs once.
    Gaussian variances add even when signed energy shifts cancel.
    """
    if (isinstance(max_phonons, (bool, np.bool_))
            or not isinstance(max_phonons, (int, np.integer))
            or max_phonons not in (1, 2, 3, 4)):
        raise ValueError("Maximum phonons must be an integer: 1, 2, 3 or 4.")
    plus, minus = phonon_event_means(coupling, temperature_k, enabled_bands)
    means = np.r_[plus, minus]
    try:
        mu_total = float(math.fsum(means))
    except OverflowError as exc:
        raise ValueError("Total phonon event mean overflows; reduce coupling or temperature.") from exc
    if not np.isfinite(mu_total):
        raise ValueError("Total phonon event mean must be finite.")
    if mu_total == 0:
        log_terms = np.r_[0.0, np.full(max_phonons, -np.inf)]
    else:
        log_terms = np.array([q*math.log(mu_total)-math.lgamma(q+1)
                              for q in range(max_phonons+1)])
    log_denominator = float(np.logaddexp.reduce(log_terms))
    order_probabilities = np.exp(log_terms-log_denominator)
    # Positive-tail sum avoids subtracting nearly equal numbers for weak coupling.
    if 0 < mu_total < 1:
        q = max_phonons + 1
        term = math.exp(-mu_total + q*math.log(mu_total)-math.lgamma(q+1))
        omitted = term
        while term > omitted * np.finfo(float).eps:
            q += 1
            term *= mu_total / q
            omitted += term
    else:
        omitted = float(-np.expm1(min(0.0, log_denominator-mu_total)))
    active = np.flatnonzero(means > 0).tolist()
    signed_centers = np.r_[PHONON_CENTERS_CM1, -PHONON_CENTERS_CM1]
    variances = np.tile(PHONON_SIGMAS_CM1**2, 2)
    components = []
    for order in range(max_phonons+1):
        for events in combinations_with_replacement(active, order):
            counts = np.bincount(events, minlength=8)
            log_weight = -log_denominator
            for event in np.flatnonzero(counts):
                count = int(counts[event])
                log_weight += count*math.log(means[event])-math.lgamma(count+1)
            probability = math.exp(log_weight)
            if probability > 0:
                components.append({
                    "order": order,
                    "counts_create": counts[:4].tolist(),
                    "counts_consume": counts[4:].tolist(),
                    "shift_cm1": float(counts @ signed_centers),
                    "variance_cm2": float(counts @ variances),
                    "probability": probability,
                })
    diagnostics = {
        "zero_phonon": float(order_probabilities[0]),
        "create_one": (order_probabilities[1]*plus/mu_total).tolist() if mu_total else [0.0]*4,
        "consume_one": (order_probabilities[1]*minus/mu_total).tolist() if mu_total else [0.0]*4,
        "mean_created_events": plus.tolist(), "mean_consumed_events": minus.tolist(),
        "mean_untruncated_events": mu_total,
        "order_probabilities": order_probabilities.tolist(),
        "poisson_omitted_probability": float(omitted),
        "retained_probability_before_renormalization": float(math.exp(log_denominator-mu_total)),
        "combination_count": len(components),
    }
    return components, diagnostics


def transition_sticks(lower, upper, temperature_k):
    """Conditional lower-multiplet occupation; no upper Boltzmann factor."""
    if temperature_k == 0:
        p = (lower == lower.min()).astype(float)
    else:
        with np.errstate(over="ignore", divide="ignore"):
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


class MultiPhononSpectrum:
    """Electronic lines convolved with all retained phonon combinations.

    Components hold (shifted centers, weighted strengths, sigma, is_zero).
    The True flag identifies no phonon events, not all zero net shifts.
    """

    def __init__(self, channels, coupling=DEFAULT_COUPLING, temperature_k=TEMPERATURE_K,
                 enabled_bands=None, max_phonons=DEFAULT_MAX_PHONONS):
        self.enabled_bands = band_mask(enabled_bands)
        multiphonon_kernel(coupling, temperature_k, self.enabled_bands, max_phonons)
        self.max_phonons = max_phonons
        self.components = []
        self.channel_kernels = {}
        self.channel_probabilities = {}
        channel_weights = np.array([w for _, _, w in channels.values()], dtype=float)
        if (not np.all(np.isfinite(channel_weights)) or np.any(channel_weights < 0)
                or channel_weights.sum() <= 0):
            raise ValueError("Channel weights must be finite, nonnegative and nonzero in total.")
        for name, (lower, upper, weight) in channels.items():
            multiplier = CHANNEL_COUPLING_MULTIPLIERS[name]
            kernel, diagnostics = multiphonon_kernel(
                coupling * multiplier, temperature_k, self.enabled_bands, max_phonons)
            self.channel_probabilities[name] = diagnostics
            self.channel_kernels[name] = kernel
            if weight == 0:
                continue
            centers, strengths = transition_sticks(lower, upper, temperature_k)
            strengths = strengths * weight / channel_weights.sum()
            for term in kernel:
                sigma = math.sqrt(ELECTRONIC_SIGMA_CM1**2 + term["variance_cm2"])
                # In absorption, creation needs MORE photon energy; consumption LESS.
                self.components.append((centers + term["shift_cm1"],
                                        strengths * term["probability"], sigma,
                                        term["order"] == 0))
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
                # Beyond 40 sigmas exp(-delta**2/2) underflows to exactly zero
                # in float64. This skips only arithmetic zeros, not Gaussian tails.
                mask = (energy >= centers.min()-40*sigma) & (energy <= centers.max()+40*sigma)
                result[mask] += gaussian_sum(energy[mask], centers, strengths, sigma)
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
    np.savetxt(output_dir / "phonon_profile_four_gaussians_multiphonon.csv", data, delimiter=",",
               header="wavenumber_cm1,gaussian_1,gaussian_2,gaussian_3,gaussian_4,sum", comments="", fmt="%.12g")
    return grid, components, total


def plot_results(output_dir, wavelengths, gsa, esa, profile, normalization="global", log_y=False,
                 full_factors=None):
    """Plot the model factors directly, with one constant scale per spectrum."""
    if normalization != "global":
        raise ValueError("Only global normalization is supported; local peak equalization was removed.")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(10.5, 5.8))
        for column, (spectrum, color, label) in enumerate(
                ((gsa, "#ba4a18", "GSA"), (esa, "#1468b1", "ESA"))):
            full = spectrum.factor(wavelengths) if full_factors is None else full_factors[:, column]
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
            fig.savefig(output_dir / f"f_gsa_f_esa_four_gaussian_multiphonon.{suffix}", dpi=180)

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
            profile_fig.savefig(output_dir / f"phonon_profile_four_gaussians_multiphonon.{suffix}", dpi=180)
        return fig, profile_fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-phonons", type=int, choices=(1, 2, 3, 4), default=DEFAULT_MAX_PHONONS,
                        help="Maximum total phonons per transition, including all lower orders (default: %(default)s).")
    parser.add_argument("--coupling", type=float, default=DEFAULT_COUPLING, help="Dimensionless coupling strength; 0 disables sidebands.")
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
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory; default: results/max_N_phonons beside this script.")
    args = parser.parse_args()
    if (not np.all(np.isfinite([WAVELENGTH_MIN_NM, WAVELENGTH_MAX_NM, WAVELENGTH_STEP_NM]))
            or WAVELENGTH_MIN_NM <= 0 or WAVELENGTH_MAX_NM <= WAVELENGTH_MIN_NM or WAVELENGTH_STEP_NM <= 0):
        raise ValueError("Invalid wavelength range or step.")
    enabled = (np.zeros(4, dtype=bool) if args.no_phonons else
               np.isin(np.arange(1, 5), args.phonon_bands) if args.phonon_bands is not None else
               band_mask())
    if args.output_dir is None:
        args.output_dir = Path(__file__).resolve().parent / "results" / f"max_{args.max_phonons}_phonons"
    gsa = MultiPhononSpectrum(GSA_CHANNELS, args.coupling, args.temperature, enabled, args.max_phonons)
    esa = MultiPhononSpectrum(ESA_CHANNELS, args.coupling, args.temperature, enabled, args.max_phonons)
    n = int(np.ceil((WAVELENGTH_MAX_NM-WAVELENGTH_MIN_NM)/WAVELENGTH_STEP_NM)) + 1
    wavelengths = np.linspace(WAVELENGTH_MIN_NM, WAVELENGTH_MAX_NM, n)
    factors = np.column_stack((wavelengths, gsa.factor(wavelengths), esa.factor(wavelengths)))
    if np.any(~np.isfinite(factors)) or np.any(factors[:, 1:] < 0):
        raise RuntimeError("Invalid spectral factors.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output_dir / "f_gsa_f_esa_four_gaussian_multiphonon.csv", factors,
               delimiter=",", header="wavelength_nm,f_gsa,f_esa", comments="", fmt="%.12g")
    profile = export_profile(args.output_dir, enabled)
    settings = {
        "model": "conditional_truncated_independent_Poisson_Gaussian_bands",
        "maximum_phonons_per_absorption": args.max_phonons,
        "cutoff_counts": "total created plus consumed phonons; includes orders 0 through maximum",
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
        "channel_combinations": gsa.channel_kernels | esa.channel_kernels,
        "spectral_peaks_per_cm1": {"gsa": gsa.peak, "esa": esa.peak},
        "factors_at_1064nm": {"gsa": gsa.factor(1064), "esa": esa.factor(1064)},
        "profile_source": "Wilhelm et al. 2013 Fig.3A https://doi.org/10.7150/thno.5113",
        "assumptions": ["Four Gaussians approximate a manually traced Raman curve, not raw data.",
                        "Raman intensities are a coupling-shape proxy; no Raman Bose correction.",
                        "Bose occupation evaluated at each Gaussian center; finite-band approximation.",
                        "Independent phonon events, including repeated-band sums and signed differences.",
                        "All retained orders are normalized together; omitted Poisson mass is reported.",
                        "Gaussian phonon variances add for independently sampled events, even at zero net shift.",
                        "Full Gaussian tails retained; no artificial absorption floor."],
    }
    (args.output_dir / "four_gaussian_multiphonon_settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"f_GSA(1064 nm) = {gsa.factor(1064):.8g}; f_ESA(1064 nm) = {esa.factor(1064):.8g}")
    print(f"Results: {args.output_dir.resolve()}")
    print(f"Active phonon bands: {(np.flatnonzero(enabled) + 1).tolist()}")
    print(f"Maximum phonons per transition: {args.max_phonons} (all lower orders included)")
    for name, diagnostics in settings["channel_probabilities"].items():
        print(f"{name}: order probabilities = {diagnostics['order_probabilities']}; "
              f"omitted Poisson mass = {diagnostics['poisson_omitted_probability']:.4g}")
    if not args.data_only:
        figures = plot_results(args.output_dir, wavelengths, gsa, esa, profile,
                               args.plot_normalization, log_y=args.log_y,
                               full_factors=factors[:, 1:])
        import matplotlib.pyplot as plt
        if args.no_show:
            for figure in figures:
                plt.close(figure)
        else:
            plt.show()


if __name__ == "__main__":
    main()
