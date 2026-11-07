# -*- coding: utf-8 -*-
"""Efficient GSA/ESA spectra with up to N phonons per transition (any N >= 1).

Sums/differences include repeated and different phonon bands.
Choose DEFAULT_MAX_PHONONS below or use --max-phonons N (any integer >= 1; default below).
The cost grows quickly with N: see the estimate printed at the start of a run.
The cutoff includes every order from zero to the chosen maximum.
Each spectrum is normalized so that f(1064 nm) = 1 for both GSA and ESA
(see NORMALIZATION_WAVELENGTH_NM).
Plots show these factors directly, without equalizing local peak heights;
the spectrum is always saved twice, with a linear and with a logarithmic y-axis.
The electronic Stark levels start from the ab initio values of Table 4(B) in
J. Phys. Chem. C 129, 20446 (2025) ("Physical Chemistry.pdf").
The effective H5 width of 300 cm^-1 is replaced by explicit phonon sidebands,
not applied a second time.

ABSORPTION CALIBRATION (default; switch off with --original-model or
USE_ABSORPTION_CALIBRATION = False to get the previous model back exactly):
a room-temperature absorption spectrum of NaTmF4@NaYF4 in cyclohexane (Lambda 950)
was fitted with this model's own Stark lines and phonon kernel
(calibrate_from_absorption.py; numbers in ABSORPTION_CALIBRATION below):
  * electronic width sigma 8.5 -> 26.8 cm^-1 (3H6 -> 3H5 band; also used for ESA),
  * Huang-Rhys coupling 0.05 -> 0.12 (blue one-phonon wing of 3H5, 1115-1165 nm),
  * Stark manifolds stretched/shifted so that every GSA band matches the measured
    band (3F4 stretched about its lowest level, which stays at 5634 cm^-1);
    the ab initio levels carry errors of this size (the same paper finds an RMSE of
    64 cm^-1 against experiment for LiYF4:Tm, growing with energy, and notes that no
    experimental Stark levels exist for beta-NaYF4),
  * magnetic-dipole line strengths added to the Judd-Ofelt weights
    (3H6 -> 3H5, 3F4 -> 3H4, 3F4 -> 3F3).
The ESA is not measured: it follows from the calibrated levels and widths.
In the cyclohexane spectrum the C-H overtone 2nu(CH2) makes the cuvette opaque
at 1716-1772 nm, so that part of the 3F4 band is excluded from the fit.

CHANNEL WEIGHTS (new): the relative strength of each electronic channel is no
longer 1.0 but follows from Judd-Ofelt theory, with Omega_t (Table S4) and the
reduced matrix elements [U(t)]^2 (Table S5) of the Supplementary Information of
Lee et al., Nature 589, 230 (2021):
    S = sum_t Omega_t [U(t)]^2,   weight = S * nu_bar   (nu_bar: multiplet barycentre gap)
The SI gives no matrix elements for 3F4 -> 3F3 and 3F4 -> 3F2; these two are taken from
the original source, Carnall et al. 1978 (ANL-78-XX-95), Appendix XIII Table 2 (Tm3+:LaF3).
A PLACEHOLDER (line strength of 3F4 -> 3H4) is only used for a channel that has no entry.

The four Gaussian components approximate the dotted Yb/Tm Raman curve in
Wilhelm et al., Fig. 3A: https://doi.org/10.7150/thno.5113 . They are a
phenomenological coupling proxy, not a measured phonon DOS or coupling.
See README.md for the full theory and limitations.

Usage (NumPy; Matplotlib only for plots):
    python gsa_esa_four_gaussian_multiphonon.py --max-phonons 6 --no-show
    python gsa_esa_four_gaussian_multiphonon.py --max-phonons 10 --no-show
    python gsa_esa_four_gaussian_multiphonon.py --max-phonons 2 --data-only
    python gsa_esa_four_gaussian_multiphonon.py --max-phonons 1 --no-show
    python gsa_esa_four_gaussian_multiphonon.py --phonon-bands 1 3 --no-show
    python gsa_esa_four_gaussian_multiphonon.py --no-phonons --no-show
    python gsa_esa_four_gaussian_multiphonon.py --original-model --no-show   # uncalibrated model
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from functools import lru_cache
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np


# Editable model assumptions. Sigma is a standard deviation, not FWHM.
PHONON_CENTERS_CM1 = np.array([250.0, 306.0, 364.0, 408.0])
PHONON_SIGMAS_CM1 = np.array([14.0, 24.0, 15.0, 24.0])
PHONON_PEAK_HEIGHTS = np.array([0.68, 0.95, 0.66, 0.36])
PHONON_BANDS_ENABLED = np.array([True, True, True, True])
TEMPERATURE_K = 300.0
# Original-model width and coupling; the calibrated model uses ABSORPTION_CALIBRATION below.
ELECTRONIC_SIGMA_CM1 = 8.5
DEFAULT_COUPLING = 0.05
DEFAULT_MAX_PHONONS = 6  # Any integer >= 1; includes all lower orders.
COMBINATION_WARNING_THRESHOLD = 100_000  # Warn when the phonon-combination count exceeds this.
JSON_COMBINATION_LIMIT = 5000  # Per channel: write all combinations to the JSON only up to this count.
NORMALIZATION_WAVELENGTH_NM = 1064.0  # f_GSA = f_ESA = 1 at this wavelength.
GAUSSIAN_RELATIVE_TOLERANCE = 1e-14  # Numerical omission bound; 0 evaluates all float64 tails.
GAUSSIAN_QUERY_BLOCK_SIZE = 32
# Relative channel couplings: set any entry to 0 to disable its sidebands.
CHANNEL_COUPLING_MULTIPLIERS = {
    "3H6 -> 3F4": 1.0, "3H6 -> 3H5": 1.0,
    "3F4 -> 3H4": 1.0, "3F4 -> 3F3": 1.0, "3F4 -> 3F2": 1.0,
}
WAVELENGTH_MIN_NM, WAVELENGTH_MAX_NM, WAVELENGTH_STEP_NM = 900.0, 2000.0, 0.05
K_B_CM1_PER_K = 0.6950348005
PLOT_MIN_FACTOR = 1e-20  # Plot limit only; never added to the data.

# Stark levels [cm^-1]: ab initio values, Table 4(B) of J. Phys. Chem. C 129, 20446 (2025),
# unchanged from the supplied script. The calibrated levels are derived from these below.
STARK_LEVELS_TABLE4_CM1 = {
    "3H6": np.array([0, 9, 54, 80, 122, 149, 187, 223, 246, 296, 314, 405, 417], dtype=float),
    "3F4": np.array([5634, 5661, 5689, 5717, 5730, 5750, 5761, 5782, 5823], dtype=float),
    "3H5": np.array([8207, 8218, 8267, 8288, 8317, 8348, 8377, 8428, 8440, 8528, 8539], dtype=float),
    "3H4": np.array([12546, 12566, 12588, 12613, 12642, 12673, 12721, 12766, 12776], dtype=float),
    "3F3": np.array([14279, 14317, 14341, 14361, 14380, 14398, 14420], dtype=float),
    "3F2": np.array([14936, 14951, 15018, 15061, 15084], dtype=float),
}

# ---------------------------------------------------------------------------
# Absorption calibration (RT absorption of NaTmF4@NaYF4 in cyclohexane, 300-2000 nm)
# ---------------------------------------------------------------------------
# True: calibrated model (default). False, or --original-model on the command line:
# the previous model, reproduced exactly (sigma 8.5, coupling 0.05, Table 4 levels, ED only).
USE_ABSORPTION_CALIBRATION = True
ABSORPTION_CALIBRATION = {
    # Gaussian sigma of every electronic (zero-phonon) line, from the 3H6 -> 3H5 band
    # fit (1110-1300 nm). The GSA lines are about three times broader than 8.5 cm^-1.
    "electronic_sigma_gsa_cm1": 26.8,
    # Not measured: taken equal to the GSA width (ESA connects the same broadened levels).
    "electronic_sigma_esa_cm1": 26.8,
    # Huang-Rhys coupling of all channels, from the blue (one-phonon) wing of 3H5 at
    # 1115-1165 nm: coupling 0.05 gives only 0.42 of the measured wing. A wider 3H5
    # manifold cannot replace it (the band core wants a narrower one); with that freedom
    # the fit gives 0.15, so 0.12 is the lower end.
    "coupling": 0.12,
    # Stark corrections E_new = E_min + shift + stretch * (E - E_min), E_min = lowest
    # Table 4 level. Each from a fit of that multiplet's GSA band at the sigma and
    # coupling above. 3F4: lowest level kept (1.8-um emission level, start of the ESA).
    # 3F2 is the least certain (weak band on the 3F3 sideband): +66 +- 20 cm^-1, and
    # +26 / +88 cm^-1 for coupling 0.05 / 0.15.
    "stark_corrections": {
        "3F4": {"stretch": 2.34, "shift_cm1": 0.0},     # 5634-6076 cm^-1 (Table 4: 5634-5823)
        "3H5": {"stretch": 1.0, "shift_cm1": -21.3},    # 8186-8518 cm^-1
        "3H4": {"stretch": 1.60, "shift_cm1": -23.1},   # 12523-12891 cm^-1
        "3F3": {"stretch": 1.63, "shift_cm1": 97.5},    # 14377-14607 cm^-1
        "3F2": {"stretch": 1.0, "shift_cm1": 65.7},     # 15002-15150 cm^-1
    },
    # Add magnetic-dipole line strengths to the Judd-Ofelt (electric-dipole) weights.
    "magnetic_dipole": True,
    "refractive_index": 1.48,  # NaYF4; the MD/ED local-field ratio 9n^2/(n^2+2)^2 barely depends on n
    "source": "calibrate_from_absorption.py on NaTmF4@NaYF4 in cyclohexane (Lambda 950, 300-2000 nm, 2026)",
}


def corrected_levels(name, calibrated):
    """Table 4 Stark levels, optionally with the absorption calibration applied."""
    levels = STARK_LEVELS_TABLE4_CM1[name]
    correction = ABSORPTION_CALIBRATION["stark_corrections"].get(name) if calibrated else None
    if not correction:
        return levels.copy()
    lowest = levels.min()
    return lowest + correction["shift_cm1"] + correction["stretch"] * (levels - lowest)


# Active Stark levels (module defaults; main() rebuilds them for --original-model).
E_3H6_CM1 = corrected_levels("3H6", USE_ABSORPTION_CALIBRATION)
E_3F4_CM1 = corrected_levels("3F4", USE_ABSORPTION_CALIBRATION)
E_3H5_CM1 = corrected_levels("3H5", USE_ABSORPTION_CALIBRATION)
E_3H4_CM1 = corrected_levels("3H4", USE_ABSORPTION_CALIBRATION)
E_3F3_CM1 = corrected_levels("3F3", USE_ABSORPTION_CALIBRATION)
E_3F2_CM1 = corrected_levels("3F2", USE_ABSORPTION_CALIBRATION)

# ---------------------------------------------------------------------------
# Judd-Ofelt channel weights (Lee et al. 2021, Supplementary Information)
# ---------------------------------------------------------------------------
JO_OMEGA_1E20_CM2 = np.array([2.37, 3.05, 0.41])  # Omega_2, Omega_4, Omega_6 (SI Table S4)
# [U(2)]^2, [U(4)]^2, [U(6)]^2 (SI Table S5; Carnall et al. 1978, Tm3+ in LaF3)
JO_REDUCED_MATRIX_ELEMENTS = {
    "3H6 -> 3F4": (0.5395, 0.7261, 0.2421),
    "3H6 -> 3H5": (0.1074, 0.2314, 0.6385),
    "3F4 -> 3H4": (0.1275, 0.1311, 0.2113),
    # Not in the SI; from Carnall et al. 1978 (ANL-78-XX-95), Appendix XIII Table 2, Tm3+:LaF3
    # (same source as SI Table S5; the rows 3F3 -> 3F4 and 3F2 -> 3F4, read in the other direction):
    "3F4 -> 3F3": (0.0025, 0.0005, 0.1688),
    "3F4 -> 3F2": (0.3026, 0.0562, 0.0440),
}
# Fill in yourself (same format), e.g. from Carnall et al. 1978:
USER_REDUCED_MATRIX_ELEMENTS: dict = {}
# Or give a line strength S directly in 1e-20 cm^2 (takes precedence over everything):
USER_LINE_STRENGTH_OVERRIDE: dict = {}
JO_PLACEHOLDER_REFERENCE = "3F4 -> 3H4"  # used for channels without matrix elements
JO_USE_FREQUENCY_FACTOR = True  # weight = S * nu_bar (sigma_int ~ nu * S); False: weight = S

# Magnetic dipole (used when ABSORPTION_CALIBRATION["magnetic_dipole"] is on).
# |<psi J||L+2S||psi' J'>|^2 in intermediate coupling for Tm3+ (4f12). Eigenvectors from a
# free-ion fit (F2, F4, F6, zeta, alpha) to the Tm3+:LaF3 barycentres of Carnall et al. 1978
# (Appendix XIII Table 1); the same eigenvectors reproduce Carnall's [U(t)]^2 of all five
# channels to within 7 % (every element above 0.01; 3H6 -> 3H5 exactly). Pure LS limit for
# 3H6 -> 3H5: 10.83. Delta J = 2 channels (3H6 -> 3F4, 3F4 -> 3F2) have no MD part.
MD_REDUCED_MATRIX_ELEMENTS_SQ = {
    "3H6 -> 3H5": 10.74,
    "3F4 -> 3H4": 3.81,
    "3F4 -> 3F3": 4.33,
}
MD_PREFACTOR_1E20_CM2 = 0.03728  # (hbar/2mc)^2 = (mu_B/e)^2 = 3.728e-22 cm^2, in 1e-20 cm^2


def md_line_strength(name, refractive_index=None):
    """MD line strength in ED-equivalent units (1e-20 cm^2), ready to add to S_ED.

    Absorption: f_ED ~ (n^2+2)^2/(9n) S_ED and f_MD ~ n S'_MD, so S_MD,eff = 9n^2/(n^2+2)^2 S'_MD.
    """
    n = ABSORPTION_CALIBRATION["refractive_index"] if refractive_index is None else refractive_index
    s_md = MD_PREFACTOR_1E20_CM2 * MD_REDUCED_MATRIX_ELEMENTS_SQ.get(name, 0.0)
    return 9 * n**2 / (n**2 + 2) ** 2 * s_md


def line_strength(name):
    """Return (electric-dipole S in 1e-20 cm^2, is_placeholder)."""
    if name in USER_LINE_STRENGTH_OVERRIDE:
        return float(USER_LINE_STRENGTH_OVERRIDE[name]), False
    table = {**JO_REDUCED_MATRIX_ELEMENTS, **USER_REDUCED_MATRIX_ELEMENTS}
    if name in table:
        return float(JO_OMEGA_1E20_CM2 @ np.asarray(table[name], dtype=float)), False
    warnings.warn(
        f"No matrix elements for '{name}' in the SI; PLACEHOLDER = line strength of "
        f"'{JO_PLACEHOLDER_REFERENCE}'. Fill in USER_REDUCED_MATRIX_ELEMENTS.",
        stacklevel=2)
    return float(JO_OMEGA_1E20_CM2 @ np.asarray(table[JO_PLACEHOLDER_REFERENCE], dtype=float)), True


def channel_weight(name, lower, upper, include_md=False):
    """Relative channel weight (arbitrary scale; normalized per spectrum later).

    A USER_LINE_STRENGTH_OVERRIDE entry is taken as the total strength (no MD added).
    """
    strength, _ = line_strength(name)
    if include_md and name not in USER_LINE_STRENGTH_OVERRIDE:
        strength += md_line_strength(name)
    if JO_USE_FREQUENCY_FACTOR:
        strength *= float(np.mean(upper) - np.mean(lower))
    return strength


def build_channels(spec, include_md=False):
    return {name: (lower, upper, channel_weight(name, lower, upper, include_md))
            for name, (lower, upper) in spec.items()}


def model_configuration(calibrated=USE_ABSORPTION_CALIBRATION):
    """Levels, channels, widths and coupling of the calibrated or the original model."""
    cal = ABSORPTION_CALIBRATION
    levels = {name: corrected_levels(name, calibrated) for name in STARK_LEVELS_TABLE4_CM1}
    include_md = bool(calibrated and cal["magnetic_dipole"])
    h6, f4 = levels["3H6"], levels["3F4"]
    return {
        "calibrated": bool(calibrated),
        "levels": levels,
        "include_magnetic_dipole": include_md,
        "gsa_channels": build_channels({"3H6 -> 3F4": (h6, f4), "3H6 -> 3H5": (h6, levels["3H5"])},
                                       include_md),
        "esa_channels": build_channels({"3F4 -> 3H4": (f4, levels["3H4"]),
                                        "3F4 -> 3F3": (f4, levels["3F3"]),
                                        "3F4 -> 3F2": (f4, levels["3F2"])}, include_md),
        "electronic_sigma_gsa_cm1": cal["electronic_sigma_gsa_cm1"] if calibrated else ELECTRONIC_SIGMA_CM1,
        "electronic_sigma_esa_cm1": cal["electronic_sigma_esa_cm1"] if calibrated else ELECTRONIC_SIGMA_CM1,
        "coupling": cal["coupling"] if calibrated else DEFAULT_COUPLING,
    }


_DEFAULT_CONFIGURATION = model_configuration(USE_ABSORPTION_CALIBRATION)
GSA_CHANNELS = _DEFAULT_CONFIGURATION["gsa_channels"]
ESA_CHANNELS = _DEFAULT_CONFIGURATION["esa_channels"]


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
    if not np.isfinite(NORMALIZATION_WAVELENGTH_NM) or NORMALIZATION_WAVELENGTH_NM <= 0:
        raise ValueError("Normalization wavelength must be positive and finite.")


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


def validate_max_phonons(max_phonons):
    if (isinstance(max_phonons, (bool, np.bool_))
            or not isinstance(max_phonons, (int, np.integer))
            or max_phonons < 1):
        raise ValueError("Maximum phonons must be an integer of at least 1.")


def estimate_cost(max_phonons, enabled_bands, temperature_k, coupling):
    """Number of count combinations per channel and of Gaussian width groups.

    combinations = C(k+N, N) with k active event types (creation + consumption per band);
    width groups  = C(b+N, N) with b enabled bands (variance depends only on per-band totals).
    """
    bands = int(np.count_nonzero(enabled_bands))
    if coupling <= 0 or bands == 0:
        return 1, 1
    k = bands * (2 if temperature_k > 0 else 1)
    return math.comb(k + max_phonons, max_phonons), math.comb(bands + max_phonons, max_phonons)


def multiphonon_kernel(coupling, temperature_k, enabled_bands=None,
                      max_phonons=DEFAULT_MAX_PHONONS):
    """All count combinations with total created + consumed <= max_phonons.

    Independent Poisson weights are conditioned on this total-event cutoff.
    For counts c_a the weight is product(mu_a**c_a/c_a!)/D_N, where
    D_N=sum(mu_total**q/q!, q=0..N). Each unordered count vector occurs once.
    Gaussian variances add even when signed energy shifts cancel.
    """
    validate_max_phonons(max_phonons)
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


def validate_sum_tolerance(tolerance):
    if not np.isfinite(tolerance) or not 0 <= tolerance <= 1e-6:
        raise ValueError("Gaussian summation tolerance must be finite, between 0 and 1e-6.")


def merge_gaussian_components(components):
    """Merge equal centers only within equal-width, equal no-event groups.

    The input four Gaussian bands and every retained probability are unchanged.
    np.unique performs exact float comparisons: no energy or width binning.
    """
    groups = {}
    for centers, strengths, sigma, is_zero in components:
        center_parts, weight_parts = groups.setdefault((sigma, is_zero), ([], []))
        center_parts.append(centers)
        weight_parts.append(strengths)
    merged = []
    for (sigma, is_zero), (center_parts, weight_parts) in sorted(groups.items()):
        centers, inverse = np.unique(np.concatenate(center_parts), return_inverse=True)
        strengths = np.bincount(inverse, weights=np.concatenate(weight_parts), minlength=len(centers))
        positive = strengths > 0
        merged.append((centers[positive], strengths[positive], sigma, is_zero))
    return merged


def gaussian_mixture(energy, components, relative_tolerance=GAUSSIAN_RELATIVE_TOLERANCE):
    """Sorted local Gaussian sums with a positive relative omission bound.

    For each width group and query block, the omitted contribution is bounded
    by eps/G times the smallest already accumulated result in that block.
    Positivity bounds additional tolerance-based omissions by eps times the
    returned result; ordinary float64 rounding and underflow are separate.
    If the bound is zero, retain all non-underflow Gaussian contributions.
    """
    validate_sum_tolerance(relative_tolerance)
    values = np.asarray(energy, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Energies must be finite.")
    flat = values.ravel()
    order = np.argsort(flat)
    sorted_energy = flat[order]
    result = np.zeros_like(flat)
    group_count = max(1, len(components))
    for centers, strengths, sigma, _ in components:
        if not centers.size:
            continue
        amplitudes = strengths / (np.sqrt(2*np.pi)*sigma)
        amplitude_sum = float(amplitudes.sum())
        if amplitude_sum == 0:
            continue
        log_amplitude_sum = math.log(amplitude_sum)
        for start in range(0, flat.size, GAUSSIAN_QUERY_BLOCK_SIZE):
            stop = min(start + GAUSSIAN_QUERY_BLOCK_SIZE, flat.size)
            queries = sorted_energy[start:stop]
            if queries[-1] < centers[0]-40*sigma or queries[0] > centers[-1]+40*sigma:
                continue
            bound = float(result[start:stop].min()) * (relative_tolerance/group_count)
            if bound >= amplitude_sum:
                continue
            radius = (min(40.0, math.sqrt(max(0.0, 2*(log_amplitude_sum-math.log(bound)))))
                      if bound > 0 else 40.0)
            left = np.searchsorted(centers, queries[0]-radius*sigma, side="left")
            right = np.searchsorted(centers, queries[-1]+radius*sigma, side="right")
            if left == right:
                continue
            with np.errstate(over="ignore"):
                delta = (queries[:, None]-centers[None, left:right])/sigma
                result[start:stop] += np.exp(-0.5*delta**2) @ amplitudes[left:right]
    restored = np.empty_like(result)
    restored[order] = result
    return restored.reshape(values.shape)


@lru_cache(maxsize=32)
def _cached_channel_kernel(coupling, temperature_k, enabled_bands, max_phonons,
                           phonon_centers, phonon_sigmas, phonon_heights):
    # Include editable band parameters in the cache key to avoid stale kernels.
    return multiphonon_kernel(coupling, temperature_k, enabled_bands, max_phonons)


class MultiPhononSpectrum:
    """Electronic lines convolved with all retained phonon combinations.

    Components hold (shifted centers, weighted strengths, sigma, is_zero).
    The True flag identifies no phonon events, not all zero net shifts.
    The spectrum is normalized so that factor(NORMALIZATION_WAVELENGTH_NM) = 1.
    """

    def __init__(self, channels, coupling=None, temperature_k=TEMPERATURE_K,
                 enabled_bands=None, max_phonons=DEFAULT_MAX_PHONONS,
                 sum_tolerance=GAUSSIAN_RELATIVE_TOLERANCE, electronic_sigma_cm1=None,
                 coupling_multipliers=None):
        # Defaults follow the module configuration (calibrated unless switched off).
        if coupling is None:
            coupling = _DEFAULT_CONFIGURATION["coupling"]
        if electronic_sigma_cm1 is None:
            electronic_sigma_cm1 = _DEFAULT_CONFIGURATION["electronic_sigma_gsa_cm1"]
        if not np.isfinite(electronic_sigma_cm1) or electronic_sigma_cm1 <= 0:
            raise ValueError("Electronic Gaussian sigma must be positive and finite.")
        multipliers = CHANNEL_COUPLING_MULTIPLIERS if coupling_multipliers is None else coupling_multipliers
        self.electronic_sigma_cm1 = float(electronic_sigma_cm1)
        self.coupling = float(coupling)
        self.enabled_bands = band_mask(enabled_bands)
        validate_max_phonons(max_phonons)
        validate_sum_tolerance(sum_tolerance)
        # Validate independently even when every channel has zero weight.
        phonon_event_means(coupling, temperature_k, self.enabled_bands)
        self.max_phonons = max_phonons
        self.sum_tolerance = sum_tolerance
        self.components = []
        self.channel_kernels = {}
        self.channel_probabilities = {}
        channel_weights = np.array([w for _, _, w in channels.values()], dtype=float)
        if (not np.all(np.isfinite(channel_weights)) or np.any(channel_weights < 0)
                or channel_weights.sum() <= 0):
            raise ValueError("Channel weights must be finite, nonnegative and nonzero in total.")
        for name, (lower, upper, weight) in channels.items():
            multiplier = multipliers[name]
            kernel, diagnostics = _cached_channel_kernel(
                coupling * multiplier, temperature_k, tuple(self.enabled_bands), max_phonons,
                tuple(PHONON_CENTERS_CM1), tuple(PHONON_SIGMAS_CM1), tuple(PHONON_PEAK_HEIGHTS))
            self.channel_probabilities[name] = diagnostics
            self.channel_kernels[name] = kernel
            if weight == 0:
                continue
            centers, strengths = transition_sticks(lower, upper, temperature_k)
            strengths = strengths * weight / channel_weights.sum()
            for term in kernel:
                sigma = math.sqrt(self.electronic_sigma_cm1**2 + term["variance_cm2"])
                # In absorption, creation needs MORE photon energy; consumption LESS.
                self.components.append((centers + term["shift_cm1"],
                                        strengths * term["probability"], sigma,
                                        term["order"] == 0))
        self.unmerged_gaussian_count = sum(len(c) for c, _, _, _ in self.components)
        self.components = merge_gaussian_components(self.components)
        self.merged_gaussian_count = sum(len(c) for c, _, _, _ in self.components)
        self.gaussian_width_group_count = len(self.components)
        # Normalization: raw spectral density at the reference wavelength.
        self.reference = float(self.raw(1e7 / NORMALIZATION_WAVELENGTH_NM))
        if not np.isfinite(self.reference) or self.reference <= 0:
            raise ValueError(
                f"Spectrum is zero or invalid at {NORMALIZATION_WAVELENGTH_NM:g} nm; "
                "normalization is impossible (e.g. too few phonons or too weak coupling).")

    def raw(self, energy_cm1, zero_only=False):
        components = [c for c in self.components if c[3]] if zero_only else self.components
        return gaussian_mixture(energy_cm1, components, self.sum_tolerance)

    def factor(self, wavelengths_nm, zero_only=False):
        wavelengths = np.asarray(wavelengths_nm, dtype=float)
        if np.any(~np.isfinite(wavelengths)) or np.any(wavelengths <= 0):
            raise ValueError("Wavelengths must be finite and positive.")
        result = self.raw(1e7 / wavelengths, zero_only=zero_only) / self.reference
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


def plot_results(output_dir, wavelengths, gsa, esa, profile, normalization="global",
                 full_factors=None, variant_label=""):
    """Plot the model factors directly, with one constant scale per spectrum.

    Always saves two spectrum figures: one with a linear and one with a log y-axis.
    """
    suffix_label = f", {variant_label}" if variant_label else ""
    if normalization != "global":
        raise ValueError("Only global normalization is supported; local peak equalization was removed.")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"}):
        # Evaluate every curve once; both figures reuse the same data.
        curves = []
        for column, (spectrum, color, label) in enumerate(
                ((gsa, "#ba4a18", "GSA"), (esa, "#1468b1", "ESA"))):
            full = spectrum.factor(wavelengths) if full_factors is None else full_factors[:, column]
            zero = spectrum.factor(wavelengths, zero_only=True)
            curves.append((full, zero, color, label))
        spectrum_figs = []
        for scale_name, log_y in (("linear", False), ("log", True)):
            fig, ax = plt.subplots(figsize=(10.5, 5.8))
            for full, zero, color, label in curves:
                full_plot = np.where(full > 0, full, np.nan) if log_y else full
                zero_plot = np.where(zero > 0, zero, np.nan) if log_y else zero
                ax.plot(wavelengths, full_plot, color=color, label=label)
                ax.plot(wavelengths, zero_plot, color=color,
                        linestyle="--", alpha=0.65, label=f"{label}, zero-phonon only")
            # Upper limit is left free: after 1064 nm normalization the peaks can lie far above 1.
            ax.set(xlabel="Pump wavelength (nm)", ylabel="Relative absorption factor f",
                   xlim=(wavelengths[0], wavelengths[-1]),
                   yscale="log" if log_y else "linear",
                   title=f"GSA and ESA factors ({scale_name} y-axis{suffix_label})")
            peak = max(float(np.nanmax(c[0])) for c in curves)
            # Top is set explicitly: a log axis otherwise pads symmetrically in decades.
            ax.set_ylim(PLOT_MIN_FACTOR, 10*peak) if log_y else ax.set_ylim(0.0, 1.05*peak)
            ax.axvline(NORMALIZATION_WAVELENGTH_NM, color="#7d858c", linestyle=":", linewidth=1,
                       label=f"{NORMALIZATION_WAVELENGTH_NM:g} nm (f = 1)")
            ax.grid(alpha=0.2)
            ax.legend(fontsize=9, ncol=2)
            fig.tight_layout()
            for suffix in ("png", "svg"):
                fig.savefig(output_dir / f"f_gsa_f_esa_four_gaussian_multiphonon_{scale_name}.{suffix}", dpi=180)
            spectrum_figs.append(fig)

        profile_fig, profile_ax = plt.subplots(figsize=(9, 4.8))
        grid, components, total = profile
        enabled = gsa.enabled_bands
        # Scale against the full reference profile, also when every band is off.
        scale = components.sum(axis=1).max()
        for k in range(4):
            style = {} if enabled[k] else {"color": "#bfc4c9", "alpha": 0.55}
            status = "" if enabled[k] else " (off)"
            profile_ax.plot(grid, components[:, k]/scale, linestyle="--", **style,
                            label=f"Gaussian {k+1}: {PHONON_CENTERS_CM1[k]:g} cm⁻¹{status}")
        profile_ax.plot(grid, total/scale, color="#303844", linewidth=2, label="Sum of active bands")
        profile_ax.set(xlabel="Raman shift (cm⁻¹)", ylabel="Relative profile intensity",
                       xlim=(200, 450), ylim=(0, 1.08))
        profile_ax.legend(fontsize=9)
        profile_ax.grid(alpha=0.2)
        profile_fig.tight_layout()
        for suffix in ("png", "svg"):
            profile_fig.savefig(output_dir / f"phonon_profile_four_gaussians_multiphonon.{suffix}", dpi=180)
        return (*spectrum_figs, profile_fig)


def channel_weight_report(configuration=None):
    """Judd-Ofelt (+ MD) line strengths and (un-normalized) weights per channel, for the JSON/log."""
    configuration = _DEFAULT_CONFIGURATION if configuration is None else configuration
    include_md = configuration["include_magnetic_dipole"]
    report = {}
    for spectrum_name in ("gsa", "esa"):
        channels = configuration[f"{spectrum_name}_channels"]
        total = sum(w for _, _, w in channels.values())
        for name, (_, _, weight) in channels.items():
            strength, placeholder = line_strength(name)
            md = md_line_strength(name) if include_md and name not in USER_LINE_STRENGTH_OVERRIDE else 0.0
            report[name] = {"spectrum": spectrum_name,
                            "line_strength_1e-20_cm2": strength + md,
                            "line_strength_ed_1e-20_cm2": strength,
                            "line_strength_md_effective_1e-20_cm2": md,
                            "weight": weight,
                            "share_in_spectrum": weight / total,
                            "placeholder": placeholder}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-phonons", type=int, default=DEFAULT_MAX_PHONONS, metavar="N",
                        help="Maximum total phonons per transition, any integer >= 1, "
                             "including all lower orders (default: %(default)s).")
    parser.add_argument("--sum-tolerance", type=float, default=GAUSSIAN_RELATIVE_TOLERANCE,
                        help="Relative numerical Gaussian-sum omission bound (default: %(default)s); 0 disables adaptive omission.")
    parser.add_argument("--coupling", type=float, default=None,
                        help="Dimensionless coupling strength; 0 disables sidebands "
                             f"(default: {ABSORPTION_CALIBRATION['coupling']:g} calibrated, "
                             f"{DEFAULT_COUPLING:g} with --original-model).")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE_K, help="Temperature in K, including 0 K.")
    parser.add_argument("--original-model", action="store_true",
                        help="Previous, uncalibrated model: Table 4 levels, sigma 8.5 cm^-1, "
                             "coupling 0.05, electric-dipole weights only.")
    switches = parser.add_mutually_exclusive_group()
    switches.add_argument("--no-phonons", action="store_true", help="Disable all four phonon bands.")
    switches.add_argument("--phonon-bands", type=int, nargs="+", choices=(1, 2, 3, 4),
                          help="Enable only these bands, for example: --phonon-bands 1 3")
    parser.add_argument("--plot-normalization", choices=("global",), default="global",
                        help="Show the model factors directly (default); no local peak equalization.")
    parser.add_argument("--data-only", action="store_true", help="CSV/JSON only; no Matplotlib needed.")
    parser.add_argument("--no-show", action="store_true", help="Save plots without opening a window.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory; default: results/max_N_phonons beside this script "
                             "(results/max_N_phonons_original with --original-model).")
    args = parser.parse_args()
    if args.max_phonons < 1:
        parser.error("--max-phonons must be at least 1.")
    configuration = model_configuration(USE_ABSORPTION_CALIBRATION and not args.original_model)
    variant = "absorption-calibrated" if configuration["calibrated"] else "original model"
    if args.coupling is None:
        args.coupling = configuration["coupling"]
    if (not np.all(np.isfinite([WAVELENGTH_MIN_NM, WAVELENGTH_MAX_NM, WAVELENGTH_STEP_NM]))
            or WAVELENGTH_MIN_NM <= 0 or WAVELENGTH_MAX_NM <= WAVELENGTH_MIN_NM or WAVELENGTH_STEP_NM <= 0):
        raise ValueError("Invalid wavelength range or step.")
    enabled = (np.zeros(4, dtype=bool) if args.no_phonons else
               np.isin(np.arange(1, 5), args.phonon_bands) if args.phonon_bands is not None else
               band_mask())
    if args.output_dir is None:
        suffix = "" if configuration["calibrated"] else "_original"
        args.output_dir = (Path(__file__).resolve().parent / "results"
                           / f"max_{args.max_phonons}_phonons{suffix}")
    print(f"Model: {variant}")
    combinations, width_groups = estimate_cost(args.max_phonons, enabled, args.temperature, args.coupling)
    print(f"Estimate: {combinations:,} phonon combinations per channel, up to {width_groups:,} Gaussian width groups.")
    if combinations > COMBINATION_WARNING_THRESHOLD:
        warnings.warn(f"{combinations:,} combinations per channel: this run can take a long time and a lot of memory. "
                      "Consider a lower --max-phonons.", stacklevel=1)
    gsa = MultiPhononSpectrum(configuration["gsa_channels"], args.coupling, args.temperature, enabled,
                              args.max_phonons, args.sum_tolerance,
                              electronic_sigma_cm1=configuration["electronic_sigma_gsa_cm1"])
    esa = MultiPhononSpectrum(configuration["esa_channels"], args.coupling, args.temperature, enabled,
                              args.max_phonons, args.sum_tolerance,
                              electronic_sigma_cm1=configuration["electronic_sigma_esa_cm1"])
    n = int(np.ceil((WAVELENGTH_MAX_NM-WAVELENGTH_MIN_NM)/WAVELENGTH_STEP_NM)) + 1
    wavelengths = np.linspace(WAVELENGTH_MIN_NM, WAVELENGTH_MAX_NM, n)
    factors = np.column_stack((wavelengths, gsa.factor(wavelengths), esa.factor(wavelengths)))
    if np.any(~np.isfinite(factors)) or np.any(factors[:, 1:] < 0):
        raise RuntimeError("Invalid spectral factors.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output_dir / "f_gsa_f_esa_four_gaussian_multiphonon.csv", factors,
               delimiter=",", header="wavelength_nm,f_gsa,f_esa", comments="", fmt="%.12g")
    profile = export_profile(args.output_dir, enabled)
    weight_report = channel_weight_report(configuration)
    settings = {
        "model": "conditional_truncated_independent_Poisson_Gaussian_bands",
        "model_variant": variant,
        "absorption_calibration": (ABSORPTION_CALIBRATION if configuration["calibrated"] else None),
        "stark_levels_cm1": {name: levels.tolist() for name, levels in configuration["levels"].items()},
        "stark_levels_source": "J. Phys. Chem. C 129, 20446 (2025), Table 4(B) (ab initio)"
                               + ("; with the absorption calibration applied" if configuration["calibrated"] else ""),
        "magnetic_dipole_included": configuration["include_magnetic_dipole"],
        "md_reduced_matrix_elements_sq": (MD_REDUCED_MATRIX_ELEMENTS_SQ
                                          if configuration["include_magnetic_dipole"] else None),
        "maximum_phonons_per_absorption": args.max_phonons,
        "cutoff_counts": "total created plus consumed phonons; includes orders 0 through maximum",
        "gaussian_sum_relative_omission_bound": args.sum_tolerance,
        "gaussian_sum_rounding_note": "Bound on additional tolerance-based omissions; ordinary float64 rounding and underflow are separate. No floor, FFT or interpolation.",
        "gaussian_evaluation": "merge exact centers at equal width; sorted local positive sums with relative omission bound",
        "gaussian_compression": {
            label: {"before_merging": spectrum.unmerged_gaussian_count,
                    "after_merging": spectrum.merged_gaussian_count,
                    "width_groups": spectrum.gaussian_width_group_count}
            for label, spectrum in (("gsa", gsa), ("esa", esa))},
        "normalization": f"GSA and ESA each scaled to 1 at {NORMALIZATION_WAVELENGTH_NM:g} nm",
        "normalization_wavelength_nm": NORMALIZATION_WAVELENGTH_NM,
        "temperature_k": args.temperature, "coupling": args.coupling,
        "channel_coupling_multipliers": CHANNEL_COUPLING_MULTIPLIERS,
        "channel_weights_judd_ofelt": weight_report,
        "judd_ofelt_omega_1e-20_cm2": JO_OMEGA_1E20_CM2.tolist(),
        "judd_ofelt_use_frequency_factor": JO_USE_FREQUENCY_FACTOR,
        "judd_ofelt_source": "Lee et al., Nature 589, 230 (2021), Supplementary Tables S4 and S5",
        "phonon_bands_enabled": enabled.tolist(),
        "active_phonon_bands": (np.flatnonzero(enabled) + 1).tolist(),
        "plot_normalization": args.plot_normalization,
        "plot_y_scales": ["linear", "log"],
        "phonon_centers_cm1": PHONON_CENTERS_CM1.tolist(),
        "phonon_sigmas_cm1": PHONON_SIGMAS_CM1.tolist(),
        "phonon_peak_heights": PHONON_PEAK_HEIGHTS.tolist(),
        "phonon_area_weights": phonon_area_weights().tolist(),
        "electronic_sigma_cm1": {"gsa": gsa.electronic_sigma_cm1, "esa": esa.electronic_sigma_cm1},
        "channel_probabilities": gsa.channel_probabilities | esa.channel_probabilities,
        "channel_combinations": {
            name: (kernel if len(kernel) <= JSON_COMBINATION_LIMIT else
                   f"omitted: {len(kernel)} combinations (> JSON_COMBINATION_LIMIT = {JSON_COMBINATION_LIMIT})")
            for name, kernel in (gsa.channel_kernels | esa.channel_kernels).items()},
        "reference_raw_per_cm1_at_normalization_wavelength": {"gsa": gsa.reference, "esa": esa.reference},
        "factors_at_normalization_wavelength": {
            "gsa": gsa.factor(NORMALIZATION_WAVELENGTH_NM),
            "esa": esa.factor(NORMALIZATION_WAVELENGTH_NM)},
        "profile_source": "Wilhelm et al. 2013 Fig.3A https://doi.org/10.7150/thno.5113",
        "assumptions": ["Four Gaussians approximate a manually traced Raman curve, not raw data.",
                        "Raman intensities are a coupling-shape proxy; no Raman Bose correction.",
                        "Bose occupation evaluated at each Gaussian center; finite-band approximation.",
                        "Independent phonon events, including repeated-band sums and signed differences.",
                        "All retained orders are normalized together; omitted Poisson mass is reported.",
                        "Gaussian phonon variances add for independently sampled events, even at zero net shift.",
                        "Gaussian bands unchanged; adaptive arithmetic omissions have a relative bound, with no artificial absorption floor.",
                        "Use --sum-tolerance 0 to evaluate every non-underflow Gaussian contribution.",
                        "Channel weights from Judd-Ofelt: Omega_t from SI Table S4; [U(t)]^2 from SI Table S5 and Carnall et al. 1978 (Appendix XIII Table 2, Tm3+:LaF3)."]
                       + (["Calibrated: width, coupling and Stark corrections fitted to the RT GSA bands of NaTmF4@NaYF4 in cyclohexane; "
                           "the 1716-1772 nm part of 3F4 (solvent opaque) was excluded.",
                           "Calibrated: ESA width equals the GSA width (not measured); ESA follows from the calibrated levels.",
                           "Calibrated: magnetic-dipole strengths added (intermediate coupling), S_MD,eff = 9n^2/(n^2+2)^2 S'_MD."]
                          if configuration["calibrated"] else []),
    }
    (args.output_dir / "four_gaussian_multiphonon_settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"f_GSA({NORMALIZATION_WAVELENGTH_NM:g} nm) = {gsa.factor(NORMALIZATION_WAVELENGTH_NM):.8g}; "
          f"f_ESA({NORMALIZATION_WAVELENGTH_NM:g} nm) = {esa.factor(NORMALIZATION_WAVELENGTH_NM):.8g}")
    print(f"Results: {args.output_dir.resolve()}")
    print(f"Active phonon bands: {(np.flatnonzero(enabled) + 1).tolist()}")
    print(f"Maximum phonons per transition: {args.max_phonons} (all lower orders included)")
    print(f"Electronic sigma: GSA {gsa.electronic_sigma_cm1:g} cm-1, ESA {esa.electronic_sigma_cm1:g} cm-1; "
          f"coupling {args.coupling:g}")
    for name, info in weight_report.items():
        flag = "  [PLACEHOLDER]" if info["placeholder"] else ""
        md = info["line_strength_md_effective_1e-20_cm2"]
        md_text = f" (ED {info['line_strength_ed_1e-20_cm2']:.4f} + MD {md:.4f})" if md else ""
        print(f"{name}: S = {info['line_strength_1e-20_cm2']:.4f} e-20 cm2{md_text}, "
              f"share in {info['spectrum'].upper()} = {info['share_in_spectrum']:.3f}{flag}")
    for name, diagnostics in settings["channel_probabilities"].items():
        print(f"{name}: order probabilities = {diagnostics['order_probabilities']}; "
              f"omitted Poisson mass = {diagnostics['poisson_omitted_probability']:.4g}")
    if not args.data_only:
        figures = plot_results(args.output_dir, wavelengths, gsa, esa, profile,
                               args.plot_normalization,
                               full_factors=factors[:, 1:], variant_label=variant)
        import matplotlib.pyplot as plt
        if args.no_show:
            for figure in figures:
                plt.close(figure)
        else:
            plt.show()


if __name__ == "__main__":
    main()
