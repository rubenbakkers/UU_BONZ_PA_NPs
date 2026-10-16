# -*- coding: utf-8 -*-
"""Reproducible serial timing and numerical comparison of the Gaussian model.

Runs the backed-up original at four phonons, the optimized model at four,
and the optimized model at six. Each case constructs both spectra (including
their global normalization searches) and evaluates the same 22,001 wavelengths.
No plots or model-result exports are included in the measured time.

Usage:
    python benchmark_multiphonon.py
    python benchmark_multiphonon.py --output-dir results/performance

Only NumPy and Python's standard library are required. Timings describe this
machine and this run; they are not a guarantee for other machines or settings.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "gsa_esa_four_gaussian_multiphonon.py"
ORIGINAL_PATH = (SCRIPT_DIR / "backups" / "before_six_phonons_20261016"
                 / "gsa_esa_four_gaussian_multiphonon.py")
COMPARISON_RTOL = 2e-12
COMPARISON_ATOL = float(np.nextafter(0.0, 1.0)) * 8


def load_model(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import model: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_shared_parameters(original, optimized):
    """Refuse a timing comparison if its physical input parameters differ."""
    for name in ("PHONON_CENTERS_CM1", "PHONON_SIGMAS_CM1", "PHONON_PEAK_HEIGHTS",
                 "PHONON_BANDS_ENABLED", "E_3H6_CM1", "E_3F4_CM1", "E_3H5_CM1",
                 "E_3H4_CM1", "E_3F3_CM1", "E_3F2_CM1"):
        if not np.array_equal(getattr(original, name), getattr(optimized, name)):
            raise ValueError(f"Benchmark physical parameters differ: {name}")
    for name in ("ELECTRONIC_SIGMA_CM1", "K_B_CM1_PER_K",
                 "CHANNEL_COUPLING_MULTIPLIERS"):
        if getattr(original, name) != getattr(optimized, name):
            raise ValueError(f"Benchmark physical parameters differ: {name}")
    for name in ("GSA_CHANNELS", "ESA_CHANNELS"):
        old_channels, new_channels = getattr(original, name), getattr(optimized, name)
        if old_channels.keys() != new_channels.keys():
            raise ValueError(f"Benchmark channels differ: {name}")
        for key in old_channels:
            if not all(np.array_equal(a, b) for a, b in
                       zip(old_channels[key], new_channels[key])):
                raise ValueError(f"Benchmark channel parameters differ: {key}")


def spectrum_statistics(spectrum):
    components = spectrum.components
    unmerged = getattr(spectrum, "unmerged_gaussian_count",
                       sum(len(item[0]) for item in components))
    merged = getattr(spectrum, "merged_gaussian_count", unmerged)
    return {
        "peak_per_cm1": spectrum.peak,
        "factor_at_1064nm": spectrum.factor(1064.0),
        "unmerged_gaussian_count": unmerged,
        "evaluated_gaussian_count": merged,
        "component_groups": len(components),
        "gaussian_count_compression_ratio": unmerged / merged,
    }


def run_case(label, model, max_phonons, wavelengths, coupling, temperature):
    # Start each optimized case with a cold kernel cache. Sharing that cache
    # between its GSA and ESA constructors is part of the implemented model.
    cache = getattr(model, "_cached_channel_kernel", None)
    if cache is not None:
        cache.cache_clear()
    spectra, timings = {}, {}
    for kind in ("gsa", "esa"):
        print(f"{label}: constructing {kind.upper()} and finding its global peak...", flush=True)
        started = time.perf_counter()
        spectra[kind] = model.MultiPhononSpectrum(
            getattr(model, f"{kind.upper()}_CHANNELS"), coupling=coupling,
            temperature_k=temperature, max_phonons=max_phonons)
        timings[f"construct_{kind}_seconds"] = time.perf_counter() - started
        print(f"{label}: {kind.upper()} construction "
              f"{timings[f'construct_{kind}_seconds']:.3f} s", flush=True)
    factors = {}
    for kind in ("gsa", "esa"):
        print(f"{label}: evaluating {kind.upper()} at {len(wavelengths):,} wavelengths...", flush=True)
        started = time.perf_counter()
        factors[kind] = spectra[kind].factor(wavelengths)
        timings[f"scan_{kind}_seconds"] = time.perf_counter() - started
        if np.any(~np.isfinite(factors[kind])) or np.any(factors[kind] < 0):
            raise RuntimeError(f"Invalid {kind.upper()} factors in {label}.")
        print(f"{label}: {kind.upper()} scan {timings[f'scan_{kind}_seconds']:.3f} s", flush=True)
    timings["construction_seconds"] = sum(timings[f"construct_{k}_seconds"] for k in spectra)
    timings["scan_seconds"] = sum(timings[f"scan_{k}_seconds"] for k in spectra)
    timings["total_seconds"] = timings["construction_seconds"] + timings["scan_seconds"]
    report = {
        "maximum_phonons": max_phonons,
        "timings": timings,
        "gaussian_relative_tolerance": getattr(model, "GAUSSIAN_RELATIVE_TOLERANCE", 0.0),
        "spectra": {kind: spectrum_statistics(spectrum) for kind, spectrum in spectra.items()},
    }
    print(f"{label}: measured total {timings['total_seconds']:.3f} s", flush=True)
    return report, factors


def compare_four_phonons(reference, optimized):
    comparison = {}
    for kind in ("gsa", "esa"):
        before, after = reference[kind], optimized[kind]
        absolute = np.abs(after - before)
        normal = before > 1e-280
        relative = absolute[normal] / before[normal]
        comparison[kind] = {
            "allclose": bool(np.allclose(after, before, rtol=COMPARISON_RTOL,
                                         atol=COMPARISON_ATOL)),
            "maximum_absolute_factor_difference": float(absolute.max()),
            "maximum_relative_factor_difference_for_reference_gt_1e_minus_280":
                float(relative.max()) if relative.size else None,
            "relative_comparison_point_count": int(normal.sum()),
            "reference_minimum_positive_factor": float(before[before > 0].min()),
            "optimized_minimum_positive_factor": float(after[after > 0].min()),
            "zero_nonzero_disagreement_count": int(np.count_nonzero((before == 0) != (after == 0))),
        }
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results" / "performance")
    args = parser.parse_args()
    if not ORIGINAL_PATH.is_file():
        raise FileNotFoundError(f"Original backup required for comparison: {ORIGINAL_PATH}")
    original = load_model("four_gaussian_benchmark_original", ORIGINAL_PATH)
    optimized = load_model("four_gaussian_benchmark_optimized", MODEL_PATH)
    check_shared_parameters(original, optimized)
    wavelengths = np.linspace(900.0, 2000.0, 22001)
    coupling, temperature = 0.05, 300.0
    print("Serial benchmark: original N=4, optimized N=4, optimized N=6.", flush=True)
    old_report, old_factors = run_case("original_four", original, 4, wavelengths, coupling, temperature)
    four_report, four_factors = run_case("optimized_four", optimized, 4, wavelengths, coupling, temperature)
    six_report, _ = run_case("optimized_six", optimized, 6, wavelengths, coupling, temperature)
    comparison = compare_four_phonons(old_factors, four_factors)
    passed = all(item["allclose"] for item in comparison.values())
    report = {
        "description": "Serial single-run timing; global-peak search plus full GSA and ESA scans; no plots or export.",
        "environment": {"python_version": platform.python_version(), "numpy_version": np.__version__,
                        "platform": platform.platform(), "processor": platform.processor()},
        "parameters": {"temperature_k": temperature, "coupling": coupling,
                       "wavelength_min_nm": float(wavelengths[0]), "wavelength_max_nm": float(wavelengths[-1]),
                       "wavelength_step_nm": 0.05, "wavelength_count": int(wavelengths.size),
                       "phonon_centers_cm1": optimized.PHONON_CENTERS_CM1.tolist(),
                       "phonon_sigmas_cm1": optimized.PHONON_SIGMAS_CM1.tolist(),
                       "phonon_peak_heights": optimized.PHONON_PEAK_HEIGHTS.tolist(),
                       "electronic_sigma_cm1": optimized.ELECTRONIC_SIGMA_CM1,
                       "phonon_bands_enabled": optimized.PHONON_BANDS_ENABLED.tolist()},
        "source_sha256": {"original": hashlib.sha256(ORIGINAL_PATH.read_bytes()).hexdigest(),
                          "optimized": hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()},
        "runs": {"original_four": old_report, "optimized_four": four_report, "optimized_six": six_report},
        "comparison_four_phonons": {"rtol": COMPARISON_RTOL, "atol": COMPARISON_ATOL,
                                    "passed": passed, "spectra": comparison},
        "measured_total_speedup_four_phonons": old_report["timings"]["total_seconds"] / four_report["timings"]["total_seconds"],
        "optimized_six_vs_original_four_total_speedup": old_report["timings"]["total_seconds"] / six_report["timings"]["total_seconds"],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "benchmark_report.json"
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Four-phonon numerical agreement: {'PASS' if passed else 'FAIL'}", flush=True)
    print(f"Measured speedup at four phonons: {report['measured_total_speedup_four_phonons']:.2f}x", flush=True)
    print(f"Report: {output.resolve()}", flush=True)
    if not passed:
        raise SystemExit("The numerical agreement check failed; inspect the saved report.")


if __name__ == "__main__":
    main()
