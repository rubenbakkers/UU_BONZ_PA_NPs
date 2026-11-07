# -*- coding: utf-8 -*-
"""Independent numerical checks for the selectable multiphonon expansion.

Run with NumPy installed:
    python verify_multiphonon.py

The checks compare probabilities against closed-form Poisson/multinomial
results and compare N=1 spectra against the unmodified supplied script.
They verify the implementation, not the physical validity of the proxy.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import unittest

import numpy as np

import gsa_esa_four_gaussian_multiphonon as model


def original_module():
    path = Path(__file__).resolve().parent / "backups" / "gsa_esa_four_gaussian_single_phonon_original.py"
    spec = importlib.util.spec_from_file_location("original_single_phonon_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def independent_means(coupling=0.05, temperature_k=300.0, enabled=None):
    """Calculate the event means directly from the documented assumptions."""
    weights = model.PHONON_PEAK_HEIGHTS * model.PHONON_SIGMAS_CM1
    weights = weights / weights.sum()
    if enabled is not None:
        weights = weights * np.asarray(enabled, dtype=bool)
    occupations = (np.zeros(4) if temperature_k == 0 else
                   1.0 / np.expm1(model.PHONON_CENTERS_CM1 /
                                  (model.K_B_CM1_PER_K * temperature_k)))
    return coupling * weights * (occupations + 1), coupling * weights * occupations


def normalization(mu, max_phonons):
    return math.fsum(mu**q / math.factorial(q) for q in range(max_phonons + 1))


def find_component(components, create=(0, 0, 0, 0), consume=(0, 0, 0, 0)):
    matches = [c for c in components if tuple(c["counts_create"]) == create
               and tuple(c["counts_consume"]) == consume]
    if len(matches) != 1:
        raise AssertionError(f"Expected one component for {create}, {consume}; got {len(matches)}")
    return matches[0]


class MultiphononKernelChecks(unittest.TestCase):
    def test_normalization_all_orders_and_independent_order_probabilities(self):
        plus, minus = independent_means()
        mu = float(plus.sum() + minus.sum())
        for maximum in range(1, 5):
            with self.subTest(max_phonons=maximum):
                components, diagnostics = model.multiphonon_kernel(0.05, 300, max_phonons=maximum)
                # All eight event types have nonzero means at finite T.
                self.assertEqual(len(components), math.comb(maximum + 8, 8))
                probabilities = np.asarray([c["probability"] for c in components])
                self.assertTrue(np.all(np.isfinite(probabilities)))
                self.assertTrue(np.all(probabilities >= 0))
                self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=13)
                denominator = normalization(mu, maximum)
                expected = [mu**q / math.factorial(q) / denominator
                            for q in range(maximum + 1)]
                actual = [math.fsum(c["probability"] for c in components if c["order"] == q)
                          for q in range(maximum + 1)]
                np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=1e-16)
                np.testing.assert_allclose(diagnostics["order_probabilities"], expected,
                                           rtol=3e-14, atol=1e-16)
                self.assertAlmostEqual(diagnostics["zero_phonon"], 1 / denominator, places=14)
                self.assertAlmostEqual(diagnostics["mean_untruncated_events"], mu, places=14)
                self.assertAlmostEqual(diagnostics["poisson_omitted_probability"],
                                       1 - math.exp(-mu) * denominator, places=14)
                # The first moment is also available independently of the enumeration.
                expected_mean_shift = float(np.dot(plus - minus, model.PHONON_CENTERS_CM1))
                expected_mean_shift *= normalization(mu, maximum - 1) / denominator
                mean_shift = math.fsum(c["probability"] * c["shift_cm1"] for c in components)
                self.assertAlmostEqual(mean_shift, expected_mean_shift, places=12)

    def test_repetitions_sums_differences_widths_and_combinatorics(self):
        components, _ = model.multiphonon_kernel(0.05, 300, max_phonons=4)
        plus, minus = independent_means()
        denominator = normalization(float(plus.sum() + minus.sum()), 4)
        repeat = find_component(components, create=(2, 0, 0, 0))
        self.assertEqual(repeat["shift_cm1"], 2 * 253)
        self.assertEqual(repeat["variance_cm2"], 2 * 14**2)
        self.assertAlmostEqual(repeat["probability"], plus[0]**2 / (2 * denominator), places=15)
        mixed = find_component(components, create=(1, 1, 0, 0))
        self.assertEqual(mixed["shift_cm1"], 253 + 306)
        self.assertEqual(mixed["variance_cm2"], 14**2 + 24**2)
        self.assertAlmostEqual(mixed["probability"], plus[0] * plus[1] / denominator, places=15)
        difference = find_component(components, create=(1, 0, 0, 0), consume=(0, 1, 0, 0))
        self.assertEqual(difference["shift_cm1"], 253 - 306)
        self.assertEqual(difference["variance_cm2"], 14**2 + 24**2)
        self.assertAlmostEqual(difference["probability"], plus[0] * minus[1] / denominator, places=15)
        fourth = find_component(components, create=(4, 0, 0, 0))
        self.assertEqual(fourth["order"], 4)
        self.assertEqual(fourth["shift_cm1"], 4 * 253)
        self.assertEqual(fourth["variance_cm2"], 4 * 14**2)
        self.assertAlmostEqual(fourth["probability"], plus[0]**4 / (24 * denominator), places=15)

    def test_zero_events_are_distinct_from_zero_net_energy(self):
        components, diagnostics = model.multiphonon_kernel(0.05, 300, max_phonons=2)
        zero = find_component(components)
        balanced = find_component(components, create=(1, 0, 0, 0), consume=(1, 0, 0, 0))
        self.assertEqual(zero["order"], 0)
        self.assertEqual(zero["variance_cm2"], 0)
        self.assertEqual(balanced["order"], 2)
        self.assertEqual(balanced["shift_cm1"], 0)
        self.assertEqual(balanced["variance_cm2"], 2 * 14**2)
        self.assertGreater(balanced["probability"], 0)
        self.assertEqual(diagnostics["zero_phonon"], zero["probability"])

    def test_zero_coupling_disabled_bands_and_zero_temperature(self):
        for maximum in range(1, 5):
            for coupling, enabled in ((0, None), (0.05, [False] * 4)):
                with self.subTest(max_phonons=maximum, coupling=coupling, enabled=enabled):
                    components, diagnostic = model.multiphonon_kernel(
                        coupling, 300, enabled_bands=enabled, max_phonons=maximum)
                    self.assertEqual(find_component(components)["probability"], 1)
                    self.assertEqual(sum(c["probability"] for c in components if c["order"] > 0), 0)
                    self.assertEqual(diagnostic["poisson_omitted_probability"], 0)
        components, _ = model.multiphonon_kernel(0.05, 0, max_phonons=4)
        self.assertTrue(all(not any(c["counts_consume"]) for c in components if c["probability"] > 0))
        self.assertTrue(all(c["shift_cm1"] >= 0 for c in components if c["probability"] > 0))
        # Disabling bands must not redistribute their original area weights.
        enabled = [True, False, False, False]
        components, _ = model.multiphonon_kernel(0.05, 300, enabled_bands=enabled, max_phonons=2)
        plus, minus = independent_means(enabled=enabled)
        denominator = normalization(float(plus.sum() + minus.sum()), 2)
        self.assertAlmostEqual(find_component(components, create=(1, 0, 0, 0))["probability"],
                               plus[0] / denominator, places=15)

    def test_invalid_inputs(self):
        for maximum in (0, 5, -1, 1.5, True, np.bool_(True)):
            with self.subTest(max_phonons=maximum), self.assertRaises(ValueError):
                model.multiphonon_kernel(0.05, 300, max_phonons=maximum)
        for coupling in (-0.01, float("nan"), float("inf")):
            with self.subTest(coupling=coupling), self.assertRaises(ValueError):
                model.multiphonon_kernel(coupling, 300)
        for temperature in (-1, float("nan"), float("inf")):
            with self.subTest(temperature=temperature), self.assertRaises(ValueError):
                model.multiphonon_kernel(0.05, temperature)


class OriginalSinglePhononRegression(unittest.TestCase):
    def test_n1_probabilities_and_complete_gsa_esa_spectra(self):
        original = original_module()
        components, diagnostic = model.multiphonon_kernel(0.05, 300, max_phonons=1)
        p0, plus, minus = original.single_phonon_probabilities(0.05, 300)
        self.assertAlmostEqual(diagnostic["zero_phonon"], p0, places=14)
        np.testing.assert_allclose(diagnostic["create_one"], plus, rtol=2e-14, atol=1e-16)
        np.testing.assert_allclose(diagnostic["consume_one"], minus, rtol=2e-14, atol=1e-16)
        wavelengths = np.unique(np.r_[np.linspace(900, 2000, 177), 1064.0])
        for label in ("GSA", "ESA"):
            with self.subTest(spectrum=label):
                channels = getattr(model, label + "_CHANNELS")
                old = original.SinglePhononSpectrum(channels)
                new = model.MultiPhononSpectrum(channels, max_phonons=1)
                for zero_only in (False, True):
                    np.testing.assert_allclose(new.raw(1e7 / wavelengths, zero_only=zero_only),
                                               old.raw(1e7 / wavelengths, zero_only=zero_only),
                                               rtol=2e-12, atol=1e-17)
                    np.testing.assert_allclose(new.factor(wavelengths, zero_only=zero_only),
                                               old.factor(wavelengths, zero_only=zero_only),
                                               rtol=2e-12, atol=1e-15)
                self.assertTrue(np.all(np.isfinite(new.factor(wavelengths))))
                self.assertTrue(np.all(new.factor(wavelengths) >= 0))
                # The support mask must preserve scalar and multidimensional APIs.
                scalar = new.factor(1064.0)
                self.assertIsInstance(scalar, float)
                self.assertEqual(scalar, new.factor(np.array(1064.0)))
                matrix = np.array([[950.0, 1064.0], [1450.0, 1900.0]])
                np.testing.assert_array_equal(new.factor(matrix).ravel(), new.factor(matrix.ravel()))
                self.assertEqual(new.raw(np.array([])).shape, (0,))
                self.assertEqual(new.factor(np.empty((0, 2))).shape, (0, 2))
                for invalid in (0, -1, float("nan"), float("inf")):
                    with self.assertRaises(ValueError):
                        new.factor(invalid)
                for invalid in (float("nan"), float("inf")):
                    with self.assertRaises(ValueError):
                        new.raw(invalid)


if __name__ == "__main__":
    unittest.main(verbosity=2)
