"""Tests for bearloc.samples: each sample behaves as its description says."""

import unittest

import numpy as np

from bearloc import model, run_analysis
from bearloc.samples import sample_description, sample_names, sample_scenario


class TestSamples(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = {name: run_analysis(sample_scenario(name)) for name in sample_names()}

    def test_all_samples_build_and_have_descriptions(self):
        self.assertGreaterEqual(len(sample_names()), 5)
        for name in sample_names():
            self.assertTrue(sample_description(name))
            self.assertEqual(sample_scenario(name).seed, 1)

    def test_well_posed_samples_pass_both_tests(self):
        for name in ("Close target, low noise", "Zig-zag, two maneuver legs",
                     "Target due west (bearing crosses 180 deg)"):
            res = self.results[name]
            self.assertTrue(res.truth.observability.invertible, name)
            self.assertTrue(res.mle.converged, name)
            self.assertTrue(res.fit.passed, name)
            self.assertTrue(res.nees.passed, name)

    def test_no_maneuver_is_unobservable(self):
        self.assertFalse(self.results["No maneuver (unobservable)"].truth.observability.invertible)

    def test_west_sample_crosses_plus_minus_pi(self):
        s = sample_scenario("Target due west (bearing crosses 180 deg)")
        h = model.bearings(s.x_true, s.times, s.platform_track)
        self.assertTrue(np.any(h > 3.0) and np.any(h < -3.0))

    def test_hard_sample_is_much_less_accurate(self):
        easy = self.results["Close target, low noise"].truth.P
        hard = self.results["High noise, weak maneuver"].truth.P
        self.assertGreater(np.sqrt(hard[1, 1]), 10 * np.sqrt(easy[1, 1]))

    def test_unknown_sample(self):
        with self.assertRaises(KeyError):
            sample_scenario("no such sample")


if __name__ == "__main__":
    unittest.main()
