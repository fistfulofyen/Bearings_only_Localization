"""Tests for bearloc.analysis: the B1-B8 pipeline and Monte Carlo consistency."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bearloc import analysis, estimation, model, report
from bearloc.model import Maneuver
from tests.helpers import observable_scenario, unobservable_scenario


class TestRunAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.s = observable_scenario()
        cls.res = analysis.run_analysis(cls.s)

    def test_b1_to_b3_at_truth(self):
        t = self.res.truth
        self.assertTrue(t.observability.invertible)
        np.testing.assert_allclose(t.P @ t.J, np.eye(4), atol=1e-8)
        self.assertEqual(t.position_covs.shape, (self.s.n, 2, 2))

    def test_b4_measurements(self):
        self.assertEqual(self.res.z.shape, (self.s.n,))

    def test_b5_and_b6(self):
        self.assertTrue(self.res.mle.converged)
        np.testing.assert_array_equal(self.res.estimate.x, self.res.mle.x)
        self.assertTrue(self.res.estimate.observability.invertible)

    def test_b7_and_b8(self):
        self.assertEqual(self.res.fit.dof, self.s.n - 4)
        self.assertAlmostEqual(self.res.fit.value, self.res.mle.cost)
        self.assertEqual(self.res.nees.dof, 4)
        self.assertGreaterEqual(self.res.nees.value, 0.0)

    def test_same_seed_same_result(self):
        again = analysis.run_analysis(self.s)
        np.testing.assert_array_equal(again.z, self.res.z)
        np.testing.assert_array_equal(again.mle.x, self.res.mle.x)

    def test_given_measurements_are_used(self):
        z = model.bearings(self.s.x_true, self.s.times, self.s.platform_track)
        res = analysis.run_analysis(self.s, z=z)
        np.testing.assert_allclose(res.mle.x, self.s.x_true, rtol=1e-6)
        self.assertLess(res.nees.value, 1e-6)

    def test_report_contains_every_step(self):
        text = report.format_report(self.res, every=10)
        for tag in ("INPUTS", "B1-B3", "B4.", "B5.", "B6.", "B7.", "B8.", "OBSERVABLE"):
            self.assertIn(tag, text)

    def test_json_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "res.json"
            analysis.save_results(self.res, path)
            data = analysis.load_results(path)
        np.testing.assert_allclose(data["mle"]["x"], self.res.mle.x)
        np.testing.assert_allclose(data["estimate"]["P"], self.res.estimate.P)
        self.assertEqual(len(data["measurements"]), self.s.n)


class TestUnobservable(unittest.TestCase):
    def test_singular_fim_handled(self):
        res = analysis.run_analysis(unobservable_scenario())
        self.assertFalse(res.truth.observability.invertible)
        self.assertIsNone(res.truth.P)
        self.assertIsNone(res.truth.position_covs)
        text = report.format_report(res)
        self.assertIn("NOT OBSERVABLE", text)
        self.assertIn("not unique", text)
        with tempfile.TemporaryDirectory() as d:  # JSON must cope with inf / None
            path = Path(d) / "res.json"
            analysis.save_results(res, path)
            self.assertIsNone(json.loads(path.read_text())["truth"]["P"])

    def test_too_few_samples_for_fit_test(self):
        # 4 bearings, observable (maneuver inside the window): rank 4, no dof left
        res = analysis.run_analysis(observable_scenario(n=4, maneuvers=[Maneuver(15, 0, 1, 3)]))
        self.assertEqual(res.estimate.observability.rank, 4)
        self.assertIsNone(res.fit)
        self.assertIn("no degrees of freedom", report.format_fit(res))

    def test_unobservable_statistics(self):
        # only 3 combinations are estimable: fit has n - 3 dof; NEES not meaningful
        s = unobservable_scenario()
        res = analysis.run_analysis(s)
        self.assertEqual(res.fit.dof, s.n - 3)
        self.assertIn(f"n - rank(J) = {s.n} - 3", report.format_fit(res))
        self.assertFalse(res.nees_valid)
        self.assertIn("NOT MEANINGFUL", report.format_nees(res))
        mc = analysis.monte_carlo(s, 20, np.random.default_rng(0))
        self.assertIsNone(mc.avg_nees)
        self.assertIn("NOT MEANINGFUL", report.format_monte_carlo(mc))


class TestStatisticalConsistency(unittest.TestCase):
    """The MLE should be (nearly) efficient: errors ~ N(0, J^-1)."""

    @classmethod
    def setUpClass(cls):
        cls.s = observable_scenario()
        cls.mc = analysis.monte_carlo(cls.s, 200, np.random.default_rng(123))

    def test_all_runs_converge(self):
        self.assertTrue(self.mc.converged.all())

    def test_average_nees_within_bounds(self):
        self.assertTrue(self.mc.avg_nees.passed, msg=str(self.mc.avg_nees))

    def test_average_fitting_error_close_to_dof(self):
        # mean of chi2(n-4) over 200 runs: std of the average = sqrt(2*56/200) ~ 0.75
        self.assertAlmostEqual(self.mc.fit_errors.mean(), self.s.n - 4, delta=3.0)

    def test_error_covariance_close_to_crlb(self):
        s = self.s
        rng = np.random.default_rng(321)
        errs = []
        for _ in range(300):
            z = s.simulate_measurements(rng)
            res = estimation.gauss_newton(z, s.times, s.platform_track, s.r, s.x_init)
            errs.append(s.x_true - res.x)
        sample_var = np.var(np.array(errs), axis=0)
        crlb_var = np.diag(analysis.analyze_fim(s.x_true, s).P)
        # 300 runs: sample variance accurate to roughly +-16 % (2 sigma)
        np.testing.assert_allclose(sample_var / crlb_var, 1.0, atol=0.25)

    def test_invalid_runs(self):
        with self.assertRaises(ValueError):
            analysis.monte_carlo(self.s, 0)


if __name__ == "__main__":
    unittest.main()
