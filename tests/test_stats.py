"""Tests for bearloc.stats: observability, FIM inverse, chi-square tools, NEES."""

import unittest

import numpy as np

from bearloc import model, stats
from tests.helpers import observable_scenario, unobservable_scenario


class TestObservability(unittest.TestCase):
    def test_identity_is_invertible(self):
        obs = stats.check_observability(np.eye(4))
        self.assertTrue(obs.invertible)
        self.assertEqual(obs.rank, 4)
        self.assertAlmostEqual(obs.condition, 1.0)

    def test_rank_deficient_matrix(self):
        v = np.array([1.0, 2.0, 3.0, 4.0])
        J = np.eye(4) - np.outer(v, v) / (v @ v)  # projector of rank 3
        obs = stats.check_observability(J)
        self.assertFalse(obs.invertible)
        self.assertEqual(obs.rank, 3)
        self.assertEqual(obs.condition, float("inf"))

    def test_zero_diagonal_is_unobservable(self):
        self.assertFalse(stats.check_observability(np.diag([1.0, 1.0, 1.0, 0.0])).invertible)

    def test_test_is_invariant_to_parameter_scaling(self):
        # Badly scaled but perfectly invertible: a plain cond() would be ~1e24.
        D = np.diag([1e-6, 1e-6, 1e6, 1e6])
        J = D @ np.array([[2, 1, 0, 0], [1, 2, 0, 0], [0, 0, 2, 1], [0, 0, 1, 2.0]]) @ D
        self.assertGreater(np.linalg.cond(J), 1e20)
        self.assertTrue(stats.check_observability(J).invertible)

    def test_bearings_scenarios(self):
        for scen, expected in ((observable_scenario(), True), (unobservable_scenario(), False)):
            J = model.fisher_information(scen.x_true, scen.times, scen.platform_track, scen.r)
            self.assertEqual(stats.check_observability(J).invertible, expected)


class TestInverse(unittest.TestCase):
    def test_matches_numpy_inverse(self):
        s = observable_scenario()
        J = model.fisher_information(s.x_true, s.times, s.platform_track, s.r)
        P = stats.invert_fim(J)
        np.testing.assert_allclose(P @ J, np.eye(4), atol=1e-8)
        np.testing.assert_array_equal(P, P.T)

    def test_singular_returns_none(self):
        s = unobservable_scenario()
        J = model.fisher_information(s.x_true, s.times, s.platform_track, s.r)
        self.assertIsNone(stats.invert_fim(J))


class TestChiSquare(unittest.TestCase):
    def test_textbook_values(self):
        lo, hi = stats.chi2_interval(4, 0.95)
        self.assertAlmostEqual(lo, 0.4844, places=3)
        self.assertAlmostEqual(hi, 11.1433, places=3)
        lo, hi = stats.chi2_interval(1, 0.95)
        self.assertAlmostEqual(hi, 5.0239, places=3)

    def test_interval_widens_with_confidence(self):
        a = stats.chi2_interval(10, 0.9)
        b = stats.chi2_interval(10, 0.99)
        self.assertLess(b[0], a[0])
        self.assertGreater(b[1], a[1])

    def test_invalid_dof(self):
        with self.assertRaises(ValueError):
            stats.chi2_interval(0)

    def test_chi2_test_verdict(self):
        self.assertTrue(stats.chi2_test(4.0, 4).passed)
        self.assertFalse(stats.chi2_test(0.1, 4).passed)
        self.assertFalse(stats.chi2_test(20.0, 4).passed)


class TestNees(unittest.TestCase):
    def test_quadratic_form(self):
        J = np.diag([4.0, 1.0, 1.0, 1.0])
        self.assertAlmostEqual(stats.nees([1, 0, 0, 0], [0, 0, 0, 0], J), 4.0)
        self.assertAlmostEqual(stats.nees([1, 1, 0, 0], [1, 1, 0, 0], J), 0.0)

    def test_nees_of_gaussian_errors_is_chi2_4(self):
        # errors drawn from N(0, J^-1) give eps ~ chi2(4): mean 4, variance 8
        rng = np.random.default_rng(3)
        s = observable_scenario()
        J = model.fisher_information(s.x_true, s.times, s.platform_track, s.r)
        P = np.linalg.inv(J)
        errs = rng.multivariate_normal(np.zeros(4), P, size=20000)
        eps = np.einsum("ij,jk,ik->i", errs, J, errs)
        self.assertAlmostEqual(eps.mean(), 4.0, delta=0.1)
        self.assertAlmostEqual(eps.var(), 8.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()
