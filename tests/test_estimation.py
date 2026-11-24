"""Tests for bearloc.estimation: the maximum likelihood estimator (B5)."""

import unittest

import numpy as np

from bearloc import estimation, model, stats
from bearloc.model import Maneuver
from tests.helpers import observable_scenario


def noise_free(s):
    """Exact bearings for scenario ``s``."""
    return model.bearings(s.x_true, s.times, s.platform_track)


class TestCost(unittest.TestCase):
    def test_zero_at_truth_without_noise(self):
        s = observable_scenario()
        self.assertAlmostEqual(estimation.cost(s.x_true, noise_free(s), s.times, s.platform_track, s.r), 0.0)

    def test_residuals_are_wrapped(self):
        # measured just below +pi, predicted just above -pi: true residual is small
        t, plat = np.array([1.0]), np.zeros((1, 2))
        x = np.array([-1.0, -1e-3, 0.0, 0.0])            # bearing slightly below -pi + 0
        z = np.array([np.pi - 1e-3])
        e = estimation.residuals(x, z, t, plat)
        self.assertLess(abs(e[0]), 0.01)

    def test_analytic_gradient_matches_finite_differences(self):
        s = observable_scenario()
        z = s.simulate_measurements(np.random.default_rng(0))
        x = s.x_true + np.array([100.0, -200.0, 0.3, -0.2])
        e = estimation.residuals(x, z, s.times, s.platform_track)
        H = model.bearing_jacobian(x, s.times, s.platform_track)
        grad = -2.0 * H.T @ e / s.r
        for j, h in enumerate([1e-2, 1e-2, 1e-5, 1e-5]):
            dx = np.zeros(4)
            dx[j] = h
            fd = (estimation.cost(x + dx, z, s.times, s.platform_track, s.r)
                  - estimation.cost(x - dx, z, s.times, s.platform_track, s.r)) / (2 * h)
            self.assertAlmostEqual(grad[j], fd, delta=1e-5 * max(1.0, abs(fd)))


class TestGaussNewton(unittest.TestCase):
    def test_recovers_truth_from_noise_free_data(self):
        s = observable_scenario()
        res = estimation.gauss_newton(noise_free(s), s.times, s.platform_track, s.r, s.x_init)
        self.assertTrue(res.converged)
        np.testing.assert_allclose(res.x, s.x_true, rtol=1e-6, atol=1e-6)
        self.assertLess(res.cost, 1e-8)
        self.assertLess(res.iterations, 20)

    def test_converges_from_distant_initial_estimates(self):
        s = observable_scenario()
        z = s.simulate_measurements(np.random.default_rng(1))
        ref = estimation.gauss_newton(z, s.times, s.platform_track, s.r, s.x_true)
        for x0 in ([10000, 20000, 0, 0], [1000, 3000, 0, 0], [-5000, 5000, 5, 5]):
            res = estimation.gauss_newton(z, s.times, s.platform_track, s.r, x0)
            self.assertTrue(res.converged, msg=f"from {x0}: {res.message}")
            np.testing.assert_allclose(res.x, ref.x, rtol=1e-6, atol=1e-6)

    def test_history_starts_at_initial_estimate(self):
        s = observable_scenario()
        res = estimation.gauss_newton(noise_free(s), s.times, s.platform_track, s.r, s.x_init)
        np.testing.assert_array_equal(res.history[0], s.x_init)
        np.testing.assert_array_equal(res.history[-1], res.x)

    def test_cost_never_increases(self):
        s = observable_scenario()
        z = s.simulate_measurements(np.random.default_rng(2))
        res = estimation.gauss_newton(z, s.times, s.platform_track, s.r, s.x_init)
        costs = [estimation.cost(x, z, s.times, s.platform_track, s.r) for x in res.history]
        self.assertTrue(np.all(np.diff(costs) <= 1e-12))

    def test_bearings_crossing_plus_minus_pi(self):
        # Target west of the platform, crossing the -xi axis: bearings jump
        # from +pi to -pi.  Without residual wrapping the fit would fail.
        s = observable_scenario(
            x_true=[-12000.0, 3000.0, 0.0, -8.0],
            platform_vel=[0.0, 5.0],
            maneuvers=[Maneuver(-10.0, 0.0, 20, 40)],
            x_init=[-9000.0, 0.0, 0.0, 0.0],
        )
        h = noise_free(s)
        self.assertTrue(np.any(h > 3.0) and np.any(h < -3.0))  # really crosses +-pi
        z = s.simulate_measurements(np.random.default_rng(5))
        res = estimation.gauss_newton(z, s.times, s.platform_track, s.r, s.x_init)
        self.assertTrue(res.converged)
        self.assertTrue(stats.chi2_test(res.cost, s.n - 4, 0.999).passed)
        J = model.fisher_information(res.x, s.times, s.platform_track, s.r)
        self.assertLess(stats.nees(s.x_true, res.x, J), 25.0)


class TestBfgs(unittest.TestCase):
    def test_agrees_with_gauss_newton(self):
        s = observable_scenario()
        z = s.simulate_measurements(np.random.default_rng(4))
        gn = estimation.gauss_newton(z, s.times, s.platform_track, s.r, s.x_init)
        bf = estimation.bfgs(z, s.times, s.platform_track, s.r, s.x_init)
        self.assertEqual(bf.method, "bfgs")
        # agreement to a tiny fraction of the estimation standard deviation
        std = np.sqrt(np.diag(np.linalg.inv(model.fisher_information(gn.x, s.times, s.platform_track, s.r))))
        np.testing.assert_array_less(np.abs(gn.x - bf.x), 1e-3 * std)
        self.assertAlmostEqual(gn.cost, bf.cost, places=6)


class TestRangeDivergence(unittest.TestCase):
    """No finite MLE: the range runs off to infinity and must be reported, not 'converged'."""

    def setUp(self):
        # Bearings of the relative trajectory *without* the platform maneuver: the
        # infinite-range limit reproduces them exactly, so the cost keeps decreasing
        # towards infinite range and the likelihood has no finite maximum.
        self.s = observable_scenario()
        s = self.s
        d = (s.x_true[:2] - s.platform_pos) + np.outer(s.times, s.x_true[2:] - s.platform_vel)
        self.z = np.arctan2(d[:, 1], d[:, 0])

    def test_both_methods_report_divergence(self):
        s = self.s
        bound = estimation.DIVERGENCE_FACTOR * estimation.length_scale(s.x_init, s.times, s.platform_track)
        for method in ("gauss-newton", "bfgs"):
            res = estimation.maximum_likelihood(self.z, s.times, s.platform_track, s.r, s.x_init, method=method)
            with self.subTest(method=method):
                self.assertTrue(res.diverged)
                self.assertFalse(res.converged)
                self.assertIn("diverged", res.message)
                self.assertGreater(estimation.max_range(res.x, s.times, s.platform_track), bound)

    def test_cost_decreases_towards_infinite_range(self):
        s = self.s
        rel0 = s.x_true[:2] - s.platform_pos
        relv = s.x_true[2:] - s.platform_vel
        costs = []
        for alpha in (1, 10, 100, 1000):
            x = np.concatenate([s.platform_pos + alpha * rel0, s.platform_vel + alpha * relv])
            costs.append(estimation.cost(x, self.z, s.times, s.platform_track, s.r))
        self.assertTrue(np.all(np.diff(costs) < 0), costs)

    def test_normal_data_do_not_diverge(self):
        s = self.s
        res = estimation.gauss_newton(s.simulate_measurements(np.random.default_rng(3)),
                                      s.times, s.platform_track, s.r, s.x_init)
        self.assertFalse(res.diverged)
        self.assertTrue(res.converged)


class TestBfgsConvergenceFlag(unittest.TestCase):
    """SciPy often reports 'precision loss' at the optimum; BEAR must still say converged."""

    def test_flag_is_reliable_over_many_draws(self):
        s = observable_scenario()
        rng = np.random.default_rng(31)
        precision_loss = 0
        for _ in range(40):
            z = s.simulate_measurements(rng)
            gn = estimation.gauss_newton(z, s.times, s.platform_track, s.r, s.x_init)
            bf = estimation.bfgs(z, s.times, s.platform_track, s.r, s.x_init)
            self.assertTrue(bf.converged, bf.message)
            precision_loss += "precision loss" in bf.message
            std = np.sqrt(np.diag(np.linalg.inv(
                model.fisher_information(gn.x, s.times, s.platform_track, s.r))))
            np.testing.assert_array_less(np.abs(bf.x - gn.x), 1e-4 * std)  # same optimum
        self.assertGreater(precision_loss, 0)  # the situation really occurs

    def test_decrement_rejects_non_optimal_points(self):
        s = observable_scenario()
        z = s.simulate_measurements(np.random.default_rng(1))
        self.assertGreater(estimation.gauss_newton_decrement(s.x_init, z, s.times, s.platform_track, s.r),
                           1e3 * estimation.BFGS_DECREMENT_TOL)


class TestDispatcher(unittest.TestCase):
    def test_methods_and_unknown_method(self):
        s = observable_scenario()
        z = noise_free(s)
        for m in ("gauss-newton", "bfgs"):
            res = estimation.maximum_likelihood(z, s.times, s.platform_track, s.r, s.x_init, method=m)
            self.assertEqual(res.method, m)
        with self.assertRaises(ValueError):
            estimation.maximum_likelihood(z, s.times, s.platform_track, s.r, s.x_init, method="newton")


if __name__ == "__main__":
    unittest.main()
