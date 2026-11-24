"""Checks of the estimation theory behind BEAR/DBEAR (Monte Carlo against formulas).

Each test verifies one theoretical statement made in the programs and docs:

* the FIM equals the covariance of the score (definition of Fisher information);
* the fitting error is chi2(n - 4): mean n - 4 *and* variance 2 (n - 4);
* NEES is distributed as chi2(4) (Kolmogorov-Smirnov test);
* a g-sigma position ellipse contains the truth with probability 1 - exp(-g^2/2);
* the ellipses are elongated along the line of sight;
* the CRLB standard deviations scale with sigma, and with 1/sqrt(n) for denser
  sampling of the same time window;
* without a maneuver: rank 3, and the fitting error is chi2(n - 3).

All random numbers are seeded, so the tests are deterministic.  Tolerances are
several Monte Carlo standard errors wide.
"""

import unittest

import numpy as np
from scipy import stats as sps

from bearloc import analysis, estimation, model
from bearloc.ellipse import ellipse_axes, ellipse_probability
from bearloc.model import Maneuver
from tests.helpers import observable_scenario, unobservable_scenario


class TestFisherInformationDefinition(unittest.TestCase):
    def test_fim_equals_score_covariance(self):
        # score = d/dx ln p(z|x) = H'(z - h)/r at the truth; its covariance is J
        s = observable_scenario()
        rng = np.random.default_rng(11)
        H = model.bearing_jacobian(s.x_true, s.times, s.platform_track)
        w = rng.normal(0.0, np.sqrt(s.r), size=(20000, s.n))
        scores = w @ H / s.r
        S = np.cov(scores.T)
        J = model.fisher_information(s.x_true, s.times, s.platform_track, s.r)
        d = np.sqrt(np.diag(J))
        self.assertLess(np.abs((S - J) / np.outer(d, d)).max(), 0.04)


class TestMonteCarloAgainstTheory(unittest.TestCase):
    """1500 independent runs of the default scenario."""

    @classmethod
    def setUpClass(cls):
        cls.s = s = observable_scenario()
        rng = np.random.default_rng(2024)
        cls.t_ell = 300.0
        nees, fits, inside = [], [], []
        for _ in range(1500):
            z = s.simulate_measurements(rng)
            res = estimation.gauss_newton(z, s.times, s.platform_track, s.r, s.x_init)
            J_hat = model.fisher_information(res.x, s.times, s.platform_track, s.r)
            err = s.x_true - res.x
            nees.append(err @ J_hat @ err)
            fits.append(res.cost)
            C = model.position_covariances(np.linalg.inv(J_hat), cls.t_ell)
            dp = model.target_position(err, cls.t_ell)  # A(t) x_tilde = true - estimated position
            inside.append(dp @ np.linalg.solve(C, dp) <= 4.0)
        cls.nees, cls.fits, cls.inside = np.array(nees), np.array(fits), np.array(inside)

    def test_fitting_error_is_chi2_n_minus_4(self):
        dof = self.s.n - 4
        self.assertAlmostEqual(self.fits.mean() / dof, 1.0, delta=0.03)
        self.assertAlmostEqual(self.fits.var() / (2 * dof), 1.0, delta=0.15)

    def test_nees_is_chi2_4(self):
        self.assertAlmostEqual(self.nees.mean(), 4.0, delta=0.25)
        self.assertGreater(sps.kstest(self.nees, "chi2", args=(4,)).pvalue, 0.001)

    def test_two_sigma_ellipse_coverage(self):
        # binomial standard error sqrt(0.865 * 0.135 / 1500) ~ 0.009
        self.assertAlmostEqual(self.inside.mean(), ellipse_probability(2.0), delta=0.035)


class TestEllipseGeometry(unittest.TestCase):
    def test_major_axis_along_line_of_sight(self):
        s = observable_scenario()
        P = np.linalg.inv(model.fisher_information(s.x_true, s.times, s.platform_track, s.r))
        for t in (300.0, 450.0, 600.0):
            theta = np.degrees(ellipse_axes(model.position_covariances(P, t)).theta)
            d = model.target_position(s.x_true, t) - s.platform_at(t)
            los = np.degrees(np.arctan2(d[1], d[0]))
            diff = (theta - los + 90.0) % 180.0 - 90.0   # axes are lines: compare mod 180
            self.assertLess(abs(diff), 10.0, f"t={t}")


class TestCrlbScaling(unittest.TestCase):
    def setUp(self):
        self.s = observable_scenario()
        J = model.fisher_information(self.s.x_true, self.s.times, self.s.platform_track, self.s.r)
        self.std = np.sqrt(np.diag(np.linalg.inv(J)))

    def test_std_proportional_to_sigma(self):
        s = self.s
        for k in (0.5, 3.0):
            J = model.fisher_information(s.x_true, s.times, s.platform_track, s.r * k**2)
            np.testing.assert_allclose(np.sqrt(np.diag(np.linalg.inv(J))) / self.std, k, rtol=1e-9)

    def test_denser_sampling_of_same_window(self):
        s = self.s
        for f in (2, 4):
            d = s.copy(T=s.T / f, n=s.n * f,
                       maneuvers=[Maneuver(m.vx, m.vy, m.k1 * f, m.k2 * f) for m in s.maneuvers])
            J = model.fisher_information(s.x_true, d.times, d.platform_track, s.r)
            ratio = np.sqrt(np.diag(np.linalg.inv(J))) / self.std
            np.testing.assert_allclose(ratio, 1 / np.sqrt(f), rtol=0.05)


class TestUnobservableTheory(unittest.TestCase):
    def test_fitting_error_is_chi2_n_minus_3(self):
        s = unobservable_scenario()
        mc = analysis.monte_carlo(s, 400, np.random.default_rng(9))
        fits = mc.fit_errors[mc.converged]
        self.assertAlmostEqual(fits.mean() / (s.n - 3), 1.0, delta=0.04)

    def test_scaled_relative_trajectory_gives_identical_bearings(self):
        s = unobservable_scenario()
        rel0, relv = s.x_true[:2] - s.platform_pos, s.x_true[2:] - s.platform_vel
        x2 = np.concatenate([s.platform_pos + 2 * rel0, s.platform_vel + 2 * relv])
        np.testing.assert_allclose(model.bearings(x2, s.times, s.platform_track),
                                   model.bearings(s.x_true, s.times, s.platform_track), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
