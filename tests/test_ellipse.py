"""Tests for bearloc.ellipse: the g-sigma ellipse algorithm used by DBEAR."""

import unittest

import numpy as np

from bearloc import ellipse


def random_spd(rng, scale=1.0):
    """Random 2x2 symmetric positive definite matrix."""
    L = rng.normal(size=(2, 2)) * scale
    return L @ L.T + 1e-3 * scale**2 * np.eye(2)


class TestEllipseAxes(unittest.TestCase):
    def test_eigenvalues_match_numpy(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            P = random_spd(rng, scale=10 ** rng.uniform(-3, 3))
            ax = ellipse.ellipse_axes(P)
            lam = np.linalg.eigvalsh(P)  # ascending
            np.testing.assert_allclose([ax.lam2, ax.lam1], lam, rtol=1e-9)

    def test_theta_is_major_eigenvector_direction(self):
        rng = np.random.default_rng(1)
        for _ in range(200):
            P = random_spd(rng)
            ax = ellipse.ellipse_axes(P)
            u = np.array([np.cos(ax.theta), np.sin(ax.theta)])
            np.testing.assert_allclose(P @ u, ax.lam1 * u, rtol=1e-8, atol=1e-10)
            self.assertTrue(-np.pi / 2 <= ax.theta <= np.pi / 2)

    def test_axis_aligned_and_circular_cases(self):
        self.assertAlmostEqual(ellipse.ellipse_axes(np.diag([4.0, 1.0])).theta, 0.0)
        self.assertAlmostEqual(ellipse.ellipse_axes(np.diag([1.0, 4.0])).theta, np.pi / 2)
        ax = ellipse.ellipse_axes(np.eye(2))
        self.assertAlmostEqual(ax.lam1, 1.0)
        self.assertAlmostEqual(ax.lam2, 1.0)

    def test_forty_five_degrees(self):
        ax = ellipse.ellipse_axes(np.array([[2.0, 1.0], [1.0, 2.0]]))
        self.assertAlmostEqual(ax.theta, np.pi / 4)
        self.assertAlmostEqual(ax.lam1, 3.0)
        self.assertAlmostEqual(ax.lam2, 1.0)

    def test_very_thin_ellipse_keeps_small_eigenvalue_accurate(self):
        # Highly correlated covariance like those in bearings-only problems
        P = np.array([[1e6, 1e6 - 1e-2], [1e6 - 1e-2, 1e6]])
        ax = ellipse.ellipse_axes(P)
        self.assertAlmostEqual(ax.lam2 / 1e-2, 1.0, places=5)

    def test_invalid_matrices_rejected(self):
        for bad in (np.eye(3), np.array([[1.0, 2.0], [2.0, 1.0]]),   # wrong shape, indefinite
                    np.array([[1.0, 0.5], [0.0, 1.0]]),              # not symmetric
                    np.array([[-1.0, 0.0], [0.0, -1.0]]),            # negative definite
                    np.array([[np.nan, 0.0], [0.0, 1.0]])):
            with self.assertRaises(ValueError):
                ellipse.ellipse_axes(bad)


class TestSigmaEllipse(unittest.TestCase):
    def test_points_satisfy_quadratic_form(self):
        rng = np.random.default_rng(2)
        for _ in range(50):
            P = random_spd(rng, scale=100.0)
            g = rng.uniform(0.5, 4.0)
            c = rng.normal(size=2) * 1000
            pts = ellipse.sigma_ellipse(P, g, c, num_points=73)
            d = pts - c
            q = np.einsum("ij,jk,ik->i", d, np.linalg.inv(P), d)
            np.testing.assert_allclose(q, g**2, rtol=1e-8)

    def test_axis_aligned_extent(self):
        pts = ellipse.sigma_ellipse(np.diag([4.0, 1.0]), g=3.0, num_points=401)
        self.assertAlmostEqual(pts[:, 0].max(), 6.0, places=6)   # 3 * sqrt(4)
        self.assertAlmostEqual(pts[:, 1].max(), 3.0, places=3)   # 3 * sqrt(1)

    def test_curve_is_closed_and_has_requested_size(self):
        pts = ellipse.sigma_ellipse(np.eye(2), 1.0, (5.0, -2.0), num_points=50)
        self.assertEqual(pts.shape, (50, 2))
        np.testing.assert_allclose(pts[0], pts[-1], atol=1e-12)
        np.testing.assert_allclose(pts.mean(axis=0), [5.0, -2.0], atol=0.05)

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            ellipse.sigma_ellipse(np.eye(2), g=0.0)
        with self.assertRaises(ValueError):
            ellipse.sigma_ellipse(np.eye(2), num_points=2)


class TestProbability(unittest.TestCase):
    def test_known_values(self):
        self.assertAlmostEqual(ellipse.ellipse_probability(1.0), 0.3935, places=4)
        self.assertAlmostEqual(ellipse.ellipse_probability(2.0), 0.8647, places=4)
        self.assertAlmostEqual(ellipse.ellipse_probability(3.0), 0.9889, places=4)

    def test_inverse(self):
        for g in (0.5, 1.0, 2.0, 3.5):
            self.assertAlmostEqual(ellipse.g_for_probability(ellipse.ellipse_probability(g)), g)
        with self.assertRaises(ValueError):
            ellipse.g_for_probability(1.0)

    def test_empirical_coverage(self):
        # fraction of N(0, P) samples inside the 2-sigma ellipse ~ 86.5 %
        rng = np.random.default_rng(3)
        P = np.array([[9.0, 4.0], [4.0, 3.0]])
        x = rng.multivariate_normal(np.zeros(2), P, size=200000)
        q = np.einsum("ij,jk,ik->i", x, np.linalg.inv(P), x)
        self.assertAlmostEqual(np.mean(q <= 4.0), ellipse.ellipse_probability(2.0), delta=0.004)


if __name__ == "__main__":
    unittest.main()
