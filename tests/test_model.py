"""Tests for bearloc.model: kinematics, bearings, Jacobian, FIM (B1, B3)."""

import unittest

import numpy as np

from bearloc import model
from bearloc.model import Maneuver
from tests.helpers import R_1DEG, observable_scenario, unobservable_scenario


class TestMeasurementTimes(unittest.TestCase):
    def test_times_are_k_times_T_for_k_1_to_n(self):
        np.testing.assert_allclose(model.measurement_times(10.0, 3), [10.0, 20.0, 30.0])

    def test_invalid_arguments_raise(self):
        with self.assertRaises(ValueError):
            model.measurement_times(0.0, 3)
        with self.assertRaises(ValueError):
            model.measurement_times(1.0, 0)


class TestManeuver(unittest.TestCase):
    def test_end_must_follow_start(self):
        with self.assertRaises(ValueError):
            Maneuver(1.0, 0.0, 5, 5)
        with self.assertRaises(ValueError):
            Maneuver(1.0, 0.0, -1, 5)

    def test_overlapping_legs_rejected(self):
        legs = [Maneuver(1, 0, 10, 20), Maneuver(0, 1, 15, 30)]
        with self.assertRaises(ValueError):
            model.validate_maneuvers(legs)

    def test_adjacent_legs_allowed(self):
        model.validate_maneuvers([Maneuver(1, 0, 10, 20), Maneuver(0, 1, 20, 30)])


class TestPlatformMotion(unittest.TestCase):
    p0, v0, T = np.array([0.0, 0.0]), np.array([0.0, 15.0]), 10.0
    legs = [Maneuver(15.0, 0.0, 20, 40)]  # east during 200 s <= t < 400 s

    def pos(self, t):
        return model.platform_position(t, self.p0, self.v0, self.legs, self.T)

    def test_no_maneuver_is_straight_line(self):
        t = np.array([0.0, 7.0, 123.0])
        p = model.platform_position(t, [1.0, 2.0], [3.0, -4.0], [], 1.0)
        np.testing.assert_allclose(p, [[1, 2], [22, -26], [370, -490]])

    def test_positions_before_during_after_leg(self):
        np.testing.assert_allclose(self.pos(100.0), [0.0, 1500.0])   # before: north
        np.testing.assert_allclose(self.pos(300.0), [1500.0, 3000.0])  # during: east
        np.testing.assert_allclose(self.pos(500.0), [3000.0, 4500.0])  # after: north again

    def test_position_is_continuous_at_leg_boundaries(self):
        for tb in (200.0, 400.0):
            np.testing.assert_allclose(self.pos(tb - 1e-9), self.pos(tb + 1e-9), atol=1e-6)

    def test_velocity_matches_legs(self):
        v = model.platform_velocity([100.0, 200.0, 399.0, 400.0], self.v0, self.legs, self.T)
        np.testing.assert_allclose(v, [[0, 15], [15, 0], [15, 0], [0, 15]])

    def test_position_derivative_equals_velocity(self):
        t = np.array([50.0, 250.0, 450.0])
        h = 1e-3
        dp = (self.pos(t + h) - self.pos(t - h)) / (2 * h)
        np.testing.assert_allclose(dp, model.platform_velocity(t, self.v0, self.legs, self.T), atol=1e-6)

    def test_scalar_time_returns_vector(self):
        self.assertEqual(self.pos(5.0).shape, (2,))
        self.assertEqual(self.pos(np.array([5.0])).shape, (1, 2))


class TestBearings(unittest.TestCase):
    def test_known_geometry(self):
        # target fixed at (1, 1), (-1, 0), (0, -2); platform at the origin
        t = np.array([1.0])
        plat = np.zeros((1, 2))
        self.assertAlmostEqual(model.bearings([1, 1, 0, 0], t, plat)[0], np.pi / 4)
        self.assertAlmostEqual(model.bearings([-1, 0, 0, 0], t, plat)[0], np.pi)
        self.assertAlmostEqual(model.bearings([0, -2, 0, 0], t, plat)[0], -np.pi / 2)

    def test_bearing_uses_target_motion(self):
        # target starts at (0, 1) moving +xi at 1/s: at t = 1 it is at (1, 1)
        h = model.bearings([0, 1, 1, 0], np.array([1.0]), np.zeros((1, 2)))
        self.assertAlmostEqual(h[0], np.pi / 4)

    def test_wrap_angle(self):
        self.assertAlmostEqual(model.wrap_angle(3 * np.pi / 2), -np.pi / 2)
        self.assertAlmostEqual(model.wrap_angle(-np.pi), np.pi)
        self.assertAlmostEqual(model.wrap_angle(np.pi), np.pi)
        self.assertAlmostEqual(model.wrap_angle(0.1 + 4 * np.pi), 0.1)
        w = model.wrap_angle(np.linspace(-20, 20, 101))
        self.assertTrue(np.all((w > -np.pi) & (w <= np.pi)))


class TestJacobian(unittest.TestCase):
    def test_matches_central_finite_differences(self):
        s = observable_scenario()
        t, plat, x = s.times, s.platform_track, s.x_true
        H = model.bearing_jacobian(x, t, plat)
        steps = np.array([1e-2, 1e-2, 1e-5, 1e-5])
        for j in range(4):
            dx = np.zeros(4)
            dx[j] = steps[j]
            fd = (model.bearings(x + dx, t, plat) - model.bearings(x - dx, t, plat)) / (2 * steps[j])
            np.testing.assert_allclose(H[:, j], fd, rtol=1e-6, atol=1e-12)

    def test_target_on_platform_raises(self):
        with self.assertRaises(ValueError):
            model.bearing_jacobian([0, 0, 0, 0], np.array([1.0]), np.zeros((1, 2)))


class TestFisherInformation(unittest.TestCase):
    def test_equals_HtH_over_r_and_is_symmetric_psd(self):
        s = observable_scenario()
        J = model.fisher_information(s.x_true, s.times, s.platform_track, s.r)
        H = model.bearing_jacobian(s.x_true, s.times, s.platform_track)
        np.testing.assert_allclose(J, H.T @ H / s.r)
        np.testing.assert_array_equal(J, J.T)
        self.assertGreater(np.linalg.eigvalsh(J).min(), 0.0)

    def test_scales_inversely_with_noise_variance(self):
        s = observable_scenario()
        J1 = model.fisher_information(s.x_true, s.times, s.platform_track, R_1DEG)
        J4 = model.fisher_information(s.x_true, s.times, s.platform_track, 4 * R_1DEG)
        np.testing.assert_allclose(J1, 4 * J4)

    def test_invalid_variance_raises(self):
        s = observable_scenario()
        with self.assertRaises(ValueError):
            model.fisher_information(s.x_true, s.times, s.platform_track, 0.0)

    def test_singular_without_platform_maneuver(self):
        # Both target and platform at constant velocity: scaling the relative
        # state [target0 - p0, v_target - v_platform] leaves every bearing
        # unchanged, so that direction is in the null space of J.
        s = unobservable_scenario()
        J = model.fisher_information(s.x_true, s.times, s.platform_track, s.r)
        null = np.concatenate([s.x_true[:2] - s.platform_pos, s.x_true[2:] - s.platform_vel])
        Jn = J @ null
        self.assertLess(np.linalg.norm(Jn), 1e-10 * np.linalg.norm(J) * np.linalg.norm(null))
        J_scaled = J / np.sqrt(np.outer(np.diag(J), np.diag(J)))
        self.assertEqual(np.linalg.matrix_rank(J_scaled, tol=1e-10), 3)


class TestPositionCovariance(unittest.TestCase):
    def test_position_matrix_reproduces_target_position(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        for t in (0.0, 2.5, 10.0):
            np.testing.assert_allclose(model.position_matrix(t) @ x, model.target_position(x, t))

    def test_covariance_is_APA(self):
        rng = np.random.default_rng(0)
        L = rng.normal(size=(4, 4))
        P = L @ L.T + np.eye(4)
        covs = model.position_covariances(P, np.array([0.0, 3.0]))
        np.testing.assert_allclose(covs[0], P[:2, :2])
        A = model.position_matrix(3.0)
        np.testing.assert_allclose(covs[1], A @ P @ A.T)
        self.assertEqual(model.position_covariances(P, 3.0).shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
