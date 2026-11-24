"""Tests for bearloc.scenario: inputs A1-A7 and the BEAR.DEF file."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from bearloc import model
from bearloc.model import Maneuver
from bearloc.scenario import Scenario, load_defaults, parse_vector, save_defaults
from tests.helpers import DEFAULT_DEF, observable_scenario


class TestParseVector(unittest.TestCase):
    def test_separators_and_brackets(self):
        for text in ("1, 2, 3", "1 2 3", "[1,2,3]", " (1 ,2,  3) ", "1,2 3"):
            np.testing.assert_array_equal(parse_vector(text), [1, 2, 3])

    def test_scientific_and_negative(self):
        np.testing.assert_array_equal(parse_vector("-1e3, 2.5E-2"), [-1000.0, 0.025])

    def test_errors(self):
        with self.assertRaises(ValueError):
            parse_vector("1, two")
        with self.assertRaises(ValueError):
            parse_vector("1, 2", length=4)


class TestBearDef(unittest.TestCase):
    def test_project_defaults_file(self):
        s = load_defaults(DEFAULT_DEF)
        np.testing.assert_array_equal(s.x_true, [5000, 12000, -6, -2])
        self.assertEqual((s.T, s.n), (10.0, 60))
        self.assertAlmostEqual(np.degrees(np.sqrt(s.r)), 1.0, places=6)
        np.testing.assert_array_equal(s.platform_vel, [0, 15])
        self.assertEqual(s.maneuvers, [Maneuver(15.0, 0.0, 20, 40)])
        self.assertEqual((s.method, s.seed, s.confidence), ("gauss-newton", 1, 0.95))

    def test_round_trip(self):
        s = observable_scenario(
            maneuvers=[Maneuver(1.5, -2.0, 5, 10), Maneuver(0.0, 3.0, 30, 45)],
            seed=None, method="bfgs", confidence=0.99,
        )
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "test.def"
            save_defaults(s, path)
            self.assertEqual(load_defaults(path).to_dict(), s.to_dict())

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_defaults("does/not/exist.def")

    def test_missing_entry_and_bad_value(self):
        text = DEFAULT_DEF.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "bad.def"
            path.write_text(text.replace("x_true =", "x_tru ="), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "x_true"):
                load_defaults(path)
            path.write_text(text.replace("n = 60", "n = sixty"), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_defaults(path)

    def test_no_maneuver_sections(self):
        s = observable_scenario(maneuvers=[])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "nm.def"
            save_defaults(s, path)
            self.assertEqual(load_defaults(path).maneuvers, [])


class TestScenario(unittest.TestCase):
    def test_validation(self):
        bad_changes = [
            {"T": 0.0}, {"n": 0}, {"r": -1.0}, {"x_true": [1, 2, 3]}, {"x_init": [1, 2]},
            {"platform_pos": [0, 0, 0]}, {"method": "simplex"}, {"confidence": 1.5},
            {"maneuvers": [Maneuver(1, 0, 10, 20), Maneuver(0, 1, 15, 25)]},
        ]
        for change in bad_changes:
            with self.assertRaises(ValueError, msg=str(change)):
                observable_scenario(**change)

    def test_dict_round_trip(self):
        s = observable_scenario()
        self.assertEqual(Scenario.from_dict(s.to_dict()).to_dict(), s.to_dict())

    def test_platform_track_matches_times(self):
        s = observable_scenario()
        self.assertEqual(s.platform_track.shape, (s.n, 2))
        np.testing.assert_allclose(s.platform_track[0], [0.0, 150.0])

    def test_simulated_measurements(self):
        s = observable_scenario(n=2000, T=0.3)
        z1 = s.simulate_measurements(np.random.default_rng(7))
        z2 = s.simulate_measurements(np.random.default_rng(7))
        np.testing.assert_array_equal(z1, z2)  # reproducible
        noise = model.wrap_angle(z1 - model.bearings(s.x_true, s.times, s.platform_track))
        self.assertAlmostEqual(noise.mean() / np.sqrt(s.r), 0.0, delta=0.1)
        self.assertAlmostEqual(noise.var() / s.r, 1.0, delta=0.1)


if __name__ == "__main__":
    unittest.main()
