"""End-to-end tests of the BEAR and DBEAR programs (bear.py, dbear.py)."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # never open windows during tests

import numpy as np

import bear
import dbear
from bearloc import analysis
from bearloc.scenario import load_defaults
from tests.helpers import DEFAULT_DEF, ROOT, ScriptedInput


def run_bear(argv, script=None):
    """Run bear.main; return (exit code, stdout text, stderr text)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stderr(err):
        code = bear.main(argv, input_fn=script or ScriptedInput(), out=out)
    return code, out.getvalue(), err.getvalue()


def run_dbear(argv, script=None):
    """Run dbear.main; return (exit code, stdout text, stderr text)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stderr(err):
        code = dbear.main(argv, input_fn=script or ScriptedInput(), out=out)
    return code, out.getvalue(), err.getvalue()


class TestBearParsers(unittest.TestCase):
    def test_variance_in_degrees_or_rad2(self):
        self.assertAlmostEqual(bear.parse_variance("1deg"), np.radians(1) ** 2)
        self.assertAlmostEqual(bear.parse_variance(" 0.5 DEG "), np.radians(0.5) ** 2)
        self.assertAlmostEqual(bear.parse_variance("3e-4"), 3e-4)
        for bad in ("-1", "0", "-2deg", "abc"):
            with self.assertRaises(ValueError):
                bear.parse_variance(bad)

    def test_seed_and_method(self):
        self.assertIsNone(bear.parse_seed("random"))
        self.assertEqual(bear.parse_seed("42"), 42)
        self.assertEqual(bear.parse_method("BFGS"), "bfgs")
        with self.assertRaises(ValueError):
            bear.parse_method("simplex")


class TestDefaultPaths(unittest.TestCase):
    """Default files live in the project folder, whatever the working directory."""

    def test_bear_defaults_next_to_script(self):
        args = bear.build_parser().parse_args([])
        self.assertEqual(args.defaults, ROOT / "BEAR.DEF")
        self.assertEqual(args.output, ROOT / "bear_results.json")

    def test_dbear_reads_what_bear_writes(self):
        self.assertEqual(dbear.build_parser().parse_args([]).results,
                         bear.build_parser().parse_args([]).output)


class TestBearProgram(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.out = self.dir / "res.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_batch_run_writes_results(self):
        code, text, _ = run_bear(["--batch", "--def", str(DEFAULT_DEF), "-o", str(self.out), "--every", "20"])
        self.assertEqual(code, 0)
        for tag in ("B1-B3", "B4.", "B5.", "B6.", "B7.", "B8.", "Results written"):
            self.assertIn(tag, text)
        data = analysis.load_results(self.out)
        self.assertTrue(data["truth"]["invertible"])
        self.assertTrue(data["mle"]["converged"])

    def test_interactive_accepts_defaults(self):
        script = ScriptedInput()
        code, _, _ = run_bear(["--def", str(DEFAULT_DEF), "-o", str(self.out)], script)
        self.assertEqual(code, 0)
        # every input A1-A7 was offered
        for item in ("A1", "A2", "A3", "A4", "A5", "A6", "A7"):
            self.assertTrue(any(p.startswith(item) for p in script.prompts), item)
        saved = analysis.load_results(self.out)["scenario"]
        self.assertEqual(saved, load_defaults(DEFAULT_DEF).to_dict())

    def test_interactive_changes_and_reprompts_on_bad_input(self):
        script = ScriptedInput({
            "A3": ["many", "0", "30"],          # two invalid answers, then 30
            "A4": ["0.5deg"],
            "A7": ["3000 10000 -5 0"],
            "    Minimization": ["bfgs"],
        })
        code, text, _ = run_bear(["--def", str(DEFAULT_DEF), "-o", str(self.out)], script)
        self.assertEqual(code, 0)
        self.assertEqual(text.count("invalid input"), 2)
        s = analysis.load_results(self.out)["scenario"]
        self.assertEqual(s["n"], 30)
        self.assertAlmostEqual(s["r"], np.radians(0.5) ** 2)
        self.assertEqual(s["x_init"], [3000, 10000, -5, 0])
        self.assertEqual(s["method"], "bfgs")

    def test_overlapping_legs_are_asked_again(self):
        script = ScriptedInput({
            "A6  Number": ["2", "2"],
            "A6  Leg 2: velocity": ["0 15", "0 15"],
            "A6  Leg 2: start": ["30", "45"],   # 30 overlaps leg 1 (20..40); 45 is fine
            "A6  Leg 2: end": ["50", "50"],
        })
        code, text, _ = run_bear(["--def", str(DEFAULT_DEF), "-o", str(self.out)], script)
        self.assertEqual(code, 0)
        self.assertIn("Inconsistent input", text)
        legs = analysis.load_results(self.out)["scenario"]["maneuvers"]
        self.assertEqual([(m["k1"], m["k2"]) for m in legs], [(20, 40), (45, 50)])

    def test_removing_maneuvers_reports_unobservable(self):
        script = ScriptedInput({"A6  Number": ["0"]})
        code, text, _ = run_bear(["--def", str(DEFAULT_DEF), "-o", str(self.out)], script)
        self.assertEqual(code, 0)
        self.assertIn("NOT OBSERVABLE", text)

    def test_save_def_and_monte_carlo(self):
        new_def = self.dir / "new.def"
        code, text, _ = run_bear(["--batch", "--def", str(DEFAULT_DEF), "-o", str(self.out),
                                  "--save-def", str(new_def), "--mc", "20"])
        self.assertEqual(code, 0)
        self.assertIn("MONTE CARLO", text)
        self.assertEqual(load_defaults(new_def).to_dict(), load_defaults(DEFAULT_DEF).to_dict())

    def test_missing_defaults_file(self):
        code, _, err = run_bear(["--batch", "--def", str(self.dir / "nope.def")])
        self.assertEqual(code, 2)
        self.assertIn("not found", err)


class TestDbearProgram(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.results = cls.dir / "res.json"
        code, _, _ = run_bear(["--batch", "--def", str(DEFAULT_DEF), "-o", str(cls.results)])
        assert code == 0

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_batch_saves_figure(self):
        png = self.dir / "ell.png"
        code, text, _ = run_dbear([str(self.results), "--batch", "--no-show", "--save", str(png),
                                   "--times", "10", "300", "800", "-g", "3", "--rays"])
        self.assertEqual(code, 0)
        self.assertGreater(png.stat().st_size, 10000)
        self.assertIn("98.9%", text)            # probability content of 3 sigma
        rows = [ln for ln in text.splitlines() if ln.rstrip().endswith((" yes", " no"))]
        self.assertEqual(len(rows), 3)          # one table row per requested time

    def test_interactive_times_and_size(self):
        script = ScriptedInput({"Times": ["-5", "100 200"], "Ellipse": ["0", "1"]})
        code, text, _ = run_dbear([str(self.results), "--no-show"], script)
        self.assertEqual(code, 0)
        self.assertEqual(text.count("invalid input"), 2)
        self.assertIn("1-sigma", text)
        self.assertIn("39.3%", text)

    def test_truth_source(self):
        code, text, _ = run_dbear([str(self.results), "--batch", "--no-show", "--source", "truth"])
        self.assertEqual(code, 0)
        self.assertIn("B3", text)

    def test_unobservable_results_rejected(self):
        bad = self.dir / "unobs.json"
        script = ScriptedInput({"A6  Number": ["0"]})
        run_bear(["--def", str(DEFAULT_DEF), "-o", str(bad)], script)
        code, _, err = run_dbear([str(bad), "--batch", "--no-show"])
        self.assertEqual(code, 2)
        self.assertIn("singular", err)

    def test_missing_results_file(self):
        code, _, err = run_dbear([str(self.dir / "none.json"), "--batch", "--no-show"])
        self.assertEqual(code, 2)
        self.assertIn("run bear.py first", err)


if __name__ == "__main__":
    unittest.main()
