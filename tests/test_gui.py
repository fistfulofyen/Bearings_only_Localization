"""Tests for the Tkinter windows (bearloc.gui), driven programmatically.

The windows are created hidden (``withdraw``) and message boxes are replaced
by mocks, so the tests never block.  They are skipped if Tk cannot open a
display (e.g. on a headless server).
"""

import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

try:
    import tkinter as tk

    _probe = tk.Tk()
    _probe.destroy()
    HAVE_TK = True
except Exception:  # no display / no Tk
    HAVE_TK = False

from bearloc import analysis
from bearloc.samples import sample_names, sample_scenario
from tests.helpers import DEFAULT_DEF, ROOT, observable_scenario, unobservable_scenario

if HAVE_TK:
    from bearloc.gui import common
    from bearloc.gui.bear_app import NOISE_UNITS, BearApp
    from bearloc.gui.common import FieldError
    from bearloc.gui.dbear_app import DbearApp


def assert_same_scenario(test, a, b):
    """Scenarios equal, except r which passes through degrees (~1e-10 relative)."""
    da, db = a.to_dict(), b.to_dict()
    test.assertAlmostEqual(da.pop("r") / db.pop("r"), 1.0, places=8)
    test.assertEqual(da, db)


@unittest.skipUnless(HAVE_TK, "Tk display not available")
class TestCommon(unittest.TestCase):
    def test_paths_from_source(self):
        self.assertFalse(common.is_frozen())
        self.assertEqual(common.app_dir(), ROOT)
        self.assertEqual(common.default_def_path(), DEFAULT_DEF)
        self.assertEqual(common.default_results_path(), ROOT / "bear_results.json")

    def test_startup_defaults_come_from_bear_def(self):
        scenario, path, warning = common.load_startup_defaults()
        self.assertEqual(path, DEFAULT_DEF)
        self.assertEqual(warning, "")
        self.assertEqual(scenario.n, 60)

    def test_launch_missing_program(self):
        with self.assertRaises(FileNotFoundError):
            common.launch_program("NOPE", "no_such_script.py", [])

    def _frozen(self, exe_dir: Path, bundle_dir: Path):
        """Context manager pretending we run as <exe_dir>/BEAR.exe (bundle in bundle_dir)."""
        import contextlib

        stack = contextlib.ExitStack()
        stack.enter_context(mock.patch.object(common.sys, "frozen", True, create=True))
        stack.enter_context(mock.patch.object(common.sys, "executable", str(exe_dir / "BEAR.exe")))
        stack.enter_context(mock.patch.object(common.sys, "_MEIPASS", str(bundle_dir), create=True))
        return stack

    def test_packaged_exe_paths_and_dbear_launch(self):
        with tempfile.TemporaryDirectory() as d:
            exe_dir, bundle = Path(d).resolve(), Path(d).resolve() / "bundle"
            (exe_dir / "DBEAR.exe").write_bytes(b"")
            with self._frozen(exe_dir, bundle), mock.patch.object(common.subprocess, "Popen") as popen:
                self.assertTrue(common.is_frozen())
                self.assertEqual(common.app_dir(), exe_dir)
                self.assertEqual(common.default_results_path(), exe_dir / "bear_results.json")
                self.assertEqual(common.bundled_path("BEAR.DEF"), bundle / "BEAR.DEF")
                common.launch_program("DBEAR", "dbear_gui.py", ["r.json"])
                popen.assert_called_once_with([str(exe_dir / "DBEAR.exe"), "r.json"])
                (exe_dir / "DBEAR.exe").unlink()
                with self.assertRaisesRegex(FileNotFoundError, "DBEAR.exe"):
                    common.launch_program("DBEAR", "dbear_gui.py", ["r.json"])

    def test_packaged_exe_defaults_fallbacks(self):
        with tempfile.TemporaryDirectory() as d:
            exe_dir, bundle = Path(d).resolve(), Path(d).resolve() / "bundle"
            bundle.mkdir()
            with self._frozen(exe_dir, bundle):
                # nothing anywhere: built-in sample with a warning
                scenario, path, warning = common.load_startup_defaults()
                self.assertIsNone(path)
                self.assertIn("No usable BEAR.DEF", warning)
                # copy bundled inside the .exe
                (bundle / "BEAR.DEF").write_text(DEFAULT_DEF.read_text(encoding="utf-8"), encoding="utf-8")
                self.assertEqual(common.load_startup_defaults()[1], bundle / "BEAR.DEF")
                # a broken BEAR.DEF next to the .exe: warn, then use the bundled copy
                (exe_dir / "BEAR.DEF").write_text("[target]\nx_true = 1, 2\n", encoding="utf-8")
                scenario, path, warning = common.load_startup_defaults()
                self.assertEqual(path, bundle / "BEAR.DEF")
                self.assertIn("Could not read", warning)
                # a good BEAR.DEF next to the .exe wins
                (exe_dir / "BEAR.DEF").write_text(DEFAULT_DEF.read_text(encoding="utf-8"), encoding="utf-8")
                loaded, path, warning = common.load_startup_defaults()
                self.assertEqual((path, warning), (exe_dir / "BEAR.DEF", ""))
                self.assertEqual(loaded.to_dict(), scenario.to_dict())


class _TkCase(unittest.TestCase):
    """Creates a hidden root window and mocks the message boxes."""

    module = ""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.root = tk.Tk()
        self.root.withdraw()
        patcher = mock.patch(f"bearloc.gui.{self.module}.messagebox")
        self.msg = patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self):
        self.root.update()  # let pending (idle) redraws finish before closing
        self.root.destroy()
        self.tmp.cleanup()


@unittest.skipUnless(HAVE_TK, "Tk display not available")
class TestBearWindow(_TkCase):
    module = "bear_app"

    def setUp(self):
        super().setUp()
        self.results = self.dir / "res.json"
        self.app = BearApp(self.root, observable_scenario(), None, self.results)

    def test_form_round_trip(self):
        assert_same_scenario(self, self.app.read_form(), observable_scenario())

    def test_every_sample_loads_into_the_form(self):
        for name in sample_names():
            self.app.sample_box.set(name)
            self.app._on_sample()
            assert_same_scenario(self, self.app.read_form(), sample_scenario(name))

    def test_invalid_field_is_named(self):
        self.app.e_n.delete(0, "end")
        self.app.e_n.insert(0, "abc")
        with self.assertRaisesRegex(FieldError, "A3"):
            self.app.read_form()
        self.app.run(background=False)          # shows an error, computes nothing
        self.msg.showerror.assert_called_once()
        self.assertIsNone(self.app.result)

    def test_run_fills_results_and_saves_file(self):
        self.app.run(background=False)
        self.assertIsNotNone(self.app.result)
        self.assertTrue(self.app.summary["obs"].cget("text").startswith("OBSERVABLE"))
        self.assertIn("within", self.app.summary["nees"].cget("text"))
        self.assertEqual(len(self.app.est_tree.get_children()), 4)
        self.assertIn("B8.", self.app.texts["report"].get("1.0", "end"))
        self.assertTrue(self.results.is_file())
        self.assertEqual(str(self.app.dbear_btn.cget("state")), "normal")
        self.msg.showerror.assert_not_called()

    def test_background_run(self):
        self.app.run()                          # worker thread + polling
        deadline = time.time() + 20
        while self.app.result is None and time.time() < deadline:
            self.root.update()
            time.sleep(0.01)
        self.assertIsNotNone(self.app.result)
        self.assertEqual(str(self.app.run_btn.cget("state")), "normal")

    def test_monte_carlo(self):
        self.app.mc_runs.delete(0, "end")
        self.app.mc_runs.insert(0, "25")
        self.app.run(background=False)
        self.assertEqual(self.app.mc.runs, 25)
        self.assertIn("average NEES", self.app.summary["mc"].cget("text"))

    def test_no_legs_is_unobservable(self):
        self.app.clear_legs()
        self.app.run(background=False)
        self.assertTrue(self.app.summary["obs"].cget("text").startswith("NOT OBSERVABLE"))

    def test_leg_editor(self):
        self.app.f_leg.set([1, 2, 50, 55])
        self.app.add_leg()
        self.assertEqual([(m.k1, m.k2) for m in self.app.legs], [(20, 40), (50, 55)])
        self.app.f_leg.set([0, 1, 30, 35])      # overlaps leg 1: rejected
        self.app.add_leg()
        self.msg.showerror.assert_called_once()
        self.assertEqual(len(self.app.legs), 2)
        self.app.legs_tree.selection_set("0")
        self.app.root.update()
        self.app.remove_leg()
        self.assertEqual([(m.k1, m.k2) for m in self.app.legs], [(50, 55)])

    def test_noise_unit_switch_keeps_value(self):
        r_before = self.app.read_noise_variance()
        self.app.noise_unit.set(NOISE_UNITS[1])
        self.app._on_noise_unit()
        self.assertAlmostEqual(float(self.app.e_noise.get()), np.radians(1) ** 2)
        self.assertAlmostEqual(self.app.read_noise_variance() / r_before, 1.0, places=8)

    def test_invalid_seed(self):
        self.app.e_seed.delete(0, "end")
        self.app.e_seed.insert(0, "x")
        self.app.run(background=False)
        self.msg.showerror.assert_called_once()


@unittest.skipUnless(HAVE_TK, "Tk display not available")
class TestDbearWindow(_TkCase):
    module = "dbear_app"

    def setUp(self):
        super().setUp()
        self.results = self.dir / "res.json"
        analysis.save_results(analysis.run_analysis(observable_scenario()), self.results)

    def test_opens_and_plots_default_times(self):
        app = DbearApp(self.root, self.results)
        self.assertEqual(len(app.table.get_children()), 5)
        self.assertIn("observable", app.info.cget("text"))

    def test_user_times_and_size(self):
        app = DbearApp(self.root, self.results)
        app.e_times.delete(0, "end")
        app.e_times.insert(0, "100 200")
        app.g_box.delete(0, "end")
        app.g_box.insert(0, "3")
        app.plot()
        self.assertEqual(len(app.table.get_children()), 2)
        self.assertIn("98.9", app.prob_label.cget("text"))
        for zoom, rays, source in ((False, True, "truth"), (True, False, "estimate")):
            app.zoom.set(zoom)
            app.rays.set(rays)
            app.source.set(source)
            app.plot()
            self.assertEqual(len(app.specs), 2)

    def test_invalid_times(self):
        app = DbearApp(self.root, self.results)
        app.e_times.delete(0, "end")
        app.e_times.insert(0, "-5")
        app.plot()
        self.msg.showerror.assert_called_once()

    def test_unobservable_results_show_message(self):
        path = self.dir / "unobs.json"
        analysis.save_results(analysis.run_analysis(unobservable_scenario()), path)
        app = DbearApp(self.root, path)
        self.assertEqual(app.specs, [])
        self.assertEqual(len(app.table.get_children()), 0)
        self.assertIn("singular", app.status.get())

    def test_missing_and_invalid_files(self):
        app = DbearApp(self.root, self.dir / "none.json")
        self.assertIsNone(app.results)
        bad = self.dir / "bad.json"
        bad.write_text('{"hello": 1}', encoding="utf-8")
        self.assertFalse(app.load(bad))
        self.msg.showerror.assert_called_once()

    def test_save_figure(self):
        app = DbearApp(self.root, self.results)
        png = self.dir / "fig.png"
        with mock.patch("bearloc.gui.dbear_app.filedialog.asksaveasfilename", return_value=str(png)):
            app.save_figure()
        self.assertGreater(png.stat().st_size, 10000)


if __name__ == "__main__":
    unittest.main()
