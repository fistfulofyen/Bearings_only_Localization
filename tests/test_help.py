"""Tests for the in-program help: texts, "?" icons, tooltips and the Help window."""

import unittest

from bearloc.gui.help_text import BEAR_TOPICS, DBEAR_TOPICS
from bearloc.samples import sample_names
from tests.test_gui import HAVE_TK, _TkCase
from tests.helpers import observable_scenario

if HAVE_TK:
    from bearloc import analysis
    from bearloc.gui.bear_app import BearApp
    from bearloc.gui.dbear_app import DbearApp
    from bearloc.gui.help import HelpIcon


def help_icons(widget):
    """All HelpIcon widgets below ``widget``."""
    found = []
    for child in widget.winfo_children():
        if isinstance(child, HelpIcon):
            found.append(child)
        found += help_icons(child)
    return found


class TestHelpTexts(unittest.TestCase):
    def test_every_topic_is_complete(self):
        for topics in (BEAR_TOPICS, DBEAR_TOPICS):
            for key, t in topics.items():
                with self.subTest(key=key):
                    self.assertTrue(t.group and t.title and t.tip and t.body)
                    self.assertLessEqual(len(t.tip), 200, "tooltips must stay short")
                    self.assertGreater(len(t.body), len(t.tip))

    def test_both_programs_start_with_an_overview(self):
        self.assertEqual(next(iter(BEAR_TOPICS)), "overview")
        self.assertEqual(next(iter(DBEAR_TOPICS)), "overview")

    def test_inputs_A1_to_A7_and_steps_documented(self):
        titles = " ".join(t.title for t in BEAR_TOPICS.values())
        for item in ("A1", "A2", "A3", "A4", "A5", "A6", "A7", "B1", "B5", "B7", "B8"):
            self.assertIn(f"({item})", titles)

    def test_sample_help_lists_every_sample(self):
        for name in sample_names():
            self.assertIn(name, BEAR_TOPICS["sample"].body)


@unittest.skipUnless(HAVE_TK, "Tk display not available")
class TestBearHelp(_TkCase):
    module = "bear_app"

    def setUp(self):
        super().setUp()
        self.app = BearApp(self.root, observable_scenario(), None, self.dir / "res.json")
        self.icons = {icon.key: icon for icon in help_icons(self.root)}

    def test_every_topic_has_an_icon(self):
        self.assertEqual(set(self.icons), set(BEAR_TOPICS))

    def test_tooltip_shows_and_hides(self):
        tip = self.icons["A6"].tooltip
        tip.show()
        self.assertIsNotNone(tip.window)
        self.assertTrue(tip.window.winfo_exists())
        tip.hide()
        self.assertIsNone(tip.window)

    def test_click_opens_one_reusable_help_window(self):
        window = self.icons["B8"].open()
        self.assertEqual(window.key, "B8")
        self.assertIn("NEES (B8)", window.text.get("1.0", "end"))
        self.assertEqual(window.tree.selection(), ("B8",))
        again = self.icons["A1"].open()
        self.assertIs(again, window)
        self.assertIn("True target parameter (A1)", window.text.get("1.0", "end"))

    def test_tree_selection_and_f1(self):
        self.app.show_help()
        window = self.root._help_window
        self.assertEqual(window.key, "overview")
        window.tree.selection_set("B7")
        self.root.update()
        self.assertEqual(window.key, "B7")
        window.close()
        self.assertFalse(window.exists())
        self.app.show_help("seed")                       # a closed window is re-created
        self.assertIsNot(self.root._help_window, window)
        self.assertEqual(self.root._help_window.key, "seed")


@unittest.skipUnless(HAVE_TK, "Tk display not available")
class TestDbearHelp(_TkCase):
    module = "dbear_app"

    def test_every_topic_has_an_icon_and_f1_works(self):
        path = self.dir / "res.json"
        analysis.save_results(analysis.run_analysis(observable_scenario()), path)
        app = DbearApp(self.root, path)
        self.assertEqual({i.key for i in help_icons(self.root)}, set(DBEAR_TOPICS))
        app.show_help("g")
        self.assertIn("86.5 %", self.root._help_window.text.get("1.0", "end"))


if __name__ == "__main__":
    unittest.main()
