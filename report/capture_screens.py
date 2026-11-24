"""Capture screenshots of the BEAR and DBEAR windows for the report (Windows).

Usage (from the project root)::

    python report/capture_screens.py

Opens each window for a few seconds, runs the default scenario, and saves
``report/figures/screen_bear.png``, ``screen_help.png`` and
``screen_dbear.png``.  Requires a display (the windows appear briefly).
"""

from __future__ import annotations

import sys
import tempfile
import time
import tkinter as tk
from pathlib import Path

from PIL import ImageGrab

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bearloc.gui import common  # noqa: E402
from bearloc.gui.bear_app import BearApp  # noqa: E402
from bearloc.gui.dbear_app import DbearApp  # noqa: E402
from bearloc.gui.help import HelpIcon  # noqa: E402

FIG = ROOT / "report" / "figures"


def grab(win: tk.Misc, path: Path) -> None:
    """Screenshot the client area of ``win``."""
    win.update()
    time.sleep(0.7)
    win.update()
    x, y = win.winfo_rootx(), win.winfo_rooty()
    ImageGrab.grab(bbox=(x, y, x + win.winfo_width(), y + win.winfo_height()), all_screens=True).save(path)
    print("saved", path.name)


def icons(widget: tk.Misc) -> list[HelpIcon]:
    found = []
    for child in widget.winfo_children():
        if isinstance(child, HelpIcon):
            found.append(child)
        found += icons(child)
    return found


def main() -> None:
    common.enable_high_dpi()
    results = Path(tempfile.gettempdir()) / "bear_report_screens.json"

    root = tk.Tk()
    common.setup_window(root, "BEAR", 1450, 920)
    root.attributes("-topmost", True)
    defaults, path, _ = common.load_startup_defaults()
    app = BearApp(root, defaults, path, results)
    common.set_entry(app.mc_runs, "500")
    app.run(background=False)
    app.set_status("Done.  Results saved for DBEAR.")  # keep local paths out of the picture
    a6 = next(i for i in icons(root) if i.key == "A6")
    a6.tooltip.show()
    a6.tooltip.window.wm_geometry(f"+{a6.winfo_rootx() + 18}+{a6.winfo_rooty() + 18}")
    grab(root, FIG / "screen_bear.png")
    a6.tooltip.hide()
    window = next(i for i in icons(root) if i.key == "B8").open()
    window.win.attributes("-topmost", True)
    grab(window.win, FIG / "screen_help.png")
    root.destroy()

    root = tk.Tk()
    common.setup_window(root, "DBEAR", 1250, 860)
    root.attributes("-topmost", True)
    dbear = DbearApp(root, results)
    common.set_entry(dbear.e_file, "bear_results.json")
    dbear.rays.set(True)
    dbear.plot()
    grab(root, FIG / "screen_dbear.png")
    root.destroy()


if __name__ == "__main__":
    main()
