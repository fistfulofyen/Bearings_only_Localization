"""Helpers shared by the BEAR and DBEAR windows.

File locations
--------------
The programs look for their files in the *application folder*:

* packaged ``BEAR.exe`` / ``DBEAR.exe`` -- the folder containing the ``.exe``;
* run from source (``python bear_gui.py``) -- the project root.

``BEAR.DEF`` is read from there; if it is missing, the copy bundled inside
the ``.exe`` is used, and if that fails too, a built-in sample.  Results are
written to ``bear_results.json`` in the same folder (or to the temporary
folder if the application folder is read-only), which is where DBEAR looks
for them by default.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Sequence

import numpy as np

from ..samples import sample_names, sample_scenario
from ..scenario import Scenario, load_defaults

DEF_NAME = "BEAR.DEF"
RESULTS_NAME = "bear_results.json"

#: Colours of the pass / fail / warning status texts.
GOOD, BAD, WARN = "#1a7f37", "#c62828", "#b26a00"


# ------------------------------------------------------------------ locations
def is_frozen() -> bool:
    """True when running as a PyInstaller-built ``.exe``."""
    return bool(getattr(sys, "frozen", False))


def app_dir() -> Path:
    """Application folder: the ``.exe``'s folder, or the project root from source."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def bundled_path(name: str) -> Path:
    """Data file bundled inside the ``.exe`` (the project root when run from source)."""
    return Path(getattr(sys, "_MEIPASS", app_dir())) / name


def default_def_path() -> Path:
    """``BEAR.DEF`` in the application folder."""
    return app_dir() / DEF_NAME


def default_results_path() -> Path:
    """``bear_results.json`` in the application folder (shared by BEAR and DBEAR)."""
    return app_dir() / RESULTS_NAME


def fallback_results_path() -> Path:
    """Where results go if the application folder is not writable."""
    return Path(tempfile.gettempdir()) / RESULTS_NAME


def load_startup_defaults() -> tuple[Scenario, Path | None, str]:
    """Defaults to show at start-up.

    Returns
    -------
    (scenario, path, warning)
        ``path`` is the file the defaults came from (``None`` for the built-in
        sample); ``warning`` is a non-empty message if a file was unusable.
    """
    warnings = []
    for path in dict.fromkeys([default_def_path(), bundled_path(DEF_NAME)]):
        if path.is_file():
            try:
                return load_defaults(path), path, "\n".join(warnings)
            except ValueError as exc:
                warnings.append(f"Could not read {path}:\n{exc}")
    warnings.append("No usable BEAR.DEF found; using a built-in sample scenario.")
    return sample_scenario(sample_names()[0]), None, "\n".join(warnings)


def launch_program(name: str, script: str, args: Sequence[str]) -> None:
    """Start the sibling program (``DBEAR``) without waiting for it.

    Runs ``<name>.exe`` from the application folder when packaged, otherwise
    ``python <script>`` from the project root.

    Raises
    ------
    FileNotFoundError
        If the program cannot be found.
    """
    if is_frozen():
        exe = app_dir() / f"{name}.exe"
        if not exe.is_file():
            raise FileNotFoundError(f"{exe.name} not found next to this program ({app_dir()}).")
        subprocess.Popen([str(exe), *args])
    else:
        path = app_dir() / script
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found.")
        subprocess.Popen([sys.executable, str(path), *args])


# ------------------------------------------------------------------ window setup
def close_splash() -> None:
    """Close the "Loading..." splash screen shown by a packaged .exe (no-op otherwise).

    PyInstaller shows the splash image while the one-file .exe unpacks itself
    (a few seconds); the program closes it once its own window is ready.
    """
    try:
        import pyi_splash  # only exists inside a .exe built with --splash
    except ImportError:
        return
    try:
        pyi_splash.close()
    except Exception:
        pass


def enable_high_dpi() -> None:
    """Make Tk render sharply on scaled Windows displays (no-op elsewhere)."""
    if sys.platform == "win32":
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass


def setup_window(root: tk.Tk, title: str, width: int, height: int) -> None:
    """Title, DPI-aware initial size (capped to the screen), styles and error handler."""
    root.title(title)
    scale = root.winfo_fpixels("1i") / 96.0
    w = min(int(width * scale), int(root.winfo_screenwidth() * 0.92))
    h = min(int(height * scale), int(root.winfo_screenheight() * 0.88))
    root.geometry(f"{w}x{h}")
    root.minsize(min(w, int(800 * scale)), min(h, int(560 * scale)))
    icon = bundled_path("bear.ico")  # bundled in the .exe by build_exe.py
    if icon.is_file():
        try:
            root.iconbitmap(default=str(icon))
        except tk.TclError:
            pass
    style = ttk.Style(root)
    for name, colour in (("Good", GOOD), ("Bad", BAD), ("Warn", WARN)):
        style.configure(f"{name}.TLabel", foreground=colour, font=("Segoe UI", 10, "bold"))
    style.configure("Heading.TLabel", font=("Segoe UI", 10, "bold"))
    style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))

    def report_error(exc_type, exc, tb):
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        messagebox.showerror(title, f"Unexpected error:\n\n{text[-2500:]}", parent=root)

    root.report_callback_exception = report_error


# ------------------------------------------------------------------ input fields
class FieldError(ValueError):
    """Invalid user input in a named field; :meth:`focus` selects the widget."""

    def __init__(self, field: str, message: str, widget: tk.Widget | None = None):
        super().__init__(f"{field}: {message}")
        self.widget = widget

    def focus(self) -> None:
        """Put the keyboard focus on the offending entry and select its text."""
        if self.widget is not None:
            self.widget.focus_set()
            if isinstance(self.widget, (ttk.Entry, tk.Entry)):
                self.widget.select_range(0, "end")


def fmt(value: float) -> str:
    """Compact text for a number shown in an entry (round-trips to ~10 digits)."""
    return f"{float(value):.10g}"


def set_entry(entry: ttk.Entry, value) -> None:
    """Replace the text of an entry."""
    entry.delete(0, "end")
    entry.insert(0, value if isinstance(value, str) else fmt(value))


def get_float(entry: ttk.Entry, field: str) -> float:
    """Read a float from an entry, raising :class:`FieldError` if invalid."""
    text = entry.get().strip()
    try:
        value = float(text)
    except ValueError:
        raise FieldError(field, f"{text!r} is not a number", entry) from None
    if not np.isfinite(value):
        raise FieldError(field, "must be a finite number", entry)
    return value


def get_int(entry: ttk.Entry, field: str, minimum: int | None = None) -> int:
    """Read an integer from an entry, raising :class:`FieldError` if invalid."""
    text = entry.get().strip()
    try:
        value = int(text)
    except ValueError:
        raise FieldError(field, f"{text!r} is not a whole number", entry) from None
    if minimum is not None and value < minimum:
        raise FieldError(field, f"must be at least {minimum}", entry)
    return value


class VectorField(ttk.Frame):
    """A row of labelled numeric entries, e.g. ``xi0 [   ] eta0 [   ] ...``."""

    def __init__(self, parent: tk.Widget, labels: Sequence[str], width: int = 9):
        super().__init__(parent)
        self.labels = list(labels)
        self.entries: list[ttk.Entry] = []
        for i, label in enumerate(self.labels):
            ttk.Label(self, text=label).grid(row=0, column=2 * i, sticky="e", padx=(8 if i else 0, 3))
            entry = ttk.Entry(self, width=width, justify="right")
            entry.grid(row=0, column=2 * i + 1, sticky="w")
            self.entries.append(entry)

    def get(self, field: str) -> np.ndarray:
        """Values as a float array; :class:`FieldError` names the bad component."""
        return np.array(
            [get_float(e, f"{field} ({lab})") for e, lab in zip(self.entries, self.labels)]
        )

    def set(self, values: Sequence[float]) -> None:
        """Fill the entries."""
        for entry, value in zip(self.entries, values):
            set_entry(entry, value)


def scrolled_text(parent: tk.Widget) -> tk.Text:
    """Read-only, monospaced, scrollable text area placed in ``parent`` (fills it)."""
    frame = ttk.Frame(parent)
    frame.pack(fill="both", expand=True)
    text = tk.Text(frame, wrap="none", font=("Consolas", 10), state="disabled", relief="flat")
    ys = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
    xs = ttk.Scrollbar(frame, orient="horizontal", command=text.xview)
    text.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
    text.grid(row=0, column=0, sticky="nsew")
    ys.grid(row=0, column=1, sticky="ns")
    xs.grid(row=1, column=0, sticky="ew")
    frame.rowconfigure(0, weight=1)
    frame.columnconfigure(0, weight=1)
    return text


def set_text(widget: tk.Text, content: str) -> None:
    """Replace the content of a read-only text area."""
    widget.configure(state="normal")
    widget.delete("1.0", "end")
    widget.insert("1.0", content)
    widget.configure(state="disabled")
