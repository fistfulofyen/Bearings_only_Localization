"""DBEAR window: target position uncertainty ellipses from BEAR results.

The user picks a results file (default: ``bear_results.json`` in the
application folder), the times at which to draw ellipses and their size g in
sigmas.  The g-sigma ellipses ``(p - p_hat)' C(t)^{-1} (p - p_hat) = g^2`` are
drawn centred at the estimated position, with ``C(t) = A(t) P A(t)'`` and ``P``
the inverse FIM from B6 (or from B3).  A table lists each ellipse's centre,
semi-axes, orientation and whether the true position lies inside.  The ellipse
algorithm is documented in :mod:`bearloc.ellipse`.
"""

from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from .. import analysis
from ..ellipse import ellipse_probability
from ..plotting import SOURCES, default_times, ellipse_specs, plot_ellipses
from ..scenario import parse_vector
from . import common
from .common import FieldError, set_entry
from .help import HelpIcon, open_help
from .help_text import DBEAR_TOPICS


class DbearApp:
    """The DBEAR main window.

    Parameters
    ----------
    root : tkinter.Tk
        Root window.
    results_path : pathlib.Path or None
        Results file to open at start-up (ignored if it does not exist).
    """

    def __init__(self, root: tk.Tk, results_path: Path | None):
        self.root = root
        self.results: dict | None = None
        self.specs = []

        self._build_menu()
        root.bind("<F1>", lambda _e: self.show_help())
        self.status = tk.StringVar()
        ttk.Label(root, textvariable=self.status, anchor="w", relief="sunken",
                  padding=(6, 2)).pack(side="bottom", fill="x")
        self._build_controls()
        self._build_body()

        if results_path is not None and Path(results_path).is_file():
            self.load(Path(results_path))
        else:
            set_entry(self.e_file, str(results_path or ""))
            self._placeholder("No results yet.\n\nRun BEAR first, or use “Browse…” to open a results file.")
            self.set_status("No results file loaded.")

    # ================================================================ layout
    def _help(self, parent: tk.Misc, key: str) -> HelpIcon:
        """A "?" icon explaining DBEAR topic ``key``."""
        return HelpIcon(parent, DBEAR_TOPICS, key, "DBEAR help")

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Open results…", accelerator="Ctrl+O", command=self.browse)
        file_menu.add_command(label="Reload", accelerator="Ctrl+R", command=self.reload)
        file_menu.add_command(label="Save figure…", accelerator="Ctrl+S", command=self.save_figure)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.destroy)
        menubar.add_cascade(label="File", menu=file_menu)
        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label="Help topics", accelerator="F1", command=self.show_help)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self._about)
        menubar.add_cascade(label="Help", menu=help_menu)
        self.root.config(menu=menubar)
        self.root.bind("<Control-o>", lambda _e: self.browse())
        self.root.bind("<Control-r>", lambda _e: self.reload())
        self.root.bind("<Control-s>", lambda _e: self.save_figure())

    def _build_controls(self) -> None:
        bar = ttk.Frame(self.root, padding=(8, 8, 8, 4))
        bar.pack(fill="x")
        ttk.Label(bar, text="Results file").grid(row=0, column=0, sticky="e")
        self.e_file = ttk.Entry(bar, width=70)
        self.e_file.grid(row=0, column=1, columnspan=5, sticky="ew", padx=4)
        ttk.Button(bar, text="Browse…", command=self.browse).grid(row=0, column=6, padx=2)
        ttk.Button(bar, text="Reload", command=self.reload).grid(row=0, column=7, padx=2)
        self._help(bar, "results_file").grid(row=0, column=8, padx=(4, 0))

        ttk.Label(bar, text="Times").grid(row=1, column=0, sticky="e", pady=6)
        times_box = ttk.Frame(bar)
        times_box.grid(row=1, column=1, sticky="w", padx=4)
        self.e_times = ttk.Entry(times_box, width=34)
        self.e_times.pack(side="left")
        self._help(times_box, "times").pack(side="left", padx=(4, 0))
        self.e_times.bind("<Return>", lambda _e: self.plot())
        ttk.Label(bar, text="g (sigmas)").grid(row=1, column=2, sticky="e", padx=(10, 3))
        self.g_box = ttk.Spinbox(bar, from_=0.5, to=6.0, increment=0.5, width=5,
                                 justify="right", command=self.plot)
        self.g_box.grid(row=1, column=3, sticky="w")
        self.g_box.bind("<Return>", lambda _e: self.plot())
        set_entry(self.g_box, "2")
        g_extra = ttk.Frame(bar)
        g_extra.grid(row=1, column=4, sticky="w")
        self._help(g_extra, "g").pack(side="left", padx=(4, 0))
        self.prob_label = ttk.Label(g_extra, text="", foreground="#555")
        self.prob_label.pack(side="left", padx=6)

        opts = ttk.Frame(bar)
        opts.grid(row=1, column=5, columnspan=3, sticky="e")
        self.source = tk.StringVar(value="estimate")
        ttk.Radiobutton(opts, text="Covariance at estimate (B6)", value="estimate",
                        variable=self.source, command=self.plot).pack(side="left")
        ttk.Radiobutton(opts, text="at truth (B3)", value="truth",
                        variable=self.source, command=self.plot).pack(side="left", padx=(4, 0))
        self._help(opts, "source").pack(side="left", padx=(4, 12))
        self.rays = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Bearing rays", variable=self.rays, command=self.plot).pack(side="left")
        self._help(opts, "rays").pack(side="left", padx=(2, 12))
        self.zoom = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Zoom to ellipses", variable=self.zoom,
                        command=self.plot).pack(side="left")
        self._help(opts, "zoom").pack(side="left", padx=(2, 0))

        act = ttk.Frame(bar)
        act.grid(row=2, column=0, columnspan=8, sticky="w")
        ttk.Button(act, text="▶  Plot", style="Accent.TButton", command=self.plot).pack(side="left", ipadx=8)
        ttk.Button(act, text="Save figure…", command=self.save_figure).pack(side="left", padx=6)
        ttk.Label(act, text="How it works").pack(side="left", padx=(6, 3))
        self._help(act, "overview").pack(side="left")
        self.info = ttk.Label(act, text="", foreground="#333")
        self.info.pack(side="left", padx=10)
        bar.columnconfigure(1, weight=1)

    def _build_body(self) -> None:
        paned = ttk.PanedWindow(self.root, orient="vertical")
        paned.pack(fill="both", expand=True, padx=8, pady=(0, 6))
        top = ttk.Frame(paned)
        bottom = ttk.Frame(paned)
        paned.add(top, weight=4)
        paned.add(bottom, weight=1)

        self.fig = Figure(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=top)
        toolbar = NavigationToolbar2Tk(self.canvas, top, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        head = ttk.Frame(bottom)
        head.pack(side="top", fill="x", pady=(2, 2))
        ttk.Label(head, text="Ellipses", style="Heading.TLabel").pack(side="left")
        self._help(head, "ellipse_table").pack(side="left", padx=(4, 0))
        self._help(head, "algorithm").pack(side="right")
        ttk.Label(head, text="How an ellipse is computed", foreground="#555").pack(side="right", padx=4)
        cols = ("t", "cx", "cy", "major", "minor", "angle", "inside")
        heads = ("t", "centre ξ", "centre η", "semi-major", "semi-minor", "angle (deg)", "truth inside")
        self.table = ttk.Treeview(bottom, columns=cols, show="headings", height=5)
        for col, head in zip(cols, heads):
            self.table.heading(col, text=head)
            self.table.column(col, width=110, anchor="e" if col != "inside" else "center")
        ys = ttk.Scrollbar(bottom, orient="vertical", command=self.table.yview)
        self.table.configure(yscrollcommand=ys.set)
        self.table.pack(side="left", fill="both", expand=True)
        ys.pack(side="right", fill="y")

    def show_help(self, key: str = "overview") -> None:
        """Open the Help window (F1) on topic ``key``."""
        open_help(self.root, DBEAR_TOPICS, key, "DBEAR help")

    def _about(self) -> None:
        messagebox.showinfo(
            "About DBEAR",
            "DBEAR — target position uncertainty ellipses\n\n"
            "Project 3.8.3 of Bar-Shalom, Li & Kirubarajan, Estimation with Applications "
            "to Tracking and Navigation.\n\n"
            "Draws the g-sigma ellipses of the target position computed by BEAR, at "
            "user-specified times.\n\n"
            f"Application folder: {common.app_dir()}",
            parent=self.root)

    # ================================================================ loading
    def set_status(self, text: str) -> None:
        """Show a message in the status bar."""
        self.status.set(text)

    def load(self, path: Path) -> bool:
        """Open a results file, reset the times to their defaults and plot."""
        try:
            results = analysis.load_results(path)
            for key in ("scenario", "mle", "truth", "estimate", "times", "measurements"):
                if key not in results:
                    raise KeyError(f"missing entry '{key}' (not a BEAR results file?)")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            messagebox.showerror("Open results", f"Cannot read {path}:\n{exc}", parent=self.root)
            return False
        self.results = results
        set_entry(self.e_file, str(path))
        set_entry(self.e_times, " ".join(f"{t:g}" for t in default_times(results)))
        s, mle = results["scenario"], results["mle"]
        x = ", ".join(f"{v:.5g}" for v in mle["x"])
        obs = "observable" if results["truth"]["invertible"] else "NOT observable"
        conv = "converged" if mle["converged"] else "NOT converged"
        self.info.configure(text=f"x̂ = ({x})   ·   {conv}   ·   {obs}   ·   n = {s['n']}, T = {s['T']:g}")
        self.plot()
        return True

    def browse(self) -> None:
        """Choose a results file."""
        current = Path(self.e_file.get() or common.default_results_path())
        path = filedialog.askopenfilename(
            parent=self.root, title="Open BEAR results", initialdir=str(current.parent),
            filetypes=[("BEAR results", "*.json"), ("All files", "*.*")])
        if path:
            self.load(Path(path))

    def reload(self) -> None:
        """Re-read the results file (e.g. after a new BEAR run)."""
        text = self.e_file.get().strip()
        if text:
            self.load(Path(text))

    # ================================================================ plotting
    def read_times(self) -> list[float]:
        """Times from the entry (non-negative numbers)."""
        try:
            times = parse_vector(self.e_times.get())
        except ValueError:
            raise FieldError("Times", "enter numbers separated by spaces", self.e_times) from None
        if times.size == 0:
            raise FieldError("Times", "enter at least one time", self.e_times)
        if np.any(times < 0):
            raise FieldError("Times", "times must be ≥ 0", self.e_times)
        return [float(t) for t in times]

    def read_g(self) -> float:
        """Ellipse size g (> 0)."""
        g = common.get_float(self.g_box, "g")
        if g <= 0:
            raise FieldError("g", "must be positive", self.g_box)
        return g

    def _placeholder(self, message: str) -> None:
        self.fig.clear()
        ax = self.fig.add_subplot()
        ax.axis("off")
        ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=12, wrap=True,
                transform=ax.transAxes)
        self.canvas.draw_idle()
        self.table.delete(*self.table.get_children())

    def plot(self) -> None:
        """Draw the ellipses for the current settings and fill the table."""
        if self.results is None:
            return
        try:
            times, g = self.read_times(), self.read_g()
        except FieldError as exc:
            exc.focus()
            messagebox.showerror("Invalid input", str(exc), parent=self.root)
            return
        self.prob_label.configure(text=f"= {100 * ellipse_probability(g):.1f}% probability")
        source = self.source.get()
        try:
            self.specs = ellipse_specs(self.results, times, g, source)
        except ValueError as exc:
            self.specs = []
            self._placeholder(f"Cannot draw ellipses:\n{exc}")
            self.set_status(str(exc))
            return
        self.fig.clear()
        plot_ellipses(self.results, self.specs, source=source, show_rays=self.rays.get(),
                      ax=self.fig.add_subplot(), zoom=self.zoom.get())
        self.canvas.draw_idle()
        self.table.delete(*self.table.get_children())
        for e in self.specs:
            a, b = e.semi_axes
            self.table.insert("", "end", values=(
                f"{e.t:g}", f"{e.center[0]:.6g}", f"{e.center[1]:.6g}", f"{a:.5g}", f"{b:.5g}",
                f"{e.orientation_deg:.2f}", "yes" if e.contains_truth else "no"))
        inside = sum(e.contains_truth for e in self.specs)
        self.set_status(f"{len(self.specs)} ellipse(s), {g:g}-sigma, covariance from {SOURCES[source]}; "
                        f"true position inside {inside} of {len(self.specs)}.")

    def save_figure(self) -> None:
        """Save the current figure (PNG, PDF or SVG)."""
        path = filedialog.asksaveasfilename(
            parent=self.root, title="Save figure", defaultextension=".png", initialfile="ellipses.png",
            filetypes=[("PNG image", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg")])
        if path:
            self.fig.savefig(path, dpi=150)
            self.set_status(f"Figure saved to {path}")


def selftest(results: Path, out_png: Path) -> int:
    """Non-interactive check of a build: plot ``results`` in a hidden window.

    Saves the figure to ``out_png``; returns 0 if ellipses were drawn, else 1.
    """
    common.close_splash()
    root = tk.Tk()
    root.withdraw()
    try:
        app = DbearApp(root, results)
        ok = bool(app.specs)
        if ok:
            app.fig.savefig(out_png, dpi=100)
        root.update()
    finally:
        root.destroy()
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    """Start the DBEAR window; an optional first argument names the results file.

    ``--selftest results.json out.png`` runs the build check instead.
    """
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] == "--selftest":
        results = Path(args[1]) if len(args) > 1 else common.default_results_path()
        out = Path(args[2]) if len(args) > 2 else results.with_suffix(".png")
        return selftest(results, out)
    path = Path(args[0]) if args else common.default_results_path()
    common.enable_high_dpi()
    root = tk.Tk()
    common.setup_window(root, "DBEAR — Target position uncertainty ellipses", 1250, 900)
    DbearApp(root, path)
    root.after(100, common.close_splash)
    root.mainloop()
    return 0
