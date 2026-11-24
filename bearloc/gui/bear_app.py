"""BEAR window: enter inputs A1-A7, run steps B1-B8, inspect the results.

Layout
------
* **Left**: the input form, pre-filled from ``BEAR.DEF`` (or a sample from
  the "Load sample" list).  Maneuver legs (A6) are edited in a small table.
* **Right**: result tabs

  - *Summary* -- pass/fail status of observability (B1), convergence (B5),
    fitting error (B7), NEES (B8) and the optional Monte Carlo test; the
    estimate next to the truth; a plot of the geometry and estimated track.
  - *B1-B3*, *B4*, *B6* and *Full report* -- the text output of the CLI.

The computation runs in a background thread so the window stays responsive
during long Monte Carlo runs.  After every run the results are saved to
``bear_results.json`` and the "Plot ellipses (DBEAR)" button opens DBEAR on
them.
"""

from __future__ import annotations

import queue
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from .. import analysis, report
from ..model import Maneuver, validate_maneuvers
from ..plotting import plot_geometry
from ..samples import sample_description, sample_names, sample_scenario
from ..scenario import METHODS, Scenario, load_defaults, save_defaults
from . import common
from .help import HelpIcon, open_help, titled_frame
from .help_text import BEAR_TOPICS
from .common import FieldError, VectorField, get_float, get_int, set_entry

#: Labels of the parameter components in the form.
PARAM_LABELS = ("ξ0", "η0", "dξ/dt", "dη/dt")
#: Units in which the noise (A4) can be entered.
NOISE_UNITS = ("σ (deg)", "r (rad²)")
_CHOOSE = "(choose a sample…)"


class BearApp:
    """The BEAR main window.

    Parameters
    ----------
    root : tkinter.Tk
        Root window.
    defaults : Scenario
        Initial values of the form (and target of "Reset").
    def_path : pathlib.Path or None
        File the defaults came from (shown in the status bar).
    results_path : pathlib.Path
        Where results are saved after each run (read by DBEAR).
    """

    def __init__(self, root: tk.Tk, defaults: Scenario, def_path: Path | None, results_path: Path):
        self.root = root
        self.defaults = defaults
        self.def_path = def_path
        self.results_path = results_path
        self.result: analysis.AnalysisResult | None = None
        self.mc: analysis.MonteCarloResult | None = None
        self.legs: list[Maneuver] = []
        self._queue: queue.Queue = queue.Queue()
        self._busy = False

        self._build_menu()
        self.status = tk.StringVar()
        ttk.Label(root, textvariable=self.status, anchor="w", relief="sunken",
                  padding=(6, 2)).pack(side="bottom", fill="x")
        paned = ttk.PanedWindow(root, orient="horizontal")
        paned.pack(fill="both", expand=True)
        left = ttk.Frame(paned, padding=8)
        right = ttk.Frame(paned, padding=(4, 8, 8, 8))
        paned.add(left, weight=0)
        paned.add(right, weight=1)
        self._build_inputs(left)
        self._build_results(right)
        root.bind("<F5>", lambda _e: self.run())
        root.bind("<F1>", lambda _e: self.show_help())
        root.bind("<Control-o>", lambda _e: self.open_defaults())
        root.bind("<Control-s>", lambda _e: self.save_inputs())

        self.fill_form(defaults)
        self.preview()
        src = def_path if def_path else "built-in sample"
        self.set_status(f"Defaults loaded from {src}.  Edit the inputs and press Run (F5).")

    # ================================================================ layout
    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Open defaults file…", accelerator="Ctrl+O", command=self.open_defaults)
        file_menu.add_command(label="Save inputs as…", accelerator="Ctrl+S", command=self.save_inputs)
        file_menu.add_separator()
        file_menu.add_command(label="Save results (JSON) as…", command=self.save_results_as)
        file_menu.add_command(label="Save report (text) as…", command=self.save_report)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.destroy)
        menubar.add_cascade(label="File", menu=file_menu)
        run_menu = tk.Menu(menubar, tearoff=False)
        run_menu.add_command(label="Run B1–B8", accelerator="F5", command=self.run)
        run_menu.add_command(label="Preview geometry", command=self.preview)
        run_menu.add_command(label="Plot ellipses (DBEAR)", command=self.open_dbear)
        menubar.add_cascade(label="Run", menu=run_menu)
        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label="Help topics", accelerator="F1", command=self.show_help)
        help_menu.add_separator()
        help_menu.add_command(label="About", command=self._about)
        menubar.add_cascade(label="Help", menu=help_menu)
        self.root.config(menu=menubar)

    def _help(self, parent: tk.Misc, key: str) -> HelpIcon:
        """A "?" icon explaining BEAR topic ``key``."""
        return HelpIcon(parent, BEAR_TOPICS, key, "BEAR help")

    def _frame(self, parent: tk.Misc, text: str, key: str) -> ttk.LabelFrame:
        """A titled input group with a "?" icon in its title."""
        return titled_frame(parent, text, BEAR_TOPICS, key, "BEAR help")

    def _build_inputs(self, parent: ttk.Frame) -> None:
        pad = {"padx": 4, "pady": 3}

        # samples
        top = ttk.Frame(parent)
        top.pack(fill="x", pady=(0, 6))
        ttk.Label(top, text="Load sample:").pack(side="left")
        self.sample_box = ttk.Combobox(top, state="readonly", width=38,
                                       values=[_CHOOSE, *sample_names()])
        self.sample_box.current(0)
        self.sample_box.pack(side="left", padx=4)
        self.sample_box.bind("<<ComboboxSelected>>", self._on_sample)
        self._help(top, "sample").pack(side="left")
        self._help(top, "overview").pack(side="right")
        ttk.Label(top, text="How it works").pack(side="right", padx=4)
        self.sample_info = ttk.Label(parent, text="", foreground="#555", wraplength=520)
        self.sample_info.pack(fill="x")

        # A1
        f = self._frame(parent, "A1   True target parameter at t = 0  (position, velocity)", "A1")
        f.pack(fill="x", **pad)
        self.f_x_true = VectorField(f, PARAM_LABELS)
        self.f_x_true.pack(anchor="w", padx=6, pady=4)

        # A2-A4
        f = ttk.LabelFrame(parent, text="A2–A4   Sampling and measurement noise")
        f.pack(fill="x", **pad)
        ttk.Label(f, text="T").grid(row=0, column=0, sticky="e", padx=(6, 3), pady=4)
        self.e_T = ttk.Entry(f, width=8, justify="right")
        self.e_T.grid(row=0, column=1, sticky="w")
        self._help(f, "T").grid(row=0, column=2, padx=(3, 0))
        ttk.Label(f, text="n").grid(row=0, column=3, sticky="e", padx=(12, 3))
        self.e_n = ttk.Entry(f, width=6, justify="right")
        self.e_n.grid(row=0, column=4, sticky="w")
        self._help(f, "n").grid(row=0, column=5, padx=(3, 0))
        ttk.Label(f, text="Noise").grid(row=0, column=6, sticky="e", padx=(12, 3))
        self.e_noise = ttk.Entry(f, width=12, justify="right")
        self.e_noise.grid(row=0, column=7, sticky="w")
        self.noise_unit = ttk.Combobox(f, state="readonly", width=9, values=NOISE_UNITS)
        self.noise_unit.current(0)
        self.noise_unit.grid(row=0, column=8, sticky="w", padx=(4, 0))
        self.noise_unit.bind("<<ComboboxSelected>>", self._on_noise_unit)
        self._noise_unit_shown = NOISE_UNITS[0]
        self._help(f, "noise").grid(row=0, column=9, padx=(3, 6))
        ttk.Label(f, text="samples at t = kT, k = 1..n", foreground="#555").grid(
            row=1, column=0, columnspan=10, sticky="w", padx=6, pady=(0, 4))

        # A5
        f = self._frame(parent, "A5   Platform initial motion", "A5")
        f.pack(fill="x", **pad)
        ttk.Label(f, text="Position").grid(row=0, column=0, sticky="w", padx=6, pady=3)
        self.f_pos = VectorField(f, ("ξ", "η"))
        self.f_pos.grid(row=0, column=1, sticky="w")
        ttk.Label(f, text="Velocity").grid(row=1, column=0, sticky="w", padx=6, pady=3)
        self.f_vel = VectorField(f, ("vξ", "vη"))
        self.f_vel.grid(row=1, column=1, sticky="w")

        # A6
        f = self._frame(parent, "A6   Platform maneuvers  (velocity in effect for k1·T ≤ t < k2·T)", "A6")
        f.pack(fill="x", **pad)
        cols = ("leg", "vx", "vy", "k1", "k2", "time")
        self.legs_tree = ttk.Treeview(f, columns=cols, show="headings", height=4, selectmode="browse")
        for col, head, w in zip(cols, ("#", "vξ", "vη", "k1", "k2", "time window"),
                                (30, 70, 70, 50, 50, 130)):
            self.legs_tree.heading(col, text=head)
            self.legs_tree.column(col, width=w, anchor="center", stretch=col == "time")
        self.legs_tree.grid(row=0, column=0, columnspan=2, sticky="ew", padx=6, pady=(4, 2))
        self.legs_tree.bind("<<TreeviewSelect>>", self._on_leg_select)
        self.e_T.bind("<FocusOut>", lambda _e: self._refresh_legs())  # time windows depend on T
        self.f_leg = VectorField(f, ("vξ", "vη", "k1", "k2"), width=7)
        self.f_leg.grid(row=1, column=0, sticky="w", padx=6, pady=2)
        btns = ttk.Frame(f)
        btns.grid(row=2, column=0, columnspan=2, sticky="w", padx=6, pady=(2, 6))
        ttk.Button(btns, text="Add leg", command=self.add_leg).pack(side="left")
        ttk.Button(btns, text="Update selected", command=self.update_leg).pack(side="left", padx=4)
        ttk.Button(btns, text="Remove selected", command=self.remove_leg).pack(side="left")
        ttk.Button(btns, text="Remove all", command=self.clear_legs).pack(side="left", padx=4)
        f.columnconfigure(0, weight=1)

        # A7
        f = self._frame(parent, "A7   Initial estimate for the minimization", "A7")
        f.pack(fill="x", **pad)
        self.f_x_init = VectorField(f, PARAM_LABELS)
        self.f_x_init.pack(anchor="w", padx=6, pady=4)

        # options
        f = ttk.LabelFrame(parent, text="Options")
        f.pack(fill="x", **pad)
        ttk.Label(f, text="Method").grid(row=0, column=0, sticky="e", padx=(6, 3), pady=4)
        self.method_box = ttk.Combobox(f, state="readonly", width=13, values=list(METHODS))
        self.method_box.grid(row=0, column=1, sticky="w")
        self._help(f, "method").grid(row=0, column=2, padx=(3, 0))
        ttk.Label(f, text="Seed").grid(row=0, column=3, sticky="e", padx=(12, 3))
        self.e_seed = ttk.Entry(f, width=8, justify="right")
        self.e_seed.grid(row=0, column=4, sticky="w")
        self._help(f, "seed").grid(row=0, column=5, padx=(3, 0))
        ttk.Label(f, text="(blank = random)", foreground="#555").grid(row=0, column=6, sticky="w", padx=3)
        ttk.Label(f, text="Confidence").grid(row=1, column=0, sticky="e", padx=(6, 3), pady=4)
        self.e_conf = ttk.Entry(f, width=8, justify="right")
        self.e_conf.grid(row=1, column=1, sticky="w")
        self._help(f, "confidence").grid(row=1, column=2, padx=(3, 0))
        ttk.Label(f, text="Monte Carlo runs").grid(row=1, column=3, sticky="e", padx=(12, 3))
        self.mc_runs = ttk.Spinbox(f, from_=0, to=100000, increment=100, width=8, justify="right")
        self.mc_runs.grid(row=1, column=4, sticky="w")
        set_entry(self.mc_runs, "0")
        self._help(f, "mc").grid(row=1, column=5, padx=(3, 0))
        ttk.Label(f, text="(0 = off)", foreground="#555").grid(row=1, column=6, sticky="w", padx=3)

        # actions
        act = ttk.Frame(parent)
        act.pack(fill="x", pady=(10, 0))
        self.run_btn = ttk.Button(act, text="▶  Run B1–B8  (F5)", style="Accent.TButton", command=self.run)
        self.run_btn.pack(side="left", ipadx=8, ipady=3)
        ttk.Button(act, text="Preview geometry", command=self.preview).pack(side="left", padx=6)
        ttk.Button(act, text="Reset to defaults", command=lambda: self.fill_form(self.defaults)).pack(side="left")
        self.dbear_btn = ttk.Button(parent, text="Plot uncertainty ellipses (DBEAR)…",
                                    command=self.open_dbear, state="disabled")
        self.dbear_btn.pack(anchor="w", pady=(8, 0))
        self.progress = ttk.Progressbar(parent, mode="indeterminate")

    def _build_results(self, parent: ttk.Frame) -> None:
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill="both", expand=True)

        # --- summary tab
        tab = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(tab, text="Summary")
        grid = ttk.Frame(tab)
        grid.pack(fill="x")
        self.summary: dict[str, ttk.Label] = {}
        rows = (("obs", "Observability (B1)", "B1"), ("mle", "Estimation (B5)", "B5"),
                ("fit", "Fitting error (B7)", "B7"), ("nees", "NEES (B8)", "B8"),
                ("mc", "Monte Carlo", "mc_result"))
        for i, (key, title, topic) in enumerate(rows):
            ttk.Label(grid, text=title, style="Heading.TLabel").grid(row=i, column=0, sticky="w", pady=2)
            self._help(grid, topic).grid(row=i, column=1, padx=(4, 0))
            lab = ttk.Label(grid, text="–")
            lab.grid(row=i, column=2, sticky="w", padx=10)
            self.summary[key] = lab

        head = ttk.Frame(tab)
        head.pack(fill="x", pady=(10, 0))
        ttk.Label(head, text="Estimate vs. truth", style="Heading.TLabel").pack(side="left")
        self._help(head, "est_table").pack(side="left", padx=(4, 0))

        cols = ("param", "initial", "estimate", "true", "error", "std", "ratio")
        heads = ("Parameter", "Initial (A7)", "Estimate (B5)", "True (A1)", "Error", "Std (B6)", "|Error| / Std")
        self.est_tree = ttk.Treeview(tab, columns=cols, show="headings", height=4)
        for col, head in zip(cols, heads):
            self.est_tree.heading(col, text=head)
            self.est_tree.column(col, width=105, anchor="e" if col != "param" else "w")
        self.est_tree.pack(fill="x", pady=(4, 6))

        head = ttk.Frame(tab)
        head.pack(fill="x")
        ttk.Label(head, text="Geometry", style="Heading.TLabel").pack(side="left")
        self._help(head, "geometry").pack(side="left", padx=(4, 0))
        self._help(head, "tabs").pack(side="right")
        ttk.Label(head, text="What the other tabs show", foreground="#555").pack(side="right", padx=4)

        self.fig = Figure(figsize=(6, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=tab)
        toolbar = NavigationToolbar2Tk(self.canvas, tab, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        # --- text tabs
        self.texts: dict[str, tk.Text] = {}
        for key, title in (("truth", "B1–B3: FIM at truth"), ("meas", "B4: Measurements"),
                           ("est", "B6: FIM at estimate"), ("report", "Full report")):
            tab = ttk.Frame(self.notebook)
            self.notebook.add(tab, text=title)
            self.texts[key] = common.scrolled_text(tab)
            common.set_text(self.texts[key], "Press Run (F5) to compute.")

    # ================================================================ form <-> scenario
    def fill_form(self, s: Scenario) -> None:
        """Show scenario ``s`` in the input form."""
        self.f_x_true.set(s.x_true)
        set_entry(self.e_T, s.T)
        set_entry(self.e_n, str(s.n))
        self.noise_unit.current(0)
        self._noise_unit_shown = NOISE_UNITS[0]
        set_entry(self.e_noise, np.degrees(np.sqrt(s.r)))
        self.f_pos.set(s.platform_pos)
        self.f_vel.set(s.platform_vel)
        self.legs = list(s.maneuvers)
        self._refresh_legs()
        self.f_x_init.set(s.x_init)
        self.method_box.set(s.method)
        set_entry(self.e_seed, "" if s.seed is None else str(s.seed))
        set_entry(self.e_conf, s.confidence)

    def read_noise_variance(self) -> float:
        """Noise variance r (rad^2) from the A4 entry and its unit."""
        value = get_float(self.e_noise, "A4 noise")
        if value <= 0:
            raise FieldError("A4 noise", "must be positive", self.e_noise)
        return float(np.radians(value) ** 2) if self.noise_unit.get() == NOISE_UNITS[0] else value

    def read_form(self) -> Scenario:
        """Build a :class:`Scenario` from the form.

        Raises
        ------
        FieldError
            For an unreadable field (the widget can be focused).
        ValueError
            For an inconsistent combination (e.g. overlapping legs).
        """
        seed_text = self.e_seed.get().strip()
        try:
            seed = int(seed_text) if seed_text else None
        except ValueError:
            raise FieldError("Seed", f"{seed_text!r} is not a whole number", self.e_seed) from None
        return Scenario(
            x_true=self.f_x_true.get("A1 true parameter"),
            T=get_float(self.e_T, "A2 sampling period T"),
            n=get_int(self.e_n, "A3 number of samples n", minimum=1),
            r=self.read_noise_variance(),
            platform_pos=self.f_pos.get("A5 platform position"),
            platform_vel=self.f_vel.get("A5 platform velocity"),
            maneuvers=list(self.legs),
            x_init=self.f_x_init.get("A7 initial estimate"),
            seed=seed,
            method=self.method_box.get(),
            max_iter=self.defaults.max_iter,
            tol=self.defaults.tol,
            confidence=get_float(self.e_conf, "Confidence"),
        )

    def read_mc_runs(self) -> int:
        """Number of Monte Carlo runs (0 = none)."""
        return get_int(self.mc_runs, "Monte Carlo runs", minimum=0)

    def _on_noise_unit(self, _event=None) -> None:
        """Convert the noise value when the unit is switched."""
        new = self.noise_unit.get()
        if new == self._noise_unit_shown:
            return
        try:
            value = get_float(self.e_noise, "A4 noise")
        except FieldError:
            self._noise_unit_shown = new
            return
        if new == NOISE_UNITS[1]:          # sigma (deg) -> r (rad^2)
            set_entry(self.e_noise, float(np.radians(value) ** 2))
        else:                              # r (rad^2) -> sigma (deg)
            set_entry(self.e_noise, float(np.degrees(np.sqrt(abs(value)))))
        self._noise_unit_shown = new

    def _on_sample(self, _event=None) -> None:
        name = self.sample_box.get()
        if name == _CHOOSE:
            self.sample_info.configure(text="")
            return
        self.fill_form(sample_scenario(name))
        self.sample_info.configure(text=sample_description(name))
        self.preview()
        self.set_status(f"Sample “{name}” loaded.  Press Run (F5).")

    # ================================================================ maneuver legs
    def _refresh_legs(self) -> None:
        self.legs.sort(key=lambda m: m.k1)
        self.legs_tree.delete(*self.legs_tree.get_children())
        try:
            T = float(self.e_T.get())
        except ValueError:
            T = float("nan")
        for i, m in enumerate(self.legs, 1):
            window = f"{m.k1 * T:g} ≤ t < {m.k2 * T:g}" if np.isfinite(T) else "?"
            self.legs_tree.insert("", "end", iid=str(i - 1),
                                  values=(i, f"{m.vx:g}", f"{m.vy:g}", m.k1, m.k2, window))

    def _read_leg_editor(self) -> Maneuver:
        vx, vy = (get_float(e, f"Leg {lab}") for e, lab in zip(self.f_leg.entries[:2], ("vξ", "vη")))
        k1 = get_int(self.f_leg.entries[2], "Leg k1", minimum=0)
        k2 = get_int(self.f_leg.entries[3], "Leg k2", minimum=k1 + 1)
        return Maneuver(vx, vy, k1, k2)

    def _selected_leg(self) -> int | None:
        sel = self.legs_tree.selection()
        return int(sel[0]) if sel else None

    def _on_leg_select(self, _event=None) -> None:
        i = self._selected_leg()
        if i is not None:
            m = self.legs[i]
            self.f_leg.set([m.vx, m.vy, m.k1, m.k2])

    def _try_set_legs(self, legs: list[Maneuver]) -> bool:
        """Accept ``legs`` if they do not overlap; otherwise explain why not."""
        try:
            validate_maneuvers(legs)
        except ValueError as exc:
            messagebox.showerror("Maneuver legs", str(exc), parent=self.root)
            return False
        self.legs = legs
        self._refresh_legs()
        return True

    def add_leg(self) -> None:
        """Add the leg typed in the editor row."""
        try:
            leg = self._read_leg_editor()
        except FieldError as exc:
            exc.focus()
            messagebox.showerror("Maneuver legs", str(exc), parent=self.root)
            return
        self._try_set_legs([*self.legs, leg])

    def update_leg(self) -> None:
        """Replace the selected leg by the editor values."""
        i = self._selected_leg()
        if i is None:
            messagebox.showinfo("Maneuver legs", "Select a leg in the table first.", parent=self.root)
            return
        try:
            leg = self._read_leg_editor()
        except FieldError as exc:
            exc.focus()
            messagebox.showerror("Maneuver legs", str(exc), parent=self.root)
            return
        legs = list(self.legs)
        legs[i] = leg
        self._try_set_legs(legs)

    def remove_leg(self) -> None:
        """Delete the selected leg."""
        i = self._selected_leg()
        if i is not None:
            self._try_set_legs([m for j, m in enumerate(self.legs) if j != i])

    def clear_legs(self) -> None:
        """Delete all legs (the target becomes unobservable)."""
        self._try_set_legs([])

    # ================================================================ running
    def _read_inputs(self) -> tuple[Scenario, int] | None:
        """Read and validate the form; show an error and return None if invalid."""
        try:
            return self.read_form(), self.read_mc_runs()
        except FieldError as exc:
            exc.focus()
            messagebox.showerror("Invalid input", str(exc), parent=self.root)
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc), parent=self.root)
        return None

    @staticmethod
    def compute(scenario: Scenario, mc_runs: int):
        """Run B1-B8 (and the Monte Carlo test if ``mc_runs > 0``)."""
        result = analysis.run_analysis(scenario)
        mc = None
        if mc_runs > 0:
            seed = None if scenario.seed is None else scenario.seed + 1
            mc = analysis.monte_carlo(scenario, mc_runs, np.random.default_rng(seed))
        return result, mc

    def run(self, background: bool = True) -> None:
        """Validate the inputs and run the analysis.

        With ``background=False`` the computation runs in the calling thread
        (used by the tests).
        """
        if self._busy:
            return
        inputs = self._read_inputs()
        if inputs is None:
            return
        scenario, mc_runs = inputs
        if not background:
            t0 = time.perf_counter()
            self.show_results(*self.compute(scenario, mc_runs), time.perf_counter() - t0)
            return
        self._set_busy(True, mc_runs)
        t0 = time.perf_counter()

        def worker():
            try:
                self._queue.put(("ok", self.compute(scenario, mc_runs), time.perf_counter() - t0))
            except Exception as exc:  # reported in the GUI thread
                self._queue.put(("error", exc, 0.0))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(50, self._poll)

    def _poll(self) -> None:
        try:
            kind, payload, elapsed = self._queue.get_nowait()
        except queue.Empty:
            self.root.after(50, self._poll)
            return
        self._set_busy(False)
        if kind == "ok":
            self.show_results(*payload, elapsed)
        else:
            self.set_status("Computation failed.")
            messagebox.showerror("Computation failed", str(payload), parent=self.root)

    def _set_busy(self, busy: bool, mc_runs: int = 0) -> None:
        self._busy = busy
        self.run_btn.configure(state="disabled" if busy else "normal")
        self.root.configure(cursor="watch" if busy else "")
        if busy:
            self.progress.pack(fill="x", pady=(8, 0))
            self.progress.start(12)
            extra = f" and {mc_runs} Monte Carlo runs" if mc_runs else ""
            self.set_status(f"Computing B1–B8{extra}…")
        else:
            self.progress.stop()
            self.progress.pack_forget()

    # ================================================================ results display
    def show_results(self, result: analysis.AnalysisResult,
                     mc: analysis.MonteCarloResult | None, elapsed: float = 0.0) -> None:
        """Fill every result tab and save the results for DBEAR."""
        self.result, self.mc = result, mc
        s = result.scenario
        self._show_summary(result, mc)
        self._show_estimate_table(result)
        common.set_text(self.texts["truth"], report.format_fim_analysis(result.truth, s.times, "the TRUE parameter"))
        common.set_text(self.texts["meas"], report.format_measurements(result))
        common.set_text(self.texts["est"], report.format_fim_analysis(result.estimate, s.times, "the ESTIMATE"))
        full = report.format_report(result)
        if mc is not None:
            full += "\n" + report.heading(f"MONTE CARLO CONSISTENCY CHECK ({mc.runs} runs)")
            full += "\n" + report.format_monte_carlo(mc)
        common.set_text(self.texts["report"], full)
        self._draw(s, result.mle.x, result.z)
        saved = self._save_for_dbear(result)
        self.dbear_btn.configure(state="normal")
        self.set_status(f"Done in {elapsed:.2f} s.  Results saved to {saved}")

    def _set_summary(self, key: str, text: str, style: str) -> None:
        self.summary[key].configure(text=text, style=f"{style}.TLabel")

    def _show_summary(self, res: analysis.AnalysisResult, mc) -> None:
        obs = res.truth.observability
        if obs.invertible:
            self._set_summary("obs", f"OBSERVABLE — FIM invertible (rank 4/4, scaled condition "
                                     f"number {obs.condition:.3g})", "Good")
        else:
            self._set_summary("obs", f"NOT OBSERVABLE — FIM singular (rank {obs.rank}/4): "
                                     "no CRLB, range undetermined", "Bad")

        m = res.mle
        if m.diverged:
            self._set_summary("mle", "FAILED — the range estimate ran off to infinity: no finite maximum "
                                     "likelihood (maneuver too weak for this noise)", "Bad")
        elif not m.converged:
            self._set_summary("mle", f"NOT converged after {m.iterations} iterations: {m.message}", "Bad")
        elif not res.estimate.observability.invertible:
            self._set_summary("mle", f"Converged in {m.iterations} iterations, but the minimum is "
                                     "not unique (FIM singular)", "Warn")
        else:
            self._set_summary("mle", f"Converged in {m.iterations} iterations ({m.method})", "Good")

        if res.fit is None:
            self._set_summary("fit", "n/a — needs more samples than estimable parameters", "Warn")
        else:
            f = res.fit
            verdict = "within" if f.passed else "OUTSIDE"
            self._set_summary("fit", f"{f.value:.2f}  (dof {f.dof})  — {verdict} the {100 * f.confidence:g}% "
                                     f"interval [{f.lower:.2f}, {f.upper:.2f}]", "Good" if f.passed else "Bad")

        t = res.nees
        verdict = "within" if t.passed else "OUTSIDE"
        if not res.nees_valid:
            self._set_summary("nees", f"{t.value:.3f}  — not meaningful: FIM at the estimate is singular "
                                      "(unobservable)", "Warn")
        else:
            self._set_summary("nees", f"{t.value:.3f}  (dof {t.dof})  — {verdict} the {100 * t.confidence:g}% "
                                      f"interval [{t.lower:.3f}, {t.upper:.3f}]", "Good" if t.passed else "Bad")

        if mc is None:
            self.summary["mc"].configure(text="not run (set Monte Carlo runs > 0)", style="TLabel")
        elif mc.avg_nees is None:
            self._set_summary("mc", f"{mc.runs} runs: NEES not meaningful (no run with an invertible FIM "
                                    "at its estimate)", "Warn")
        else:
            a = mc.avg_nees
            n_ok = int(mc.valid.sum())
            verdict = "within" if a.passed else "OUTSIDE"
            n_div = int(mc.diverged.sum())
            div = f" ({n_div} diverged)" if n_div else ""
            self._set_summary("mc", f"average NEES {a.value:.3f} over {n_ok}/{mc.runs} runs{div} — {verdict} "
                                    f"[{a.lower:.3f}, {a.upper:.3f}];  average fit error "
                                    f"{mc.fit_errors[mc.converged].mean():.2f}", "Good" if a.passed else "Bad")

    def _show_estimate_table(self, res: analysis.AnalysisResult) -> None:
        self.est_tree.delete(*self.est_tree.get_children())
        s = res.scenario
        P = res.estimate.P
        std = np.sqrt(np.diag(P)) if P is not None else np.full(4, np.nan)
        for i, name in enumerate(PARAM_LABELS):
            err = res.error[i]
            ratio = abs(err) / std[i] if np.isfinite(std[i]) and std[i] > 0 else np.nan
            self.est_tree.insert("", "end", values=(
                name, f"{s.x_init[i]:.6g}", f"{res.mle.x[i]:.6g}", f"{s.x_true[i]:.6g}",
                f"{err:.4g}", f"{std[i]:.4g}", f"{ratio:.2f}"))

    def _draw(self, scenario: Scenario, x_hat=None, z=None) -> None:
        self.fig.clear()
        ax = self.fig.add_subplot()
        plot_geometry(scenario, x_hat=x_hat, z=z, ax=ax)
        self.canvas.draw_idle()

    def preview(self) -> None:
        """Draw the geometry of the current form without estimating."""
        try:
            scenario = self.read_form()
        except ValueError:
            return  # incomplete form: keep the previous picture
        self._draw(scenario)

    def _save_for_dbear(self, result: analysis.AnalysisResult) -> Path:
        """Save results where DBEAR looks for them (fallback: temp folder)."""
        for path in (self.results_path, common.fallback_results_path()):
            try:
                analysis.save_results(result, path)
                self.results_path = path
                return path
            except OSError:
                continue
        raise OSError("could not save the results file")

    # ================================================================ menu actions
    def show_help(self, key: str = "overview") -> None:
        """Open the Help window (F1) on topic ``key``."""
        open_help(self.root, BEAR_TOPICS, key, "BEAR help")

    def set_status(self, text: str) -> None:
        """Show a message in the status bar."""
        self.status.set(text)

    def open_defaults(self) -> None:
        """Load another defaults (.DEF) file into the form."""
        path = filedialog.askopenfilename(
            parent=self.root, title="Open defaults file",
            initialdir=str(self.def_path.parent if self.def_path else common.app_dir()),
            filetypes=[("BEAR defaults", "*.DEF *.def"), ("All files", "*.*")])
        if not path:
            return
        try:
            scenario = load_defaults(path)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Open defaults", str(exc), parent=self.root)
            return
        self.defaults, self.def_path = scenario, Path(path)
        self.fill_form(scenario)
        self.preview()
        self.set_status(f"Defaults loaded from {path}")

    def save_inputs(self) -> None:
        """Save the current form as a defaults (.DEF) file."""
        inputs = self._read_inputs()
        if inputs is None:
            return
        path = filedialog.asksaveasfilename(
            parent=self.root, title="Save inputs as defaults file", defaultextension=".DEF",
            initialdir=str(common.app_dir()), initialfile="BEAR.DEF",
            filetypes=[("BEAR defaults", "*.DEF *.def"), ("All files", "*.*")])
        if path:
            save_defaults(inputs[0], path)
            self.set_status(f"Inputs saved to {path}")

    def save_results_as(self) -> None:
        """Save the last results as a JSON file (for DBEAR)."""
        if self.result is None:
            messagebox.showinfo("Save results", "Run the analysis first.", parent=self.root)
            return
        path = filedialog.asksaveasfilename(
            parent=self.root, title="Save results", defaultextension=".json",
            initialfile="bear_results.json", filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if path:
            analysis.save_results(self.result, path)
            self.set_status(f"Results saved to {path}")

    def save_report(self) -> None:
        """Save the full text report."""
        if self.result is None:
            messagebox.showinfo("Save report", "Run the analysis first.", parent=self.root)
            return
        path = filedialog.asksaveasfilename(
            parent=self.root, title="Save report", defaultextension=".txt",
            initialfile="bear_report.txt", filetypes=[("Text", "*.txt"), ("All files", "*.*")])
        if path:
            Path(path).write_text(self.texts["report"].get("1.0", "end"), encoding="utf-8")
            self.set_status(f"Report saved to {path}")

    def open_dbear(self) -> None:
        """Open DBEAR on the results of the last run."""
        if self.result is None:
            messagebox.showinfo("DBEAR", "Run the analysis first.", parent=self.root)
            return
        try:
            common.launch_program("DBEAR", "dbear_gui.py", [str(self.results_path)])
            self.set_status(f"DBEAR started on {self.results_path}")
        except (OSError, FileNotFoundError) as exc:
            messagebox.showerror("DBEAR", str(exc), parent=self.root)

    def _about(self) -> None:
        messagebox.showinfo(
            "About BEAR",
            "BEAR — bearings-only target localization\n\n"
            "Project 3.8.3 of Bar-Shalom, Li & Kirubarajan, Estimation with Applications "
            "to Tracking and Navigation.\n\n"
            "B1 Fisher information & observability · B2 CRLB · B3 position covariances · "
            "B4 noisy bearings · B5 maximum likelihood estimate · B6 B1–B3 at the estimate · "
            "B7 fitting error · B8 NEES.\n\n"
            f"Application folder: {common.app_dir()}",
            parent=self.root)


def selftest(out: Path) -> int:
    """Non-interactive check of a build: run B1-B8 in a hidden window.

    Uses the start-up defaults with 100 Monte Carlo runs, saves the results to
    ``out`` (JSON) and the full report next to it (``.txt``).  Returns 0 if
    the estimation converged and the report was produced, else 1.
    """
    common.close_splash()
    root = tk.Tk()
    root.withdraw()
    try:
        defaults, def_path, _ = common.load_startup_defaults()
        app = BearApp(root, defaults, def_path, out)
        set_entry(app.mc_runs, "100")
        app.run(background=False)
        ok = app.result is not None and app.result.mle.converged and app.mc is not None
        if app.result is not None:
            out.with_suffix(".txt").write_text(app.texts["report"].get("1.0", "end"), encoding="utf-8")
        root.update()
    finally:
        root.destroy()
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    """Start the BEAR window (``--selftest [out.json]`` runs the build check instead)."""
    args = sys.argv[1:] if argv is None else argv
    if args and args[0] == "--selftest":
        out = Path(args[1]) if len(args) > 1 else common.app_dir() / "selftest_results.json"
        return selftest(out)
    common.enable_high_dpi()
    root = tk.Tk()
    common.setup_window(root, "BEAR — Bearings-only target localization", 1450, 920)
    defaults, def_path, warning = common.load_startup_defaults()
    BearApp(root, defaults, def_path, common.default_results_path())
    root.after(100, common.close_splash)
    if warning:
        root.after(200, lambda: messagebox.showwarning("BEAR.DEF", warning, parent=root))
    root.mainloop()
    return 0
