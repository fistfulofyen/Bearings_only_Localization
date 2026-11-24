"""DBEAR -- display the target position uncertainty ellipses computed by BEAR.

Implements DBEAR.EXE of the project in Section 3.8.3 of Bar-Shalom, Li &
Kirubarajan.  It reads the results file written by ``bear.py`` (by default
``bear_results.json`` in the project folder, next to this script) and draws, at
user-specified times, the g-sigma ellipses

    (p - p_hat(t))' C(t)^{-1} (p - p_hat(t)) = g^2

centred at the estimated target position ``p_hat(t) = A(t) x_hat``, with
``C(t) = A(t) P A(t)'`` and ``P`` the inverse FIM at the estimate (step B6).
Times need not be sampling times (``A(t)`` is defined for any ``t``), so the
uncertainty can also be extrapolated beyond the observation window.

The ellipse construction is documented in :mod:`bearloc.ellipse` and in
``docs/algorithms.md``.

Usage::

    python dbear.py                          # interactive: asks for times and g
    python dbear.py run1.json --times 10 200 400 600 -g 2
    python dbear.py --batch --save ellipses.png --no-show
    python dbear.py --source truth --rays    # use the B3 covariance, draw bearings
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, TextIO

import numpy as np

from bearloc import analysis
from bearloc.ellipse import ellipse_probability
from bearloc.plotting import SOURCES, default_times, ellipse_specs, format_ellipse_table
from bearloc.scenario import parse_vector

#: Default results file: the one bear.py writes by default (next to this script).
DEFAULT_RESULTS = Path(__file__).resolve().with_name("bear_results.json")

InputFn = Callable[[str], str]


def _ask(prompt: str, default, parse: Callable, show: Callable, input_fn: InputFn, out: TextIO):
    """Ask until the answer parses; empty input (or EOF) keeps ``default``."""
    while True:
        try:
            text = input_fn(f"{prompt} [{show(default)}]: ")
        except EOFError:
            return default
        if not text.strip():
            return default
        try:
            return parse(text)
        except ValueError as exc:
            print(f"    invalid input: {exc}. Please try again.", file=out)


def parse_times(text: str) -> list[float]:
    """Parse a list of non-negative times."""
    times = parse_vector(text)
    if times.size == 0:
        raise ValueError("enter at least one time")
    if np.any(times < 0):
        raise ValueError("times must be >= 0")
    return [float(t) for t in times]


def parse_sigmas(text: str) -> float:
    """Parse the ellipse size g (> 0)."""
    g = float(text)
    if g <= 0:
        raise ValueError("g must be positive")
    return g


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface definition."""
    p = argparse.ArgumentParser(
        prog="dbear",
        description="Plot target position uncertainty ellipses from BEAR results.",
    )
    p.add_argument("results", nargs="?", type=Path, default=DEFAULT_RESULTS,
                   help="results file written by bear.py (default: bear_results.json in the project folder)")
    p.add_argument("--times", type=float, nargs="+", metavar="T",
                   help="times at which to draw ellipses (default: 5 evenly spaced sampling times)")
    p.add_argument("-g", "--sigmas", type=float, metavar="G",
                   help="ellipse size in sigmas (default: 2)")
    p.add_argument("--source", choices=sorted(SOURCES), default="estimate",
                   help="covariance from the estimate (B6, default) or the truth (B3)")
    p.add_argument("--rays", action="store_true", help="draw the measured bearings at those times")
    p.add_argument("--zoom", action="store_true", help="zoom the view on the ellipses")
    p.add_argument("-b", "--batch", action="store_true", help="do not ask; use options/defaults")
    p.add_argument("--save", type=Path, metavar="FILE", help="save the figure (PNG, PDF, SVG, ...)")
    p.add_argument("--no-show", action="store_true", help="do not open a plot window")
    return p


def main(argv: list[str] | None = None, input_fn: InputFn = input, out: TextIO = sys.stdout) -> int:
    """Run DBEAR.  Returns the process exit code (0 = success, 2 = input error)."""
    args = build_parser().parse_args(argv)
    try:
        results = analysis.load_results(args.results)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    times = args.times if args.times else default_times(results)
    g = args.sigmas if args.sigmas else 2.0
    if g <= 0:
        print("error: -g must be positive", file=sys.stderr)
        return 2
    if not args.batch:
        print("DBEAR -- target position uncertainty ellipses", file=out)
        print("Press Enter to keep the default in brackets.\n", file=out)
        show_list = lambda v: " ".join(f"{t:g}" for t in v)  # noqa: E731
        times = _ask("Times at which to draw ellipses", times, parse_times, show_list, input_fn, out)
        g = _ask("Ellipse size g (number of sigmas)", g, parse_sigmas, lambda v: f"{v:g}", input_fn, out)

    try:
        specs = ellipse_specs(results, times, g, args.source)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"\n{g:g}-sigma ellipses (probability content {100 * ellipse_probability(g):.1f}%), "
          f"covariance from {SOURCES[args.source]}:", file=out)
    print(format_ellipse_table(specs), file=out)

    import matplotlib

    if args.no_show:
        matplotlib.use("Agg")  # no window needed
    import matplotlib.pyplot as plt

    from bearloc.plotting import plot_ellipses

    fig = plot_ellipses(results, specs, source=args.source, show_rays=args.rays, zoom=args.zoom)
    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Figure saved to {args.save}", file=out)
    if not args.no_show:
        plt.show()
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
