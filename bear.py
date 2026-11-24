"""BEAR -- interactive program for bearings-only target localization.

Implements BEAR.EXE of the project in Section 3.8.3 of Bar-Shalom, Li &
Kirubarajan, *Estimation with Applications to Tracking and Navigation*.

The program

1. reads default inputs A1-A7 from ``BEAR.DEF``;
2. lets the user confirm or change every input (press Enter to keep the
   default shown in brackets);
3. carries out steps B1-B8 and prints the results;
4. writes all results to a JSON file that the DBEAR program plots (by
   default ``bear_results.json`` in the project folder, next to this script,
   whatever the current directory is).

Usage::

    python bear.py                       # interactive, defaults from BEAR.DEF
    python bear.py --batch               # no questions, use BEAR.DEF as is
    python bear.py --def my.def -o run1.json --every 5
    python bear.py --batch --mc 500      # add a 500-run Monte Carlo NEES check
    python bear.py --save-def new.def    # also store the inputs as new defaults

Input syntax: vectors are numbers separated by commas and/or spaces.  The
noise variance accepts a suffix ``deg`` meaning "standard deviation in
degrees", e.g. ``0.5deg`` for ``r = (0.5 pi/180)^2``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, TextIO

import numpy as np

from bearloc import analysis, report
from bearloc.model import Maneuver
from bearloc.scenario import METHODS, Scenario, format_vector, load_defaults, parse_vector, save_defaults

#: Default defaults file: BEAR.DEF next to this script.
DEFAULT_DEF = Path(__file__).resolve().with_name("BEAR.DEF")
#: Default results file: bear_results.json next to this script (read by DBEAR).
DEFAULT_OUTPUT = Path(__file__).resolve().with_name("bear_results.json")

InputFn = Callable[[str], str]


# ------------------------------------------------------------------ parsing
def parse_variance(text: str) -> float:
    """Parse the noise variance: ``"3e-4"`` (rad^2) or ``"1deg"`` (std in degrees)."""
    t = text.strip().lower()
    if t.endswith("deg"):
        sigma = float(t[:-3])
        if sigma <= 0:
            raise ValueError("standard deviation must be positive")
        return float(np.radians(sigma) ** 2)
    r = float(t)
    if r <= 0:
        raise ValueError("variance must be positive")
    return r


def parse_positive_float(text: str) -> float:
    """Parse a strictly positive float."""
    v = float(text)
    if v <= 0:
        raise ValueError("value must be positive")
    return v


def parse_int(minimum: int) -> Callable[[str], int]:
    """Return a parser for integers ``>= minimum``."""

    def parse(text: str) -> int:
        v = int(text.strip())
        if v < minimum:
            raise ValueError(f"value must be >= {minimum}")
        return v

    return parse


def parse_seed(text: str) -> int | None:
    """Parse a random seed; ``none`` or ``random`` means a random seed."""
    t = text.strip().lower()
    return None if t in ("none", "random") else int(t)


def parse_method(text: str) -> str:
    """Parse the minimization method name."""
    t = text.strip().lower()
    if t not in METHODS:
        raise ValueError(f"choose one of {', '.join(METHODS)}")
    return t


# ------------------------------------------------------------------ prompting
def ask(prompt: str, default, parse: Callable, show: Callable, input_fn: InputFn, out: TextIO):
    """Ask for one value until it parses; empty input (or EOF) keeps ``default``."""
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


def interactive_scenario(defaults: Scenario, input_fn: InputFn = input, out: TextIO = sys.stdout) -> Scenario:
    """Let the user review and edit inputs A1-A7, starting from ``defaults``.

    If the resulting combination is inconsistent (e.g. overlapping maneuver
    legs) the error is shown and the questions are asked again, with the
    values just entered offered as defaults.
    """
    vec = format_vector
    num = lambda v: f"{v:g}"  # noqa: E731 - tiny display helper
    d = defaults
    while True:
        print("\nEnter new values or press Enter to keep the default in brackets.\n", file=out)
        x_true = ask("A1  True target parameter xi0, eta0, xi_dot, eta_dot", d.x_true,
                     lambda s: parse_vector(s, 4), vec, input_fn, out)
        T = ask("A2  Sampling period T", d.T, parse_positive_float, num, input_fn, out)
        n = ask("A3  Number of samples n", d.n, parse_int(1), str, input_fn, out)
        r = ask(f"A4  Noise variance r in rad^2, or e.g. '1deg' (now sigma = "
                f"{np.degrees(np.sqrt(d.r)):.3g} deg)", d.r, parse_variance, lambda v: f"{v:.6g}",
                input_fn, out)
        pos = ask("A5  Platform initial position", d.platform_pos,
                  lambda s: parse_vector(s, 2), vec, input_fn, out)
        vel = ask("A5  Platform initial velocity", d.platform_vel,
                  lambda s: parse_vector(s, 2), vec, input_fn, out)
        n_legs = ask("A6  Number of platform maneuver legs", len(d.maneuvers), parse_int(0), str,
                     input_fn, out)
        legs = []
        for i in range(n_legs):
            if i < len(d.maneuvers):
                dm = d.maneuvers[i]
            else:  # sensible default for an extra leg: right after the previous one
                start = legs[-1].k2 if legs else n // 3
                dm = Maneuver(float(vel[1]), float(-vel[0]), start, start + max(1, n // 6))
            v = ask(f"A6  Leg {i + 1}: velocity vx, vy", np.array([dm.vx, dm.vy]),
                    lambda s: parse_vector(s, 2), vec, input_fn, out)
            k1 = ask(f"A6  Leg {i + 1}: start sample k1", dm.k1, parse_int(0), str, input_fn, out)
            k2 = ask(f"A6  Leg {i + 1}: end sample k2", max(dm.k2, k1 + 1), parse_int(k1 + 1), str,
                     input_fn, out)
            legs.append(Maneuver(float(v[0]), float(v[1]), k1, k2))
        x_init = ask("A7  Initial estimate xi0, eta0, xi_dot, eta_dot", d.x_init,
                     lambda s: parse_vector(s, 4), vec, input_fn, out)
        method = ask(f"    Minimization method ({' / '.join(METHODS)})", d.method, parse_method, str,
                     input_fn, out)
        seed = ask("    Random seed ('random' for none)", d.seed, parse_seed,
                   lambda v: "random" if v is None else str(v), input_fn, out)
        try:
            return d.copy(x_true=x_true, T=T, n=n, r=r, platform_pos=pos, platform_vel=vel,
                          maneuvers=legs, x_init=x_init, method=method, seed=seed)
        except ValueError as exc:
            print(f"\n*** Inconsistent input: {exc}. Please re-enter.", file=out)
            # keep what was typed as the new defaults (without re-validating)
            d = d.copy()
            d.x_true, d.T, d.n, d.r = np.asarray(x_true), T, n, r
            d.platform_pos, d.platform_vel, d.x_init = np.asarray(pos), np.asarray(vel), np.asarray(x_init)
            d.method, d.seed, d.maneuvers = method, seed, legs


# ------------------------------------------------------------------ main
def build_parser() -> argparse.ArgumentParser:
    """Command-line interface definition."""
    p = argparse.ArgumentParser(
        prog="bear",
        description="Bearings-only target localization: FIM, CRLB, MLE, fitting error and NEES.",
    )
    p.add_argument("--def", dest="defaults", type=Path, default=DEFAULT_DEF,
                   help="defaults file (default: BEAR.DEF next to this script)")
    p.add_argument("-b", "--batch", action="store_true",
                   help="non-interactive: use the defaults file without asking")
    p.add_argument("-o", "--output", type=Path, default=DEFAULT_OUTPUT,
                   help="results file for DBEAR (default: bear_results.json in the project folder)")
    p.add_argument("--save-def", type=Path, metavar="PATH",
                   help="save the inputs used in this run as a defaults file")
    p.add_argument("--every", type=int, default=1, metavar="M",
                   help="print only every M-th row of the per-sample tables (default: 1)")
    p.add_argument("--mc", type=int, default=0, metavar="N",
                   help="also run N Monte Carlo runs and test the average NEES")
    return p


def main(argv: list[str] | None = None, input_fn: InputFn = input, out: TextIO = sys.stdout) -> int:
    """Run BEAR.  Returns the process exit code (0 = success, 2 = input error)."""
    args = build_parser().parse_args(argv)
    if args.every < 1:
        print("error: --every must be >= 1", file=sys.stderr)
        return 2
    try:
        defaults = load_defaults(args.defaults)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print("BEAR -- bearings-only target localization", file=out)
    print(f"Defaults read from {args.defaults}", file=out)
    scenario = defaults if args.batch else interactive_scenario(defaults, input_fn, out)
    if args.save_def:
        save_defaults(scenario, args.save_def)
        print(f"Inputs saved to {args.save_def}", file=out)

    result = analysis.run_analysis(scenario)
    print(report.format_report(result, every=args.every), file=out)

    if args.mc > 0:
        print(report.heading(f"MONTE CARLO CONSISTENCY CHECK ({args.mc} runs)"), file=out)
        mc = analysis.monte_carlo(scenario, args.mc,
                                  np.random.default_rng(None if scenario.seed is None else scenario.seed + 1))
        print(report.format_monte_carlo(mc), file=out)

    analysis.save_results(result, args.output)
    output = args.output.resolve()
    plot_cmd = "python dbear.py" if output == DEFAULT_OUTPUT else f'python dbear.py "{output}"'
    print(f"\nResults written to {output}\n(plot them with: {plot_cmd})", file=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
