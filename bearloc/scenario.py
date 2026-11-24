"""Scenario definition (inputs A1-A7) and the ``BEAR.DEF`` defaults file.

A :class:`Scenario` bundles every user-specified input of the BEAR program:

====  =========================================================  ==================
Item  Meaning                                                    Field
====  =========================================================  ==================
A1    True target parameter at ``t = 0``                         ``x_true``
A2    Sampling period ``T``                                      ``T``
A3    Number of samples ``n``                                    ``n``
A4    Measurement noise variance ``r`` (rad^2)                   ``r``
A5    Platform initial position and velocity                     ``platform_pos``,
                                                                 ``platform_vel``
A6    Platform maneuver legs (velocity, ``k1``, ``k2``)          ``maneuvers``
A7    Initial estimate for the minimization                      ``x_init``
====  =========================================================  ==================

plus a few run-control settings (random seed, optimizer, confidence level).

``BEAR.DEF`` file format
------------------------
An INI file (parsed with :mod:`configparser`); ``#`` and ``;`` start comments
and vectors are comma- or whitespace-separated numbers.  Every maneuver leg is
its own section whose name starts with ``maneuver``::

    [target]
    x_true = 5000, 12000, -6, -2

    [sampling]
    T = 10
    n = 60

    [noise]
    r = 3.0462e-4
    seed = 1

    [platform]
    position = 0, 0
    velocity = 0, 15

    [maneuver 1]
    velocity = 15, 0
    k1 = 20
    k2 = 40

    [estimation]
    x_init = 3000, 9000, 0, 0
    method = gauss-newton
    max_iter = 200
    tol = 1e-10

    [statistics]
    confidence = 0.95
"""

from __future__ import annotations

import configparser
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from . import model
from .model import Maneuver

#: Optimizers accepted by :func:`bearloc.estimation.maximum_likelihood`.
METHODS = ("gauss-newton", "bfgs")


def parse_vector(text: str, length: int | None = None) -> np.ndarray:
    """Parse ``"1, 2 3"`` (commas and/or whitespace, optional brackets) to floats.

    Parameters
    ----------
    text : str
        Text to parse.
    length : int, optional
        Required number of elements.

    Raises
    ------
    ValueError
        If the text is not a list of numbers or has the wrong length.
    """
    tokens = [tok for tok in re.split(r"[,\s]+", text.strip().strip("[]()")) if tok]
    try:
        values = np.array([float(tok) for tok in tokens], dtype=float)
    except ValueError as exc:
        raise ValueError(f"cannot parse {text!r} as a list of numbers") from exc
    if length is not None and values.size != length:
        raise ValueError(f"expected {length} numbers, got {values.size} in {text!r}")
    return values


def format_vector(v: Sequence[float]) -> str:
    """Format a vector as comma-separated numbers (inverse of :func:`parse_vector`).

    Uses the shortest representation that reads back to exactly the same
    float, so ``parse_vector(format_vector(v))`` reproduces ``v`` exactly.
    """
    return ", ".join(_num(a) for a in v)


def _num(a: float) -> str:
    """Shortest exact text for a float, without a trailing ``.0``."""
    text = repr(float(a))
    return text[:-2] if text.endswith(".0") else text


@dataclass
class Scenario:
    """All inputs of a bearings-only localization run.

    Units are arbitrary but must be consistent (the defaults use metres,
    seconds and radians).
    """

    x_true: np.ndarray                      # A1
    T: float                                # A2
    n: int                                  # A3
    r: float                                # A4
    platform_pos: np.ndarray                # A5
    platform_vel: np.ndarray                # A5
    maneuvers: list[Maneuver] = field(default_factory=list)  # A6
    x_init: np.ndarray = field(default_factory=lambda: np.zeros(4))  # A7
    seed: int | None = None
    method: str = "gauss-newton"
    max_iter: int = 200
    tol: float = 1e-10
    confidence: float = 0.95

    def __post_init__(self) -> None:
        self.x_true = np.asarray(self.x_true, dtype=float)
        self.platform_pos = np.asarray(self.platform_pos, dtype=float)
        self.platform_vel = np.asarray(self.platform_vel, dtype=float)
        self.x_init = np.asarray(self.x_init, dtype=float)
        self.maneuvers = list(self.maneuvers)
        self.n = int(self.n)
        self.T = float(self.T)
        self.r = float(self.r)
        self.validate()

    # ------------------------------------------------------------------ checks
    def validate(self) -> None:
        """Raise :class:`ValueError` if any input is inconsistent."""
        if self.x_true.shape != (model.N_PARAMS,):
            raise ValueError("x_true must have 4 elements [xi0, eta0, xi_dot, eta_dot]")
        if self.x_init.shape != (model.N_PARAMS,):
            raise ValueError("x_init must have 4 elements [xi0, eta0, xi_dot, eta_dot]")
        if self.platform_pos.shape != (2,) or self.platform_vel.shape != (2,):
            raise ValueError("platform position and velocity must have 2 elements")
        if self.T <= 0:
            raise ValueError(f"sampling period T must be positive, got {self.T}")
        if self.n < 1:
            raise ValueError(f"number of samples n must be >= 1, got {self.n}")
        if self.r <= 0:
            raise ValueError(f"noise variance r must be positive, got {self.r}")
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
        if not 0 < self.confidence < 1:
            raise ValueError(f"confidence must be in (0, 1), got {self.confidence}")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}")
        model.validate_maneuvers(self.maneuvers)

    # -------------------------------------------------------------- geometry
    @property
    def times(self) -> np.ndarray:
        """Measurement times ``t_k = k T``, ``k = 1..n``."""
        return model.measurement_times(self.T, self.n)

    def platform_at(self, t: np.ndarray | float) -> np.ndarray:
        """Platform position at arbitrary time(s) ``t``."""
        return model.platform_position(
            t, self.platform_pos, self.platform_vel, self.maneuvers, self.T
        )

    @property
    def platform_track(self) -> np.ndarray:
        """Platform positions at the measurement times, shape ``(n, 2)``."""
        return self.platform_at(self.times)

    def simulate_measurements(self, rng: np.random.Generator) -> np.ndarray:
        """Generate noisy bearings ``z(k) = h(k, x_true) + w(k)`` (step B4).

        The result is wrapped into ``(-pi, pi]``.
        """
        h = model.bearings(self.x_true, self.times, self.platform_track)
        w = rng.normal(0.0, np.sqrt(self.r), size=self.n)
        return model.wrap_angle(h + w)

    def copy(self, **changes) -> "Scenario":
        """Return a copy with some fields replaced (validated again)."""
        return replace(self, **changes)

    # ----------------------------------------------------------- serialization
    def to_dict(self) -> dict:
        """JSON-friendly representation."""
        return {
            "x_true": self.x_true.tolist(),
            "T": self.T,
            "n": self.n,
            "r": self.r,
            "platform_pos": self.platform_pos.tolist(),
            "platform_vel": self.platform_vel.tolist(),
            "maneuvers": [
                {"vx": m.vx, "vy": m.vy, "k1": m.k1, "k2": m.k2} for m in self.maneuvers
            ],
            "x_init": self.x_init.tolist(),
            "seed": self.seed,
            "method": self.method,
            "max_iter": self.max_iter,
            "tol": self.tol,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        """Inverse of :meth:`to_dict`."""
        d = dict(d)
        d["maneuvers"] = [Maneuver(**m) for m in d.get("maneuvers", [])]
        return cls(**d)


# ---------------------------------------------------------------- BEAR.DEF I/O
def _maneuver_sections(cfg: configparser.ConfigParser) -> list[str]:
    """Section names describing maneuver legs, in file order."""
    return [s for s in cfg.sections() if s.lower().startswith("maneuver")]


def load_defaults(path: str | Path) -> Scenario:
    """Read a scenario from a ``BEAR.DEF`` file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If a required entry is missing or malformed.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"defaults file not found: {path}")
    cfg = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    cfg.optionxform = str  # keep key case (T vs t)
    cfg.read(path, encoding="utf-8")

    def get(section: str, key: str, fallback: str | None = None) -> str:
        if cfg.has_option(section, key):
            return cfg.get(section, key)
        if fallback is not None:
            return fallback
        raise ValueError(f"{path}: missing '{key}' in section [{section}]")

    try:
        maneuvers = []
        for s in _maneuver_sections(cfg):
            vx, vy = parse_vector(get(s, "velocity"), 2)
            maneuvers.append(
                Maneuver(float(vx), float(vy), k1=int(get(s, "k1")), k2=int(get(s, "k2")))
            )
        seed_text = get("noise", "seed", "").strip()
        return Scenario(
            x_true=parse_vector(get("target", "x_true"), 4),
            T=float(get("sampling", "T")),
            n=int(get("sampling", "n")),
            r=float(get("noise", "r")),
            platform_pos=parse_vector(get("platform", "position"), 2),
            platform_vel=parse_vector(get("platform", "velocity"), 2),
            maneuvers=maneuvers,
            x_init=parse_vector(get("estimation", "x_init"), 4),
            seed=int(seed_text) if seed_text else None,
            method=get("estimation", "method", "gauss-newton").strip().lower(),
            max_iter=int(get("estimation", "max_iter", "200")),
            tol=float(get("estimation", "tol", "1e-10")),
            confidence=float(get("statistics", "confidence", "0.95")),
        )
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def save_defaults(scenario: Scenario, path: str | Path) -> None:
    """Write a scenario to a ``BEAR.DEF`` file readable by :func:`load_defaults`."""
    s = scenario
    lines = [
        "# BEAR.DEF -- default inputs for BEAR (bearings-only target localization)",
        "# Units: metres, seconds, radians.  Vectors: comma-separated numbers.",
        "",
        "[target]",
        "# A1: true parameter at t = 0: xi0, eta0, xi_dot, eta_dot",
        f"x_true = {format_vector(s.x_true)}",
        "",
        "[sampling]",
        "# A2: sampling period T;  A3: number of samples n (t_k = k T, k = 1..n)",
        f"T = {_num(s.T)}",
        f"n = {s.n}",
        "",
        "[noise]",
        "# A4: bearing noise variance r (rad^2);  seed: blank = random",
        f"r = {_num(s.r)}",
        f"seed = {'' if s.seed is None else s.seed}",
        "",
        "[platform]",
        "# A5: platform initial position and velocity",
        f"position = {format_vector(s.platform_pos)}",
        f"velocity = {format_vector(s.platform_vel)}",
    ]
    for i, m in enumerate(s.maneuvers, start=1):
        lines += [
            "",
            f"[maneuver {i}]",
            "# A6: velocity in effect for k1*T <= t < k2*T",
            f"velocity = {format_vector([m.vx, m.vy])}",
            f"k1 = {m.k1}",
            f"k2 = {m.k2}",
        ]
    lines += [
        "",
        "[estimation]",
        "# A7: initial estimate; method = gauss-newton | bfgs",
        f"x_init = {format_vector(s.x_init)}",
        f"method = {s.method}",
        f"max_iter = {s.max_iter}",
        f"tol = {_num(s.tol)}",
        "",
        "[statistics]",
        "# probability level for the chi-square acceptance intervals",
        f"confidence = {_num(s.confidence)}",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
