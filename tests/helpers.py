"""Shared fixtures for the test-suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bearloc.model import Maneuver
from bearloc.scenario import Scenario

#: Project root (where BEAR.DEF, bear.py and dbear.py live).
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEF = ROOT / "BEAR.DEF"

#: One degree of bearing noise, as a variance in rad^2.
R_1DEG = float(np.radians(1.0) ** 2)


def observable_scenario(**changes) -> Scenario:
    """Maneuvering platform: the target parameter is observable."""
    s = Scenario(
        x_true=[5000.0, 12000.0, -6.0, -2.0],
        T=10.0,
        n=60,
        r=R_1DEG,
        platform_pos=[0.0, 0.0],
        platform_vel=[0.0, 15.0],
        maneuvers=[Maneuver(15.0, 0.0, 20, 40)],
        x_init=[2000.0, 8000.0, 0.0, 0.0],
        seed=1,
    )
    return s.copy(**changes) if changes else s


def unobservable_scenario(**changes) -> Scenario:
    """Constant-velocity platform: the target parameter is NOT observable."""
    return observable_scenario(maneuvers=[], **changes)


class ScriptedInput:
    """Replacement for :func:`input` that answers prompts from a script.

    ``responses`` maps a prompt prefix (e.g. ``"A3"``) to the list of answers
    to give, in order, the successive times a prompt with that prefix is
    shown.  Any other prompt (or an exhausted list) gets an empty answer,
    i.e. "keep the default".  All prompts are recorded in ``prompts``.
    """

    def __init__(self, responses: dict[str, list[str]] | None = None):
        self.responses = {k: list(v) for k, v in (responses or {}).items()}
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        for prefix, answers in self.responses.items():
            if prompt.startswith(prefix) and answers:
                return answers.pop(0)
        return ""
