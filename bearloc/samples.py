"""Ready-made sample scenarios, each illustrating one aspect of the problem.

They are offered in the "Load sample" drop-down of the BEAR window and can be
used directly from Python::

    from bearloc.samples import sample_scenario
    s = sample_scenario("Zig-zag, two maneuver legs")

All samples use metres, seconds and radians, seed 1 and the Gauss-Newton
optimizer, so their results are reproducible.
"""

from __future__ import annotations

import numpy as np

from .model import Maneuver
from .scenario import Scenario


def _var_deg(sigma_deg: float) -> float:
    """Noise variance (rad^2) for a standard deviation given in degrees."""
    return float(np.radians(sigma_deg) ** 2)


#: name -> (description, Scenario keyword arguments)
SAMPLES: dict[str, tuple[str, dict]] = {
    "Close target, low noise": (
        "Easy case: close target, 0.5 deg noise, clear maneuver. Small ellipses.",
        dict(x_true=[2000, 6000, 4, -3], T=5, n=80, r=_var_deg(0.5),
             platform_pos=[0, 0], platform_vel=[10, 0],
             maneuvers=[Maneuver(0, 10, 30, 50)], x_init=[1000, 4000, 0, 0]),
    ),
    "Zig-zag, two maneuver legs": (
        "Platform turns east, north again, then west. Two legs of maneuver.",
        dict(x_true=[-8000, 15000, 7, -5], T=10, n=90, r=_var_deg(1.0),
             platform_pos=[0, 0], platform_vel=[0, 12],
             maneuvers=[Maneuver(12, 0, 25, 45), Maneuver(-12, 0, 60, 80)],
             x_init=[-4000, 10000, 0, 0]),
    ),
    "Target due west (bearing crosses 180 deg)": (
        "Bearings jump from +180 to -180 deg; tests the residual wrapping.",
        dict(x_true=[-12000, 3000, 0, -8], T=10, n=60, r=_var_deg(1.0),
             platform_pos=[0, 0], platform_vel=[0, 5],
             maneuvers=[Maneuver(-10, 0, 20, 40)], x_init=[-9000, 0, 0, 0]),
    ),
    "No maneuver (unobservable)": (
        "Platform never maneuvers: the FIM is singular and the range is undetermined.",
        dict(x_true=[5000, 12000, -6, -2], T=10, n=60, r=_var_deg(1.0),
             platform_pos=[0, 0], platform_vel=[0, 15],
             maneuvers=[], x_init=[2000, 8000, 0, 0]),
    ),
    "High noise, weak maneuver": (
        "Hard case: far target, 3 deg noise, small short turn. The estimate is often "
        "badly off (NEES fails), and in about a third of noise draws the range estimate "
        "runs off to infinity (no finite maximum likelihood).",
        dict(x_true=[10000, 25000, -5, -8], T=10, n=50, r=_var_deg(3.0),
             platform_pos=[0, 0], platform_vel=[0, 10],
             maneuvers=[Maneuver(4, 9, 20, 30)], x_init=[8000, 20000, 0, 0]),
    ),
}


def sample_names() -> list[str]:
    """Names of the available samples, in display order."""
    return list(SAMPLES)


def sample_description(name: str) -> str:
    """One-line description of sample ``name``."""
    return SAMPLES[name][0]


def sample_scenario(name: str) -> Scenario:
    """Build the :class:`Scenario` of sample ``name`` (seed 1, Gauss-Newton).

    Raises
    ------
    KeyError
        If there is no sample with that name.
    """
    _, kwargs = SAMPLES[name]
    return Scenario(seed=1, method="gauss-newton", **kwargs)
