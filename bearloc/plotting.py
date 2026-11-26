"""Graphics for DBEAR: target uncertainty ellipses and trajectories.

The functions here work on the results dictionary written by
:func:`bearloc.analysis.save_results`, so DBEAR can be run any time after
BEAR without repeating the estimation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import model
from .ellipse import ellipse_axes, ellipse_probability, sigma_ellipse
from .scenario import Scenario

#: Covariance sources selectable in DBEAR.
SOURCES = {
    "estimate": "B6 (FIM at the estimate)",
    "truth": "B3 (FIM at the true parameter)",
}


@dataclass(frozen=True)
class EllipseSpec:
    """One uncertainty ellipse to draw.

    Attributes
    ----------
    t : float
        Time of the ellipse.
    center : numpy.ndarray
        Estimated target position at ``t`` (ellipse centre).
    true_position : numpy.ndarray
        True target position at ``t``.
    cov : numpy.ndarray
        2x2 position covariance at ``t``.
    g : float
        Size in sigmas.
    """

    t: float
    center: np.ndarray
    true_position: np.ndarray
    cov: np.ndarray
    g: float

    @property
    def semi_axes(self) -> tuple[float, float]:
        """Major and minor semi-axis lengths ``g*sqrt(lam1), g*sqrt(lam2)``."""
        ax = ellipse_axes(self.cov)
        return self.g * float(np.sqrt(ax.lam1)), self.g * float(np.sqrt(ax.lam2))

    @property
    def orientation_deg(self) -> float:
        """Major-axis angle from the xi axis, degrees."""
        return float(np.degrees(ellipse_axes(self.cov).theta))

    @property
    def contains_truth(self) -> bool:
        """Whether the true position lies inside the ellipse."""
        d = self.true_position - self.center
        return bool(d @ np.linalg.solve(self.cov, d) <= self.g**2)


def default_times(results: dict, count: int = 5) -> list[float]:
    """``count`` sampling times spread evenly over the observation window."""
    s = results["scenario"]
    ks = np.unique(np.round(np.linspace(1, s["n"], min(count, s["n"]))).astype(int))
    return [float(k * s["T"]) for k in ks]


def ellipse_specs(
    results: dict, times: Sequence[float], g: float, source: str = "estimate"
) -> list[EllipseSpec]:
    """Build the ellipses for the requested times.

    The position covariance at an arbitrary time ``t`` (not necessarily a
    sampling time) is ``A(t) P A(t)'`` with ``P`` the inverse FIM from B6
    (``source="estimate"``) or B3 (``source="truth"``).  Ellipses are always
    centred at the *estimated* position ``A(t) x_hat``.

    Raises
    ------
    ValueError
        If the requested covariance does not exist (singular FIM) or the
        source is unknown.
    """
    if source not in SOURCES:
        raise ValueError(f"source must be one of {sorted(SOURCES)}, got {source!r}")
    P = results[source]["P"]
    if P is None:
        raise ValueError(
            f"the FIM of {SOURCES[source]} is singular: no covariance, no ellipses"
        )
    P = np.asarray(P)
    x_hat = np.asarray(results["mle"]["x"])
    x_true = np.asarray(results["scenario"]["x_true"])
    specs = []
    for t in times:
        specs.append(
            EllipseSpec(
                t=float(t),
                center=model.target_position(x_hat, float(t)),
                true_position=model.target_position(x_true, float(t)),
                cov=model.position_covariances(P, float(t)),
                g=float(g),
            )
        )
    return specs


def format_ellipse_table(specs: Sequence[EllipseSpec]) -> str:
    """Text table of the ellipse parameters (printed by DBEAR)."""
    lines = [
        f"{'t':>9} {'centre xi':>12} {'centre eta':>12} {'semi-major':>11} "
        f"{'semi-minor':>11} {'angle(deg)':>11} {'truth inside':>13}"
    ]
    for e in specs:
        a, b = e.semi_axes
        lines.append(
            f"{e.t:>9.4g} {e.center[0]:>12.6g} {e.center[1]:>12.6g} {a:>11.5g} "
            f"{b:>11.5g} {e.orientation_deg:>11.2f} {('yes' if e.contains_truth else 'no'):>13}"
        )
    return "\n".join(lines)


def _new_axes(ax):
    """Return ``(figure, axes)``, creating a pyplot figure only if ``ax`` is None."""
    if ax is not None:
        return ax.figure, ax
    import matplotlib.pyplot as plt  # only needed for stand-alone figures

    fig, ax = plt.subplots(figsize=(9, 8))
    return fig, ax


def _draw_tracks(ax, scenario: Scenario, x_hat: np.ndarray | None, t_end: float,
                 platform_in_legend: bool = True) -> None:
    """Draw the platform track (maneuvers highlighted) and the target tracks.

    ``platform_in_legend=False`` keeps the platform out of the legend (used
    when the view is zoomed to the target and the platform is off-screen).
    """
    t_dense = np.linspace(0.0, t_end, 400)
    plat_dense = scenario.platform_at(t_dense)
    ax.plot(plat_dense[:, 0], plat_dense[:, 1], color="tab:blue", lw=2,
            label="platform" if platform_in_legend else None)
    ax.plot(*scenario.platform_pos, "o", color="tab:blue", ms=6)
    for leg in scenario.maneuvers:
        t_leg = np.linspace(leg.k1 * scenario.T, min(leg.k2 * scenario.T, t_end), 50)
        if t_leg[0] < t_end:
            seg = scenario.platform_at(t_leg)
            ax.plot(seg[:, 0], seg[:, 1], color="tab:cyan", lw=4, alpha=0.7)
    if scenario.maneuvers and platform_in_legend:
        ax.plot([], [], color="tab:cyan", lw=4, alpha=0.7, label="platform maneuver")

    true_track = model.target_position(scenario.x_true, t_dense)
    ax.plot(true_track[:, 0], true_track[:, 1], "k--", lw=1.5, label="true target")
    ax.plot(*true_track[0], "ks", ms=6)
    if x_hat is not None:
        est_track = model.target_position(x_hat, t_dense)
        ax.plot(est_track[:, 0], est_track[:, 1], color="tab:red", lw=1.5, label="estimated target")
        ax.plot(*est_track[0], "s", color="tab:red", ms=6)


def _draw_ray(ax, scenario: Scenario, t: float, bearing: float, x_ref: np.ndarray) -> None:
    """Draw a measured line of sight from the platform at time ``t``.

    The ray ends a little past the target position predicted by ``x_ref``.
    """
    p = scenario.platform_at(t)
    length = 1.15 * np.hypot(*(model.target_position(x_ref, t) - p))
    end = p + length * np.array([np.cos(bearing), np.sin(bearing)])
    ax.plot([p[0], end[0]], [p[1], end[1]], color="tab:gray", lw=0.7, alpha=0.6)


def _finish_axes(ax, title: str) -> None:
    """Common axis labels, equal scaling, grid and legend."""
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("xi")
    ax.set_ylabel("eta")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)


def plot_geometry(
    scenario: Scenario,
    x_hat: np.ndarray | None = None,
    z: np.ndarray | None = None,
    num_rays: int = 8,
    ax=None,
):
    """Draw the scenario geometry: platform path, true (and estimated) target track.

    Parameters
    ----------
    scenario : Scenario
        Inputs A1-A6.
    x_hat : array_like, optional
        Estimated parameter; its track is drawn if given.
    z : array_like, optional
        Measured bearings; ``num_rays`` evenly spaced lines of sight are drawn.
    num_rays : int
        Number of measured bearings to draw when ``z`` is given.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _new_axes(ax)
    times = scenario.times
    _draw_tracks(ax, scenario, None if x_hat is None else np.asarray(x_hat), float(times[-1]))
    if z is not None and num_rays > 0:
        ref = scenario.x_true if x_hat is None else np.asarray(x_hat)
        for k in np.unique(np.linspace(0, scenario.n - 1, num_rays).round().astype(int)):
            _draw_ray(ax, scenario, times[k], z[k], ref)
        ax.plot([], [], color="tab:gray", lw=0.7, label="measured bearing")
    _finish_axes(ax, "Scenario geometry" + ("" if x_hat is None else " and estimated track"))
    fig.tight_layout()
    return fig


def plot_ellipses(
    results: dict,
    specs: Sequence[EllipseSpec],
    source: str = "estimate",
    show_rays: bool = False,
    ax=None,
    zoom: bool = False,
):
    """Draw trajectories, measurements and uncertainty ellipses.

    Parameters
    ----------
    results : dict
        Results dictionary from :func:`bearloc.analysis.load_results`.
    specs : sequence of EllipseSpec
        Ellipses to draw (from :func:`ellipse_specs`).
    source : str
        Covariance source, only used in the title.
    show_rays : bool
        Also draw the measured lines of sight at the ellipse times.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    zoom : bool
        Limit the view to the ellipses (and true positions) instead of the
        whole scenario.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _new_axes(ax)
    scenario = Scenario.from_dict(results["scenario"])
    times = np.asarray(results["times"])
    x_hat = np.asarray(results["mle"]["x"])
    _draw_tracks(ax, scenario, x_hat, max([times[-1], *[e.t for e in specs]]),
                 platform_in_legend=not zoom)

    if show_rays:
        z = np.asarray(results["measurements"])
        for e in specs:
            k = int(np.argmin(np.abs(times - e.t)))  # measurement closest to the ellipse time
            _draw_ray(ax, scenario, times[k], z[k], x_hat)
        ax.plot([], [], color="tab:gray", lw=0.7, label="measured bearing")

    for i, e in enumerate(specs):
        pts = sigma_ellipse(e.cov, e.g, e.center, num_points=200)
        ax.plot(pts[:, 0], pts[:, 1], color="tab:green", lw=1.5,
                label=f"{e.g:g}-sigma ellipse" if i == 0 else None)
        ax.plot(*e.center, "+", color="tab:red", ms=10)
        ax.plot(*e.true_position, "kx", ms=7)
        # label below-right of the estimate, on a light box so that markers
        # and ellipse lines underneath do not make it unreadable
        ax.annotate(f"t={e.t:g}", e.center, textcoords="offset points", xytext=(7, -9),
                    va="top", fontsize=8,
                    bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.8})

    if zoom and specs:
        pts = np.vstack([sigma_ellipse(e.cov, e.g, e.center, num_points=100) for e in specs]
                        + [e.true_position[None, :] for e in specs])
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        margin = 0.08 * max(hi - lo)
        # Restrict the data limits (not the axis limits) so that autoscaling can
        # still widen one direction to keep the equal aspect ratio.
        ax.dataLim.set_points(np.array([lo - margin, hi + margin]))
        ax.autoscale_view()

    g = specs[0].g if specs else 1.0
    _finish_axes(
        ax,
        f"Target position uncertainty: {g:g}-sigma ellipses "
        f"(P = {100 * ellipse_probability(g):.1f}%), covariance from {SOURCES[source]}",
    )
    fig.tight_layout()
    return fig
