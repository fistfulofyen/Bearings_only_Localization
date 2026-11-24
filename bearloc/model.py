"""Kinematic and measurement model for bearings-only target localization.

This module implements the model of Bar-Shalom, Li & Kirubarajan,
*Estimation with Applications to Tracking and Navigation*, Section 3.7.

Coordinates and conventions
---------------------------
* Cartesian plane with coordinates ``(xi, eta)`` (think East, North).
* The target moves with constant velocity.  Its parameter vector, defined at
  ``t = 0``, is::

      x = [xi0, eta0, xi_dot, eta_dot]'

  so that ``xi(t) = xi0 + xi_dot * t`` and ``eta(t) = eta0 + eta_dot * t``.
* Measurements are taken at ``t_k = k * T`` for ``k = 1, ..., n``.
* The bearing is the angle of the line of sight from the platform to the
  target, measured counter-clockwise from the ``xi`` axis, in radians, in
  ``(-pi, pi]``::

      h(k, x) = atan2(eta(t_k) - eta_p(t_k), xi(t_k) - xi_p(t_k))

* The measurement is ``z(k) = h(k, x) + w(k)`` with ``w(k) ~ N(0, r)`` i.i.d.

Platform motion
---------------
The platform starts at ``p0`` with velocity ``v0``.  A list of
:class:`Maneuver` legs overrides that velocity: leg ``j`` imposes velocity
``v_j`` for ``k1_j * T <= t < k2_j * T``.  Outside every leg the platform
moves with ``v0``.  Because the velocity is piecewise constant the position is
obtained in closed form (see :func:`platform_position`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

#: Dimension of the target parameter vector ``[xi0, eta0, xi_dot, eta_dot]``.
N_PARAMS = 4


@dataclass(frozen=True)
class Maneuver:
    """One platform maneuver leg (input A6).

    Attributes
    ----------
    vx, vy : float
        Platform velocity components (``xi`` and ``eta``) while the leg is
        active.
    k1, k2 : int
        Start and end sample indices.  The velocity is in effect for
        ``k1 * T <= t < k2 * T``.
    """

    vx: float
    vy: float
    k1: int
    k2: int

    def __post_init__(self) -> None:
        if self.k1 < 0:
            raise ValueError(f"maneuver start k1 must be >= 0, got {self.k1}")
        if self.k2 <= self.k1:
            raise ValueError(
                f"maneuver end k2 must be greater than k1, got k1={self.k1}, k2={self.k2}"
            )

    @property
    def velocity(self) -> np.ndarray:
        """Leg velocity as a length-2 array ``[vx, vy]``."""
        return np.array([self.vx, self.vy], dtype=float)


def validate_maneuvers(maneuvers: Sequence[Maneuver]) -> None:
    """Check that maneuver legs do not overlap in time.

    Overlapping legs would make the platform velocity ambiguous.

    Raises
    ------
    ValueError
        If two legs overlap.
    """
    legs = sorted(maneuvers, key=lambda m: m.k1)
    for prev, cur in zip(legs, legs[1:]):
        if cur.k1 < prev.k2:
            raise ValueError(
                f"maneuver legs overlap: [{prev.k1}, {prev.k2}) and [{cur.k1}, {cur.k2})"
            )


def measurement_times(T: float, n: int) -> np.ndarray:
    """Return the sampling times ``t_k = k * T`` for ``k = 1..n``.

    Parameters
    ----------
    T : float
        Sampling period (> 0).
    n : int
        Number of samples (>= 1).
    """
    if T <= 0:
        raise ValueError(f"sampling period T must be positive, got {T}")
    if n < 1:
        raise ValueError(f"number of samples n must be >= 1, got {n}")
    return T * np.arange(1, n + 1, dtype=float)


def platform_velocity(
    t: np.ndarray | float,
    v0: Sequence[float],
    maneuvers: Sequence[Maneuver],
    T: float,
) -> np.ndarray:
    """Platform velocity at time(s) ``t``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(len(t), 2)`` (or ``(2,)`` for scalar ``t``).
    """
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    v = np.tile(np.asarray(v0, dtype=float), (t_arr.size, 1))
    for leg in maneuvers:
        active = (t_arr >= leg.k1 * T) & (t_arr < leg.k2 * T)
        v[active] = leg.velocity
    return v[0] if np.ndim(t) == 0 else v


def platform_position(
    t: np.ndarray | float,
    p0: Sequence[float],
    v0: Sequence[float],
    maneuvers: Sequence[Maneuver],
    T: float,
) -> np.ndarray:
    """Platform position at time(s) ``t`` (inputs A5 and A6).

    The velocity is ``v0`` except during maneuver legs, so integrating gives

    .. math::

        p(t) = p_0 + v_0 t + \\sum_j (v_j - v_0)\\,
               \\operatorname{clip}(t - k_{1,j}T,\\ 0,\\ (k_{2,j}-k_{1,j})T)

    where the clipped term is the time spent so far inside leg ``j``.

    Parameters
    ----------
    t : float or array_like
        Time(s) at which to evaluate the position (``t >= 0``).
    p0, v0 : sequence of 2 floats
        Initial position and velocity.
    maneuvers : sequence of Maneuver
        Non-overlapping maneuver legs.
    T : float
        Sampling period, used to convert ``k1, k2`` into times.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(len(t), 2)`` (or ``(2,)`` for scalar ``t``).
    """
    validate_maneuvers(maneuvers)
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    p0 = np.asarray(p0, dtype=float)
    v0 = np.asarray(v0, dtype=float)
    pos = p0 + np.outer(t_arr, v0)
    for leg in maneuvers:
        t1, t2 = leg.k1 * T, leg.k2 * T
        time_in_leg = np.clip(t_arr - t1, 0.0, t2 - t1)
        pos += np.outer(time_in_leg, leg.velocity - v0)
    return pos[0] if np.ndim(t) == 0 else pos


def target_position(x: Sequence[float], t: np.ndarray | float) -> np.ndarray:
    """Target position at time(s) ``t`` for parameter ``x``.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(len(t), 2)`` (or ``(2,)`` for scalar ``t``).
    """
    x = np.asarray(x, dtype=float)
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    pos = x[:2] + np.outer(t_arr, x[2:])
    return pos[0] if np.ndim(t) == 0 else pos


def position_matrix(t: float) -> np.ndarray:
    """Matrix ``A(t)`` mapping the parameter to the position at time ``t``.

    ``target_position(x, t) == position_matrix(t) @ x``::

        A(t) = [[1, 0, t, 0],
                [0, 1, 0, t]]
    """
    return np.array([[1.0, 0.0, t, 0.0], [0.0, 1.0, 0.0, t]])


def wrap_angle(a: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle(s) into the interval ``(-pi, pi]``.

    Bearing residuals must be wrapped, otherwise a target near the ``+-pi``
    direction would produce spurious residuals of about ``2*pi``.
    """
    wrapped = np.pi - np.mod(np.pi - np.asarray(a, dtype=float), 2.0 * np.pi)
    return float(wrapped) if np.ndim(wrapped) == 0 else wrapped


def _relative_geometry(
    x: Sequence[float], t: np.ndarray, platform_xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return target-minus-platform offsets ``(d_xi, d_eta)`` at times ``t``."""
    t = np.asarray(t, dtype=float)
    platform_xy = np.asarray(platform_xy, dtype=float).reshape(t.size, 2)
    rel = target_position(x, t).reshape(t.size, 2) - platform_xy
    return rel[:, 0], rel[:, 1]


def bearings(x: Sequence[float], t: np.ndarray, platform_xy: np.ndarray) -> np.ndarray:
    """Noise-free bearings ``h(k, x)`` (radians, in ``(-pi, pi]``).

    Parameters
    ----------
    x : sequence of 4 floats
        Target parameter ``[xi0, eta0, xi_dot, eta_dot]``.
    t : array_like, shape (n,)
        Measurement times.
    platform_xy : array_like, shape (n, 2)
        Platform positions at those times.
    """
    d_xi, d_eta = _relative_geometry(x, t, platform_xy)
    return np.arctan2(d_eta, d_xi)


def bearing_jacobian(
    x: Sequence[float], t: np.ndarray, platform_xy: np.ndarray
) -> np.ndarray:
    """Jacobian of the bearings with respect to the parameter ``x``.

    With ``d_xi = xi(t) - xi_p(t)``, ``d_eta = eta(t) - eta_p(t)`` and
    ``rho^2 = d_xi^2 + d_eta^2``::

        dh/dx = [-d_eta, d_xi, -d_eta * t, d_xi * t] / rho^2

    Returns
    -------
    numpy.ndarray
        Array ``H`` of shape ``(n, 4)``; row ``k`` is ``dh(k, x)/dx``.

    Raises
    ------
    ValueError
        If the target coincides with the platform (bearing undefined).
    """
    t = np.asarray(t, dtype=float)
    d_xi, d_eta = _relative_geometry(x, t, platform_xy)
    rho2 = d_xi**2 + d_eta**2
    if np.any(rho2 == 0.0):
        raise ValueError("target coincides with the platform; bearing is undefined")
    return np.column_stack([-d_eta, d_xi, -d_eta * t, d_xi * t]) / rho2[:, None]


def fisher_information(
    x: Sequence[float], t: np.ndarray, platform_xy: np.ndarray, r: float
) -> np.ndarray:
    """Fisher information matrix (FIM) for the parameter ``x`` (step B1).

    For i.i.d. Gaussian bearing noise with variance ``r``:

    .. math::

        J(x) = \\frac{1}{r} \\sum_{k=1}^{n} h_x(k, x)'\\, h_x(k, x) = \\frac{1}{r} H'H

    Its inverse, when it exists, is the Cramer-Rao lower bound on the
    covariance of any unbiased estimator of ``x``.

    Returns
    -------
    numpy.ndarray
        Symmetric positive semi-definite ``(4, 4)`` matrix.
    """
    if r <= 0:
        raise ValueError(f"noise variance r must be positive, got {r}")
    H = bearing_jacobian(x, t, platform_xy)
    J = H.T @ H / r
    return 0.5 * (J + J.T)  # enforce exact symmetry


def position_covariances(P: np.ndarray, t: np.ndarray | float) -> np.ndarray:
    """Target position covariance at time(s) ``t`` (step B3).

    Since the position is ``A(t) x``, its covariance is ``A(t) P A(t)'``.

    Parameters
    ----------
    P : array_like, shape (4, 4)
        Parameter covariance (e.g. the inverse FIM).
    t : float or array_like
        Time(s).

    Returns
    -------
    numpy.ndarray
        Shape ``(len(t), 2, 2)`` (or ``(2, 2)`` for scalar ``t``).
    """
    P = np.asarray(P, dtype=float)
    t_arr = np.atleast_1d(np.asarray(t, dtype=float))
    covs = np.empty((t_arr.size, 2, 2))
    for i, ti in enumerate(t_arr):
        A = position_matrix(ti)
        C = A @ P @ A.T
        covs[i] = 0.5 * (C + C.T)
    return covs[0] if np.ndim(t) == 0 else covs
