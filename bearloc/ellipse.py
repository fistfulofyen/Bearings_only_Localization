"""The g-sigma uncertainty ellipse of a 2x2 covariance matrix (used by DBEAR).

Problem
-------
Given a symmetric positive definite matrix

.. math::  P = \\begin{bmatrix} P_{11} & P_{12} \\\\ P_{12} & P_{22} \\end{bmatrix}

find the curve :math:`\\{x : x' P^{-1} x = g^2\\}` (shifted to a centre ``c``).

Algorithm
---------
1. **Eigenvalues** (closed form for a symmetric 2x2 matrix)::

       m      = (P11 + P22) / 2
       d      = sqrt( ((P11 - P22) / 2)^2 + P12^2 )
       lam1   = m + d          (largest,  >= lam2)
       lam2   = m - d          (smallest, > 0 because P is positive definite)

2. **Orientation** of the major axis (eigenvector of ``lam1``), measured from
   the first coordinate axis::

       theta = 0.5 * atan2(2 * P12, P11 - P22)

   Proof: writing ``P = R(theta) diag(lam1, lam2) R(theta)'`` gives
   ``P11 - P22 = (lam1 - lam2) cos(2 theta)`` and
   ``2 P12 = (lam1 - lam2) sin(2 theta)``.

3. **Semi-axes**: ``a = g * sqrt(lam1)`` (major), ``b = g * sqrt(lam2)`` (minor).

4. **Points**: for ``phi`` in ``[0, 2 pi)``::

       x(phi) = c + R(theta) [a cos(phi), b sin(phi)]'
       R(theta) = [[cos theta, -sin theta],
                   [sin theta,  cos theta]]

   Check: with ``y = x - c`` and ``P^{-1} = R diag(1/lam1, 1/lam2) R'``,
   ``y' P^{-1} y = (a cos phi)^2 / lam1 + (b sin phi)^2 / lam2 = g^2``.

Probability content
-------------------
If ``x ~ N(c, P)`` then ``(x - c)' P^{-1} (x - c) ~ chi2(2)``, whose CDF is
``1 - exp(-q/2)``.  Hence the probability that ``x`` falls inside the g-sigma
ellipse is ``1 - exp(-g^2 / 2)``: 39.3 % (g = 1), 86.5 % (g = 2),
98.9 % (g = 3).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EllipseAxes:
    """Principal axes of a 2x2 covariance matrix.

    Attributes
    ----------
    lam1, lam2 : float
        Eigenvalues, ``lam1 >= lam2 > 0`` (variances along the axes).
    theta : float
        Angle (rad) of the major axis from the first coordinate axis, in
        ``(-pi/2, pi/2]``.
    """

    lam1: float
    lam2: float
    theta: float


def _check_covariance(P: np.ndarray) -> np.ndarray:
    """Validate that ``P`` is a symmetric positive definite 2x2 matrix."""
    P = np.asarray(P, dtype=float)
    if P.shape != (2, 2):
        raise ValueError(f"P must be 2x2, got shape {P.shape}")
    if not np.all(np.isfinite(P)):
        raise ValueError("P contains non-finite values")
    if not np.isclose(P[0, 1], P[1, 0], rtol=1e-9, atol=1e-12 * np.abs(P).max()):
        raise ValueError("P must be symmetric")
    if P[0, 0] <= 0 or np.linalg.det(P) <= 0:
        raise ValueError("P must be positive definite")
    return P


def ellipse_axes(P: np.ndarray) -> EllipseAxes:
    """Eigenvalues and orientation of a 2x2 SPD matrix (steps 1-2 above).

    Raises
    ------
    ValueError
        If ``P`` is not a symmetric positive definite 2x2 matrix.
    """
    P = _check_covariance(P)
    p11, p22 = P[0, 0], P[1, 1]
    p12 = 0.5 * (P[0, 1] + P[1, 0])
    mean = 0.5 * (p11 + p22)
    half_diff = 0.5 * (p11 - p22)
    d = np.hypot(half_diff, p12)
    lam1 = mean + d
    # lam1 * lam2 = det(P): avoids cancellation in mean - d for thin ellipses
    lam2 = (p11 * p22 - p12 * p12) / lam1
    theta = 0.5 * np.arctan2(2.0 * p12, p11 - p22)
    return EllipseAxes(float(lam1), float(lam2), float(theta))


def sigma_ellipse(
    P: np.ndarray,
    g: float = 1.0,
    center: np.ndarray | tuple[float, float] = (0.0, 0.0),
    num_points: int = 100,
) -> np.ndarray:
    """Points on the g-sigma ellipse ``(x - c)' P^{-1} (x - c) = g^2``.

    Parameters
    ----------
    P : array_like, shape (2, 2)
        Symmetric positive definite covariance matrix.
    g : float
        Size of the ellipse in "sigmas" (> 0).
    center : array_like, shape (2,)
        Ellipse centre ``c`` (e.g. the estimated target position).
    num_points : int
        Number of points; the curve is closed (first point repeated last).

    Returns
    -------
    numpy.ndarray
        Array of shape ``(num_points, 2)``.
    """
    if g <= 0:
        raise ValueError(f"g must be positive, got {g}")
    if num_points < 3:
        raise ValueError(f"num_points must be >= 3, got {num_points}")
    axes = ellipse_axes(P)
    a, b = g * np.sqrt(axes.lam1), g * np.sqrt(axes.lam2)
    phi = np.linspace(0.0, 2.0 * np.pi, num_points)
    c, s = np.cos(axes.theta), np.sin(axes.theta)
    R = np.array([[c, -s], [s, c]])
    unit = np.vstack([a * np.cos(phi), b * np.sin(phi)])
    return (R @ unit).T + np.asarray(center, dtype=float)


def ellipse_probability(g: float) -> float:
    """Probability mass of a 2-D Gaussian inside its g-sigma ellipse: ``1 - exp(-g^2/2)``."""
    return float(1.0 - np.exp(-0.5 * g * g))


def g_for_probability(p: float) -> float:
    """Ellipse size ``g`` that contains probability ``p``: ``sqrt(-2 ln(1 - p))``."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"probability must be in (0, 1), got {p}")
    return float(np.sqrt(-2.0 * np.log1p(-p)))
