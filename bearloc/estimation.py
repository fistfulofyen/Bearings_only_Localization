"""Maximum likelihood estimation of the target parameter (step B5).

With i.i.d. Gaussian bearing noise the negative log-likelihood is, up to a
constant, half of the weighted least-squares cost

.. math::

    \\lambda(x) = \\frac{1}{r} \\sum_{k=1}^{n} \\big[z(k) - h(k, x)\\big]^2,

so the MLE is the nonlinear least-squares solution
:math:`\\hat x = \\arg\\min_x \\lambda(x)`.  At the MLE, ``lambda(x_hat)`` is the
normalized fitting error of step B7 (chi-square with ``n - 4`` degrees of
freedom if the model is correct).

Two minimizers are provided:

* :func:`gauss_newton` -- Gauss-Newton with Levenberg-Marquardt damping
  (recommended).  Its Hessian approximation is exactly the FIM, so it
  converges in a handful of iterations.
* :func:`bfgs` -- the quasi-Newton BFGS method of :func:`scipy.optimize.minimize`
  using the analytic gradient.

Residuals ``z(k) - h(k, x)`` are always wrapped into ``(-pi, pi]``.

Range divergence (no finite MLE)
--------------------------------
Scaling the target-minus-platform trajectory by ``alpha -> infinity`` makes
the predicted bearings tend to those of a target seen without any parallax
from the platform maneuver, so ``lambda`` tends to a finite limit along that
ray.  When the maneuver is weak compared with the noise, a noisy data set can
fit this limit better than any finite range: the likelihood then has **no
finite maximum** and every descent method runs off to infinite range (where
the gradient vanishes, which a naive test would call "converged").  Both
minimizers therefore stop and report ``diverged`` once the range exceeds
:data:`DIVERGENCE_FACTOR` times the problem's length scale
(:func:`length_scale`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize

from . import model

#: Largest Levenberg-Marquardt damping factor tried before giving up.
_MU_MAX = 1e16

#: An estimate whose range exceeds this multiple of :func:`length_scale` is
#: treated as running off to infinite range (no finite MLE).
DIVERGENCE_FACTOR = 1e3

_DIVERGED_MESSAGE = (
    "range diverged to infinity: the likelihood has no finite maximum "
    "(the bearings do not determine the range)"
)


def length_scale(x0: np.ndarray, t: np.ndarray, platform_xy: np.ndarray) -> float:
    """Length scale of the problem, used by the divergence test.

    The larger of the initial-estimate range at the first measurement and the
    size (diagonal of the bounding box) of the platform track.
    """
    platform_xy = np.asarray(platform_xy, dtype=float)
    rho0 = float(np.hypot(*(model.target_position(x0, float(t[0])) - platform_xy[0])))
    extent = float(np.hypot(*np.ptp(platform_xy, axis=0)))
    return max(rho0, extent, 1e-300)


#: A BFGS result is accepted as a minimum when a Gauss-Newton step could lower
#: the (chi-square scaled) cost by no more than this.
BFGS_DECREMENT_TOL = 1e-6


def gauss_newton_decrement(x: np.ndarray, z: np.ndarray, t: np.ndarray,
                           platform_xy: np.ndarray, r: float) -> float:
    """Cost reduction predicted by one undamped Gauss-Newton step from ``x``.

    ``delta = g' A^+ g`` with ``A = H'H/r`` and ``g = H'e/r``.  It is zero at a
    minimum and, like the cost, dimensionless, so it is a unit-free
    optimality measure.
    """
    H = model.bearing_jacobian(x, t, platform_xy)
    e = residuals(x, z, t, platform_xy)
    A, g = H.T @ H / r, H.T @ e / r
    return float(g @ np.linalg.lstsq(A, g, rcond=None)[0])


def max_range(x: np.ndarray, t: np.ndarray, platform_xy: np.ndarray) -> float:
    """Largest target-to-platform distance over the measurement times."""
    d = model.target_position(x, np.asarray(t, dtype=float)) - np.asarray(platform_xy, dtype=float)
    return float(np.max(np.hypot(d[:, 0], d[:, 1])))


@dataclass
class MLEResult:
    """Result of the maximum likelihood minimization.

    Attributes
    ----------
    x : numpy.ndarray
        Estimate ``x_hat`` (length 4).
    cost : float
        Normalized fitting error ``lambda(x_hat) = sum(residuals**2) / r``.
    residuals : numpy.ndarray
        Wrapped residuals ``z(k) - h(k, x_hat)``.
    iterations : int
        Number of iterations performed.
    converged : bool
        Whether the convergence criterion was met (always False if diverged).
    message : str
        Human-readable termination reason.
    method : str
        ``"gauss-newton"`` or ``"bfgs"``.
    history : list of numpy.ndarray
        Iterates, starting with the initial estimate.
    diverged : bool
        True if the range ran off to infinity (no finite MLE, see the module
        docstring); ``x`` is then the last iterate.
    """

    x: np.ndarray
    cost: float
    residuals: np.ndarray
    iterations: int
    converged: bool
    message: str
    method: str
    history: list[np.ndarray] = field(default_factory=list)
    diverged: bool = False


def residuals(x: np.ndarray, z: np.ndarray, t: np.ndarray, platform_xy: np.ndarray) -> np.ndarray:
    """Wrapped bearing residuals ``z(k) - h(k, x)``."""
    return model.wrap_angle(np.asarray(z, dtype=float) - model.bearings(x, t, platform_xy))


def cost(x: np.ndarray, z: np.ndarray, t: np.ndarray, platform_xy: np.ndarray, r: float) -> float:
    """Normalized least-squares cost ``lambda(x) = sum(residuals**2) / r``."""
    e = residuals(x, z, t, platform_xy)
    return float(e @ e / r)


def gauss_newton(
    z: np.ndarray,
    t: np.ndarray,
    platform_xy: np.ndarray,
    r: float,
    x0: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> MLEResult:
    """Minimize ``lambda(x)`` by Gauss-Newton with Levenberg-Marquardt damping.

    Each iteration linearizes ``h`` about the current iterate ``x^i``
    (Jacobian ``H``, wrapped residual ``e``) and solves

    .. math::

        \\big(A + \\mu\\, \\operatorname{diag}(A)\\big)\\, \\Delta x = g,
        \\qquad A = H'H / r = J(x^i),\\quad g = H'e / r .

    A step that lowers the cost is accepted and ``mu`` is decreased ten-fold;
    otherwise ``mu`` is increased ten-fold and the step is recomputed.  For
    ``mu -> 0`` this is the Gauss-Newton step; for large ``mu`` it is a short,
    scaled steepest-descent step, which makes the method robust far from the
    minimum.  Scaling the damping by ``diag(A)`` makes it independent of the
    units of the parameters.

    Convergence is declared when the *Gauss-Newton decrement*
    ``delta = g' A^+ g`` -- the cost reduction predicted by the undamped
    linearized model -- drops below ``tol``.  Because the cost is
    dimensionless (chi-square scale), so is this criterion.

    The iteration stops with ``diverged=True`` if the range exceeds
    ``DIVERGENCE_FACTOR * length_scale`` (no finite MLE; see module docstring).

    Parameters
    ----------
    z : array_like, shape (n,)
        Measured bearings (rad).
    t : array_like, shape (n,)
        Measurement times.
    platform_xy : array_like, shape (n, 2)
        Platform positions at the measurement times.
    r : float
        Measurement noise variance (rad^2).
    x0 : array_like, shape (4,)
        Initial estimate (input A7).
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence threshold on the Gauss-Newton decrement.
    """
    x = np.asarray(x0, dtype=float).copy()
    e = residuals(x, z, t, platform_xy)
    c = float(e @ e / r)
    mu = 1e-3
    history = [x.copy()]
    converged = diverged = False
    message = f"maximum number of iterations ({max_iter}) reached"
    range_bound = DIVERGENCE_FACTOR * length_scale(x, t, platform_xy)
    it = 0
    for it in range(1, max_iter + 1):
        H = model.bearing_jacobian(x, t, platform_xy)
        A = H.T @ H / r
        g = H.T @ e / r
        decrement = float(g @ np.linalg.lstsq(A, g, rcond=None)[0])
        if decrement <= tol:
            converged = True
            message = "converged (Gauss-Newton decrement below tolerance)"
            it -= 1  # this pass only verified convergence
            break
        damping = np.diag(np.maximum(np.diag(A), np.finfo(float).tiny))
        accepted = False
        while mu <= _MU_MAX:
            try:
                dx = np.linalg.solve(A + mu * damping, g)
            except np.linalg.LinAlgError:
                dx = np.linalg.lstsq(A + mu * damping, g, rcond=None)[0]
            x_new = x + dx
            try:
                e_new = residuals(x_new, z, t, platform_xy)
            except ValueError:  # step put the target on the platform: shorten it
                mu *= 10.0
                continue
            c_new = float(e_new @ e_new / r)
            if c_new < c:
                x, e, c = x_new, e_new, c_new
                mu = max(mu / 10.0, 1e-12)
                accepted = True
                break
            mu *= 10.0
        history.append(x.copy())
        if not accepted:
            message = "stopped: no step reduces the cost (possible numerical minimum)"
            break
        if max_range(x, t, platform_xy) > range_bound:
            diverged, message = True, _DIVERGED_MESSAGE
            break
    return MLEResult(
        x=x,
        cost=c,
        residuals=e,
        iterations=it,
        converged=converged and not diverged,
        message=message,
        method="gauss-newton",
        history=history,
        diverged=diverged,
    )


def bfgs(
    z: np.ndarray,
    t: np.ndarray,
    platform_xy: np.ndarray,
    r: float,
    x0: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-10,
) -> MLEResult:
    """Minimize ``lambda(x)`` with the quasi-Newton BFGS method.

    The parameters are rescaled to ``u = sqrt(diag J(x0)) * (x - x0)`` so that
    all components have comparable magnitudes (positions and velocities can
    differ by many orders of magnitude), which BFGS needs for good
    performance.  The analytic gradient

    .. math::  \\nabla \\lambda(x) = -\\frac{2}{r} H' e

    is used.  ``tol`` is passed to SciPy as the gradient tolerance (``gtol``)
    in the scaled coordinates (with a floor of ``1e-8``).  The final estimate
    is checked for range divergence like in :func:`gauss_newton`.

    When SciPy stops with "precision loss" (status 2) -- which often happens
    exactly at the minimum -- the result is accepted as converged if the
    Gauss-Newton decrement is below :data:`BFGS_DECREMENT_TOL`.

    Parameters are as for :func:`gauss_newton`.
    """
    x0 = np.asarray(x0, dtype=float).copy()
    H0 = model.bearing_jacobian(x0, t, platform_xy)
    s = np.sqrt(np.maximum(np.einsum("ij,ij->j", H0, H0) / r, np.finfo(float).tiny))
    history = [x0.copy()]

    def to_x(u: np.ndarray) -> np.ndarray:
        return x0 + u / s

    def fun(u: np.ndarray) -> tuple[float, np.ndarray]:
        x = to_x(u)
        e = residuals(x, z, t, platform_xy)
        H = model.bearing_jacobian(x, t, platform_xy)
        grad_x = -2.0 * H.T @ e / r
        return float(e @ e / r), grad_x / s

    res = optimize.minimize(
        fun,
        np.zeros(model.N_PARAMS),
        jac=True,
        method="BFGS",
        callback=lambda u: history.append(to_x(u)),
        options={"maxiter": max_iter, "gtol": max(tol, 1e-8)},
    )
    x = to_x(res.x)
    e = residuals(x, z, t, platform_xy)
    diverged = max_range(x, t, platform_xy) > DIVERGENCE_FACTOR * length_scale(x0, t, platform_xy)
    converged, message = bool(res.success), str(res.message)
    if not converged and res.status == 2:
        # SciPy's line search often stops with "precision loss" exactly at the
        # optimum.  Verify optimality independently with the (unit-free)
        # Gauss-Newton decrement: the cost reduction still achievable.
        if gauss_newton_decrement(x, z, t, platform_xy, r) <= BFGS_DECREMENT_TOL:
            converged = True
            message = ("converged (BFGS stopped with precision loss; the Gauss-Newton "
                       "decrement confirms the minimum)")
    return MLEResult(
        x=x,
        cost=float(e @ e / r),
        residuals=e,
        iterations=int(res.nit),
        converged=converged and not diverged,
        message=_DIVERGED_MESSAGE if diverged else message,
        method="bfgs",
        history=history,
        diverged=diverged,
    )


def maximum_likelihood(
    z: np.ndarray,
    t: np.ndarray,
    platform_xy: np.ndarray,
    r: float,
    x0: np.ndarray,
    method: str = "gauss-newton",
    max_iter: int = 200,
    tol: float = 1e-10,
) -> MLEResult:
    """Compute the MLE with the chosen ``method`` (``"gauss-newton"`` or ``"bfgs"``)."""
    solvers = {"gauss-newton": gauss_newton, "bfgs": bfgs}
    try:
        solver = solvers[method]
    except KeyError:
        raise ValueError(f"unknown method {method!r}; choose from {sorted(solvers)}") from None
    return solver(z, t, platform_xy, r, x0, max_iter=max_iter, tol=tol)
