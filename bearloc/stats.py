"""Statistical tools: observability test, chi-square bounds, NEES.

Used by steps B1 (observability), B7 (fitting error) and B8 (NEES).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

#: Default relative singular-value threshold for declaring the FIM singular.
DEFAULT_RCOND = 1e-10


@dataclass(frozen=True)
class Observability:
    """Outcome of the FIM invertibility test (step B1).

    Attributes
    ----------
    invertible : bool
        ``True`` if the FIM is (numerically) non-singular, i.e. the
        parameter is observable.
    rank : int
        Numerical rank of the FIM.
    condition : float
        Condition number of the *scaled* FIM (``inf`` if singular).
    singular_values : numpy.ndarray
        Singular values of the scaled FIM, descending.
    """

    invertible: bool
    rank: int
    condition: float
    singular_values: np.ndarray


def check_observability(J: np.ndarray, rcond: float = DEFAULT_RCOND) -> Observability:
    """Decide whether the Fisher information matrix is invertible.

    The FIM mixes position and velocity components whose magnitudes differ by
    orders of magnitude, so a plain determinant or condition number depends
    on the units.  The test is therefore performed on the diagonally scaled
    matrix

    .. math::  \\tilde J = D^{-1/2} J D^{-1/2},\\qquad D = \\operatorname{diag}(J)

    (a "correlation-like" matrix with unit diagonal), which is invariant to
    re-scaling of individual parameters.  ``J`` is declared singular if
    ``sigma_min / sigma_max <= rcond`` for the singular values of
    :math:`\\tilde J`, or if any diagonal element of ``J`` is zero.

    Parameters
    ----------
    J : array_like, shape (m, m)
        Symmetric positive semi-definite information matrix.
    rcond : float
        Relative singular-value threshold.
    """
    J = np.asarray(J, dtype=float)
    d = np.diag(J)
    if np.any(d <= 0.0) or not np.all(np.isfinite(J)):
        return Observability(False, int(np.sum(d > 0)), float("inf"), np.zeros(J.shape[0]))
    scale = 1.0 / np.sqrt(d)
    J_scaled = J * scale[:, None] * scale[None, :]
    sv = np.linalg.svd(J_scaled, compute_uv=False)
    rank = int(np.sum(sv > rcond * sv[0]))
    invertible = rank == J.shape[0]
    condition = float(sv[0] / sv[-1]) if invertible else float("inf")
    return Observability(invertible, rank, condition, sv)


def invert_fim(J: np.ndarray, rcond: float = DEFAULT_RCOND) -> np.ndarray | None:
    """Return ``J^{-1}`` (the CRLB) or ``None`` if ``J`` is singular (step B2).

    The inverse is computed in the scaled coordinates of
    :func:`check_observability` for numerical accuracy and symmetrized.
    """
    J = np.asarray(J, dtype=float)
    if not check_observability(J, rcond).invertible:
        return None
    scale = 1.0 / np.sqrt(np.diag(J))
    J_scaled = J * scale[:, None] * scale[None, :]
    P = np.linalg.inv(J_scaled) * scale[:, None] * scale[None, :]
    return 0.5 * (P + P.T)


def chi2_interval(dof: int, confidence: float = 0.95) -> tuple[float, float]:
    """Two-sided probability interval of a chi-square random variable.

    Returns ``(a, b)`` with ``P{a <= X <= b} = confidence`` and equal tail
    probabilities, for ``X ~ chi2(dof)``.

    Example: ``chi2_interval(4, 0.95) == (0.484, 11.14)`` (approximately).
    """
    if dof < 1:
        raise ValueError(f"degrees of freedom must be >= 1, got {dof}")
    alpha = 1.0 - confidence
    return float(sps.chi2.ppf(alpha / 2, dof)), float(sps.chi2.ppf(1 - alpha / 2, dof))


@dataclass(frozen=True)
class Chi2Test:
    """Result of comparing a statistic with its chi-square probability interval."""

    value: float
    dof: int
    lower: float
    upper: float
    confidence: float

    @property
    def passed(self) -> bool:
        """``True`` if ``lower <= value <= upper``."""
        return self.lower <= self.value <= self.upper


def chi2_test(value: float, dof: int, confidence: float = 0.95) -> Chi2Test:
    """Check whether ``value`` lies within the two-sided chi-square interval."""
    lo, hi = chi2_interval(dof, confidence)
    return Chi2Test(float(value), int(dof), lo, hi, float(confidence))


def nees(x_true: np.ndarray, x_hat: np.ndarray, J: np.ndarray) -> float:
    """Normalized estimation error squared (step B8).

    .. math::  \\epsilon = \\tilde x' J \\tilde x,\\qquad \\tilde x = x - \\hat x

    With ``J`` the FIM (inverse covariance) of an efficient estimator,
    ``epsilon ~ chi2(dim x)``.
    """
    err = np.asarray(x_true, dtype=float) - np.asarray(x_hat, dtype=float)
    return float(err @ np.asarray(J, dtype=float) @ err)
