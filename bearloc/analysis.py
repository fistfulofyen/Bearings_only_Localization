"""The BEAR processing pipeline, steps B1-B8, independent of any user interface.

:func:`run_analysis` performs, for a :class:`~bearloc.scenario.Scenario`:

* **B1** Fisher information matrix at the true parameter; observability test.
* **B2** Its inverse (Cramer-Rao lower bound), if it exists.
* **B3** Target position covariance at every sampling time.
* **B4** Noisy bearing measurements.
* **B5** Maximum likelihood estimate from the initial estimate (A7).
* **B6** B1-B3 repeated at the estimate.
* **B7** Fitting error ``lambda(x_hat)`` vs. ``chi2(n - 4)`` interval.
* **B8** NEES ``x_tilde' J x_tilde`` vs. ``chi2(4)`` interval.

:func:`monte_carlo` repeats B4, B5, B7 and B8 over many independent noise
realizations to check statistical consistency of the estimator (an optional
extension of the project).

Results can be written to JSON (:func:`save_results`) for the DBEAR plotting
program.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import estimation, model, stats
from .scenario import Scenario


@dataclass
class FIMAnalysis:
    """Outcome of steps B1-B3 at a given parameter value.

    Attributes
    ----------
    x : numpy.ndarray
        Parameter at which the FIM is evaluated.
    J : numpy.ndarray
        Fisher information matrix (4x4).
    observability : bearloc.stats.Observability
        Invertibility test result.
    P : numpy.ndarray or None
        ``J^{-1}`` (CRLB), or ``None`` if ``J`` is singular.
    position_covs : numpy.ndarray or None
        Position covariances at the sampling times, shape ``(n, 2, 2)``, or
        ``None`` if ``J`` is singular.
    """

    x: np.ndarray
    J: np.ndarray
    observability: stats.Observability
    P: np.ndarray | None
    position_covs: np.ndarray | None

    def to_dict(self) -> dict:
        """JSON-friendly representation."""
        return {
            "x": self.x.tolist(),
            "J": self.J.tolist(),
            "invertible": self.observability.invertible,
            "rank": self.observability.rank,
            "condition": self.observability.condition,
            "P": None if self.P is None else self.P.tolist(),
            "position_covs": None if self.position_covs is None else self.position_covs.tolist(),
        }


def analyze_fim(x: np.ndarray, scenario: Scenario) -> FIMAnalysis:
    """Steps B1-B3 at parameter ``x``: FIM, observability, inverse, position covariances."""
    x = np.asarray(x, dtype=float)
    J = model.fisher_information(x, scenario.times, scenario.platform_track, scenario.r)
    obs = stats.check_observability(J)
    P = stats.invert_fim(J) if obs.invertible else None
    covs = None if P is None else model.position_covariances(P, scenario.times)
    return FIMAnalysis(x=x, J=J, observability=obs, P=P, position_covs=covs)


@dataclass
class AnalysisResult:
    """Everything computed by one BEAR run (steps B1-B8)."""

    scenario: Scenario
    truth: FIMAnalysis                  # B1-B3
    z: np.ndarray                       # B4
    mle: estimation.MLEResult           # B5
    estimate: FIMAnalysis               # B6
    fit: stats.Chi2Test | None          # B7 (None when n <= rank J)
    nees: stats.Chi2Test                # B8, FIM evaluated at x_hat
    nees_true_fim: float                # B8 variant, FIM evaluated at x_true

    @property
    def nees_valid(self) -> bool:
        """Whether the NEES test is meaningful.

        Only if the FIM at the estimate is invertible.  Otherwise the estimate
        lies somewhere along an unobservable direction, the error along it is
        arbitrary, and ``x_tilde' J x_tilde`` follows no known distribution.
        """
        return self.estimate.observability.invertible

    @property
    def error(self) -> np.ndarray:
        """Estimation error ``x_tilde = x_true - x_hat``."""
        return self.scenario.x_true - self.mle.x

    def to_dict(self) -> dict:
        """JSON-friendly representation (consumed by DBEAR)."""
        s = self.scenario
        return {
            "scenario": s.to_dict(),
            "times": s.times.tolist(),
            "platform_track": s.platform_track.tolist(),
            "measurements": self.z.tolist(),
            "truth": self.truth.to_dict(),
            "mle": {
                "x": self.mle.x.tolist(),
                "cost": self.mle.cost,
                "iterations": self.mle.iterations,
                "converged": self.mle.converged,
                "message": self.mle.message,
                "method": self.mle.method,
                "history": [h.tolist() for h in self.mle.history],
            },
            "estimate": self.estimate.to_dict(),
            "fit": None if self.fit is None else _chi2_dict(self.fit),
            "nees": _chi2_dict(self.nees),
            "nees_valid": self.nees_valid,
            "nees_true_fim": self.nees_true_fim,
        }


def _chi2_dict(test: stats.Chi2Test) -> dict:
    return {
        "value": test.value,
        "dof": test.dof,
        "lower": test.lower,
        "upper": test.upper,
        "confidence": test.confidence,
        "passed": test.passed,
    }


def run_analysis(
    scenario: Scenario,
    rng: np.random.Generator | None = None,
    z: np.ndarray | None = None,
) -> AnalysisResult:
    """Run steps B1-B8 for ``scenario``.

    Parameters
    ----------
    scenario : Scenario
        Inputs A1-A7.
    rng : numpy.random.Generator, optional
        Random generator for the measurement noise.  Defaults to
        ``numpy.random.default_rng(scenario.seed)``.
    z : array_like, optional
        Use these bearings instead of simulating new ones (skips the noise
        generation of B4).
    """
    if rng is None:
        rng = np.random.default_rng(scenario.seed)
    t, plat = scenario.times, scenario.platform_track

    truth = analyze_fim(scenario.x_true, scenario)                      # B1-B3
    if z is None:
        z = scenario.simulate_measurements(rng)                         # B4
    z = np.asarray(z, dtype=float)
    mle = estimation.maximum_likelihood(                                # B5
        z, t, plat, scenario.r, scenario.x_init,
        method=scenario.method, max_iter=scenario.max_iter, tol=scenario.tol,
    )
    est = analyze_fim(mle.x, scenario)                                  # B6
    # The bearings determine rank(J) independent parameter combinations: 4 if
    # observable, 3 without a platform maneuver.  That many are fitted, so the
    # residuals keep n - rank(J) degrees of freedom.
    dof = scenario.n - max(est.observability.rank, 1)
    fit = stats.chi2_test(mle.cost, dof, scenario.confidence) if dof >= 1 else None  # B7
    # NEES ~ chi2(4) for an efficient estimator; only meaningful when J(x_hat)
    # is invertible (see AnalysisResult.nees_valid).
    nees_test = stats.chi2_test(                                        # B8
        stats.nees(scenario.x_true, mle.x, est.J), model.N_PARAMS, scenario.confidence
    )
    nees_true = stats.nees(scenario.x_true, mle.x, truth.J)
    return AnalysisResult(scenario, truth, z, mle, est, fit, nees_test, nees_true)


@dataclass
class MonteCarloResult:
    """Summary of :func:`monte_carlo`.

    Attributes
    ----------
    nees : numpy.ndarray
        NEES of every run (FIM at the estimate).
    fit_errors : numpy.ndarray
        Fitting error of every run.
    converged : numpy.ndarray of bool
        Convergence flag of every run.
    diverged : numpy.ndarray of bool
        Runs whose range estimate ran off to infinity (no finite MLE).
    valid : numpy.ndarray of bool
        Runs used for the NEES average: converged, with an invertible FIM at
        the estimate.
    avg_nees : bearloc.stats.Chi2Test or None
        Average NEES over the N valid runs.  If the estimator is efficient,
        ``N * avg_nees ~ chi2(4 N)``, so the interval is that of
        ``chi2(4 N) / N`` (``avg_nees.dof`` holds ``4 N``).  ``None`` if no
        run is valid (e.g. an unobservable scenario).
    """

    nees: np.ndarray
    fit_errors: np.ndarray
    converged: np.ndarray
    diverged: np.ndarray
    valid: np.ndarray
    avg_nees: stats.Chi2Test | None

    @property
    def runs(self) -> int:
        """Total number of runs."""
        return int(self.nees.size)


def monte_carlo(
    scenario: Scenario, runs: int, rng: np.random.Generator | None = None
) -> MonteCarloResult:
    """Repeat B4, B5 and B8 over ``runs`` independent noise realizations.

    The average NEES over the ``N`` valid runs (converged, FIM at the estimate
    invertible) is compared with the two-sided interval of ``chi2(4 N) / N``.
    """
    if runs < 1:
        raise ValueError(f"runs must be >= 1, got {runs}")
    if rng is None:
        rng = np.random.default_rng(scenario.seed)
    t, plat = scenario.times, scenario.platform_track
    nees = np.empty(runs)
    fit_errors = np.empty(runs)
    converged = np.zeros(runs, dtype=bool)
    diverged = np.zeros(runs, dtype=bool)
    invertible = np.zeros(runs, dtype=bool)
    for i in range(runs):
        z = scenario.simulate_measurements(rng)
        mle = estimation.maximum_likelihood(
            z, t, plat, scenario.r, scenario.x_init,
            method=scenario.method, max_iter=scenario.max_iter, tol=scenario.tol,
        )
        J_hat = model.fisher_information(mle.x, t, plat, scenario.r)
        nees[i] = stats.nees(scenario.x_true, mle.x, J_hat)
        fit_errors[i] = mle.cost
        converged[i] = mle.converged
        diverged[i] = mle.diverged
        invertible[i] = stats.check_observability(J_hat).invertible
    valid = converged & invertible
    n_valid = int(valid.sum())
    avg = None
    if n_valid > 0:
        dof = model.N_PARAMS * n_valid
        lo, hi = stats.chi2_interval(dof, scenario.confidence)
        avg = stats.Chi2Test(float(nees[valid].mean()), dof, lo / n_valid, hi / n_valid,
                             scenario.confidence)
    return MonteCarloResult(nees, fit_errors, converged, diverged, valid, avg)


def save_results(result: AnalysisResult, path: str | Path) -> None:
    """Write an :class:`AnalysisResult` to a JSON file for DBEAR."""

    def default(obj):  # numpy scalars / infinities
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"cannot serialize {type(obj).__name__}")

    data = result.to_dict()
    Path(path).write_text(json.dumps(data, indent=1, default=default), encoding="utf-8")


def load_results(path: str | Path) -> dict:
    """Read a JSON results file written by :func:`save_results`."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"results file not found: {path} (run bear.py first)")
    return json.loads(path.read_text(encoding="utf-8"))
