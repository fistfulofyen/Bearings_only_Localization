"""Plain-text formatting of BEAR results (what the BEAR program displays)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from . import model
from .analysis import AnalysisResult, FIMAnalysis, MonteCarloResult
from .scenario import Scenario

#: Names of the parameter components, in order.
PARAM_NAMES = ("xi0", "eta0", "xi_dot", "eta_dot")

_RULE = "=" * 78


def heading(title: str) -> str:
    """Section heading."""
    return f"\n{_RULE}\n{title}\n{_RULE}"


def format_matrix(M: np.ndarray, labels: Sequence[str] = PARAM_NAMES, fmt: str = "{:>13.5g}") -> str:
    """Format a square matrix with row and column labels."""
    M = np.asarray(M, dtype=float)
    width = max(len(s) for s in labels)
    head = " " * (width + 2) + "".join(f"{s:>13}" for s in labels)
    rows = [f"  {lab:<{width}}" + "".join(fmt.format(v) for v in row) for lab, row in zip(labels, M)]
    return "\n".join([head, *rows])


def format_vector(v: np.ndarray, labels: Sequence[str] = PARAM_NAMES) -> str:
    """Format a vector as ``name = value`` pairs on one line."""
    return "  ".join(f"{lab} = {val:.6g}" for lab, val in zip(labels, np.asarray(v, dtype=float)))


def format_scenario(s: Scenario) -> str:
    """Summary of the inputs A1-A7."""
    lines = [
        heading("INPUTS"),
        f"A1  true parameter x      : {format_vector(s.x_true)}",
        f"A2  sampling period T     : {s.T:g}",
        f"A3  number of samples n   : {s.n}",
        f"A4  noise variance r      : {s.r:.6g} rad^2  (sigma = {np.degrees(np.sqrt(s.r)):.4g} deg)",
        f"A5  platform position     : ({s.platform_pos[0]:g}, {s.platform_pos[1]:g})",
        f"    platform velocity     : ({s.platform_vel[0]:g}, {s.platform_vel[1]:g})",
    ]
    if s.maneuvers:
        for i, m in enumerate(s.maneuvers, 1):
            lines.append(
                f"A6  maneuver leg {i}        : velocity ({m.vx:g}, {m.vy:g}) "
                f"for t in [{m.k1 * s.T:g}, {m.k2 * s.T:g})  (k = {m.k1}..{m.k2})"
            )
    else:
        lines.append("A6  maneuver legs         : none")
    lines += [
        f"A7  initial estimate      : {format_vector(s.x_init)}",
        f"    method / seed         : {s.method} / {s.seed if s.seed is not None else 'random'}",
    ]
    return "\n".join(lines)


def format_observability(fa: FIMAnalysis) -> str:
    """Observability statement for step B1/B6."""
    obs = fa.observability
    if obs.invertible:
        return (
            f"The FIM is INVERTIBLE (rank {obs.rank}/4, scaled condition number "
            f"{obs.condition:.3g}).\n=> The target parameter is OBSERVABLE."
        )
    return (
        f"The FIM is SINGULAR (numerical rank {obs.rank}/4).\n"
        "=> The target parameter is NOT OBSERVABLE: the bearings cannot determine it\n"
        "   uniquely (typically the platform must maneuver to resolve the range)."
    )


def format_inverse(fa: FIMAnalysis) -> str:
    """Inverse FIM with standard deviations and correlation coefficients (B2)."""
    if fa.P is None:
        return "The FIM is singular: its inverse (the CRLB) does not exist."
    std = np.sqrt(np.diag(fa.P))
    corr = fa.P / np.outer(std, std)
    return "\n".join(
        [
            "P = J^-1 (Cramer-Rao lower bound):",
            format_matrix(fa.P),
            "",
            "Standard deviations: " + format_vector(std),
            "Correlation coefficients:",
            format_matrix(corr, fmt="{:>13.4f}"),
        ]
    )


def format_position_covariances(fa: FIMAnalysis, times: np.ndarray, every: int = 1) -> str:
    """Table of position covariance matrices at the sampling times (B3)."""
    if fa.position_covs is None:
        return "Not available: the FIM is singular."
    lines = [
        "Covariance of the target position [xi(t), eta(t)] = A(t) P A(t)'",
        f"{'k':>4} {'t':>9} {'P11':>12} {'P12':>12} {'P22':>12} "
        f"{'sigma_xi':>10} {'sigma_eta':>10} {'corr':>7}",
    ]
    n = len(times)
    for i in range(n):
        if i % every and i != n - 1:
            continue
        C = fa.position_covs[i]
        sx, sy = np.sqrt(C[0, 0]), np.sqrt(C[1, 1])
        lines.append(
            f"{i + 1:>4} {times[i]:>9.4g} {C[0, 0]:>12.5g} {C[0, 1]:>12.5g} {C[1, 1]:>12.5g} "
            f"{sx:>10.4g} {sy:>10.4g} {C[0, 1] / (sx * sy):>7.3f}"
        )
    return "\n".join(lines)


def format_fim_analysis(fa: FIMAnalysis, times: np.ndarray, label: str, every: int = 1) -> str:
    """Steps B1-B3 (or B6) for one parameter value."""
    return "\n".join(
        [
            f"FIM J evaluated at {label}:  {format_vector(fa.x)}",
            format_matrix(fa.J),
            "",
            format_observability(fa),
            "",
            format_inverse(fa),
            "",
            format_position_covariances(fa, times, every),
        ]
    )


def format_measurements(res: AnalysisResult, every: int = 1) -> str:
    """Table of the simulated bearings (B4)."""
    s = res.scenario
    h = model.bearings(s.x_true, s.times, s.platform_track)
    plat = s.platform_track
    lines = [
        f"{'k':>4} {'t':>9} {'platform xi':>12} {'platform eta':>13} "
        f"{'true (deg)':>11} {'meas. (deg)':>12} {'noise (deg)':>12}"
    ]
    for i in range(s.n):
        if i % every and i != s.n - 1:
            continue
        noise = np.degrees(model.wrap_angle(res.z[i] - h[i]))
        lines.append(
            f"{i + 1:>4} {s.times[i]:>9.4g} {plat[i, 0]:>12.6g} {plat[i, 1]:>13.6g} "
            f"{np.degrees(h[i]):>11.4f} {np.degrees(res.z[i]):>12.4f} {noise:>12.4f}"
        )
    return "\n".join(lines)


def format_mle(res: AnalysisResult) -> str:
    """MLE result (B5)."""
    m = res.mle
    lines = [
        f"Method: {m.method};  iterations: {m.iterations};  converged: {'yes' if m.converged else 'NO'}",
        f"Termination: {m.message}",
        "",
        f"{'':>10}{'initial':>14}{'estimate':>14}{'true':>14}{'error':>14}{'std (B6)':>14}",
    ]
    std = (
        np.sqrt(np.diag(res.estimate.P)) if res.estimate.P is not None else np.full(4, np.nan)
    )
    for i, name in enumerate(PARAM_NAMES):
        lines.append(
            f"{name:>10}{res.scenario.x_init[i]:>14.6g}{m.x[i]:>14.6g}"
            f"{res.scenario.x_true[i]:>14.6g}{res.error[i]:>14.4g}{std[i]:>14.4g}"
        )
    if m.diverged:
        lines.append(
            "\nWARNING: the range estimate ran off to infinity.  For these bearings the\n"
            "likelihood has no finite maximum: the platform maneuver is too weak compared\n"
            "with the noise to determine the range.  Use a stronger/longer maneuver, more\n"
            "measurements or less noise.  The numbers below are the last iterate."
        )
    elif not m.converged:
        lines.append("\nWARNING: the minimization did not converge; try another initial estimate.")
    if not res.estimate.observability.invertible:
        lines.append(
            "\nWARNING: the FIM at the estimate is singular, so the minimum is not unique:\n"
            "a whole family of parameters fits the bearings equally well (no range\n"
            "information).  The estimate shown is just one point of that family."
        )
    return "\n".join(lines)


def _verdict(passed: bool) -> str:
    return "WITHIN the probability limits" if passed else "OUTSIDE the probability limits"


def format_fit(res: AnalysisResult) -> str:
    """Fitting error (B7)."""
    if res.fit is None:
        return ("Not available: n <= rank(J) (the number of estimable parameters) leaves\n"
                "no degrees of freedom for the residuals.")
    f = res.fit
    rank = res.scenario.n - f.dof
    return "\n".join(
        [
            "Fitting error  lambda(x_hat) = sum_k [z(k) - h(k, x_hat)]^2 / r",
            f"  value = {f.value:.4f},  degrees of freedom = n - rank(J) = {res.scenario.n} - {rank} = {f.dof}"
            f"  (expected value {f.dof})",
            f"  {100 * f.confidence:g}% interval of chi2({f.dof}): [{f.lower:.3f}, {f.upper:.3f}]",
            f"  => {_verdict(f.passed)}"
            + ("" if f.passed else " (model mismatch or a local minimum?)"),
        ]
    )


def format_nees(res: AnalysisResult) -> str:
    """NEES (B8)."""
    t = res.nees
    return "\n".join(
        [
            "NEES  eps = x_tilde' J x_tilde,  x_tilde = x_true - x_hat",
            f"  with J = J(x_hat) (B6): eps = {t.value:.4f}   (dof = {t.dof}, expected value {t.dof})",
            f"  with J = J(x_true) (B1): eps = {res.nees_true_fim:.4f}",
            f"  {100 * t.confidence:g}% interval of chi2({t.dof}): [{t.lower:.3f}, {t.upper:.3f}]",
            f"  => NEES is {_verdict(t.passed)}"
            if res.nees_valid else
            "  => NOT MEANINGFUL: the FIM at the estimate is singular, so the error along\n"
            "     the unobservable direction is arbitrary.  No test is made.",
        ]
    )


def format_monte_carlo(mc: MonteCarloResult) -> str:
    """Monte Carlo consistency summary."""
    n_ok = int(mc.converged.sum())
    n_valid = int(mc.valid.sum())
    head = f"Runs: {mc.runs}  (converged: {n_ok}, range diverged: {int(mc.diverged.sum())})"
    a = mc.avg_nees
    if a is None:
        return "\n".join([
            head,
            "Average NEES: NOT MEANINGFUL - no converged run has an invertible FIM at its",
            "estimate (unobservable or failed estimation).",
        ] + ([f"Average fitting error: {mc.fit_errors[mc.converged].mean():.3f}"] if n_ok else []))
    return "\n".join(
        [
            head,
            f"Average NEES over {n_valid} valid runs: {a.value:.4f}  (expected 4)",
            f"  {100 * a.confidence:g}% interval of chi2({a.dof})/{n_valid}: [{a.lower:.3f}, {a.upper:.3f}]",
            f"  => average NEES is {_verdict(a.passed)}",
            f"Average fitting error: {mc.fit_errors[mc.converged].mean():.3f}",
        ]
    )


def format_report(res: AnalysisResult, every: int = 1) -> str:
    """Full BEAR report, steps B1-B8."""
    s = res.scenario
    return "\n".join(
        [
            format_scenario(s),
            heading("B1-B3. FISHER INFORMATION, OBSERVABILITY, CRLB AT THE TRUE PARAMETER"),
            format_fim_analysis(res.truth, s.times, "the TRUE parameter", every),
            heading("B4. NOISY BEARING MEASUREMENTS"),
            format_measurements(res, every),
            heading("B5. MAXIMUM LIKELIHOOD ESTIMATE"),
            format_mle(res),
            heading("B6. FISHER INFORMATION, OBSERVABILITY, COVARIANCES AT THE ESTIMATE"),
            format_fim_analysis(res.estimate, s.times, "the ESTIMATE", every),
            heading("B7. FITTING ERROR"),
            format_fit(res),
            heading("B8. NORMALIZED ESTIMATION ERROR SQUARED"),
            format_nees(res),
        ]
    )
