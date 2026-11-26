"""Reproduce every experiment, figure and number of the project report.

Usage (from the project root)::

    python report/make_figures.py              # PDF figures + generated tables/macros
    python report/make_figures.py --png DIR    # also PNG previews in DIR

Writes

* ``report/figures/*.pdf``     -- the figures included by ``report.tex``;
* ``report/generated/*.tex``   -- result macros (``results.tex``) and tables.

All random numbers are seeded, so the output is identical on every run.  The
experiments take about one to two minutes.
"""

from __future__ import annotations

import re
import sys
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats as sps  # noqa: E402

from bearloc import analysis, estimation, model, stats  # noqa: E402
from bearloc.ellipse import ellipse_axes, ellipse_probability  # noqa: E402
from bearloc.model import Maneuver  # noqa: E402
from bearloc.plotting import ellipse_specs, plot_ellipses, plot_geometry  # noqa: E402
from bearloc.samples import sample_names, sample_scenario  # noqa: E402
from bearloc.scenario import load_defaults  # noqa: E402

FIG = ROOT / "report" / "figures"
GEN = ROOT / "report" / "generated"
FIG.mkdir(parents=True, exist_ok=True)
GEN.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- figure style
# Categorical slots 1-3 of the validated reference palette; ink and chrome.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SHADE = "#f0efec"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.linewidth": 0.6,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "grid.linestyle": "-",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "lines.linewidth": 1.6,
    "lines.markersize": 4.5,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})
FULL_W, HALF_W = 6.3, 3.1  # inches (text width of the report)

MACROS: dict[str, str] = {}


def macro(name: str, value, fmt: str = "{}") -> None:
    """Record a LaTeX macro ``\\name`` -> formatted value."""
    assert name.isalpha(), name
    MACROS[name] = fmt.format(value)


#: Optional folder for PNG previews (``--png DIR`` on the command line).
PNG_DIR = Path(sys.argv[sys.argv.index("--png") + 1]) if "--png" in sys.argv else None


def save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.pdf")
    if PNG_DIR is not None:
        PNG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PNG_DIR / f"{name}.png", dpi=110)
    plt.close(fig)
    print(f"  figure {name}.pdf")


def write_table(name: str, rows: list[str]) -> None:
    """Write table rows; a hyphen before a digit outside math becomes a minus sign."""
    rows = [re.sub(r"(?<![\w$^{\\])-(?=\d)", "$-$", row) for row in rows]
    (GEN / f"{name}.tex").write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"  table  {name}.tex")


def tex_num(x: float, digits: int = 3) -> str:
    """Number for LaTeX text: plain for moderate values, a \\times 10^{k} form otherwise."""
    if x == 0 or 1e-3 <= abs(x) < 1e5:
        return np.format_float_positional(x, precision=digits, unique=False, fractional=False, trim="-")
    mant, exp = f"{x:.{digits - 1}e}".split("e")
    return f"{mant}\\times10^{{{int(exp)}}}"


S = load_defaults(ROOT / "BEAR.DEF")
T_, PLAT, X = S.times, S.platform_track, S.x_true
J_TRUE = model.fisher_information(X, T_, PLAT, S.r)
P_TRUE = np.linalg.inv(J_TRUE)
STD_TRUE = np.sqrt(np.diag(P_TRUE))
NAMES = (r"$\xi_0$", r"$\eta_0$", r"$\dot\xi$", r"$\dot\eta$")
UNITS = ("m", "m", "m/s", "m/s")


# ============================================================ E1 single run
def e1_single_run():
    print("E1  single run of the default scenario")
    res = analysis.run_analysis(S)
    obs = res.truth.observability
    macro("resCond", tex_num(obs.condition, 4))
    macro("resIters", res.mle.iterations)
    macro("resFit", res.fit.value, "{:.2f}")
    macro("resFitDof", res.fit.dof)
    macro("resFitLo", res.fit.lower, "{:.2f}")
    macro("resFitHi", res.fit.upper, "{:.2f}")
    macro("resNees", res.nees.value, "{:.3f}")
    macro("resNeesTrue", res.nees_true_fim, "{:.3f}")
    std_hat = np.sqrt(np.diag(res.estimate.P))
    rows = []
    for i in range(4):
        rows.append(
            f"{NAMES[i]} & {UNITS[i]} & {S.x_init[i]:.6g} & {res.mle.x[i]:.6g} & {X[i]:.6g} & "
            f"{res.error[i]:.3g} & {std_hat[i]:.3g} & {STD_TRUE[i]:.3g} & "
            f"{abs(res.error[i]) / std_hat[i]:.2f} \\\\"
        )
    write_table("tab_estimate", rows)

    # convergence histories (cost per iteration) of both methods
    z = res.z
    gn = estimation.gauss_newton(z, T_, PLAT, S.r, S.x_init)
    bf = estimation.bfgs(z, T_, PLAT, S.r, S.x_init)
    c_gn = [estimation.cost(x, z, T_, PLAT, S.r) for x in gn.history]
    c_bf = [estimation.cost(x, z, T_, PLAT, S.r) for x in bf.history]
    c_min = min(c_gn[-1], c_bf[-1])
    fig, ax = plt.subplots(figsize=(HALF_W, 2.3))
    ax.semilogy(range(len(c_gn)), np.maximum(np.array(c_gn) - c_min, 1e-12), "-o", color=BLUE,
                label="Gauss–Newton / LM")
    ax.semilogy(range(len(c_bf)), np.maximum(np.array(c_bf) - c_min, 1e-12), "-s", color=ORANGE,
                markersize=3, label="BFGS")
    ax.set_xlabel("iteration")
    ax.set_ylabel(r"$\lambda(x^i) - \lambda_{\min}$")
    ax.legend(loc="lower center", bbox_to_anchor=(0.6, 0.02))  # clear of both curves' ends
    save(fig, "fig_convergence")
    macro("resItersBfgs", bf.iterations)

    # geometry and ellipses, exactly as drawn by BEAR / DBEAR
    fig = plot_geometry(S, x_hat=res.mle.x, z=res.z, num_rays=8)
    fig.set_size_inches(HALF_W + 0.2, 3.3)
    for ax in fig.axes:
        ax.set_title("Geometry, measured bearings and estimate", fontsize=8.5)
        ax.legend(fontsize=6.5, loc="lower right")
    save(fig, "fig_geometry")
    data = res.to_dict()
    specs = ellipse_specs(data, [10, 160, 300, 450, 600], 2.0, "estimate")
    fig = plot_ellipses(data, specs, zoom=True)
    fig.set_size_inches(HALF_W + 0.2, 3.3)
    for ax in fig.axes:
        ax.set_title("2-sigma position ellipses (DBEAR)", fontsize=8.5)
        ax.legend(fontsize=6.5, loc="lower left")
    save(fig, "fig_ellipses")
    rows = []
    for e in specs:
        a, b = e.semi_axes
        rows.append(f"{e.t:g} & {e.center[0]:.0f} & {e.center[1]:.0f} & {a:.0f} & {b:.0f} & "
                    f"{e.orientation_deg:.1f} & {'yes' if e.contains_truth else 'no'} \\\\")
    write_table("tab_ellipses", rows)
    return res


# ============================================================ E2 Monte Carlo consistency
def e2_monte_carlo(n_runs: int = 2000):
    print(f"E2  Monte Carlo consistency ({n_runs} runs)")
    rng = np.random.default_rng(2026)
    errs, fits, nees, iters, cover = [], [], [], [], {1: [], 2: [], 3: []}
    t_ell = 300.0
    t0 = time.perf_counter()
    for _ in range(n_runs):
        z = S.simulate_measurements(rng)
        res = estimation.gauss_newton(z, T_, PLAT, S.r, S.x_init)
        J_hat = model.fisher_information(res.x, T_, PLAT, S.r)
        err = X - res.x
        errs.append(err)
        fits.append(res.cost)
        nees.append(err @ J_hat @ err)
        iters.append(res.iterations)
        C = model.position_covariances(np.linalg.inv(J_hat), t_ell)
        dp = model.target_position(err, t_ell)
        q = dp @ np.linalg.solve(C, dp)
        for g in cover:
            cover[g].append(q <= g * g)
    elapsed = time.perf_counter() - t0
    errs, fits, nees = np.array(errs), np.array(fits), np.array(nees)
    dof = S.n - 4
    macro("mcRuns", n_runs)
    macro("mcMsPerRun", 1e3 * elapsed / n_runs, "{:.1f}")
    macro("mcItersMean", np.mean(iters), "{:.1f}")
    macro("mcItersMax", np.max(iters))
    macro("mcBiasMax", np.max(np.abs(errs.mean(0) / STD_TRUE)), "{:.3f}")
    ratio = errs.var(0) / np.diag(P_TRUE)
    macro("mcVarRatioMin", ratio.min(), "{:.2f}")
    macro("mcVarRatioMax", ratio.max(), "{:.2f}")
    macro("mcFitMean", fits.mean(), "{:.2f}")
    macro("mcFitVar", fits.var(), "{:.1f}")
    lo, hi = stats.chi2_interval(dof)
    macro("mcFitOut", 100 * np.mean((fits < lo) | (fits > hi)), "{:.1f}")
    macro("mcNeesMean", nees.mean(), "{:.3f}")
    macro("mcNeesVar", nees.var(), "{:.2f}")
    lo4, hi4 = stats.chi2_interval(4)
    macro("mcNeesOut", 100 * np.mean((nees < lo4) | (nees > hi4)), "{:.1f}")
    macro("mcKsNees", sps.kstest(nees, "chi2", args=(4,)).pvalue, "{:.2f}")
    macro("mcKsFit", sps.kstest(fits, "chi2", args=(dof,)).pvalue, "{:.2f}")
    for g, name in ((1, "One"), (2, "Two"), (3, "Three")):
        macro(f"mcCover{name}", 100 * np.mean(cover[g]), "{:.1f}")
        macro(f"theoryCover{name}", 100 * ellipse_probability(g), "{:.1f}")
    z2 = np.mean(np.abs(errs) / STD_TRUE > 2, axis=0)
    macro("mcZtwoMin", 100 * z2.min(), "{:.1f}")
    macro("mcZtwoMax", 100 * z2.max(), "{:.1f}")
    # avg NEES over the MC runs and its interval
    lo_a, hi_a = stats.chi2_interval(4 * n_runs)
    macro("mcAvgLo", lo_a / n_runs, "{:.3f}")
    macro("mcAvgHi", hi_a / n_runs, "{:.3f}")

    # per-parameter table
    rows = []
    for i in range(4):
        rows.append(f"{NAMES[i]} & {STD_TRUE[i]:.3g} & {errs[:, i].mean():.3g} & "
                    f"{errs[:, i].mean() / STD_TRUE[i]:.3f} & {errs[:, i].std():.3g} & {ratio[i]:.3f} \\\\")
    write_table("tab_mc_params", rows)

    # histograms vs theory
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.3))
    ax = axes[0]
    ax.hist(nees, bins=np.linspace(0, 16, 41), density=True, color=BLUE, alpha=0.85,
            edgecolor="white", linewidth=0.6, label=f"NEES, {n_runs} runs")
    u = np.linspace(0.01, 16, 300)
    ax.plot(u, sps.chi2.pdf(u, 4), color=ORANGE, label=r"$\chi^2_4$ density")
    ax.set_xlabel(r"NEES $\epsilon = \tilde x' J \tilde x$")
    ax.set_ylabel("density")
    ax.legend()
    ax = axes[1]
    ax.hist(fits, bins=40, density=True, color=BLUE, alpha=0.85, edgecolor="white",
            linewidth=0.6, label="fitting error")
    u = np.linspace(fits.min(), fits.max(), 300)
    ax.plot(u, sps.chi2.pdf(u, dof), color=ORANGE, label=fr"$\chi^2_{{{dof}}}$ density")
    ax.set_xlabel(r"fitting error $\lambda(\hat x)$")
    ax.legend()
    fig.tight_layout()
    save(fig, "fig_consistency")


# ============================================================ E3 observability & CRLB over time
def e3_observability():
    print("E3  observability and CRLB over time")
    u = sample_scenario("No maneuver (unobservable)")
    J_u = model.fisher_information(u.x_true, u.times, u.platform_track, u.r)
    sv_obs = stats.check_observability(J_TRUE).singular_values
    sv_un = stats.check_observability(J_u).singular_values
    rows = [
        "with maneuver & " + " & ".join(f"${tex_num(v)}$" for v in sv_obs)
        + f" & {stats.check_observability(J_TRUE).rank} & ${tex_num(sv_obs[0] / sv_obs[-1])}$ \\\\",
        "no maneuver & " + " & ".join(f"${tex_num(v)}$" for v in sv_un)
        + f" & {stats.check_observability(J_u).rank} & $\\infty$ \\\\",
    ]
    write_table("tab_singular", rows)
    v = np.concatenate([u.x_true[:2] - u.platform_pos, u.x_true[2:] - u.platform_vel])
    macro("nullResidual", tex_num(np.linalg.norm(J_u @ v) / (np.linalg.norm(J_u) * np.linalg.norm(v)), 2))
    macro("plainCondM", tex_num(np.linalg.cond(J_TRUE), 2))
    D = np.diag([1e-3, 1e-3, 1.0, 1.0])
    J_km = np.linalg.inv(D) @ J_TRUE @ np.linalg.inv(D)
    macro("plainCondKm", tex_num(np.linalg.cond(J_km), 2))
    macro("scaledCond", tex_num(stats.check_observability(J_km).condition, 4))

    # 1-sigma semi-axes and line-of-sight alignment over time
    tt = np.linspace(0, 800, 401)
    major, minor, misalign = [], [], []
    for t in tt:
        C = model.position_covariances(P_TRUE, t)
        ax_ = ellipse_axes(C)
        major.append(np.sqrt(ax_.lam1))
        minor.append(np.sqrt(ax_.lam2))
        d = model.target_position(X, t) - S.platform_at(t)
        los = np.degrees(np.arctan2(d[1], d[0]))
        misalign.append(abs((np.degrees(ax_.theta) - los + 90) % 180 - 90))
    major, minor = np.array(major), np.array(minor)
    fig, ax = plt.subplots(figsize=(FULL_W, 2.4))
    m = S.maneuvers[0]
    ax.axvspan(m.k1 * S.T, m.k2 * S.T, color=SHADE, zorder=0)
    ax.text((m.k1 + m.k2) * S.T / 2, 2.2e3, "maneuver", ha="center", color=INK2, fontsize=8)
    ax.axvline(T_[-1], color=AXIS, linewidth=0.8)
    ax.text(T_[-1] + 6, 2.2e3, "last bearing", color=INK2, fontsize=8)
    ax.semilogy(tt, major, color=BLUE, label=r"semi-major $\sqrt{\lambda_1}$ (≈ range)")
    ax.semilogy(tt, minor, color=ORANGE, linestyle=(0, (5, 2)), label=r"semi-minor $\sqrt{\lambda_2}$ (≈ cross-range)")
    ax.set_xlabel("time $t$ [s]")
    ax.set_ylabel(r"1$\sigma$ position std [m]")
    ax.set_ylim(5, 4e3)
    ax.legend(loc="lower left")
    save(fig, "fig_crlb_time")
    misalign = np.array(misalign)
    macro("losMisalignMax", np.max(misalign[tt <= T_[-1]]), "{:.1f}")
    macro("losMisalignEnd", misalign[-1], "{:.0f}")
    macro("aspectEightHundred", major[-1] / minor[-1], "{:.1f}")
    k_min = int(np.argmin(major[tt <= T_[-1]]))
    macro("majorMinT", tt[k_min], "{:.0f}")
    macro("majorMin", major[k_min], "{:.0f}")
    macro("majorStart", major[0], "{:.0f}")
    macro("aspectAtEnd", major[tt == 600][0] / minor[tt == 600][0], "{:.0f}")


# ============================================================ E4 maneuver design
def e4_maneuver_design():
    print("E4  maneuver design sweeps")
    t_end = T_[-1]

    def final_std(scen, t_eval=t_end):
        J = model.fisher_information(X, scen.times, scen.platform_track, scen.r)
        if not stats.check_observability(J).invertible:
            return np.inf
        C = model.position_covariances(np.linalg.inv(J), t_eval)
        return float(np.sqrt(np.trace(C)))

    def turn(psi):
        return S.copy(maneuvers=[Maneuver(15 * np.sin(np.radians(psi)), 15 * np.cos(np.radians(psi)), 20, 40)])

    def cond(scen):
        J = model.fisher_information(X, scen.times, scen.platform_track, scen.r)
        return stats.check_observability(J).condition

    psis = np.r_[np.arange(3, 30, 1), np.arange(30, 181, 5)]
    by_angle = [final_std(turn(p)) for p in psis]
    starts = np.arange(0, 41, 2)
    by_start = [final_std(S.copy(maneuvers=[Maneuver(15, 0, int(k), int(k) + 20)])) for k in starts]
    by_start0 = [final_std(S.copy(maneuvers=[Maneuver(15, 0, int(k), int(k) + 20)]), 0.0) for k in starts]
    durations = np.arange(1, 41, 1)
    by_dur = [final_std(S.copy(maneuvers=[Maneuver(15, 0, 20, 20 + int(d))])) for d in durations]
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.2), sharey=True)
    axes[0].semilogy(psis, by_angle, "-", color=BLUE)
    axes[0].set_xlabel(r"turn angle $\psi$ [deg]")
    axes[0].set_ylabel(r"$\sqrt{\mathrm{tr}\,C(t_n)}$ [m]")
    axes[0].set_title("(a) heading change, legs 20–40")
    axes[1].semilogy(starts * S.T, by_start, "-o", color=BLUE, markersize=3, label=r"at $t_n$")
    axes[1].semilogy(starts * S.T, by_start0, "--s", color=ORANGE, markersize=3, label=r"at $t=0$")
    axes[1].legend(loc="upper center", fontsize=7)
    axes[1].set_xlabel("start of 200 s turn [s]")
    axes[1].set_title("(b) timing of a 90° turn")
    axes[2].semilogy(durations * S.T, by_dur, "-", color=BLUE)
    axes[2].set_xlabel("duration of turn [s]")
    axes[2].set_title("(c) duration, start 200 s")
    for ax in axes:
        ax.title.set_fontsize(8)
    fig.tight_layout()
    save(fig, "fig_maneuver")
    by_angle = np.array(by_angle)
    macro("angleNinety", by_angle[psis == 90][0], "{:.0f}")
    big = psis >= 30
    macro("angleBest", by_angle[big].min(), "{:.0f}")
    macro("angleBestPsi", psis[big][np.argmin(by_angle[big])])
    small = psis < 30
    peak = int(np.argmax(by_angle[small]))
    macro("anglePeakPsi", psis[small][peak])
    macro("anglePeak", by_angle[small][peak], "{:.0f}")
    macro("angleDipPsi", psis[small][np.argmin(by_angle[small][:peak])])
    macro("angleDip", by_angle[small][:peak].min(), "{:.0f}")
    macro("condPeak", tex_num(cond(turn(psis[small][peak])), 2))
    macro("condNinety", tex_num(cond(turn(90)), 2))
    macro("condFive", tex_num(cond(turn(5)), 2))
    by_dur = np.array(by_dur)
    macro("durPeakS", durations[np.argmax(by_dur)] * S.T, "{:.0f}")
    macro("durPeak", by_dur.max(), "{:.0f}")
    macro("startBest", starts[np.argmin(by_start)] * S.T, "{:.0f}")
    macro("startBestVal", min(by_start), "{:.0f}")
    macro("startWorst", starts[np.argmax(by_start)] * S.T, "{:.0f}")
    macro("startWorstVal", max(by_start), "{:.0f}")
    macro("startBestZero", starts[np.argmin(by_start0)] * S.T, "{:.0f}")
    macro("startBestZeroVal", min(by_start0), "{:.0f}")
    macro("startLateZeroVal", by_start0[-1], "{:.0f}")
    macro("durShort", by_dur[0], "{:.0f}")
    macro("durLong", by_dur[-1], "{:.0f}")


# ============================================================ E5 noise sweep
def e5_noise(n_runs: int = 400):
    print(f"E5  noise sweep ({n_runs} runs per level)")
    sigmas = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0]
    rng = np.random.default_rng(77)
    rmse_ratio, med_nees, mean_nees, div_pct, conv_pct = [], [], [], [], []
    rows = []
    for sig in sigmas:
        sc = S.copy(r=float(np.radians(sig) ** 2))
        Jt = model.fisher_information(X, T_, PLAT, sc.r)
        crlb_pos = np.sqrt(np.trace(model.position_covariances(np.linalg.inv(Jt), T_[-1])))
        e_pos, nees, n_div, n_conv = [], [], 0, 0
        for _ in range(n_runs):
            z = sc.simulate_measurements(rng)
            res = estimation.gauss_newton(z, T_, PLAT, sc.r, sc.x_init)
            n_div += res.diverged
            if not res.converged:
                continue
            n_conv += 1
            err = X - res.x
            e_pos.append(np.sum(model.target_position(err, T_[-1]) ** 2))
            Jh = model.fisher_information(res.x, T_, PLAT, sc.r)
            if stats.check_observability(Jh).invertible:
                nees.append(err @ Jh @ err)
        rr = np.sqrt(np.mean(e_pos)) / crlb_pos
        rmse_ratio.append(rr)
        med_nees.append(np.median(nees))
        mean_nees.append(np.mean(nees))
        div_pct.append(100 * n_div / n_runs)
        conv_pct.append(100 * n_conv / n_runs)
        rows.append(f"{sig:g} & {crlb_pos:.0f} & {rr:.2f} & {np.mean(nees):.2f} & {np.median(nees):.2f} & "
                    f"{100 * n_conv / n_runs:.1f} & {100 * n_div / n_runs:.1f} \\\\")
    write_table("tab_noise", rows)
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.2))
    axes[0].plot(sigmas, rmse_ratio, "-o", color=BLUE, markersize=3)
    axes[0].axhline(1.0, color=AXIS, linewidth=0.8)
    axes[0].set_ylabel("RMSE / CRLB")
    axes[0].set_title("(a) efficiency, position at $t_n$")
    axes[1].plot(sigmas, med_nees, "-o", color=BLUE, markersize=3)
    axes[1].axhline(sps.chi2.median(4), color=AXIS, linewidth=0.8)
    axes[1].text(sigmas[0], sps.chi2.median(4) - 0.03, r"$\chi^2_4$ median 3.36", ha="left", va="top",
                 color=INK2, fontsize=7.5)
    axes[1].set_ylim(2.6, 3.7)
    axes[1].set_ylabel("median NEES")
    axes[1].set_title("(b) typical NEES")
    axes[2].plot(sigmas, mean_nees, "-o", color=BLUE, markersize=3)
    axes[2].axhline(4.0, color=AXIS, linewidth=0.8)
    axes[2].text(sigmas[0], 4.12, r"$\chi^2_4$ mean 4", ha="left", va="bottom", color=INK2, fontsize=7.5)
    axes[2].set_ylabel("mean NEES")
    axes[2].set_title("(c) mean NEES (tail)")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(sigmas)
        ax.set_xticklabels([f"{s:g}" for s in sigmas])
        ax.minorticks_off()
        ax.set_xlabel(r"bearing noise $\sigma$ [deg]")
        ax.title.set_fontsize(8)
    fig.tight_layout()
    save(fig, "fig_noise")
    macro("noiseRatioOne", rmse_ratio[sigmas.index(1.0)], "{:.2f}")
    macro("noiseRatioSix", rmse_ratio[-1], "{:.2f}")
    macro("noiseDivMax", max(div_pct), "{:.1f}")
    macro("noiseConvMin", min(conv_pct), "{:.1f}")
    macro("noiseMedNeesSix", med_nees[-1], "{:.1f}")
    macro("noiseMeanNeesSix", mean_nees[-1], "{:.1f}")
    macro("noiseRuns", n_runs)


# ============================================================ E6 range divergence
def e6_divergence(n_runs: int = 600):
    print(f"E6  hard sample and range divergence ({n_runs} runs)")
    h = sample_scenario("High noise, weak maneuver")
    Jt = model.fisher_information(h.x_true, h.times, h.platform_track, h.r)
    macro("hardCond", tex_num(stats.check_observability(Jt).condition, 2))
    rng = np.random.default_rng(7)
    n_div = n_conv = n_lower = 0
    nees = []
    div_draw = ok_draw = None
    for _ in range(n_runs):
        z = h.simulate_measurements(rng)
        res = estimation.gauss_newton(z, h.times, h.platform_track, h.r, h.x_init)
        if res.diverged:
            n_div += 1
            n_lower += res.cost < estimation.cost(h.x_true, z, h.times, h.platform_track, h.r)
            if div_draw is None:
                div_draw = (z, res)
        elif res.converged:
            n_conv += 1
            Jh = model.fisher_information(res.x, h.times, h.platform_track, h.r)
            if stats.check_observability(Jh).invertible:
                e = h.x_true - res.x
                nees.append(e @ Jh @ e)
            if ok_draw is None and abs(np.log(np.hypot(*res.x[:2]) / np.hypot(*h.x_true[:2]))) < 0.3:
                ok_draw = (z, res)
    macro("hardRuns", n_runs)
    macro("hardDivPct", 100 * n_div / n_runs, "{:.1f}")
    macro("hardConvPct", 100 * n_conv / n_runs, "{:.1f}")
    macro("hardLowerPct", 100 * n_lower / max(n_div, 1), "{:.0f}")
    macro("hardMedNees", np.median(nees), "{:.1f}")

    # cost along the ray through the final estimate, relative to its infinite-range limit:
    # a finite MLE exists only if the curve dips below zero somewhere.
    fig, ax = plt.subplots(figsize=(HALF_W + 0.2, 2.3))
    p0, v0 = h.platform_pos, h.platform_vel
    ranges = np.logspace(3.5, 8, 160)
    dips = {}
    for key, (z, res), colour, mark, label in (("ok", ok_draw, BLUE, "o", "draw with a finite MLE"),
                                               ("div", div_draw, ORANGE, "s", "draw that diverged")):
        rel0, relv = res.x[:2] - p0, res.x[2:] - v0
        scale = np.hypot(*rel0)
        rel0, relv = rel0 / scale, relv / scale

        def lam(rho):
            return estimation.cost(np.concatenate([p0 + rho * rel0, v0 + rho * relv]), z,
                                   h.times, h.platform_track, h.r)

        lam_inf = lam(1e12)
        rel = np.array([lam(rho) - lam_inf for rho in ranges])
        dips[key] = rel.min()
        ax.semilogx(ranges, rel, color=colour, label=label, marker=mark, markevery=16, markersize=3.5)
    ax.axhline(0.0, color=AXIS, linewidth=0.8)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylim(-300, 3000)
    ax.text(ranges[-1], 0.15, r"limit $\lambda_\infty$", ha="right", va="bottom", color=INK2, fontsize=7.5)
    ax.set_xlabel("initial range along the estimate's ray [m]")
    ax.set_ylabel(r"$\lambda - \lambda_\infty$")
    ax.legend(loc="upper right")
    save(fig, "fig_divergence")
    macro("rayDipOk", dips["ok"], "{:.1f}")
    macro("rayDipDiv", dips["div"], "{:.2f}")


# ============================================================ E6b predicting divergence
def range_snr(scen) -> float:
    """rho_0 / sigma_rho0: initial range over its CRLB standard deviation (t = 0, before any maneuver)."""
    J = model.fisher_information(scen.x_true, scen.times, scen.platform_track, scen.r)
    d = scen.x_true[:2] - scen.platform_pos
    rho = float(np.hypot(*d))
    grad = np.r_[d / rho, 0.0, 0.0]                      # d rho(0) / dx
    return rho / float(np.sqrt(grad @ np.linalg.inv(J) @ grad))


def e6b_divergence_probability(n_runs: int = 400):
    print(f"E6b predicted vs observed divergence probability ({n_runs} runs per point)")
    rng = np.random.default_rng(99)
    cases = (("hard sample, varying noise", sample_scenario("High noise, weak maneuver"),
              [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], BLUE, "o"),
             ("default scenario, high noise", S, [8.0, 12.0, 16.0, 24.0], ORANGE, "s"))
    fig, ax = plt.subplots(figsize=(HALF_W + 0.2, 2.4))
    s_grid = np.linspace(0.05, 3.2, 200)
    ax.plot(s_grid, 100 * sps.norm.cdf(-s_grid), color=INK2, linewidth=1.2,
            label=r"theory $\Phi(-\rho_0/\sigma_{\rho_0})$")
    rows, max_dev = [], 0.0
    for label, base, sigmas, colour, mark in cases:
        snrs, obs, err = [], [], []
        for sig in sigmas:
            sc = base.copy(r=float(np.radians(sig) ** 2))
            n_div = sum(estimation.gauss_newton(sc.simulate_measurements(rng), sc.times, sc.platform_track,
                                                sc.r, sc.x_init).diverged for _ in range(n_runs))
            p = n_div / n_runs
            s = range_snr(sc)
            snrs.append(s)
            obs.append(100 * p)
            err.append(100 * np.sqrt(max(p * (1 - p), 1e-4) / n_runs))
            pred = 100 * sps.norm.cdf(-s)
            max_dev = max(max_dev, abs(100 * p - pred))
            rows.append(f"{label} & {sig:g} & {s:.2f} & {pred:.1f} & {100 * p:.1f} \\\\")
        ax.errorbar(snrs, obs, yerr=2 * np.array(err), fmt=mark, color=colour, markersize=4,
                    elinewidth=0.8, capsize=2, label=label)
    ax.set_xlabel(r"$\rho_0/\sigma_{\rho_0}$ (initial range / its CRLB std)")
    ax.set_ylabel("runs diverged [%]")
    ax.set_xlim(0, 3.2)
    ax.set_ylim(-2, 52)
    ax.legend(loc="upper right", fontsize=7)
    save(fig, "fig_divprob")
    write_table("tab_divprob", rows)
    macro("divProbRuns", n_runs)
    macro("divProbMaxDev", max_dev, "{:.1f}")
    h = sample_scenario("High noise, weak maneuver")
    macro("hardSnr", range_snr(h), "{:.2f}")
    macro("hardPred", 100 * sps.norm.cdf(-range_snr(h)), "{:.1f}")
    macro("defaultSnr", range_snr(S), "{:.1f}")


# ============================================================ E7/E8 initial guess and methods
def e7_init_and_methods(n_runs: int = 100):
    print(f"E7  initial-guess robustness ({n_runs} draws per point, both methods)")
    factors = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    rng = np.random.default_rng(31)
    draws = [S.simulate_measurements(rng) for _ in range(n_runs)]
    refs = [estimation.gauss_newton(z, T_, PLAT, S.r, X).x for z in draws]
    rho_true = np.hypot(*(model.target_position(X, T_[0]) - PLAT[0]))
    success = {"gauss-newton": [], "bfgs": []}
    iters = {"gauss-newton": [], "bfgs": []}
    timing = {"gauss-newton": [], "bfgs": []}
    flag_wrong = {"gauss-newton": 0, "bfgs": 0}
    agree = []
    for f in factors:
        for method in success:
            ok, its = 0, []
            for z, ref in zip(draws, refs):
                b = z[0]
                x0 = np.array([PLAT[0, 0] + f * rho_true * np.cos(b), PLAT[0, 1] + f * rho_true * np.sin(b), 0, 0])
                t0 = time.perf_counter()
                res = estimation.maximum_likelihood(z, T_, PLAT, S.r, x0, method=method)
                timing[method].append(time.perf_counter() - t0)
                # success = the global optimum was found, whatever the method's own flag says
                good = bool(np.all(np.abs(res.x - ref) < 1e-2 * STD_TRUE))
                ok += good
                flag_wrong[method] += good and not res.converged
                if good:
                    its.append(res.iterations)
                if method == "bfgs" and f == 1.0:
                    agree.append(np.max(np.abs(res.x - ref) / STD_TRUE))
            success[method].append(100 * ok / n_runs)
            iters[method].append(np.mean(its) if its else np.nan)
    fig, ax = plt.subplots(figsize=(HALF_W, 2.3))
    ax.semilogx(factors, iters["gauss-newton"], "-o", color=BLUE, label="Gauss–Newton / LM")
    ax.semilogx(factors, iters["bfgs"], "--s", color=ORANGE, markersize=3.5, label="BFGS")
    ax.set_xticks(factors[::2])
    ax.set_xticklabels([f"{f:g}" for f in factors[::2]])
    ax.minorticks_off()
    ax.set_ylim(0, 72)
    ax.set_xlabel("initial range / true range")
    ax.set_ylabel("mean iterations")
    ax.legend(loc="upper center", ncol=2)
    save(fig, "fig_init")
    macro("initSuccessMin", min(min(success["gauss-newton"]), min(success["bfgs"])), "{:.0f}")
    rows = []
    for i, f in enumerate(factors):
        rows.append(f"{f:g} & {success['gauss-newton'][i]:.0f} & {iters['gauss-newton'][i]:.1f} & "
                    f"{success['bfgs'][i]:.0f} & {iters['bfgs'][i]:.1f} \\\\")
    write_table("tab_init", rows)
    macro("initRuns", n_runs)
    macro("flagWrongGn", flag_wrong["gauss-newton"])
    macro("flagWrongBfgs", flag_wrong["bfgs"])
    macro("initTotal", n_runs * len(factors))
    macro("gnMs", 1e3 * np.mean(timing["gauss-newton"]), "{:.1f}")
    macro("bfgsMs", 1e3 * np.mean(timing["bfgs"]), "{:.1f}")
    macro("bfgsAgree", tex_num(max(agree), 2))
    macro("gnItersTrue", iters["gauss-newton"][factors.index(1.0)], "{:.1f}")
    macro("bfgsItersTrue", iters["bfgs"][factors.index(1.0)], "{:.1f}")


# ============================================================ samples, tests, code size
def e9_samples_tests_code():
    print("E9  samples, tests and code size")
    rows = []
    for name in sample_names():
        sc = sample_scenario(name)
        res = analysis.run_analysis(sc)
        obs = res.truth.observability
        cond = tex_num(obs.condition, 2) if obs.invertible else r"\infty"
        if res.mle.diverged:
            b5 = "diverged"
        elif not res.mle.converged:
            b5 = "not conv."
        elif not res.estimate.observability.invertible:
            b5 = "not unique"
        else:
            b5 = f"{res.mle.iterations} it."
        fit = f"{res.fit.value:.1f} ({'pass' if res.fit.passed else 'fail'})" if res.fit else "n/a"
        nees = (f"{res.nees.value:.2f} ({'pass' if res.nees.passed else 'fail'})"
                if res.nees_valid else "n/m")
        rows.append(f"{name} & ${cond}$ & {b5} & {fit} & {nees} \\\\")
    write_table("tab_samples", rows)

    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), top_level_dir=str(ROOT))
    counts: dict[str, int] = {}

    def walk(s):
        for item in s:
            if isinstance(item, unittest.TestSuite):
                walk(item)
            else:
                mod = type(item).__module__.split(".")[-1]
                counts[mod] = counts.get(mod, 0) + 1

    walk(suite)
    macro("testsTotal", sum(counts.values()))
    rows = [f"\\texttt{{{m.replace('_', chr(92) + '_')}}} & {c} \\\\" for m, c in sorted(counts.items())]
    write_table("tab_tests", rows)

    def loc(paths):
        return sum(len(p.read_text(encoding="utf-8").splitlines()) for p in paths)

    core = sorted((ROOT / "bearloc").glob("*.py"))
    gui = sorted((ROOT / "bearloc" / "gui").glob("*.py"))
    apps = [ROOT / n for n in ("bear.py", "dbear.py", "bear_gui.py", "dbear_gui.py", "build_exe.py")]
    tests = sorted((ROOT / "tests").glob("*.py"))
    macro("locCore", loc(core))
    macro("locGui", loc(gui))
    macro("locApps", loc(apps))
    macro("locTests", loc(tests))


def main() -> None:
    t0 = time.time()
    e1_single_run()
    e2_monte_carlo()
    e3_observability()
    e4_maneuver_design()
    e5_noise()
    e6_divergence()
    e6b_divergence_probability()
    e7_init_and_methods()
    e9_samples_tests_code()
    lines = ["% Generated by report/make_figures.py -- do not edit by hand."]
    lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(MACROS.items())]
    (GEN / "results.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(MACROS)} macros to report/generated/results.tex in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
