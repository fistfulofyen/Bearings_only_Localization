"""Help texts shown by the "?" icons and the Help window of BEAR and DBEAR.

Every topic has

* ``tip``  -- one or two sentences, shown when the mouse hovers over the "?";
* ``body`` -- the full explanation, shown in the Help window when the "?" is
  clicked (or via Help > Help topics, F1).

This module contains only text (no Tk code), so it can be read and tested
without a display.  Keep it consistent with ``docs/user_guide.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..samples import sample_description, sample_names


@dataclass(frozen=True)
class HelpTopic:
    """One help entry.

    Attributes
    ----------
    group : str
        Heading under which the topic is listed in the Help window.
    title : str
        Topic title.
    tip : str
        Short text for the hover tooltip.
    body : str
        Full explanation (plain text; paragraphs separated by blank lines).
    """

    group: str
    title: str
    tip: str
    body: str


def _samples_body() -> str:
    lines = [
        "Fills every input with a ready-made scenario. Each one illustrates a "
        "different aspect of the problem:",
        "",
    ]
    lines += [f"•  {name} — {sample_description(name)}" for name in sample_names()]
    lines += [
        "",
        "All samples use seed 1, so their results are reproducible. After loading "
        "you can change any value before pressing Run.",
    ]
    return "\n".join(lines)


# ====================================================================== BEAR
BEAR_TOPICS: dict[str, HelpTopic] = {
    # ------------------------------------------------------------ overview
    "overview": HelpTopic(
        "Getting started", "How BEAR works",
        "Estimates a target's position and velocity from bearings measured by a moving platform.",
        "BEAR estimates where a target is and how it moves, using only the directions "
        "(bearings) to it, measured from a moving platform. Range is never measured.\n\n"
        "1.  Enter the scenario on the left, or choose one in “Load sample”. The true "
        "target (A1) is used only to simulate the measurements and to judge the result; "
        "the estimator never sees it.\n\n"
        "2.  Press Run (F5). BEAR then\n"
        "    B1–B3  computes the Fisher information and the best achievable accuracy,\n"
        "    B4       simulates noisy bearings,\n"
        "    B5       computes the maximum likelihood estimate, starting from A7,\n"
        "    B6       repeats B1–B3 at the estimate,\n"
        "    B7       checks how well the bearings are fitted,\n"
        "    B8       checks the estimation error against the claimed accuracy (NEES).\n\n"
        "3.  Read the Summary tab: green = pass, red = fail. The other tabs hold the "
        "detailed numbers.\n\n"
        "4.  Click “Plot uncertainty ellipses (DBEAR)” to see the position uncertainty.\n\n"
        "Hover over any  ?  for a short explanation; click it for the details shown "
        "here. Units are free but must be consistent; the defaults use metres, seconds "
        "and m/s.",
    ),
    # ------------------------------------------------------------ inputs
    "sample": HelpTopic(
        "Inputs", "Load sample",
        "Fill the form with a ready-made scenario.",
        _samples_body(),
    ),
    "A1": HelpTopic(
        "Inputs", "True target parameter (A1)",
        "The target's real position and velocity at t = 0. Used to simulate the "
        "measurements and to compute the error; the estimator does not see it.",
        "x = [ξ0, η0, dξ/dt, dη/dt]\n\n"
        "•  ξ0, η0 — target position at t = 0 (e.g. metres). ξ is the horizontal axis "
        "(think east), η the vertical axis (think north).\n"
        "•  dξ/dt, dη/dt — constant target velocity (e.g. m/s). The target moves in a "
        "straight line: ξ(t) = ξ0 + t·dξ/dt,  η(t) = η0 + t·dη/dt.\n\n"
        "BEAR uses this “truth” for three things: to simulate the noisy bearings (B4), "
        "to compute the best achievable accuracy (B1–B3) and to compute the estimation "
        "error for the NEES test (B8). The estimator (B5) only sees the bearings and "
        "the initial estimate (A7).\n\n"
        "Example: 5000, 12000, −6, −2 is a target about 13 km away, moving at about "
        "6.3 m/s towards the west-south-west.",
    ),
    "T": HelpTopic(
        "Inputs", "Sampling period T (A2)",
        "Time between two bearing measurements.",
        "Measurements are taken at t_k = k·T for k = 1, 2, …, n. The first one is at "
        "t = T, not at t = 0.\n\n"
        "Use the time unit of your velocities (seconds in the defaults). Maneuver legs "
        "(A6) are given in samples, so a leg from k1 to k2 lasts from k1·T to k2·T.\n\n"
        "For a fixed n, a larger T gives a longer observation window. The geometry "
        "then changes more during the observation, which usually improves the accuracy.",
    ),
    "n": HelpTopic(
        "Inputs", "Number of samples n (A3)",
        "How many bearings are measured. The observation lasts n·T.",
        "The total number of bearing measurements.\n\n"
        "The estimator fits 4 numbers, so the fitting-error test (B7) needs at least 5 "
        "measurements; it has n − 4 degrees of freedom.\n\n"
        "Sampling the same time window more densely (smaller T, larger n) shrinks the "
        "uncertainty like 1/√n. A longer observation window, which also changes the "
        "geometry, usually helps more than denser sampling.",
    ),
    "noise": HelpTopic(
        "Inputs", "Measurement noise (A4)",
        "How noisy each bearing is: standard deviation σ in degrees, or variance r in rad².",
        "Every measured bearing is the true bearing plus Gaussian noise with variance "
        "r (rad²). Choose the unit in the drop-down next to the value:\n\n"
        "•  σ (deg) — standard deviation in degrees, e.g. 1 means 1°. Then "
        "r = (σ·π/180)².\n"
        "•  r (rad²) — the variance, as in the book, e.g. 3.046e-4 for 1°.\n\n"
        "Switching the unit converts the value already entered.\n\n"
        "Typical values: 0.1°–0.5° for a good sensor, 1°–3° for a poor one. The "
        "uncertainty of the estimate grows in proportion to σ. The same r is used to "
        "simulate the noise and inside the estimator.",
    ),
    "A5": HelpTopic(
        "Inputs", "Platform initial motion (A5)",
        "Where the platform (the sensor) starts, and its velocity outside the maneuver legs.",
        "Position (ξ, η) of the platform at t = 0, and its velocity (vξ, vη). Outside "
        "the maneuver legs (A6) the platform moves with this velocity.\n\n"
        "Example: position 0, 0 and velocity 0, 15 means the platform starts at the "
        "origin, heading north at 15 m/s.\n\n"
        "Bearings are measured from the platform to the target, counter-clockwise from "
        "the ξ axis: 0° points along +ξ (east) and 90° along +η (north).",
    ),
    "A6": HelpTopic(
        "Inputs", "Platform maneuvers (A6)",
        "Changes of platform velocity. At least one is needed to make the target observable.",
        "Each leg gives the platform velocity (vξ, vη) that is in effect for "
        "k1·T ≤ t < k2·T. Outside every leg, the platform moves with its initial "
        "velocity (A5). Legs may not overlap.\n\n"
        "Why maneuver? With bearings only, a constant-velocity target seen from a "
        "platform that also moves at constant velocity is ambiguous. A target twice as "
        "far away, whose velocity relative to the platform is also twice as large, gives "
        "exactly the same bearings. The range cannot be determined: the target is "
        "unobservable and the Fisher information matrix is singular (B1). A platform "
        "maneuver breaks this ambiguity; larger or longer maneuvers usually give better "
        "accuracy.\n\n"
        "If the maneuver is too weak compared with the noise, the bearings may still not "
        "determine the range: the estimate then runs off to infinity (B5 reports it).\n\n"
        "Editing the table:\n"
        "•  Type vξ, vη, k1 and k2 in the row below the table, then click “Add leg”.\n"
        "•  Click a leg in the table to load it into that row, change it, then click "
        "“Update selected”.\n"
        "•  “Remove selected” deletes the chosen leg. “Remove all” gives the "
        "unobservable case.\n"
        "The “time window” column shows k1·T ≤ t < k2·T.",
    ),
    "A7": HelpTopic(
        "Inputs", "Initial estimate (A7)",
        "Starting guess for the estimator; same four components as A1.",
        "The maximum likelihood estimate (B5) is found by an iterative minimization that "
        "starts from this guess [ξ0, η0, dξ/dt, dη/dt]. In practice it would come from "
        "prior knowledge, for example an assumed range along the first bearing.\n\n"
        "A reasonable guess converges in a few iterations. A poor one can need more "
        "iterations, or end in a wrong local minimum. In that case B5 reports “not "
        "converged”, or the fitting error (B7) comes out too large.\n\n"
        "When the minimization reaches the global minimum, the starting guess does not "
        "change the answer.",
    ),
    "method": HelpTopic(
        "Inputs", "Minimization method",
        "Algorithm used to compute the maximum likelihood estimate.",
        "•  gauss-newton (recommended) — Gauss–Newton with Levenberg–Marquardt "
        "damping. It uses the Fisher information as its curvature, typically converges "
        "in 5–15 iterations, and never increases the cost.\n\n"
        "•  bfgs — the quasi-Newton BFGS method (SciPy) with the exact gradient. It "
        "builds its own curvature estimate from gradients and typically needs about 30 "
        "iterations.\n\n"
        "Both give the same estimate when they converge.",
    ),
    "seed": HelpTopic(
        "Inputs", "Random seed",
        "Makes the random measurement noise repeatable. Leave blank for new noise on every run.",
        "The noise in B4 comes from a random-number generator started from this seed. "
        "The same seed gives exactly the same measurements and results every time, which "
        "is useful to compare settings or to reproduce a result.\n\n"
        "Leave it blank to get new random noise on every run. NEES and the fitting error "
        "then change from run to run, as they should.\n\n"
        "The Monte Carlo runs use seed + 1.",
    ),
    "confidence": HelpTopic(
        "Inputs", "Confidence level",
        "Probability of the acceptance intervals in B7, B8 and the Monte Carlo test (e.g. 0.95).",
        "The fitting error and NEES are random. A test passes when the value lies inside "
        "the two-sided interval that holds this probability of its chi-square "
        "distribution. With 0.95, 2.5 % is cut off at each end.\n\n"
        "Even a perfect estimator falls outside a 95 % interval in about 5 % of runs, so "
        "a single failure is not alarming. A failure of the Monte Carlo test, which "
        "averages many runs, is.",
    ),
    "mc": HelpTopic(
        "Inputs", "Monte Carlo runs",
        "Repeat the estimation this many times with new noise to test statistical "
        "consistency (0 = off).",
        "A single NEES value (B8) says little, because NEES is random. With N > 0, BEAR "
        "also runs N independent noise realizations (B4, B5 and B8 each time) and "
        "averages the NEES.\n\n"
        "If the estimator is consistent, the average is close to 4 and lies inside the "
        "interval of χ²(4N)/N. A few hundred runs take a few seconds. The result appears "
        "in the “Monte Carlo” row of the Summary.",
    ),
    # ------------------------------------------------------------ results
    "B1": HelpTopic(
        "Results", "Observability (B1)",
        "Can the target be located from the bearings at all? Yes if the Fisher "
        "information matrix (FIM) is invertible.",
        "The Fisher information matrix J (4×4) measures how much information the "
        "bearings carry about the four target parameters:\n"
        "J = (1/r) · Σ h_x' h_x, where h_x is the sensitivity of each bearing to the "
        "parameters.\n\n"
        "•  OBSERVABLE — J is invertible. Its inverse is the Cramér–Rao lower bound "
        "(CRLB): the smallest covariance any unbiased estimator can achieve.\n"
        "•  NOT OBSERVABLE — J is singular (rank 3). Some combination of the parameters, "
        "essentially the range, does not affect the bearings. Usually the platform did "
        "not maneuver (A6).\n\n"
        "The scaled condition number shows how close J is to singular. It is computed "
        "after scaling J, so that the units of the parameters do not matter. Larger means "
        "weaker observability. For the samples: about 1e3 (default, accurate), about 1e4 "
        "(zig-zag), about 3e5 (high noise, weak maneuver: the estimate often fails).",
    ),
    "B5": HelpTopic(
        "Results", "Estimation (B5)",
        "Did the maximum likelihood estimation converge?",
        "The maximum likelihood estimate (MLE) is the target parameter that best fits "
        "the measured bearings: it minimizes the sum of squared bearing residuals "
        "divided by r. It is found iteratively, starting from A7.\n\n"
        "•  Converged in k iterations — normal.\n"
        "•  Converged, but the minimum is not unique — the FIM at the estimate is "
        "singular, so many parameters fit equally well (the unobservable case).\n"
        "•  NOT converged — try another initial estimate (A7) or the other method.\n"
        "•  FAILED, range ran off to infinity — for these bearings the likelihood has "
        "no finite maximum: the platform maneuver is too weak compared with the noise to "
        "determine the range. No initial estimate helps; the geometry must improve "
        "(stronger or longer maneuver, more measurements, less noise).",
    ),
    "B7": HelpTopic(
        "Results", "Fitting error (B7)",
        "Do the measurements fit the model? Compares the sum of squared residuals with "
        "its expected χ²(n − 4) range.",
        "Fitting error  λ = Σ (measured − predicted bearing)² / r  at the estimate.\n\n"
        "If the model and the noise level r are right, λ follows a chi-square "
        "distribution with n − 4 degrees of freedom (n measurements minus 4 estimated "
        "parameters). Its expected value is n − 4 and its standard deviation "
        "√(2(n − 4)). In the unobservable case only 3 parameter combinations are "
        "estimated, and n − 3 is used.\n\n"
        "•  within the interval — consistent.\n"
        "•  above — the residuals are too large: r is too small, the target "
        "maneuvers, or the estimator stopped in a wrong local minimum.\n"
        "•  below — the residuals are too small: r is too large, or it is chance.",
    ),
    "B8": HelpTopic(
        "Results", "NEES (B8)",
        "Is the estimation error consistent with the claimed accuracy? NEES should be about 4.",
        "Normalized estimation error squared:  ε = x̃' J x̃, where x̃ = true − "
        "estimated parameter and J is the FIM at the estimate.\n\n"
        "It measures the error in units of the estimator's own claimed uncertainty: "
        "roughly the squared number of sigmas, combined over the four parameters and "
        "their correlations. For an efficient, unbiased estimator with Gaussian errors, "
        "ε follows χ²(4), with mean 4 and 95 % interval [0.48, 11.14].\n\n"
        "When the FIM at the estimate is singular (unobservable), NEES is not "
        "meaningful: the estimate can lie anywhere along the unobservable direction, so "
        "the error along it is arbitrary. BEAR then shows the value but makes no test.\n\n"
        "•  within — the error is as large as the covariance claims.\n"
        "•  above — the error is larger than claimed (the estimator is "
        "overconfident). This is typical of poor geometry or high noise, where the "
        "problem is strongly nonlinear.\n\n"
        "NEES exists only in simulations, because it needs the true value.",
    ),
    "mc_result": HelpTopic(
        "Results", "Monte Carlo test",
        "Average NEES over many runs; should be close to 4.",
        "Shows the average NEES over the converged Monte Carlo runs and its interval, "
        "χ²(4N)/N. Inside the interval means the estimator is consistent: its covariance "
        "(the inverse FIM) correctly describes its actual errors.\n\n"
        "Only valid runs are averaged: converged, with an invertible FIM at the estimate. "
        "Runs whose range estimate ran off to infinity are counted as diverged. Many "
        "diverged runs mean the geometry is too weak.\n\n"
        "The average fitting error shown next to it should be close to n − 4.\n\n"
        "Set the number of runs under Options → Monte Carlo runs.",
    ),
    "est_table": HelpTopic(
        "Results", "Estimate table",
        "The estimate next to the truth, with the error and the standard deviation from B6.",
        "Columns:\n"
        "•  Initial (A7) — the starting guess.\n"
        "•  Estimate (B5) — the maximum likelihood estimate.\n"
        "•  True (A1) — the value used in the simulation.\n"
        "•  Error — true − estimate.\n"
        "•  Std (B6) — standard deviation: the square root of the diagonal of the "
        "inverse FIM at the estimate.\n"
        "•  |Error| / Std — how many standard deviations each component is off. Values "
        "up to about 2 are normal.\n\n"
        "NEES (B8) combines all four components and their correlations into one number.",
    ),
    "geometry": HelpTopic(
        "Results", "Geometry plot",
        "The platform path, the true target track and, after a run, the estimated track.",
        "•  Blue — platform track. Maneuver legs are highlighted in cyan; the dot marks "
        "the start.\n"
        "•  Black dashed — true target track; the square marks the position at t = 0.\n"
        "•  Red — estimated target track.\n"
        "•  Grey lines — a few of the measured bearings.\n\n"
        "Before a run, “Preview geometry” draws the scenario from the current inputs, "
        "which helps when setting up maneuvers. Use the toolbar below the plot to zoom, "
        "pan and save the picture.",
    ),
    "tabs": HelpTopic(
        "Results", "Detailed result tabs",
        "The full numerical output of steps B1–B8.",
        "•  B1–B3: FIM at truth — the FIM J; its inverse P (the CRLB) with standard "
        "deviations and correlation coefficients (B2); and for every sampling time the "
        "2×2 covariance of the target position, A(t)·P·A(t)', with σξ, ση and their "
        "correlation (B3).\n"
        "•  B4: Measurements — for each k: platform position, true bearing, measured "
        "bearing and the noise.\n"
        "•  B6: FIM at estimate — the same as B1–B3, evaluated at the estimate. This "
        "is what you could report in practice, where the truth is unknown.\n"
        "•  Full report — everything, as text. Save it with File → Save report.",
    ),
}


# ====================================================================== DBEAR
DBEAR_TOPICS: dict[str, HelpTopic] = {
    "overview": HelpTopic(
        "Getting started", "How DBEAR works",
        "Draws the target position uncertainty ellipses computed by BEAR.",
        "DBEAR shows how uncertain the estimated target position is at chosen times.\n\n"
        "1.  BEAR saves its results to bear_results.json after every run, and DBEAR "
        "opens that file. Start DBEAR from BEAR (“Plot uncertainty ellipses”), or "
        "use Browse….\n\n"
        "2.  Enter the times and the ellipse size g, then press Plot or Enter.\n\n"
        "3.  At each time t, the ellipse is centred at the estimated position (red +). "
        "Its shape comes from the position covariance C(t) = A(t)·P·A(t)', where P is "
        "the inverse Fisher information matrix. The black × is the true position.\n\n"
        "4.  The table below the plot lists each ellipse's size, orientation, and "
        "whether it contains the true position.\n\n"
        "Hover over any  ?  for a short explanation; click it for the details.",
    ),
    "results_file": HelpTopic(
        "Controls", "Results file",
        "The file written by BEAR (bear_results.json). Use Reload after a new BEAR run.",
        "The JSON file written by BEAR after every run. By default it is "
        "bear_results.json in the program folder, where BEAR saves it.\n\n"
        "•  Browse… — open another results file, for example one saved with BEAR's "
        "File → Save results.\n"
        "•  Reload — read the file again after a new BEAR run. The times are reset to "
        "their defaults.",
    ),
    "times": HelpTopic(
        "Controls", "Times",
        "Times at which to draw ellipses, separated by spaces (e.g. 10 300 600).",
        "Use the same time unit as the sampling period T. Any t ≥ 0 is allowed: "
        "sampling times, times in between, and times after the last measurement. After "
        "the last measurement the ellipses are extrapolations, and the uncertainty "
        "grows.\n\n"
        "By default DBEAR proposes five times spread over the observation window. Press "
        "Enter or Plot to redraw.",
    ),
    "g": HelpTopic(
        "Controls", "Ellipse size g",
        "Size in standard deviations (sigmas). g = 2 contains the true position with 86.5 % probability.",
        "DBEAR draws the curve  (p − p̂)' C⁻¹ (p − p̂) = g².\n\n"
        "For a 2-D Gaussian error, the probability that the true position lies inside "
        "is 1 − exp(−g²/2):\n"
        "    g = 1     →  39.3 %\n"
        "    g = 2     →  86.5 %\n"
        "    g = 2.45  →  95 %\n"
        "    g = 3     →  98.9 %\n\n"
        "Note that a 1-sigma ellipse contains only 39 %, not 68 % as in one dimension.",
    ),
    "source": HelpTopic(
        "Controls", "Covariance source",
        "Which inverse FIM shapes the ellipses: at the estimate (B6) or at the truth (B3).",
        "•  At estimate (B6), the default — computed at the estimated parameter. This is "
        "what is available in practice.\n"
        "•  At truth (B3) — computed at the true parameter: the theoretical best "
        "accuracy (CRLB).\n\n"
        "Both are centred at the estimated position. They differ noticeably only when "
        "the estimate is far from the truth.",
    ),
    "rays": HelpTopic(
        "Controls", "Bearing rays",
        "Draw the measured line of sight from the platform at each ellipse time.",
        "Draws the measured bearing from the platform, taken at the measurement closest "
        "to each ellipse time.\n\n"
        "Notice that the ellipses are elongated along the line of sight. Bearings fix "
        "the direction to the target well, but the range poorly.",
    ),
    "zoom": HelpTopic(
        "Controls", "Zoom to ellipses",
        "On: show only the ellipses. Off: show the whole scenario including the platform.",
        "•  On — the view is limited to the ellipses and the true positions, so their "
        "shapes are easy to see.\n"
        "•  Off — the whole scenario, including the platform track, is shown.\n\n"
        "The toolbar below the plot also offers zoom and pan; its Home button resets "
        "the view.",
    ),
    "ellipse_table": HelpTopic(
        "Results", "Ellipse table",
        "Centre, axis lengths and orientation of every ellipse, and whether it contains the truth.",
        "One row per time:\n"
        "•  centre ξ, centre η — the estimated target position at that time.\n"
        "•  semi-major, semi-minor — half-lengths of the axes, g·√λ1 and g·√λ2, where "
        "λ1 ≥ λ2 are the eigenvalues of C(t).\n"
        "•  angle — orientation of the major axis, in degrees counter-clockwise from "
        "the ξ axis.\n"
        "•  truth inside — whether the true position lies inside the ellipse.\n\n"
        "Over many independent runs, a fraction 1 − exp(−g²/2) of the ellipses should "
        "contain the truth. Ellipses at different times of the same run are strongly "
        "correlated, though, so they usually all agree.",
    ),
    "algorithm": HelpTopic(
        "Results", "How an ellipse is computed",
        "Eigen-decomposition of the 2×2 position covariance.",
        "For the 2×2 covariance C = [[C11, C12], [C12, C22]]:\n\n"
        "1.  Eigenvalues  λ1,2 = (C11 + C22)/2 ± √(((C11 − C22)/2)² + C12²)\n"
        "2.  Orientation  θ = ½·atan2(2·C12, C11 − C22)\n"
        "3.  Semi-axes  a = g·√λ1,  b = g·√λ2\n"
        "4.  Points  p(φ) = p̂ + R(θ)·[a·cos φ, b·sin φ]',  for φ from 0 to 2π\n\n"
        "Every point satisfies (p − p̂)' C⁻¹ (p − p̂) = g². The full derivation is "
        "in docs/algorithms.md.",
    ),
}
