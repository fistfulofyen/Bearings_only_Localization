# User Guide — BEAR and DBEAR

This guide explains how to use the two programs of the bearings-only target
localization project. The mathematics behind them is in
[algorithms.md](algorithms.md).

1. [What the programs do](#1-what-the-programs-do)
2. [Starting the programs](#2-starting-the-programs)
3. [BEAR — estimation](#3-bear--estimation)
4. [DBEAR — uncertainty ellipses](#4-dbear--uncertainty-ellipses)
5. [Files](#5-files)
6. [Experiments to try](#6-experiments-to-try)
7. [Troubleshooting](#7-troubleshooting)

---

## 1. What the programs do

A **platform** (a ship, aircraft or sensor) moves through the plane and
repeatedly measures the **bearing**, the direction, to a **target**. It never
measures the range. The target moves in a straight line at constant speed.
From the noisy bearings we want the target's position and velocity:

```
x = [ξ0, η0, dξ/dt, dη/dt]      (position at t = 0, velocity)
```

| Program | Job |
|---|---|
| **BEAR** | Takes the scenario (inputs A1–A7), simulates noisy bearings, computes the maximum likelihood estimate, and evaluates it (steps B1–B8): observability, best achievable accuracy, fitting error and NEES. |
| **DBEAR** | Draws the **position uncertainty ellipses** of the estimated target at times you choose. |

BEAR saves its results to `bear_results.json`, and DBEAR reads that file.

Both programs exist in two forms with the same calculations:

* **Windows** (graphical): `bear_gui.py`, `dbear_gui.py`. Every input and result
  has a **?** icon: hover over it for a short explanation, or click it for the
  full text. Press **F1** for all help topics.
* **Command line** (text prompts): `bear.py`, `dbear.py`. Useful for scripts and
  batch runs.

## 2. Starting the programs

**Requirements:** Python 3.10 or newer, with NumPy, SciPy and Matplotlib.
Install them once from the project folder:

```bash
pip install -r requirements.txt
```

**Graphical versions:**

```bash
python bear_gui.py
```

```bash
python dbear_gui.py
```

**Command-line versions:**

```bash
python bear.py
```

```bash
python dbear.py
```

The programs find their files in the project folder, whatever folder you start
them from:

| File | Written by | Read by | Content |
|---|---|---|---|
| `BEAR.DEF` | you (or *Save inputs*) | BEAR at start-up | default inputs A1–A7 |
| `bear_results.json` | BEAR, after every run | DBEAR | all results of the last run |

---

## 3. BEAR — estimation

### 3.1 The window at a glance

* **Left — inputs.** Grouped as in the assignment (A1–A7), plus *Options*.
  Everything is pre-filled from `BEAR.DEF`. *Load sample* fills in one of five
  ready-made scenarios.
* **Bottom left — actions.**
  * **Run B1–B8 (F5)** computes everything.
  * **Preview geometry** draws the scenario without estimating.
  * **Reset to defaults** restores the values from `BEAR.DEF`.
  * **Plot uncertainty ellipses (DBEAR)** opens DBEAR on the results.
* **Right — results.**
  * The **Summary** tab shows the pass/fail status lines, the estimate next to
    the truth, and the geometry plot.
  * The other tabs hold the detailed numbers of B1–B3, B4 and B6, and the full
    text report.
* **Status bar** (bottom) — what happened last, and where the results were saved.

### 3.2 A first run

1. Start BEAR. The defaults describe a target about 13 km away. The platform
   heads north at 15 m/s and turns east between samples 20 and 40.
2. Press **F5**. Within a fraction of a second the Summary shows:

   | Line | Result | Meaning |
   |---|---|---|
   | Observability (B1) | OBSERVABLE — rank 4/4 | the bearings determine the target |
   | Estimation (B5) | converged in 5 iterations | the estimator worked |
   | Fitting error (B7) | 44.20 (dof 56), within [37.21, 78.57] | the bearings fit the model |
   | NEES (B8) | 0.690, within [0.484, 11.143] | the error matches the claimed accuracy |

   The estimate is ξ0 = 4870, η0 = 11802, dξ/dt = −5.65, dη/dt = −1.99. The true
   value is 5000, 12000, −6, −2, and the standard deviations are 244, 451, 0.58
   and 0.56. Every error is below one standard deviation.
3. Click **Plot uncertainty ellipses (DBEAR)** to see how uncertain the position
   is over time (section 4).
4. Now click **Remove all** in the A6 box and press F5 again. B1 turns red, *NOT
   OBSERVABLE*: without a maneuver the range cannot be determined.

### 3.3 Inputs

Units are free but must be consistent. The defaults use metres, seconds and m/s.

| Item | Field | Meaning | Default |
|---|---|---|---|
| A1 | ξ0, η0, dξ/dt, dη/dt | True target position at t = 0 and velocity. Used only to simulate the bearings and to judge the result; the estimator never sees it. | 5000, 12000, −6, −2 |
| A2 | T | Time between measurements. Measurements are taken at t = kT, k = 1..n. | 10 |
| A3 | n | Number of measurements. At least 5 are needed for the B7 test. | 60 |
| A4 | Noise | Bearing noise as σ in degrees, or as r in rad². The drop-down switches the unit and converts the value. | 1° (r = 3.046e-4) |
| A5 | Position, Velocity | Platform at t = 0, and its velocity outside the maneuver legs. | (0, 0), (0, 15) |
| A6 | Maneuver legs | Platform velocity (vξ, vη) in effect for k1·T ≤ t < k2·T. Legs may not overlap. At least one leg is needed for observability. | (15, 0) from k = 20 to 40 |
| A7 | ξ0, η0, dξ/dt, dη/dt | Starting guess for the estimator. | 2000, 8000, 0, 0 |
| — | Method | `gauss-newton` (recommended) or `bfgs`. | gauss-newton |
| — | Seed | Makes the noise repeatable. Blank = new noise on every run. | 1 |
| — | Confidence | Probability of the χ² acceptance intervals. | 0.95 |
| — | Monte Carlo runs | Extra consistency test over N runs (0 = off). | 0 |

**Editing maneuver legs (A6).**
* Type vξ, vη, k1 and k2 in the row under the table, then click **Add leg**.
* To change a leg, click it in the table, edit the row, then click **Update
  selected**.
* **Remove selected** and **Remove all** delete legs.
* The *time window* column shows each leg in time units.

**Invalid input.** If an entry cannot be read, BEAR names the field, puts the
cursor in it, and computes nothing.

### 3.4 Reading the results

**Summary tab.** Green means the check passed, red means it failed, and orange
is a warning.

* **Observability (B1).** BEAR computes the Fisher information matrix (FIM) J at
  the true parameter.
  * If J is invertible, the target is observable, and J⁻¹ is the Cramér–Rao
    lower bound (CRLB): the best covariance any unbiased estimator can reach.
  * If J is singular (rank 3), the range is undetermined. Usually there is no
    platform maneuver.
* **Estimation (B5).** The maximum likelihood estimate minimizes the sum of
  squared bearing residuals divided by r, starting from A7.
  * *Not converged*: try another A7 or the other method.
  * *Minimum not unique*: the FIM at the estimate is singular.
  * *FAILED — range ran off to infinity*: for these bearings the likelihood
    has no finite maximum. The maneuver is too weak compared with the noise
    to determine the range, so no initial estimate helps; only a better
    geometry does.
* **Fitting error (B7).** λ = Σ (measured − predicted bearing)² / r. It should
  follow χ² with n − 4 degrees of freedom (n − 3 when unobservable, because
  only 3 combinations are then fitted).
  * Too large: r is too small, the target maneuvers, or the estimator found a
    wrong minimum.
  * Too small: r is too large.
* **NEES (B8).** ε = x̃'Jx̃, the estimation error in units of the claimed
  uncertainty. It should follow χ²(4), with mean 4.
  * Too large: the estimator is overconfident, which is typical of poor geometry
    or high noise.
  * When unobservable, NEES is shown but marked *not meaningful*. The
    estimate can then sit anywhere along the unobservable direction, so its
    error there is arbitrary.
* **Monte Carlo.** The average NEES over the valid runs (converged, with an
  invertible FIM) should be close to 4 and inside the χ²(4N)/N interval. Runs
  whose range diverged are counted separately. The average fitting error
  should be close to n − 4.

About 5 % of runs fall outside a 95 % interval even when everything is right, so
judge consistency with the Monte Carlo test, not with a single run.

**Estimate vs. truth table.**
* Each parameter appears with its initial value, estimate, true value, error and
  standard deviation (from B6).
* The last column, |error|/std, says how many standard deviations each
  component is off. Values up to about 2 are normal.

**Geometry plot.**
* Blue: platform track, with maneuvers in cyan.
* Black dashed: true target track.
* Red: estimated target track.
* Grey: some measured bearings.

**Detailed tabs.**
* *B1–B3: FIM at truth* — J; J⁻¹ with standard deviations and correlations
  (B2); and the 2×2 position covariance A(t)·J⁻¹·A(t)' at every sampling time
  (B3).
* *B4: Measurements* — platform position, true bearing, measured bearing and
  noise for each sample.
* *B6: FIM at estimate* — the same as B1–B3, evaluated at the estimate. This is
  what is available in practice.
* *Full report* — all of the above as text.

### 3.5 Menus and shortcuts

| Menu | Command | Shortcut |
|---|---|---|
| File | Open defaults file… (load a `.DEF`) | Ctrl+O |
| File | Save inputs as… (write a `.DEF`) | Ctrl+S |
| File | Save results (JSON) as… / Save report (text) as… | |
| Run | Run B1–B8 / Preview geometry / Plot ellipses (DBEAR) | F5 |
| Help | Help topics / About | F1 |

### 3.6 Command-line version (`bear.py`)

`bear.py` asks for each input in turn, with the default in brackets. Press
**Enter** to keep it. Vectors are numbers separated by commas or spaces. The
noise accepts a variance in rad² or a standard deviation with the suffix `deg`,
for example `0.5deg`. Invalid answers are asked again.

```
A1  True target parameter xi0, eta0, xi_dot, eta_dot [5000, 12000, -6, -2]:
A2  Sampling period T [10]:
A3  Number of samples n [60]: 80
A4  Noise variance r in rad^2, or e.g. '1deg' (now sigma = 1 deg) [0.000304617]: 0.5deg
...
```

| Option | Meaning |
|---|---|
| `-b`, `--batch` | no questions; use the defaults file as it is |
| `--def PATH` | another defaults file |
| `-o PATH` | results file (default `bear_results.json` in the project folder) |
| `--save-def PATH` | save the inputs of this run as a defaults file |
| `--every M` | print only every M-th row of the per-sample tables |
| `--mc N` | add an N-run Monte Carlo NEES test |

---

## 4. DBEAR — uncertainty ellipses

### 4.1 The window

* **Top.**
  * The results file, with *Browse…* and *Reload*.
  * The **times**, and the ellipse size **g** with its probability.
  * The **covariance source**: at the estimate (B6) or at the truth (B3).
  * Checkboxes for **bearing rays** and **zoom to ellipses**.
* **Middle.** The plot. The toolbar below it zooms, pans and saves.
* **Bottom.** The ellipse table.
* **Menus.**
  * File → Open results… (Ctrl+O), Reload (Ctrl+R), Save figure… (Ctrl+S).
  * Help → Help topics (F1).

### 4.2 Drawing ellipses

1. Open DBEAR from BEAR's **Plot uncertainty ellipses** button, or start it
   yourself; it opens `bear_results.json`. After a new BEAR run, press
   **Reload**.
2. Enter the **times**, separated by spaces, for example `10 300 600`. Any t ≥ 0
   works: between samples, or after the last measurement (extrapolation).
3. Choose **g** and press **Plot** or Enter. The plot redraws immediately when
   you click the g arrows, the source, rays or zoom. After typing a value, press
   Enter.
4. Use **Save figure…** to export PNG, PDF or SVG.

### 4.3 Reading the ellipses

At each time t the ellipse is centred at the **estimated** position (red +).
The black × is the true position. The ellipse is the curve

```
(p − p̂)' C(t)⁻¹ (p − p̂) = g²,     C(t) = A(t) P A(t)'
```

where P is the inverse FIM. Its size g sets the probability that the true
position lies inside:

| g | 1 | 2 | 2.45 | 3 |
|---|---|---|---|---|
| probability | 39.3 % | 86.5 % | 95 % | 98.9 % |

Things to notice:

* The ellipses are long and thin, and **elongated along the line of sight**
  (turn on *Bearing rays*). Bearings fix the direction well and the range
  poorly.
* They are **smallest around and after the maneuver**, when the geometry
  changes most. They **grow after the last measurement**.
* The table gives each ellipse's centre, semi-axes (g·√λ of C), orientation,
  and whether the truth is inside.

With the default scenario and g = 2, the semi-major axes at t = 10, 160, 300,
450 and 600 are about 1005, 851, 742, 692 and 721 m, and all five ellipses
contain the true position.

### 4.4 Command-line version (`dbear.py`)

Without options, `dbear.py` asks for the times and g, prints the ellipse table,
and opens the plot.

| Option | Meaning |
|---|---|
| `results` | results file (default `bear_results.json` in the project folder) |
| `--times T ...` | ellipse times |
| `-g G` | ellipse size in sigmas (default 2) |
| `--source estimate\|truth` | covariance from B6 (default) or B3 |
| `--rays`, `--zoom` | bearing rays, zoom to the ellipses |
| `--save FILE`, `--no-show`, `-b` | save the figure / no window / no questions |

Example:

```bash
python dbear.py --batch --times 10 300 600 -g 2 --zoom --save ellipses.png
```

---

## 5. Files

### 5.1 `BEAR.DEF`

`BEAR.DEF` is a plain text (INI) file that you can edit in any text editor. `#`
starts a comment. Every section whose name starts with `maneuver` is one leg.
Delete all of them for a platform that never maneuvers.

```ini
[target]
x_true = 5000, 12000, -6, -2      # A1
[sampling]
T = 10                            # A2
n = 60                            # A3
[noise]
r = 0.0003046174198               # A4, rad^2 (1 deg)
seed = 1                          # blank = random
[platform]
position = 0, 0                   # A5
velocity = 0, 15
[maneuver 1]
velocity = 15, 0                  # A6
k1 = 20
k2 = 40
[estimation]
x_init = 2000, 8000, 0, 0         # A7
method = gauss-newton
max_iter = 200
tol = 1e-10
[statistics]
confidence = 0.95
```

To keep several scenarios, save them with *File → Save inputs as…* (window) or
`--save-def` (command line). Load them with *File → Open defaults file…* or
`--def`.

### 5.2 `bear_results.json`

A JSON file with everything DBEAR needs, and more:

* `scenario` — the inputs.
* `times`, `platform_track`, `measurements` — the samples, platform positions
  and measured bearings.
* `truth` and `estimate` — FIM, invertibility, inverse FIM and position
  covariances, evaluated at the truth and at the estimate.
* `mle` — the estimate, cost, iterations, convergence and iteration history.
* `fit`, `nees` — the test values and their intervals.

It can be read from Python with `json.load` for your own analysis.

---

## 6. Experiments to try

The five samples (*Load sample* in the window) each show one effect:

| Sample | What to look at |
|---|---|
| Close target, low noise | Small ellipses; all tests pass. |
| Zig-zag, two maneuver legs | Two turns give good accuracy far away. |
| Target due west (bearing crosses 180°) | In the B4 tab the bearings jump from +180° to −180°, and the estimate is still correct. |
| No maneuver (unobservable) | B1 is red, and DBEAR cannot draw ellipses. |
| High noise, weak maneuver | The estimate is often far off and NEES fails: the CRLB is not reliable in this strongly nonlinear case. Run 200 Monte Carlo runs: about a third diverge (no finite maximum likelihood). |

Further ideas:
* Move or lengthen the maneuver leg and watch the ellipses.
* Set the seed to blank and run several times to see the run-to-run spread.
* Set Monte Carlo runs to 500 and check that the average NEES is about 4.
* Give a poor initial estimate (A7) and watch the iteration count in B5.

---

## 7. Troubleshooting

| Symptom | Cause | What to do |
|---|---|---|
| B1: *NOT OBSERVABLE* | No platform maneuver, or a very weak one | Add a maneuver leg (A6), or make it larger or longer. |
| B5: *NOT converged* | Poor initial estimate, or a nearly unobservable geometry | Change A7 (for example a range closer to the truth), try the other method, or improve the geometry. |
| B5: *range ran off to infinity* | No finite maximum likelihood: the maneuver is too weak for this noise level | Stronger or longer maneuver, more measurements, or less noise. A7 does not help. |
| B8: *not meaningful* | The FIM at the estimate is singular (unobservable) | Add a maneuver. |
| B7 above its interval | r too small, a wrong local minimum, or a model mismatch | Check A4; try another A7. |
| B8 above its interval in one run | Chance (about 5 % of runs) | Run the Monte Carlo test. |
| Monte Carlo NEES above its interval | Strongly nonlinear case (far target, high noise, weak maneuver): the inverse FIM understates the true error | An expected limitation. Use more measurements or a stronger maneuver. |
| DBEAR: *FIM is singular, no ellipses* | The BEAR run was unobservable | Add a maneuver and run BEAR again. |
| DBEAR: *No results yet* | BEAR has not been run, or the file is elsewhere | Run BEAR, or use *Browse…*. |
| DBEAR shows old results | It still has the previous file loaded | Press *Reload*. |
| "Overlapping maneuver legs" | Two legs share sample times | Let each leg start at or after the previous k2. |

---

*Optional:* the graphical programs can also be packaged as stand-alone Windows
programs (`BEAR.exe`, `DBEAR.exe`) with `python build_exe.py`; see the README.
