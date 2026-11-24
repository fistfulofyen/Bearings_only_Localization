# Algorithms

This note documents the mathematics behind BEAR and DBEAR: the model
(Bar-Shalom, Li & Kirubarajan, Sec. 3.7), each processing step B1–B8 of the
project (Sec. 3.8.3), and the g-sigma ellipse algorithm used by DBEAR.
The code module implementing each part is given in brackets.

---

## 1. Model  [`bearloc/model.py`]

**Target.** Constant velocity in the plane, parameter defined at `t = 0`:

```
x = [ξ0, η0, ξ̇, η̇]'           ξ(t) = ξ0 + ξ̇ t,   η(t) = η0 + η̇ t
```

Equivalently the target position is `p(t) = A(t) x` with

```
A(t) = | 1  0  t  0 |
       | 0  1  0  t |
```

**Platform.** Starts at `p_p(0)` with velocity `v0`. Maneuver leg *j* imposes
velocity `v_j` for `k1_j T ≤ t < k2_j T`; outside all legs the velocity is `v0`.
Legs may not overlap. Integrating the piecewise-constant velocity gives the
closed form

```
p_p(t) = p_p(0) + v0 t + Σ_j (v_j − v0) · clip(t − k1_j T, 0, (k2_j − k1_j) T)
```

where the clipped term is the time spent so far in leg *j*.

**Measurements.** At `t_k = kT`, `k = 1..n`:

```
z(k) = h(k, x) + w(k),   h(k, x) = atan2(Δη_k, Δξ_k),   w(k) ~ N(0, r) i.i.d.
Δξ_k = ξ(t_k) − ξ_p(t_k),   Δη_k = η(t_k) − η_p(t_k),   ρ_k² = Δξ_k² + Δη_k²
```

The bearing is measured counter-clockwise from the ξ axis, in (−π, π].
(For a navigation convention — clockwise from North — use
`β = π/2 − h`; nothing else changes.)

**Jacobian.** Since `∂atan2(Δη, Δξ)/∂Δξ = −Δη/ρ²` and `∂/∂Δη = Δξ/ρ²`, and
`∂Δξ/∂ξ0 = 1`, `∂Δξ/∂ξ̇ = t`, etc.:

```
h_x(k) = ∂h(k, x)/∂x = [ −Δη_k,  Δξ_k,  −Δη_k t_k,  Δξ_k t_k ] / ρ_k²
```

The code checks this against central finite differences
(`tests/test_model.py`).

**Angle wrapping.** Every bearing difference (residual) is wrapped into
(−π, π] by `wrap(a) = π − mod(π − a, 2π)`. Without this, a target whose
line of sight crosses the −ξ axis would produce residuals near ±2π and break
the estimator.

---

## 2. B1 — Fisher information matrix and observability  [`model.fisher_information`, `stats.check_observability`]

For i.i.d. Gaussian noise the log-likelihood is
`ln Λ(x) = −(1/2r) Σ_k [z(k) − h(k,x)]² + const`, and the FIM is

```
J(x) = E[ ∇ ln Λ ∇ ln Λ' ] = (1/r) Σ_k h_x(k)' h_x(k) = H'H / r
```

with `H` the n×4 matrix whose rows are `h_x(k)`.

**Observability = invertibility of J.** The parameter is (locally)
identifiable iff `J` is non-singular.

*Why a non-maneuvering platform is unobservable.* If the platform also moves
at constant velocity, the relative position is
`d(t) = (p_t0 − p_p0) + (v_t − v_p) t`. Scaling both terms by any α > 0
scales `d(t)` and leaves every bearing unchanged. Hence the direction

```
v_null = [ p_t0 − p_p0 ;  v_t − v_p ]
```

satisfies `H v_null = 0`, so `J v_null = 0`: rank `J` = 3. Range cannot be
determined, only the "shape" of the relative trajectory. A platform maneuver
breaks the linearity of `d(t)` and (generically) makes `J` full rank.
`tests/test_model.py` verifies this null vector numerically.

**Numerical test.** `J` mixes position entries (~1/(ρ² r)) and velocity
entries (~t²/(ρ² r)) that differ by many orders of magnitude, so `det J` or
`cond J` depend on the units and are meaningless thresholds. BEAR therefore
tests the *diagonally scaled* matrix

```
J̃ = D^{-1/2} J D^{-1/2},     D = diag(J)
```

which has unit diagonal and is invariant to rescaling any parameter. `J` is
declared invertible iff `σ_min(J̃) / σ_max(J̃) > 1e-10`. The reported
"scaled condition number" is `σ_max/σ_min` of `J̃`.

## 3. B2 — Inverse FIM (CRLB)  [`stats.invert_fim`]

If `J` is invertible, `P = J⁻¹` is the Cramér–Rao lower bound: any unbiased
estimator has covariance ≥ `P`. It is computed as
`P = D^{-1/2} J̃⁻¹ D^{-1/2}` (better conditioned) and symmetrized. BEAR also
prints the standard deviations `√P_ii` and the correlation coefficients
`P_ij / √(P_ii P_jj)`.

## 4. B3 — Position uncertainty at each sampling time  [`model.position_covariances`]

Since `p(t) = A(t) x`, the position covariance is

```
C(t) = A(t) P A(t)'
     = | P11 + 2t P13 + t² P33        P12 + t(P14 + P23) + t² P34 |
       | (sym.)                       P22 + 2t P24 + t² P44       |
```

BEAR prints `C(t_k)` for every `k` (elements, standard deviations,
correlation).

## 5. B4 — Measurement generation  [`Scenario.simulate_measurements`]

`z(k) = wrap( h(k, x_true) + √r · ν_k )`, `ν_k ~ N(0,1)` from a seeded
NumPy generator (seed in `BEAR.DEF`, so runs are reproducible).

## 6. B5 — Maximum likelihood estimate  [`bearloc/estimation.py`]

Maximizing `Λ(x)` is equivalent to the nonlinear least-squares problem

```
x̂ = argmin λ(x),     λ(x) = (1/r) Σ_k [ wrap(z(k) − h(k,x)) ]²
```

### Gauss–Newton / Levenberg–Marquardt (default)

At iterate `x^i` with residual vector `e` and Jacobian `H`:

```
A = H'H / r  (= J(x^i)),   g = H'e / r  (= −½ ∇λ)
(A + μ diag(A)) Δx = g
```

* If `λ(x^i + Δx) < λ(x^i)` the step is accepted and `μ ← μ/10`;
  otherwise `μ ← 10μ` and the step is recomputed.
* `μ → 0` gives the Gauss–Newton step (Newton's method with the expected
  Hessian = the FIM, i.e. Fisher scoring); large `μ` gives a short scaled
  gradient step, robust far from the solution.
* Damping by `diag(A)` (Marquardt) makes the iteration unit-invariant.
* **Convergence**: stop when the Gauss–Newton decrement
  `δ = g' A⁺ g` — the cost reduction predicted by the linearized model — is
  below `tol` (default 1e-10). Because `λ` is a dimensionless χ²-type
  quantity, `δ` is too, so `tol` does not depend on the units.
* The cost never increases (checked in `tests/test_estimation.py`).

### BFGS (quasi-Newton, optional)

`scipy.optimize.minimize(method="BFGS")` with the analytic gradient
`∇λ = −(2/r) H'e`. Positions and velocities are rescaled first,
`u = √diag(J(x0)) ⊙ (x − x0)`, so all components are O(1); without this
BFGS performs poorly on this badly scaled problem.

Both methods reach the same minimum (tested); Gauss–Newton typically needs
5–15 iterations versus ~30 for BFGS.

SciPy's BFGS often stops with the status "precision loss" exactly at the
minimum. In about half of the runs of the default scenario, BEAR used to
report "not converged" for a correct estimate. BEAR therefore verifies
such a stop independently: the result is accepted when the Gauss–Newton
decrement `δ = g' A⁺ g` is at most `10⁻⁶`.

**Initial estimate (A7).** `λ(x)` is not convex. A poor initial estimate
can lead to a local minimum; BEAR reports the
convergence status, and B7 helps detect a wrong minimum (large fit error).

### Range divergence: when no finite MLE exists

Scale the target-minus-platform trajectory by α, so that
`x(α) = [p_p0 + α(p_t0 − p_p0); v_p + α(v_t − v_p)]` with `v_p` the initial
platform velocity. The relative position at time t is then

```
d_α(t) = α·[(p_t0 − p_p0) + (v_t − v_p) t] − m(t)
```

where `m(t)` is the extra platform displacement caused by the maneuvers.
As α → ∞, the term `m(t)` becomes negligible. The predicted bearings tend to
those of a straight-line relative motion *without* the maneuver's parallax,
and `λ(x(α))` tends to a finite limit.

When the maneuver is weak compared with the noise (small `m(t)`, large σ), a
noisy data set can fit this limit better than any finite range. In that case
`λ` keeps decreasing toward infinite range and **the likelihood has no finite
maximum**: the MLE does not exist.

Every descent method then runs off along the ray. Far out the gradient
vanishes, so a naive stopping test would report convergence. This is a
property of the problem, not of the optimizer. In the *High noise, weak
maneuver* sample it happens in about one third of the noise draws, and the
diverged estimate has a lower cost than the true parameter in every such run.

**How often does it happen?** Parametrize the target by its inverse initial
range `κ = 1/ρ₀` and three scale-invariant coordinates (initial bearing, and
relative velocity divided by range). These are the modified polar
coordinates of Aidala and Hammel. In these coordinates:

- Scaling the relative trajectory changes only `κ`.
- The bearings are smooth in `κ` through `κ = 0`, the point at infinity.
- Divergence is the event that the unconstrained estimate `κ̂ ≤ 0`.

With `κ̂` approximately Gaussian, and `σ_κ/κ = σ_ρ₀/ρ₀` (delta method), we
get

```
P(divergence) ≈ Φ(−ρ₀ / σ_ρ₀)
```

Here `σ_ρ₀` is the CRLB standard deviation of the initial range. The
probability tends to 50 % as observability vanishes. Monte Carlo runs of two
different geometries agree with this law within 3.5 percentage points (see
the project report). For the default scenario `ρ₀/σ_ρ₀ ≈ 25`, so divergence
is practically impossible.

BEAR therefore stops and reports **diverged** (not converged) once the
estimated range exceeds `10³ ×` the problem's length scale. That scale is the
larger of the initial-estimate range and the size of the platform track.
Bearings generated without the maneuver's parallax are the extreme case:
there `λ → 0` as α → ∞. They are used in the tests to check this detection.

## 7. B6 — B1–B3 at the estimate

Identical computations with `x̂` in place of `x_true`. This is what would be
available in practice, where the true parameter is unknown; DBEAR uses this
covariance.

## 8. B7 — Fitting error  [`stats.chi2_test`]

```
λ(x̂) = Σ_k [z(k) − h(k, x̂)]² / r   ~   χ²(n − 4)
```

approximately, when the model is correct: n measurements minus 4 estimated
parameters (a linear LS result that holds asymptotically for the nonlinear
case).

More generally the residuals keep `n − rank J(x̂)` degrees of freedom. When
the FIM is singular (no maneuver) only 3 parameter combinations are fitted,
and BEAR uses `n − 3`. Monte Carlo check: the mean fitting error is 56.7 for
n − 3 = 57. BEAR prints `λ(x̂)`, its degrees of freedom, and the two-sided
interval `[χ²_{n−4}(α/2), χ²_{n−4}(1−α/2)]` (α = 1 − confidence). A value
above the interval suggests a model mismatch (wrong r, maneuvering target)
or a local minimum.

## 9. B8 — Normalized estimation error squared  [`stats.nees`]

```
ε = x̃' J x̃,   x̃ = x_true − x̂
```

If `x̂` is efficient (unbiased with Gaussian errors of covariance `J⁻¹`),
`ε ~ χ²(4)`. J is evaluated at `x̂` (B6, primary) and at `x_true` (B1, shown
for comparison). The 95% two-sided interval is `[0.484, 11.14]`.

**NEES is only tested when `J(x̂)` is invertible.** With a singular FIM the
estimate lies somewhere along the unobservable direction, so the error along
it is arbitrary. The null direction of `J(x̂)` is also not that of
`J(x_true)`, so `ε` follows no χ² law: in a Monte Carlo test of the
no-maneuver sample its mean is 5.7, not 3. BEAR shows the value but reports
"not meaningful". The Monte Carlo average uses only the converged runs with
an invertible FIM.

A single run proves little (1 in 20 consistent runs falls outside). The
`--mc N` option runs N independent noise realizations and tests the average:

```
ε̄ = (1/N) Σ ε_i,    N ε̄ ~ χ²(4N)   ⇒   ε̄ ∈ [χ²_{4N}(α/2)/N, χ²_{4N}(1−α/2)/N]
```

### Verification against theory

A Monte Carlo study of the default scenario (3000 independent runs, 1° noise)
confirms every statement above. The key checks are repeated in
`tests/test_theory.py`.

| Statement | Theory | Monte Carlo |
|---|---|---|
| FIM = covariance of the score `H'w/r` | exact | max scaled difference 0.014 (sampling error) |
| MLE unbiased | bias / std = 0 | |bias| / std ≤ 0.03 |
| MLE efficient (variance = CRLB) | ratio 1 | 0.99 – 1.04 |
| fitting error ~ χ²(56) | mean 56, variance 112 | 56.1, 112.3; 5.2 % outside the 95 % interval |
| NEES ~ χ²(4) | mean 4, variance 8 | 4.09, 8.6; 4.7 % outside; Kolmogorov–Smirnov p = 0.25 |
| 2-σ position ellipse coverage | 86.5 % | 86.3 % |
| \|error\| / std > 2, per component | 4.6 % | 4.4 – 4.9 % |
| ellipse major axis vs line of sight | aligned | within 1 – 5° |
| CRLB std ∝ σ | exact | exact |
| denser sampling of the same window, n × 2 / × 4 | std × 0.707 / 0.500 | 0.70 – 0.72 / 0.49 – 0.51 |

**Where the theory stops applying.** In the *High noise, weak maneuver*
sample (3°, scaled condition number 3·10⁵), the problem is strongly
nonlinear:
- about a third of the runs have no finite MLE (range divergence, see B5);
- among the runs that do converge, the median NEES is about 46 instead of
  3.4.

There the inverse FIM badly understates the real error, and the NEES test
correctly reports that the estimator is not consistent.

---

## 10. DBEAR — the g-sigma ellipse of a 2×2 covariance  [`bearloc/ellipse.py`]

**Problem.** Given a symmetric positive definite

```
P = | P11  P12 |
    | P12  P22 |
```

draw the curve `x' P⁻¹ x = g²` (translated to a centre `c`).

### Algorithm

1. **Eigenvalues** (closed form for a symmetric 2×2 matrix):

   ```
   m   = (P11 + P22) / 2
   d   = √( ((P11 − P22)/2)² + P12² )
   λ1  = m + d                       (largest)
   λ2  = det(P) / λ1 = (P11 P22 − P12²) / λ1      (= m − d)
   ```

   `λ2` is computed from `λ1 λ2 = det P` rather than `m − d` to avoid
   catastrophic cancellation for very thin ellipses (highly correlated
   covariances, common in bearings-only problems).

2. **Orientation** of the major axis (angle from the first axis):

   ```
   θ = ½ · atan2(2 P12, P11 − P22)          θ ∈ (−π/2, π/2]
   ```

   *Proof.* With `R(θ) = [[cos θ, −sin θ], [sin θ, cos θ]]` and
   `P = R diag(λ1, λ2) R'`, expanding gives
   `P11 − P22 = (λ1 − λ2) cos 2θ` and `2 P12 = (λ1 − λ2) sin 2θ`;
   since `λ1 − λ2 ≥ 0`, `atan2` returns `2θ` in the correct quadrant.
   (If `λ1 = λ2` the ellipse is a circle and any θ works.)

3. **Semi-axes:** `a = g √λ1` (major), `b = g √λ2` (minor).

4. **Points:** for `φ ∈ [0, 2π]` (N equally spaced values, closed curve):

   ```
   x(φ) = c + R(θ) [ a cos φ ;  b sin φ ]
   ```

### Correctness

Let `y = x(φ) − c = R u` with `u = [a cos φ, b sin φ]'`. Since
`P⁻¹ = R diag(1/λ1, 1/λ2) R'` and `R'R = I`:

```
y' P⁻¹ y = u' diag(1/λ1, 1/λ2) u = g² λ1 cos²φ / λ1 + g² λ2 sin²φ / λ2 = g²
```

So every generated point lies on the required curve (verified numerically
for random matrices in `tests/test_ellipse.py`).

An equivalent alternative is the Cholesky factor `P = L L'`:
`x(φ) = c + g L [cos φ, sin φ]'` (since `y'P⁻¹y = g² u'L'L'⁻¹L⁻¹Lu = g²`);
the eigen-form is used because it also yields the axis lengths and the
orientation, which DBEAR prints.

### Probability content

If `x ~ N(c, P)`, then `q = (x − c)' P⁻¹ (x − c) ~ χ²(2)` whose CDF is
`1 − e^{−q/2}`. Hence

```
P{ x inside the g-sigma ellipse } = 1 − exp(−g²/2)
```

| g | 1 | 2 | 3 |
|---|---|---|---|
| probability | 39.3 % | 86.5 % | 98.9 % |

and the `g` that contains probability `p` is `g = √(−2 ln(1 − p))`
(e.g. 95 % → g = 2.448).

### Use in DBEAR

At each user-specified time `t` (not necessarily a sampling time):
centre `c = A(t) x̂`, covariance `C(t) = A(t) P̂ A(t)'` with `P̂ = J(x̂)⁻¹`
from B6 (or `J(x_true)⁻¹` from B3 with `--source truth`). DBEAR prints the
centre, semi-axes, orientation and whether the true position falls inside,
and plots the ellipses together with the platform, true and estimated target
trajectories (and optionally the measured lines of sight).
