# Steady-state force laws

Level 1 of the [stack](index): given a slip, what force does the tire make? Four
classical answers, in increasing order of fidelity and parameter count. All four are
implemented, differentiable and free of learnable parameters, so they can be used as
baselines, as priors inside a grey-box model, or as the analytical core of a parameter
network.

## 1. Linear

```{math}
:label: eq-linear
F_x = C_\kappa\,\kappa, \qquad F_y = -C_\alpha\,\alpha
```

Two parameters: the longitudinal and cornering stiffnesses. This is the tangent of every
other model at zero slip, and it is the model implicitly assumed by any controller
linearised about straight-line running.

**Good:** exact where it is valid, trivially invertible, and the only model whose
parameters are directly measurable from a low-slip sweep.
**Bad:** unbounded. Force grows without limit, so a planner using it will happily
request 3 g of lateral acceleration from a tire that can deliver 1.1 g. It is unusable
anywhere near the limit — which is where a racing controller lives.

## 2. Brush model

The tread is a row of elastic bristles that stick until the local shear stress exceeds
$\mu p(x)$, then slide. With a parabolic pressure distribution this integrates in closed
form. With $\theta = C/(3\mu F_z)$ and $z = \theta\sigma$ for a theoretical slip
$\sigma$:

```{math}
:label: eq-brush
F(\sigma) = \begin{cases}
\mu F_z\,\big(3z - 3z^2 + z^3\big), & z < 1 \quad\text{(partial sliding)}\\[4pt]
\mu F_z, & z \ge 1 \quad\text{(full sliding)}
\end{cases}
```

**Good:** *derived*, not fitted. Odd symmetry, zero force at zero slip and the bound
$|F| \le \mu F_z$ all fall out of the derivation rather than being imposed — which is
precisely the argument for encoding them in a network. Two parameters ($C$, $\mu$).
**Bad:** no post-peak decay. Real tires lose force after the peak; the brush model
saturates and stays flat. The parabolic pressure assumption and the rigid-carcass
assumption both break down at high load and high camber.

## 3. Dugoff

A closed-form combined-slip model with a single friction bound. With
$s = \kappa/(1+\kappa)$ and $t = \tan\alpha/(1+\kappa)$, the unsaturated forces are
$F_{x0} = C_\kappa s$ and $F_{y0} = -C_\alpha t$, and the boundary value

```{math}
:label: eq-dugoff-lambda
\lambda = \frac{\mu F_z (1+\kappa)}{2\sqrt{(C_\kappa \kappa)^2 + (C_\alpha \tan\alpha)^2}}
```

decides whether the patch is fully adhering. The forces are scaled by

```{math}
:label: eq-dugoff-f
f(\lambda) = \begin{cases} 1, & \lambda \ge 1\\ \lambda(2-\lambda), & \lambda < 1 \end{cases}
```

**Good:** combined slip for free, cheap to evaluate, physically bounded, only three
parameters. Popular in real-time vehicle control for exactly these reasons.
**Bad:** $f$ is only $C^1$ at $\lambda = 1$ — there is a curvature discontinuity on the
adhesion/sliding boundary that a Newton-type solver can find. Like the brush model it
has no post-peak decay.

:::{warning}
The $(1+\kappa)$ factor in {eq}`eq-dugoff-lambda` must **not** be applied to the already
normalised forces. Dividing twice makes $\lambda$ too large, the model under-saturates,
and it then exceeds $\mu F_z$ under combined slip — defeating the point of the model.
This was a real bug in this codebase, caught by
`tests/test_physics.py::test_dugoff_stays_within_the_friction_circle`.
:::

## 4. Magic Formula

The empirical standard {cite}`pacejka2012tire`:

```{math}
:label: eq-mf
F(x) = D \sin\Big( C \arctan\big( Bx' - E\,(Bx' - \arctan Bx')\big)\Big) + S_v,
\qquad x' = x + S_h
```

```{math}
:label: eq-mf-roles
\underbrace{B}_{\text{stiffness}},\quad
\underbrace{C}_{\text{shape}},\quad
\underbrace{D = \mu(F_z)\,F_z}_{\text{peak}},\quad
\underbrace{E}_{\text{curvature}},
\qquad
C_\alpha = \left.\frac{\partial F_y}{\partial\alpha}\right|_0 = B\,C\,D
```

```{figure} ../_static/figures/magic_formula_parameters.png
:alt: effect of the Magic Formula coefficients
:width: 100%

What each coefficient does. $B$ sets the initial slope, $C$ how far the curve falls back
after the peak, $E$ the curvature near the peak, $\mu$ the peak height. Note that $D$ is
stored here as $\mu$, not as a free force — see load sensitivity below.
```

**Good:** it can fit essentially any measured curve shape, including the post-peak decay
the physical models miss, and it is the format every tire supplier and vehicle-dynamics
tool speaks.
**Bad:** it is a *curve fit*, not a derivation. The coefficients have no independent
physical meaning beyond their roles above, they do not extrapolate outside the measured
envelope, and full MF 6.x needs tens of coefficients and a proper measurement campaign.
$B$ and $C$ are only jointly identifiable near the origin (the product $BCD$ is the
stiffness), so a naive fit on sparse data is ill-conditioned — see
[benchmarks](../comparison/benchmarks).

## Side by side

```{figure} ../_static/figures/steady_state_laws.png
:alt: linear, brush, Dugoff and Magic Formula compared
:width: 100%

All four laws matched to the same cornering stiffness $C_\alpha = 45$ kN/rad and the
same $\mu = 1.0$ at $F_z = 1$ kN. Left: the linear model leaves the physical envelope
almost immediately; brush and Dugoff saturate and stay flat; only the Magic Formula
reproduces the post-peak decay. Right: near zero slip all four agree by construction —
which is exactly why low-slip data cannot distinguish them, and why a model fitted only
on gentle driving tells you nothing about the limit.
```

| | parameters | bounded by $\mu F_z$ | post-peak decay | smoothness | derived or fitted |
|---|---|---|---|---|---|
| linear | 2 | ✗ | ✗ | $C^\infty$ | derived (small-slip limit) |
| brush | 2 | ✓ | ✗ | $C^1$ at $z=1$ | derived |
| Dugoff | 3 | ✓ | ✗ | $C^1$ at $\lambda=1$ | derived |
| Magic Formula | 4+ | ✓ (by $D$) | ✓ | $C^\infty$ | fitted |

## Load sensitivity

Peak friction falls as the tire is loaded, so $D$ is not a free parameter:

```{math}
:label: eq-load-sensitivity
\mu(F_z) = \mu_0 \left[1 - k_\mu\left(\frac{F_z}{F_{z0}} - 1\right)\right],
\qquad D = \mu(F_z)\,F_z
```

```{figure} ../_static/figures/load_sensitivity.png
:alt: load sensitivity of friction and peak force
:width: 100%

Left: $\mu(F_z)$. Right: peak force is sub-linear in load. This is why lateral load
transfer *reduces* an axle's total grip, and therefore why a four-wheel model with load
transfer predicts understeer that a bicycle model with the same tire does not.
```

Encoding {eq}`eq-load-sensitivity` leaves only $k_\mu$ to identify, instead of asking a
network to discover the load dependence from the three or four load levels a rig
typically provides.

## Fitting the analytical baseline

A learned model must beat a *properly fitted* Magic Formula, not a guessed one.
{py:func}`tire_nn.physics.fitting.fit_magic_formula` does bounded least squares with
`scipy.optimize.least_squares` on **load-normalised** residuals $F/F_z$ — without that
normalisation, high-load samples dominate and $\mu$ is biased toward the heavy end.

```bash
python scripts/fit_magic_formula.py --source synthetic
```

## Code

```python
from tire_nn.physics import (linear_tire, brush_combined, dugoff_tire,
                             MagicFormulaTire, MFParams, pacejka_lateral)

Fy_lin = linear_tire(alpha, kappa, Fz, C_alpha=45e3, C_kappa=50e3)[1]
Fy_brush = brush_combined(alpha, kappa, Fz, 50e3, 45e3, mu=1.0)[1]
Fy_dugoff = dugoff_tire(alpha, kappa, Fz, 45e3, 50e3, mu=1.0)[1]
Fy_mf, mu_eff = pacejka_lateral(alpha, Fz, MFParams(B=9., C=1.6, E=0.4, mu=1.0))
```
