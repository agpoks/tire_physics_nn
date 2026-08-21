# Combined slip and the friction ellipse

Level 2: the tire is asked for longitudinal and lateral force at the same time. This is
where models differ most, and where a wrong model is most dangerous — a controller
planning a corner-exit will sit exactly here.

## The physical constraint

The resultant force is bounded by available friction:

```{math}
:label: eq-ellipse
\left(\frac{F_x}{\mu_x F_z}\right)^{2} + \left(\frac{F_y}{\mu_y F_z}\right)^{2} \le 1
```

If $\mu_x = \mu_y$ this is the friction *circle*; the ellipse form allows the usual
observation that a tire generates slightly more longitudinal than lateral force.

## Three ways to get combined slip

### Similarity (normalised slip vector)

The theoretical slip vector sets the force *direction*; the pure-slip curve evaluated at
its magnitude sets the *size*:

```{math}
:label: eq-combined
\sigma_x = \kappa,\quad \sigma_y = \tan\alpha,\quad \sigma = \sqrt{\sigma_x^2+\sigma_y^2},
\qquad
F_x = \frac{\sigma_x}{\sigma}\,\big|F_{x0}(\sigma)\big|, \quad
F_y = -\frac{\sigma_y}{\sigma}\,\big|F_{y0}(\sigma)\big|
```

This is what the brush model implies {cite}`svendenius2007tire` and it needs no extra
coefficients. Used by `MagicFormulaTire` and `ParameterTireNet` in this framework.

:::{admonition} Theoretical slip breaks odd symmetry — a subtlety worth knowing
:class: important

The textbook normalisation divides by $1+\kappa$:
$(\sigma_x, \sigma_y) = (\kappa, \tan\alpha)/(1+\kappa)$. It is more accurate at large
*braking* slip, but it is **not odd in $\kappa$**, because the practical slip ratio is
itself an asymmetric definition — driving slip is unbounded above, braking slip is
bounded below by $-1$.

That asymmetry is *kinematic*, living in the definition of $\kappa$, not *constitutive*,
living in the tire. Since this project compares models under a symmetry claim, the
symmetric form is the **default** and the textbook form is opt-in:

```python
pacejka_combined(alpha, kappa, Fz, px, py, theoretical_slip=True)
```

Measured side by side, the symmetric form reports an odd-symmetry violation of exactly
0; the theoretical-slip form reports 0.22 in load-normalised force units.
:::

### Dugoff's built-in coupling

The $\lambda$ of {eq}`eq-dugoff-lambda` already contains both slips, so combined slip
comes for free with no extra construction — the model's main attraction.

### Magic Formula weighting functions

Full MF 6.x multiplies each pure-slip force by a cosine-form weighting function of the
other slip, with roughly twenty additional coefficients
{cite}`besselink2010magicformula`. Highest fidelity, and the only option if you have a
supplier's full coefficient set — but it needs a proper combined-slip measurement
campaign, which most projects do not have.

## What they actually do

```{figure} ../_static/figures/combined_slip_methods.png
:alt: combined slip force locus of three models
:width: 100%

Force locus over a grid of $(\alpha, \kappa)$, at $\mu F_z = 1$ kN. All three respect
the friction circle, but they fill it differently. The brush model reaches the boundary
smoothly from inside. Dugoff's locus shows the radial "spokes" of its piecewise
$f(\lambda)$. The Magic Formula's locus pulls *inside* the circle at large combined slip
because its post-peak decay reduces the force magnitude — physically real, and something
neither of the other two can represent.
```

## Why this must be a hard constraint in a learned model

A network fitted to tire data will happily predict
$\sqrt{F_x^2 + F_y^2} > \mu F_z$ somewhere. That matters more than it sounds, because an
optimiser inside an MPC *searches for the best achievable force* — so it will find
exactly the region where the model is unphysically optimistic and plan a lap the car
cannot drive.

This framework enforces {eq}`eq-ellipse` by a differentiable radial projection. Given a
raw network output $(q_x, q_y)$, define the normalised radius $\rho$ and scale:

```{math}
:label: eq-projection
\rho = \sqrt{\left(\frac{q_x}{\mu_x F_z}\right)^{2} + \left(\frac{q_y}{\mu_y F_z}\right)^{2}},
\qquad
F_x = q_x\, s(\rho), \quad F_y = q_y\, s(\rho), \quad s(\rho) = \frac{\tanh \rho}{\rho}
```

The projected radius is $\rho\,s(\rho) = \tanh\rho < 1$ for every finite input, so
{eq}`eq-ellipse` holds by construction, for any weights, anywhere.

```{figure} ../_static/figures/envelope_scaling.png
:alt: envelope scaling functions
:width: 100%

Left: the scaling factor. Right: the resulting utilisation, which can never reach 1.
The hard clip $\min(1, 1/\rho)$ also bounds the output but has zero gradient beyond the
limit — so learning dies exactly on the saturating samples that carry the information
about $\mu$ — and puts a kink on the ellipse that a Newton-type solver will find.
$\tanh\rho/\rho$ is $C^\infty$ and equals $1 - \rho^2/3 + O(\rho^4)$ near the origin, so
the linear-range stiffness is untouched.
```

The scaling is **radial**, preserving force direction: contact-patch shear opposes the
slip-velocity vector, so the direction is set by kinematics and only the magnitude is
limited by friction. Clipping one component alone would rotate the force away from the
slip direction.

```{figure} ../_static/figures/friction_ellipse.png
:alt: locus with and without the envelope
:width: 100%

Two *untrained* networks on a 1 kN tire. Left: symmetry alone bounds nothing — the model
reaches 40 kN, forty times the physical limit. Right: with the projection every point
lies inside the learned ellipse. Note the different axis scales.
```

## Numerical caveat

In exact arithmetic $\tanh\rho < 1$ strictly. In float32 $\tanh$ rounds to exactly $1.0$
for $\rho \gtrsim 8$, so deep in saturation the bound is *attained* rather than
approached. Pass `max_utilization=0.999` where a solver needs a strictly interior point;
the missing 0.1 % is absorbed by the learned $\mu$.

## Code

```python
from tire_nn.layers import FrictionEnvelope, project_into_ellipse, ellipse_radius

Fx, Fy = project_into_ellipse(qx, qy, mu_x, mu_y, Fz, mode="tanh")
rho = ellipse_radius(Fx, Fy, mu_x, mu_y, Fz)      # <= 1 always
```

## Tests

`test_friction_envelope_never_violated_for_extreme_inputs` (inputs to $10^{12}$ N),
`test_projection_preserves_force_direction`,
`test_projection_is_identity_like_in_the_linear_range`,
`test_projection_is_differentiable_in_saturation`,
`test_symmetry_only_model_does_violate_the_envelope`.
