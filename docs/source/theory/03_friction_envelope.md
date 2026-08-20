# 3. The hard friction envelope (P3)

## The physics

The resultant tire force is bounded by the available friction. In elliptical form:

```{math}
:label: eq-ellipse
\left(\frac{F_x}{\mu_x F_z}\right)^{2} + \left(\frac{F_y}{\mu_y F_z}\right)^{2} \le 1 .
```

This is the constraint that a controller relies on most, and the one a black-box model
violates most damagingly — an optimiser actively seeks the region where the model
promises more grip than exists.

## Radial projection

Define the **utilisation radius** of a raw, unconstrained network output $(q_x,q_y)$:

```{math}
:label: eq-rho
\rho = \sqrt{\left(\frac{q_x}{\mu_x F_z}\right)^{2} + \left(\frac{q_y}{\mu_y F_z}\right)^{2}} ,
```

and map it into the ellipse with a smooth radial squashing:

```{math}
:label: eq-projection
\boxed{\;
F_x = q_x\, s(\rho), \qquad F_y = q_y\, s(\rho), \qquad
s(\rho) = \frac{\tanh \rho}{\rho}\; }
```

The projected utilisation is then $\hat\rho = \rho\, s(\rho) = \tanh\rho < 1$ for every
finite input — the constraint {eq}`eq-ellipse` holds **by construction**, for any
network weights, at any operating point, in or out of distribution.

```{figure} ../_static/figures/envelope_scaling.png
:alt: envelope scaling functions
:width: 100%

Left: the scaling factor $s(\rho)$. Right: the resulting utilisation. Both smooth
options are strictly bounded by 1; the hard clip $\min(1, 1/\rho)$ reaches the bound
too, but with a kink and with zero gradient beyond it.
```

## Why not a hard clip, and why not a penalty

**Why $\tanh$ and not $\min(1,1/\rho)$:**

1. A hard clip has **zero gradient** outside the ellipse. Learning dies exactly on the
   saturating samples — the ones that carry the information about $\mu$.
2. It puts a kink on the ellipse boundary. A Newton-type MPC will find that kink and
   stall on it.
3. $\tanh\rho/\rho$ is $C^\infty$ and equals $1 - \rho^2/3 + O(\rho^4)$ near the
   origin, so the linear-range cornering stiffness is untouched — the projection does
   nothing where nothing needs doing.

An algebraic alternative $s(\rho) = 1/\sqrt{1+\rho^2}$ is available
(`mode="algebraic"`) when a softer approach to the limit is wanted.

**Why the scaling is radial:** the contact-patch shear stress opposes the slip-velocity
vector, so the force direction is set by kinematics and only the magnitude is limited
by friction. Scaling both components equally preserves that direction; clipping one
component alone would rotate the force away from the slip direction and produce a
physically wrong combined-slip response.

**Why not a penalty.** A penalty term $\lambda\,\mathbb{E}[(\rho-1)_+^2]$:

- is satisfied only *in expectation over the training distribution*;
- is silently violated on extrapolation — the racing operating point;
- trades against RMSE through an arbitrary $\lambda$, so "how much physics" becomes a
  hyperparameter.

The penalty is still implemented in {py:func}`tire_nn.training.losses.friction_penalty`
— as the control experiment, not as the method.

```{figure} ../_static/figures/friction_ellipse.png
:alt: combined slip locus with and without the envelope
:width: 100%

Combined-slip force locus of two **untrained** models on a $1000\,\mathrm{N}$ tire.
Left: symmetry alone bounds nothing — the model reaches 40 kN, forty times the
physical limit. Right: with the projection every point lies inside the learned
ellipse. Note the different axis scales.
```

## Numerical caveat

In exact arithmetic $\tanh\rho < 1$ strictly. In float32, $\tanh$ rounds to exactly
$1.0$ for $\rho \gtrsim 8$, so deep in saturation the bound is *attained* rather than
approached. Where a solver needs a strictly interior point, pass
`max_utilization=0.999`; the missing 0.1 % is absorbed by the learned $\mu$.

## Where $\mu$ comes from

$\mu_x, \mu_y$ are not constants. They are produced by a bounded parameter head
({doc}`04_parameter_network`) from load and context, so the ellipse itself is learned
while staying inside a declared physical range — and, when the condition model of
{doc}`07_thermal_wear_graining` is enabled, they are modulated by temperature, wear
and graining.

## Code

```python
from tire_nn.layers import FrictionEnvelope, project_into_ellipse

Fx, Fy = project_into_ellipse(qx, qy, mu_x, mu_y, Fz, mode="tanh")
```

Implementation: {py:mod}`tire_nn.layers.friction_envelope`.

## Guaranteed by tests

- `test_friction_envelope_never_violated_for_extreme_inputs` — inputs up to $10^{12}$ N
- `test_friction_envelope_holds_for_random_network_output`
- `test_projection_preserves_force_direction` — cross product is zero
- `test_projection_is_identity_like_in_the_linear_range`
- `test_projection_is_differentiable_in_saturation`
- `test_symmetry_only_model_does_violate_the_envelope` — the ablation control

## References

The friction ellipse and combined-slip behaviour follow {cite}`pacejka2012tire` and
the brush-model derivation in {cite}`svendenius2007tire`. Physics-constrained network
outputs for racing vehicle dynamics are also used in {cite}`chrosniak2024deepdynamics`.
