# 2. Odd symmetry and zero force at zero slip (P2)

## The physics

An isotropic tire on a symmetric road, at zero camber and with no ply steer or
conicity, satisfies

```{math}
:label: eq-odd
F_y(-\alpha, \kappa, F_z) = -F_y(\alpha, \kappa, F_z),
\qquad
F_x(\alpha, -\kappa, F_z) = -F_x(\alpha, \kappa, F_z),
```

and, as an immediate consequence,

```{math}
:label: eq-zero
F_x(\kappa = 0) = 0, \qquad F_y(\alpha = 0) = 0 .
```

Equation {eq}`eq-zero` is the statement that a tire rolling without slip transmits no
shear force. A second, equally physical requirement is **dissipativity**: the contact
shear opposes the slip, so

```{math}
:label: eq-dissipativity
F_x\,\kappa \ge 0, \qquad F_y\,\alpha \le 0 .
```

## The encoded parameterisation

Write the force as an odd factor times an even function:

```{math}
:label: eq-encoded-force
\boxed{
\begin{aligned}
q_x &= \kappa \; g_x(\kappa^2,\, \alpha^2,\, F_z,\, c) \\
q_y &= -\alpha \; g_y(\alpha^2,\, \kappa^2,\, F_z,\, c)
\end{aligned}}
```

with $g_x, g_y > 0$ produced by a small MLP whose output passes through a `softplus`,
and $c$ an optional context vector (speed, pressure, temperature, tire id).

The network sees only the **even invariants** $\kappa^2$ and $\alpha^2$. Therefore:

- {eq}`eq-odd` holds identically, because $g_x, g_y$ are unchanged when
  $(\alpha,\kappa)\to(-\alpha,-\kappa)$ while the prefactors flip sign;
- {eq}`eq-zero` holds identically, because of the explicit factor $\kappa$ (resp.
  $-\alpha$) — not approximately, but to the last bit of the floating-point
  representation;
- {eq}`eq-dissipativity` holds because $g_x, g_y > 0$.

None of this depends on training. It is true for random weights, for diverged weights,
and outside the training envelope.

```{figure} ../_static/figures/symmetry_zero_slip.png
:alt: MLP vs symmetry-encoded network near zero slip
:width: 100%

Four random weight initialisations of each model, **untrained**. Left: the plain MLP
predicts a large lateral force at zero slip angle and has no symmetry. Right: the
encoded network passes through the origin and is exactly odd for every seed. After
training the MLP's offset shrinks but never vanishes, and it returns as soon as the
model is evaluated off-distribution.
```

### Load scaling

By default $g$ is multiplied by $F_z$, so the network predicts a *normalised
stiffness* rather than an absolute force:

```{math}
q_y = -\alpha\, F_z\, \hat g_y(\cdot).
```

Force then scales linearly with load by construction, which is the correct
leading-order behaviour; the remaining nonlinearity is the load sensitivity of $\mu$,
handled by the envelope in {doc}`03_friction_envelope`. This removes most of the load
dependence from the learning problem, which matters when a rig has measured only three
or four load levels.

### Where the real asymmetries go

Real tires *are* slightly asymmetric: ply steer, conicity and camber thrust all
produce a force at zero slip angle. These are **not** folded into $g$. They enter
through a separate, switchable, additive term $S_v(F_z,\gamma,c)$ that is even in
$(\alpha,\kappa)$:

```python
field = OddSymmetricForceField(asymmetry=True)   # off by default
```

Keeping the asymmetry separable means it is (a) individually reportable, (b)
switchable off for the symmetry tests, and (c) physically named instead of smeared
into the weights — so "this tire has 40 N of ply steer" remains a statement you can
read off the model.

## Code

```python
from tire_nn.models import SymmetryTireNet

model = SymmetryTireNet(context_keys=("p",), hidden=(32, 32))
out = model(alpha, kappa, Fz, {"p": pressure})
```

Implementation: {py:mod}`tire_nn.layers.symmetry`,
{py:mod}`tire_nn.models.encoded_tire`.

## Guaranteed by tests

`tests/test_layers.py` and `tests/test_models.py`, all run with **adversarially
randomised weights** (`conftest.randomize_`, $\sigma = 3$):

- `test_zero_slip_gives_exactly_zero_force` — asserts bitwise equality with zero
- `test_odd_symmetry_is_exact_for_random_weights`
- `test_Fy_is_zero_whenever_alpha_is_zero_even_under_longitudinal_slip`
- `test_force_opposes_slip_direction_dissipativity`

If a guarantee only held after training, it would be a penalty in disguise — hence
the random weights.

## References

The symmetry and zero-slip properties are standard tire modelling assumptions
{cite}`pacejka2012tire`; the brush model {cite}`svendenius2007tire` satisfies them from
first principles, which is why it is the reference baseline here.
