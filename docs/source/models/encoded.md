# Symmetry- and envelope-encoded models

Two rungs of the ablation ladder that share a trunk, so comparing them isolates the
friction envelope.

```{figure} ../_static/diagrams/encoded_tire.png
:alt: EncodedTireNet architecture
:width: 100%
```

## `SymmetryTireNet` — P1 + P2

```{math}
:label: eq-encoded-force
q_x = \kappa \; g_x(\kappa^2,\, \alpha^2,\, F_z,\, c), \qquad
q_y = -\alpha \; g_y(\alpha^2,\, \kappa^2,\, F_z,\, c)
```

with $g_x, g_y > 0$ from a small MLP through a `softplus`. Because the network sees only
the **even invariants** and the result is multiplied by the odd factor:

```{math}
F_y(-\alpha) = -F_y(\alpha), \qquad F_x(-\kappa) = -F_x(\kappa), \qquad
F_x(\kappa{=}0) = F_y(\alpha{=}0) = 0
```

hold *identically* — not approximately, but to the last bit of the floating-point
representation, for any weights. Positivity of $g$ additionally gives dissipativity
($F_x \kappa \ge 0$, $F_y \alpha \le 0$): the force can never point along the slip.

### Load scaling

By default $g$ is multiplied by $F_z$, so the network predicts a normalised *stiffness*
rather than an absolute force. Force then scales linearly with load by construction —
the correct leading-order behaviour — leaving only the load sensitivity of $\mu$ to be
learned. This matters when a rig has measured three or four load levels.

### Where the real asymmetries go

Real tires *are* slightly asymmetric: ply steer, conicity and camber thrust all produce
force at zero slip angle. These are **not** folded into $g$; they enter through a
separate, switchable, additive term that is even in $(\alpha, \kappa)$:

```python
SymmetryTireNet(asymmetry=True)      # off by default
```

Keeping it separable means it stays individually reportable and physically named
("this tire has 40 N of ply steer") instead of being smeared into the weights.

### What it does *not* give you

Symmetry constrains the *shape*, not the *magnitude*. An untrained `SymmetryTireNet`
exceeds the friction limit by a factor of ~8, and it is the **worst** model in the
envelope-violation column of the [benchmark](../comparison/benchmarks) — worse than the
plain MLP. A correct shape with an unbounded magnitude is not a safe model.

## `EncodedTireNet` — P1 + P2 + P3

Adds the radial projection of [Combined slip](../physics/combined-slip):

```{math}
F_x = q_x\,s(\rho), \quad F_y = q_y\,s(\rho), \quad s(\rho) = \tanh(\rho)/\rho
```

with $\mu_x, \mu_y$ produced by a bounded head from $(F_z, c)$ — so the ellipse itself is
learned, but always inside a declared physical range, and the force is inside that
ellipse for any weights.

```python
from tire_nn.models import EncodedTireNet

model = EncodedTireNet(context_keys=("p", "Ts"), hidden=(32, 32),
                       envelope_mode="tanh", max_utilization=1.0)
out = model(alpha, kappa, Fz, ctx)
out.params["mu_x"], out.params["mu_y"], out.params["rho"]   # readable, plottable
```

`context["mu_scale"]`, when present, multiplies the ellipse — this is how the
[condition model](dynamic) and an external $\mu$ estimator feed in, without touching the
force afterwards, so the guarantee still holds with respect to the scaled ellipse.

## When to use which

| | `SymmetryTireNet` | `EncodedTireNet` |
|---|---|---|
| shape guarantees | ✓ | ✓ |
| bounded force | ✗ | ✓ |
| exposes $\mu$ | ✗ | ✓ |
| use it when | you want to isolate the effect of P2, or the force limit is enforced elsewhere | always, in practice |

## Tests

Every assertion runs with **adversarially randomised weights** — if a guarantee only
held after training it would be a penalty in disguise:
`test_zero_slip_gives_exactly_zero_force` (asserts bitwise equality with zero),
`test_odd_symmetry_is_exact_for_random_weights`,
`test_force_opposes_slip_direction_dissipativity`,
`test_friction_envelope_holds_for_random_network_output`.
