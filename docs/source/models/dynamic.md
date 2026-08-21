# Dynamic models: relaxation and condition states

Two wrappers that add a time dimension to any constitutive model. Both are **universal
differential equations** {cite}`rackauckas2020ude` — the equation structure is fixed
physics and only the closure terms are learned:

```{math}
\dot z = \underbrace{f_{\text{known}}(z, u)}_{\text{balance / structure}}
        \;+\; \underbrace{g_\theta(z, u)}_{\text{unknown kinetics}}
```

See [integration pattern 6](../methods/integration.md#this-pattern-has-a-name-ude) for
why this matters and how it compares with a fully learned Neural ODE.

## `RelaxationTireCell`

```{figure} ../_static/diagrams/relaxation_cell.png
:alt: relaxation cell block diagram
:width: 100%
```

```{math}
\tau_i = \frac{\sigma_i}{|v_x| + \varepsilon}, \qquad
\frac{\mathrm{d}F_i}{\mathrm{d}t} = \frac{F_{i,\text{ss}} - F_i}{\tau_i}
```

The physics, the integrator options and the measured evidence are in
[Transient](../physics/transient). What matters at the model level:

```python
from tire_nn.models import EncodedTireNet, ParameterTireNet
from tire_nn.models.relaxation_tire import RelaxationTireCell

cell = RelaxationTireCell(EncodedTireNet(context_keys=("vx",)))
F = cell.rollout(alpha, kappa, Fz, vx, dt=0.002, context={"vx": vx}, method="rk4")

# or reuse the relaxation length the parameter network already predicts, instead of
# learning a second, inconsistent set:
cell = RelaxationTireCell(ParameterTireNet(), sigma_from_steady=True)
```

The cell wraps *any* static model, so relaxation composes with every other prior. Because
the relaxed force contracts toward $F_{ss}$, and $F_{ss}$ is already inside the friction
ellipse, the transient force never leaves it either.

`forward()` returns the steady state, so the cell is drop-in compatible with the static
models wherever a quasi-static evaluation is wanted.

## `ThermoGrainingTire`

```{figure} ../_static/diagrams/condition_states.png
:alt: condition state block diagram
:width: 100%
```

Latent state $z = [T_s, T_c, w, g]$ with the structure fixed and only the rates learned.
Equations, gating and the demonstrator: [Thermal, wear, graining](../physics/thermal-wear).

```python
from tire_nn.models.thermo_graining_tire import ThermoGrainingTire

model = ThermoGrainingTire(EncodedTireNet(context_keys=("vx",)),
                           enable_thermal=True, enable_wear=True, enable_graining=True)

forces, states, extra = model.rollout_condition(
    alpha, kappa, Fz, vx, dt=0.05, T_road=305.0, T_air=298.0)
Ts, Tc, wear, graining = (states[..., i] for i in range(4))
```

The condition only **scales the friction ellipse**, through `context["mu_scale"]`, so the
shape guarantees of the wrapped model are untouched.

### Structural guarantees, checked adversarially

With weights drawn from $\mathcal{N}(0, 20^2)$ and a deliberately absurd $\Delta t = 0.5$ s:

```text
wear monotone non-decreasing : True
graining in [0, 1]           : 0.0000 .. 0.1789
slip power non-negative      : True
temperatures finite          : True
```

These are properties of the equations, not of training.

## Cost

Both wrappers integrate sequentially, which is the main runtime expense in this
framework — a 400-step rollout is 400 forward passes with autograd through each. Keep
training windows short, prefer the `"exact"` integrator when the sampling is coarse, and
see the [experiment guide](../guides/experiments) for the budgets used here.
