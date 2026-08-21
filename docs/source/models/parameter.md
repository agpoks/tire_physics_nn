# `ParameterTireNet` — predicting a tire, not a force

The interpretable end of the ladder. The network never outputs a force; it outputs the
*coefficients* of the Magic Formula, and the analytical law does the rest.

```{figure} ../_static/diagrams/parameter_net.png
:alt: parameter network architecture
:width: 100%
```

## Equations

```{math}
:label: eq-paramnet
\big(\mu, B, C, E, k_\mu, \sigma_x, \sigma_y\big)_{x,y}
    = \Pi_{\text{bounds}}\Big(f_\theta(F_z,\, c)\Big),
\qquad
(F_x, F_y) = \mathrm{MF}\big(\alpha, \kappa, F_z;\ \cdot\big)
```

with the bounding map

```{math}
:label: eq-bounds
p = p_{\min} + \mathrm{softplus}(z), \qquad
p = p_{\min} + (p_{\max}-p_{\min})\,\sigma(z)
```

```{figure} ../_static/figures/bounded_parameters.png
:alt: bounded parameter transforms
:width: 100%

The parameter is valid for *any* raw network output, including $z = \pm 30$. There is no
training step at which the model holds a non-physical tire.
```

## Why bounded transforms rather than clipping in the loss

A negative $C$ flips the curve; a negative $D$ is a negative peak force. In those regions
the loss landscape is meaningless and the gradients are actively misleading. Clipping in
the loss leaves the *raw* parameter free to wander there and only masks the symptom. The
monotone transform makes the parameter valid at **every step**, and each head's bias is
initialised so an untrained model already starts at a sensible tire ($\mu = 1.0$,
$B = 10$, $C = 1.6$, $\sigma_y = 0.30\,$m).

Default ranges bracket published passenger-car values and the fitted small-scale racing
values in `On-Track-SysID` ($B \approx 7\text{–}8$, $C \approx 1.6\text{–}2.1$,
$E \approx 0.4\text{–}0.5$) {cite}`ontracksysid` with margin.

:::{admonition} The coefficients depend on the condition, never on the slip
:class: important
At a given operating point a tire has *one* coefficient set. `parameters_at(Fz, context)`
therefore receives load, pressure, temperature and tire identity — but never $\alpha$ or
$\kappa$. Enforced structurally, and tested by
`test_parameter_model_coefficients_depend_only_on_condition_not_on_slip`.
:::

## What you get back

```python
from tire_nn.models import ParameterTireNet

model = ParameterTireNet(context_keys=("p", "Ts"))
out = model(alpha, kappa, Fz, ctx)

out.params["mu_y"]      # effective (load-corrected) peak friction
out.params["mu_y0"]     # nominal value at Fz_ref
out.params["C_alpha"]   # cornering stiffness B*C*D [N/rad]
out.params["sigma_y"]   # relaxation length [m], feeds RelaxationTireCell
out.params["B_y"], out.params["C_y"], out.params["E_y"]
```

Example output across load, from notebook 1:

| $F_z$ [N] | $\mu_y$ | $B_y$ | $C_y$ | $C_\alpha$ [N/rad] | $\sigma_y$ [m] |
|---|---|---|---|---|---|
| 300 | 1.161 | 8.59 | 1.52 | 4 548 | 0.30 |
| 1 200 | 1.084 | 8.51 | 1.52 | 16 844 | 0.30 |
| 3 000 | 0.932 | 8.55 | 1.52 | 36 387 | 0.30 |

That is a tire you can hand to an MPC that expects Pacejka coefficients, or plot against
a rig measurement.

:::{note}
`params["mu_x"]`/`["mu_y"]` are the **load-corrected** values $\mu(F_z)$; the nominal
values at $F_{z0}$ are `["mu_x0"]`/`["mu_y0"]`. Reporting the nominal value understates
the friction ellipse at light load and overstates it at high load — a bug this codebase
had and fixed.
:::

## Strengths and limits

**Strengths.** Every guarantee of the Magic Formula, plus interpretability, plus strong
regularisation: on 210 training samples it scores 17.4 N where a direct `scipy` fit of
the *same* Magic Formula scores 196.9 N. Bounding the parameter space is doing real work,
not just bookkeeping.

**Limits.** You inherit the law's blind spots — whatever the Magic Formula cannot
express, this model cannot express either. On data generated *by* a Magic Formula it
wins by construction, so its benchmark result must be read with that caveat
([benchmarks](../comparison/benchmarks)).

## Optional hard envelope

The Magic Formula is already bounded by $D = \mu F_z$, so the projection is off by
default. `hard_envelope=True` adds it for combined slip beyond what the similarity
method guarantees.
