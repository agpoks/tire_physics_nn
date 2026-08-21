# Thermal, wear and graining

Level 5: the tire is not stationary. Over a stint its grip changes because it heats up,
wears out and — in the right conditions — grains.

:::{warning}
The graining results in this framework come from **synthetic, weakly supervised** data.
They demonstrate that the structure can represent and identify these effects. They are
**not** validated real motorsport graining, and no quantitative claim about a real tire
may be based on them.
:::

## Slip power: the only energy input

```{math}
:label: eq-slip-power
P_{\text{slip}} = -\big(F_x v_{sx} + F_y v_{sy}\big) \;\ge\; 0
```

With SAE signs the force opposes the slip velocity, so this is non-negative for a
dissipative tire. It is clamped at zero because a negative value would mean the tire
*generates* energy, and a bad gradient step could otherwise cool a tire by sliding it.

This matters more than it looks. A model heated by "speed", or by "lateral
acceleration", gets braking, coasting and combined slip wrong, because none of those is
the actual dissipation. $-F \cdot v_{slip}$ is.

## Two-node thermal model

```{math}
:label: eq-thermal
\begin{aligned}
C_s\,\frac{\mathrm{d}T_s}{\mathrm{d}t} &= \eta\,P_{\text{slip}}
    - h_{sc}\,(T_s - T_c) - h_{sa}\,(T_s - T_{\text{road}}) \\
C_c\,\frac{\mathrm{d}T_c}{\mathrm{d}t} &= h_{sc}\,(T_s - T_c) - h_{ca}\,(T_c - T_{\text{air}})
\end{aligned}
```

```{figure} ../_static/figures/thermal_two_time_scales.png
:alt: two thermal time scales
:width: 75%

One energy input, two very different responses. The surface node responds within a
corner and drives grip; the core drifts over a stint and sets the operating window.
Reproducing this is why two nodes are the minimum — a single lumped temperature cannot
be both fast enough for the corner and slow enough for the stint.
```

The coupling $h_{sc}$ appears with opposite signs in both equations, so the pair
conserves energy apart from the explicit environment losses (tested).

**Modelling options.** A single node is simpler but cannot represent both time scales.
Three or more nodes (tread / belt / carcass) or a 1-D through-thickness diffusion model
are more faithful and are what tire manufacturers use, at the cost of parameters you
cannot identify from vehicle data. Two nodes is the smallest model that reproduces the
qualitative behaviour engineers actually reason about.

:::{note}
The default coefficients are **plausible engineering magnitudes for a racing tire, not
identified values** — sized so that ~1 kW of sustained slip power lifts the surface
roughly 15–25 K above the road. They exist so the model runs and plots sensibly.
Identify them before quoting any temperature quantitatively.
:::

## Wear: irreversible by construction

```{math}
:label: eq-wear
\frac{\mathrm{d}w}{\mathrm{d}t} = \mathrm{softplus}\big(f_w(\cdot)\big) \;\ge\; 0
```

Wear is a thermodynamic one-way street. `softplus` makes the monotonicity **exact**
rather than probable: there is no weight vector, and no gradient step, for which the
model can un-wear a tire. A penalty on $\dot w < 0$ would merely make it unlikely.

## Graining: reversible, but confined to $[0,1]$

```{math}
:label: eq-graining
\frac{\mathrm{d}g}{\mathrm{d}t} = (1-g)\,R_{\text{form}}(\cdot) - g\,R_{\text{clean}}(\cdot),
\qquad R_{\text{form}}, R_{\text{clean}} = \mathrm{softplus}(\cdot) \ge 0
```

The interval $[0,1]$ is an **invariant set**, and it costs nothing to see why:

- at $g = 0$ the sink term $-g R_{\text{clean}}$ vanishes, so $\dot g \ge 0$;
- at $g = 1$ the source term $(1-g)R_{\text{form}}$ vanishes, so $\dot g \le 0$.

No clamping and no penalty — and unlike a clamp, the gradient stays informative at the
boundary. For the *discrete* update the exact zero-order-hold solution is used, so the
bound survives any step size:

```{math}
:label: eq-graining-step
g_{k+1} = g_\infty + (g_k - g_\infty)\,e^{-(R_{\text{form}}+R_{\text{clean}})\Delta t},
\qquad
g_\infty = \frac{R_{\text{form}}}{R_{\text{form}}+R_{\text{clean}}} \in [0,1]
```

**Gating.** The rate networks receive *monotone* features — how far the surface is below
the working window, how far above, normalised slip power, load. So cold plus high slip
energy can drive formation and a warm surface can drive cleaning, but the *shape* of
each dependence stays learnable. The sign of the effect is imposed; the curve is not.

## Effective friction

```{math}
:label: eq-mu-eff
\mu_{\text{eff}} = \underbrace{\mu_{\text{base}}(T_s, F_z, p)}_{\text{grip window}}
\;\cdot\; e^{-k_w w} \;\cdot\; (1 - k_g\,g)
```

```{figure} ../_static/figures/effective_friction.png
:alt: temperature, wear and graining effects on friction
:width: 100%

The three factors. The temperature term is an inverted parabola, not a monotone
function: a *monotone* model would predict that a tire keeps gaining grip as it heats,
which is wrong at both ends (cold graining, hot overheating). The product form is
strictly positive for $k_g < 1$, so degradation can never invert the sign of friction —
the [friction ellipse](combined-slip) stays well posed for every reachable state.
```

Crucially the condition model only **scales the ellipse**, so odd symmetry and zero
force at zero slip still hold exactly, now with respect to a condition-dependent limit.

## The demonstrator

```{figure} ../_static/figures/graining_stint.png
:alt: synthetic graining stint
:width: 80%

A four-phase synthetic stint. **1** cold and aggressive: graining forms, reaching 0.74.
**2** hot and aggressive: the surface enters its window and cleaning overtakes formation
while wear keeps accumulating. **3** cool-down: low slip, the surface cools, graining
creeps back. **4** aggressive again: with the accumulated wear, grip never returns to
the phase-2 level. Wear is monotone throughout — structurally, not by luck.
```

## Why degradation is a natural UDE problem

Degradation modelling splits unusually cleanly into what is known and what is not, which
is precisely the setting a **universal differential equation** targets
{cite}`rackauckas2020ude`:

| | |
|---|---|
| **Known with confidence** | energy balance; dissipation is $-F \cdot v_{slip}$; wear is monotone; a fractional surface state lives in $[0,1]$; grip degrades multiplicatively |
| **Genuinely unknown** | the *kinetics* — how fast graining forms at a given temperature and slip energy, how cleaning scales with the operating point, how wear rate depends on compound and load |

Encoding the first column and learning the second is exactly what
{py:class}`~tire_nn.models.thermo_graining_tire.ThermoGrainingTire` does. It is also why
the guarantees survive training: the learned rates are wrapped in `softplus`, so no
weight vector can make wear reverse or push graining outside $[0,1]$.

The natural extension — not implemented here — is to recover a *symbolic* rate law from
the trained closure with sparse regression {cite}`brunton2016sindy`, turning "a network
that predicts the wear rate" into "an Arrhenius-type expression for the wear rate".

## Everything here is optional

```python
ThermoGrainingTire(steady, enable_thermal=True, enable_wear=False, enable_graining=False)
```

With all three off, the model reduces exactly to its steady-state core.

## Tests

`tests/test_thermo_graining.py`:
`test_wear_never_decreases_over_a_long_rollout_with_random_weights`,
`test_graining_stays_in_the_unit_interval_with_adversarial_rates` (with a deliberately
coarse $\Delta t = 0.5\,$s), `test_graining_boundaries_are_invariant_under_the_exact_update`,
`test_condition_only_scales_the_friction_ellipse_and_keeps_the_shape_guarantees`.
