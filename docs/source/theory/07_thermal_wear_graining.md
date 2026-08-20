# 7. Thermal, wear and graining states (P7)

:::{warning}
This chapter's experiment uses **synthetic, weakly supervised** data. It demonstrates
that the encoded structure can represent and identify these effects. It is **not**
validated real motorsport graining, and no quantitative claim about a real tire may be
based on it.
:::

## The state

```{math}
z = \big[\,T_s,\; T_c,\; w,\; g\,\big]
```
surface temperature, core temperature, wear and graining. The **structure is fixed**;
only the rates are learned.

## Slip power

The single energy input is the dissipated slip power:

```{math}
:label: eq-slip-power
P_{\text{slip}} = -\big(F_x v_{sx} + F_y v_{sy}\big) \;\ge\; 0 .
```

With the SAE signs the force opposes the slip velocity, so this is non-negative for a
dissipative tire; it is clamped at zero because a negative value would mean the tire
*generates* energy, and a bad gradient step could otherwise cool a tire by sliding it.

This term matters more than it looks: a model heated by "speed" or by "lateral
acceleration" gets braking, coasting and combined slip wrong, because none of those is
the actual dissipation.

## Two-node thermal model

```{math}
:label: eq-thermal
\begin{aligned}
C_s\,\frac{\mathrm{d}T_s}{\mathrm{d}t} &= \eta\,P_{\text{slip}}
    - h_{sc}\,(T_s - T_c) - h_{sa}\,(T_s - T_{\text{road}}) \\
C_c\,\frac{\mathrm{d}T_c}{\mathrm{d}t} &= h_{sc}\,(T_s - T_c) - h_{ca}\,(T_c - T_{\text{air}})
\end{aligned}
```

Two nodes is the minimal structure with the two time scales that are actually observed:
a light surface node that responds within a corner and drives grip
($C_s/(h_{sc}+h_{sa}) \sim 30\,$s), and a core node an order of magnitude heavier that
drifts over a stint and sets the operating window. The coupling $h_{sc}$ appears with
opposite signs in both equations, so the pair conserves energy apart from the explicit
environment losses (tested).

:::{note}
The default coefficients are **plausible engineering magnitudes for a racing tire, not
identified values** — sized so ~1 kW of sustained slip power lifts the surface roughly
15–25 K above the road. They exist so the model runs and plots sensibly. Identify them
(or learn a correction) before quoting any temperature quantitatively.
:::

## Wear: irreversible by construction

```{math}
:label: eq-wear
\frac{\mathrm{d}w}{\mathrm{d}t} = \mathrm{softplus}\big(f_w(\cdot)\big) \;\ge\; 0 .
```

Wear is a thermodynamic one-way street. `softplus` makes the monotonicity **exact**
rather than probable: there is no weight vector, and no gradient step, for which the
model can un-wear a tire.

## Graining: reversible, but confined to $[0,1]$

```{math}
:label: eq-graining
\frac{\mathrm{d}g}{\mathrm{d}t} = (1-g)\,R_{\text{form}}(\cdot) - g\,R_{\text{clean}}(\cdot),
\qquad R_{\text{form}}, R_{\text{clean}} = \mathrm{softplus}(\cdot) \ge 0 .
```

The interval $[0,1]$ is an **invariant set**, and it costs nothing to see why:

- at $g = 0$ the sink term $-g R_{\text{clean}}$ vanishes, so $\dot g \ge 0$;
- at $g = 1$ the source term $(1-g)R_{\text{form}}$ vanishes, so $\dot g \le 0$.

No clamping and no penalty are needed, and — unlike a clamp — the gradient stays
informative at the boundary. For the *discrete* update the exact zero-order-hold
solution is used, so the bound survives any step size:

```{math}
:label: eq-graining-step
g_{k+1} = g_\infty + (g_k - g_\infty)\,e^{-(R_{\text{form}}+R_{\text{clean}})\Delta t},
\qquad
g_\infty = \frac{R_{\text{form}}}{R_{\text{form}}+R_{\text{clean}}} \in [0,1] .
```

### Gating

The rate networks receive **monotone gating features**: how far the surface is below
the working window, how far above, the normalised slip power, and the load. So cold +
high slip energy can drive formation, and a warm surface can drive cleaning — but the
*shape* of each dependence stays learnable. The sign of the effect is imposed; the
curve is not.

## Effective friction

```{math}
:label: eq-mu-eff
\mu_{\text{eff}} = \underbrace{\mu_{\text{base}}(T_s, F_z, p)}_{\text{grip window}}
\;\cdot\; e^{-k_w w} \;\cdot\; (1 - k_g\,g)
```

with the temperature term an inverted parabola clamped from below,

```{math}
\mu_{\text{base}} \propto \max\!\left(1 - c_T\left(\frac{T_s - T_{\text{opt}}}{T_{\text{width}}}\right)^{2},\ 0.05\right).
```

A *monotone* temperature model would predict that a tire keeps gaining grip as it
heats, which is wrong at both ends (cold graining, hot overheating). The product form
is strictly positive for $k_g < 1$, so degradation can never invert the sign of the
friction — the envelope of {doc}`03_friction_envelope` stays well posed for every state
the condition model can reach. The condition **only scales the ellipse**, so odd
symmetry and zero force at zero slip still hold exactly.

## The demonstrator

```{figure} ../_static/figures/graining_stint.png
:alt: synthetic graining stint
:width: 85%

A four-phase synthetic stint. **Phase 1** (cold, aggressive): graining forms rapidly,
reaching 0.74. **Phase 2** (hot, aggressive): the surface enters its window and
cleaning overtakes formation while wear keeps accumulating. **Phase 3** (cool-down):
low slip, the surface cools and graining creeps back up. **Phase 4**: with the
accumulated wear, grip never returns to the phase-2 level. Wear is monotone
throughout — structurally, not by luck.
```

Experiment 4 trains the neural rate networks on noisy weak labels for $T_s$ and $g$
plus the force, then **asserts the structural guarantees on the trained model**:

```text
[checks] wear_monotone=True  graining_in_unit_interval=True
         temperatures_finite=True  slip_power_non_negative=True
```

## Everything here is optional

```python
ThermoGrainingTire(steady, enable_thermal=True, enable_wear=False, enable_graining=False)
```

Each state can be switched off independently; with all three off, the model reduces
exactly to its steady-state core.

## Guaranteed by tests

`tests/test_thermo_graining.py`:

- `test_wear_never_decreases_over_a_long_rollout_with_random_weights`
- `test_graining_stays_in_the_unit_interval_with_adversarial_rates` — including a
  deliberately coarse $\Delta t = 0.5\,$s
- `test_graining_boundaries_are_invariant_under_the_exact_update`
- `test_temperature_stays_finite_and_rises_under_slip`
- `test_condition_only_scales_the_friction_ellipse_and_keeps_the_shape_guarantees`
- `test_synthetic_scenario_shows_the_intended_qualitative_behaviour`
