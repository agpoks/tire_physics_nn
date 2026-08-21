# Transient behaviour: relaxation

Level 4: force does not follow slip instantaneously, because the contact patch has to
deform first.

## The physics

The key fact is that the transient is parameterised by **travelled distance**, not by
time. With relaxation lengths $\sigma_x, \sigma_y$:

```{math}
:label: eq-tau
\tau_i = \frac{\sigma_i}{|v_x| + \varepsilon}, \qquad i \in \{x, y\}
```

```{math}
:label: eq-relaxation
\frac{\mathrm{d}F_i}{\mathrm{d}t} = \frac{F_{i,\text{ss}} - F_i}{\tau_i}
\qquad\Longleftrightarrow\qquad
\sigma_i \frac{\mathrm{d}F_i}{\mathrm{d}s} + F_i = F_{i,\text{ss}}
```

where $s$ is travelled distance and $F_{ss}$ comes from any [steady-state
law](steady-state). The relaxation length is the distance the tire must roll to cover
$1 - 1/e \approx 63\%$ of a step: typically $0.1$–$0.6\,$m for a passenger tire, a few
centimetres for a small-scale racing tire {cite}`pacejka2012tire`.

```{figure} ../_static/figures/relaxation_step.png
:alt: step response versus time and versus distance
:width: 100%

The same model, the same slip step, three speeds. **Left**, against time: the response
looks completely different at each speed. **Right**, against travelled distance: the
three curves collapse onto one, and the 63 % point *is* the relaxation length.
```

## Modelling options

| option | what it captures | cost |
|---|---|---|
| **none** (quasi-static) | nothing; force jumps with slip | free; wrong during transients, and the error is worst exactly during a steering input |
| **first-order relaxation** {eq}`eq-relaxation` | the dominant lag, correct speed scaling | one length per axis; sequential integration |
| **carcass / rigid-ring models** | belt vibration modes, ~40–80 Hz | many parameters, needs dedicated measurement |
| **generic RNN / Neural ODE** | whatever is in the data | many parameters, no guaranteed speed scaling or stability |

First-order relaxation is the sweet spot for vehicle control: it captures the effect
that matters at handling frequencies with one interpretable parameter per axis.

## Why encode it rather than learn it

Three properties come free from {eq}`eq-tau`–{eq}`eq-relaxation`:

**Stability.** $\sigma_i = \sigma_{\min} + \mathrm{softplus}(z) > 0$, so $\tau_i > 0$ and
the ODE is contractive. The model *cannot* learn a divergent transient, whatever the
data or the optimiser does.

**Speed extrapolation.** The speed dependence is exact, so a model identified at 15 m/s
is still right at 40 m/s.

**Interpretability.** What is learned is one length per axis — a number comparable with a
rig measurement — instead of a gate matrix.

**Correct standstill limit.** The regularisation sits on $|v_x|$, so as the wheel stops
$\tau \to \sigma/\varepsilon$ grows and the force freezes rather than blowing up. A
non-rolling tire does not relax.

## Integration

Four integrators share one interface:

`"exact"`
: $F \leftarrow F_{ss} + (F - F_{ss})e^{-\Delta t/\tau}$ — exact under a zero-order-hold
  input and unconditionally stable for any step size. The right default for coarsely
  sampled data.

`"rk4"`
: Classical fourth order, zero-order hold within a step.

`"euler"`
: Explicit Euler; needs $\Delta t < 2\tau$. `check_step_size(dt, vx_max)` refuses unsafe
  step sizes, since $\tau$ is smallest at the highest speed.

`"odeint"`
: Optional {py:mod}`torchdiffeq` path {cite}`torchdiffeq` with linear input
  interpolation, for stiff or irregularly sampled data. Imported lazily.

## Measured: distance or time?

The decisive diagnostic is the **rise-distance ratio** — the distance to reach 63 % of a
step at 30 m/s divided by the same at 10 m/s.

```{figure} ../_static/figures/transient_ratio.png
:alt: rise distance ratio by model
:width: 75%

Ratio $\approx 1$ means the transient is parameterised by distance (physically correct);
$\approx 3$ means a fixed *time* constant was learned. Two independent runs. The Neural
ODE consistently learned a time constant; the GRU produced 2.25 in one run and 0.53 in
the other — i.e. **no stable scaling law at all**. Only the encoded cell lands near 1
every time, because it cannot do anything else.
```

That is the real result: without the structure, whatever speed dependence the model ends
up with is a property of the run, not of the tire. The encoded cell is also the most
accurate (110 N rollout RMSE against the GRU's 203 N) and recovers
$\sigma_y \approx 0.25$ m against a true $0.30$ m.

## Training on trajectories, not one step ahead

```python
TrainConfig(mode="sequence", dt=0.002, integrator="rk4", targets=("Fx", "Fy"))
```

A one-step-ahead loss is dominated by the steady-state map and barely constrains $\tau$.
The trajectory loss is what makes the relaxation length identifiable.

## Tests

`tests/test_relaxation.py`: `test_time_constant_scales_inversely_with_speed`,
`test_rise_distance_is_speed_invariant`,
`test_integrators_agree_on_a_smooth_trajectory`,
`test_relaxed_force_never_leaves_the_friction_ellipse`,
`test_relaxation_recovers_the_true_relaxation_length_on_synthetic_data`.
