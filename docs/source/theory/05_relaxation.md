# 5. Relaxation dynamics (P5)

## The physics

Tire force does not respond instantaneously to a change in slip: the contact patch has
to deform first. Crucially, the transient is parameterised by **travelled distance**,
not by time. With relaxation lengths $\sigma_x, \sigma_y$:

```{math}
:label: eq-tau
\tau_i = \frac{\sigma_i}{|v_x| + \varepsilon}, \qquad i \in \{x, y\}
```

```{math}
:label: eq-relaxation
\boxed{\;\frac{\mathrm{d}F_i}{\mathrm{d}t} = \frac{F_{i,\text{ss}} - F_i}{\tau_i}\;}
```

where $F_{i,\text{ss}}$ is the steady-state force from any static tire model.
Equivalently, in the space domain, $\sigma_i \,\mathrm{d}F_i/\mathrm{d}s + F_i =
F_{i,\text{ss}}$ — the relaxation length is the distance the tire must roll to cover
$1 - 1/e \approx 63\%$ of a step.

Typical values: $0.1$–$0.6\,$m for a passenger tire, shorter for small-scale racing
tires {cite}`pacejka2012tire`.

```{figure} ../_static/figures/relaxation_step.png
:alt: step response versus time and versus distance
:width: 100%

The same model, the same step in slip angle, three speeds. **Left**, against time: the
response looks completely different at each speed. **Right**, against travelled
distance: the three curves collapse onto one, and the 63 % point is the relaxation
length. A generic recurrent model has to *learn* this collapse from data — and usually
only learns it for the speeds it saw.
```

## Why this is encoded

Three properties come for free from {eq}`eq-tau`–{eq}`eq-relaxation`:

Stability
: $\sigma_i = \sigma_{\min} + \text{softplus}(z) > 0$, so $\tau_i > 0$ and the ODE is
  contractive. The model **cannot** learn a divergent transient, whatever the data or
  the optimiser does.

Speed extrapolation
: The speed dependence is exact, so a model identified at 15 m/s is still right at
  40 m/s. This is the property that Experiment 2 measures.

Interpretability
: What is learned is one length per axis — a number an engineer can compare with a rig
  measurement — instead of a gate matrix.

Correct standstill limit
: The regularisation $\varepsilon$ sits on $|v_x|$, so as the wheel stops, $\tau \to
  \sigma/\varepsilon$ becomes large and the force *freezes* rather than blowing up. A
  non-rolling tire does not relax, because relaxation is a rolling-distance phenomenon.

## Integration

Four integrators share one interface, so the choice is a config flag, not a rewrite:

`"exact"`
: $F \leftarrow F_{ss} + (F - F_{ss})\,e^{-\Delta t/\tau}$. Exact for a
  zero-order-hold input and **unconditionally stable** for any step size. The right
  default when the data is sampled coarsely relative to $\tau$.

`"rk4"`
: Classical fourth order with zero-order hold within a step.

`"euler"`
: Explicit Euler. Requires $\Delta t < 2\tau$; the model refuses unsafe step sizes via
  `check_step_size(dt, vx_max)`, since $\tau$ is smallest at the highest speed.

`"odeint"`
: Optional {py:mod}`torchdiffeq` path {cite}`torchdiffeq` with linear input
  interpolation, for stiff or irregularly sampled data. Imported lazily — the package
  is optional and the fixed-step integrators are the default.

## Composition

The cell wraps *any* static model, so relaxation composes with every other prior:

```python
from tire_nn.models import EncodedTireNet, ParameterTireNet
from tire_nn.models.relaxation_tire import RelaxationTireCell

cell = RelaxationTireCell(EncodedTireNet(context_keys=("vx",)))
F = cell.rollout(alpha, kappa, Fz, vx, dt=0.002, context={"vx": vx}, method="rk4")

# or reuse the relaxation length the parameter network already predicts:
cell = RelaxationTireCell(ParameterTireNet(), sigma_from_steady=True)
```

Because the relaxed force contracts toward $F_{ss}$, and $F_{ss}$ is already inside the
friction ellipse ({doc}`03_friction_envelope`), the transient force never leaves the
ellipse either.

## Training on trajectories, not one step ahead

The sequence training mode supervises the **whole rollout**:

```python
TrainConfig(mode="sequence", dt=0.002, integrator="rk4", targets=("Fx", "Fy"))
```

A one-step-ahead loss is dominated by the steady-state map and barely constrains
$\tau$; the trajectory loss is what makes the relaxation length identifiable.

## What Experiment 2 measures

The decisive diagnostic is the **rise distance ratio**: the distance travelled to reach
63 % of a step, measured at 30 m/s divided by the same at 10 m/s.

- ratio $\approx 1$ — the transient is parameterised by distance (physically correct);
- ratio $\approx 3$ — a fixed *time* constant was learned, so the distance scales with
  speed;
- any other value — the model has no consistent speed law at all.

Two independent runs (the Experiment 2 script, and notebook 2 with a shorter budget),
true $\sigma_x = 0.15$, $\sigma_y = 0.30\,$m:

| model | rise-distance ratio (run 1) | ratio (run 2) | rollout $F_y$ RMSE [N], run 2 |
|---|---|---|---|
| static tire net | – (no transient) | – | 293 |
| generic GRU {cite}`cho2014gru` | 2.25 | 0.53 | 203 |
| generic Neural ODE {cite}`chen2018neuralode` | 3.00 | 3.00 | 252 |
| **relaxation cell** | **1.25** | **1.15** | **110** |
| **relaxation + ParameterNet** | **0.92** | – | – |

The result to take away is not that the baselines are uniformly worse at one number —
it is that **their speed law is a property of the run rather than of the tire**. The
Neural ODE consistently learned a fixed time constant; the GRU produced 2.25 in one run
and 0.53 in the other, i.e. no stable scaling at all. Only the encoded cell lands near 1
every time, because it cannot do anything else.

The encoded cell also recovers the physical parameter: $\sigma_y \approx 0.25$–$0.28\,$m
against a true $0.30\,$m, from a few epochs of step tests. Rerun with
`python experiments/train_relaxation.py`.

## Guaranteed by tests

`tests/test_relaxation.py`:

- `test_relaxation_lengths_are_positive_for_adversarial_weights`
- `test_time_constants_are_positive_and_finite_including_standstill`
- `test_time_constant_scales_inversely_with_speed`
- `test_rise_distance_is_speed_invariant`
- `test_integrators_agree_on_a_smooth_trajectory`
- `test_relaxed_force_never_leaves_the_friction_ellipse`
- `test_relaxation_recovers_the_true_relaxation_length_on_synthetic_data`
