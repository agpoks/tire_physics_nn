# 6. Four wheels, one tire model (P6)

## The physics

Planar rigid-body dynamics, with no learned correction anywhere:

```{math}
:label: eq-newton-euler
\begin{aligned}
m\left(\dot v_x - r v_y\right) &= \sum_{i} F_{x,i}^{\text{body}} - F_{\text{drag}} \\
m\left(\dot v_y + r v_x\right) &= \sum_{i} F_{y,i}^{\text{body}} \\
I_z\, \dot r &= \sum_{i} \left(x_i F_{y,i}^{\text{body}} - y_i F_{x,i}^{\text{body}}\right)
\end{aligned}
```

with the wheel-to-body rotation by the per-corner steering angle,

```{math}
:label: eq-wheel-to-body
\begin{aligned}
F_{x,i}^{\text{body}} &= F_{x,i}\cos\delta_i - F_{y,i}\sin\delta_i\\
F_{y,i}^{\text{body}} &= F_{x,i}\sin\delta_i + F_{y,i}\cos\delta_i
\end{aligned}
```

and the exact corner geometry (wheel order is `FL, FR, RL, RR` project-wide):

```{math}
:label: eq-corner-geometry
x_i = \big(l_f,\; l_f,\; -l_r,\; -l_r\big), \qquad
y_i = \Big(\tfrac{t_f}{2},\; -\tfrac{t_f}{2},\; \tfrac{t_r}{2},\; -\tfrac{t_r}{2}\Big).
```

Each corner's contact velocity follows from rigid-body kinematics,

```{math}
:label: eq-corner-velocity
\mathbf{v}_i = \mathbf{v}_{\text{CoG}} + \boldsymbol\omega \times \mathbf{r}_i
\;\Longrightarrow\;
v_{x,i} = v_x - r\, y_i,\qquad v_{y,i} = v_y + r\, x_i ,
```

and feeds the slip definitions of {doc}`01_slip_kinematics`.

## Load transfer

```{math}
:label: eq-load-transfer
\begin{aligned}
F_{z,f} &= \frac{m\,(g\,l_r - a_x h_{cg})}{L}, &
F_{z,r} &= \frac{m\,(g\,l_f + a_x h_{cg})}{L}, & L &= l_f + l_r\\[4pt]
\Delta F_{z,f} &= \frac{F_{z,f}\, a_y\, h_{cg}}{g\, t_f}, &
\Delta F_{z,r} &= \frac{F_{z,r}\, a_y\, h_{cg}}{g\, t_r} &&
\end{aligned}
```

with $F_{z,\text{FL}} = F_{z,f}/2 - \Delta F_{z,f}$ and so on. The transfers cancel in
the sum by construction, so $\sum_i F_{z,i} = mg$ **exactly**, at every operating point.

```{figure} ../_static/figures/load_transfer.png
:alt: per-corner vertical loads during braking and cornering
:width: 80%

Braking transfers load forward, cornering transfers it across each axle, and the total
stays at $mg$ throughout. The per-corner loads computed here are the same ones fed to
the tire model, so the chassis and the tire always see a consistent state.
```

Three modes are available: `"measured"` (default — use the logged $a_x, a_y$, which
every vehicle-level dataset has, and which avoids an implicit algebraic loop),
`"static"`, and `"iterate"` (Picard iterations for the loop). Using the measured
acceleration for the *load transfer* is not target leakage: load transfer is a
kinematic consequence of acceleration, and the tire model never sees $a_x, a_y$.

## One shared tire model

The same `TireNet` instance is evaluated on all four corners:

```python
FourWheelVehicle(tire, vp, share_tire=True)      # default
FourWheelVehicle(tire, vp, share_tire=False)     # ablation: 4 independent nets
```

Why sharing is the default:

1. **Statistical.** Each vehicle-level sample becomes four constraints on one
   constitutive law. Evidence about the friction limit — which only one corner visits
   at a time — is pooled instead of split four ways.
2. **Identifiability.** Four independent networks can absorb *chassis* errors (a wrong
   $I_z$, a wrong $h_{cg}$) into per-wheel weights, producing a model that fits the
   training laps and is wrong about the tire. With one shared law and exact mechanics,
   a chassis error shows up as a fit error instead of being quietly absorbed.
3. **Physical.** The four tires are the same product. The differences between corners
   are differences in $F_{z}$, $\alpha$, $\kappa$ and $\delta$ — all already inputs.

Per-corner differences can still be modelled explicitly through a small learned corner
embedding (`corner_embedding=True`), which is **off by default** because it is the
loophole through which per-wheel behaviour returns.

## IMU-only identification

The tire model is identified without ever measuring a tire force. The observable
signals are $a_x$, $a_y$, $\dot r$, wheel speeds, steering angle and velocity;
gradients flow backwards through {eq}`eq-newton-euler`, {eq}`eq-wheel-to-body` and the
slip definitions into the shared tire model.

The IMU convention is applied in exactly one place
({py:func}`tire_nn.training.losses.imu_accelerations`), because dropping or
double-counting the centripetal term $r v_y$ is a classic silent bug in vehicle-level
identification:

```{math}
a_x^{\text{IMU}} = \dot v_x - r v_y, \qquad a_y^{\text{IMU}} = \dot v_y + r v_x .
```

The loss weights $a_x$, $a_y$ and $\dot r$ against **fixed physical scales**
(≈ 1 g, and a typical yaw acceleration), so the weights are dimensionless and the unit
choice does not silently decide the trade-off.

## Experiment 3 result (smoke run)

Trained on IMU signals only at $\mu \in \{1.0, 0.85\}$, tested on an unseen
$\mu = 0.65$ and an unseen tire set:

| model | params | val $a_y$ RMSE | unseen-$\mu$ $a_y$ RMSE |
|---|---|---|---|
| fitted Magic Formula (no vehicle data) | 0 | 1.04 | 1.59 |
| shared TireNet | 1 323 | 0.59 | 1.84 |
| per-wheel nets (ablation) | 5 292 | 0.62 | 1.92 |
| shared ParameterNet | 1 548 | **0.38** | **1.12** |

The per-wheel ablation uses four times the parameters and is *worse* on both, and the
parameter network — the most constrained model — generalises best to the friction level
it never saw.

## Guaranteed by tests

`tests/test_four_wheel.py`:

- `test_all_four_wheels_share_the_same_parameter_objects` — identity, not value
- `test_shared_model_has_a_quarter_of_the_per_wheel_parameter_count`
- `test_aggregation_matches_a_hand_computed_reference`
- `test_corner_positions_are_the_exact_geometry`
- `test_load_transfer_conserves_total_vertical_load`
- `test_vehicle_model_has_no_learnable_chassis_parameters`
- `test_gradients_reach_the_tire_from_vehicle_level_supervision_only`

## References

Vehicle dynamics and load transfer follow {cite}`rajamani2012vehicle`; learning vehicle
dynamics for racing from vehicle-level logs is the setting of
{cite}`chrosniak2024deepdynamics` and {cite}`kabzan2019lbmpc`.
