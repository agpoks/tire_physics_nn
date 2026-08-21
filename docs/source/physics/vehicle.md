# From four tires to a vehicle

Level 6: the tire model has to explain measurements taken at the vehicle, not at the
contact patch.

## Rigid-body equations

```{math}
:label: eq-newton-euler
\begin{aligned}
m\left(\dot v_x - r v_y\right) &= \sum_{i} F_{x,i}^{\text{body}} - F_{\text{drag}} \\
m\left(\dot v_y + r v_x\right) &= \sum_{i} F_{y,i}^{\text{body}} \\
I_z\, \dot r &= \sum_{i} \left(x_i F_{y,i}^{\text{body}} - y_i F_{x,i}^{\text{body}}\right)
\end{aligned}
```

with the wheel$\to$body rotation by the per-corner steering angle

```{math}
:label: eq-wheel-to-body
F_{x,i}^{\text{body}} = F_{x,i}\cos\delta_i - F_{y,i}\sin\delta_i, \qquad
F_{y,i}^{\text{body}} = F_{x,i}\sin\delta_i + F_{y,i}\cos\delta_i
```

and the exact corner geometry (wheel order `FL, FR, RL, RR` project-wide)

```{math}
:label: eq-corner-geometry
x_i = \big(l_f,\; l_f,\; -l_r,\; -l_r\big), \qquad
y_i = \Big(\tfrac{t_f}{2},\; -\tfrac{t_f}{2},\; \tfrac{t_r}{2},\; -\tfrac{t_r}{2}\Big)
```

Each corner's contact velocity follows from rigid-body kinematics:

```{math}
:label: eq-corner-velocity
\mathbf{v}_i = \mathbf{v}_{\text{CoG}} + \boldsymbol\omega \times \mathbf{r}_i
\;\Longrightarrow\;
v_{x,i} = v_x - r\, y_i,\qquad v_{y,i} = v_y + r\, x_i
```

## Model choices at this level

| model | states | captures | misses |
|---|---|---|---|
| **kinematic bicycle** | $x, y, \psi$ | low-speed geometry | any tire force at all |
| **dynamic bicycle** | $+\,v_y, r$ | axle-level cornering | load transfer, per-wheel slip, individual traction |
| **four-wheel** (used here) | $+\,\omega_i$ | per-corner load and slip, torque vectoring, split-$\mu$ | suspension kinematics, compliance, roll dynamics |
| **multi-body** | many | everything | needs a parameter set nobody has for a research car |

The four-wheel model is the smallest one in which *load transfer* and *per-corner slip*
both exist, which is what makes the tire identifiable from vehicle data.

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

The transfers cancel in the sum by construction, so $\sum_i F_{z,i} = mg$ **exactly**, at
every operating point.

```{figure} ../_static/figures/load_transfer.png
:alt: per-corner vertical loads during braking and cornering
:width: 78%

Braking transfers load forward, cornering transfers it across each axle, and the total
stays at $mg$ throughout. Combined with the [load sensitivity](steady-state) of $\mu$,
this is what makes a heavily loaded outer tire deliver less than proportionally more
grip — the origin of the understeer/oversteer balance.
```

Three modes are available: `"measured"` (default — use the logged $a_x, a_y$, avoiding
an implicit algebraic loop), `"static"`, and `"iterate"`. Using measured acceleration
for the *load transfer* is not target leakage: load transfer is a kinematic consequence
of acceleration, and the tire model never sees $a_x, a_y$.

## Why nothing here may be learned

Equations {eq}`eq-newton-euler`–{eq}`eq-corner-velocity` are exact given the geometry.
If a "learned correction" is added to them, then a wrong $I_z$, a wrong $h_{cg}$ or a
mis-measured track width can be absorbed by that correction — and the tire model, which
is what you were trying to identify, silently takes the blame or the credit. Keeping the
mechanics exact is what makes the tire *identifiable*.

## What is observable from vehicle data

With supervision only on $a_x$, $a_y$, $\dot r$ you observe **sums and moments** of tire
forces, not individual forces. Consequences worth knowing before trusting a result:

- Only the slip range the vehicle actually visited is identified. A car that never
  exceeded 4° of slip angle tells you nothing about the peak.
- Front/rear force split is identifiable through the yaw moment; left/right split only
  through asymmetric manoeuvres.
- Absolute $\mu$ is only identifiable if the vehicle got near the limit at some point.

This is why [Experiment 3](../guides/experiments) supports warm-starting from rig data:
rig data for the shape, vehicle data for this particular tire on this particular
surface.

## Tests

`tests/test_four_wheel.py`: `test_aggregation_matches_a_hand_computed_reference`,
`test_corner_positions_are_the_exact_geometry`,
`test_load_transfer_conserves_total_vertical_load`,
`test_vehicle_model_has_no_learnable_chassis_parameters`.

## References

{cite}`rajamani2012vehicle` for the vehicle dynamics; {cite}`chrosniak2024deepdynamics`
and {cite}`kabzan2019lbmpc` for learning vehicle dynamics from racing logs.
