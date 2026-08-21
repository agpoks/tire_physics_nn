# Conventions: frames, signs and slip

Everything in this framework is SI and uses one sign convention, fixed at the dataset
adapter boundary. Getting this wrong is the single most expensive class of bug in tire
work, because a sign error produces a model that trains beautifully and steers the wrong
way.

## Slip definitions

Slip is a *definition*, not an uncertain physical effect. Given wheel angular velocity
$\omega$, effective rolling radius $R_e$, and the contact-point velocity $(v_x, v_y)$:

```{math}
:label: eq-slip-ratio
\kappa = \frac{R_e\,\omega - v_x}{\max(|v_x|,\, v_\varepsilon)}
```

```{math}
:label: eq-slip-angle
\alpha = \arctan\!\left(\frac{v_y}{\max(|v_x|,\, v_\varepsilon)}\right) - \delta
```

The contact-patch slip velocity, which the thermal model needs because dissipation is
$-F \cdot v_{slip}$:

```{math}
:label: eq-slip-velocity
v_{sx} = R_e\,\omega - v_{x,w}, \qquad v_{sy} = v_{y,w}
```

with $(v_{x,w}, v_{y,w})$ the contact velocity rotated by the steering angle $\delta$.

## The two choices that must be documented

`v_eps` — low-speed regularisation
: Default $0.5\,\mathrm{m/s}$. Both definitions divide by speed and are singular at
  standstill. Clamping $|v_x|$ (rather than clamping the resulting slip) keeps the
  expression smooth, and makes the standstill limit physically right: combined with
  the relaxation dynamics of [Transient](transient), the force *freezes* as the wheel
  stops, which is what a non-rolling tire does.

sign convention — **SAE**
: A positive slip angle gives a **negative** lateral force; a positive slip ratio is
  **driving**. This choice fixes the sign in the encoded parameterisation
  $F_y = -\alpha\,g_y(\cdot)$ used throughout the models.

Other conventions exist and are not wrong, just different. ISO flips the lateral sign;
CommonRoad (and the `scuderia_gymnasium` simulator) uses braking-positive slip. Every
dataset adapter declares its source convention and converts exactly once, through
{py:func}`tire_nn.data.common.flip_sign_convention`.

## Reference frames

| frame | $x$ | used for |
|---|---|---|
| wheel | along the wheel's heading (rotated by $\delta$) | the tire force law itself |
| body | along the vehicle's longitudinal axis | Newton–Euler aggregation |
| inertial | fixed to the ground | trajectory integration |

The wheel$\to$body rotation is applied in exactly one place,
{py:func}`tire_nn.physics.vehicle_dynamics.wheel_to_body`.

## Code

```python
from tire_nn.layers import SlipKinematics, slip_angle, slip_ratio, slip_velocity

sk = SlipKinematics(R_e=0.32, v_eps=0.5)
alpha, kappa = sk(vx, vy, omega, delta)
assert list(sk.parameters()) == []      # nothing here is learned
```

`SlipKinematics` is an `nn.Module` with **zero parameters** deliberately: the absence of
learning is then visible in `model.named_parameters()`, and the module keeps the
gradient path open from raw wheel speeds and steering angles through to the tire model,
which is what makes IMU-only training possible ([Vehicle](vehicle.md)).

## Validation

`validate_schema` enforces the contract on every dataset, including a check that
`alpha` is not accidentally in degrees:

```python
>>> df["alpha"] = np.rad2deg(df["alpha"])
>>> validate_schema(df, ("Fy",))
ValueError: alpha out of range — is it in degrees? adapters must convert to rad
```

## Tests

`tests/test_layers.py::test_slip_ratio_zero_at_free_rolling_and_signs`,
`::test_slip_is_finite_at_standstill`,
`::test_slip_velocity_matches_kinematic_definition`,
`::test_slip_kinematics_has_no_learnable_parameters`.

## References

{cite}`pacejka2012tire` for the slip definitions and rolling-radius convention;
{cite}`rajamani2012vehicle` for the vehicle-level derivation.
