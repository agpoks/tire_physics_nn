# 1. Slip kinematics (P1)

## The physics

Slip is not an uncertain physical effect — it is a *definition*. Given the wheel
angular velocity $\omega$, the effective rolling radius $R_e$ and the contact-point
velocity $(v_x, v_y)$ in the wheel frame:

```{math}
:label: eq-slip-ratio
\kappa = \frac{R_e\,\omega - v_x}{\max(|v_x|,\, v_\varepsilon)}
```

```{math}
:label: eq-slip-angle
\alpha = \arctan\!\left(\frac{v_y}{\max(|v_x|,\, v_\varepsilon)}\right) - \delta
```

and the contact-patch slip velocity, which the thermal model needs:

```{math}
:label: eq-slip-velocity
v_{sx} = R_e\,\omega - v_{x,w}, \qquad v_{sy} = v_{y,w},
```

where $(v_{x,w}, v_{y,w})$ is the contact velocity rotated by the steering angle
$\delta$.

## Why it is computed and not learned

Learning {eq}`eq-slip-ratio` would spend network capacity reproducing an identity, and
it would destroy the interpretability of everything downstream: if $\kappa$ itself is
approximate, a "learned cornering stiffness" is no longer a cornering stiffness.

The only genuine modelling choices here are made explicit rather than hidden:

`v_eps`
: Low-speed regularisation, default $0.5\,\mathrm{m/s}$. Both slip definitions divide
  by speed and are singular at standstill. Clamping $|v_x|$ (rather than clamping the
  resulting slip) keeps the expression smooth and makes the standstill limit
  *physically* right: at $v_x\to 0$ the computed slip goes to a finite value and the
  relaxation dynamics of {doc}`05_relaxation` freeze the force, which is what a
  non-rolling tire does.

sign convention
: This project uses **SAE**: a positive slip angle produces a *negative* lateral
  force, and a positive slip ratio is *driving*. The choice matters because it fixes
  the sign in the symmetric parameterisation of {doc}`02_symmetry`. Every dataset
  adapter declares its source convention and converts once, in
  {py:func}`tire_nn.data.common.flip_sign_convention`. (The `scuderia_gymnasium`
  simulator, for instance, uses the CommonRoad braking-positive convention.)

## Code

{py:mod}`tire_nn.layers.slip_kinematics` provides the functions and a parameter-free
`nn.Module` wrapper:

```python
from tire_nn.layers import SlipKinematics, slip_angle, slip_ratio

sk = SlipKinematics(R_e=0.32, v_eps=0.5)
alpha, kappa = sk(vx, vy, omega, delta)
assert list(sk.parameters()) == []      # nothing here is learned
```

Being an `nn.Module` with zero parameters is deliberate: the absence of learning is
visible in `model.named_parameters()`, and the module keeps the gradient path open
from raw wheel speeds and steering angles all the way to the shared tire model —
which is what makes IMU-only training possible ({doc}`06_four_wheel`).

## Guaranteed by tests

`tests/test_layers.py`:

- `test_slip_kinematics_has_no_learnable_parameters`
- `test_slip_ratio_zero_at_free_rolling_and_signs`
- `test_slip_is_finite_at_standstill`
- `test_slip_velocity_matches_kinematic_definition`

## References

Slip definitions and the rolling-radius convention follow {cite}`pacejka2012tire`;
see also {cite}`rajamani2012vehicle` for the vehicle-level derivation.
