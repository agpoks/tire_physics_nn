# `FourWheelVehicle` — one shared tire, exact mechanics

```{figure} ../_static/diagrams/four_wheel.png
:alt: four-wheel vehicle architecture
:width: 100%
```

The equations, the load-transfer model and the observability discussion are in
[From four tires to a vehicle](../physics/vehicle). This page is about the two modelling
decisions.

## Decision 1: one shared tire model, not four

```python
FourWheelVehicle(tire, vp, share_tire=True)      # default
FourWheelVehicle(tire, vp, share_tire=False)     # ablation: 4 independent nets
```

**Statistical.** Each vehicle-level sample becomes four constraints on one constitutive
law. Evidence about the friction limit — which only one corner visits at a time — is
pooled rather than split four ways.

**Identifiability.** Four independent networks can absorb *chassis* errors (a wrong
$I_z$, a wrong $h_{cg}$) into per-wheel weights, producing a model that fits the training
laps and is wrong about the tire.

**Physical.** The four tires are the same product. The differences between corners are
differences in $F_z$, $\alpha$, $\kappa$ and $\delta$ — all already inputs.

Per-corner differences can still be modelled explicitly through a small learned
embedding (`corner_embedding=True`), off by default because it is the loophole through
which per-wheel behaviour returns.

Measured: the per-wheel ablation uses 4× the parameters and generalises worse to an
unseen friction level ([benchmarks](../comparison/benchmarks)).

## Decision 2: nothing in the chassis is learnable

```python
tire_params = {id(p) for p in veh.tire.parameters()}
assert all(id(p) in tire_params for p in veh.parameters())   # tested
```

Newton–Euler with known geometry is exact. A learned correction to it would let a wrong
inertia be absorbed silently, and the tire model — the thing you were identifying — takes
the blame or the credit.

## IMU-only training

```python
out = vehicle(vx, vy, r, delta, omega, ax_meas=ax, ay_meas=ay)
loss = vehicle_loss((out["ax"], out["ay"], out["r_dot"]), (ax, ay, r_dot))
```

Gradients flow backwards through the rigid-body equations and the slip definitions into
the shared tire model. `ax_meas`/`ay_meas` are used **only** for the quasi-static load
transfer — a kinematic consequence of acceleration — never as an input to the tire model.

The IMU convention is applied in exactly one place
({py:func}`tire_nn.training.losses.imu_accelerations`), because dropping or
double-counting the centripetal term $r v_y$ is a classic silent bug:

```{math}
a_x^{\text{IMU}} = \dot v_x - r v_y, \qquad a_y^{\text{IMU}} = \dot v_y + r v_x
```

The loss weights $a_x$, $a_y$ and $\dot r$ against **fixed physical scales** (≈1 g and a
typical yaw acceleration), so the weights are dimensionless and the unit choice does not
silently decide the trade-off.

## What comes out

After training on IMU signals only, the recovered constitutive model passes the full
consistency audit (all violations zero) — but it is only trustworthy over the slip range
the vehicle actually visited. Notebook 3 shows the recovered $F_y(\alpha)$ curve against
the ground-truth tire, and the divergence beyond the visited range is visible and
expected. Warm-start from rig data when you have it
(`evaluation.pretrain_with_tire_data: true`).
