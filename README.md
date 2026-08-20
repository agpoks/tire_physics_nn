# tire_physics_nn

**Physics-encoded neural tire models for autonomous racing and motorsport.**

A modular research framework where the physics lives in the *architecture*, not in the loss:
slip kinematics are computed analytically, odd symmetry and zero-slip-zero-force hold exactly
for any weights, the friction ellipse is enforced by a differentiable radial projection
(never a penalty), relaxation is a first-order ODE in travelled distance, and one shared
`TireNet` serves all four corners inside exact Newton–Euler vehicle equations.

Read **[PLAN.md](PLAN.md)** first — it documents the physical justification for every
architectural decision, the module interfaces, and the canonical dataset format.

## Status

Milestone M0 complete (inspection, plan, skeleton). See PLAN.md §8 for the milestone list.

## Install

```bash
python -m pip install -e .            # core
python -m pip install -e ".[ode]"     # + torchdiffeq (optional Neural-ODE path)
python -m pip install -e ".[dev]"     # + pytest, jupyter, pyarrow
```

Python 3.10+, PyTorch 2.2+. `torchdiffeq` and `nnodely` are optional; the framework runs
fully without them (fixed-step Euler/RK4 is the default integrator).

## Layout

```
tire_nn/
  layers/      symmetry, friction_envelope, bounded_parameters, slip_kinematics
  models/      mlp / encoded / parameter / residual / relaxation / thermo-graining / four-wheel
  physics/     pacejka, brush, vehicle_dynamics, thermal, wear   (no learnable parameters)
  data/        dataset adapters -> one canonical schema (PLAN.md §4)
  training/    losses, trainer, metrics
  evaluation/  plots, extrapolation, physical_consistency
experiments/   train_direct_force / train_relaxation / train_vehicle_supervised / train_graining
scripts/       download_<dataset>.py, fit_magic_formula.py
notebooks/     01 encoded tire force, 02 relaxation+graining cell, 03 four-wheel supervision
tests/         invariant tests (symmetry, envelope, bounds, monotonicity, shared weights)
```

## Datasets

No dataset is committed and nothing large downloads automatically. Each source has a
`scripts/download_<name>.py` that fetches a small subset or prints exact manual steps.
Every dataset is labelled **real measurement / simulated / game telemetry / synthetic** —
see `data/README.md` and PLAN.md §4.4. All experiments also run on in-repo synthetic data.

## Caveat

The graining/wear demonstrator (Experiment 4) uses synthetic, weakly supervised states.
It illustrates the model structure — it is **not** validated real motorsport graining.
