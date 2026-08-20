# tire_physics_nn

**Physics-encoded neural tire models for autonomous racing and motorsport.**

A modular research framework where the physics lives in the *architecture*, not in the loss:
slip kinematics are computed analytically, odd symmetry and zero-slip-zero-force hold exactly
for any weights, the friction ellipse is enforced by a differentiable radial projection
(never a penalty), relaxation is a first-order ODE in travelled distance, and one shared
`TireNet` serves all four corners inside exact Newton–Euler vehicle equations.

**Documentation:** the `docs/` tree builds a ReadTheDocs site with a full theory
tutorial — every prior derived, justified, plotted and linked to the test that
guarantees it — plus a quickstart, the experiment guide, the dataset reference and an
auto-generated API reference:

```bash
cd docs && make figures && make html && open _build/html/index.html
```

Read **[PLAN.md](PLAN.md)** first — it documents the physical justification for every
architectural decision, the module interfaces, and the canonical dataset format.

## Status

All milestones M0–M7 complete: physics core, constrained layers, the model ablation
ladder, dataset adapters, four experiments, the ReadTheDocs theory tutorial and three
executed notebooks. 152 tests pass. See PLAN.md §8.

## What the priors buy

Four *untrained* models (default initialisation, `set_seed(0)`) audited on a grid wider
than any training range — violations are load-normalised, the audit is
`tire_nn.evaluation.audit`:

| model | force at zero slip | odd-symmetry violation | friction-envelope violation |
|---|---|---|---|
| fitted Magic Formula | 0 | 0 | 0 |
| plain MLP | 0.22 | 0.39 | 0 * |
| symmetry-encoded (P2) | **0** | **0** | 7.76 |
| symmetry + hard envelope (P2+P3) | **0** | **0** | **0** |

\* the untrained MLP happens to output small forces, so it does not exceed a *generous*
reference ellipse; after training on saturating data it does (0.26 in Experiment 1).
Symmetry alone is the worst case for the envelope — a correct shape with an
unconstrained magnitude.

Experiment 1 on synthetic Magic-Formula data, all rungs trained identically:

| model | params | test $F_y$ RMSE [N] | zero-slip force | envelope violation |
|---|---|---|---|---|
| Magic Formula (fitted) | 0 | 16.2 | 0 | 0 |
| plain MLP | 4 546 | 204.9 | 0.081 | 0.258 |
| MLP + friction **penalty** | 4 546 | 208.9 | 0.069 | 0.252 |
| symmetry-encoded | 1 250 | 120.1 | **0** | 0.431 |
| symmetry + hard envelope | 1 254 | 28.2 | **0** | **0** |
| ParameterNet + Magic Formula | 1 483 | **15.7** | **0** | **0** |

The soft penalty reduces the envelope violation by 2 %; the structural projection
removes it. The encoded models use a quarter of the parameters and are seven times more
accurate. (The data is Magic-Formula generated, so the ParameterNet result is expected
to be strong — re-run on real rig data before comparing accuracy.)

Experiment 2 measures the **rise-distance ratio** between 30 m/s and 10 m/s: ≈1 means
the transient is parameterised by travelled distance (physically correct), ≈3 means a
fixed time constant was learned. GRU 2.25, Neural ODE 3.00, encoded relaxation cells
0.92–1.25.

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
docs/          Sphinx/ReadTheDocs theory tutorial + API reference
papers/        references.bib
```

## Datasets

No dataset is committed and nothing large downloads automatically. Each source has a
`scripts/download_<name>.py` that fetches a small subset or prints exact manual steps.
Every dataset is labelled **real measurement / simulated / game telemetry / synthetic** —
see `data/README.md` and PLAN.md §4.4. All experiments also run on in-repo synthetic data.

## Caveat

The graining/wear demonstrator (Experiment 4) uses synthetic, weakly supervised states.
It illustrates the model structure — it is **not** validated real motorsport graining.
