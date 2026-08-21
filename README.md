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

**Structural guarantees.** Four *untrained* models (default initialisation, `set_seed(0)`)
audited on a grid wider than any training range — violations are load-normalised, the
audit is `tire_nn.evaluation.audit`:

| model | force at zero slip | odd-symmetry violation | friction-envelope violation |
|---|---|---|---|
| fitted Magic Formula | 0 | 0 | 0 |
| plain MLP | 0.22 | 0.39 | 0 * |
| symmetry-encoded (P2) | **0** | **0** | 7.76 |
| symmetry + hard envelope (P2+P3) | **0** | **0** | **0** |

\* against a deliberately generous μ=1.5 reference ellipse. Symmetry alone is the worst
case for the envelope — a correct shape with an unconstrained magnitude.

**Experiment 1**, full budget (6 000 synthetic samples, 300 epochs, identical
optimiser/seed/budget per rung). RMSE in newtons on a tire loaded to ~1.4 kN;
`envelope (μ=1.1)` is measured against the *true* friction limit of the data, i.e. what
a controller would plan against:

| model | params | test $F_y$ RMSE | extrap. RMSE | zero-slip force | symmetry | envelope (μ=1.1) |
|---|---|---|---|---|---|---|
| Magic Formula (fitted) | 0 | 17.71 | 17.87 | 0 | 0 | 0 |
| plain MLP | 4 546 | 18.61 | 18.32 | 0.016 | 0.076 | 0.065 |
| MLP + friction **penalty** | 4 546 | 18.14 | 18.10 | 0.014 | 0.100 | 0.057 |
| symmetry-encoded | 1 250 | 17.61 | 17.60 | **0** | **0** | 0.426 |
| symmetry + hard envelope | 1 254 | 18.55 | 19.21 | **0** | **0** | **0** |
| ParameterNet + Magic Formula | 1 483 | **17.39** | **17.49** | **0** | **0** | **0** |
| residual grey-box | 1 254 | 17.47 | 17.48 | **0** | **0** | **0** |

Read it carefully, because it does *not* say the priors make the model much more
accurate:

1. **On clean, plentiful, in-distribution data, accuracy is close to a wash** — 17.4 to
   18.6 N across every rung. What separates them is the guarantee column, not the RMSE
   column. Anyone claiming a large accuracy win from physics encoding on data like this
   is probably comparing against an under-trained baseline.
2. **The friction penalty does not deliver the constraint.** It cut the violation of the
   true limit by 13 % (0.065 → 0.057) and made the *symmetry* violation worse
   (0.076 → 0.100) — it is a soft trade-off, not a constraint. The structural version is
   exactly zero, for every weight vector.
3. **Data efficiency is where the priors win decisively**, and it is the only place the
   accuracy gap is large — see the learning curve below.

Test $F_y$ RMSE [N] vs training-set size — the regime a real tire programme lives in,
because rig time is expensive:

| training samples | Magic Formula (fitted) | plain MLP | symmetry | residual | encoded | ParameterNet |
|---|---|---|---|---|---|---|
| 210 | 196.9 | 95.2 | 75.3 | 25.2 | 20.6 | **17.4** |
| 570 | 17.6 | 20.3 | 42.8 | 19.0 | 19.0 | **17.3** |
| 1 547 | 17.6 | 18.4 | 26.8 | 18.1 | 18.8 | **17.3** |
| 4 200 | 17.7 | 18.1 | 17.7 | 17.5 | 18.6 | **17.4** |

Two things stand out:

* **The encoded models are near their final accuracy from 210 samples.** The plain MLP
  needs roughly an order of magnitude more data to catch up.
* **The analytical baseline is not automatically data-efficient.** The `scipy`
  Magic-Formula fit is the *worst* model at 210 samples (196.9 N) — five free parameters
  on sparse, noisy data is an ill-conditioned fit — and only recovers by 570. The
  ParameterNet contains the same Magic Formula but predicts its coefficients through
  bounded transforms, and is stable at every size. The bounded parameterisation is doing
  real regularisation work, not just bookkeeping.

(The `mlp_penalty` column is omitted: a bug meant the learning-curve loop dropped the
penalty, making that rung an exact duplicate of `mlp`. Fixed in
`experiments/train_direct_force.py`; the main table above is unaffected, as it takes a
separate code path.)

**Experiment 2** measures the **rise-distance ratio** between 30 m/s and 10 m/s: ≈1 means
the transient is parameterised by travelled distance (physically correct), ≈3 means a
fixed *time* constant was learned. Across two runs the encoded relaxation cells score
0.92–1.25, the Neural ODE 3.00 both times, and the GRU 2.25 and 0.53 — i.e. the
unstructured baselines' speed law is a property of the run, not of the tire. The encoded
cell also has the lowest rollout error (110 N vs the GRU's 203 N) and recovers
$\sigma_y \approx 0.25$ m against a true 0.30 m.

All numbers above come from synthetic, Magic-Formula-generated data with clean Gaussian
noise on a dense slip grid — kinder than any real rig, and structurally matched to
`ParameterTireNet`. Re-run on real measurements before drawing conclusions about
relative accuracy.

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
