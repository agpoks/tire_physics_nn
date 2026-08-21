<p align="center">
  <img src="docs/source/_static/logo.svg" alt="tire_physics_nn" width="440">
</p>

<h1 align="center">tire_physics_nn</h1>

<p align="center"><strong>Physics-encoded neural tire models for autonomous racing and motorsport.</strong></p>

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch 2.2+](https://img.shields.io/badge/pytorch-2.2%2B-ee4c2c)](https://pytorch.org/)
[![tests](https://img.shields.io/badge/tests-218%20passing-brightgreen)](tests/)
[![docs](https://img.shields.io/badge/docs-sphinx-informational)](docs/)

A research framework for tire models in which the physics lives in the **architecture**
rather than in the loss. Properties such as *no force at zero slip* and *never exceed the
friction ellipse* hold for every weight vector — before training, after training, and
outside the training data.

---

## Contents

- [Why](#why) · [Features](#features) · [Installation](#installation) · [Quick start](#quick-start)
- [Documentation](#documentation) · [Repository layout](#repository-layout) · [Results](#results)
- [Datasets](#datasets) · [Testing](#testing) · [Citation](#citation) · [License](#license)

## Why

A tire model for a racing controller is not judged by its RMSE. It is judged by what it
does at the friction limit, where the controller operates and where data is thinnest.
Three failure modes matter and none of them appear in an interpolation score:

1. **Force at zero slip.** A black-box fit predicts $F_y(\alpha{=}0) \neq 0$. Linearise an
   MPC about straight-line running and that offset becomes a phantom steering command.
2. **Broken symmetry.** $F_y(-\alpha) \neq -F_y(\alpha)$ means the model learned a
   left-turn tire and a different right-turn tire from the same object.
3. **Forces outside the friction ellipse.** An optimiser *searches for the best
   achievable force*, so it finds exactly the region where the model is unphysically
   optimistic and plans a lap the car cannot drive.

Adding these as penalty terms makes them likely. Building them into the computation makes
them certain. That is the whole idea.

## Features

- **Analytical tire laws**, differentiable and parameter-free: linear, brush, Dugoff,
  Magic Formula with load sensitivity and similarity combined slip.
- **Encoded neural models** — exact odd symmetry, exact zero force at zero slip, a hard
  friction envelope via differentiable radial projection, and bounded physical parameters.
- **Dynamic extensions** — relaxation as a first-order ODE in travelled distance;
  optional thermal, wear and graining states with structural irreversibility and bounds.
- **Degradation as a universal differential equation** — known observation structure,
  learned kinetics, identified from **real Formula 1 stint data** where the tyre state
  is never measured.
- **The contact patch as a PDE** — the brush model discretised onto a chain, with the
  pressure distribution learned under an exact load balance (`softmax`), beating the
  parabolic assumption 9× on a tyre whose patch is not parabolic.
- **Condition from imagery** — a monotone wear index with ordered ordinal thresholds,
  recovering continuous wear (r = 0.98) from the three ordered classes that real tyre
  datasets actually provide.
- **Vehicle-level learning** — one shared tire model across four corners inside exact
  Newton–Euler equations, trainable from IMU signals alone.
- **Honest baselines** — plain MLP, MLP + friction penalty, GRU and Neural ODE controls,
  plus a properly fitted analytical model.
- **Dataset adapters** for seven public tire, vehicle and stint datasets, onto canonical
  schemas — including a working, no-API-key path to real F1 timing data.
- **218 tests**, with every structural guarantee checked under adversarially random
  weights.

## Installation

```bash
git clone https://github.com/agpoks/tire_physics_nn.git
cd tire_physics_nn
python -m pip install -e .            # core
python -m pip install -e ".[ode]"     # + torchdiffeq (optional Neural-ODE integrator)
python -m pip install -e ".[dev]"     # + pytest, jupyter, pyarrow
```

Python 3.10+, PyTorch 2.2+. `torchdiffeq` and `nnodely` are optional and never imported
at module level.

## Quick start

```python
import torch
from tire_nn.data import TireDataset, make_synthetic, split_by_group
from tire_nn.models import EncodedTireNet
from tire_nn.training import TrainConfig, train_model
from tire_nn.evaluation import audit

df = make_synthetic(4000, mu=1.1, seed=0)
train_df, val_df, test_df = split_by_group(df)
make_ds = lambda d: TireDataset(d, targets=("Fx", "Fy"))

model = EncodedTireNet(hidden=(32, 32))
train_model(model, make_ds(train_df), make_ds(val_df),
            TrainConfig(epochs=100, targets=("Fx", "Fy"), lr=2e-3))

print(audit(model, n=4096, alpha_max=0.6, kappa_max=0.6))
# {'sym_violation_y': 0.0, 'zero_slip_force': 0.0, 'envelope_violation': 1.2e-07, ...}
```

Every violation is zero to machine precision — and would have been before training too.
Swap in `MLPTireModel` and the same call returns finite values for all of them.

## Documentation

```bash
cd docs && python generate_figures.py && python diagrams.py && make html
open _build/html/index.html
```

The documentation is organised in four parts:

| | |
|---|---|
| **Tire physics** | every physical effect available for modelling — slip conventions, the four steady-state laws, combined slip, transients, thermal/wear/graining, vehicle dynamics — with equations, plots and an honest account of what each gets wrong |
| **Method** | precise definitions of physics-guided, physics-informed and physics-encoded, and six concrete patterns for integrating a physical law into a network |
| **Models** | every model, its equations, the physics it contains and an architecture diagram |
| **Comparisons** | measured dynamic behaviour, benchmark tables and a trade-off matrix |

Three executed notebooks are in [`notebooks/`](notebooks/).

## Repository layout

```
tire_nn/
  physics/     analytical laws, vehicle dynamics, thermal/wear  (no learnable parameters)
  layers/      slip kinematics, symmetry, friction envelope, bounded parameters
  models/      the model catalogue, from black box to fully encoded
  data/        dataset adapters onto one canonical schema
  training/    losses, metrics, deterministic trainer
  evaluation/  consistency audit, extrapolation protocol, plots
configs/       one YAML per experiment
experiments/   five runnable experiments
notebooks/     six executed notebooks
scripts/       dataset download helpers, Magic-Formula fitting
docs/          Sphinx documentation and figure generators
tests/         invariant and contract tests
```

## Results

On synthetic steady-state data with a full training budget, **accuracy is close to a
wash** across every model (17.4–18.6 N test RMSE) — what separates them is the guarantee
columns:

| model | params | test $F_y$ RMSE | zero-slip force | symmetry | envelope violation |
|---|---|---|---|---|---|
| Magic Formula (fitted) | 0 | 17.71 | 0 | 0 | 0 |
| plain MLP | 4 546 | 18.61 | 0.016 | 0.076 | 0.065 |
| MLP + friction **penalty** | 4 546 | 18.14 | 0.014 | 0.100 | 0.057 |
| symmetry + hard envelope | 1 254 | 18.55 | **0** | **0** | **0** |
| ParameterNet + Magic Formula | 1 483 | **17.39** | **0** | **0** | **0** |

The soft penalty cut the violation it targeted by 13 % and made the symmetry violation
*worse*. The structural version is exactly zero at no cost in accuracy.

**Data efficiency is where the priors win decisively** — test $F_y$ RMSE by training-set
size:

| samples | Magic Formula (fitted) | plain MLP | encoded | ParameterNet |
|---|---|---|---|---|
| 210 | 196.9 | 95.2 | 20.6 | **17.4** |
| 4 200 | 17.7 | 18.1 | 18.6 | **17.4** |

**Degradation from real data.** Tyre condition is never measured, but lap time carries
its consequence. A UDE — known observation structure, learned kinetics — is the most
accurate model on synthetic stints (0.258 s vs 0.290 s for a linear baseline) and
recovers the fuel coefficient as 3.48 s against a true 3.50 s, stably across seeds.

But repeating the fit across six seeds on identical data shows the **wear/graining
decomposition is not reliably identifiable** — the lap-time fit is stable (RMSE
0.237–0.241) while the wear correlation with the hidden truth swings between 0.29 and
0.63. One scalar per lap cannot robustly separate two latent states. Trust the total
degradation and the known parameters; do not trust an individual channel without a
second observation.

**Identifiability, and how to fix it.** The wear/graining split is not identifiable from
lap time alone — across seeds the fit is stable while the recovered wear swings between
0.29 and 0.63 correlation with the hidden truth. Adding **one photograph per pit stop**
collapses that spread 28× (to 0.934–0.952) while the lap-time fit barely moves: the
image adds information, not flexibility.

On **real 2023 F1 data** (8 races, 7 223 dry laps, via FastF1) the 45-parameter linear
baseline generalises best (0.82 s against the UDE's 1.10 s) — over a 14-lap stint real
degradation is close to linear — while the black box overfits by more than 3×. Graining
is not identifiable from this data at all, and the UDE reports ~0.

Full tables, including the transient and vehicle-level experiments, are in the
documentation.

> **Caveat.** All published numbers come from synthetic, Magic-Formula-generated data
> with clean Gaussian noise — kinder than any real rig, and structurally matched to
> `ParameterTireNet`. Re-run on real measurements before drawing conclusions about
> relative accuracy.

## Datasets

Every dataset is in one registry — {py:data}`tire_nn.data.registry.DATASETS` — which the
loader, the download helper and the documentation table all read from, so they cannot
drift apart:

```python
from tire_nn.data import registry

registry.describe()                       # all 13, with type, licence and status
registry.describe("kit")                  # url, size, licence, exact steps, target folder
df = registry.get("f1_stints")            # downloads via FastF1 (no API key) and loads
df = registry.get("kit", root="data/raw") # loads if present, else prints the steps
```

`fetch()` is deliberately conservative: it will not click a licence, use your Kaggle
credentials or pull a multi-gigabyte archive on your behalf, and it never silently
substitutes another dataset. **Start with `f1_stints`** — it is the only real dataset
here that needs no manual step.

No dataset is committed and nothing large downloads automatically. Each source has a
`scripts/download_<name>.py` that fetches a small subset or prints exact manual steps.
Adapters exist for the KIT inner-drum dataset, VeTyT bicycle tyre measurements, a TUM
cargo-bicycle set, Deep Dynamics (BayesRace + Indy Autonomous Challenge), RoboRacer and
Q-Motion. **Formula 1 stint data is the exception that needs no manual step** — it comes
from the MIT-licensed FastF1 package with no API key:

```bash
python -m pip install fastf1
python scripts/download_f1_stints.py --seasons 2023 --rounds 1 2 3 4 5 9 13 17
python experiments/train_degradation_ude.py
```

Every dataset is labelled **real measurement / simulated / game telemetry / synthetic**,
and that label travels with it into any results table. Sources whose primary reference
could not be verified are marked `UNVERIFIED` rather than guessed.

## Running your own comparison

```bash
cp experiments/template_experiment.py experiments/my_experiment.py
python experiments/my_experiment.py
```

Three places to edit — the dataset, the models, the budget. Or call the harness directly:

```python
from tire_nn.benchmark import compare, DEFAULT_MODELS

table = compare({**DEFAULT_MODELS, "mine": lambda: MyTireModel()}, data, epochs=150)
```

Every comparison reports the physical-violation columns next to the error columns,
because on clean in-distribution data a good black box and a good encoded model usually
land within noise of each other on RMSE — the difference is in the guarantees. See
[Running your own comparison](docs/source/guides/your-own-experiment.md).

## Testing

```bash
python -m pytest -q        # ~90 s
```

Every structural guarantee is tested with **adversarially randomised weights** — if a
property only held after training, it would be a penalty in disguise.

## Citation

```bibtex
@software{poks_tire_physics_nn,
  author = {Poks, Agnes},
  title  = {tire_physics_nn: physics-encoded neural tire models for autonomous racing},
  year   = {2026},
  url    = {https://github.com/agpoks/tire_physics_nn}
}
```

References for the underlying tire and vehicle theory are in
[`papers/references.bib`](papers/references.bib).

## License

No license file has been added yet — choose one before publishing or sharing the
repository. Note that the KIT dataset is CC BY-NC-SA 4.0, which constrains commercial
use of anything derived from it.
