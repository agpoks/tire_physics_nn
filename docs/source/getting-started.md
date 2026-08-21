# Getting started

## Install

```bash
git clone https://github.com/agpoks/tire_physics_nn.git
cd tire_physics_nn
python -m pip install -e .            # core
python -m pip install -e ".[ode]"     # + torchdiffeq (optional Neural-ODE integrator)
python -m pip install -e ".[dev]"     # + pytest, jupyter, pyarrow
```

Python 3.10+, PyTorch 2.2+. `torchdiffeq` and `nnodely` are optional and are never
imported at module level; the fixed-step integrators are the default.

```bash
python -m pytest -q       # ~90 s
```

## Five minutes

```python
import torch
from tire_nn.data import TireDataset, make_synthetic, split_by_group
from tire_nn.models import EncodedTireNet
from tire_nn.training import TrainConfig, train_model

df = make_synthetic(4000, mu=1.1, seed=0)
train_df, val_df, test_df = split_by_group(df)

make_ds = lambda d: TireDataset(d, targets=("Fx", "Fy"))
model = EncodedTireNet(hidden=(32, 32))

train_model(model, make_ds(train_df), make_ds(val_df),
            TrainConfig(epochs=100, targets=("Fx", "Fy"), lr=2e-3))
```

Then check that the physics survived training — on a grid **wider** than the training
range, because that is where violations live:

```python
from tire_nn.evaluation import audit
print(audit(model, n=4096, alpha_max=0.6, kappa_max=0.6))
```

```text
{'sym_violation_x': 0.0, 'sym_violation_y': 0.0, 'zero_slip_force': 0.0,
 'zero_alpha_Fy': 0.0, 'zero_kappa_Fx': 0.0, 'envelope_violation': 1.2e-07,
 'envelope_violation_frac': 0.0, 'dissipativity_violation': 0.0}
```

Every violation is zero to machine precision — and would have been zero before training
too. Swap in `MLPTireModel` and the same call returns finite values for all of them.

## The common interface

Every model, analytical or learned, has one signature:

```python
out = model(alpha, kappa, Fz, context=None)   # -> TireForces
Fx, Fy = out                                  # TireForces unpacks like a tuple
out.params["mu_y"]                            # physical quantities, where exposed
```

`context` is an open dict with a documented vocabulary — `vx`, `Ts`, `Tc`, `p`,
`gamma`, `mu_est`, `wear`, `graining`, `tire_id`. Each model declares which keys it
consumes and ignores the rest.

:::{admonition} Missing context is never zero-filled
:class: tip
Each declared key contributes a `(value, present_flag)` pair, and an absent key is
filled by a *learned* per-key constant, so the model can tell a real measurement of 0
from a missing one. This matters because the target datasets expose different subsets —
VeTyT has camber and pressure but no temperature, the KIT set has no camber.
:::

## Choosing a model

| You have | Use | Why |
|---|---|---|
| Rig data, want interpretability | [`ParameterTireNet`](models/parameter) | outputs $\mu$, $B$, $C$, $E$, $\sigma$ — a tire, not a fit |
| Rig data, want flexibility with guarantees | [`EncodedTireNet`](models/encoded) | free shape, exact symmetry and bound |
| A decent analytical model + a mismatch | [`ResidualTireNet`](models/residual) | network learns only the correction |
| Transient / step-test data | [`RelaxationTireCell`](models/dynamic) | adds $\tau = \sigma/v$ |
| Only vehicle-level logs | [`FourWheelVehicle`](models/vehicle) | IMU-only identification |
| A stint with temperature effects | [`ThermoGrainingTire`](models/dynamic) | latent $[T_s, T_c, w, g]$ |
| A baseline to beat | `MagicFormulaTire` + `fit_magic_formula` | properly fitted, not guessed |

## Reproducibility

```python
from tire_nn.training import set_seed
set_seed(0)      # torch, numpy, random, cudnn.deterministic
```

Every experiment writes `config.yaml`, `norm.json` (statistics computed on the **train
split only**), `best.pt`, `history.csv` and a row in `summary.csv`. Splits are grouped
by `sequence_id` so a window never straddles two trajectories, and extrapolation
holdouts are grouped by *condition* rather than randomly.
