# Quickstart

## Install

```bash
git clone https://github.com/agpoks/tire_physics_nn.git
cd tire_physics_nn
python -m pip install -e .            # core: torch, numpy, scipy, pandas, matplotlib, sklearn, pyyaml
python -m pip install -e ".[ode]"     # + torchdiffeq (optional Neural-ODE integrator)
python -m pip install -e ".[dev]"     # + pytest, jupyter, pyarrow
```

Python 3.10+, PyTorch 2.2+. `torchdiffeq` and `nnodely` are optional; nothing in
`tire_nn/` imports them at module level.

```bash
python -m pytest -q       # ~2 minutes, 124 tests
```

## Five minutes: train a physics-encoded tire model

```python
import torch
from tire_nn.data import TireDataset, make_synthetic, split_by_group
from tire_nn.models import EncodedTireNet
from tire_nn.training import TrainConfig, train_model, violation_metrics

df = make_synthetic(4000, mu=1.1, seed=0)          # Magic-Formula ground truth + noise
train_df, val_df, test_df = split_by_group(df)

make_ds = lambda d: TireDataset(d, targets=("Fx", "Fy"))
model = EncodedTireNet(hidden=(32, 32))            # P1 + P2 + P3

train_model(model, make_ds(train_df), make_ds(val_df),
            TrainConfig(epochs=100, targets=("Fx", "Fy"), lr=2e-3))
```

Now check that the physics survived training — on a grid **wider** than the training
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
too. Swap `EncodedTireNet` for `MLPTireModel` and the same call reports finite values
for all of them.

## The common interface

Every model, analytical or learned, has the same signature:

```python
out = model(alpha, kappa, Fz, context=None)   # -> TireForces
Fx, Fy = out                                  # TireForces unpacks like a tuple
out.params["mu_y"]                            # physical quantities, when exposed
```

`context` is an open dict with a documented vocabulary — `vx`, `Ts`, `Tc`, `p`,
`gamma`, `mu_est`, `wear`, `graining`, `tire_id`. A model declares which keys it
consumes and ignores the rest.

:::{admonition} Missing context is never zero-filled
:class: tip

The target datasets expose different subsets (VeTyT has camber and pressure but no
temperature; the KIT set has no camber; Q-Motion is pressure-centric). Each declared
key contributes a `(value, present_flag)` pair, and an absent key is filled by a
*learned per-key constant*. A model can therefore tell a real measurement of 0 from a
missing one — silent zero-filling would put fabricated data into the network.
:::

## Picking a model

| You have | Use | Why |
|---|---|---|
| Rig data, want maximum interpretability | `ParameterTireNet` | Outputs $\mu$, $B$, $C$, $E$, $\sigma$ — a tire, not a curve fit |
| Rig data, want maximum flexibility with guarantees | `EncodedTireNet` | Free shape, exact symmetry and envelope |
| A decent analytical model + a mismatch | `ResidualTireNet` | Network only learns the correction |
| Transient / step-test data | `RelaxationTireCell(...)` | Adds $\tau = \sigma/v$ |
| Only vehicle-level logs | `FourWheelVehicle(tire, vp)` | IMU-only identification |
| A stint with temperature effects | `ThermoGrainingTire(...)` | Latent $[T_s, T_c, w, g]$ |
| A baseline to beat | `MagicFormulaTire` + `fit_magic_formula` | Properly fitted, not guessed |

## Composing the priors

They stack, because each wraps the previous one:

```python
from tire_nn.models import ParameterTireNet
from tire_nn.models.relaxation_tire import RelaxationTireCell
from tire_nn.models.thermo_graining_tire import ThermoGrainingTire
from tire_nn.models.four_wheel_vehicle import FourWheelVehicle
from tire_nn.data.vehicle import DEFAULT_VEHICLE

tire = ParameterTireNet(context_keys=("vx", "p"))       # P1+P2+P4
tire = ThermoGrainingTire(tire)                          # + P7
cell = RelaxationTireCell(tire.steady, sigma_from_steady=True)   # + P5
vehicle = FourWheelVehicle(tire, DEFAULT_VEHICLE)        # + P6, one shared tire
```

## Reproducibility

```python
from tire_nn.training import set_seed
set_seed(0)      # torch, numpy, random, cudnn.deterministic
```

Every experiment writes `config.yaml`, `norm.json` (normalisation statistics computed
on the **train split only**), `best.pt`, `history.csv` and a row in `summary.csv`.
Splits are grouped by `sequence_id` so a window never straddles two trajectories, and
extrapolation holdouts are grouped by *condition* (a load level, a pressure, a tire
set) rather than randomly.
