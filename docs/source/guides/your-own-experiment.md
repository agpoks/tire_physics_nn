# Running your own comparison

Everything in this project is set up so a new model or a new dataset can be compared
against the existing ladder without rewriting the plumbing.

## The fastest path

```bash
cp experiments/template_experiment.py experiments/my_experiment.py
python experiments/my_experiment.py
```

The template has three places to edit, marked `EDIT`: the dataset, the models, and the
budget. Everything else — training, the held-out split, the metrics, the physical audit
and the CSV summary — is shared with the built-in experiments, so your numbers are
directly comparable with the ones in this documentation.

## The benchmark harness

If you would rather work in a notebook or your own script, the same machinery is one
call:

```python
from tire_nn.benchmark import compare, DEFAULT_MODELS
from tire_nn.data import registry

data = registry.get("synthetic_force", n=6000, seed=0)

table = compare(
    {**DEFAULT_MODELS, "mine": lambda: MyTireModel()},
    data,
    targets=("Fx", "Fy"),
    epochs=150,
    extrapolation="slip",     # hold out the saturated region
    audit_mu=1.1,             # measure violations against the true friction limit
)
print(table.round(4).to_string(index=False))
```

Pass **factories**, not instances, so every model is freshly initialised under the same
seed.

Example output — the standard ladder plus one custom model:

```text
        model  n_params  test_Fy_rmse  extrap_Fy_rmse  zero_slip_force  sym_violation_y  envelope_violation
magic_formula         0       74.2775         81.6976           0.0000           0.0000              0.0000
          mlp      4546       18.9002         18.7923           0.0147           0.0823              0.0644
     symmetry      1250       22.2513         19.9777           0.0000           0.0000              0.9097
      encoded      1254       18.5173         19.1177           0.0000           0.0000              0.0000
    parameter      1483       17.4197         17.4878           0.0000           0.0000              0.0000
         mine      2646       18.4324         19.0119           0.0000           0.0000              0.0000
```

:::{important}
**The violation columns are not optional.** On clean, in-distribution data a good black
box and a good encoded model usually land within noise of each other on RMSE — look at
`mlp` and `encoded` above. The difference is in the guarantee columns, and in what
happens outside the data. A comparison that reports only accuracy will mislead you, and
that is not a hypothetical: an early version of this project's own headline table was
wrong because the baseline had been under-trained.
:::

## Choosing what to hold out

`extrapolation` decides what "generalisation" means for your table:

`"slip"`
: train on small slip, test on the saturated region. The right default: it asks whether
  the model works where a controller actually operates.

`"load"`
: train on low and medium load, test on high load. Probes whether load sensitivity
  transferred.

`"none"`
: a random split. Measures interpolation, which is the metric that makes every model
  look equivalent. Use it only when you want that.

For sequence or vehicle data, split by *group* — {py:func}`tire_nn.data.common.split_by_group`
keeps whole trajectories together, because consecutive laps or samples of a stint are
nearly identical and a random row split leaks badly.

## Bringing your own data

Map it onto the canonical schema once, at the boundary
([Datasets](datasets.md#the-canonical-schema)):

```python
from tire_nn.data.adapters import ColumnSpec, finalise, map_columns
import numpy as np, pandas as pd

MY_COLUMNS = {
    "alpha": ColumnSpec(("slip_angle_deg", "SA"), np.pi / 180, required=True),
    "Fz":    ColumnSpec(("Fz", "vertical_load"), 1.0, required=True),
    "Fy":    ColumnSpec(("Fy", "lateral_force"), 1.0),
    "p":     ColumnSpec(("pressure",), 1000.0),          # kPa -> Pa
}

raw = pd.read_csv("my_rig.csv")
df = finalise(map_columns(raw, MY_COLUMNS), "my_rig", ("Fy",))
```

Then it works with everything above. Two things worth getting right first:

1. **The sign convention.** This project uses SAE: positive slip angle gives *negative*
   lateral force. Check one pure-lateral sweep before trusting a whole run, and convert
   with {py:func}`tire_nn.data.common.flip_sign_convention` if needed.
2. **The units.** `validate_schema` will catch degrees-as-radians, but not bar-as-Pa.

## Bringing your own model

Anything with the standard signature works:

```python
from tire_nn.models.base import BaseTireModel
from tire_nn.types import TireForces

class MyTireModel(BaseTireModel):
    def forward(self, alpha, kappa, Fz, context=None) -> TireForces:
        ...
        return TireForces(Fx=Fx, Fy=Fy, params={"mu_y": mu_y})
```

See [Extending the framework](extending.md) for the building blocks — the symmetry
layer, the friction envelope, the bounded parameter transforms — and for the project's
one convention: **whatever physical property you claim, add a test that asserts it under
randomised weights.** A property that only holds after training is a penalty in
disguise.

## Comparing shapes, not just numbers

When the errors are close, the models may still disagree about the tire:

```python
from tire_nn.benchmark import compare_curves
fig = compare_curves({"mine": my_model, "reference": magic_formula}, Fz=1000.0)
```

The plot set in {py:mod}`tire_nn.evaluation.plots` also covers the friction ellipse,
learned $\mu$ against load, residuals and time series.

## Where results go

Experiments write to `results/<name>/`: `config.yaml`, `norm.json` (normalisation
statistics from the **train split only**), `best.pt`, `history.csv`, `summary.csv` and
`plots/`. `results/` is gitignored, so nothing large lands in the repository.
