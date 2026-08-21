# Extending the framework

## Adding a tire model

Subclass {py:class}`tire_nn.models.base.BaseTireModel` and implement one method:

```python
from tire_nn.models.base import BaseTireModel, ContextEncoder
from tire_nn.types import TireForces

class MyTireModel(BaseTireModel):
    encodes = ("slip_kinematics", "odd_symmetry")     # documentation, shown by describe()

    def __init__(self, context_keys=(), n_tires=0, hidden=(32, 32)):
        super().__init__()
        self.context = ContextEncoder(context_keys, n_tires)
        ...

    def forward(self, alpha, kappa, Fz, context=None) -> TireForces:
        c = self.context(context, alpha)
        ...
        return TireForces(Fx=Fx, Fy=Fy, params={"mu_y": mu_y})
```

Then register it so configs can name it:

```python
# tire_nn/models/registry.py
MODEL_REGISTRY["mine"] = MyTireModel
```

**Add the invariant tests you claim.** The convention in this project is that a claimed
guarantee is tested with adversarially randomised weights (`tests/conftest.py::randomize_`),
because a property that only holds after training is a penalty in disguise:

```python
def test_my_model_is_odd_for_random_weights():
    model = randomize_(MyTireModel(), std=3.0)
    out, out_m = model(a, k, Fz), model(-a, -k, Fz)
    assert torch.allclose(out.Fy, -out_m.Fy, atol=1e-6)
```

## Adding a dataset adapter

Return a `pandas.DataFrame` in the canonical schema ([Datasets](datasets)), in SI units
and the SAE sign convention:

```python
from tire_nn.data.adapters import ColumnSpec, finalise, map_columns, read_any

MYRIG_COLUMNS = {
    "alpha": ColumnSpec(("slip_angle_deg", "SA"), DEG2RAD, required=True),
    "Fz":    ColumnSpec(("Fz", "vertical_load"), 1.0, required=True),
    "Fy":    ColumnSpec(("Fy", "lateral_force"), 1.0),
    "p":     ColumnSpec(("pressure",), KPA2PA),
}

def load_myrig(root, columns=None, flip_signs=False):
    raw = read_any(Path(root) / "myrig", ("*.csv",), "MyRig", INSTRUCTIONS)
    df = map_columns(raw, columns or MYRIG_COLUMNS)
    if flip_signs:
        df = flip_sign_convention(df)
    return finalise(df, "myrig", ("Fy",))
```

Then add it to `tire_nn.data.adapters.load`. **Document the source's sign convention in
the docstring** — that one sentence prevents the most expensive class of bug in this
domain.

## Adding a physical law

Put it in `tire_nn/physics/`, with **no learnable parameters** — parameters are passed
in. That is what lets the same equations serve the analytical baseline, the parameter
network and the `scipy` fitter.

```python
def my_law(alpha, kappa, Fz, param_a, param_b) -> tuple[Tensor, Tensor]:
    """One-line summary.

    Assumptions, validity range, and what it gets wrong. Cite the source.
    """
```

`physics/` must never import from `models/`. `models/` must never re-implement an
equation that exists in `physics/`. This is what keeps the ablations honest: every rung
evaluates the same analytical core.

## Adding a guarantee

If you want a new structural property, the [integration patterns](../methods/integration)
page lists the techniques. The checklist:

1. Can it be written as algebra on the output? → structural map (pattern 5).
2. Is it a sign or a bound on a rate? → `softplus` / `exp` (pattern 6).
3. Is it an invariant interval? → write the ODE so both boundaries are absorbing, **and
   check the discrete update preserves it** — the continuous invariance of $g \in [0,1]$
   does not survive a large explicit Euler step, which is why the graining update uses
   the exact exponential form.
4. None of the above? → it belongs in the loss, and you should report how much it is
   actually violated.
