#!/usr/bin/env python3
"""TEMPLATE — copy this to start your own experiment.

    cp experiments/template_experiment.py experiments/my_experiment.py
    python experiments/my_experiment.py

Three places to edit, marked EDIT below:

  1. ``load_data``   — which dataset. Any registry key, or your own DataFrame.
  2. ``MODELS``      — which models to compare. Add your own factory.
  3. ``CONFIG``      — budget, targets, what to hold out.

Everything else — training, the held-out split, the metrics, the physical-violation
audit and the CSV summary — is shared with the built-in experiments, so your results
are directly comparable with the ones in the documentation.

Why the violation columns are always printed: on clean in-distribution data a good black
box and a good encoded model usually land within noise of each other on RMSE. The
difference is in the guarantees, and in what happens outside the data. A comparison that
reports only accuracy will mislead you.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tire_nn.benchmark import DEFAULT_MODELS, compare  # noqa: E402
from tire_nn.data import registry  # noqa: E402
from tire_nn.models import EncodedTireNet, build_model  # noqa: E402
from tire_nn.training import append_summary  # noqa: E402

OUT = ROOT / "results" / "my_experiment"

# --------------------------------------------------------------- 1. EDIT: the data
def load_data() -> pd.DataFrame:
    """Return a DataFrame in the canonical schema (see tire_nn/data/common.py).

    Options:
      * synthetic, always available:
            registry.get("synthetic_force", n=6000, seed=0)
      * a real dataset once downloaded (registry.describe() lists them all):
            registry.get("kit", root="data/raw")
      * your own file, mapped onto the canonical columns:
            from tire_nn.data.adapters import ColumnSpec, map_columns, finalise
    """
    return registry.get("synthetic_force", n=6000, seed=0)


# ------------------------------------------------------------- 2. EDIT: the models
class MyTireModel(EncodedTireNet):
    """Your model. Anything with ``forward(alpha, kappa, Fz, context) -> TireForces``
    works; subclassing an existing one is the quickest start.

    See docs 'Extending the framework' for writing one from scratch, and note the
    project convention: whatever physical property you claim, add a test that asserts it
    under randomised weights.
    """


MODELS = {
    **DEFAULT_MODELS,                       # the standard ladder, for context
    "mine": lambda: MyTireModel(hidden=(48, 48)),
}

# --------------------------------------------------------------- 3. EDIT: the budget
CONFIG = dict(
    targets=("Fx", "Fy"),
    context_keys=(),           # e.g. ("p",) to give the models inflation pressure
    epochs=150,
    lr=2e-3,
    seed=0,
    extrapolation="slip",      # "slip" | "load" | "none" — hold out something meaningful
    audit_mu=1.1,              # the friction limit violations are measured against
)


def main() -> int:
    data = load_data()
    print(f"[data] {len(data)} samples, source={data['source'].iloc[0]}, "
          f"tires={sorted(data['tire_id'].unique())}\n")

    table = compare(MODELS, data, **CONFIG)

    print("\n" + table.round(4).to_string(index=False))
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "comparison.csv", index=False)
    for row in table.to_dict("records"):
        append_summary(OUT / "summary.csv", {"experiment": "my_experiment", **row})
    print(f"\n[out] {OUT / 'comparison.csv'}")

    # Optional: overlay the curves. Numbers that are close can still be different shapes.
    try:
        import matplotlib
        matplotlib.use("Agg")
        from tire_nn.benchmark import compare_curves
        from tire_nn.evaluation import plots

        trained = {name: build_model(name) for name in ("magic_formula",)}
        fig = compare_curves(trained, Fz=float(data["Fz"].median()))
        plots.save(fig, OUT / "curves.png")
        print(f"[out] {OUT / 'curves.png'}")
    except Exception as exc:                                  # noqa: BLE001
        print(f"[plot] skipped: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
