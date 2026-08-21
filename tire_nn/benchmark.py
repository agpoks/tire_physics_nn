"""Compare tire models on equal footing — including models you wrote yourself.

The experiment scripts each hard-code their own comparison loop. This module factors
that out so a new model can be dropped in and evaluated against the whole ladder with
one call, on the same data, the same budget, the same seed and the same metrics.

The one rule worth stating: a comparison that reports only accuracy is not a comparison.
Every table this module produces carries the physical-violation columns alongside the
error columns, because on clean in-distribution data the accuracies of a good black box
and a good encoded model are usually within noise of each other — the difference is in
the guarantees, and in what happens off-distribution.

Example::

    from tire_nn.benchmark import compare, DEFAULT_MODELS
    from tire_nn.data import make_synthetic

    df = make_synthetic(4000, seed=0)
    table = compare({**DEFAULT_MODELS, "mine": lambda: MyTireModel()}, df, epochs=100)
    print(table.round(4).to_string(index=False))
"""

from __future__ import annotations

from typing import Callable

import pandas as pd
import torch

from tire_nn.data.common import TireDataset, split_by_group
from tire_nn.evaluation.extrapolation import evaluate_on, load_range_holdout, slip_range_holdout
from tire_nn.evaluation.physical_consistency import audit
from tire_nn.models.registry import build_model
from tire_nn.training.trainer import TrainConfig, set_seed, train_model

__all__ = ["DEFAULT_MODELS", "compare", "compare_curves"]

#: The standard ablation ladder, as factories so each run gets a fresh model.
DEFAULT_MODELS: dict[str, Callable[[], torch.nn.Module]] = {
    "magic_formula": lambda: build_model("magic_formula"),
    "mlp": lambda: build_model("mlp"),
    "symmetry": lambda: build_model("symmetry"),
    "encoded": lambda: build_model("encoded"),
    "parameter": lambda: build_model("parameter"),
    "residual": lambda: build_model("residual"),
}


def compare(
    models: dict[str, Callable[[], torch.nn.Module]],
    data: pd.DataFrame,
    targets: tuple[str, ...] = ("Fx", "Fy"),
    context_keys: tuple[str, ...] = (),
    epochs: int = 150,
    lr: float = 2e-3,
    batch_size: int = 256,
    seed: int = 0,
    extrapolation: str = "slip",
    audit_mu: float = 1.1,
    verbose: bool = True,
) -> pd.DataFrame:
    """Train every model on the same data and return one tidy comparison table.

    Args:
        models: name -> zero-argument factory. Use a factory, not an instance, so each
            model is freshly initialised under the same seed.
        data: a DataFrame in the canonical schema (see :py:mod:`tire_nn.data.common`).
        extrapolation: ``"slip"`` holds out the saturated region, ``"load"`` the high
            loads, ``"none"`` uses the random test split. Interpolation error on a
            random split is the metric that makes every model look equivalent, so the
            default holds out something meaningful.
        audit_mu: the friction limit the envelope violation is measured against. Set it
            to the limit a controller would actually plan against, not a generous one —
            a model can look clean against a loose bound while still promising more grip
            than the tire has.

    Returns:
        One row per model with parameter count, test and extrapolation errors, and the
        physical-violation columns.
    """
    train_df, val_df, test_df = split_by_group(data, seed=seed)
    if extrapolation == "slip":
        _, outer_df = slip_range_holdout(data)
    elif extrapolation == "load":
        _, outer_df = load_range_holdout(data)
    else:
        outer_df = test_df

    tire_index = {name: i for i, name in enumerate(sorted(data["tire_id"].unique()))}
    make_ds = lambda d: TireDataset(d, targets=targets, context_keys=context_keys,
                                    tire_index=tire_index)

    rows = []
    for name, factory in models.items():
        set_seed(seed)
        model = factory()
        trainable = sum(p.numel() for p in model.parameters())
        if trainable:
            train_model(model, make_ds(train_df), make_ds(val_df),
                        TrainConfig(epochs=epochs, lr=lr, batch_size=batch_size,
                                    targets=targets, seed=seed),
                        verbose=False)
        model.eval()

        row = {"model": name, "n_params": trainable}
        row.update({f"test_{k}": v for k, v in
                    evaluate_on(model, test_df, targets, context_keys, tire_index).items()})
        row.update({f"extrap_{k}": v for k, v in
                    evaluate_on(model, outer_df, targets, context_keys, tire_index).items()})
        report = audit(model, n=4096, alpha_max=0.6, kappa_max=0.6, mu_ref=audit_mu)
        row.update({k: report[k] for k in
                    ("zero_slip_force", "sym_violation_y", "envelope_violation",
                     "dissipativity_violation")})
        rows.append(row)
        if verbose:
            print(f"  {name:16s} params {trainable:6d}  "
                  f"test {row.get('test_Fy_rmse', float('nan')):7.2f}  "
                  f"extrap {row.get('extrap_Fy_rmse', float('nan')):7.2f}  "
                  f"violations {max(row['zero_slip_force'], row['sym_violation_y'], row['envelope_violation']):.3g}")

    columns = ["model", "n_params"]
    for suffix in ("test_Fy_rmse", "test_Fx_rmse", "extrap_Fy_rmse"):
        if any(suffix in r for r in rows):
            columns.append(suffix)
    columns += ["zero_slip_force", "sym_violation_y", "envelope_violation",
                "dissipativity_violation"]
    table = pd.DataFrame(rows)
    return table[[c for c in columns if c in table.columns]]


@torch.no_grad()
def compare_curves(models: dict[str, torch.nn.Module], Fz: float = 1000.0,
                   alpha_max: float = 0.35, context=None):
    """Overlay the lateral characteristics of several trained models.

    Returns a matplotlib figure; useful when the numbers are close and the difference is
    in the shape — particularly outside the training range.
    """
    from tire_nn.evaluation import plots

    return plots.plot_lateral_curve(models, Fz=Fz, alpha_max=alpha_max, context=context)
