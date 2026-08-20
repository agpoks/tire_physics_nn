"""Extrapolation protocol (PLAN.md §5).

Interpolation error on a random split is the metric that makes every model look
equivalent. The questions this project asks — does the encoded structure survive
outside the data? — are only answered by holding out whole regions of the operating
envelope.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

__all__ = ["slip_range_holdout", "load_range_holdout", "learning_curve_sizes", "evaluate_on"]


def slip_range_holdout(df: pd.DataFrame, alpha_train_max: float = 0.12, kappa_train_max: float = 0.12):
    """Train on small slip only; test on the saturated region the controller lives in."""
    inner = (df["alpha"].abs() <= alpha_train_max) & (df["kappa"].abs() <= kappa_train_max)
    return df[inner].reset_index(drop=True), df[~inner].reset_index(drop=True)


def load_range_holdout(df: pd.DataFrame, quantile: float = 0.75):
    """Train on low/medium load, test on high load — probes load-sensitivity transfer."""
    cut = df["Fz"].quantile(quantile)
    return df[df["Fz"] <= cut].reset_index(drop=True), df[df["Fz"] > cut].reset_index(drop=True)


def learning_curve_sizes(n: int, points: int = 5, min_frac: float = 0.05) -> list[int]:
    """Log-spaced training-set sizes for the 'performance vs training-data amount' study."""
    return sorted({int(round(f * n)) for f in np.geomspace(min_frac, 1.0, points)})


@torch.no_grad()
def evaluate_on(model, df: pd.DataFrame, targets=("Fx", "Fy"), context_keys=(), tire_index=None) -> dict:
    """Metrics of a model on a canonical DataFrame, without building a DataLoader."""
    from tire_nn.training.metrics import regression_metrics

    t = lambda c: torch.as_tensor(df[c].to_numpy(), dtype=torch.float32)
    alpha, kappa, Fz = t("alpha"), t("kappa"), t("Fz")
    ctx = {k: t(k) for k in context_keys if k in df.columns}
    if tire_index is not None:
        ctx["tire_id"] = torch.as_tensor([tire_index[x] for x in df["tire_id"]], dtype=torch.long)
    out = model(alpha, kappa, Fz, ctx or None)
    metrics = {}
    for name in targets:
        if name in df.columns:
            metrics.update(regression_metrics(getattr(out, name), t(name), Fz, prefix=f"{name}_"))
    return metrics
