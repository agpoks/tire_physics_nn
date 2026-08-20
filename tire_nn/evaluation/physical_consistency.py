"""Physical-consistency audit of a trained model (PLAN.md §5, §6).

Reports the violations of the properties the encoded models guarantee. For an
encoded model every number here must be ~1e-7 or below; for the black-box baselines
they are the quantitative statement of what the priors buy.
"""

from __future__ import annotations

import torch

from tire_nn.training.metrics import violation_metrics

__all__ = ["audit", "audit_table"]


@torch.no_grad()
def audit(
    model,
    n: int = 4096,
    alpha_max: float = 0.6,
    kappa_max: float = 0.6,
    Fz_range: tuple[float, float] = (200.0, 3000.0),
    context=None,
    mu_ref: float = 1.5,
    seed: int = 0,
) -> dict:
    """Audit on a *wide* grid — deliberately wider than any training range.

    Violations concentrate outside the training distribution, which is exactly where
    a racing controller operates during a limit-handling event, so auditing only on
    the training support would hide the failure mode this project is about.
    """
    g = torch.Generator().manual_seed(seed)
    alpha = (torch.rand(n, generator=g) * 2 - 1) * alpha_max
    kappa = (torch.rand(n, generator=g) * 2 - 1) * kappa_max
    Fz = torch.rand(n, generator=g) * (Fz_range[1] - Fz_range[0]) + Fz_range[0]
    if context is not None:
        context = {k: v.expand(n) if v.numel() == 1 else v for k, v in context.items()}
    return violation_metrics(model, alpha, kappa, Fz, context, mu_ref=mu_ref)


def audit_table(models: dict, **kwargs):
    """Audit several models and return a tidy ``pandas.DataFrame``."""
    import pandas as pd

    rows = []
    for name, model in models.items():
        row = {"model": name}
        row.update(audit(model, **kwargs))
        rows.append(row)
    return pd.DataFrame(rows)
