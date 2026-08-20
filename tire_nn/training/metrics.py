"""Evaluation metrics, including the physical-violation metrics (PLAN.md §5).

The violation metrics are computed for *every* model, encoded or not. For the encoded
models they should come out at machine precision — reporting them anyway is what
makes the claim checkable rather than asserted.
"""

from __future__ import annotations

import torch
from torch import Tensor

from tire_nn.layers.friction_envelope import ellipse_radius

__all__ = ["rmse", "mae", "r2", "regression_metrics", "violation_metrics"]


def rmse(pred: Tensor, target: Tensor) -> float:
    return float(torch.sqrt(torch.mean((pred - target) ** 2)))


def mae(pred: Tensor, target: Tensor) -> float:
    return float(torch.mean(torch.abs(pred - target)))


def r2(pred: Tensor, target: Tensor) -> float:
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - target.mean()) ** 2)
    return float(1.0 - ss_res / torch.clamp(ss_tot, min=1e-12))


def regression_metrics(pred: Tensor, target: Tensor, Fz: Tensor | None = None, prefix: str = "") -> dict:
    out = {
        f"{prefix}rmse": rmse(pred, target),
        f"{prefix}mae": mae(pred, target),
        f"{prefix}r2": r2(pred, target),
    }
    if Fz is not None:
        out[f"{prefix}rmse_norm"] = rmse(pred / Fz, target / Fz)
    return out


@torch.no_grad()
def violation_metrics(
    model,
    alpha: Tensor,
    kappa: Tensor,
    Fz: Tensor,
    context=None,
    mu_ref: float = 1.5,
) -> dict:
    """Physical-consistency metrics, all in load-normalised force units [-].

    ``mu_ref`` is the friction limit the envelope violation is measured against for
    models that do not expose their own ``mu`` (the black-box baselines). It should be
    chosen generously — a violation of a *generous* limit is unambiguous.
    """
    f = model(alpha, kappa, Fz, context)
    fm = model(-alpha, -kappa, Fz, context)
    z = torch.zeros_like(Fz)
    f0 = model(z, z, Fz, context)
    f_a0 = model(z, kappa, Fz, context)
    f_k0 = model(alpha, z, Fz, context)

    mu_x = f.params.get("mu_x", torch.full_like(Fz, mu_ref))
    mu_y = f.params.get("mu_y", torch.full_like(Fz, mu_ref))
    rho = ellipse_radius(f.Fx, f.Fy, mu_x, mu_y, Fz)

    return {
        "sym_violation_x": float(((f.Fx + fm.Fx).abs() / Fz).max()),
        "sym_violation_y": float(((f.Fy + fm.Fy).abs() / Fz).max()),
        "zero_slip_force": float(((f0.Fx.abs() + f0.Fy.abs()) / Fz).max()),
        "zero_alpha_Fy": float((f_a0.Fy.abs() / Fz).max()),
        "zero_kappa_Fx": float((f_k0.Fx.abs() / Fz).max()),
        # Tolerance: ellipse_radius adds a numerical EPS inside the sqrt, so rho sits a
        # few 1e-10 above 1 exactly on the boundary. Anything above 1e-6 is a real violation.
        "envelope_violation": float(torch.clamp(rho - 1.0, min=0.0).max()),
        "envelope_violation_frac": float((rho > 1.0 + 1e-6).float().mean()),
        "dissipativity_violation": float(
            torch.clamp(-(f.Fx * kappa), min=0.0).max() / Fz.max() +
            torch.clamp(f.Fy * alpha, min=0.0).max() / Fz.max()
        ),
    }
