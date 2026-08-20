"""Loss terms. Physical constraints appear here only when they *cannot* be encoded.

Rule (PLAN.md §"Important"): a constraint that the architecture can guarantee is
never expressed as a penalty. The penalties in this module exist for two reasons
only:

1. the ``MLP + friction penalty`` ablation, which is the empirical control showing
   what a soft constraint does and does not buy;
2. weak supervision of latent states (graining/temperature) where no architectural
   encoding is possible because the quantity is simply unobserved.
"""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F

from tire_nn.layers.friction_envelope import ellipse_radius

__all__ = [
    "force_loss",
    "friction_penalty",
    "symmetry_penalty",
    "zero_slip_penalty",
    "imu_accelerations",
    "vehicle_loss",
]


def force_loss(pred: Tensor, target: Tensor, Fz: Tensor | None = None, kind: str = "mse") -> Tensor:
    """Force regression loss, optionally normalised by load.

    Load-normalised errors weight a 100 N error on a lightly loaded tire the same as
    on a heavily loaded one, which matches how the model is used (the controller
    cares about the friction coefficient, not the absolute force).
    """
    if Fz is not None:
        pred, target = pred / Fz, target / Fz
    if kind == "mse":
        return F.mse_loss(pred, target)
    if kind == "mae":
        return F.l1_loss(pred, target)
    if kind == "huber":
        return F.smooth_l1_loss(pred, target)
    raise ValueError(f"unknown loss kind {kind!r}")


def friction_penalty(Fx: Tensor, Fy: Tensor, mu_x: Tensor, mu_y: Tensor, Fz: Tensor) -> Tensor:
    """Soft friction-ellipse penalty — **ablation only**, never used by encoded models."""
    rho = ellipse_radius(Fx, Fy, mu_x, mu_y, Fz)
    return torch.clamp(rho - 1.0, min=0.0).pow(2).mean()


def symmetry_penalty(model, alpha: Tensor, kappa: Tensor, Fz: Tensor, context=None) -> Tensor:
    """Soft odd-symmetry penalty — ablation only (encoded models satisfy it exactly)."""
    f = model(alpha, kappa, Fz, context)
    fm = model(-alpha, -kappa, Fz, context)
    return ((f.Fx + fm.Fx) ** 2 + (f.Fy + fm.Fy) ** 2).mean()


def zero_slip_penalty(model, Fz: Tensor, context=None) -> Tensor:
    """Soft zero-slip penalty — ablation only."""
    z = torch.zeros_like(Fz)
    f = model(z, z, Fz, context)
    return (f.Fx ** 2 + f.Fy ** 2).mean()


def imu_accelerations(ax_inertial: Tensor, ay_inertial: Tensor, vx: Tensor, vy: Tensor, r: Tensor):
    """Convert inertial accelerations to what a body-mounted IMU measures.

        ax_imu = dvx/dt - r vy      ay_imu = dvy/dt + r vx

    Applied in exactly one place so the centripetal term can never be dropped or
    double-counted between the model and the data (a classic silent bug in
    vehicle-level identification).
    """
    return ax_inertial - r * vy, ay_inertial + r * vx


def vehicle_loss(
    pred: tuple[Tensor, Tensor, Tensor],
    target: tuple[Tensor, Tensor, Tensor],
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    scales: tuple[float, float, float] = (10.0, 10.0, 2.0),
) -> Tensor:
    """Weighted ``(ax, ay, r_dot)`` loss with fixed physical scales.

    ``ax, ay`` are in m/s^2 and ``r_dot`` in rad/s^2 — quantities with different units
    and magnitudes. Dividing by fixed, documented scales (roughly 1 g and a typical
    yaw-acceleration magnitude) makes the weights dimensionless and comparable, rather
    than letting the unit choice silently decide the trade-off.
    """
    total = 0.0
    for p, t, w, s in zip(pred, target, weights, scales):
        total = total + w * F.mse_loss(p / s, t / s)
    return total
