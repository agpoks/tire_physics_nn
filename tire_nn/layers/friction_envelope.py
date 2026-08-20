"""Hard friction envelope by differentiable radial projection (PLAN.md §3, P3).

Guarantees, **by construction and for any network weights**::

    (Fx / (mu_x Fz))^2 + (Fy / (mu_y Fz))^2 < 1

This is deliberately *not* a penalty term. A penalty is satisfied only in
expectation over the training distribution, is silently violated exactly where an
aggressive racing controller operates (the saturated, data-sparse limit), and
trades off against RMSE through an arbitrary weight. A structural projection has
none of those failure modes.

Why a smooth squashing rather than ``min(1, 1/rho)``:

* a hard clip has **zero gradient** outside the ellipse — learning dies precisely
  on the saturating samples that carry the information about ``mu``;
* it puts a kink on the ellipse that a Newton-type MPC will find and stall on;
* ``tanh(rho)/rho`` is ``C^inf``, equals ``1 - rho^2/3 + ...`` near the origin (so the
  linear-range cornering stiffness is untouched), and satisfies ``rho tanh(rho)/rho
  = tanh(rho) < 1`` strictly, for all inputs.

The projection is **radial**, preserving force direction: the contact-patch shear
stress opposes the slip-velocity vector, so scaling both components equally is the
physically correct way to hit the limit (clipping one component alone would rotate
the force vector away from the slip direction).
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch import nn

__all__ = ["ellipse_radius", "project_into_ellipse", "FrictionEnvelope"]

EPS = 1e-9


def ellipse_radius(Fx: Tensor, Fy: Tensor, mu_x: Tensor, mu_y: Tensor, Fz: Tensor) -> Tensor:
    """Normalised friction-ellipse radius ``rho``; ``rho <= 1`` means inside."""
    ax = torch.clamp(mu_x * Fz, min=EPS)
    ay = torch.clamp(mu_y * Fz, min=EPS)
    return torch.sqrt((Fx / ax) ** 2 + (Fy / ay) ** 2 + EPS)


def project_into_ellipse(
    qx: Tensor,
    qy: Tensor,
    mu_x: Tensor,
    mu_y: Tensor,
    Fz: Tensor,
    mode: str = "tanh",
    max_utilization: float = 1.0,
) -> tuple[Tensor, Tensor]:
    """Map unconstrained ``(qx, qy)`` into the open friction ellipse.

    ``mode="tanh"``      : ``s = tanh(rho)/rho`` (default; sharpest approach to the limit)
    ``mode="algebraic"`` : ``s = 1/sqrt(1 + rho^2)`` (softer saturation, cheaper)

    ``max_utilization`` < 1 shrinks the admissible ellipse. In exact arithmetic the
    projection is already strict (``tanh(rho) < 1``), but ``tanh`` rounds to exactly
    1.0 in float32 for ``rho > ~8``, so the deep-saturation limit is attained rather
    than approached. Set e.g. 0.999 when a downstream solver needs a strictly
    interior point; the missing 0.1% is absorbed by the learned ``mu``.
    """
    rho = ellipse_radius(qx, qy, mu_x, mu_y, Fz)
    if mode == "tanh":
        scale = torch.tanh(rho) / rho
    elif mode == "algebraic":
        scale = torch.rsqrt(1.0 + rho * rho)
    else:
        raise ValueError(f"unknown projection mode {mode!r}")
    scale = scale * max_utilization
    return qx * scale, qy * scale


class FrictionEnvelope(nn.Module):
    """Module form of :func:`project_into_ellipse` (no learnable parameters).

    ``mu_x``/``mu_y`` are supplied by the caller — typically a ``BoundedParameterHead``
    (P4) and, when the condition model is enabled, modulated by wear/graining (P7).
    """

    def __init__(self, mode: str = "tanh", max_utilization: float = 1.0):
        super().__init__()
        self.mode = mode
        self.max_utilization = float(max_utilization)

    def forward(self, qx, qy, mu_x, mu_y, Fz):
        return project_into_ellipse(qx, qy, mu_x, mu_y, Fz, self.mode, self.max_utilization)

    def extra_repr(self) -> str:
        return f"mode={self.mode}, max_utilization={self.max_utilization}, guarantee='(Fx/muxFz)^2+(Fy/muyFz)^2 < 1', learnable_parameters=0"
