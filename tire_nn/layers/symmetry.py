"""Odd-symmetry-encoded force field (PLAN.md §3, P2).

    Fx = kappa * gx(kappa^2, alpha^2, Fz, c)
    Fy = -alpha * gy(alpha^2, kappa^2, Fz, c)

Because the network sees only the **even** invariants ``kappa^2, alpha^2`` and the
result is multiplied by the odd factor ``kappa`` / ``-alpha``:

* ``Fx(kappa=0) = 0`` and ``Fy(alpha=0) = 0`` hold to machine precision;
* ``Fx(-kappa) = -Fx(kappa)`` and ``Fy(-alpha) = -Fy(alpha)`` hold exactly;

for **any** weights, before, during and after training, in and out of distribution.
A plain MLP violates these worst where data is sparsest — near zero and at the
extremes — and a non-zero force at zero slip is exactly what destabilises an MPC
linearised about straight-line running.

``gx, gy`` are passed through ``softplus`` so they are positive: the force can never
point *against* the slip direction (``Fx kappa >= 0``, ``Fy alpha <= 0``), which is
the dissipativity requirement the brush model satisfies from first principles.

Asymmetric real effects (ply steer, conicity, camber thrust) are intentionally
excluded from ``g``. They enter through an explicit, separable, switchable offset
head so that they stay physically named and individually reportable instead of
being smeared into the weights — the symmetry tests apply to the base term.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

__all__ = ["mlp", "OddSymmetricForceField"]


def mlp(in_features: int, out_features: int, hidden=(32, 32), activation=nn.Tanh) -> nn.Sequential:
    """Small MLP. Defaults are deliberately tiny (PLAN.md §9: prefer interpretable models)."""
    layers: list[nn.Module] = []
    last = in_features
    for h in hidden:
        layers += [nn.Linear(last, h), activation()]
        last = h
    layers += [nn.Linear(last, out_features)]
    return nn.Sequential(*layers)


class OddSymmetricForceField(nn.Module):
    """Symmetry-encoded raw (pre-envelope) force generator.

    Args:
        context_dim: width of the extra context vector appended to the invariants.
        hidden: hidden layer sizes of the shared trunk.
        scale_with_load: multiply ``g`` by ``Fz`` so the network predicts a
            *normalised stiffness* rather than an absolute force. Force then scales
            linearly with load by construction, which is the correct leading-order
            behaviour and removes most of the load dependence from the learning
            problem (the remaining nonlinearity is the load sensitivity of ``mu``,
            handled in the envelope).
        asymmetry: enable the separable offset head for ply steer / conicity / camber.
        alpha_ref, kappa_ref: normalisation scales for the invariants.
    """

    def __init__(
        self,
        context_dim: int = 0,
        hidden: tuple[int, ...] = (32, 32),
        scale_with_load: bool = True,
        asymmetry: bool = False,
        alpha_ref: float = 0.2,
        kappa_ref: float = 0.2,
        Fz_ref: float = 1000.0,
        stiffness_scale: float = 20.0,
    ):
        super().__init__()
        self.context_dim = context_dim
        self.scale_with_load = scale_with_load
        self.asymmetry = asymmetry
        self.alpha_ref = alpha_ref
        self.kappa_ref = kappa_ref
        self.Fz_ref = Fz_ref
        self.stiffness_scale = stiffness_scale

        in_dim = 3 + context_dim          # [kappa^2, alpha^2, Fz_n] + context
        self.trunk = mlp(in_dim, 2, hidden)
        self.offset = mlp(in_dim, 2, hidden) if asymmetry else None

    def invariants(self, alpha: Tensor, kappa: Tensor, Fz: Tensor, context: Tensor | None) -> Tensor:
        feats = [
            (kappa / self.kappa_ref) ** 2,
            (alpha / self.alpha_ref) ** 2,
            Fz / self.Fz_ref,
        ]
        x = torch.stack(feats, dim=-1)
        if context is not None and context.numel() > 0:
            x = torch.cat([x, context], dim=-1)
        return x

    def forward(
        self,
        alpha: Tensor,
        kappa: Tensor,
        Fz: Tensor,
        context: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        x = self.invariants(alpha, kappa, Fz, context)
        raw = self.trunk(x)
        gx = F.softplus(raw[..., 0]) * self.stiffness_scale
        gy = F.softplus(raw[..., 1]) * self.stiffness_scale
        if self.scale_with_load:
            gx = gx * Fz
            gy = gy * Fz

        qx = kappa * gx
        qy = -alpha * gy

        if self.offset is not None:
            off = self.offset(x)
            # Even in (alpha, kappa) by construction -> a pure offset, cleanly separable
            # from the odd base term and reportable on its own.
            qx = qx + off[..., 0] * Fz * 0.01
            qy = qy + off[..., 1] * Fz * 0.01
        return qx, qy
