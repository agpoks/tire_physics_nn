"""Unconstrained MLP baseline — the bottom of the ablation ladder (PLAN.md §2.3).

Deliberately encodes **nothing**. It exists to quantify what each physical prior
buys, and to demonstrate the failure modes the priors remove: non-zero force at zero
slip, broken odd symmetry, and forces outside the friction ellipse — all of which
appear mostly *outside* the training distribution, which is where a racing
controller operates.

``friction_penalty=True`` does not change the architecture; it only marks the model
so the trainer adds a soft envelope penalty (``training/losses.py``). That ablation
is the empirical argument for why P3 is structural here.
"""

from __future__ import annotations

import torch
from torch import Tensor

from tire_nn.layers.symmetry import mlp
from tire_nn.models.base import BaseTireModel, ContextEncoder
from tire_nn.types import TireForces

__all__ = ["MLPTireModel"]


class MLPTireModel(BaseTireModel):
    encodes = ()

    def __init__(
        self,
        context_keys: tuple[str, ...] = (),
        n_tires: int = 0,
        hidden: tuple[int, ...] = (64, 64),
        alpha_ref: float = 0.2,
        kappa_ref: float = 0.2,
        Fz_ref: float = 1000.0,
        force_scale: float = 1.0,
        scale_with_load: bool = True,
        friction_penalty: bool = False,
    ):
        super().__init__()
        self.context = ContextEncoder(context_keys, n_tires)
        self.alpha_ref, self.kappa_ref, self.Fz_ref = alpha_ref, kappa_ref, Fz_ref
        self.force_scale = force_scale
        self.scale_with_load = scale_with_load
        self.friction_penalty = friction_penalty
        self.net = mlp(3 + self.context.out_dim, 2, hidden)

    def forward(self, alpha: Tensor, kappa: Tensor, Fz: Tensor, context=None) -> TireForces:
        feats = [alpha / self.alpha_ref, kappa / self.kappa_ref, Fz / self.Fz_ref]
        x = torch.stack(feats, dim=-1)
        c = self.context(context, alpha)
        if c is not None:
            x = torch.cat([x, c], dim=-1)
        out = self.net(x) * self.force_scale
        Fx, Fy = out[..., 0], out[..., 1]
        if self.scale_with_load:
            Fx, Fy = Fx * Fz, Fy * Fz
        return TireForces(Fx=Fx, Fy=Fy)
