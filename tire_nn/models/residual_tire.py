"""Grey-box residual model: analytical baseline + bounded learned correction.

    F = clip_envelope( F_analytical(alpha, kappa, Fz) + F_residual(...) )

The residual uses the same odd-symmetric parameterisation as ``EncodedTireNet``, so
the *sum* still satisfies odd symmetry and zero force at zero slip exactly, and the
envelope is applied to the sum — the correction can reshape the curve but cannot push
the tire past its friction limit.

Why this rung exists: with little data the analytical model carries the structure and
the network only has to explain the mismatch, which is the regime most real projects
start in. It also separates "the tire differs from Pacejka" from "the tire is
unknown", and the residual magnitude is a directly reportable diagnostic.
"""

from __future__ import annotations

import torch
from torch import Tensor

from tire_nn.layers.friction_envelope import FrictionEnvelope
from tire_nn.models.base import BaseTireModel, ContextEncoder, mu_head
from tire_nn.layers.symmetry import OddSymmetricForceField
from tire_nn.physics.pacejka import MFParams, pacejka_combined
from tire_nn.types import TireForces

__all__ = ["ResidualTireNet"]


class ResidualTireNet(BaseTireModel):
    encodes = ("slip_kinematics", "odd_symmetry", "magic_formula_prior", "friction_envelope")

    def __init__(
        self,
        px: MFParams | None = None,
        py: MFParams | None = None,
        context_keys: tuple[str, ...] = (),
        n_tires: int = 0,
        hidden: tuple[int, ...] = (32, 32),
        residual_scale: float = 0.3,
        hard_envelope: bool = True,
        **field_kwargs,
    ):
        super().__init__()
        self.px = px or MFParams()
        self.py = py or MFParams()
        self.context = ContextEncoder(context_keys, n_tires)
        self.residual_scale = float(residual_scale)
        self.field = OddSymmetricForceField(
            context_dim=self.context.out_dim, hidden=hidden, **field_kwargs
        )
        self.envelope = FrictionEnvelope() if hard_envelope else None
        self.mu = mu_head(1 + self.context.out_dim)

    def forward(self, alpha: Tensor, kappa: Tensor, Fz: Tensor, context=None) -> TireForces:
        Fx0, Fy0 = pacejka_combined(alpha, kappa, Fz, self.px, self.py)
        c = self.context(context, alpha)
        rx, ry = self.field(alpha, kappa, Fz, c)
        Fx = Fx0 + self.residual_scale * rx
        Fy = Fy0 + self.residual_scale * ry
        x = (Fz / self.field.Fz_ref).unsqueeze(-1)
        if c is not None:
            x = torch.cat([x, c], dim=-1)
        p = self.mu(x)
        if self.envelope is not None:
            Fx, Fy = self.envelope(Fx, Fy, p["mu_x"], p["mu_y"], Fz)
        return TireForces(
            Fx=Fx, Fy=Fy,
            params={"mu_x": p["mu_x"], "mu_y": p["mu_y"],
                    "residual_fraction": (self.residual_scale * rx).abs() / (Fx0.abs() + 1e-6)},
        )
