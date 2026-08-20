"""Symmetry- and envelope-encoded tire networks (PLAN.md §3, P2 + P3).

Two rungs of the ablation ladder:

``SymmetryTireNet``  P1 + P2         — odd symmetry and zero-slip-zero-force exact
``EncodedTireNet``   P1 + P2 + P3    — plus a friction ellipse that cannot be exceeded

Both share the same trunk, so the comparison isolates the envelope.
"""

from __future__ import annotations

from torch import Tensor

from tire_nn.layers.friction_envelope import FrictionEnvelope, ellipse_radius
from tire_nn.layers.symmetry import OddSymmetricForceField
from tire_nn.models.base import BaseTireModel, ContextEncoder, mu_head
from tire_nn.types import TireForces

__all__ = ["SymmetryTireNet", "EncodedTireNet"]


class SymmetryTireNet(BaseTireModel):
    """``Fx = kappa gx(kappa^2, alpha^2, Fz, c)``, ``Fy = -alpha gy(alpha^2, kappa^2, Fz, c)``."""

    encodes = ("slip_kinematics", "odd_symmetry", "dissipativity")

    def __init__(
        self,
        context_keys: tuple[str, ...] = (),
        n_tires: int = 0,
        hidden: tuple[int, ...] = (32, 32),
        asymmetry: bool = False,
        **field_kwargs,
    ):
        super().__init__()
        self.context = ContextEncoder(context_keys, n_tires)
        self.field = OddSymmetricForceField(
            context_dim=self.context.out_dim, hidden=hidden, asymmetry=asymmetry, **field_kwargs
        )

    def _raw(self, alpha, kappa, Fz, context):
        c = self.context(context, alpha)
        return self.field(alpha, kappa, Fz, c), c

    def forward(self, alpha: Tensor, kappa: Tensor, Fz: Tensor, context=None) -> TireForces:
        (qx, qy), _ = self._raw(alpha, kappa, Fz, context)
        return TireForces(Fx=qx, Fy=qy)


class EncodedTireNet(SymmetryTireNet):
    """Symmetry-encoded field followed by a hard, differentiable friction envelope.

    ``mu_x``/``mu_y`` are produced by a bounded head from ``(Fz, context)`` — so the
    ellipse itself is learned, but always inside the declared physical range, and the
    resulting force is inside that ellipse for *any* weights (P3). ``params`` exposes
    the learned friction values and the envelope utilisation ``rho``, both of which
    are directly plottable and comparable with the fitted Magic Formula.
    """

    encodes = ("slip_kinematics", "odd_symmetry", "dissipativity", "friction_envelope")

    def __init__(self, *args, envelope_mode: str = "tanh", max_utilization: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.envelope = FrictionEnvelope(envelope_mode, max_utilization)
        self.mu = mu_head(1 + self.context.out_dim)

    def friction(self, Fz: Tensor, c: Tensor | None) -> tuple[Tensor, Tensor]:
        import torch

        x = (Fz / self.field.Fz_ref).unsqueeze(-1)
        if c is not None:
            x = torch.cat([x, c], dim=-1)
        p = self.mu(x)
        return p["mu_x"], p["mu_y"]

    def forward(self, alpha: Tensor, kappa: Tensor, Fz: Tensor, context=None) -> TireForces:
        (qx, qy), c = self._raw(alpha, kappa, Fz, context)
        mu_x, mu_y = self.friction(Fz, c)
        if context is not None and "mu_scale" in context:
            # Externally supplied road-friction scaling (e.g. from the condition model
            # in P7, or a mu estimator). Applied to the envelope, never to the force
            # afterwards, so the guarantee still holds w.r.t. the scaled ellipse.
            mu_x = mu_x * context["mu_scale"]
            mu_y = mu_y * context["mu_scale"]
        Fx, Fy = self.envelope(qx, qy, mu_x, mu_y, Fz)
        return TireForces(
            Fx=Fx,
            Fy=Fy,
            params={"mu_x": mu_x, "mu_y": mu_y, "rho": ellipse_radius(Fx, Fy, mu_x, mu_y, Fz)},
        )
