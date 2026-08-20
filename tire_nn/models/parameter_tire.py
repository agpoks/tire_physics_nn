"""ParameterNet + differentiable Magic Formula (PLAN.md §3, P4).

The network never outputs a force. It outputs *physically meaningful parameters* —
``mu``, ``B``, ``C``, ``E``, load sensitivity ``k_mu`` and relaxation length ``sigma`` —
through bounded transforms, and those parameters are pushed through the same
``physics/pacejka.py`` equations that the analytical baseline uses.

Why this is the interpretable end of the ladder: the output is a *tire*, not just a
fit. Cornering stiffness ``B C D``, peak friction and relaxation length can be read
off, plotted against operating conditions, compared with a scipy Magic-Formula fit,
and handed to an MPC that expects Pacejka coefficients. Bounded transforms keep every
parameter valid at every step, so optimisation never crosses a region where the tire
law is meaningless (``C < 0`` flips the curve; ``D < 0`` is a negative peak force).
"""

from __future__ import annotations

import torch
from torch import Tensor

from tire_nn.layers.bounded_parameters import BoundedParameterHead, ParamSpec
from tire_nn.layers.friction_envelope import FrictionEnvelope
from tire_nn.layers.symmetry import mlp
from tire_nn.models.base import BaseTireModel, ContextEncoder
from tire_nn.physics.pacejka import MFParams, cornering_stiffness, load_sensitive_mu, pacejka_combined
from tire_nn.types import TireForces

__all__ = ["MF_SPECS", "ParameterTireNet"]

#: Per-axis Magic Formula parameter ranges. Bounds bracket published passenger-car and
#: fitted F1TENTH values (On-Track-SysID: B 7-8, C 1.6-2.1, E 0.4-0.5) with margin.
MF_SPECS = (
    ParamSpec("mu_x", lo=0.05, hi=2.5, init=1.0),
    ParamSpec("B_x", lo=0.5, hi=40.0, init=10.0),
    ParamSpec("C_x", lo=0.5, hi=2.5, init=1.6),
    ParamSpec("E_x", lo=-2.0, hi=1.0, init=0.5),
    ParamSpec("mu_y", lo=0.05, hi=2.5, init=1.0),
    ParamSpec("B_y", lo=0.5, hi=40.0, init=9.0),
    ParamSpec("C_y", lo=0.5, hi=2.5, init=1.5),
    ParamSpec("E_y", lo=-2.0, hi=1.0, init=0.5),
    ParamSpec("k_mu", lo=0.0, hi=0.5, init=0.05),
    ParamSpec("sigma_x", lo=0.01, hi=None, init=0.2),
    ParamSpec("sigma_y", lo=0.01, hi=None, init=0.3),
)


class ParameterTireNet(BaseTireModel):
    """Predicts bounded Magic Formula parameters from ``(Fz, context)``.

    Note that the parameter network is a function of the *operating condition only*
    (load, pressure, temperature, tire id) — never of the slip itself. That is what
    keeps the result a tire *model* rather than a curve fit: at a given condition the
    tire has one set of coefficients, and the slip dependence comes entirely from the
    analytical Magic Formula.
    """

    encodes = ("slip_kinematics", "odd_symmetry", "magic_formula", "bounded_parameters")

    def __init__(
        self,
        context_keys: tuple[str, ...] = (),
        n_tires: int = 0,
        hidden: tuple[int, ...] = (32, 32),
        Fz_ref: float = 1000.0,
        hard_envelope: bool = False,
        envelope_mode: str = "tanh",
        theoretical_slip: bool = False,
    ):
        super().__init__()
        self.context = ContextEncoder(context_keys, n_tires)
        self.Fz_ref = Fz_ref
        self.theoretical_slip = theoretical_slip
        in_dim = 1 + self.context.out_dim
        self.trunk = mlp(in_dim, max(8, hidden[-1]), hidden[:-1] or (hidden[0],)) if len(hidden) > 1 else None
        feat_dim = max(8, hidden[-1]) if self.trunk is not None else in_dim
        self.head = BoundedParameterHead(feat_dim, MF_SPECS)
        # The Magic Formula is already bounded by mu*Fz; the envelope is optional and
        # only matters for combined slip beyond the similarity method's own bound.
        self.envelope = FrictionEnvelope(envelope_mode) if hard_envelope else None

    def parameters_at(self, Fz: Tensor, context=None) -> dict[str, Tensor]:
        c = self.context(context, Fz)
        x = (Fz / self.Fz_ref).unsqueeze(-1)
        if c is not None:
            x = torch.cat([x, c], dim=-1)
        if self.trunk is not None:
            x = torch.tanh(self.trunk(x))
        return self.head(x)

    def forward(self, alpha: Tensor, kappa: Tensor, Fz: Tensor, context=None) -> TireForces:
        p = self.parameters_at(Fz, context)
        px = MFParams(B=p["B_x"], C=p["C_x"], E=p["E_x"], mu=p["mu_x"], k_mu=p["k_mu"], Fz0=self.Fz_ref)
        py = MFParams(B=p["B_y"], C=p["C_y"], E=p["E_y"], mu=p["mu_y"], k_mu=p["k_mu"], Fz0=self.Fz_ref)
        Fx, Fy = pacejka_combined(alpha, kappa, Fz, px, py, self.theoretical_slip)
        # The friction limit that actually applies at this load is the load-sensitive
        # one, not the nominal value at Fz_ref. Reporting (and enforcing) the nominal
        # mu would understate the ellipse at light load and overstate it at high load.
        mu_x_eff = load_sensitive_mu(Fz, p["mu_x"], p["k_mu"], self.Fz_ref)
        mu_y_eff = load_sensitive_mu(Fz, p["mu_y"], p["k_mu"], self.Fz_ref)
        if self.envelope is not None:
            Fx, Fy = self.envelope(Fx, Fy, mu_x_eff, mu_y_eff, Fz)
        out = dict(p)
        out["mu_x0"], out["mu_y0"] = p["mu_x"], p["mu_y"]      # nominal, at Fz_ref
        out["mu_x"], out["mu_y"] = mu_x_eff, mu_y_eff          # effective, at this load
        out["C_alpha"] = cornering_stiffness(py, Fz)
        out["C_kappa"] = cornering_stiffness(px, Fz)
        return TireForces(Fx=Fx, Fy=Fy, params=out)
