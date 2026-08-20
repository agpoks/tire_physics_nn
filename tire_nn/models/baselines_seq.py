"""Sequence baselines for Experiment 2: generic GRU and generic Neural ODE.

Both are given exactly the same inputs as ``RelaxationTireCell`` and the same
``rollout`` signature, so the comparison isolates one thing: whether the *structure*
``tau = sigma / v`` is encoded or has to be learned.

Neither encodes any tire physics. That is deliberate — they are the controls.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch import nn

from tire_nn.layers.symmetry import mlp
from tire_nn.models.base import BaseTireModel, ContextEncoder
from tire_nn.types import TireForces

__all__ = ["GRUTireModel", "NeuralODETireModel"]


def _inputs(alpha, kappa, Fz, vx, refs) -> Tensor:
    a_ref, k_ref, fz_ref, v_ref = refs
    return torch.stack([alpha / a_ref, kappa / k_ref, Fz / fz_ref, vx / v_ref], dim=-1)


class GRUTireModel(BaseTireModel):
    """Generic recurrent baseline: hidden state -> force, no physical structure."""

    encodes = ()

    def __init__(self, hidden_size: int = 32, context_keys=(), n_tires: int = 0,
                 force_scale: float = 1000.0, refs=(0.2, 0.2, 1000.0, 20.0)):
        super().__init__()
        self.context = ContextEncoder(context_keys, n_tires)
        self.refs = refs
        self.force_scale = force_scale
        self.cell = nn.GRUCell(4 + self.context.out_dim, hidden_size)
        self.readout = nn.Linear(hidden_size, 2)
        self.hidden_size = hidden_size

    def rollout(self, alpha, kappa, Fz, vx, dt, context=None, F0=None, method=None) -> Tensor:
        B, T = alpha.shape
        h = torch.zeros(B, self.hidden_size, device=alpha.device, dtype=alpha.dtype)
        out = []
        for t in range(T):
            x = _inputs(alpha[..., t], kappa[..., t], Fz[..., t], vx[..., t], self.refs)
            c = self.context({k: v[..., t] for k, v in context.items()} if context else None, alpha[..., t])
            if c is not None:
                x = torch.cat([x, c], dim=-1)
            h = self.cell(x, h)
            out.append(self.readout(h) * self.force_scale)
        return torch.stack(out, dim=-2)

    def forward(self, alpha, kappa, Fz, context=None) -> TireForces:
        """Single-sample call: one GRU step from a zero hidden state (no steady state exists)."""
        vx = (context or {}).get("vx", torch.full_like(alpha, self.refs[3]))
        F = self.rollout(alpha.unsqueeze(-1), kappa.unsqueeze(-1), Fz.unsqueeze(-1),
                         vx.unsqueeze(-1), 0.0,
                         {k: v.unsqueeze(-1) for k, v in context.items()} if context else None)
        return TireForces(Fx=F[..., 0, 0], Fy=F[..., 0, 1])


class NeuralODETireModel(BaseTireModel):
    """Generic Neural ODE: ``dF/dt = f_theta(F, alpha, kappa, Fz, vx)``.

    Continuous-time like the relaxation cell, but with an unconstrained right-hand
    side — nothing forces the time constant to scale with speed, nothing forces the
    dynamics to be contractive, and the steady state is only implicit.
    """

    encodes = ()

    def __init__(self, hidden=(32, 32), context_keys=(), n_tires: int = 0,
                 force_scale: float = 1000.0, rate_scale: float = 100.0,
                 refs=(0.2, 0.2, 1000.0, 20.0)):
        super().__init__()
        self.context = ContextEncoder(context_keys, n_tires)
        self.refs = refs
        self.force_scale = force_scale
        self.rate_scale = rate_scale
        self.net = mlp(6 + self.context.out_dim, 2, hidden)

    def rates(self, F, alpha, kappa, Fz, vx, context=None) -> Tensor:
        x = _inputs(alpha, kappa, Fz, vx, self.refs)
        c = self.context(context, alpha)
        x = torch.cat([x, F / self.force_scale] + ([c] if c is not None else []), dim=-1)
        return self.net(x) * self.rate_scale * self.force_scale

    def rollout(self, alpha, kappa, Fz, vx, dt, context=None, F0=None, method="rk4") -> Tensor:
        T = alpha.shape[-1]
        ctx_t = lambda t: ({k: v[..., t] for k, v in context.items()} if context else None)
        F = torch.zeros(*alpha.shape[:-1], 2, device=alpha.device, dtype=alpha.dtype) if F0 is None else F0
        out = [F]
        for t in range(T - 1):
            f = lambda x: self.rates(x, alpha[..., t], kappa[..., t], Fz[..., t], vx[..., t], ctx_t(t))
            if method == "euler":
                F = F + dt * f(F)
            else:
                k1 = f(F)
                k2 = f(F + 0.5 * dt * k1)
                k3 = f(F + 0.5 * dt * k2)
                k4 = f(F + dt * k3)
                F = F + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            out.append(F)
        return torch.stack(out, dim=-2)

    def forward(self, alpha, kappa, Fz, context=None) -> TireForces:
        """No closed-form steady state: report the state reached after a short settle."""
        T = 40
        expand = lambda v: v.unsqueeze(-1).expand(*v.shape, T)
        vx = (context or {}).get("vx", torch.full_like(alpha, self.refs[3]))
        F = self.rollout(expand(alpha), expand(kappa), expand(Fz), expand(vx), 0.005,
                         {k: expand(v) for k, v in context.items()} if context else None)
        return TireForces(Fx=F[..., -1, 0], Fy=F[..., -1, 1])
