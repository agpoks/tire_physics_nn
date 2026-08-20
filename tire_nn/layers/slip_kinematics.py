"""Analytical slip kinematics — computed, never learned (PLAN.md §3, P1).

Slip is a *definition* that follows from rigid-body kinematics plus the measured
wheel speed. Learning it would spend network capacity on reproducing an identity
and would destroy the interpretability of everything downstream. The only modelling
choices in this module are the low-speed regularisation ``v_eps`` and the sign
convention, both documented and both explicit.

Sign convention (SAE, PLAN.md §4.1):
    kappa = (R_e omega - vx) / max(abs(vx), v_eps)   -> positive when driving
    alpha = atan2(vy, abs(vx)) - delta               -> positive nose-out at the wheel
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch import nn

__all__ = ["slip_ratio", "slip_angle", "slip_velocity", "SlipKinematics"]

V_EPS_DEFAULT = 0.5   # [m/s] low-speed regularisation


def slip_ratio(omega: Tensor, vx: Tensor, R_e: float | Tensor, v_eps: float = V_EPS_DEFAULT) -> Tensor:
    """Longitudinal slip ratio, driving-positive."""
    v_ref = torch.clamp(vx.abs(), min=v_eps)
    return (R_e * omega - vx) / v_ref


def slip_angle(vx: Tensor, vy: Tensor, delta: Tensor | float = 0.0, v_eps: float = V_EPS_DEFAULT) -> Tensor:
    """Side-slip angle at the wheel [rad], measured in the wheel frame."""
    v_ref = torch.clamp(vx.abs(), min=v_eps)
    return torch.atan2(vy, v_ref) - delta


def slip_velocity(
    vx: Tensor,
    vy: Tensor,
    omega: Tensor,
    delta: Tensor | float,
    R_e: float | Tensor,
) -> tuple[Tensor, Tensor]:
    """Contact-patch slip velocity in the **wheel** frame ``(vsx, vsy)`` [m/s].

    Needed by the thermal model, where the dissipated power is ``-F . v_slip``
    (``physics/thermal.slip_power``) — an approximation by ``kappa * vx`` would get
    the sign wrong under braking.
    """
    c, s = torch.cos(torch.as_tensor(delta)), torch.sin(torch.as_tensor(delta))
    vx_w = vx * c + vy * s
    vy_w = -vx * s + vy * c
    return R_e * omega - vx_w, vy_w


class SlipKinematics(nn.Module):
    """Stateless module wrapper so slip computation appears explicitly in a model graph.

    Being an ``nn.Module`` with **no parameters** is the point: it makes the absence
    of learning here visible in ``model.named_parameters()`` and lets vehicle-level
    gradients (Experiment 3) flow from raw wheel speeds/steering through to the
    shared ``TireNet``.
    """

    def __init__(self, R_e: float, v_eps: float = V_EPS_DEFAULT):
        super().__init__()
        self.R_e = float(R_e)
        self.v_eps = float(v_eps)

    def forward(self, vx: Tensor, vy: Tensor, omega: Tensor, delta: Tensor | float = 0.0):
        return (
            slip_angle(vx, vy, delta, self.v_eps),
            slip_ratio(omega, vx, self.R_e, self.v_eps),
        )

    def extra_repr(self) -> str:
        return f"R_e={self.R_e}, v_eps={self.v_eps}, learnable_parameters=0"
