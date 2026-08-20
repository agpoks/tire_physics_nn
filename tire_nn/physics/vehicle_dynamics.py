"""Exact rigid-body vehicle equations — no learnable parameters (PLAN.md §3, P6).

Everything here is geometry and Newton-Euler. The tire model is the only thing that
may be learned; if any of these equations were learned as well, vehicle-level
training could hide a wrong tire model behind a wrong chassis model.

Wheel order is fixed project-wide: ``(FL, FR, RL, RR)`` along the last axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = [
    "VehicleParams",
    "corner_positions",
    "corner_velocities",
    "static_loads",
    "quasi_static_loads",
    "wheel_to_body",
    "newton_euler",
]

WHEELS = ("FL", "FR", "RL", "RR")
G = 9.81


@dataclass
class VehicleParams:
    m: float            # mass [kg]
    Iz: float           # yaw inertia [kg m^2]
    lf: float           # CoG -> front axle [m]
    lr: float           # CoG -> rear axle [m]
    t_f: float          # front track width [m]
    t_r: float          # rear track width [m]
    h_cg: float = 0.0   # CoG height [m] (0 disables load transfer)
    R_e: float = 0.3    # effective rolling radius [m]

    @property
    def wheelbase(self) -> float:
        return self.lf + self.lr


def corner_positions(vp: VehicleParams, device=None, dtype=torch.float32) -> tuple[Tensor, Tensor]:
    """Exact geometric corner positions ``(x_i, y_i)`` in the body frame, shape (4,)."""
    x = torch.tensor([vp.lf, vp.lf, -vp.lr, -vp.lr], device=device, dtype=dtype)
    y = torch.tensor([vp.t_f / 2, -vp.t_f / 2, vp.t_r / 2, -vp.t_r / 2], device=device, dtype=dtype)
    return x, y


def corner_velocities(vx: Tensor, vy: Tensor, r: Tensor, vp: VehicleParams) -> tuple[Tensor, Tensor]:
    """``v_i = v_cog + omega x r_i`` for the four corners. Inputs (B,), outputs (B,4)."""
    x, y = corner_positions(vp, device=vx.device, dtype=vx.dtype)
    vxi = vx.unsqueeze(-1) - r.unsqueeze(-1) * y
    vyi = vy.unsqueeze(-1) + r.unsqueeze(-1) * x
    return vxi, vyi


def static_loads(vp: VehicleParams, device=None, dtype=torch.float32) -> Tensor:
    """Static per-corner vertical loads (4,), summing to ``m g``."""
    Ff = vp.m * G * vp.lr / vp.wheelbase
    Fr = vp.m * G * vp.lf / vp.wheelbase
    return torch.tensor([Ff / 2, Ff / 2, Fr / 2, Fr / 2], device=device, dtype=dtype)


def quasi_static_loads(ax: Tensor, ay: Tensor, vp: VehicleParams, fz_min: float = 1.0) -> Tensor:
    """Quasi-static load transfer. ``ax, ay`` (B,) [m/s^2] -> ``Fz`` (B,4) [N].

    Longitudinal transfer moves load front<->rear, lateral transfer moves it
    left<->right within each axle. Both are constructed so the total stays ``m g``
    exactly (the transfers cancel in the sum), which is what makes the four
    per-corner loads consistent with the same chassis used in ``newton_euler``.
    """
    L = vp.wheelbase
    Ff = vp.m * (G * vp.lr - ax * vp.h_cg) / L
    Fr = vp.m * (G * vp.lf + ax * vp.h_cg) / L
    dFf = Ff * ay * vp.h_cg / (G * max(vp.t_f, 1e-6))
    dFr = Fr * ay * vp.h_cg / (G * max(vp.t_r, 1e-6))
    Fz = torch.stack([Ff / 2 - dFf, Ff / 2 + dFf, Fr / 2 - dFr, Fr / 2 + dFr], dim=-1)
    return torch.clamp(Fz, min=fz_min)


def wheel_to_body(Fx_w: Tensor, Fy_w: Tensor, delta: Tensor) -> tuple[Tensor, Tensor]:
    """Rotate wheel-frame forces into the body frame. All tensors (B,4)."""
    c, s = torch.cos(delta), torch.sin(delta)
    Fx_b = Fx_w * c - Fy_w * s
    Fy_b = Fx_w * s + Fy_w * c
    return Fx_b, Fy_b


def newton_euler(
    Fx_b: Tensor,
    Fy_b: Tensor,
    vx: Tensor,
    vy: Tensor,
    r: Tensor,
    vp: VehicleParams,
    F_drag: Tensor | float = 0.0,
) -> tuple[Tensor, Tensor, Tensor]:
    """Planar rigid-body equations. Inputs (B,4) body-frame forces -> ``(ax, ay, r_dot)``.

        m (dvx/dt - r vy) = sum Fx_b - F_drag
        m (dvy/dt + r vx) = sum Fy_b
        Iz dr/dt          = sum (x_i Fy_b_i - y_i Fx_b_i)

    Returns the *inertial* accelerations ``dvx/dt, dvy/dt`` and ``dr/dt``. Note that
    a body-mounted IMU measures ``ax_imu = dvx/dt - r vy``; conversion helpers live
    in ``training/losses.py`` so the convention is applied in exactly one place.
    """
    x, y = corner_positions(vp, device=Fx_b.device, dtype=Fx_b.dtype)
    sumFx = Fx_b.sum(-1) - F_drag
    sumFy = Fy_b.sum(-1)
    Mz = (x * Fy_b - y * Fx_b).sum(-1)
    ax = sumFx / vp.m + r * vy
    ay = sumFy / vp.m - r * vx
    r_dot = Mz / vp.Iz
    return ax, ay, r_dot
