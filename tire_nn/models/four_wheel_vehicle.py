"""Four-wheel vehicle model around ONE shared TireNet (PLAN.md §3, P6).

    m (dvx/dt - r vy) = sum_i Fx_body_i
    m (dvy/dt + r vx) = sum_i Fy_body_i
    Iz dr/dt          = sum_i (x_i Fy_body_i - y_i Fx_body_i)

Why one shared tire model instead of four:

* four independent networks have 4x the parameters and cannot share the scarce
  evidence about the friction limit, which only one corner visits at a time;
* they can absorb *chassis* errors (wrong ``Iz``, wrong load transfer) into per-wheel
  weights, producing a model that fits the training laps and is wrong about the tire;
* physically the four tires are the same product — the differences between corners are
  differences in ``Fz``, ``alpha``, ``kappa`` and steering, all of which are already
  inputs.

Sharing turns every vehicle-level sample into four constraints on one constitutive
law, which is what makes IMU-only identification (Experiment 3) work at all.

The aggregation is exact rigid-body mechanics with no learned correction, so a fit
that looks good cannot be hiding a wrong tire behind a learned chassis.
"""

from __future__ import annotations

import copy

import torch
from torch import Tensor
from torch import nn

from tire_nn.layers.slip_kinematics import slip_angle, slip_ratio
from tire_nn.models.base import BaseTireModel
from tire_nn.physics.vehicle_dynamics import (
    VehicleParams,
    corner_velocities,
    newton_euler,
    quasi_static_loads,
    static_loads,
    wheel_to_body,
)

__all__ = ["FourWheelVehicle", "WHEELS"]

WHEELS = ("FL", "FR", "RL", "RR")


class FourWheelVehicle(nn.Module):
    """Vehicle-level wrapper: shared tire model + exact Newton-Euler aggregation.

    Args:
        tire: the shared constitutive model (any ``BaseTireModel``).
        vp: exact vehicle geometry and inertia.
        share_tire: ``True`` (default) evaluates one module four times. ``False`` deep-copies
            it into four independent networks — the ablation, not the recommendation.
        load_transfer: ``"measured"`` uses the measured ``ax``/``ay`` for the quasi-static
            load transfer (available in every vehicle-level dataset and avoids an
            implicit algebraic loop), ``"static"`` uses the static loads, ``"iterate"``
            solves the loop with a fixed number of Picard iterations.
        corner_embedding: adds a learned per-corner index to the tire context. Off by
            default: it is the loophole through which per-wheel weights come back.
        drag: quadratic drag coefficient ``F = drag * vx^2`` [N s^2/m^2].
        roll_resistance: constant rolling resistance coefficient [-].
    """

    def __init__(
        self,
        tire: BaseTireModel,
        vp: VehicleParams,
        share_tire: bool = True,
        load_transfer: str = "measured",
        corner_embedding: bool = False,
        drag: float = 0.0,
        roll_resistance: float = 0.0,
        n_iter: int = 2,
    ):
        super().__init__()
        self.vp = vp
        self.share_tire = share_tire
        self.load_transfer = load_transfer
        self.corner_embedding = corner_embedding
        self.drag = float(drag)
        self.roll_resistance = float(roll_resistance)
        self.n_iter = int(n_iter)
        if share_tire:
            self.tire = tire
            self.tires = None
        else:
            self.tire = None
            self.tires = nn.ModuleList([copy.deepcopy(tire) for _ in WHEELS])

    # -- kinematics ---------------------------------------------------------

    def corner_slips(self, vx, vy, r, delta, omega) -> tuple[Tensor, Tensor]:
        """Exact per-corner slip from the rigid-body corner velocities. All (B,4)."""
        vxi, vyi = corner_velocities(vx, vy, r, self.vp)
        if delta.dim() == 1:
            delta = torch.stack([delta, delta, torch.zeros_like(delta), torch.zeros_like(delta)], dim=-1)
        # Wheel-frame longitudinal velocity (steering rotates the front corners).
        c, s = torch.cos(delta), torch.sin(delta)
        vx_w = vxi * c + vyi * s
        alpha = slip_angle(vxi, vyi, delta)
        kappa = slip_ratio(omega, vx_w, self.vp.R_e)
        return alpha, kappa, delta

    def corner_loads(self, ax: Tensor | None, ay: Tensor | None, like: Tensor) -> Tensor:
        if self.load_transfer == "static" or ax is None or ay is None:
            return static_loads(self.vp, device=like.device, dtype=like.dtype).expand(*like.shape, 4)
        return quasi_static_loads(ax, ay, self.vp)

    # -- forces -------------------------------------------------------------

    def tire_forces(self, alpha, kappa, Fz, context=None) -> tuple[Tensor, Tensor, dict]:
        """Evaluate the tire model on all four corners. Inputs (B,4) -> forces (B,4)."""
        if self.share_tire:
            ctx = dict(context or {})
            if self.corner_embedding:
                idx = torch.arange(4, device=alpha.device).expand_as(alpha)
                ctx["tire_id"] = idx
            out = self.tire(alpha, kappa, Fz, ctx or None)
            return out.Fx, out.Fy, out.params
        fx, fy = [], []
        for i in range(4):
            ctx_i = {k: v[..., i] for k, v in (context or {}).items()} or None
            o = self.tires[i](alpha[..., i], kappa[..., i], Fz[..., i], ctx_i)
            fx.append(o.Fx)
            fy.append(o.Fy)
        return torch.stack(fx, -1), torch.stack(fy, -1), {}

    def forward(
        self,
        vx: Tensor,
        vy: Tensor,
        r: Tensor,
        delta: Tensor,
        omega: Tensor,
        ax_meas: Tensor | None = None,
        ay_meas: Tensor | None = None,
        context: dict | None = None,
    ) -> dict:
        """Predict ``(ax, ay, r_dot)`` and the per-corner forces.

        ``ax_meas``/``ay_meas`` are used **only** for the quasi-static load transfer, never
        as a target-leaking input to the tire model — the load transfer is a
        kinematic consequence of acceleration, and using the measured value avoids an
        algebraic loop without giving the network the answer.
        """
        alpha, kappa, delta4 = self.corner_slips(vx, vy, r, delta, omega)
        Fz = self.corner_loads(ax_meas, ay_meas, vx)

        for _ in range(self.n_iter if self.load_transfer == "iterate" else 1):
            ctx = dict(context or {})
            ctx.setdefault("vx", vx.unsqueeze(-1).expand_as(alpha))
            Fx_w, Fy_w, params = self.tire_forces(alpha, kappa, Fz, ctx)
            Fx_b, Fy_b = wheel_to_body(Fx_w, Fy_w, delta4)
            drag = self.drag * vx * vx.abs() + self.roll_resistance * self.vp.m * 9.81 * torch.sign(vx)
            ax, ay, r_dot = newton_euler(Fx_b, Fy_b, vx, vy, r, self.vp, drag)
            if self.load_transfer == "iterate":
                Fz = quasi_static_loads(ax.detach(), ay.detach(), self.vp)

        return {
            "ax": ax, "ay": ay, "r_dot": r_dot,
            "Fx_wheel": Fx_w, "Fy_wheel": Fy_w, "Fx_body": Fx_b, "Fy_body": Fy_b,
            "Fz": Fz, "alpha": alpha, "kappa": kappa, "params": params,
        }

    def shared_parameter_ids(self) -> set[int]:
        """Identity of the tire parameter tensors — used by the shared-weights test."""
        module = self.tire if self.share_tire else self.tires[0]
        return {id(p) for p in module.parameters()}

    def extra_repr(self) -> str:
        return (f"share_tire={self.share_tire}, load_transfer={self.load_transfer}, "
                f"corner_embedding={self.corner_embedding}")
