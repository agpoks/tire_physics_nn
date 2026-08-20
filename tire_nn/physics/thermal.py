"""Two-node (surface/core) tire thermal model — structure fixed, rates supplied.

PLAN.md §3, P7. The split into a fast surface node and a slow core node is the
minimal structure that reproduces the two distinct time scales a race engineer
actually observes: surface temperature responds within a corner and drives grip,
core temperature drifts over a stint and drives the operating window.

The only energy input is the slip power ``P_slip = -F_tire . v_slip``, which is the
*correct* dissipation term: a model heated by "speed" or "lateral acceleration"
gets braking, coasting and combined slip wrong.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = ["ThermalParams", "slip_power", "thermal_rates"]


@dataclass
class ThermalParams:
    Cs: float = 8.0e3     # surface heat capacity        [J/K]
    Cc: float = 6.0e4     # core heat capacity           [J/K]
    h_sc: float = 250.0   # surface <-> core conductance [W/K]
    h_sa: float = 120.0   # surface <-> road/air         [W/K]
    h_ca: float = 30.0    # core <-> air                 [W/K]
    eta: float = 0.6      # fraction of slip power into the surface node [-]


def slip_power(Fx: Tensor, Fy: Tensor, vsx: Tensor, vsy: Tensor, clamp_positive: bool = True) -> Tensor:
    """``P_slip = -(Fx vsx + Fy vsy)`` [W].

    With SAE signs the tire force opposes the slip velocity, so this is
    non-negative for a dissipative tire. It is clamped at zero by default: a
    negative value would mean the tire *generates* energy, which is unphysical and
    would let a bad gradient step cool the tire by sliding it.
    """
    P = -(Fx * vsx + Fy * vsy)
    return torch.clamp(P, min=0.0) if clamp_positive else P


def thermal_rates(
    Ts: Tensor,
    Tc: Tensor,
    P_slip: Tensor,
    T_road: Tensor,
    T_air: Tensor,
    p: ThermalParams,
) -> tuple[Tensor, Tensor]:
    """Return ``(dTs/dt, dTc/dt)`` [K/s].

        Cs dTs/dt = eta P_slip - h_sc (Ts - Tc) - h_sa (Ts - T_road)
        Cc dTc/dt = h_sc (Ts - Tc) - h_ca (Tc - T_air)

    The ``h_sc`` coupling appears with opposite signs in the two equations, so the
    pair conserves energy apart from the explicit environment losses.
    """
    dTs = (p.eta * P_slip - p.h_sc * (Ts - Tc) - p.h_sa * (Ts - T_road)) / p.Cs
    dTc = (p.h_sc * (Ts - Tc) - p.h_ca * (Tc - T_air)) / p.Cc
    return dTs, dTc
