"""Thermal, wear and graining condition model (PLAN.md §3, P7) — optional extension.

Latent state ``z = [Ts, Tc, wear, graining]``. The **structure is fixed** and only the
rates are learned:

    P_slip    = -(Fx vsx + Fy vsy)                     >= 0 for a dissipative tire
    Cs dTs/dt = eta P_slip - h_sc (Ts-Tc) - h_sa (Ts-T_road)
    Cc dTc/dt = h_sc (Ts-Tc) - h_ca (Tc-T_air)
    dwear/dt  = softplus(f_w(.))                       >= 0, irreversible
    dg/dt     = (1-g) R_form(.) - g R_clean(.)         R_* = softplus(.) >= 0
    mu_eff    = mu_base(Ts, Fz, p) exp(-kw wear) (1 - kg g)

Why each piece is structural rather than learned or penalised:

* **Two thermal nodes.** The minimal structure with the two time scales that are
  actually observed: surface temperature responds within a corner and drives grip,
  core temperature drifts over a stint and sets the operating window.
* **``P_slip = -F . v_slip``.** The only correct energy input. A model heated by
  "speed" or "lateral acceleration" gets braking, coasting and combined slip wrong.
* **Wear via softplus.** Irreversibility is a thermodynamic one-way street; softplus
  makes ``dwear/dt >= 0`` exact rather than probable, so wear can never "heal" during
  a bad gradient step.
* **Graining boundary invariance.** At ``g = 0`` the sink term vanishes and at ``g = 1``
  the source term vanishes, so with non-negative rates ``[0, 1]`` is an invariant set —
  no clamping, no penalty, and the gradient stays informative at the boundary.
* **Gating, not hard-coded curves.** Cold surface and high slip energy are fed to
  ``R_form`` as monotone features, and surface temperature above the window bottom is
  fed to ``R_clean``. The *sign* of each effect is imposed; the *shape* stays learnable.

Experiment 4 runs this on synthetic, weakly supervised states. It is a demonstrator of
the model structure and is **not** validated real motorsport graining.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

from tire_nn.layers.bounded_parameters import BoundedParameter, ParamSpec
from tire_nn.layers.symmetry import mlp
from tire_nn.models.base import BaseTireModel
from tire_nn.physics.thermal import ThermalParams, slip_power, thermal_rates
from tire_nn.physics.wear import effective_friction, graining_rate, wear_rate
from tire_nn.types import TireForces

__all__ = ["ThermoGrainingTire", "CONDITION_SPECS", "STATE_NAMES"]

STATE_NAMES = ("Ts", "Tc", "wear", "graining")

CONDITION_SPECS = (
    ParamSpec("T_opt", lo=310.0, hi=420.0, init=360.0),     # peak-grip surface temperature [K]
    ParamSpec("T_width", lo=5.0, hi=None, init=40.0),       # width of the grip window [K]
    ParamSpec("c_T", lo=0.0, hi=0.8, init=0.3),             # depth of the temperature penalty [-]
    ParamSpec("k_wear", lo=0.0, hi=2.0, init=0.1),          # grip loss per unit wear [-]
    ParamSpec("k_grain", lo=0.0, hi=0.9, init=0.3),         # grip loss at full graining [-]
)


class ThermoGrainingTire(BaseTireModel):
    """Condition-state wrapper around any static tire model.

    The condition never changes the *shape* guarantees of the wrapped model: it only
    scales the friction ellipse through ``context["mu_scale"]``, so odd symmetry, zero
    force at zero slip and the hard envelope all still hold, now with respect to the
    condition-dependent limit.
    """

    encodes = ("two_node_thermal", "slip_power", "irreversible_wear", "bounded_graining")

    def __init__(
        self,
        steady: BaseTireModel,
        thermal: ThermalParams | None = None,
        enable_thermal: bool = True,
        enable_wear: bool = True,
        enable_graining: bool = True,
        hidden: tuple[int, ...] = (16, 16),
        wear_scale: float = 1e-6,
        grain_scale: float = 1.0,
        T_ref_cold: float = 340.0,
        P_ref: float = 5000.0,
    ):
        super().__init__()
        self.steady = steady
        self.thermal = thermal or ThermalParams()
        self.enable_thermal = enable_thermal
        self.enable_wear = enable_wear
        self.enable_graining = enable_graining
        self.wear_scale = float(wear_scale)
        self.grain_scale = float(grain_scale)
        self.T_ref_cold = float(T_ref_cold)
        self.P_ref = float(P_ref)

        # Rate networks: 4 features in, 1 unconstrained rate out (made positive outside).
        self.wear_net = mlp(4, 1, hidden)
        self.form_net = mlp(4, 1, hidden)
        self.clean_net = mlp(4, 1, hidden)
        for spec in CONDITION_SPECS:
            setattr(self, spec.name, BoundedParameter(spec))

    # -- friction modulation ------------------------------------------------

    def temperature_factor(self, Ts: Tensor) -> Tensor:
        """Inverted-parabola grip window, clamped to stay positive.

        Grip peaks at an optimal surface temperature and falls off on both sides; a
        monotone temperature model would predict that a tire keeps gaining grip as it
        heats, which is wrong in both directions (cold graining, hot overheating).
        """
        x = (Ts - self.T_opt()) / self.T_width()
        return torch.clamp(1.0 - self.c_T() * x * x, min=0.05)

    def mu_scale(self, z: Tensor) -> Tensor:
        """Condition multiplier on the friction ellipse, strictly positive."""
        Ts, wear, g = z[..., 0], z[..., 2], z[..., 3]
        base = self.temperature_factor(Ts) if self.enable_thermal else torch.ones_like(Ts)
        return effective_friction(base, wear, g, self.k_wear(), self.k_grain())

    # -- dynamics -----------------------------------------------------------

    def _features(self, z: Tensor, P_slip: Tensor, Fz: Tensor) -> Tensor:
        """Monotone gating features. Signs are imposed here; shapes stay learnable."""
        Ts = z[..., 0]
        cold = F.relu(self.T_ref_cold - Ts) / 50.0          # how far below the window
        hot = F.relu(Ts - self.T_ref_cold) / 50.0           # how far above it
        return torch.stack([cold, hot, P_slip / self.P_ref, Fz / 1000.0], dim=-1)

    def state_rates(self, z: Tensor, Fx: Tensor, Fy: Tensor, vsx: Tensor, vsy: Tensor,
                    Fz: Tensor, T_road: Tensor, T_air: Tensor) -> tuple[Tensor, dict]:
        """``dz/dt`` for ``z = [Ts, Tc, wear, graining]``."""
        Ts, Tc, wear, g = z[..., 0], z[..., 1], z[..., 2], z[..., 3]
        P = slip_power(Fx, Fy, vsx, vsy)
        feats = self._features(z, P, Fz)

        if self.enable_thermal:
            dTs, dTc = thermal_rates(Ts, Tc, P, T_road, T_air, self.thermal)
        else:
            dTs = dTc = torch.zeros_like(Ts)

        dwear = (wear_rate(self.wear_net(feats).squeeze(-1)) * self.wear_scale
                 if self.enable_wear else torch.zeros_like(wear))

        if self.enable_graining:
            R_form = F.softplus(self.form_net(feats).squeeze(-1)) * self.grain_scale
            R_clean = F.softplus(self.clean_net(feats).squeeze(-1)) * self.grain_scale
            dg = graining_rate(g, R_form, R_clean)
        else:
            R_form = R_clean = torch.zeros_like(g)
            dg = torch.zeros_like(g)

        return torch.stack([dTs, dTc, dwear, dg], dim=-1), {
            "P_slip": P, "R_form": R_form, "R_clean": R_clean}

    def step(self, z: Tensor, dz: Tensor, dt: float) -> Tensor:
        """Explicit Euler on the condition states, with the graining step made safe.

        The ``[0, 1]`` invariance of the continuous graining ODE only transfers to a
        *discrete* step if the step is small enough. Rather than clamp (which would
        kill the gradient at the boundary), the graining increment is scaled by the
        exact linear-ODE solution over the step: with ``R_form, R_clean`` held constant
        the update is a convex interpolation toward ``R_form/(R_form+R_clean)``, which
        is in ``[0, 1]`` for any ``dt``.
        """
        return z + dt * dz

    def graining_step(self, g: Tensor, R_form: Tensor, R_clean: Tensor, dt: float) -> Tensor:
        """Exact zero-order-hold graining update — unconditionally inside ``[0, 1]``."""
        total = R_form + R_clean
        g_inf = R_form / torch.clamp(total, min=1e-12)
        decay = torch.exp(-torch.clamp(total, min=0.0) * dt)
        return torch.where(total > 1e-12, g_inf + (g - g_inf) * decay, g)

    def initial_state(self, Fz: Tensor, T0: float = 300.0) -> Tensor:
        """Cold tire, unworn, clean."""
        return torch.stack([
            torch.full_like(Fz, T0), torch.full_like(Fz, T0),
            torch.zeros_like(Fz), torch.zeros_like(Fz)], dim=-1)

    # -- interfaces ---------------------------------------------------------

    def forward(self, alpha, kappa, Fz, context=None) -> TireForces:
        """Static call at the condition given in ``context`` (``z`` key), or a clean tire."""
        z = (context or {}).get("z")
        if z is None:
            z = self.initial_state(Fz, T0=float(self.T_opt().detach()))
        ctx = dict(context or {})
        ctx.pop("z", None)
        ctx["mu_scale"] = self.mu_scale(z)
        out = self.steady(alpha, kappa, Fz, ctx)
        out.params.update({name: z[..., i] for i, name in enumerate(STATE_NAMES)})
        out.params["mu_scale"] = ctx["mu_scale"]
        return out

    def rollout_condition(
        self,
        alpha: Tensor,
        kappa: Tensor,
        Fz: Tensor,
        vx: Tensor,
        dt: float,
        T_road: Tensor | float = 300.0,
        T_air: Tensor | float = 295.0,
        z0: Tensor | None = None,
        context: dict | None = None,
    ) -> tuple[Tensor, Tensor, dict]:
        """Integrate forces and condition states over a sequence.

        Inputs ``(B, T)`` -> forces ``(B, T, 2)``, states ``(B, T, 4)``.

        Slip velocities are derived from the slip definitions
        (``vsx = kappa vx``, ``vsy = vx tan(alpha)``) so the dissipated power is
        consistent with the same kinematics the force model uses.
        """
        B, T = alpha.shape
        z = self.initial_state(Fz[..., 0]) if z0 is None else z0
        T_road_t = torch.as_tensor(T_road, dtype=alpha.dtype, device=alpha.device).expand(B)
        T_air_t = torch.as_tensor(T_air, dtype=alpha.dtype, device=alpha.device).expand(B)

        forces, states, extras = [], [], []
        for t in range(T):
            ctx = {k: v[..., t] for k, v in (context or {}).items()}
            ctx["z"] = z
            out = self(alpha[..., t], kappa[..., t], Fz[..., t], ctx)
            vsx = kappa[..., t] * vx[..., t]
            vsy = vx[..., t] * torch.tan(alpha[..., t])
            dz, aux = self.state_rates(z, out.Fx, out.Fy, vsx, vsy, Fz[..., t], T_road_t, T_air_t)

            forces.append(torch.stack([out.Fx, out.Fy], dim=-1))
            states.append(z)
            extras.append(aux["P_slip"])

            z_next = self.step(z, dz, dt)
            if self.enable_graining:
                # Replace the Euler graining component with the exact bounded update.
                g_next = self.graining_step(z[..., 3], aux["R_form"], aux["R_clean"], dt)
                z_next = torch.cat([z_next[..., :3], g_next.unsqueeze(-1)], dim=-1)
            z = z_next

        return (torch.stack(forces, dim=-2), torch.stack(states, dim=-2),
                {"P_slip": torch.stack(extras, dim=-1)})
