"""Physics-encoded transient tire model (PLAN.md §3, P5).

    tau_i = sigma_i / (abs(v_x) + eps),    dF_i/dt = (F_i,ss - F_i) / tau_i,   i in {x, y}

Why this structure rather than a generic recurrent cell: tire force does not respond
instantaneously to a slip step, because the contact patch has to deform, and the
transient is parameterised by **travelled distance**, not by time — hence
``tau = sigma / v``. A GRU has to discover that speed dependence from data, and
typically discovers it only for the speeds it saw. Encoding it means the learned
quantity is one interpretable length per axis (~0.1-0.6 m for a passenger tire,
shorter for small-scale racing tires) instead of a gate matrix, and the model
extrapolates across speed for free.

``sigma > 0`` by construction (softplus, ``layers/bounded_parameters``), so ``tau > 0``,
so the ODE is contractive and unconditionally stable — the model cannot learn a
divergent transient.

At standstill ``tau -> sigma/eps`` is large and the force is frozen rather than
singular: a non-rolling tire does not relax, because relaxation is a rolling-distance
phenomenon. That is the physically correct limit and it is why the regularisation is
placed on the speed ``abs(v_x)`` rather than on ``tau``.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch import nn

from tire_nn.layers.bounded_parameters import BoundedParameter, ParamSpec
from tire_nn.models.base import BaseTireModel
from tire_nn.types import TireForces

__all__ = ["RelaxationTireCell", "SIGMA_SPECS"]

V_EPS = 0.5     # [m/s] — same regularisation scale as layers/slip_kinematics

SIGMA_SPECS = (
    ParamSpec("sigma_x", lo=0.005, hi=None, init=0.15),
    ParamSpec("sigma_y", lo=0.005, hi=None, init=0.30),
)


class RelaxationTireCell(BaseTireModel):
    """Wraps any static tire model in first-order relaxation dynamics.

    The steady-state model is used unchanged, so relaxation composes with every rung
    of the ablation ladder (and the envelope guarantee still holds for ``F_ss``; the
    relaxed force is a convex-combination-like contraction toward ``F_ss``, so it can
    never leave the ellipse either once it starts inside it — see
    ``tests/test_relaxation.py``).
    """

    encodes = ("relaxation_dynamics",)

    def __init__(
        self,
        steady: BaseTireModel,
        sigma_from_steady: bool = False,
        v_eps: float = V_EPS,
    ):
        super().__init__()
        self.steady = steady
        self.v_eps = float(v_eps)
        #: When the steady model already predicts ``sigma_x/sigma_y`` (ParameterTireNet
        #: does), reuse them instead of learning a second, inconsistent set.
        self.sigma_from_steady = bool(sigma_from_steady)
        if not sigma_from_steady:
            self.sigma_x = BoundedParameter(SIGMA_SPECS[0])
            self.sigma_y = BoundedParameter(SIGMA_SPECS[1])

    # -- physics ------------------------------------------------------------

    def relaxation_lengths(self, params: dict) -> tuple[Tensor, Tensor]:
        if self.sigma_from_steady:
            if "sigma_x" not in params:
                raise ValueError("sigma_from_steady=True but the steady model exposes no sigma")
            return params["sigma_x"], params["sigma_y"]
        return self.sigma_x(), self.sigma_y()

    def time_constants(self, vx: Tensor, params: dict) -> tuple[Tensor, Tensor]:
        """``tau = sigma / (abs(vx) + eps)`` — strictly positive by construction."""
        sx, sy = self.relaxation_lengths(params)
        v = vx.abs() + self.v_eps
        return sx / v, sy / v

    def rates(self, F: Tensor, alpha, kappa, Fz, vx, context=None) -> tuple[Tensor, dict]:
        """``dF/dt`` for the stacked state ``F = [Fx, Fy]`` (last dim = 2)."""
        ss = self.steady(alpha, kappa, Fz, context)
        tau_x, tau_y = self.time_constants(vx, ss.params)
        dFx = (ss.Fx - F[..., 0]) / tau_x
        dFy = (ss.Fy - F[..., 1]) / tau_y
        return torch.stack([dFx, dFy], dim=-1), {"Fx_ss": ss.Fx, "Fy_ss": ss.Fy,
                                                 "tau_x": tau_x, "tau_y": tau_y}

    # -- interfaces ---------------------------------------------------------

    def forward(self, alpha, kappa, Fz, context=None) -> TireForces:
        """Steady-state call, so the cell is drop-in compatible with the static models."""
        return self.steady(alpha, kappa, Fz, context)

    def step(self, F, alpha, kappa, Fz, vx, dt, context=None, method: str = "rk4") -> Tensor:
        """Advance the force state by ``dt``.

        ``method``:

        * ``"exact"`` — ``F_ss + (F - F_ss) exp(-dt/tau)``. Exact for a zero-order-hold
          input, unconditionally stable for any ``dt``, and the right default when the
          data is sampled coarsely relative to ``tau``.
        * ``"rk4"``   — classical 4th order, zero-order hold on the inputs within a step.
        * ``"euler"`` — explicit Euler; requires ``dt < 2 tau`` for stability, which the
          trainer checks (``check_step_size``).
        """
        ss = self.steady(alpha, kappa, Fz, context)
        F_ss = torch.stack([ss.Fx, ss.Fy], dim=-1)
        tau = torch.stack(self.time_constants(vx, ss.params), dim=-1)

        if method == "exact":
            return F_ss + (F - F_ss) * torch.exp(-dt / tau)
        if method == "euler":
            return F + dt * (F_ss - F) / tau
        if method == "rk4":
            f = lambda x: (F_ss - x) / tau
            k1 = f(F)
            k2 = f(F + 0.5 * dt * k1)
            k3 = f(F + 0.5 * dt * k2)
            k4 = f(F + dt * k3)
            return F + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        raise ValueError(f"unknown integration method {method!r}")

    def rollout(
        self,
        alpha: Tensor,
        kappa: Tensor,
        Fz: Tensor,
        vx: Tensor,
        dt: float | Tensor,
        context: dict | None = None,
        F0: Tensor | None = None,
        method: str = "rk4",
    ) -> Tensor:
        """Integrate over a sequence. Inputs ``(B, T)`` -> forces ``(B, T, 2)``.

        The initial force defaults to the steady-state value at the first sample,
        which is the correct assumption for a rig test that starts from equilibrium;
        pass ``F0`` explicitly when it is not.
        """
        if method == "odeint":
            return self._rollout_odeint(alpha, kappa, Fz, vx, dt, context, F0)

        T = alpha.shape[-1]
        ctx_t = lambda t: ({k: v[..., t] for k, v in context.items()} if context else None)
        if F0 is None:
            ss0 = self.steady(alpha[..., 0], kappa[..., 0], Fz[..., 0], ctx_t(0))
            F = torch.stack([ss0.Fx, ss0.Fy], dim=-1)
        else:
            F = F0
        out = [F]
        for t in range(T - 1):
            F = self.step(F, alpha[..., t], kappa[..., t], Fz[..., t], vx[..., t],
                          dt, ctx_t(t), method)
            out.append(F)
        return torch.stack(out, dim=-2)

    def _rollout_odeint(self, alpha, kappa, Fz, vx, dt, context, F0):
        """Optional ``torchdiffeq`` path with linear input interpolation.

        Imported lazily: the package is optional (PLAN.md §1) and the fixed-step
        integrators above are the default.
        """
        try:
            from torchdiffeq import odeint
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "method='odeint' needs torchdiffeq: pip install -e '.[ode]'. "
                "The fixed-step 'rk4'/'exact' integrators need no extra dependency."
            ) from exc

        T = alpha.shape[-1]
        t_grid = torch.arange(T, dtype=alpha.dtype, device=alpha.device) * float(dt)

        def interp(seq: Tensor, t: Tensor) -> Tensor:
            pos = torch.clamp(t / float(dt), 0.0, T - 1 - 1e-6)
            i0 = pos.floor().long()
            w = pos - i0.to(pos.dtype)
            return seq[..., i0] * (1 - w) + seq[..., i0 + 1] * w

        def func(t, F):
            ctx = {k: interp(v, t) for k, v in context.items()} if context else None
            return self.rates(F, interp(alpha, t), interp(kappa, t), interp(Fz, t), interp(vx, t), ctx)[0]

        if F0 is None:
            ctx0 = {k: v[..., 0] for k, v in context.items()} if context else None
            ss0 = self.steady(alpha[..., 0], kappa[..., 0], Fz[..., 0], ctx0)
            F0 = torch.stack([ss0.Fx, ss0.Fy], dim=-1)
        return odeint(func, F0, t_grid).movedim(0, -2)

    def check_step_size(self, dt: float, vx_max: float) -> None:
        """Raise if explicit Euler would be unstable at this sample rate.

        ``dt < 2 tau_min`` is the stability limit of explicit Euler on ``dF/dt = -F/tau``;
        ``tau`` is smallest at the highest speed, so that is the case to check.
        """
        with torch.no_grad():
            sx, sy = self.relaxation_lengths({})
            tau_min = float(min(sx.detach(), sy.detach())) / (abs(vx_max) + self.v_eps)
        if dt >= 2 * tau_min:
            raise ValueError(
                f"dt={dt:g}s is not stable for explicit Euler: tau_min={tau_min:.4g}s at "
                f"vx={vx_max:g} m/s requires dt < {2 * tau_min:.4g}s. Use method='exact' or 'rk4'."
            )
