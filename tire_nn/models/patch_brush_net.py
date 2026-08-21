r"""Learned contact-patch brush model — a physics-encoded PDE/quadrature layer.

The classical brush model assumes a **parabolic** pressure distribution and a uniform
bristle stiffness. Real contact patches are neither: load, inflation pressure, camber
and wear all reshape them, and a worn or over-inflated tire can carry a visibly
non-parabolic profile. This model keeps every equation of
:py:mod:`tire_nn.physics.brush_patch` and learns only the things that are genuinely
unknown — the *shape* of the pressure distribution and the material parameters.

What is encoded, and how
------------------------

:math:`p_i > 0`
    ``softmax`` over the patch elements.
:math:`\sum_i p_i \Delta\xi = F_z`
    the same ``softmax`` — it sums to one by definition.
:math:`\|\tau_i\| \le \mu p_i`
    elementwise ``min`` against the friction bound.
force opposes slip
    the shear is built along :math:`-\boldsymbol\sigma`.
:math:`F(0) = 0` and odd symmetry
    zero slip gives zero deflection, hence zero shear, and the construction is odd in
    the slip vector.
:math:`\mu`, :math:`k_b`, :math:`a` physical
    bounded parameter transforms.

The load-balance constraint is worth dwelling on. Written as a penalty
:math:`\\lambda(\\sum p_i \\Delta\\xi - F_z)^2` it would be satisfied only on average, and a
model that quietly invents or destroys vertical load can fit almost anything. As a
``softmax`` it is exact for every weight vector, and costs one line.

This is the PDE/graph analogue of the whole project's thesis: discretise the governing
equation, keep its structure, and learn only the closure.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch import nn

from tire_nn.layers.bounded_parameters import BoundedParameter, ParamSpec
from tire_nn.layers.symmetry import mlp
from tire_nn.models.base import BaseTireModel, ContextEncoder
from tire_nn.physics.brush_patch import patch_forces, pressure_from_logits, patch_coordinates
from tire_nn.types import TireForces

__all__ = ["PatchBrushNet", "PATCH_SPECS"]

PATCH_SPECS = (
    #: peak friction coefficient
    ParamSpec("mu", lo=0.05, hi=2.5, init=1.0),
    #: bristle line stiffness [N/m per m of deflection], scaled by ``stiffness_ref``
    ParamSpec("stiffness_scale", lo=0.1, hi=20.0, init=1.0),
    #: contact half-length at the reference load [m]
    ParamSpec("half_length", lo=0.01, hi=0.20, init=0.06),
    #: exponent of the load dependence of patch length, a ~ Fz^q; q = 1/3 for a
    #: Hertzian contact, 1/2 is often measured on tires, so the range brackets both
    ParamSpec("length_exponent", lo=0.0, hi=1.0, init=0.4),
)


class PatchBrushNet(BaseTireModel):
    """Brush model over a discretised contact patch, with a learned pressure profile.

    Args:
        n_elements: number of patch elements (chain nodes). 32-64 is plenty; the
            quadrature converges at second order.
        learn_pressure: if False the classical parabolic profile is used and only the
            scalar parameters are learned — a useful ablation, since it isolates what
            the pressure freedom is worth.
        symmetric_pressure: constrain the profile to be symmetric about the patch
            centre. Real profiles are slightly asymmetric under braking/traction, so
            this is off by default, but it halves the parameters when data is scarce.
    """

    encodes = ("brush_pde", "load_balance", "friction_bound", "odd_symmetry",
               "bounded_parameters")

    def __init__(
        self,
        n_elements: int = 48,
        context_keys: tuple[str, ...] = (),
        n_tires: int = 0,
        hidden: tuple[int, ...] = (32, 32),
        learn_pressure: bool = True,
        symmetric_pressure: bool = False,
        Fz_ref: float = 1000.0,
        stiffness_ref: float = 6.0e6,
    ):
        # Defaults chosen so an untrained model is already a plausible tire: with
        # a = 60 mm and k_b = 6e6 N/m^2 the cornering stiffness C_alpha = 2 a^2 k_b is
        # about 43 kN/rad at the 1 kN reference load, the right order for a small
        # racing tire. (Line quantities throughout — no width factor.)
        super().__init__()
        self.n_elements = int(n_elements)
        self.learn_pressure = bool(learn_pressure)
        self.symmetric_pressure = bool(symmetric_pressure)
        self.Fz_ref = float(Fz_ref)
        self.stiffness_ref = float(stiffness_ref)

        self.context = ContextEncoder(context_keys, n_tires)
        for spec in PATCH_SPECS:
            setattr(self, spec.name, BoundedParameter(spec))

        if learn_pressure:
            out_dim = (n_elements + 1) // 2 if symmetric_pressure else n_elements
            self.pressure_net = mlp(1 + self.context.out_dim, out_dim, hidden)
            # Start from a near-uniform profile and let the data reshape it; a
            # zero-init last layer means the initial softmax is exactly uniform.
            nn.init.zeros_(self.pressure_net[-1].weight)
            nn.init.zeros_(self.pressure_net[-1].bias)

    # -- geometry and material ---------------------------------------------

    def contact_half_length(self, Fz: Tensor) -> Tensor:
        """Patch length grows with load: ``a = a_ref (Fz/Fz_ref)^q``.

        A longer patch at higher load is why cornering stiffness rises with load while
        peak friction falls — encoding it means the model gets that trade-off right
        without seeing a load sweep.
        """
        ratio = torch.clamp(Fz / self.Fz_ref, min=1e-3)
        return self.half_length() * ratio.pow(self.length_exponent())

    def pressure_profile(self, Fz: Tensor, context=None) -> Tensor:
        """Positive pressure profile integrating to ``Fz``, for any weights."""
        a = self.contact_half_length(Fz)
        if not self.learn_pressure:
            xi, _ = patch_coordinates(self.n_elements, a, device=Fz.device, dtype=Fz.dtype)
            from tire_nn.physics.brush_patch import parabolic_pressure
            return parabolic_pressure(xi, a, Fz)

        c = self.context(context, Fz)
        x = (Fz / self.Fz_ref).unsqueeze(-1)
        if c is not None:
            x = torch.cat([x, c], dim=-1)
        logits = self.pressure_net(x)
        if self.symmetric_pressure:
            half = logits
            mirror = torch.flip(half, dims=[-1])
            logits = torch.cat([half, mirror], dim=-1)[..., :self.n_elements]
        return pressure_from_logits(logits, a, Fz, self.n_elements)

    # -- interface ----------------------------------------------------------

    def forward(self, alpha: Tensor, kappa: Tensor, Fz: Tensor, context=None) -> TireForces:
        # Theoretical slip, symmetric form (see physics/combined-slip in the docs).
        sigma_x = kappa
        sigma_y = torch.tan(alpha)

        a = self.contact_half_length(Fz)
        pressure = self.pressure_profile(Fz, context)
        stiffness = self.stiffness_scale() * self.stiffness_ref
        mu = self.mu().expand_as(Fz)

        out = patch_forces(sigma_x, sigma_y, Fz, a,
                           stiffness.expand_as(Fz), mu,
                           pressure=pressure, n_elements=self.n_elements)
        return TireForces(
            Fx=out["Fx"], Fy=out["Fy"],
            params={"mu_x": mu, "mu_y": mu,
                    "half_length": a,
                    "sliding_fraction": out["sliding_fraction"],
                    "pressure": out["pressure"],
                    "xi": out["xi"]},
        )

    def describe(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return (f"PatchBrushNet(elements={self.n_elements}, params={n}, "
                f"learn_pressure={self.learn_pressure})")
