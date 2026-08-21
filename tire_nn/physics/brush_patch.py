"""Discretised brush model: the contact patch as a 1-D chain (PDE / graph form).

The closed-form brush model in :py:mod:`tire_nn.physics.brush` is the *integral* of a
local problem. This module keeps that local problem explicit, which is what makes it
extensible: the pressure distribution, the bristle stiffness and the friction
coefficient can vary along the patch, and any of them can be learned.

The physics
-----------
Take a contact patch of length :math:`2a`, with a coordinate :math:`\\xi` running from the
leading edge (:math:`\\xi = 0`) to the trailing edge (:math:`\\xi = 2a`). Tread material
enters at the leading edge undeflected and is carried through the patch, so in the
**adhesion** region a bristle's deflection grows with the distance it has travelled:

.. math:: \\frac{\\mathrm{d}\\boldsymbol\\delta}{\\mathrm{d}\\xi} = \\boldsymbol\\sigma,
          \\qquad \\boldsymbol\\delta(0) = \\mathbf{0}

for a theoretical slip vector :math:`\\boldsymbol\\sigma`. The local shear stress is
:math:`\\boldsymbol\\tau = k_b \\boldsymbol\\delta`, and it cannot exceed what friction can
carry:

.. math:: \\|\\boldsymbol\\tau(\\xi)\\| \\le \\mu\\, p(\\xi)

Where that bound binds, the bristle **slides** and the stress sits exactly on the bound,
still opposing the slip direction. The tire force is the integral over the patch:

.. math:: \\mathbf{F} = w \\int_0^{2a} \\boldsymbol\\tau(\\xi)\\, \\mathrm{d}\\xi

Discretising :math:`\\xi` into ``n_elements`` turns the ODE into a cumulative sum along a
chain and the integral into a quadrature — a 1-D graph in which each element talks only
to its predecessor.

What this buys over the closed form
-----------------------------------
The closed form assumes a parabolic pressure distribution and a uniform bristle
stiffness. Real contact patches are not parabolic — load, inflation pressure, camber and
wear all reshape them — and the closed form cannot represent that. The discretised
version can, while keeping every guarantee: the friction bound is enforced elementwise,
the force still opposes the slip, and the pressure still integrates to :math:`F_z`.

No learnable parameters live here; see
:py:mod:`tire_nn.models.patch_brush_net` for the learned version.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "patch_coordinates",
    "parabolic_pressure",
    "patch_shear",
    "patch_forces",
    "pressure_from_logits",
]

EPS = 1e-9


def patch_coordinates(n_elements: int, half_length: Tensor | float,
                      device=None, dtype=torch.float32) -> tuple[Tensor, Tensor]:
    """Midpoint quadrature nodes along the patch.

    Returns ``(xi, dxi)`` with ``xi`` of shape ``(..., n)`` running from the leading to
    the trailing edge, and ``dxi`` the element length. Midpoints are used rather than
    endpoints so the quadrature stays second-order and never samples the singular
    leading edge exactly.
    """
    a = torch.as_tensor(half_length, device=device, dtype=dtype)
    index = torch.arange(n_elements, device=a.device, dtype=a.dtype) + 0.5
    dxi = (2.0 * a) / n_elements
    return index * dxi.unsqueeze(-1), dxi


def parabolic_pressure(xi: Tensor, half_length: Tensor, Fz: Tensor) -> Tensor:
    """Classical parabolic line load [N/m], normalised so it integrates to ``Fz``.

    .. math:: p(\\xi) = \\frac{3 F_z}{4a}\\left[1 - \\left(\\frac{\\xi - a}{a}\\right)^2\\right]

    This is the assumption behind the closed-form brush model, and the thing the learned
    version is allowed to depart from.
    """
    a = half_length.unsqueeze(-1)
    shape = 1.0 - ((xi - a) / a) ** 2
    return (3.0 * Fz.unsqueeze(-1) / (4.0 * a)) * torch.clamp(shape, min=0.0)


def pressure_from_logits(logits: Tensor, half_length: Tensor, Fz: Tensor,
                         n_elements: int) -> Tensor:
    """Turn unconstrained logits into a pressure distribution that is positive **and**
    integrates to exactly ``Fz``.

    A ``softmax`` over the elements gives non-negative weights summing to one; dividing
    by the element length converts them to a pressure. So

    .. math:: p_i > 0 \\quad\\text{and}\\quad \\sum_i p_i\\,\\Delta\\xi = F_z

    hold identically, for any logits — the same trick as the bounded parameters
    elsewhere in this project, applied to a whole distribution rather than a scalar.
    Enforcing the load balance with a penalty instead would let the model quietly
    invent or destroy vertical load.
    """
    weights = torch.softmax(logits, dim=-1)
    dxi = (2.0 * half_length) / n_elements
    return weights * Fz.unsqueeze(-1) / dxi.unsqueeze(-1)


def patch_shear(
    xi: Tensor,
    pressure: Tensor,
    sigma_x: Tensor,
    sigma_y: Tensor,
    stiffness: Tensor,
    mu: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Local shear stress along the patch, with the friction bound applied elementwise.

    In adhesion the stress magnitude is ``k_b |sigma| xi``; where that exceeds
    ``mu p(xi)`` the element slides and sits on the bound. Both branches point along
    ``-sigma``, so the shear always opposes the slip.

    Returns ``(tau_x, tau_y, sliding)`` where ``sliding`` is the fraction-of-patch
    indicator, useful for plotting the adhesion/sliding split.
    """
    magnitude = torch.sqrt(sigma_x ** 2 + sigma_y ** 2 + EPS).unsqueeze(-1)
    adhesion = stiffness.unsqueeze(-1) * magnitude * xi
    limit = mu.unsqueeze(-1) * pressure
    tau = torch.minimum(adhesion, limit)
    sliding = (adhesion > limit).to(tau.dtype)

    direction_x = (sigma_x.unsqueeze(-1) / magnitude)
    direction_y = (sigma_y.unsqueeze(-1) / magnitude)
    return tau * direction_x, tau * direction_y, sliding


def patch_forces(
    sigma_x: Tensor,
    sigma_y: Tensor,
    Fz: Tensor,
    half_length: Tensor,
    stiffness: Tensor,
    mu: Tensor,
    pressure: Tensor | None = None,
    n_elements: int = 64,
) -> dict:
    """Integrate the local shear over the patch to get the tire force.

    ``pressure`` may be supplied (e.g. from a learned distribution); if omitted the
    classical parabolic profile is used, in which case this reproduces the closed-form
    brush model to quadrature accuracy.

    Returns a dict with ``Fx``, ``Fy``, the local ``tau_x``/``tau_y``/``pressure``
    arrays and ``sliding_fraction`` — the share of the patch that is sliding, which is
    the physical quantity behind the shape of the force curve.
    """
    xi, dxi = patch_coordinates(n_elements, half_length, device=Fz.device, dtype=Fz.dtype)
    if pressure is None:
        pressure = parabolic_pressure(xi, half_length, Fz)

    tau_x, tau_y, sliding = patch_shear(xi, pressure, sigma_x, sigma_y, stiffness, mu)
    scale = dxi.unsqueeze(-1)      # line quantities: no width factor (see module docstring)

    # SAE signs: the force opposes the slip, so Fy is negative for positive alpha.
    return {
        "Fx": (tau_x * scale).sum(-1),
        "Fy": -(tau_y * scale).sum(-1),
        "tau_x": tau_x,
        "tau_y": tau_y,
        "pressure": pressure,
        "xi": xi,
        "sliding_fraction": sliding.mean(-1),
    }
