"""Differentiable brush tire model (physical baseline, no learnable parameters).

The brush model derives the force from first principles: the tread is a row of
elastic bristles that stick until the local shear stress exceeds ``mu * p(x)``,
after which they slide. With a parabolic pressure distribution this integrates in
closed form and yields, for free, the two properties this project cares about:
odd symmetry and a force magnitude bounded by ``mu * Fz``.

It is the cheapest honest reference for *why* the encoded network is built the way
it is (PLAN.md §3), and a useful initialiser for the residual model.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["brush_forces", "brush_combined"]

EPS = 1e-9


def _brush_scalar(sigma: Tensor, C_stiff: Tensor | float, mu_Fz: Tensor) -> Tensor:
    """Closed-form brush characteristic for a scalar theoretical slip ``sigma >= 0``."""
    theta = C_stiff / (3.0 * torch.clamp(mu_Fz, min=EPS))
    z = theta * sigma
    # Adhesion branch (z < 1) blended into full sliding at z >= 1 by construction.
    full_slide = z >= 1.0
    adhesion = mu_Fz * (3.0 * z - 3.0 * z.pow(2) + z.pow(3))
    return torch.where(full_slide, mu_Fz, adhesion)


def brush_combined(
    alpha: Tensor,
    kappa: Tensor,
    Fz: Tensor,
    C_kappa: Tensor | float,
    C_alpha: Tensor | float,
    mu: Tensor | float,
) -> tuple[Tensor, Tensor]:
    """Combined-slip brush force, SAE signs (see ``physics/pacejka``).

    ``C_kappa``/``C_alpha`` are the longitudinal/cornering stiffnesses [N] and [N/rad].
    Anisotropy is handled by scaling the lateral slip so a single isotropic
    characteristic can be used on the normalised slip vector.
    """
    mu_Fz = mu * Fz
    denom = 1.0 + torch.clamp(kappa, min=-0.99)
    sx = kappa / denom
    sy = torch.tan(alpha) / denom

    C_kappa_t = torch.as_tensor(C_kappa, dtype=Fz.dtype, device=Fz.device)
    C_alpha_t = torch.as_tensor(C_alpha, dtype=Fz.dtype, device=Fz.device)
    # Anisotropic scaling: work in a frame where the characteristic is isotropic.
    ux = C_kappa_t * sx
    uy = C_alpha_t * sy
    u = torch.sqrt(ux * ux + uy * uy + EPS)
    sigma = u / torch.clamp(C_kappa_t, min=EPS)

    F_mag = _brush_scalar(sigma, C_kappa_t, mu_Fz)
    Fx = (ux / u) * F_mag
    Fy = -(uy / u) * F_mag
    return Fx, Fy


def brush_forces(alpha: Tensor, kappa: Tensor, Fz: Tensor, C_kappa, C_alpha, mu):
    """Alias of :func:`brush_combined` kept for symmetry with ``pacejka`` naming."""
    return brush_combined(alpha, kappa, Fz, C_kappa, C_alpha, mu)
