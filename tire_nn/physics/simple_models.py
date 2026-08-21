"""Linear and Dugoff tire models — the two remaining analytical baselines.

Together with :py:mod:`tire_nn.physics.brush` and :py:mod:`tire_nn.physics.pacejka`
these give the four classical steady-state laws the documentation compares:

``linear``
    Parameters ``C_alpha``, ``C_kappa``. Exact near zero slip, unbounded — unusable at
    the limit.
``brush``
    Parameters ``C``, ``mu``. Physically derived; saturates at ``mu*Fz``; no post-peak
    decay.
``Dugoff``
    Parameters ``C_alpha``, ``C_kappa``, ``mu``. Closed-form combined slip, cheap; kink
    at the sliding boundary.
``Magic Formula``
    Parameters ``B, C, D, E``. Fits any measured shape including post-peak decay, but
    empirical.

All are differentiable and carry no learnable parameters.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["linear_tire", "dugoff_tire"]

EPS = 1e-9


def linear_tire(
    alpha: Tensor,
    kappa: Tensor,
    Fz: Tensor,
    C_alpha: Tensor | float,
    C_kappa: Tensor | float,
) -> tuple[Tensor, Tensor]:
    """Linear tire: ``Fx = C_kappa kappa``, ``Fy = -C_alpha alpha``.

    The tangent of every other model at zero slip, and the model implicitly assumed by
    any controller linearised about straight-line running. It is exact where it is
    valid and catastrophically wrong outside: force grows without bound, so a planner
    using it will happily ask for 3 g of lateral acceleration.
    """
    return C_kappa * kappa, -C_alpha * alpha


def dugoff_tire(
    alpha: Tensor,
    kappa: Tensor,
    Fz: Tensor,
    C_alpha: Tensor | float,
    C_kappa: Tensor | float,
    mu: Tensor | float,
    eps_r: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Dugoff model — closed-form combined slip with a single friction bound.

    With ``s = kappa/(1+kappa)`` and ``t = tan(alpha)/(1+kappa)`` the unsaturated forces
    are ``Fx0 = C_kappa s`` and ``Fy0 = -C_alpha t``. The boundary value

    .. math::

        \\lambda = \\frac{\\mu F_z (1 + \\kappa)}{2\\sqrt{(C_\\kappa \\kappa)^2 + (C_\\alpha \\tan\\alpha)^2}}

    decides whether the contact patch is fully adhering (``lambda >= 1``) or partly
    sliding (``lambda < 1``), and the forces are scaled by

    .. math:: f(\\lambda) = \\begin{cases} 1 & \\lambda \\ge 1 \\\\ \\lambda(2-\\lambda) & \\lambda < 1\\end{cases}

    Cheaper than the Magic Formula and physically bounded, but ``f`` is only
    :math:`C^1` at ``lambda = 1`` and the model cannot represent the post-peak decay of
    a real tire — its force is monotone up to the limit and then flat.

    ``eps_r`` is the optional velocity-dependent friction reduction of the original
    formulation; left at 0 here since the project models friction changes through
    ``mu`` and the condition states instead.
    """
    denom = 1.0 + torch.clamp(kappa, min=-0.99)
    s = kappa / denom
    t = torch.tan(alpha) / denom

    Fx0 = C_kappa * s
    Fy0 = -C_alpha * t

    # The lambda denominator uses the *un-normalised* stiffness-weighted slip
    # sqrt((C_kappa kappa)^2 + (C_alpha tan alpha)^2), NOT the already-divided forces:
    # dividing twice by (1 + kappa) makes lambda too large and the model under-saturates
    # (it then exceeds mu*Fz under combined slip, which defeats the point of the model).
    magnitude = torch.sqrt((C_kappa * kappa) ** 2 + (C_alpha * torch.tan(alpha)) ** 2 + EPS)
    lam = (mu * Fz * denom) / (2.0 * magnitude)
    scale = torch.where(lam >= 1.0, torch.ones_like(lam), lam * (2.0 - lam))
    if eps_r:
        scale = scale * (1.0 - eps_r)
    return Fx0 * scale, Fy0 * scale
