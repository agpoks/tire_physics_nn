"""Wear and graining state dynamics — irreversibility and boundedness by construction.

PLAN.md §3, P7.

``wear``     : monotone non-decreasing, unbounded above (a one-way thermodynamic street).
``graining`` : reversible, confined to ``[0, 1]`` *structurally* — at ``g = 0`` the sink
               term vanishes and at ``g = 1`` the source term vanishes, so with
               non-negative rates the interval is an invariant set. No clamping and no
               penalty are needed, and the gradient stays informative at the boundary.

The **rates** may come from small positive networks (``models/thermo_graining_tire.py``);
this module only fixes the structure they are plugged into.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["wear_rate", "graining_rate", "effective_friction"]


def wear_rate(raw: Tensor) -> Tensor:
    """``dwear/dt = softplus(raw) >= 0``. Irreversible for any network output."""
    return torch.nn.functional.softplus(raw)


def graining_rate(g: Tensor, R_form: Tensor, R_clean: Tensor) -> Tensor:
    """``dg/dt = (1-g) R_form - g R_clean`` with ``R_form, R_clean >= 0``.

    Callers must pass non-negative rates (use ``softplus``); the invariance of
    ``[0, 1]`` depends on it.
    """
    return (1.0 - g) * R_form - g * R_clean


def effective_friction(
    mu_base: Tensor,
    wear: Tensor,
    graining: Tensor,
    kw: Tensor | float = 0.1,
    kg: Tensor | float = 0.2,
    mu_min: float = 1e-3,
) -> Tensor:
    """``mu_eff = mu_base * exp(-kw * wear) * (1 - kg * graining)``.

    Multiplicative and strictly positive for ``kg < 1``: tire condition degrades grip
    but can never invert its sign or push it through zero, so the friction envelope
    (P3) stays well posed for every state the condition model can reach.
    """
    factor = torch.exp(-kw * torch.clamp(wear, min=0.0)) * (1.0 - kg * torch.clamp(graining, 0.0, 1.0))
    return torch.clamp(mu_base * factor, min=mu_min)
