"""Shared, dependency-free data types (PLAN.md §2.2)."""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import Tensor

__all__ = ["TireForces"]


@dataclass
class TireForces:
    """Output of every tire model.

    Supports tuple unpacking (``Fx, Fy = model(...)``) so the analytical baselines and
    the learned models are drop-in interchangeable.

    ``params`` carries the physically meaningful quantities a model chooses to expose
    (``mu_x``, ``mu_y``, ``B``, ``C``, ``E``, ``sigma``, ...). It is always populated by
    ``ParameterTireNet``, partially by the encoded models (``mu`` at minimum) and may be
    empty for the plain MLP — which is exactly the interpretability gradient the
    ablation ladder is meant to expose.
    """

    Fx: Tensor
    Fy: Tensor
    Mz: Tensor | None = None
    params: dict[str, Tensor] = field(default_factory=dict)

    def __iter__(self):
        yield self.Fx
        yield self.Fy
