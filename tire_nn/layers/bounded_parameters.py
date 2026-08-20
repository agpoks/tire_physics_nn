"""Bounded parameter transforms (PLAN.md §3, P4).

The network emits unconstrained reals; a fixed monotone map sends them into the
physically valid range. The parameter is then valid **at every training step**, so
optimisation never traverses a region where the tire law is meaningless (``C < 0``
flips the curve, ``D < 0`` is a negative peak force) and where gradients would
therefore be actively misleading. Clipping inside the loss would leave the raw
parameter free and only hide the symptom.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

__all__ = [
    "ParamSpec",
    "DEFAULT_SPECS",
    "to_positive",
    "to_bounded",
    "BoundedParameter",
    "BoundedParameterHead",
]


def to_positive(z: Tensor, minimum: float = 0.0, scale: float = 1.0) -> Tensor:
    """``minimum + scale * softplus(z)`` — never below ``minimum``.

    In exact arithmetic the result is strictly greater than ``minimum``; in float32
    ``softplus`` underflows to 0 for ``z < -87``, so the bound is attained rather than
    approached. The guarantee this project relies on is therefore the **closed**
    range ``[minimum, inf)``, and every ``ParamSpec`` is declared so that its
    endpoints are themselves physically valid (e.g. ``sigma_min = 0.01 m > 0``, so
    ``tau = sigma/v`` stays positive even at saturation).
    """
    return minimum + scale * F.softplus(z)


def to_bounded(z: Tensor, lo: float, hi: float) -> Tensor:
    """``lo + (hi - lo) * sigmoid(z)`` — never outside the closed range ``[lo, hi]``.

    As for :func:`to_positive`, float32 saturation makes the endpoints attainable
    (``sigmoid`` rounds to 0/1 for ``|z| > 89``); the endpoints are chosen to be
    valid values, so attaining them is harmless.
    """
    return lo + (hi - lo) * torch.sigmoid(z)


@dataclass
class ParamSpec:
    """Declared range of one physical parameter.

    ``lo``/``hi`` both given -> sigmoid mapping; ``hi is None`` -> softplus above ``lo``.
    The guaranteed range is **closed**: ``lo <= p <= hi``. Declare ``lo``/``hi`` as values
    that are themselves physically admissible.
    ``init`` is the desired value at raw output 0 and is used to pre-bias the head, so
    an untrained model starts at a sensible tire instead of an arbitrary one.
    """

    name: str
    lo: float
    hi: float | None = None
    init: float | None = None
    scale: float = 1.0

    def transform(self, z: Tensor) -> Tensor:
        if self.hi is None:
            return to_positive(z, self.lo, self.scale)
        return to_bounded(z, self.lo, self.hi)

    def inverse(self, value: float) -> float:
        """Raw value that maps to ``value`` (used to initialise a head's bias)."""
        import math

        if self.hi is None:
            y = (value - self.lo) / self.scale
            y = max(y, 1e-6)
            return math.log(math.expm1(y)) if y < 20 else y
        frac = min(max((value - self.lo) / (self.hi - self.lo), 1e-6), 1 - 1e-6)
        return math.log(frac / (1 - frac))


# Physically motivated default ranges. Sources: Pacejka, *Tire and Vehicle Dynamics*;
# fitted F1TENTH values in On-Track-SysID/params/pacejka_params.yaml (B~7-8, C~1.6-2.1,
# D~0.4-0.7, E~0.4-0.5) which sit comfortably inside these bounds.
DEFAULT_SPECS: tuple[ParamSpec, ...] = (
    ParamSpec("mu", lo=0.05, hi=2.5, init=1.0),
    ParamSpec("B", lo=0.5, hi=40.0, init=10.0),
    ParamSpec("C", lo=0.5, hi=2.5, init=1.6),
    ParamSpec("E", lo=-2.0, hi=1.0, init=0.5),
    ParamSpec("k_mu", lo=0.0, hi=0.5, init=0.05),
    ParamSpec("sigma", lo=0.01, hi=None, init=0.3, scale=1.0),   # relaxation length [m]
)


class BoundedParameter(nn.Module):
    """A single learnable scalar (or vector) constrained to a physical range."""

    def __init__(self, spec: ParamSpec, shape: tuple[int, ...] = ()):
        super().__init__()
        self.spec = spec
        init_raw = spec.inverse(spec.init if spec.init is not None else (spec.lo + 1.0))
        self.raw = nn.Parameter(torch.full(shape, float(init_raw)))

    def forward(self) -> Tensor:
        return self.spec.transform(self.raw)

    def extra_repr(self) -> str:
        s = self.spec
        rng = f"[{s.lo}, {s.hi}]" if s.hi is not None else f"> {s.lo}"
        return f"{s.name} in {rng}"


class BoundedParameterHead(nn.Module):
    """Maps a feature vector to a dict of physically bounded parameters.

    One linear head per parameter, bias-initialised so that a freshly constructed
    model with zero-ish features already predicts ``spec.init``.
    """

    def __init__(self, in_features: int, specs=DEFAULT_SPECS):
        super().__init__()
        self.specs = tuple(specs)
        self.heads = nn.ModuleDict()
        for spec in self.specs:
            lin = nn.Linear(in_features, 1)
            nn.init.zeros_(lin.weight)
            nn.init.constant_(lin.bias, spec.inverse(spec.init if spec.init is not None else spec.lo + 1.0))
            self.heads[spec.name] = lin

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        return {spec.name: spec.transform(self.heads[spec.name](features).squeeze(-1)) for spec in self.specs}
