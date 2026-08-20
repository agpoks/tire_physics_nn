"""Common model interface and context handling (PLAN.md §2.2)."""

from __future__ import annotations

import torch
from torch import Tensor
from torch import nn

from tire_nn.layers.bounded_parameters import BoundedParameterHead, ParamSpec
from tire_nn.types import TireForces

__all__ = ["CONTEXT_KEYS", "BaseTireModel", "ContextEncoder", "MU_SPECS", "mu_head"]

#: Documented context vocabulary. Models declare which keys they consume; unknown keys
#: are ignored, and declared-but-absent keys are handled explicitly by ContextEncoder.
CONTEXT_KEYS = ("vx", "Ts", "Tc", "p", "gamma", "mu_est", "wear", "graining")

#: Default normalisation for context keys: (offset, scale) in SI units.
CONTEXT_NORM = {
    "vx": (20.0, 20.0),
    "Ts": (330.0, 40.0),
    "Tc": (330.0, 40.0),
    "p": (2.0e5, 1.0e5),
    "gamma": (0.0, 0.05),
    "mu_est": (1.0, 0.3),
    "wear": (0.0, 1.0),
    "graining": (0.0, 1.0),
}

MU_SPECS = (
    ParamSpec("mu_x", lo=0.05, hi=2.5, init=1.0),
    ParamSpec("mu_y", lo=0.05, hi=2.5, init=1.0),
)


def mu_head(in_features: int) -> BoundedParameterHead:
    """Head producing the friction-ellipse semi-axes, bounded by construction."""
    return BoundedParameterHead(in_features, MU_SPECS)


class ContextEncoder(nn.Module):
    """Encodes the optional context dict into a fixed-width feature vector.

    Missing keys are **not** silently zero-filled. Each declared key contributes
    ``(value, present_flag)``, and when a key is absent its value slot is filled by a
    learned per-key constant. This matters because the six target datasets expose
    different subsets (VeTyT has camber and pressure but no temperature; KIT has no
    camber; Q-Motion is pressure-centric, PLAN.md §4.4) — a fixed input vector would
    force fabricated values into the model and the network could not tell a real
    measurement of 0 from a missing one.

    ``n_tires > 0`` adds a learned embedding indexed by ``context["tire_id"]``, which
    is how tire-set / compound / mass-change experiments are represented.
    """

    def __init__(self, keys: tuple[str, ...] = (), n_tires: int = 0, embed_dim: int = 4):
        super().__init__()
        unknown = [k for k in keys if k not in CONTEXT_KEYS]
        if unknown:
            raise ValueError(f"unknown context keys {unknown}; vocabulary is {CONTEXT_KEYS}")
        self.keys = tuple(keys)
        self.n_tires = int(n_tires)
        self.embed_dim = int(embed_dim) if n_tires > 0 else 0
        if self.keys:
            self.fill = nn.Parameter(torch.zeros(len(self.keys)))
            off = torch.tensor([CONTEXT_NORM[k][0] for k in self.keys])
            scale = torch.tensor([CONTEXT_NORM[k][1] for k in self.keys])
            self.register_buffer("offset", off)
            self.register_buffer("scale", scale)
        if n_tires > 0:
            self.embedding = nn.Embedding(n_tires, self.embed_dim)
            nn.init.zeros_(self.embedding.weight)

    @property
    def out_dim(self) -> int:
        return 2 * len(self.keys) + self.embed_dim

    def forward(self, context: dict[str, Tensor] | None, like: Tensor) -> Tensor | None:
        if self.out_dim == 0:
            return None
        context = context or {}
        feats: list[Tensor] = []
        for i, key in enumerate(self.keys):
            value = context.get(key)
            if value is None:
                feats.append(self.fill[i].expand_as(like))
                feats.append(torch.zeros_like(like))
            else:
                feats.append((value - self.offset[i]) / self.scale[i])
                feats.append(torch.ones_like(like))
        out = torch.stack(feats, dim=-1)
        if self.embed_dim:
            tire_id = context.get("tire_id")
            if tire_id is None:
                tire_id = torch.zeros_like(like, dtype=torch.long)
            out = torch.cat([out, self.embedding(tire_id.long())], dim=-1)
        return out

    def extra_repr(self) -> str:
        return f"keys={self.keys}, n_tires={self.n_tires}, out_dim={self.out_dim}"


class BaseTireModel(nn.Module):
    """Interface shared by every tire model in this project.

        forward(alpha, kappa, Fz, context) -> TireForces

    ``alpha`` [rad], ``kappa`` [-], ``Fz`` [N] are broadcastable tensors of identical
    shape; ``context`` is the optional dict described in PLAN.md §2.2. Sign convention
    is SAE (PLAN.md §4.1) throughout.
    """

    encodes: tuple[str, ...] = ()      # human-readable list of encoded priors

    def forward(self, alpha: Tensor, kappa: Tensor, Fz: Tensor, context=None) -> TireForces:
        raise NotImplementedError

    def rollout(self, alpha, kappa, Fz, vx, dt, context=None, F0=None, method=None) -> Tensor:
        """Quasi-static rollout: evaluate the steady-state law at every sample.

        This *is* the "static TireNet" baseline of Experiment 2 — a model with no
        transient dynamics applied to transient data. Subclasses with real dynamics
        (``RelaxationTireCell``, and the sequence baselines) override it.
        """
        out = self(alpha, kappa, Fz, context)
        return torch.stack([out.Fx, out.Fy], dim=-1)

    def describe(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return f"{type(self).__name__}(params={n}, encodes={self.encodes or ('none',)})"
