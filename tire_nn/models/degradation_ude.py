"""Tyre degradation from stint data — a universal differential equation.

The problem: tyre condition is latent. What is observed, lap after lap, is its
consequence — the car gets slower. This is the same shape of problem as the IMU-only
vehicle experiment: identify a hidden state's *dynamics* from an aggregate, confounded
measurement of its *effect*.

A universal differential equation {cite}`rackauckas2020ude` is the natural formulation,
because the split between known and unknown is unusually clean.

**Known** — the observation is additive and its non-tyre terms are understood:

.. math::

    t_{\\text{lap}} = t_{\\text{ref}}(\\text{compound})
        + c_{\\text{fuel}}\\,\\phi_{\\text{fuel}}
        + \\underbrace{w}_{\\text{wear}} + \\underbrace{a_g\\,g}_{\\text{graining}}

A full fuel tank costs a few seconds a lap and burns off linearly; a pit stop resets the
tyre state to zero; wear is monotone; graining is a bounded fraction of the surface.

**Unknown** — the *kinetics*: how fast wear and graining accumulate as a function of
compound, track temperature and tyre age. That is what the networks learn.

.. math::

    \\frac{\\mathrm{d}w}{\\mathrm{d}\\lambda} = \\mathrm{softplus}\\big(f_w(u)\\big) \\ \\ge 0,
    \\qquad
    \\frac{\\mathrm{d}g}{\\mathrm{d}\\lambda} = (1-g)R_{\\text{form}}(u) - g R_{\\text{clean}}(u)

Identifiability note
--------------------
Wear amplitude and wear rate are **not** separately identifiable from lap time alone:
only their product enters the observation, and $w$ never saturates. The state is
therefore defined directly in **seconds of lap time**, i.e. the wear amplitude is fixed
at 1 by definition and the learned quantity is the rate. Graining is different — it
saturates at $g = 1$ — so its amplitude $a_g$ *is* identifiable, and is learned.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F

from tire_nn.layers.bounded_parameters import BoundedParameter, ParamSpec
from tire_nn.layers.symmetry import mlp
from tire_nn.physics.wear import graining_rate, wear_rate

__all__ = ["LapDegradationUDE", "LinearDegradationModel", "BlackBoxDegradationModel",
           "DEGRADATION_SPECS"]

DEGRADATION_SPECS = (
    #: seconds of lap time lost per unit of remaining fuel fraction (a full tank)
    ParamSpec("c_fuel", lo=0.0, hi=8.0, init=3.0),
    #: seconds of lap time lost at fully developed graining (g = 1)
    ParamSpec("a_grain", lo=0.0, hi=5.0, init=0.8),
    #: maximum wear rate, in seconds of lap time lost per lap. A racing tyre loses
    #: order 0.01-0.1 s per lap; anything above ~0.3 is not degradation but the model
    #: absorbing pace or fuel effects into the wear channel. Bounding it is the same
    #: principle the tire models apply to mu and B (see the bounded-parameter layer):
    #: keep the parameter physical at every step rather than hoping the fit stays sane.
    ParamSpec("wear_rate_max", lo=0.0, hi=0.30, init=0.05),
)


def _features(tyre_age: Tensor, track_temp: Tensor, air_temp: Tensor,
              compound_emb: Tensor) -> Tensor:
    """Physically scaled, monotone gating features for the rate networks.

    ``age`` lets the rate depend on how far into the stint the tyre is; the two
    temperature channels are split into "how far below" and "how far above" a reference,
    so the *sign* of the thermal effect can be imposed while its shape stays learnable.
    """
    ref = 310.0
    cold = F.relu(ref - track_temp) / 10.0
    hot = F.relu(track_temp - ref) / 10.0
    return torch.cat([
        (tyre_age / 10.0).unsqueeze(-1),
        cold.unsqueeze(-1),
        hot.unsqueeze(-1),
        ((air_temp - 300.0) / 10.0).unsqueeze(-1),
        compound_emb,
    ], dim=-1)


class LapDegradationUDE(nn.Module):
    """UDE for tyre degradation observed through lap time.

    Args:
        n_compounds: number of distinct compounds in the dataset.
        embed_dim: width of the learned compound embedding fed to the rate networks.
        hidden: hidden sizes of the rate networks (kept small — these are kinetics, not
            a general function approximator).
        enable_graining: set False for a wear-only ablation.
        n_pace_groups: number of (session, driver) groups. Each gets a free additive
            offset absorbing car pace, circuit and conditions. Without it the model
            fits the pace difference between cars — which is far larger than the
            degradation signal — and reports nonsense kinetics.
    """

    def __init__(
        self,
        n_compounds: int,
        embed_dim: int = 3,
        hidden: tuple[int, ...] = (16, 16),
        enable_graining: bool = True,
        base_lap_time: float = 90.0,
        n_pace_groups: int = 0,
    ):
        super().__init__()
        self.n_compounds = int(n_compounds)
        self.enable_graining = bool(enable_graining)
        self.embedding = nn.Embedding(n_compounds, embed_dim)
        nn.init.zeros_(self.embedding.weight)

        # Per-compound reference pace: the lap time of a fresh tyre with no fuel penalty.
        self.t_ref = nn.Parameter(torch.full((n_compounds,), float(base_lap_time)))
        # Free offset per (session, driver): car pace, circuit, conditions.
        self.pace = nn.Parameter(torch.zeros(max(n_pace_groups, 1)))
        self.n_pace_groups = int(n_pace_groups)

        in_dim = 4 + embed_dim
        self.wear_net = mlp(in_dim, 1, hidden)
        self.form_net = mlp(in_dim, 1, hidden)
        self.clean_net = mlp(in_dim, 1, hidden)
        # Start with almost no graining, so it has to be *earned* from the data. With a
        # default (zero-bias) init the formation rate starts at softplus(0) = 0.69 per
        # lap, graining saturates within two laps, the resulting near-constant offset is
        # absorbed by the pace term, and the wear channel receives no gradient — the
        # model then reports zero degradation of either kind. That is an optimisation
        # artefact, not a finding, and this init removes it.
        nn.init.constant_(self.form_net[-1].bias, -3.0)      # softplus(-3) ~ 0.05 / lap
        nn.init.constant_(self.clean_net[-1].bias, -1.0)     # softplus(-1) ~ 0.31 / lap
        for spec in DEGRADATION_SPECS:
            setattr(self, spec.name, BoundedParameter(spec))

    # -- kinetics -----------------------------------------------------------

    def rates(self, u: Tensor, g: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(dw, R_form, R_clean)``, all non-negative by construction.

        The wear rate is bounded above as well as below: ``sigmoid`` into
        ``[0, wear_rate_max]`` rather than an unbounded ``softplus``. Without the upper
        bound the network drove the wear channel to ~9 s over a stint on real F1 data —
        physically absurd, compensated in-sample by the pace offsets, and badly wrong
        out-of-sample.
        """
        dw = self.wear_rate_max() * torch.sigmoid(self.wear_net(u).squeeze(-1))
        if not self.enable_graining:
            zero = torch.zeros_like(dw)
            return dw, zero, zero
        R_form = F.softplus(self.form_net(u).squeeze(-1))
        R_clean = F.softplus(self.clean_net(u).squeeze(-1))
        return dw, R_form, R_clean

    def graining_step(self, g: Tensor, R_form: Tensor, R_clean: Tensor) -> Tensor:
        """Exact zero-order-hold update over one lap — stays in ``[0, 1]`` for any rate."""
        total = R_form + R_clean
        g_inf = R_form / torch.clamp(total, min=1e-12)
        decay = torch.exp(-torch.clamp(total, min=0.0))
        return torch.where(total > 1e-12, g_inf + (g - g_inf) * decay, g)

    # -- rollout ------------------------------------------------------------

    def forward(self, batch: dict) -> dict:
        """Roll a whole stint out lap by lap. Tensors are ``(n_stints, max_len)``."""
        age = batch["tyre_age"]
        n, T = age.shape
        emb = self.embedding(batch["compound"])                     # (n, embed)
        t_ref = self.t_ref[batch["compound"]].unsqueeze(-1)          # (n, 1)

        w = torch.zeros(n, device=age.device, dtype=age.dtype)
        g = torch.zeros_like(w)
        wear_hist, grain_hist = [], []

        for k in range(T):
            wear_hist.append(w)
            grain_hist.append(g)
            u = _features(age[:, k], batch["track_temp"][:, k],
                          batch["air_temp"][:, k], emb)
            dw, R_form, R_clean = self.rates(u, g)
            w = w + dw                                               # monotone by construction
            if self.enable_graining:
                g = self.graining_step(g, R_form, R_clean)

        wear = torch.stack(wear_hist, dim=-1)
        graining = torch.stack(grain_hist, dim=-1)
        lap_time = (t_ref
                    + self._pace_offset(batch)
                    + self.c_fuel() * batch["fuel_frac"]
                    + wear
                    + self.a_grain() * graining)
        return {"lap_time": lap_time, "wear": wear, "graining": graining}

    def _pace_offset(self, batch: dict) -> Tensor:
        if not self.n_pace_groups or "pace_group" not in batch:
            return torch.zeros(1, 1, device=self.pace.device, dtype=self.pace.dtype)
        return self.pace[batch["pace_group"]].unsqueeze(-1)

    def describe(self) -> str:
        n = sum(p.numel() for p in self.parameters())
        return (f"LapDegradationUDE(params={n}, c_fuel={float(self.c_fuel().detach()):.2f}s, "
                f"a_grain={float(self.a_grain().detach()):.2f}s, graining={self.enable_graining})")


class LinearDegradationModel(nn.Module):
    """The practitioner's baseline: lap time linear in tyre age.

    .. math:: t = t_{\\text{ref}}(\\text{compound}) + c_{\\text{fuel}}\\phi
              + k(\\text{compound})\\,\\lambda_{\\text{age}}

    This is what most public race-strategy models use, and it is a strong baseline:
    wear really is close to linear in tyre age. What it cannot represent is anything
    *non-monotone* in age — such as graining, which forms early and then cleans up.
    """

    def __init__(self, n_compounds: int, base_lap_time: float = 90.0,
                 n_pace_groups: int = 0):
        super().__init__()
        self.t_ref = nn.Parameter(torch.full((n_compounds,), float(base_lap_time)))
        self.k_deg = nn.Parameter(torch.full((n_compounds,), 0.03))
        self.c_fuel = BoundedParameter(DEGRADATION_SPECS[0])
        self.pace = nn.Parameter(torch.zeros(max(n_pace_groups, 1)))
        self.n_pace_groups = int(n_pace_groups)

    def forward(self, batch: dict) -> dict:
        t_ref = self.t_ref[batch["compound"]].unsqueeze(-1)
        k = self.k_deg[batch["compound"]].unsqueeze(-1)
        offset = (self.pace[batch["pace_group"]].unsqueeze(-1)
                  if self.n_pace_groups and "pace_group" in batch else 0.0)
        lap_time = t_ref + offset + self.c_fuel() * batch["fuel_frac"] + k * batch["tyre_age"]
        return {"lap_time": lap_time,
                "wear": k * batch["tyre_age"],
                "graining": torch.zeros_like(lap_time)}


class BlackBoxDegradationModel(nn.Module):
    """Black-box control: an MLP from lap features straight to lap time.

    No state, no dynamics, no structure — it sees ``(age, temps, fuel, compound)`` and
    predicts the lap time. It can represent a non-monotone age dependence, unlike the
    linear model, but it has no notion of an accumulating state, so nothing constrains
    it to be consistent across a stint, and it cannot report a tyre condition at all.
    """

    def __init__(self, n_compounds: int, embed_dim: int = 3, hidden=(32, 32),
                 base_lap_time: float = 90.0, n_pace_groups: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(n_compounds, embed_dim)
        nn.init.zeros_(self.embedding.weight)
        self.net = mlp(5 + embed_dim, 1, hidden)
        self.base = float(base_lap_time)
        self.pace = nn.Parameter(torch.zeros(max(n_pace_groups, 1)))
        self.n_pace_groups = int(n_pace_groups)

    def forward(self, batch: dict) -> dict:
        age = batch["tyre_age"]
        emb = self.embedding(batch["compound"]).unsqueeze(1).expand(-1, age.shape[1], -1)
        x = torch.stack([age / 10.0,
                         (batch["track_temp"] - 310.0) / 10.0,
                         (batch["air_temp"] - 300.0) / 10.0,
                         batch["fuel_frac"],
                         torch.ones_like(age)], dim=-1)
        offset = (self.pace[batch["pace_group"]].unsqueeze(-1)
                  if self.n_pace_groups and "pace_group" in batch else 0.0)
        lap_time = self.base + offset + self.net(torch.cat([x, emb], dim=-1)).squeeze(-1)
        return {"lap_time": lap_time,
                "wear": torch.zeros_like(lap_time),
                "graining": torch.zeros_like(lap_time)}
