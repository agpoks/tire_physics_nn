"""Canonical dataset schema, synthetic generator, splits and normalisation (PLAN.md §4).

Every adapter in this subpackage returns a ``pandas.DataFrame`` with the columns
below, in SI units and in the project-wide SAE sign convention. Converting once, at
the adapter boundary, is what allows one training script to run on a tire test bench,
a bicycle-tyre rig and a small-scale race car without per-dataset special cases.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from tire_nn.physics.pacejka import MFParams, pacejka_combined

__all__ = [
    "REQUIRED_COLUMNS",
    "OPTIONAL_COLUMNS",
    "CONTEXT_COLUMNS",
    "validate_schema",
    "flip_sign_convention",
    "make_synthetic",
    "make_synthetic_transient",
    "TireDataset",
    "Normalizer",
    "split_by_group",
    "split_by_condition",
]

REQUIRED_COLUMNS = ("alpha", "kappa", "Fz", "tire_id", "source")
TARGET_COLUMNS = ("Fx", "Fy", "Mz")
OPTIONAL_COLUMNS = ("vx", "p", "gamma", "Ts", "Tc", "T_road", "T_air", "mu_ref", "t", "sequence_id")
#: Columns forwarded into the model ``context`` dict (names match ``models.base.CONTEXT_KEYS``).
CONTEXT_COLUMNS = ("vx", "Ts", "Tc", "p", "gamma", "mu_est", "wear", "graining")


def validate_schema(df: pd.DataFrame, targets: tuple[str, ...] = ("Fy",)) -> pd.DataFrame:
    """Raise if the canonical contract is broken. Cheap insurance against silent unit bugs."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns {missing}; got {list(df.columns)}")
    for t in targets:
        if t not in df.columns:
            raise ValueError(f"missing target column {t!r}")
    if (df["Fz"] <= 0).any():
        raise ValueError("Fz must be strictly positive (compression positive)")
    if df["alpha"].abs().max() > np.pi / 2:
        raise ValueError("alpha out of range — is it in degrees? adapters must convert to rad")
    return df


def flip_sign_convention(df: pd.DataFrame, columns=("alpha", "Fy", "kappa", "Fx")) -> pd.DataFrame:
    """Flip the listed columns. Used by adapters whose source uses the opposite convention.

    Kept as a single helper so the conversion appears in exactly one place; each
    adapter documents which convention its source uses (PLAN.md §4.1).
    """
    out = df.copy()
    for c in columns:
        if c in out.columns:
            out[c] = -out[c]
    return out


def make_synthetic(
    n: int = 4000,
    mu: float = 1.1,
    B: float = 9.0,
    C: float = 1.6,
    E: float = 0.4,
    Fz_range: tuple[float, float] = (400.0, 2500.0),
    alpha_max: float = 0.25,
    kappa_max: float = 0.25,
    pressure_range: tuple[float, float] | None = None,
    noise: float = 0.01,
    seed: int = 0,
    tire_id: str = "synthetic",
) -> pd.DataFrame:
    """Magic-Formula ground truth with additive noise — the always-available dataset.

    Labelled **synthetic** everywhere it is used (PLAN.md §4.4). Its purpose is to make
    the whole framework verifiable before any real download succeeds, and to give the
    extrapolation study a case where the true function is known outside the training
    range.

    ``pressure_range`` adds a pressure context channel with a physically plausible
    effect (stiffness rises and peak friction falls slightly with pressure), which is
    what the Q-Motion generalisation experiment probes.
    """
    rng = np.random.default_rng(seed)
    alpha = rng.uniform(-alpha_max, alpha_max, n)
    kappa = rng.uniform(-kappa_max, kappa_max, n)
    Fz = rng.uniform(*Fz_range, n)

    p_col = None
    B_eff = np.full(n, B)
    mu_eff = np.full(n, mu)
    if pressure_range is not None:
        p_col = rng.uniform(*pressure_range, n)
        p_rel = p_col / 2.0e5
        B_eff = B * (0.85 + 0.3 * p_rel)          # stiffer carcass with pressure
        mu_eff = mu * (1.02 - 0.05 * p_rel)       # smaller contact patch -> slightly less grip

    t = lambda x: torch.as_tensor(x, dtype=torch.float32)
    px = MFParams(B=t(B_eff), C=C, E=E, mu=t(mu_eff), k_mu=0.08, Fz0=1000.0)
    py = MFParams(B=t(B_eff * 0.95), C=C * 0.95, E=E, mu=t(mu_eff), k_mu=0.08, Fz0=1000.0)
    Fx, Fy = pacejka_combined(t(alpha), t(kappa), t(Fz), px, py)
    Fx = Fx.numpy()
    Fy = Fy.numpy()
    if noise > 0:
        Fx = Fx + rng.normal(0, noise, n) * mu * Fz
        Fy = Fy + rng.normal(0, noise, n) * mu * Fz

    df = pd.DataFrame(
        {"alpha": alpha, "kappa": kappa, "Fz": Fz, "Fx": Fx, "Fy": Fy,
         "tire_id": tire_id, "source": "synthetic"}
    )
    if p_col is not None:
        df["p"] = p_col
    return validate_schema(df, ("Fx", "Fy"))


def make_synthetic_transient(
    n_sequences: int = 40,
    T: int = 400,
    dt: float = 0.002,
    mu: float = 1.1,
    sigma_x: float = 0.15,
    sigma_y: float = 0.30,
    vx_range: tuple[float, float] = (5.0, 35.0),
    n_steps: int = 4,
    vary_mu: bool = True,
    noise: float = 0.005,
    seed: int = 0,
) -> pd.DataFrame:
    """Step-test sequences with **known** relaxation dynamics (Experiment 2 ground truth).

    Each sequence holds a constant speed and applies ``n_steps`` abrupt changes to
    ``alpha``, ``kappa``, ``Fz`` and (optionally) ``mu``. The reference force is the
    Magic Formula steady state passed through the exact first-order relaxation
    solution, so the transient a model has to reproduce is generated by the same
    physics ``RelaxationTireCell`` encodes — and the recovered ``sigma`` can be
    compared against the true value, which is the point of the experiment.

    Different speeds per sequence are essential: a model that learns a fixed *time*
    constant instead of a fixed *length* fits one speed and fails the others.
    """
    margin = 20
    if T - 2 * margin < n_steps:
        raise ValueError(
            f"T={T} is too short for n_steps={n_steps}: step edges are placed in "
            f"[{margin}, T-{margin}), which needs T >= {2 * margin + n_steps}. "
            "Either lengthen the sequences or ask for fewer steps."
        )
    rng = np.random.default_rng(seed)
    frames = []
    for s in range(n_sequences):
        vx = float(rng.uniform(*vx_range))
        edges = np.sort(rng.choice(np.arange(margin, T - margin), size=n_steps, replace=False))
        seg = np.zeros(T, dtype=int)
        for e in edges:
            seg[e:] += 1
        alpha = rng.uniform(-0.2, 0.2, n_steps + 1)[seg]
        kappa = rng.uniform(-0.2, 0.2, n_steps + 1)[seg]
        Fz = rng.uniform(500.0, 2000.0, n_steps + 1)[seg]
        mu_seq = (rng.uniform(0.6, 1.2, n_steps + 1)[seg] if vary_mu else np.full(T, mu))

        t = lambda x: torch.as_tensor(x, dtype=torch.float32)
        px = MFParams(B=9.0, C=1.6, E=0.4, mu=t(mu_seq), k_mu=0.08, Fz0=1000.0)
        py = MFParams(B=8.5, C=1.5, E=0.4, mu=t(mu_seq), k_mu=0.08, Fz0=1000.0)
        Fx_ss, Fy_ss = pacejka_combined(t(alpha), t(kappa), t(Fz), px, py)
        Fx_ss, Fy_ss = Fx_ss.numpy(), Fy_ss.numpy()

        # Exact zero-order-hold solution of dF/dt = (F_ss - F)/tau.
        tau_x = sigma_x / (abs(vx) + 0.5)
        tau_y = sigma_y / (abs(vx) + 0.5)
        Fx = np.empty(T)
        Fy = np.empty(T)
        Fx[0], Fy[0] = Fx_ss[0], Fy_ss[0]
        ax_, ay_ = np.exp(-dt / tau_x), np.exp(-dt / tau_y)
        for i in range(1, T):
            Fx[i] = Fx_ss[i] + (Fx[i - 1] - Fx_ss[i]) * ax_
            Fy[i] = Fy_ss[i] + (Fy[i - 1] - Fy_ss[i]) * ay_
        if noise > 0:
            Fx = Fx + rng.normal(0, noise, T) * Fz
            Fy = Fy + rng.normal(0, noise, T) * Fz

        frames.append(pd.DataFrame({
            "t": np.arange(T) * dt, "sequence_id": s, "alpha": alpha, "kappa": kappa,
            "Fz": Fz, "vx": vx, "mu_ref": mu_seq, "Fx": Fx, "Fy": Fy,
            "tire_id": "synthetic", "source": "synthetic",
        }))
    df = pd.concat(frames, ignore_index=True)
    df.attrs["dt"] = dt
    df.attrs["sigma_x"] = sigma_x
    df.attrs["sigma_y"] = sigma_y
    return validate_schema(df, ("Fx", "Fy"))


@dataclass
class Normalizer:
    """Per-column mean/std computed on the **train split only** (PLAN.md §7).

    Saved next to the checkpoint and reloaded at evaluation time; never recomputed on
    validation or test data, which would leak distribution information.
    """

    stats: dict[str, tuple[float, float]] = field(default_factory=dict)

    @classmethod
    def fit(cls, df: pd.DataFrame, columns) -> "Normalizer":
        stats = {}
        for c in columns:
            if c in df.columns:
                s = float(df[c].std())
                stats[c] = (float(df[c].mean()), s if s > 1e-12 else 1.0)
        return cls(stats)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.stats, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Normalizer":
        return cls({k: tuple(v) for k, v in json.loads(Path(path).read_text()).items()})


class TireDataset(Dataset):
    """Canonical DataFrame -> tensors, in static (per-sample) or windowed (sequence) mode.

    Static mode yields scalars per sample; windowed mode yields contiguous windows of
    length ``window`` from the same ``sequence_id``, which is what the relaxation and
    thermal models need. Windows never straddle a sequence boundary.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        targets: tuple[str, ...] = ("Fy",),
        context_keys: tuple[str, ...] = (),
        window: int | None = None,
        tire_index: dict[str, int] | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        validate_schema(df, targets)
        self.df = df.reset_index(drop=True)
        self.targets = tuple(targets)
        self.context_keys = tuple(context_keys)
        self.window = window
        self.dtype = dtype
        self.tire_index = tire_index or {name: i for i, name in enumerate(sorted(df["tire_id"].unique()))}

        if window is not None:
            if "sequence_id" not in df.columns:
                raise ValueError("windowed mode requires a 'sequence_id' column")
            self.windows: list[np.ndarray] = []
            for _, group in self.df.groupby("sequence_id", sort=True):
                idx = group.index.to_numpy()
                for start in range(0, len(idx) - window + 1):
                    self.windows.append(idx[start:start + window])

    def __len__(self) -> int:
        return len(self.windows) if self.window is not None else len(self.df)

    def _rows(self, i: int) -> pd.DataFrame:
        return self.df.loc[self.windows[i]] if self.window is not None else self.df.iloc[[i]]

    def __getitem__(self, i: int) -> dict[str, Tensor]:
        rows = self._rows(i)
        squeeze = self.window is None

        def col(name: str) -> Tensor:
            v = torch.as_tensor(rows[name].to_numpy(), dtype=self.dtype)
            return v[0] if squeeze else v

        item = {k: col(k) for k in ("alpha", "kappa", "Fz")}
        item.update({k: col(k) for k in self.targets})
        ctx = {k: col(k) for k in self.context_keys if k in rows.columns}
        ids = torch.as_tensor([self.tire_index[t] for t in rows["tire_id"]], dtype=torch.long)
        ctx["tire_id"] = ids[0] if squeeze else ids
        item["context"] = ctx
        if "t" in rows.columns:
            item["t"] = col("t")
        return item


def split_by_group(
    df: pd.DataFrame,
    group: str = "sequence_id",
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
):
    """Train/val/test split that never cuts inside a group (default: a trajectory).

    Random row-wise splitting of a time series leaks: neighbouring samples are nearly
    identical, so the test error would measure interpolation between adjacent samples
    rather than generalisation.
    """
    if group not in df.columns:
        idx = np.arange(len(df))
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
        n_tr = int(fractions[0] * len(idx))
        n_va = int(fractions[1] * len(idx))
        parts = (idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:])
        return tuple(df.iloc[np.sort(p)].reset_index(drop=True) for p in parts)

    groups = df[group].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_tr = int(fractions[0] * len(groups))
    n_va = int(fractions[1] * len(groups))
    sets = (groups[:n_tr], groups[n_tr:n_tr + n_va], groups[n_tr + n_va:])
    return tuple(df[df[group].isin(s)].reset_index(drop=True) for s in sets)


def split_by_condition(df: pd.DataFrame, column: str, holdout_values) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Extrapolation holdout: keep entire *conditions* out of training.

    This is the split that actually answers the research question. Holding out a load
    level, a pressure or a tire set tests whether the encoded physics transfers,
    whereas a random split mostly tests interpolation.
    """
    mask = df[column].isin(list(holdout_values))
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)
