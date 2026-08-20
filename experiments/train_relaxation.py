#!/usr/bin/env python3
"""Experiment 2 — transient tire modelling (PLAN.md §5).

Compares a static tire net, a generic GRU, a generic Neural ODE and the
physics-encoded ``RelaxationTireCell`` on step tests in ``alpha``, ``kappa``, ``Fz``
and ``mu``, then asks the question that matters: do they still work at speeds outside
the training range?

    python experiments/train_relaxation.py --config configs/exp2_relaxation.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tire_nn.data import TireDataset, make_synthetic_transient, split_by_group  # noqa: E402
from tire_nn.models import build_model  # noqa: E402
from tire_nn.models.baselines_seq import GRUTireModel, NeuralODETireModel  # noqa: E402
from tire_nn.models.relaxation_tire import RelaxationTireCell  # noqa: E402
from tire_nn.training import TrainConfig, append_summary, set_seed, train_model  # noqa: E402
from tire_nn.training.metrics import rmse  # noqa: E402


def build(name: str, kw: dict):
    kw = dict(kw)
    if name in ("static", "relaxation", "relaxation_parameter"):
        steady = build_model(kw.pop("steady", "encoded"), context_keys=("vx",))
        if name == "static":
            return steady
        return RelaxationTireCell(steady, **kw)
    if name == "gru":
        return GRUTireModel(context_keys=("vx",), **kw)
    if name == "neural_ode":
        return NeuralODETireModel(context_keys=("vx",), **kw)
    raise KeyError(name)


@torch.no_grad()
def rollout_rmse(model, df: pd.DataFrame, dt: float, integrator: str) -> dict:
    """Whole-sequence rollout error, per sequence, then averaged."""
    errs_x, errs_y = [], []
    for _, g in df.groupby("sequence_id"):
        t = lambda c: torch.as_tensor(g[c].to_numpy(), dtype=torch.float32).unsqueeze(0)
        F = model.rollout(t("alpha"), t("kappa"), t("Fz"), t("vx"), dt,
                          {"vx": t("vx")}, method=integrator)
        errs_x.append(rmse(F[0, :, 0], t("Fx")[0]))
        errs_y.append(rmse(F[0, :, 1], t("Fy")[0]))
    return {"Fx_rmse": float(sum(errs_x) / len(errs_x)), "Fy_rmse": float(sum(errs_y) / len(errs_y))}


@torch.no_grad()
def step_response(model, dt: float, vx: float, integrator: str, T: int = 400):
    """Lateral step response and the distance travelled to reach 63% of steady state.

    Reporting the *distance* rather than the time is the physical statement: for a
    real tire it is (approximately) speed-invariant and equal to the relaxation
    length. A model that learned a fixed time constant shows a distance that grows
    linearly with speed.
    """
    a = torch.zeros(1, T)
    a[:, T // 4:] = 0.1
    z = torch.zeros(1, T)
    Fz = torch.full((1, T), 1000.0)
    v = torch.full((1, T), vx)
    F = model.rollout(a, z, Fz, v, dt, {"vx": v}, method=integrator)
    Fy = F[0, :, 1]
    start, final = Fy[T // 4 - 1], Fy[-1]
    target = start + 0.632 * (final - start)
    after = Fy[T // 4:]
    reached = (after - start).abs() >= (target - start).abs()
    idx = int(torch.nonzero(reached)[0]) if reached.any() else T
    return float(idx * dt * vx), Fy


def main() -> int:
    import yaml

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/exp2_relaxation.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    for item in args.set:
        key, _, value = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)

    set_seed(cfg["seed"])
    out_root = ROOT / cfg["out_dir"]
    out_root.mkdir(parents=True, exist_ok=True)
    d = cfg["data"]
    dt = d["dt"]

    df = make_synthetic_transient(
        n_sequences=d["n_sequences"], T=d["T"], dt=dt, sigma_x=d["sigma_x"], sigma_y=d["sigma_y"],
        vx_range=tuple(d["vx_range"]), vary_mu=d["vary_mu"], noise=d["noise"], seed=cfg["seed"])
    print(f"[data] synthetic transient: {len(df)} samples, {d['n_sequences']} sequences, "
          f"true sigma_x={d['sigma_x']} sigma_y={d['sigma_y']}")

    # Speed holdout: sequences above vx_train_max are never trained on.
    fast_ids = df.loc[df["vx"] > d["vx_train_max"], "sequence_id"].unique()
    slow = df[~df["sequence_id"].isin(fast_ids)].reset_index(drop=True)
    fast = df[df["sequence_id"].isin(fast_ids)].reset_index(drop=True)
    train_df, val_df, test_df = split_by_group(slow, fractions=tuple(d["split"]), seed=cfg["seed"])
    print(f"[data] train {train_df['sequence_id'].nunique()} seq (vx <= {d['vx_train_max']}), "
          f"speed-extrapolation set {fast['sequence_id'].nunique()} seq")

    mk = lambda x: TireDataset(x, targets=("Fx", "Fy"), context_keys=("vx",), window=d["window"])
    tcfg_base = cfg["training"]
    summary = out_root / "summary.csv"

    for name, kw in cfg["models"].items():
        print(f"\n=== {name} ===")
        model = build(name, kw)
        tcfg = TrainConfig(
            **{k: v for k, v in tcfg_base.items() if k in TrainConfig.__dataclass_fields__},
            targets=("Fx", "Fy"), dt=dt, seed=cfg["seed"], device=cfg["device"])
        history = train_model(model, mk(train_df), mk(val_df), tcfg, out_root / name)
        model.eval()

        row = {"experiment": cfg["experiment"], "model": name,
               "n_params": sum(p.numel() for p in model.parameters()),
               "best_epoch": history["best_epoch"]}
        row.update({f"test_{k}": v for k, v in rollout_rmse(model, test_df, dt, tcfg.integrator).items()})
        row.update({f"fastvx_{k}": v for k, v in rollout_rmse(model, fast, dt, tcfg.integrator).items()})
        for vx in (10.0, 30.0):
            dist, _ = step_response(model, dt, vx, tcfg.integrator)
            row[f"rise_distance_vx{int(vx)}"] = dist
        # Speed-invariance of the rise distance is the physical signature of a
        # correctly encoded (distance-parameterised) relaxation.
        row["rise_distance_ratio"] = row["rise_distance_vx30"] / max(row["rise_distance_vx10"], 1e-9)
        if isinstance(model, RelaxationTireCell):
            if model.sigma_from_steady:
                # Condition-dependent sigma: report the mean over the test operating points.
                p_test = model.steady.parameters_at(
                    torch.as_tensor(test_df["Fz"].to_numpy(), dtype=torch.float32),
                    {"vx": torch.as_tensor(test_df["vx"].to_numpy(), dtype=torch.float32)})
                row["learned_sigma_x"] = float(p_test["sigma_x"].mean())
                row["learned_sigma_y"] = float(p_test["sigma_y"].mean())
            else:
                row["learned_sigma_x"] = float(model.sigma_x())
                row["learned_sigma_y"] = float(model.sigma_y())
            row["sigma_x_error"] = row["learned_sigma_x"] - d["sigma_x"]
            row["sigma_y_error"] = row["learned_sigma_y"] - d["sigma_y"]
        append_summary(summary, row)
        print("  " + "  ".join(f"{k}={v:.4g}" for k, v in row.items() if isinstance(v, float)))

    print(f"\n[summary] {summary}")
    print(pd.read_csv(summary).to_string(index=False))
    print("\nrise_distance_ratio ~ 1.0 means the transient is parameterised by travelled "
          "distance (physically correct); ~3.0 means the model learned a fixed time constant.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
