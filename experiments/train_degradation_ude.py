#!/usr/bin/env python3
"""Experiment 5 — tyre degradation from stint data, as a universal differential equation.

Tyre condition is never measured. Lap time is, and it carries the consequence. This
experiment asks whether the *dynamics* of degradation can be identified from that
indirect, confounded signal — the same question as the IMU-only vehicle experiment, one
level further removed.

    python experiments/train_degradation_ude.py --config configs/exp5_degradation.yaml
    python experiments/train_degradation_ude.py --set data.source=synthetic

Real data: `python scripts/download_f1_stints.py` (needs `pip install fastf1`).
Without it the experiment runs on synthetic stints generated from a known degradation
model, which additionally allows the recovered latent states to be scored against truth.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tire_nn.data.lap_degradation import (  # noqa: E402
    load_fastf1_stints,
    make_synthetic_stints,
    stint_tensors,
)
from tire_nn.models.degradation_ude import (  # noqa: E402
    BlackBoxDegradationModel,
    LapDegradationUDE,
    LinearDegradationModel,
)
from tire_nn.training import append_summary, set_seed  # noqa: E402


def load_data(dcfg: dict) -> tuple[pd.DataFrame, str]:
    if dcfg.get("source", "fastf1") == "fastf1":
        try:
            df = load_fastf1_stints(dcfg.get("root", "data/raw"))
            return df, "real measurement (F1 timing via FastF1)"
        except FileNotFoundError as exc:
            print(f"[data] {exc}\n[data] falling back to synthetic stints.\n")
    df = make_synthetic_stints(n_sessions=dcfg.get("n_sessions", 8),
                               n_drivers=dcfg.get("n_drivers", 6),
                               laps=dcfg.get("laps", 55),
                               noise=dcfg.get("noise", 0.25))
    return df, "SYNTHETIC (generated from a known degradation model)"


def build(name: str, kw: dict, n_compounds: int, base: float, n_pace_groups: int = 0):
    kw = dict(kw, base_lap_time=base, n_pace_groups=n_pace_groups)
    if name == "linear":
        return LinearDegradationModel(n_compounds, **kw)
    if name == "blackbox":
        return BlackBoxDegradationModel(n_compounds, **kw)
    if name.startswith("ude"):
        return LapDegradationUDE(n_compounds, **kw)
    raise KeyError(name)


def fit_pace_offsets(model, batch):
    """Set the per-group pace offset to its exact least-squares value on a new split.

    The offset enters the observation additively, so for a frozen model the optimal
    value for each group is simply the mean residual of that group's laps — no
    iteration, no learning rate, no risk of not converging. That matters here: held-out
    circuits differ from the training ones by tens of seconds a lap, and a gradient fit
    starting from zero cannot travel that far in a fixed number of steps.

    This is the standard profile-likelihood treatment of a nuisance parameter: the
    kinetics stay frozen and only the nuisance term is re-estimated, so the reported
    test error measures degradation rather than car and circuit pace.
    """
    if not getattr(model, "n_pace_groups", 0) or "pace_group" not in batch:
        return model
    n_groups = int(batch["n_pace_groups"])
    with torch.no_grad():
        model.pace.data = torch.zeros(n_groups)
        model.n_pace_groups = n_groups
        residual = batch["lap_time"] - model(batch)["lap_time"]      # (n_stints, T)
        mask = batch["mask"].float()
        totals = torch.zeros(n_groups)
        counts = torch.zeros(n_groups)
        totals.index_add_(0, batch["pace_group"], (residual * mask).sum(dim=-1))
        counts.index_add_(0, batch["pace_group"], mask.sum(dim=-1))
        model.pace.data = totals / torch.clamp(counts, min=1.0)
    return model


def masked_rmse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    err = ((pred - target) ** 2)[mask]
    return float(torch.sqrt(err.mean()))


def train(model, batch, cfg, verbose=True):
    """Coordinate descent: closed-form pace offsets, gradient steps for the kinetics.

    The per-(session, driver) offsets are nuisance parameters spanning tens of seconds
    across circuits, while the kinetics live in hundredths of a second per lap. Learning
    both by the same gradient step means either the offsets converge far too slowly or
    the kinetics are swamped. Since the offsets have an exact least-squares solution for
    any frozen kinetics, they are re-solved every epoch and excluded from the optimiser.
    """
    kinetic_params = [p for name, p in model.named_parameters() if name != "pace"]
    opt = torch.optim.Adam(kinetic_params, lr=cfg["lr"],
                           weight_decay=cfg.get("weight_decay", 0.0))
    mask, target = batch["mask"], batch["lap_time"]
    best, best_state, since = float("inf"), None, 0

    for epoch in range(cfg["epochs"]):
        fit_pace_offsets(model, batch)              # exact, given the current kinetics
        opt.zero_grad(set_to_none=True)
        out = model(batch)
        loss = (((out["lap_time"] - target) ** 2)[mask]).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(kinetic_params, 10.0)
        opt.step()

        value = float(loss.detach())
        if value < best - 1e-9:
            best, since = value, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= cfg.get("patience", 60):
                break
        if verbose and epoch % 100 == 0:
            print(f"  epoch {epoch:4d}  train MSE {value:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    fit_pace_offsets(model, batch)
    return model.eval()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/exp5_degradation.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--seeds", type=int, default=1,
                    help="repeat the UDE fit with this many seeds and report the spread. "
                         "Worth doing: the lap-time fit is stable but the split of "
                         "degradation between the wear and graining channels is not.")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    for item in args.set:
        key, _, value = item.partition("=")
        node, parts = cfg, key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(value)

    set_seed(cfg["seed"])
    out_root = ROOT / cfg["out_dir"]
    out_root.mkdir(parents=True, exist_ok=True)

    df, label = load_data(cfg["data"])
    synthetic = label.startswith("SYNTHETIC")
    print(f"[data] {label}")
    print(f"[data] {len(df)} laps, {df.groupby(['session_id','driver','stint']).ngroups} stints, "
          f"compounds {sorted(df['compound'].unique())}")
    print(f"[data] track temperature {df['track_temp'].min():.1f}–{df['track_temp'].max():.1f} K")

    # Hold out whole sessions where possible, otherwise whole drivers: a random split of
    # laps would leak, since consecutive laps of a stint are nearly identical.
    key = "session_id" if df["session_id"].nunique() > 2 else "driver"
    groups = sorted(df[key].unique())
    n_test = max(1, int(round(cfg["data"].get("test_fraction", 0.25) * len(groups))))
    rng = np.random.default_rng(cfg["seed"])
    test_groups = set(rng.choice(groups, size=n_test, replace=False).tolist())
    train_df = df[~df[key].isin(test_groups)].reset_index(drop=True)
    test_df = df[df[key].isin(test_groups)].reset_index(drop=True)
    print(f"[data] holding out {len(test_groups)} of {len(groups)} {key} groups "
          f"({len(test_df)} laps) for test")

    compounds = sorted(df["compound"].unique())
    train_batch, _ = stint_tensors(train_df, compounds)
    test_batch, _ = stint_tensors(test_df, compounds)
    base = float(df["lap_time"].median())

    summary = out_root / "summary.csv"
    trained = {}
    for name, kw in cfg["models"].items():
        print(f"\n=== {name} ===")
        model = build(name, kw, len(compounds), base,
                      n_pace_groups=int(train_batch["n_pace_groups"]))
        train(model, train_batch, cfg["training"])
        trained[name] = model

        with torch.no_grad():
            tr_out = model(train_batch)
        # Held-out sessions have their own cars and conditions, so the pace offset is
        # unknown there. Fit ONLY that scalar per group on the test set — the kinetics
        # stay frozen. This is the standard random-effect treatment; without it the test
        # error would measure car pace rather than degradation. It must come after the
        # training-set evaluation, since it resizes the offset vector.
        fit_pace_offsets(model, test_batch)
        with torch.no_grad():
            te_out = model(test_batch)
        row = {
            "experiment": cfg["experiment"], "data": label, "model": name,
            "n_params": sum(p.numel() for p in model.parameters()),
            "train_rmse_s": masked_rmse(tr_out["lap_time"], train_batch["lap_time"],
                                        train_batch["mask"]),
            "test_rmse_s": masked_rmse(te_out["lap_time"], test_batch["lap_time"],
                                       test_batch["mask"]),
        }
        if isinstance(model, LapDegradationUDE):
            row["learned_c_fuel_s"] = float(model.c_fuel().detach())
            row["learned_a_grain_s"] = float(model.a_grain().detach())
            w = te_out["wear"][test_batch["mask"]]
            g = te_out["graining"][test_batch["mask"]]
            row["wear_monotone"] = bool(
                (torch.diff(te_out["wear"], dim=-1) >= -1e-9).all())
            row["graining_in_unit_interval"] = bool(g.min() >= 0 and g.max() <= 1)
            row["max_wear_s"] = float(w.max())
            row["max_graining"] = float(g.max())
            if synthetic and "wear" in test_batch:
                # Only meaningful on synthetic data, where the latent truth is known.
                row["wear_corr_with_truth"] = float(np.corrcoef(
                    w.numpy(), test_batch["wear"][test_batch["mask"]].numpy())[0, 1])
                if model.enable_graining:
                    row["graining_corr_with_truth"] = float(np.corrcoef(
                        g.numpy(), test_batch["graining"][test_batch["mask"]].numpy())[0, 1])
        elif isinstance(model, LinearDegradationModel):
            row["learned_c_fuel_s"] = float(model.c_fuel().detach())
            for i, c in enumerate(compounds):
                row[f"k_deg_{c}_s_per_lap"] = float(model.k_deg[i].detach())
        append_summary(summary, row)
        print("  " + "  ".join(f"{k}={v:.4g}" for k, v in row.items() if isinstance(v, float)))

    # --- plots ---------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pdir = out_root / "plots"
    pdir.mkdir(parents=True, exist_ok=True)
    ude = trained.get("ude")
    if ude is not None:
        with torch.no_grad():
            out = ude(test_batch)
        age = test_batch["tyre_age"][0].numpy()
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
        for i in range(min(4, out["wear"].shape[0])):
            m = test_batch["mask"][i].numpy()
            axes[0].plot(age[m], out["wear"][i].numpy()[m], label=f"stint {i}")
            axes[1].plot(age[m], out["graining"][i].numpy()[m], label=f"stint {i}")
        axes[0].set_ylabel("wear [s of lap time]")
        axes[1].set_ylabel("graining [-]")
        axes[1].set_ylim(-0.05, 1.05)
        for ax in axes:
            ax.set_xlabel("tyre age [laps]")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        axes[0].set_title("Recovered latent states (held-out stints)")
        fig.tight_layout()
        fig.savefig(pdir / "latent_states.png", dpi=140)
        plt.close(fig)
        print(f"\n[plots] {pdir}")

    # --- seed-stability study -------------------------------------------------
    if args.seeds > 1:
        print(f"\n=== seed stability ({args.seeds} seeds, identical data) ===")
        stability = []
        for seed in range(args.seeds):
            set_seed(seed)
            m = build("ude", cfg["models"].get("ude", {}), len(compounds), base,
                      n_pace_groups=int(train_batch["n_pace_groups"]))
            train(m, train_batch, cfg["training"], verbose=False)
            with torch.no_grad():
                out = m(train_batch)
            mask = train_batch["mask"]
            row = {"seed": seed,
                   "rmse_s": masked_rmse(out["lap_time"], train_batch["lap_time"], mask),
                   "c_fuel_s": float(m.c_fuel().detach()),
                   "a_grain_s": float(m.a_grain().detach())}
            if "wear" in train_batch:
                row["wear_corr"] = float(np.corrcoef(
                    out["wear"][mask].numpy(), train_batch["wear"][mask].numpy())[0, 1])
                row["grain_corr"] = float(np.corrcoef(
                    out["graining"][mask].numpy(), train_batch["graining"][mask].numpy())[0, 1])
            stability.append(row)
        table = pd.DataFrame(stability)
        table.to_csv(out_root / "seed_stability.csv", index=False)
        print(table.round(3).to_string(index=False))
        print("\nspread across seeds:")
        for col in table.columns.drop("seed"):
            print(f"  {col:12s} mean {table[col].mean():.3f}   "
                  f"range {table[col].min():.3f} - {table[col].max():.3f}")
        print("\nA stable RMSE with a widely varying wear/graining split means the two "
              "latent\nchannels are not separately identifiable from lap time alone.")

    print(f"\n[summary] {summary}")
    print(pd.read_csv(summary).to_string(index=False))
    if not synthetic:
        print("\nNOTE: lap time is an indirect, confounded observation of tyre state. "
              "These results are about identifying degradation DYNAMICS, not tyre forces.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
