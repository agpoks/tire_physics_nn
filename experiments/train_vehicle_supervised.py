#!/usr/bin/env python3
"""Experiment 3 — four-wheel vehicle-supervised tire learning (PLAN.md §5).

No tire force is ever observed. The only supervision is ``ax``, ``ay`` and yaw
acceleration, connected to the tire model by the exact Newton-Euler equations, with
one shared ``TireNet`` for all four corners.

    python experiments/train_vehicle_supervised.py --config configs/exp3_vehicle.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tire_nn.data import TireDataset, make_synthetic, split_by_group  # noqa: E402
from tire_nn.data.vehicle import VehicleDataset, make_synthetic_vehicle  # noqa: E402
from tire_nn.models import build_model  # noqa: E402
from tire_nn.models.four_wheel_vehicle import FourWheelVehicle  # noqa: E402
from tire_nn.physics import VehicleParams  # noqa: E402
from tire_nn.physics.fitting import fit_magic_formula  # noqa: E402
from tire_nn.physics.pacejka import MagicFormulaTire  # noqa: E402
from tire_nn.training import TrainConfig, append_summary, set_seed, train_model  # noqa: E402
from tire_nn.training.losses import vehicle_loss  # noqa: E402
from tire_nn.training.metrics import rmse  # noqa: E402


def batch_forward(veh: FourWheelVehicle, batch: dict) -> tuple:
    out = veh(batch["vx"], batch["vy"], batch["r"], batch["delta"], batch["omega"],
              ax_meas=batch["ax"], ay_meas=batch["ay"])
    return out["ax"], out["ay"], out["r_dot"]


def train_vehicle(veh, train_ds, val_ds, cfg, out_dir=None, verbose=True):
    """Adam on the vehicle-level loss. Kept separate from ``training.trainer`` because
    the batch contract is different (vehicle states, not slip/force pairs)."""
    opt = torch.optim.Adam([p for p in veh.parameters() if p.requires_grad], lr=cfg["lr"])
    weights = tuple(cfg.get("loss_weights", (1.0, 1.0, 1.0)))
    tl = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    vl = DataLoader(val_ds, batch_size=cfg["batch_size"]) if val_ds is not None else None
    best = {"loss": float("inf"), "epoch": -1, "state": None}
    since = 0
    for epoch in range(cfg["epochs"]):
        veh.train()
        for batch in tl:
            opt.zero_grad(set_to_none=True)
            loss = vehicle_loss(batch_forward(veh, batch),
                                (batch["ax"], batch["ay"], batch["r_dot"]), weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(veh.parameters(), 10.0)
            opt.step()
        veh.eval()
        with torch.no_grad():
            total, n = 0.0, 0
            for batch in (vl or tl):
                l = vehicle_loss(batch_forward(veh, batch),
                                 (batch["ax"], batch["ay"], batch["r_dot"]), weights)
                total += float(l) * len(batch["vx"])
                n += len(batch["vx"])
            val = total / max(n, 1)
        if val < best["loss"] - 1e-9:
            best = {"loss": val, "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in veh.state_dict().items()}}
            since = 0
        else:
            since += 1
            if since >= cfg.get("patience", 40):
                break
        if verbose and epoch % 20 == 0:
            print(f"  epoch {epoch:4d}  val {val:.6g}")
    if best["state"]:
        veh.load_state_dict(best["state"])
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": veh.state_dict()}, Path(out_dir) / "best.pt")
    return best


@torch.no_grad()
def evaluate(veh, ds) -> dict:
    dl = DataLoader(ds, batch_size=4096)
    preds, targets = [[], [], []], [[], [], []]
    for batch in dl:
        p = batch_forward(veh, batch)
        t = (batch["ax"], batch["ay"], batch["r_dot"])
        for i in range(3):
            preds[i].append(p[i])
            targets[i].append(t[i])
    names = ("ax", "ay", "r_dot")
    return {f"{n}_rmse": rmse(torch.cat(preds[i]), torch.cat(targets[i])) for i, n in enumerate(names)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/exp3_vehicle.yaml")
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
    vp = VehicleParams(**cfg["vehicle"])
    d = cfg["data"]

    train_all = make_synthetic_vehicle(n_sequences=d["n_sequences"], T=d["T"], dt=d["dt"], vp=vp,
                                       mu_values=tuple(d["mu_train"]), tire_ids=tuple(d["tire_ids"]),
                                       noise_imu=d["noise_imu"], seed=cfg["seed"])
    unseen = make_synthetic_vehicle(n_sequences=2, T=d["T"], dt=d["dt"], vp=vp,
                                    mu_values=tuple(d["mu_test"]), tire_ids=("unseen",),
                                    noise_imu=d["noise_imu"], seed=cfg["seed"] + 100)
    print(f"[data] train/val {len(train_all)} rows (mu={d['mu_train']}), "
          f"unseen-condition test {len(unseen)} rows (mu={d['mu_test']})")

    tr, va, _ = split_by_group(train_all, fractions=tuple(d["split"]), seed=cfg["seed"])
    tire_index = {n: i for i, n in enumerate(sorted(set(train_all["tire_id"]) | set(unseen["tire_id"])))}
    tr_ds, va_ds = VehicleDataset(tr, tire_index), VehicleDataset(va, tire_index)
    unseen_ds = VehicleDataset(unseen, tire_index)

    summary = out_root / "summary.csv"
    for name, kw in cfg["models"].items():
        print(f"\n=== {name} ===")
        kw = dict(kw)
        if name == "analytical":
            # Fit the Magic Formula to *direct* tire data — the reference a team would
            # normally have. It never sees the vehicle data, which is the point.
            ref = make_synthetic(4000, mu=float(d["mu_train"][0]), seed=cfg["seed"])
            px, py = fit_magic_formula(ref)
            veh = FourWheelVehicle(MagicFormulaTire(px, py), vp)
            best = {"epoch": -1}
        else:
            tire = build_model(kw.pop("tire", "encoded"), context_keys=("vx",))
            if cfg["evaluation"].get("pretrain_with_tire_data", False):
                # Optional warm start from direct tire data (PLAN.md §5, Experiment 3).
                pre = make_synthetic(3000, mu=float(d["mu_train"][0]), seed=cfg["seed"] + 7)
                ptr, pva, _ = split_by_group(pre)
                mk = lambda x: TireDataset(x, targets=("Fx", "Fy"), context_keys=("vx",))
                train_model(tire, mk(ptr), mk(pva),
                            TrainConfig(epochs=cfg["evaluation"].get("pretrain_epochs", 40),
                                        targets=("Fx", "Fy"), patience=20), verbose=False)
                print("  pretrained on direct tire data")
            veh = FourWheelVehicle(tire, vp, **kw)
            best = train_vehicle(veh, tr_ds, va_ds, cfg["training"], out_root / name)

        veh.eval()
        row = {"experiment": cfg["experiment"], "model": name,
               "n_params": sum(p.numel() for p in veh.parameters()),
               "shared_tire": bool(kw.get("share_tire", True)) if name != "analytical" else True,
               "best_epoch": best["epoch"]}
        row.update({f"val_{k}": v for k, v in evaluate(veh, va_ds).items()})
        row.update({f"unseen_{k}": v for k, v in evaluate(veh, unseen_ds).items()})
        append_summary(summary, row)
        print("  " + "  ".join(f"{k}={v:.4g}" for k, v in row.items() if isinstance(v, float)))

    print(f"\n[summary] {summary}")
    print(pd.read_csv(summary).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
