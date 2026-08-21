"""Deterministic training loop shared by all experiments (PLAN.md §7)."""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from tire_nn.training.losses import force_loss, friction_penalty

__all__ = ["TrainConfig", "set_seed", "collate", "train_model", "append_summary"]


def set_seed(seed: int) -> None:
    """Seed every RNG this project can touch, and disable nondeterministic kernels.

    The CUDA calls are guarded by an availability check. Calling
    ``torch.cuda.manual_seed_all`` on a CUDA-built PyTorch with no usable device forces
    a CUDA context initialisation that can kill the process outright — it reproducibly
    killed the Jupyter kernel in this project's notebooks (torch 2.13+cu130 under WSL).
    Guarding it costs nothing on a real GPU and makes the function safe everywhere.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@dataclass
class TrainConfig:
    epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    loss: str = "mse"
    normalize_by_load: bool = True
    targets: tuple[str, ...] = ("Fy",)
    friction_penalty_weight: float = 0.0     # ablation only (PLAN.md §2.3)
    mu_ref: float = 1.5                      # limit used by the penalty ablation
    patience: int = 50
    seed: int = 0
    device: str = "cpu"
    log_every: int = 25
    grad_clip: float = 10.0
    #: "static" trains on per-sample forces; "sequence" rolls the model out over a
    #: window and supervises the whole trajectory (Experiment 2 / relaxation).
    mode: str = "static"
    dt: float = 0.002
    integrator: str = "rk4"


def collate(batch: list[dict]) -> dict:
    """Stack samples, keeping the nested ``context`` dict a dict of tensors."""
    out: dict = {}
    keys = [k for k in batch[0] if k != "context"]
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch])
    ctx_keys = batch[0]["context"].keys()
    out["context"] = {k: torch.stack([b["context"][k] for b in batch]) for k in ctx_keys}
    return out


def _step_sequence(model, batch, cfg: TrainConfig, device):
    """Trajectory loss: roll the model out over the window and compare every sample.

    Supervising the whole rollout (rather than one-step-ahead) is what forces the
    *dynamics* to be right: a one-step loss is dominated by the steady-state map and
    barely constrains the time constant.
    """
    alpha = batch["alpha"].to(device)
    kappa = batch["kappa"].to(device)
    Fz = batch["Fz"].to(device)
    ctx = {k: v.to(device) for k, v in batch["context"].items()}
    vx = ctx.get("vx")
    if vx is None:
        raise ValueError("sequence mode needs 'vx' in the context (relaxation is distance-based)")
    F = model.rollout(alpha, kappa, Fz, vx, cfg.dt, ctx, method=cfg.integrator)
    loss = torch.zeros((), device=device)
    for i, target in enumerate(("Fx", "Fy")):
        if target in cfg.targets:
            loss = loss + force_loss(F[..., i], batch[target].to(device),
                                     Fz if cfg.normalize_by_load else None, cfg.loss)
    return loss


def _step(model, batch, cfg: TrainConfig, device):
    if cfg.mode == "sequence":
        return _step_sequence(model, batch, cfg, device)
    alpha = batch["alpha"].to(device)
    kappa = batch["kappa"].to(device)
    Fz = batch["Fz"].to(device)
    ctx = {k: v.to(device) for k, v in batch["context"].items()}
    out = model(alpha, kappa, Fz, ctx)
    loss = torch.zeros((), device=device)
    for target in cfg.targets:
        pred = getattr(out, target)
        loss = loss + force_loss(pred, batch[target].to(device),
                                 Fz if cfg.normalize_by_load else None, cfg.loss)
    if cfg.friction_penalty_weight > 0:
        mu = torch.full_like(Fz, cfg.mu_ref)
        loss = loss + cfg.friction_penalty_weight * friction_penalty(out.Fx, out.Fy, mu, mu, Fz)
    return loss


def train_model(
    model: nn.Module,
    train_set,
    val_set=None,
    cfg: TrainConfig | None = None,
    out_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    """Train, early-stop on validation loss, checkpoint the best state.

    Returns a history dict; writes ``best.pt``, ``config.json`` and ``history.csv``
    into ``out_dir`` when given.
    """
    cfg = cfg or TrainConfig()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)
    model = model.to(device)

    gen = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=collate, generator=gen, drop_last=False)
    val_loader = (
        DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate)
        if val_set is not None else None
    )

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    history = {"train_loss": [], "val_loss": []}
    best = {"loss": float("inf"), "epoch": -1, "state": None}
    since_improved = 0

    for epoch in range(cfg.epochs):
        model.train()
        total, count = 0.0, 0
        for batch in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = _step(model, batch, cfg, device)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            total += float(loss.detach()) * len(batch["alpha"])
            count += len(batch["alpha"])
        train_loss = total / max(count, 1)
        history["train_loss"].append(train_loss)

        val_loss = train_loss
        if val_loader is not None:
            model.eval()
            total, count = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    total += float(_step(model, batch, cfg, device)) * len(batch["alpha"])
                    count += len(batch["alpha"])
            val_loss = total / max(count, 1)
        history["val_loss"].append(val_loss)

        if val_loss < best["loss"] - 1e-9:
            best = {"loss": val_loss, "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            since_improved = 0
        else:
            since_improved += 1
            if since_improved >= cfg.patience:
                if verbose:
                    print(f"  early stop at epoch {epoch} (best {best['loss']:.6g} @ {best['epoch']})")
                break

        if verbose and (epoch % cfg.log_every == 0 or epoch == cfg.epochs - 1):
            print(f"  epoch {epoch:4d}  train {train_loss:.6g}  val {val_loss:.6g}")

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    history["best_epoch"] = best["epoch"]
    history["best_val_loss"] = best["loss"]

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "config": asdict(cfg)}, out / "best.pt")
        (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2, default=str))
        with open(out / "history.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["epoch", "train_loss", "val_loss"])
            w.writerows(zip(range(len(history["train_loss"])), history["train_loss"], history["val_loss"]))
    return history


def append_summary(path: str | Path, row: dict) -> None:
    """Append one experiment row to a CSV summary.

    Handles a growing schema: models report different fields (only the relaxation
    models have a ``learned_sigma``, only the parameter models have ``mu``), so when a
    new key appears the file is rewritten with the union of all columns and missing
    entries left blank. Silently dropping the extra columns would lose exactly the
    model-specific diagnostics the experiment is run for.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    fields: list[str] = []
    if path.exists():
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            fields = list(reader.fieldnames or [])
            rows = list(reader)
    for key in row:
        if key not in fields:
            fields.append(key)
    rows.append({k: row.get(k, "") for k in fields})
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})
