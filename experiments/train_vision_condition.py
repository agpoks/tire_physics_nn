#!/usr/bin/env python3
"""Experiment 6 — tyre condition from imagery, on real photographs and synthetic textures.

Answers two questions that the synthetic-only version of this work could not:

1. **Does the encoded ordinal model work on real photographs?** The wear index is a
   single monotone latent with ordered thresholds; real data can only supervise the
   classes, so the question is whether the latent orders real images sensibly.
2. **Does a model trained on synthetic textures transfer to real ones?** This is the
   honest check on every synthetic result in this project.

    python -m pip install huggingface_hub
    python scripts/download_tyre_images.py --limit 250
    python experiments/train_vision_condition.py

Without the download it runs synthetic-only and says so.

The real data is ~1 850 photographs from NMiriams/Good_Tires and NMiriams/Defective_Tires
on the Hugging Face Hub, CC BY 4.0. Its labels are **binary condition**, and "defective"
mixes tread wear with cracking, bulges and punctures — so it is not a graded wear scale
and no result here should be read as a tread-depth measurement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tire_nn.data.tread_images import load_tyre_quality_images, make_tread_dataset  # noqa: E402
from tire_nn.models.condition_vision import TreadConditionNet, ordinal_loss  # noqa: E402
from tire_nn.training import append_summary, set_seed  # noqa: E402


def train(images, labels, n_classes=2, epochs=30, lr=3e-3, width=16, seed=0):
    set_seed(seed)
    model = TreadConditionNet(n_classes=n_classes, predict_graining=False, width=width)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(images, labels), batch_size=32, shuffle=True)
    for _ in range(epochs):
        for img, lab in loader:
            opt.zero_grad(set_to_none=True)
            ordinal_loss(model(img)["cumulative"], lab).backward()
            opt.step()
    return model.eval()


@torch.no_grad()
def score(model, images, labels, tag):
    out = model(images)
    accuracy = float((out["class_prob"].argmax(-1) == labels).float().mean())
    low = out["wear"][labels == 0].mean()
    high = out["wear"][labels == labels.max()].mean()
    return {
        "setting": tag,
        "accuracy": accuracy,
        "wear_index_low_class": float(low),
        "wear_index_high_class": float(high),
        "ordered_correctly": bool(high > low),
        "n_test": int(len(labels)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_dir = ROOT / "results" / "exp6_vision"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    # ---------------------------------------------------------------- synthetic
    synthetic = make_tread_dataset(n=1200, size=64, seed=args.seed)
    syn_test = make_tread_dataset(n=300, size=64, seed=args.seed + 1)
    syn_labels = (synthetic["wear"] > 0.5).long()
    syn_test_labels = (syn_test["wear"] > 0.5).long()

    model_syn = train(synthetic["images"], syn_labels, epochs=args.epochs, seed=args.seed)
    row = score(model_syn, syn_test["images"], syn_test_labels, "synthetic -> synthetic")
    with torch.no_grad():
        row["wear_corr_with_truth"] = float(np.corrcoef(
            model_syn(syn_test["images"])["wear"].numpy(), syn_test["wear"].numpy())[0, 1])
    rows.append(row)
    print(f"[synthetic -> synthetic] accuracy {row['accuracy']:.3f}  "
          f"wear corr {row['wear_corr_with_truth']:.3f}")

    # --------------------------------------------------------------------- real
    try:
        real = load_tyre_quality_images(args.root, size=64)
    except FileNotFoundError as exc:
        print(f"\n[real] {exc}")
        print("[real] skipping the real-data comparison.")
        pd.DataFrame(rows).to_csv(out_dir / "summary.csv", index=False)
        return 0

    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(real["images"]), generator=generator)
    images, labels = real["images"][order], real["label"][order]
    split = int(0.75 * len(images))
    Xtr, ytr, Xte, yte = images[:split], labels[:split], images[split:], labels[split:]
    print(f"\n[real] {len(Xtr)} train / {len(Xte)} test photographs")

    model_real = train(Xtr, ytr, epochs=args.epochs, seed=args.seed)
    rows.append(score(model_real, Xte, yte, "real -> real"))
    rows.append(score(model_syn, Xte, yte, "synthetic -> real (transfer)"))

    table = pd.DataFrame(rows)
    for row in rows:
        append_summary(out_dir / "summary.csv", {"experiment": "exp6_vision", **row})
    print()
    print(table.round(3).to_string(index=False))

    transfer = table[table.setting.str.contains("transfer")].iloc[0]
    print(f"\nchance level is {1 / len(real['classes']):.3f}.")
    if transfer["accuracy"] < 0.6:
        print("The synthetic-trained model does NOT transfer to real photographs. The "
              "synthetic\ntextures are a stand-in for the structure of the problem, not "
              "a renderer of real\ntyres — every synthetic result in this project should "
              "be read with that in mind.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
