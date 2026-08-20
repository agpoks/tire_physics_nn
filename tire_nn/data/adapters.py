"""Shared machinery for the dataset adapters (PLAN.md §4).

Every adapter maps one source onto the canonical schema of
:py:mod:`tire_nn.data.common`: SI units, SAE signs, one row per measurement instant.
Converting at the adapter boundary — and only there — is what lets a single training
script run on a drum test bench, a bicycle-tyre rig and a small-scale race car.

Because the raw column names of these datasets are not fixed (and several of the
sources still need manual download, PLAN.md §4.4), each adapter declares a **column
map** of canonical name -> candidate source names plus a unit conversion. The map is
the thing to edit when a real file turns out to use different headers, and it is
overridable per call, so nothing has to be patched in the library to read a variant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from tire_nn.data.common import validate_schema

__all__ = ["ColumnSpec", "DatasetNotAvailable", "read_table", "read_any", "map_columns", "load"]

DEG2RAD = math.pi / 180.0
BAR2PA = 1.0e5
KPA2PA = 1.0e3
PSI2PA = 6894.757
C2K = 273.15


class DatasetNotAvailable(FileNotFoundError):
    """Raised with actionable instructions when a dataset has not been downloaded."""


@dataclass
class ColumnSpec:
    """How to obtain one canonical column from a source table.

    Args:
        candidates: source column names to look for, in priority order. Matching is
            case-insensitive and ignores spaces and underscores, because measurement
            exports are inconsistent about all three.
        scale: multiplied onto the raw value (unit conversion, or -1 for a sign flip).
        offset: added *after* scaling (e.g. 273.15 for degrees Celsius).
        required: raise if no candidate is found.
        default: value used when the column is absent and not required.
    """

    candidates: tuple[str, ...]
    scale: float = 1.0
    offset: float = 0.0
    required: bool = False
    default: float | None = None


def _normalise(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def map_columns(df: pd.DataFrame, spec: dict[str, ColumnSpec]) -> pd.DataFrame:
    """Apply a column map, converting units and reporting exactly what was missing."""
    lookup = {_normalise(c): c for c in df.columns}
    out: dict[str, object] = {}
    missing_required: list[str] = []
    for canonical, cs in spec.items():
        source = next((lookup[_normalise(c)] for c in cs.candidates if _normalise(c) in lookup), None)
        if source is None:
            if cs.required:
                missing_required.append(f"{canonical} (looked for {list(cs.candidates)})")
            elif cs.default is not None:
                out[canonical] = cs.default
            continue
        out[canonical] = pd.to_numeric(df[source], errors="coerce") * cs.scale + cs.offset
    if missing_required:
        raise ValueError(
            "the source table is missing required columns:\n  - "
            + "\n  - ".join(missing_required)
            + f"\navailable columns: {list(df.columns)}\n"
            "Pass an updated column map to the adapter rather than editing the library."
        )
    return pd.DataFrame(out)


def read_table(path: Path, **kw) -> pd.DataFrame:
    """Read one CSV/TSV/Parquet/Excel file, guessing the separator for text formats."""
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, **kw)
    return pd.read_csv(path, sep=None, engine="python", **kw)


def read_any(root: Path, patterns: tuple[str, ...], dataset: str, instructions: str) -> pd.DataFrame:
    """Concatenate every file under ``root`` matching ``patterns``.

    Raises :class:`DatasetNotAvailable` with the manual-download instructions when
    nothing is found, rather than a bare ``FileNotFoundError`` — the caller is usually
    a person who has not downloaded the data yet, not a bug.
    """
    root = Path(root)
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(root.rglob(pattern)))
    if not files:
        raise DatasetNotAvailable(
            f"no {dataset} files found under {root} (patterns: {list(patterns)}).\n{instructions}"
        )
    frames = []
    for f in files:
        frame = read_table(f)
        frame["__file"] = f.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def finalise(df: pd.DataFrame, source: str, targets: tuple[str, ...],
             tire_id: str | pd.Series = "unknown") -> pd.DataFrame:
    """Attach the bookkeeping columns, drop unusable rows and validate the schema."""
    df = df.copy()
    df["source"] = source
    if "tire_id" not in df.columns:
        df["tire_id"] = tire_id
    df["tire_id"] = df["tire_id"].astype(str)
    needed = ["alpha", "kappa", "Fz", *targets]
    before = len(df)
    df = df.dropna(subset=[c for c in needed if c in df.columns]).reset_index(drop=True)
    df = df[df["Fz"] > 0].reset_index(drop=True)
    if len(df) < before:
        print(f"[{source}] dropped {before - len(df)} rows with missing values or Fz <= 0")
    return validate_schema(df, targets)


def load(name: str, root: str | Path = "data/raw", **kw) -> pd.DataFrame:
    """Dispatch to an adapter by dataset name."""
    from tire_nn.data import deep_dynamics, kit, qmotion, roboracer, tum_cargo_bike, vetyt

    table = {
        "kit": kit.load_kit,
        "vetyt": vetyt.load_vetyt,
        "tum_cargo_bike": tum_cargo_bike.load_tum_cargo_bike,
        "deep_dynamics": deep_dynamics.load_deep_dynamics,
        "roboracer": roboracer.load_roboracer,
        "qmotion": qmotion.load_qmotion,
    }
    if name not in table:
        raise KeyError(f"unknown dataset {name!r}; available: {sorted(table)}")
    return table[name](root, **kw)
