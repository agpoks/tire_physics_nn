"""KIT inner-drum tire force transmission dataset (PLAN.md §4.4, source 1).

**Type: real measurement.**

    Tire's force transmission characteristic on dry asphalt measured at the inner drum
    test bench of KIT — Institute of Vehicle System Technology, RADAR4KIT repository,
    https://radar.kit.edu/radar/en/dataset/p0rr2jc5wmf0drf8
    Licence: CC BY-NC-SA 4.0.

The release contains lateral and longitudinal force-transmission characteristics
measured on dry asphalt, transformed from the measurement coordinate system (TYDEX C)
into two vehicle-related systems (TYDEX H and W), plus a separate folder holding a
simulated slalom driving cycle for the front left and right wheels.

Coordinate system
-----------------
TYDEX defines the axis directions; this adapter assumes the **W-axis system** files,
whose signs already match the SAE convention used here (positive slip angle gives
negative lateral force). Set ``flip_signs=True`` if the file you have uses the opposite
convention — check one pure-lateral sweep before trusting a whole run.

The simulated slalom folder is **simulation output, not measurement**. It is excluded
by default (``include_simulation=False``) so a simulated cycle can never quietly enter
a table of measured results.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tire_nn.data.adapters import (
    C2K,
    DEG2RAD,
    KPA2PA,
    ColumnSpec,
    finalise,
    map_columns,
    read_any,
)
from tire_nn.data.common import flip_sign_convention

__all__ = ["KIT_COLUMNS", "load_kit"]

INSTRUCTIONS = """
Manual download (the dataset is behind a repository landing page, so it is not fetched
automatically — see scripts/download_kit.py):

  1. open https://radar.kit.edu/radar/en/dataset/p0rr2jc5wmf0drf8
  2. accept the CC BY-NC-SA 4.0 licence and download the archive
  3. extract it to  data/raw/kit/
  4. re-run with --set data.source=kit data.root=data/raw

If the column headers differ from the defaults in KIT_COLUMNS, pass your own map:
  load_kit(root, columns={**KIT_COLUMNS, "Fy": ColumnSpec(("F_y_W",), required=True)})
""".strip()

#: Canonical name -> candidate source headers and unit conversion. Edit or override
#: this map when the actual file headers differ; do not patch the loader.
KIT_COLUMNS: dict[str, ColumnSpec] = {
    "alpha": ColumnSpec(("alpha", "slip_angle", "slipangle", "SA", "alpha_deg"), DEG2RAD, required=True),
    "kappa": ColumnSpec(("kappa", "slip_ratio", "slipratio", "SR", "longitudinal_slip"), 1.0, default=0.0),
    "Fz": ColumnSpec(("Fz", "F_z", "Fz_W", "vertical_force", "wheel_load"), 1.0, required=True),
    "Fx": ColumnSpec(("Fx", "F_x", "Fx_W", "longitudinal_force"), 1.0),
    "Fy": ColumnSpec(("Fy", "F_y", "Fy_W", "lateral_force"), 1.0),
    "Mz": ColumnSpec(("Mz", "M_z", "Mz_W", "aligning_moment", "self_aligning_torque"), 1.0),
    "vx": ColumnSpec(("vx", "v_x", "velocity", "speed", "drum_speed"), 1.0),
    "p": ColumnSpec(("p", "pressure", "inflation_pressure", "p_kPa"), KPA2PA),
    "gamma": ColumnSpec(("gamma", "camber", "inclination_angle", "IA"), DEG2RAD, default=0.0),
    "Ts": ColumnSpec(("T_surface", "tire_temperature", "T_tire"), 1.0, C2K),
}


def load_kit(
    root: str | Path = "data/raw",
    subdir: str = "kit",
    columns: dict[str, ColumnSpec] | None = None,
    targets: tuple[str, ...] = ("Fx", "Fy"),
    flip_signs: bool = False,
    include_simulation: bool = False,
    tire_id: str = "kit_dry_asphalt",
) -> pd.DataFrame:
    """Load the KIT dataset into the canonical schema."""
    root = Path(root) / subdir
    raw = read_any(root, ("*.csv", "*.txt", "*.parquet", "*.xlsx"), "KIT", INSTRUCTIONS)
    if not include_simulation:
        mask = raw["__file"].str.contains("slalom|simul", case=False, na=False)
        if mask.any():
            print(f"[kit] excluding {int(mask.sum())} rows from simulated driving-cycle files "
                  f"(pass include_simulation=True to keep them, and label them 'simulated')")
            raw = raw[~mask].reset_index(drop=True)

    df = map_columns(raw, columns or KIT_COLUMNS)
    if flip_signs:
        df = flip_sign_convention(df)
    have = tuple(t for t in targets if t in df.columns and df[t].notna().any())
    if not have:
        raise ValueError(f"none of the target columns {targets} were found in the KIT files")
    return finalise(df, "kit", have, tire_id)
