"""Stint-level tyre degradation data: canonical schema, synthetic generator, FastF1 adapter.

Tyre degradation is rarely observed directly. What *is* observed, lap after lap, is the
**consequence**: the car gets slower. This module provides the data side of that
indirect observation problem, in the same spirit as the IMU-only vehicle experiment —
the quantity of interest (tyre state) is latent, and only its effect is measured.

Canonical stint schema
----------------------

===============  =====  ======================================================
column           unit   meaning
===============  =====  ======================================================
``session_id``   -      race/session identifier
``driver``       -      driver or car identifier
``stint``        -      stint index within the session (a pit stop ends a stint)
``lap_number``   -      lap number within the session
``tyre_age``     laps   laps on this set of tyres, 0 on the out-lap of a new set
``compound``     -      compound name (soft / medium / hard / ...)
``lap_time``     s      lap time
``track_temp``   K      track surface temperature
``air_temp``     K      air temperature
``fuel_frac``    -      fraction of the starting fuel load still on board, 1 -> 0
``is_valid``     -      False for laps under safety car, in/out laps, traffic
``source``       -      dataset tag
===============  =====  ======================================================

The tyre state itself is **not** a column, because nobody measures it. That is the
point of the experiment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "STINT_COLUMNS",
    "validate_stint_schema",
    "make_synthetic_stints",
    "load_fastf1_stints",
    "WET_COMPOUNDS",
    "stint_tensors",
]

STINT_COLUMNS = ("session_id", "driver", "stint", "lap_number", "tyre_age", "compound",
                 "lap_time", "track_temp", "air_temp", "fuel_frac", "is_valid", "source")

#: Reference compound behaviour for the synthetic generator. Ordering (soft fastest but
#: degrading quickest) matches how compounds are designed; the numbers are plausible
#: rather than measured, and the generator is labelled synthetic everywhere it is used.
SYNTHETIC_COMPOUNDS = {
    #                 base pace [s]  wear rate  graining susceptibility
    "soft":           (90.0,         0.030,     1.6),
    "medium":         (90.7,         0.018,     1.0),
    "hard":           (91.5,         0.011,     0.5),
}


def validate_stint_schema(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in STINT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing stint columns {missing}; got {list(df.columns)}")
    if (df["tyre_age"] < 0).any():
        raise ValueError("tyre_age must be non-negative")
    if df["track_temp"].max() < 200:
        raise ValueError("track_temp looks like degrees Celsius — the schema is kelvin")
    if not (0.0 <= df["fuel_frac"].min() and df["fuel_frac"].max() <= 1.0 + 1e-6):
        raise ValueError("fuel_frac must be a fraction in [0, 1]")
    return df


def make_synthetic_stints(
    n_sessions: int = 6,
    n_drivers: int = 4,
    laps: int = 55,
    fuel_effect: float = 0.035,
    fuel_mass: float = 100.0,
    track_temp_range: tuple[float, float] = (298.0, 325.0),
    graining_threshold: float = 310.0,
    noise: float = 0.25,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate stint data from a **known** degradation model, for validation.

    The ground-truth latent dynamics, per lap :math:`\\lambda`:

    .. math::

        \\frac{\\mathrm{d}w}{\\mathrm{d}\\lambda} = r_w(\\text{compound}) \\cdot
            \\Big(1 + \\beta\\,\\frac{T_{track} - T_{ref}}{10}\\Big)

        \\frac{\\mathrm{d}g}{\\mathrm{d}\\lambda} = (1-g) R_{form} - g R_{clean}

    with graining forming on a **cold** track (below ``graining_threshold``) and early in
    a stint, and cleaning up once the track is warm. Lap time is then

    .. math::
        t = t_{ref}(\\text{compound}) + c_{fuel}\\,m_{fuel}\\,\\phi
            + a_w w + a_g g + \\varepsilon

    Returned frames contain the true ``wear``/``graining`` columns so a model can be
    scored against them, but the training code must not use them — only ``lap_time``.
    """
    rng = np.random.default_rng(seed)
    compounds = list(SYNTHETIC_COMPOUNDS)
    rows = []

    for s in range(n_sessions):
        track_temp = float(rng.uniform(*track_temp_range))
        air_temp = track_temp - float(rng.uniform(5.0, 12.0))
        session_pace = float(rng.normal(0.0, 0.6))          # circuit / conditions offset

        for d in range(n_drivers):
            driver_pace = float(rng.normal(0.0, 0.25))
            lap, stint = 1, 0
            # Two to three stints per race.
            stint_lengths: list[int] = []
            remaining = laps
            while remaining > 0:
                if remaining <= 32:
                    stint_lengths.append(remaining)      # last stint takes the rest
                    remaining = 0
                else:
                    length = min(int(rng.integers(12, 31)), remaining - 12)
                    stint_lengths.append(length)
                    remaining -= length

            for length in stint_lengths:
                compound = compounds[int(rng.integers(0, len(compounds)))]
                base, wear_rate, grain_susceptibility = SYNTHETIC_COMPOUNDS[compound]
                w = 0.0
                g = 0.0
                for age in range(length):
                    fuel_frac = max(0.0, 1.0 - (lap - 1) / laps)
                    t = (base + session_pace + driver_pace
                         + fuel_effect * fuel_mass * fuel_frac
                         + 1.10 * w                       # wear -> lap time [s]
                         + 0.85 * g                       # graining -> lap time [s]
                         + rng.normal(0.0, noise))
                    rows.append({
                        "session_id": f"synthetic_{s:02d}", "driver": f"CAR{d:02d}",
                        "stint": stint, "lap_number": lap, "tyre_age": age,
                        "compound": compound, "lap_time": t,
                        "track_temp": track_temp, "air_temp": air_temp,
                        "fuel_frac": fuel_frac, "is_valid": age > 0,
                        "wear": w, "graining": g, "source": "synthetic",
                    })

                    # --- ground-truth latent dynamics, one lap step ---
                    thermal = 1.0 + 0.08 * (track_temp - 310.0) / 10.0
                    w += wear_rate * max(thermal, 0.2)

                    cold = max(graining_threshold - track_temp, 0.0) / 10.0
                    early = np.exp(-age / 6.0)             # worst on a fresh, cold tyre
                    R_form = 0.55 * grain_susceptibility * cold * early
                    R_clean = 0.05 + 0.11 * max(track_temp - graining_threshold, 0.0) / 10.0
                    g = float(np.clip(g + (1 - g) * R_form - g * R_clean, 0.0, 1.0))
                    lap += 1
                stint += 1

    df = pd.DataFrame(rows)
    df.attrs["fuel_effect"] = fuel_effect
    df.attrs["fuel_mass"] = fuel_mass
    return validate_stint_schema(df)


#: Wet-weather compounds. Running on these is a different regime: the tyre construction
#: differs, the track state is changing underneath the car, and lap-time evolution is
#: dominated by the drying line rather than by tyre degradation. Mixing them into a
#: degradation fit confounds it badly — in this project a wet race landing in the test
#: split moved the held-out error by a factor of four.
WET_COMPOUNDS = ("intermediate", "wet")


def load_fastf1_stints(
    root: str | Path = "data/raw",
    subdir: str = "f1_stints",
    drop_invalid: bool = True,
    max_lap_time: float | None = 1.10,
    dry_only: bool = True,
) -> pd.DataFrame:
    """Load stint data prepared by ``scripts/download_f1_stints.py``.

    **Type: real measurement** (official F1 timing data, via the MIT-licensed FastF1
    package). Note what this data is and is not: lap time is an *indirect, aggregate*
    observation of tyre state, confounded by fuel load, traffic, driver input, track
    evolution and weather. It is the right data for asking "can the degradation
    *dynamics* be identified from their consequence", and the wrong data for asking
    "what is this tyre's friction coefficient".

    ``max_lap_time`` drops laps slower than that multiple of the driver's stint median —
    a crude but effective filter for traffic, safety cars and mistakes, which otherwise
    dominate the residual.

    ``dry_only`` drops wet and intermediate running (see :data:`WET_COMPOUNDS`). Keep it
    on unless you are specifically modelling wet-weather behaviour.
    """
    path = Path(root) / subdir
    files = sorted(path.glob("*.parquet")) + sorted(path.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"no prepared F1 stint data under {path}.\n"
            "Fetch it with:\n"
            "    python -m pip install fastf1\n"
            "    python scripts/download_f1_stints.py --seasons 2023 --rounds 1 2 3\n"
            "See scripts/README.md and the docs page 'Degradation from stint data'."
        )
    df = pd.concat([pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
                    for f in files], ignore_index=True)

    if drop_invalid and "is_valid" in df.columns:
        df = df[df["is_valid"]].reset_index(drop=True)
    if dry_only:
        wet = df["compound"].isin(WET_COMPOUNDS)
        if wet.any():
            print(f"[f1_stints] dropped {int(wet.sum())} wet/intermediate laps "
                  f"(different regime — pass dry_only=False to keep them)")
            df = df[~wet].reset_index(drop=True)
    if max_lap_time is not None:
        median = df.groupby(["session_id", "driver", "stint"])["lap_time"].transform("median")
        keep = df["lap_time"] <= max_lap_time * median
        dropped = int((~keep).sum())
        if dropped:
            print(f"[f1_stints] dropped {dropped} laps slower than "
                  f"{max_lap_time:.2f}x the stint median (traffic / safety car / errors)")
        df = df[keep].reset_index(drop=True)
    return validate_stint_schema(df)


def stint_tensors(df: pd.DataFrame, compounds: list[str] | None = None):
    """Group a stint frame into padded per-stint tensors for sequence training.

    Returns ``(batch, compounds)`` where ``batch`` is a dict of tensors shaped
    ``(n_stints, max_len)``, plus a boolean ``mask``, a ``compound`` index per stint and
    a ``pace_group`` index per stint identifying the (session, driver) it belongs to.
    """
    import torch

    compounds = compounds or sorted(df["compound"].unique())
    index = {c: i for i, c in enumerate(compounds)}
    groups = list(df.groupby(["session_id", "driver", "stint"], sort=True))
    max_len = max(len(g) for _, g in groups)

    def pad(values, fill=0.0):
        out = np.full((len(groups), max_len), fill, dtype=np.float32)
        for i, v in enumerate(values):
            out[i, :len(v)] = v
        return torch.as_tensor(out)

    ordered = [g.sort_values("tyre_age") for _, g in groups]

    # Per (session, driver) index. Car pace and circuit differences are far larger than
    # the degradation signal — a second a lap between cars, against about a second of
    # degradation over a whole stint — so a model without this term fits the pace
    # difference and reports nonsense degradation.
    pace_keys = sorted({(k[0], k[1]) for k, _ in groups})
    pace_index = {k: i for i, k in enumerate(pace_keys)}
    batch = {
        "lap_time": pad([g["lap_time"].to_numpy() for g in ordered]),
        "tyre_age": pad([g["tyre_age"].to_numpy() for g in ordered]),
        "track_temp": pad([g["track_temp"].to_numpy() for g in ordered], 300.0),
        "air_temp": pad([g["air_temp"].to_numpy() for g in ordered], 295.0),
        "fuel_frac": pad([g["fuel_frac"].to_numpy() for g in ordered]),
        "mask": pad([np.ones(len(g)) for g in ordered]).bool(),
        "compound": torch.as_tensor(
            np.array([index[g["compound"].iloc[0]] for g in ordered], dtype=np.int64)),
        "pace_group": torch.as_tensor(
            np.array([pace_index[(k[0], k[1])] for k, _ in groups], dtype=np.int64)),
    }
    batch["n_pace_groups"] = len(pace_index)
    for extra in ("wear", "graining"):
        if extra in df.columns:
            batch[extra] = pad([g[extra].to_numpy() for g in ordered])
    return batch, compounds
