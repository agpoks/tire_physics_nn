#!/usr/bin/env python3
"""Fetch Formula 1 stint data into the canonical stint schema.

Type: **real measurement** — official F1 timing data, retrieved with the MIT-licensed
FastF1 package (no API key required). FastF1 caches raw responses locally, so the first
run for a session is slow and subsequent runs are offline.

    python -m pip install fastf1
    python scripts/download_f1_stints.py --seasons 2023 --rounds 1 2 3

What the data is, and is not
----------------------------
Lap time is an *indirect, aggregate and confounded* observation of tyre state: fuel
load, traffic, driver input, track evolution, wind and safety cars all move it. It is
the right data for asking whether degradation *dynamics* can be identified from their
consequence; it is the wrong data for asking what a tyre's friction coefficient is.

Tyre temperature is not published, so the thermal gating in the model uses track and air
temperature as proxies. That limitation is real and is stated in the results.

Licence and terms are F1's, not this project's — check them before redistributing
anything you download.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

C2K = 273.15


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seasons", type=int, nargs="+", default=[2023])
    ap.add_argument("--rounds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--root", default="data/raw")
    ap.add_argument("--cache", default="data/raw/fastf1_cache")
    args = ap.parse_args()

    try:
        import fastf1
    except ImportError:
        print("fastf1 is not installed.\n")
        print("  python -m pip install fastf1")
        print("  python scripts/download_f1_stints.py --seasons 2023 --rounds 1 2 3\n")
        print("Without it, the degradation experiment falls back to synthetic stint data,")
        print("which is generated from a known model and labelled as synthetic throughout.")
        return 1

    import pandas as pd

    out_dir = Path(args.root) / "f1_stints"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache))

    written = 0
    for season in args.seasons:
        for rnd in args.rounds:
            try:
                session = fastf1.get_session(season, rnd, "R")
                session.load(telemetry=False, weather=True, messages=False)
            except Exception as exc:                     # noqa: BLE001 - network/data issues
                print(f"[skip] {season} round {rnd}: {exc}")
                continue

            laps = session.laps.copy()
            if laps.empty:
                print(f"[skip] {season} round {rnd}: no laps")
                continue

            total_laps = float(laps["LapNumber"].max())
            weather = session.weather_data
            rows = []
            for _, lap in laps.iterlaps():
                if pd.isna(lap["LapTime"]) or pd.isna(lap["TyreLife"]):
                    continue
                # Nearest weather sample to this lap.
                if weather is not None and not weather.empty:
                    idx = (weather["Time"] - lap["LapStartTime"]).abs().idxmin()
                    track_t = float(weather.loc[idx, "TrackTemp"]) + C2K
                    air_t = float(weather.loc[idx, "AirTemp"]) + C2K
                else:
                    track_t, air_t = 30.0 + C2K, 22.0 + C2K

                lap_number = float(lap["LapNumber"])
                rows.append({
                    "session_id": f"{season}_{rnd:02d}_{session.event['EventName']}".replace(" ", "_"),
                    "driver": str(lap["Driver"]),
                    "stint": int(lap["Stint"]) if not pd.isna(lap["Stint"]) else 0,
                    "lap_number": lap_number,
                    "tyre_age": float(lap["TyreLife"]) - 1.0,   # FastF1 counts from 1
                    "compound": str(lap["Compound"]).lower(),
                    "lap_time": lap["LapTime"].total_seconds(),
                    "track_temp": track_t,
                    "air_temp": air_t,
                    # Fuel burns off roughly linearly with distance covered.
                    "fuel_frac": max(0.0, 1.0 - (lap_number - 1.0) / max(total_laps, 1.0)),
                    "is_valid": bool(lap["IsAccurate"]) and not pd.isna(lap["Compound"]),
                    "source": "fastf1",
                })

            if not rows:
                continue
            df = pd.DataFrame(rows)
            df["tyre_age"] = df["tyre_age"].clip(lower=0.0)
            name = f"{season}_{rnd:02d}.parquet"
            try:
                df.to_parquet(out_dir / name)
            except Exception:                            # pyarrow not installed
                name = name.replace(".parquet", ".csv")
                df.to_csv(out_dir / name, index=False)
            print(f"[ok] {name}: {len(df)} laps, {df['driver'].nunique()} drivers, "
                  f"compounds {sorted(df['compound'].unique())}")
            written += 1

    if not written:
        print("\nNothing written. Check the season/round numbers and your connection.")
        return 1
    print(f"\n{written} session(s) written to {out_dir}")
    print("Load with:  from tire_nn.data.lap_degradation import load_fastf1_stints")
    return 0


if __name__ == "__main__":
    sys.exit(main())
