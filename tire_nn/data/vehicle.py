"""Vehicle-level data: canonical schema, synthetic generator and dataset (PLAN.md §4.2).

Vehicle-level rows carry what an instrumented car actually logs — IMU accelerations,
yaw rate, wheel speeds, steering — and *not* tire forces. Identifying a tire model
from this is the point of Experiment 3.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from tire_nn.physics.pacejka import MFParams, MagicFormulaTire
from tire_nn.physics.vehicle_dynamics import VehicleParams, corner_velocities

__all__ = [
    "VEHICLE_COLUMNS",
    "WHEEL_COLUMNS",
    "validate_vehicle_schema",
    "make_synthetic_vehicle",
    "VehicleDataset",
    "DEFAULT_VEHICLE",
]

VEHICLE_COLUMNS = ("t", "sequence_id", "vx", "vy", "r", "ax", "ay", "r_dot", "delta",
                   "tire_id", "source")
WHEEL_COLUMNS = tuple(f"omega_{w}" for w in ("FL", "FR", "RL", "RR"))

#: A small-scale racing platform in the F1TENTH/RoboRacer class, scaled up to a light
#: prototype car. Replaced by the real geometry when a dataset provides it.
DEFAULT_VEHICLE = VehicleParams(m=1200.0, Iz=1500.0, lf=1.3, lr=1.4,
                                t_f=1.6, t_r=1.6, h_cg=0.45, R_e=0.32)


def validate_vehicle_schema(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in VEHICLE_COLUMNS + WHEEL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing vehicle columns {missing}")
    if df["vx"].min() < 0:
        raise ValueError("vx must be non-negative for the slip definitions used here")
    return df


def make_synthetic_vehicle(
    n_sequences: int = 8,
    T: int = 1500,
    dt: float = 0.005,
    vp: VehicleParams = DEFAULT_VEHICLE,
    mu_values=(1.0,),
    tire_ids=("baseline",),
    vx0_range: tuple[float, float] = (12.0, 30.0),
    steer_amp: float = 0.09,
    kappa_amp: float = 0.04,
    noise_imu: float = 0.05,
    seed: int = 0,
) -> pd.DataFrame:
    """Simulate a car with a **known** Magic Formula tire and log only vehicle signals.

    The ground-truth tire is never written to the file: the experiment must recover it
    from ``ax, ay, r_dot`` alone. Each sequence gets a friction level and a tire id, so
    a whole friction/tire condition can be held out for the generalisation test.

    Excitation is a two-tone steering sweep plus a slower longitudinal slip command —
    enough to visit combined slip, which pure sinusoidal steering never does.
    """
    from tire_nn.models.four_wheel_vehicle import FourWheelVehicle

    rng = np.random.default_rng(seed)
    frames = []
    for s in range(n_sequences):
        mu = float(mu_values[s % len(mu_values)])
        tire_id = tire_ids[s % len(tire_ids)]
        tire = MagicFormulaTire(MFParams(B=9.0, C=1.6, E=0.4, mu=mu, k_mu=0.08),
                                MFParams(B=8.5, C=1.5, E=0.4, mu=mu, k_mu=0.08))
        veh = FourWheelVehicle(tire, vp, load_transfer="measured")

        f1, f2 = rng.uniform(0.15, 0.5), rng.uniform(0.7, 1.6)
        phase = rng.uniform(0, 2 * np.pi)
        vx = torch.tensor([float(rng.uniform(*vx0_range))])
        vy = torch.zeros(1)
        r = torch.zeros(1)
        ax_prev = torch.zeros(1)
        ay_prev = torch.zeros(1)

        rows = []
        for i in range(T):
            t = i * dt
            delta = torch.tensor([steer_amp * (np.sin(2 * np.pi * f1 * t) +
                                               0.4 * np.sin(2 * np.pi * f2 * t + phase))])
            kappa_cmd = kappa_amp * np.sin(2 * np.pi * 0.11 * t + phase)

            # Wheel speeds consistent with the commanded slip at each corner: the
            # longitudinal slip command is realised through omega, exactly as a
            # traction/brake controller would, so kappa is never prescribed directly.
            delta4 = torch.stack([delta, delta, torch.zeros_like(delta), torch.zeros_like(delta)], dim=-1)
            vxc, vyc = corner_velocities(vx, vy, r, vp)
            c, sn = torch.cos(delta4), torch.sin(delta4)
            vx_w = vxc * c + vyc * sn
            omega = (1.0 + kappa_cmd) * vx_w / vp.R_e

            out = veh(vx, vy, r, delta, omega, ax_meas=ax_prev, ay_meas=ay_prev)
            ax, ay, r_dot = out["ax"], out["ay"], out["r_dot"]

            rows.append({
                "t": t, "sequence_id": s, "vx": float(vx), "vy": float(vy), "r": float(r),
                "ax": float(ax), "ay": float(ay), "r_dot": float(r_dot), "delta": float(delta),
                **{f"omega_{w}": float(omega[0, k]) for k, w in enumerate(("FL", "FR", "RL", "RR"))},
                "mu_ref": mu, "tire_id": tire_id, "source": "synthetic",
            })

            vx = torch.clamp(vx + dt * ax, min=1.0)
            vy = vy + dt * ay
            r = r + dt * r_dot
            ax_prev, ay_prev = ax.detach(), ay.detach()

        frame = pd.DataFrame(rows)
        if noise_imu > 0:
            for c_ in ("ax", "ay"):
                frame[c_] += rng.normal(0, noise_imu, len(frame))
            frame["r_dot"] += rng.normal(0, noise_imu * 0.2, len(frame))
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    df.attrs["vehicle"] = vp
    df.attrs["dt"] = dt
    return validate_vehicle_schema(df)


class VehicleDataset(Dataset):
    """Per-sample vehicle rows -> tensors. Newton-Euler is instantaneous, so no window."""

    def __init__(self, df: pd.DataFrame, tire_index: dict[str, int] | None = None,
                 dtype: torch.dtype = torch.float32):
        validate_vehicle_schema(df)
        self.df = df.reset_index(drop=True)
        self.dtype = dtype
        self.tire_index = tire_index or {n: i for i, n in enumerate(sorted(df["tire_id"].unique()))}
        cols = ("vx", "vy", "r", "delta", "ax", "ay", "r_dot")
        self.data = {c: torch.as_tensor(self.df[c].to_numpy(), dtype=dtype) for c in cols}
        self.data["omega"] = torch.stack(
            [torch.as_tensor(self.df[c].to_numpy(), dtype=dtype) for c in WHEEL_COLUMNS], dim=-1)
        self.data["tire_id"] = torch.as_tensor(
            [self.tire_index[t] for t in self.df["tire_id"]], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int) -> dict:
        return {k: v[i] for k, v in self.data.items()}
