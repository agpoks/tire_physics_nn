"""Dataset contract tests (PLAN.md §4).

The adapters are the boundary where units and sign conventions are fixed. A silent
error here — degrees read as radians, kPa read as Pa, a flipped lateral force —
produces a model that trains beautifully and is wrong, so the conversion is tested on
synthetic files with deliberately awkward headers.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from tire_nn.data import (
    Normalizer,
    TireDataset,
    adapters,
    make_synthetic,
    make_synthetic_transient,
    split_by_condition,
    split_by_group,
    validate_schema,
)
from tire_nn.data.adapters import ColumnSpec, DatasetNotAvailable, map_columns
from tire_nn.data.common import flip_sign_convention


# --- canonical schema ------------------------------------------------------

def test_synthetic_data_satisfies_the_canonical_schema():
    df = make_synthetic(200, pressure_range=(1.5e5, 2.5e5))
    validate_schema(df, ("Fx", "Fy"))
    assert (df["Fz"] > 0).all()
    assert df["alpha"].abs().max() < np.pi / 2
    assert set(("tire_id", "source")).issubset(df.columns)


def test_schema_validation_catches_degrees_mistaken_for_radians():
    df = make_synthetic(50)
    df["alpha"] = df["alpha"] * 180 / np.pi          # the classic unit bug
    with pytest.raises(ValueError, match="degrees"):
        validate_schema(df, ("Fy",))


def test_schema_validation_catches_non_positive_load():
    df = make_synthetic(50)
    df.loc[0, "Fz"] = -1.0
    with pytest.raises(ValueError, match="Fz"):
        validate_schema(df, ("Fy",))


def test_schema_validation_reports_a_missing_target():
    df = make_synthetic(50).drop(columns=["Fy"])
    with pytest.raises(ValueError, match="Fy"):
        validate_schema(df, ("Fy",))


def test_sign_convention_flip_is_an_involution():
    df = make_synthetic(100)
    assert np.allclose(flip_sign_convention(flip_sign_convention(df))["Fy"], df["Fy"])
    assert np.allclose(flip_sign_convention(df)["alpha"], -df["alpha"])


def test_synthetic_data_follows_the_sae_sign_convention():
    df = make_synthetic(4000, noise=0.0, seed=0)
    pure_lateral = df[df["kappa"].abs() < 0.01]
    positive = pure_lateral[pure_lateral["alpha"] > 0.05]
    assert (positive["Fy"] < 0).mean() > 0.99, "positive alpha must give negative Fy"


# --- column mapping --------------------------------------------------------

def test_column_map_converts_units_and_is_header_insensitive():
    raw = pd.DataFrame({
        "Slip Angle [deg]": [0.0, 5.0, -5.0],
        "F_Z": [1000.0, 1000.0, 1000.0],
        "pressure": [200.0, 200.0, 200.0],           # kPa
    })
    spec = {
        "alpha": ColumnSpec(("slip_angle_deg", "slipangle[deg]", "Slip Angle [deg]"),
                            np.pi / 180, required=True),
        "Fz": ColumnSpec(("Fz",), 1.0, required=True),
        "p": ColumnSpec(("p", "pressure"), 1000.0),
    }
    out = map_columns(raw, spec)
    assert np.isclose(out["alpha"][1], np.deg2rad(5.0))
    assert np.isclose(out["p"][0], 2.0e5)


def test_column_map_reports_missing_required_columns_with_the_candidates():
    raw = pd.DataFrame({"something_else": [1.0]})
    spec = {"Fz": ColumnSpec(("Fz", "vertical_load"), required=True)}
    with pytest.raises(ValueError, match="vertical_load"):
        map_columns(raw, spec)


def test_column_map_applies_offsets_for_temperature():
    raw = pd.DataFrame({"T_tire": [25.0]})
    out = map_columns(raw, {"Ts": ColumnSpec(("T_tire",), 1.0, 273.15)})
    assert np.isclose(out["Ts"][0], 298.15)


def test_adapter_round_trip_through_a_file(tmp_path):
    """Write a file in 'foreign' units and conventions; read it back canonical."""
    from tire_nn.data.kit import load_kit

    truth = make_synthetic(300, seed=2)
    raw = pd.DataFrame({
        "alpha": np.rad2deg(-truth["alpha"]),       # degrees AND opposite convention
        "kappa": -truth["kappa"],
        "Fz": truth["Fz"],
        "Fx": -truth["Fx"],
        "Fy": -truth["Fy"],
        "pressure": 200.0,                          # kPa
    })
    (tmp_path / "kit").mkdir()
    raw.to_csv(tmp_path / "kit" / "run01.csv", index=False)

    df = load_kit(tmp_path, flip_signs=True)
    assert np.allclose(df["alpha"], truth["alpha"], atol=1e-6)
    assert np.allclose(df["Fy"], truth["Fy"], atol=1e-6)
    assert np.isclose(df["p"].iloc[0], 2.0e5)
    assert df["source"].iloc[0] == "kit"


def test_kit_adapter_excludes_the_simulated_driving_cycle_by_default(tmp_path):
    from tire_nn.data.kit import load_kit

    truth = make_synthetic(100, seed=3)
    base = pd.DataFrame({"alpha": np.rad2deg(truth["alpha"]), "kappa": truth["kappa"],
                         "Fz": truth["Fz"], "Fx": truth["Fx"], "Fy": truth["Fy"]})
    (tmp_path / "kit").mkdir()
    base.to_csv(tmp_path / "kit" / "measurement.csv", index=False)
    base.to_csv(tmp_path / "kit" / "slalom_simulation.csv", index=False)

    assert len(load_kit(tmp_path)) == len(base)
    assert len(load_kit(tmp_path, include_simulation=True)) == 2 * len(base)


@pytest.mark.parametrize("name", ["kit", "vetyt", "tum_cargo_bike", "deep_dynamics",
                                  "roboracer", "qmotion"])
def test_every_adapter_fails_with_actionable_instructions(name, tmp_path):
    """A missing dataset must never look like an empty one."""
    with pytest.raises(DatasetNotAvailable) as exc:
        adapters.load(name, root=tmp_path)
    message = str(exc.value)
    assert len(message) > 200
    assert "data/raw" in message or "target directory" in message or str(tmp_path) in message


def test_unknown_dataset_name_is_rejected():
    with pytest.raises(KeyError, match="unknown dataset"):
        adapters.load("not_a_dataset")


# --- datasets and splits ---------------------------------------------------

def test_dataset_static_and_windowed_shapes():
    df = make_synthetic_transient(n_sequences=3, T=100, seed=1)
    static = TireDataset(df, targets=("Fx", "Fy"), context_keys=("vx",))
    assert static[0]["alpha"].shape == ()
    windowed = TireDataset(df, targets=("Fx", "Fy"), context_keys=("vx",), window=40)
    assert windowed[0]["alpha"].shape == (40,)


def test_windows_never_straddle_a_sequence_boundary():
    df = make_synthetic_transient(n_sequences=3, T=60, seed=1)
    ds = TireDataset(df, targets=("Fy",), window=20)
    for idx in ds.windows:
        assert df.loc[idx, "sequence_id"].nunique() == 1


def test_windowed_mode_requires_sequence_ids():
    with pytest.raises(ValueError, match="sequence_id"):
        TireDataset(make_synthetic(100), targets=("Fy",), window=10)


def test_group_split_keeps_sequences_intact():
    df = make_synthetic_transient(n_sequences=10, T=60, seed=0)
    tr, va, te = split_by_group(df, fractions=(0.6, 0.2, 0.2), seed=0)
    ids = [set(x["sequence_id"]) for x in (tr, va, te)]
    assert ids[0].isdisjoint(ids[1]) and ids[0].isdisjoint(ids[2]) and ids[1].isdisjoint(ids[2])
    assert len(tr) + len(va) + len(te) == len(df)


def test_condition_split_holds_out_entire_conditions():
    df = make_synthetic(400, pressure_range=(1.5e5, 2.5e5))
    df["level"] = pd.cut(df["p"], 4, labels=False)
    train, held = split_by_condition(df, "level", [3])
    assert set(held["level"]) == {3}
    assert 3 not in set(train["level"])


def test_normalizer_is_fitted_on_train_only_and_round_trips(tmp_path):
    df = make_synthetic(500)
    tr, _, te = split_by_group(df)
    norm = Normalizer.fit(tr, ("alpha", "Fz", "Fy"))
    path = tmp_path / "norm.json"
    norm.save(path)
    assert Normalizer.load(path).stats == norm.stats
    assert np.isclose(norm.stats["Fz"][0], tr["Fz"].mean())
    assert not np.isclose(norm.stats["Fz"][0], te["Fz"].mean())


def test_tire_index_is_stable_and_shared_across_splits():
    df = pd.concat([make_synthetic(100, tire_id="soft"), make_synthetic(100, tire_id="hard")],
                   ignore_index=True)
    index = {n: i for i, n in enumerate(sorted(df["tire_id"].unique()))}
    ds = TireDataset(df, targets=("Fy",), tire_index=index)
    assert ds.tire_index == {"hard": 0, "soft": 1}
    assert isinstance(ds[0]["context"]["tire_id"], torch.Tensor)


def test_transient_generator_rejects_a_sequence_too_short_for_its_steps():
    with pytest.raises(ValueError, match="too short"):
        make_synthetic_transient(n_sequences=1, T=40, n_steps=4)


def test_transient_generator_produces_the_documented_relaxation():
    """The generated force must actually lag the steady-state force."""
    df = make_synthetic_transient(n_sequences=1, T=400, sigma_y=0.3, noise=0.0, seed=0)
    seq = df[df["sequence_id"] == 0].reset_index(drop=True)
    steps = seq.index[seq["alpha"].diff().abs() > 1e-6]
    assert len(steps) > 0
    i = int(steps[0])
    # Immediately after a slip step the force has not yet reached its new value.
    assert abs(seq.loc[i, "Fy"] - seq.loc[i + 30, "Fy"]) > 1e-6


def test_vehicle_generator_conserves_the_documented_schema():
    from tire_nn.data.vehicle import VEHICLE_COLUMNS, WHEEL_COLUMNS, make_synthetic_vehicle

    df = make_synthetic_vehicle(n_sequences=1, T=50)
    for c in VEHICLE_COLUMNS + WHEEL_COLUMNS:
        assert c in df.columns
    assert (df["vx"] > 0).all()
