"""Tests for the tyre-degradation UDE (Experiment 5).

The structural guarantees — monotone wear, bounded graining, non-negative rates — must
hold for arbitrary weights, exactly as elsewhere in the framework.
"""

import numpy as np
import pytest
import torch

from tire_nn.data.lap_degradation import (
    STINT_COLUMNS,
    make_synthetic_stints,
    stint_tensors,
    validate_stint_schema,
)
from tire_nn.models.degradation_ude import (
    BlackBoxDegradationModel,
    LapDegradationUDE,
    LinearDegradationModel,
)
from conftest import randomize_


@pytest.fixture(scope="module")
def stints():
    df = make_synthetic_stints(n_sessions=2, n_drivers=3, seed=0)
    batch, compounds = stint_tensors(df)
    return df, batch, compounds


# --- data contract ---------------------------------------------------------

def test_synthetic_stints_satisfy_the_schema(stints):
    df, _, _ = stints
    validate_stint_schema(df)
    for column in STINT_COLUMNS:
        assert column in df.columns


def test_schema_rejects_celsius_temperatures(stints):
    df, _, _ = stints
    bad = df.copy()
    bad["track_temp"] = bad["track_temp"] - 273.15
    with pytest.raises(ValueError, match="kelvin"):
        validate_stint_schema(bad)


def test_tyre_age_resets_at_every_stint(stints):
    df, _, _ = stints
    firsts = df.sort_values("lap_number").groupby(["session_id", "driver", "stint"])["tyre_age"].first()
    assert (firsts == 0).all()


def test_ground_truth_wear_is_monotone_within_a_stint(stints):
    df, _, _ = stints
    for _, g in df.groupby(["session_id", "driver", "stint"]):
        w = g.sort_values("tyre_age")["wear"].to_numpy()
        assert np.all(np.diff(w) >= -1e-12)


def test_ground_truth_graining_is_non_monotone_somewhere(stints):
    """The point of the experiment: graining forms and then cleans up, so a model that
    is monotone in tyre age cannot represent it."""
    df, _, _ = stints
    rises_then_falls = False
    for _, g in df.groupby(["session_id", "driver", "stint"]):
        series = g.sort_values("tyre_age")["graining"].to_numpy()
        if len(series) > 4 and series.max() > 0.05 and series[-1] < series.max() - 1e-3:
            rises_then_falls = True
            break
    assert rises_then_falls


def test_stint_tensors_are_padded_and_masked(stints):
    df, batch, compounds = stints
    n_stints = df.groupby(["session_id", "driver", "stint"]).ngroups
    assert batch["lap_time"].shape[0] == n_stints
    assert batch["mask"].sum() == len(df)
    assert batch["compound"].shape == (n_stints,)
    assert int(batch["n_pace_groups"]) == df.groupby(["session_id", "driver"]).ngroups


# --- model guarantees ------------------------------------------------------

def test_wear_is_monotone_for_adversarial_weights(stints):
    _, batch, compounds = stints
    model = randomize_(LapDegradationUDE(len(compounds), n_pace_groups=int(batch["n_pace_groups"])),
                       std=8.0)
    out = model(batch)
    assert torch.all(torch.diff(out["wear"], dim=-1) >= -1e-9)


def test_graining_stays_in_the_unit_interval_for_adversarial_weights(stints):
    _, batch, compounds = stints
    model = randomize_(LapDegradationUDE(len(compounds), n_pace_groups=int(batch["n_pace_groups"])),
                       std=20.0)
    g = model(batch)["graining"]
    assert float(g.min()) >= 0.0 and float(g.max()) <= 1.0


def test_state_starts_at_zero_on_a_fresh_tyre(stints):
    _, batch, compounds = stints
    model = randomize_(LapDegradationUDE(len(compounds)), std=5.0)
    out = model(batch)
    assert torch.allclose(out["wear"][:, 0], torch.zeros_like(out["wear"][:, 0]))
    assert torch.allclose(out["graining"][:, 0], torch.zeros_like(out["graining"][:, 0]))


def test_bounded_parameters_stay_in_range(stints):
    _, _, compounds = stints
    model = randomize_(LapDegradationUDE(len(compounds)), std=30.0)
    assert 0.0 <= float(model.c_fuel().detach()) <= 8.0
    assert 0.0 <= float(model.a_grain().detach()) <= 5.0


def test_graining_can_be_disabled(stints):
    _, batch, compounds = stints
    model = randomize_(LapDegradationUDE(len(compounds), enable_graining=False), std=5.0)
    out = model(batch)
    assert torch.allclose(out["graining"], torch.zeros_like(out["graining"]))
    assert float(out["wear"].max()) > 0.0


@pytest.mark.parametrize("cls", [LapDegradationUDE, LinearDegradationModel,
                                 BlackBoxDegradationModel])
def test_all_models_share_the_batch_interface(cls, stints):
    _, batch, compounds = stints
    model = cls(len(compounds), n_pace_groups=int(batch["n_pace_groups"]))
    out = model(batch)
    assert out["lap_time"].shape == batch["lap_time"].shape
    assert torch.isfinite(out["lap_time"]).all()


def test_linear_model_cannot_represent_non_monotone_degradation(stints):
    """Structural limitation, stated as a test so it cannot be forgotten."""
    _, batch, compounds = stints
    model = LinearDegradationModel(len(compounds))
    out = model(batch)["lap_time"].detach()
    # Within the valid part of a stint the prediction is affine in lap number (tyre age
    # and fuel fraction both advance by a constant per lap), so second differences
    # vanish. Padding is excluded: padded entries reset tyre_age to 0 and would show up
    # as a large artificial curvature.
    worst = 0.0
    for i in range(out.shape[0]):
        n = int(batch["mask"][i].sum())
        if n >= 4:
            worst = max(worst, float(torch.diff(out[i, :n], n=2).abs().max()))
    assert worst < 1e-3, f"linear model is not affine along a stint (max curvature {worst:.2e})"


def test_ude_recovers_known_dynamics_end_to_end():
    """Train briefly on synthetic stints and check the latent states correlate with truth."""
    df = make_synthetic_stints(n_sessions=6, n_drivers=4, seed=1)
    batch, compounds = stint_tensors(df)
    model = LapDegradationUDE(len(compounds), n_pace_groups=int(batch["n_pace_groups"]),
                              base_lap_time=float(df["lap_time"].median()))
    params = [p for n, p in model.named_parameters() if n != "pace"]
    opt = torch.optim.Adam(params, lr=0.02)

    def fit_offsets():
        with torch.no_grad():
            model.pace.data = torch.zeros(int(batch["n_pace_groups"]))
            residual = batch["lap_time"] - model(batch)["lap_time"]
            mask = batch["mask"].float()
            totals = torch.zeros(int(batch["n_pace_groups"]))
            counts = torch.zeros(int(batch["n_pace_groups"]))
            totals.index_add_(0, batch["pace_group"], (residual * mask).sum(-1))
            counts.index_add_(0, batch["pace_group"], mask.sum(-1))
            model.pace.data = totals / torch.clamp(counts, min=1.0)

    losses = []
    for _ in range(250):
        fit_offsets()
        opt.zero_grad(set_to_none=True)
        out = model(batch)
        loss = (((out["lap_time"] - batch["lap_time"]) ** 2)[batch["mask"]]).mean()
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))

    assert losses[-1] < losses[0]
    with torch.no_grad():
        out = model(batch)
    mask = batch["mask"]
    corr = np.corrcoef(out["wear"][mask].numpy(), batch["wear"][mask].numpy())[0, 1]
    assert corr > 0.5, f"recovered wear correlates only {corr:.2f} with the truth"
    # The fuel coefficient is a known physical quantity: 0.035 s/kg on 100 kg.
    assert 2.0 < float(model.c_fuel().detach()) < 5.0
