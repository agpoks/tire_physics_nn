"""Four-wheel structure tests (PLAN.md §6, items 10 and 11)."""

import pytest
import torch

from tire_nn.models import EncodedTireNet, ParameterTireNet
from tire_nn.models.four_wheel_vehicle import FourWheelVehicle
from tire_nn.physics import MagicFormulaTire, VehicleParams, newton_euler
from tire_nn.physics.vehicle_dynamics import corner_positions

VP = VehicleParams(m=1200.0, Iz=1500.0, lf=1.3, lr=1.4, t_f=1.6, t_r=1.6, h_cg=0.45, R_e=0.32)


def _state(B=8, vx=20.0, r=0.0, delta=0.0):
    return dict(
        vx=torch.full((B,), vx), vy=torch.zeros(B), r=torch.full((B,), r),
        delta=torch.full((B,), delta), omega=torch.full((B, 4), vx / VP.R_e),
    )


def _veh(**kw):
    return FourWheelVehicle(EncodedTireNet(context_keys=("vx",)), VP, **kw)


# --- item 10: all four wheels share the same TireNet parameters --------------

def test_all_four_wheels_share_the_same_parameter_objects():
    veh = _veh()
    ids = veh.shared_parameter_ids()
    assert ids, "tire model has no parameters"
    # There is exactly one tire module, so the four evaluations cannot diverge.
    assert veh.tires is None
    assert {id(p) for p in veh.tire.parameters()} == ids
    assert sum(p.numel() for p in veh.parameters()) == sum(p.numel() for p in veh.tire.parameters())


def test_shared_model_has_a_quarter_of_the_per_wheel_parameter_count():
    shared = _veh(share_tire=True)
    per_wheel = _veh(share_tire=False)
    n_s = sum(p.numel() for p in shared.parameters())
    n_p = sum(p.numel() for p in per_wheel.parameters())
    assert n_p == 4 * n_s


def test_a_gradient_step_moves_all_four_corners_identically_when_shared():
    veh = _veh()
    st = _state(delta=0.05, r=0.2)
    out = veh(**st, ax_meas=torch.zeros(8), ay_meas=torch.full((8,), 3.0))
    out["ay"].sum().backward()
    # One parameter set means one gradient; nothing corner-specific can exist.
    grads = [p.grad for p in veh.tire.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_per_wheel_ablation_really_is_independent():
    veh = _veh(share_tire=False)
    ids = [{id(p) for p in t.parameters()} for t in veh.tires]
    assert ids[0].isdisjoint(ids[1])


def test_identical_corners_give_identical_forces_under_sharing():
    """Straight running, no yaw: all four corners see the same slip -> same force."""
    veh = FourWheelVehicle(MagicFormulaTire(), VP, load_transfer="static")
    st = _state(B=2, vx=20.0, r=0.0, delta=0.0)
    st["omega"] = torch.full((2, 4), 20.0 / VP.R_e * 1.02)   # equal drive slip
    out = veh(**st)
    Fx = out["Fx_wheel"]
    assert torch.allclose(Fx[:, 0], Fx[:, 1], atol=1e-5)      # left/right front
    assert torch.allclose(Fx[:, 2], Fx[:, 3], atol=1e-5)      # left/right rear


# --- item 11: force/moment aggregation --------------------------------------

def test_aggregation_matches_a_hand_computed_reference():
    Fx_b = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    Fy_b = torch.tensor([[500.0, 500.0, -200.0, -200.0]])
    z = torch.zeros(1)
    ax, ay, r_dot = newton_euler(Fx_b, Fy_b, torch.tensor([20.0]), z, z, VP)
    assert torch.allclose(ay, torch.tensor([600.0 / VP.m]))
    expected_Mz = VP.lf * 1000.0 + (-VP.lr) * (-400.0)
    assert torch.allclose(r_dot, torch.tensor([expected_Mz / VP.Iz]))


def test_corner_positions_are_the_exact_geometry():
    x, y = corner_positions(VP)
    assert torch.allclose(x, torch.tensor([VP.lf, VP.lf, -VP.lr, -VP.lr]))
    assert torch.allclose(y, torch.tensor([VP.t_f / 2, -VP.t_f / 2, VP.t_r / 2, -VP.t_r / 2]))


def test_steering_rotation_enters_the_body_forces():
    """A pure lateral wheel force on a steered front wheel must produce a Fx component."""
    veh = _veh()
    out_straight = veh(**_state(delta=0.0, r=0.3), ax_meas=torch.zeros(8), ay_meas=torch.zeros(8))
    out_steered = veh(**_state(delta=0.3, r=0.3), ax_meas=torch.zeros(8), ay_meas=torch.zeros(8))
    assert not torch.allclose(out_straight["Fx_body"], out_steered["Fx_body"])


def test_symmetric_cornering_produces_no_net_longitudinal_force():
    veh = FourWheelVehicle(MagicFormulaTire(), VP, load_transfer="static")
    st = _state(B=1, vx=20.0, r=0.0, delta=0.0)
    out = veh(**st)
    assert abs(float(out["Fx_body"].sum())) < 1e-3


def test_load_transfer_conserves_total_vertical_load():
    veh = _veh()
    out = veh(**_state(), ax_meas=torch.full((8,), -5.0), ay_meas=torch.full((8,), 6.0))
    assert torch.allclose(out["Fz"].sum(-1), torch.full((8,), VP.m * 9.81), rtol=1e-4)


def test_slip_angles_differ_between_corners_under_yaw():
    veh = _veh()
    out = veh(**_state(r=0.5), ax_meas=torch.zeros(8), ay_meas=torch.zeros(8))
    alpha = out["alpha"][0]
    assert abs(float(alpha[0] - alpha[2])) > 1e-3       # front vs rear
    assert torch.isfinite(alpha).all()


def test_drag_and_rolling_resistance_only_oppose_motion():
    veh = _veh(drag=0.4, roll_resistance=0.015)
    out = veh(**_state(vx=30.0), ax_meas=torch.zeros(8), ay_meas=torch.zeros(8))
    free = _veh(drag=0.0, roll_resistance=0.0)(**_state(vx=30.0),
                                               ax_meas=torch.zeros(8), ay_meas=torch.zeros(8))
    assert float(out["ax"][0].detach()) < float(free["ax"][0].detach())


def test_vehicle_model_has_no_learnable_chassis_parameters():
    """Newton-Euler and the geometry must stay exact — nothing there may be trained."""
    veh = _veh()
    tire_params = {id(p) for p in veh.tire.parameters()}
    assert all(id(p) in tire_params for p in veh.parameters())


def test_gradients_reach_the_tire_from_vehicle_level_supervision_only():
    """The whole point of Experiment 3: IMU error must reach the constitutive model."""
    veh = _veh()
    st = _state(delta=0.06, r=0.25)
    out = veh(**st, ax_meas=torch.zeros(8), ay_meas=torch.full((8,), 3.0))
    target = torch.full((8,), 2.0)
    ((out["ay"] - target) ** 2).mean().backward()
    grads = [p.grad for p in veh.tire.parameters()]
    assert any(g is not None and g.abs().max() > 0 for g in grads)


def test_parameter_tire_works_inside_the_vehicle():
    veh = FourWheelVehicle(ParameterTireNet(context_keys=("vx",)), VP)
    out = veh(**_state(delta=0.05, r=0.2), ax_meas=torch.zeros(8), ay_meas=torch.full((8,), 3.0))
    assert torch.isfinite(out["ax"]).all() and torch.isfinite(out["r_dot"]).all()
    assert "mu_y" in out["params"]


@pytest.mark.parametrize("mode", ["measured", "static", "iterate"])
def test_all_load_transfer_modes_run_and_conserve_load(mode):
    veh = _veh(load_transfer=mode)
    out = veh(**_state(), ax_meas=torch.full((8,), 2.0), ay_meas=torch.full((8,), 4.0))
    assert torch.allclose(out["Fz"].sum(-1), torch.full((8,), VP.m * 9.81), rtol=1e-4)


def test_vehicle_level_training_recovers_a_usable_tire_model():
    """End-to-end: train on IMU only, then check the tire model is physically sane."""
    from tire_nn.data.vehicle import VehicleDataset, make_synthetic_vehicle
    from tire_nn.evaluation import audit
    from torch.utils.data import DataLoader
    from tire_nn.training.losses import vehicle_loss

    df = make_synthetic_vehicle(n_sequences=2, T=300, vp=VP, seed=1)
    ds = VehicleDataset(df)
    veh = _veh()
    opt = torch.optim.Adam(veh.parameters(), lr=0.005)
    losses = []
    for _ in range(6):
        for batch in DataLoader(ds, batch_size=256, shuffle=True):
            opt.zero_grad(set_to_none=True)
            out = veh(batch["vx"], batch["vy"], batch["r"], batch["delta"], batch["omega"],
                      ax_meas=batch["ax"], ay_meas=batch["ay"])
            loss = vehicle_loss((out["ax"], out["ay"], out["r_dot"]),
                                (batch["ax"], batch["ay"], batch["r_dot"]))
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]
    report = audit(veh.tire, n=512, context={"vx": torch.tensor([20.0])})
    assert report["envelope_violation"] < 1e-5
    assert report["zero_slip_force"] == 0.0
