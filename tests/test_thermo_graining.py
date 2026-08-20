"""Condition-model tests (PLAN.md §6, items 8 and 9).

Wear irreversibility and graining boundedness must hold for *arbitrary* weights and
over long rollouts — they are structural, not a consequence of training.
"""

import pytest
import torch

from tire_nn.models import EncodedTireNet
from tire_nn.models.thermo_graining_tire import STATE_NAMES, ThermoGrainingTire
from tire_nn.physics.wear import graining_rate, wear_rate
from conftest import randomize_


def _model(**kw):
    return ThermoGrainingTire(EncodedTireNet(context_keys=("vx",)), **kw)


def _drive(T=1500, alpha=0.12, kappa=0.05, vx=40.0, Fz=3000.0):
    ones = torch.ones(1, T)
    return ones * alpha, ones * kappa, ones * Fz, ones * vx


def test_wear_never_decreases_over_a_long_rollout_with_random_weights():
    model = randomize_(_model(), std=5.0)
    F, z, _ = model.rollout_condition(*_drive(T=2000), dt=0.05)
    wear = z[0, :, 2]
    assert torch.all(torch.diff(wear) >= -1e-12)


def test_wear_rate_is_non_negative_for_any_network_output():
    assert torch.all(wear_rate(torch.linspace(-1e3, 1e3, 2001)) >= 0.0)


def test_graining_stays_in_the_unit_interval_with_adversarial_rates():
    model = randomize_(_model(), std=20.0)
    _, z, _ = model.rollout_condition(*_drive(T=2000), dt=0.5)   # deliberately coarse dt
    g = z[..., 3]
    assert float(g.min()) >= 0.0 and float(g.max()) <= 1.0


def test_graining_boundaries_are_invariant_under_the_exact_update():
    model = _model()
    R_form = torch.rand(64) * 10
    R_clean = torch.rand(64) * 10
    for g0 in (torch.zeros(64), torch.ones(64), torch.rand(64)):
        for dt in (1e-3, 1.0, 100.0):
            g = model.graining_step(g0, R_form, R_clean, dt)
            assert float(g.min()) >= 0.0 and float(g.max()) <= 1.0


def test_graining_rate_signs_at_the_boundaries():
    R_form = torch.rand(32) * 3
    R_clean = torch.rand(32) * 3
    assert torch.all(graining_rate(torch.zeros(32), R_form, R_clean) >= 0)
    assert torch.all(graining_rate(torch.ones(32), R_form, R_clean) <= 0)


def test_temperature_stays_finite_and_rises_under_slip():
    model = _model()
    _, z, extra = model.rollout_condition(*_drive(T=2000), dt=0.05)
    Ts, Tc = z[0, :, 0], z[0, :, 1]
    assert torch.isfinite(Ts).all() and torch.isfinite(Tc).all()
    assert float(Ts[-1]) > float(Ts[0]), "sustained slip power must heat the surface"
    assert float(Ts[-1]) > float(Tc[-1]), "the surface is the node being heated"


def test_slip_power_is_never_negative():
    model = randomize_(_model(), std=3.0)
    _, _, extra = model.rollout_condition(*_drive(T=500), dt=0.05)
    assert float(extra["P_slip"].min()) >= 0.0


def test_condition_only_scales_the_friction_ellipse_and_keeps_the_shape_guarantees():
    """Symmetry and zero-slip must survive the condition model."""
    model = randomize_(_model(), std=3.0)
    a = torch.linspace(-0.4, 0.4, 64)
    Fz = torch.full((64,), 2000.0)
    z = model.initial_state(Fz)
    ctx = {"z": z, "vx": torch.full((64,), 30.0)}
    out = model(a, a * 0.5, Fz, ctx)
    out_m = model(-a, -a * 0.5, Fz, ctx)
    assert torch.allclose(out.Fx, -out_m.Fx, atol=1e-5)
    assert torch.allclose(out.Fy, -out_m.Fy, atol=1e-5)
    zero = torch.zeros(64)
    out0 = model(zero, zero, Fz, ctx)
    assert torch.allclose(out0.Fy, zero, atol=1e-6)


def test_effective_friction_drops_with_wear_and_graining():
    model = _model()
    Fz = torch.ones(1)
    clean = model.mu_scale(model.initial_state(Fz, T0=float(model.T_opt())))
    worn = model.mu_scale(torch.tensor([[float(model.T_opt()), 300.0, 1.0, 0.0]]))
    grained = model.mu_scale(torch.tensor([[float(model.T_opt()), 300.0, 0.0, 0.8]]))
    assert float(clean) > float(worn) > 0
    assert float(clean) > float(grained) > 0


def test_temperature_factor_peaks_at_the_learned_optimum():
    model = _model()
    T_opt = float(model.T_opt())
    peak = float(model.temperature_factor(torch.tensor([T_opt])))
    for offset in (-60.0, -20.0, 20.0, 60.0):
        assert float(model.temperature_factor(torch.tensor([T_opt + offset]))) <= peak + 1e-6


def test_condition_states_can_be_disabled_individually():
    model = _model(enable_wear=False, enable_graining=False)
    _, z, _ = model.rollout_condition(*_drive(T=200), dt=0.05)
    assert torch.allclose(z[..., 2], torch.zeros_like(z[..., 2]))
    assert torch.allclose(z[..., 3], torch.zeros_like(z[..., 3]))
    assert float(z[0, -1, 0]) > float(z[0, 0, 0])       # thermal still active


def test_state_names_match_the_state_vector_layout():
    model = _model()
    Fz = torch.ones(3)
    z = model.initial_state(Fz)
    assert z.shape[-1] == len(STATE_NAMES) == 4
    out = model(torch.zeros(3), torch.zeros(3), Fz, {"z": z})
    for name in STATE_NAMES:
        assert name in out.params


def test_synthetic_scenario_shows_the_intended_qualitative_behaviour():
    """The reference generator must actually demonstrate what Experiment 4 claims."""
    from tire_nn.data.graining import make_synthetic_graining

    df = make_synthetic_graining(T=8000, dt=0.05, seed=0)
    q = len(df) // 4
    cold_phase_peak = df["graining"].iloc[:q].max()
    after_warmup = df["graining"].iloc[2 * q - 1]
    assert cold_phase_peak > 0.3, "cold + high slip must form graining"
    assert after_warmup < cold_phase_peak, "a warm surface must clean graining up"
    assert df["wear"].is_monotonic_increasing
    assert df["Ts"].iloc[q] > df["Ts"].iloc[0], "the tire must warm up"


def test_gradients_flow_through_the_condition_rollout():
    model = _model()
    F, z, _ = model.rollout_condition(*_drive(T=100), dt=0.05)
    (F.sum() + z[..., 3].sum()).backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), name
    assert model.form_net[0].weight.grad is not None
