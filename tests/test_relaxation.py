"""Transient-model tests (PLAN.md §6, items 7 and the integrator agreement check)."""

import pytest
import torch

from tire_nn.models import EncodedTireNet, ParameterTireNet
from tire_nn.models.baselines_seq import GRUTireModel, NeuralODETireModel
from tire_nn.models.relaxation_tire import RelaxationTireCell
from tire_nn.layers.friction_envelope import ellipse_radius
from conftest import randomize_


def _cell(**kw):
    return RelaxationTireCell(EncodedTireNet(context_keys=("vx",)), **kw)


def _seq(T=200, vx=20.0, step_at=50, alpha=0.1):
    a = torch.zeros(1, T)
    a[:, step_at:] = alpha
    return a, torch.zeros(1, T), torch.full((1, T), 1000.0), torch.full((1, T), vx)


def test_relaxation_lengths_are_positive_for_adversarial_weights():
    cell = randomize_(_cell(), std=50.0)
    assert float(cell.sigma_x()) > 0
    assert float(cell.sigma_y()) > 0


def test_time_constants_are_positive_and_finite_including_standstill():
    cell = _cell()
    for vx in (0.0, 1e-9, 1.0, 100.0):
        tx, ty = cell.time_constants(torch.tensor([vx]), {})
        assert float(tx) > 0 and float(ty) > 0
        assert torch.isfinite(tx).all() and torch.isfinite(ty).all()


def test_time_constant_scales_inversely_with_speed():
    """tau = sigma/v is the encoded structure — it must hold numerically."""
    cell = _cell()
    tau_slow = float(cell.time_constants(torch.tensor([5.0]), {})[1])
    tau_fast = float(cell.time_constants(torch.tensor([50.0]), {})[1])
    assert tau_slow > tau_fast
    sigma = float(cell.sigma_y())
    assert abs(tau_fast - sigma / (50.0 + cell.v_eps)) < 1e-6


def test_rise_distance_is_speed_invariant():
    """The physical signature of distance-parameterised relaxation."""
    cell = _cell()
    distances = []
    for vx in (10.0, 30.0):
        a, k, Fz, v = _seq(T=600, vx=vx)
        F = cell.rollout(a, k, Fz, v, 0.001, {"vx": v}, method="exact")
        Fy = F[0, :, 1]
        start, final = Fy[49], Fy[-1]
        target = start + 0.632 * (final - start)
        idx = int(torch.nonzero((Fy[50:] - start).abs() >= (target - start).abs())[0])
        distances.append(idx * 0.001 * vx)
    assert abs(distances[0] - distances[1]) / distances[0] < 0.15


@pytest.mark.parametrize("method", ["euler", "rk4", "exact"])
def test_integrators_agree_on_a_smooth_trajectory(method):
    cell = _cell()
    a, k, Fz, v = _seq(T=400)
    ref = cell.rollout(a, k, Fz, v, 1e-4, {"vx": v}, method="exact")
    got = cell.rollout(a, k, Fz, v, 1e-4, {"vx": v}, method=method)
    assert torch.allclose(ref, got, atol=1e-2 * float(ref.abs().max()))


def test_exact_and_rk4_agree_at_a_realistic_step_size():
    cell = _cell()
    a, k, Fz, v = _seq(T=300)
    exact = cell.rollout(a, k, Fz, v, 2e-3, {"vx": v}, method="exact")
    rk4 = cell.rollout(a, k, Fz, v, 2e-3, {"vx": v}, method="rk4")
    assert torch.allclose(exact, rk4, rtol=1e-3, atol=1.0)


def test_torchdiffeq_path_matches_or_is_cleanly_unavailable():
    cell = _cell()
    a, k, Fz, v = _seq(T=120)
    try:
        got = cell.rollout(a, k, Fz, v, 2e-3, {"vx": v}, method="odeint")
    except ImportError as exc:
        pytest.skip(f"torchdiffeq not installed: {exc}")
    ref = cell.rollout(a, k, Fz, v, 2e-3, {"vx": v}, method="exact")
    assert torch.allclose(ref, got, rtol=5e-2, atol=5.0)


def test_relaxed_force_never_leaves_the_friction_ellipse():
    """The relaxed force contracts toward F_ss, which the envelope already bounds."""
    cell = randomize_(_cell(), std=4.0)
    a, k, Fz, v = _seq(T=300, alpha=0.5)
    F = cell.rollout(a, k, Fz, v, 2e-3, {"vx": v}, method="exact")
    ss = cell.steady(a, k, Fz, {"vx": v})
    mu_x, mu_y = ss.params["mu_x"], ss.params["mu_y"]
    rho = ellipse_radius(F[..., 0], F[..., 1], mu_x, mu_y, Fz)
    assert float(rho.max()) <= 1.0 + 1e-5


def test_steady_state_is_reached_when_inputs_are_constant():
    cell = _cell()
    T = 2000
    a = torch.full((1, T), 0.08)
    k = torch.zeros(1, T)
    Fz = torch.full((1, T), 1000.0)
    v = torch.full((1, T), 20.0)
    F = cell.rollout(a, k, Fz, v, 2e-3, {"vx": v}, method="exact")
    ss = cell.steady(a[:, -1], k[:, -1], Fz[:, -1], {"vx": v[:, -1]})
    assert torch.allclose(F[0, -1, 1], ss.Fy[0], rtol=1e-3, atol=1e-2)


def test_step_size_check_rejects_unstable_euler_and_accepts_rk4_regime():
    cell = _cell()
    cell.check_step_size(dt=1e-3, vx_max=30.0)          # fine
    with pytest.raises(ValueError, match="not stable"):
        cell.check_step_size(dt=0.5, vx_max=30.0)


def test_sigma_from_steady_uses_the_parameter_networks_relaxation_length():
    cell = RelaxationTireCell(ParameterTireNet(), sigma_from_steady=True)
    assert not hasattr(cell, "sigma_x")
    Fz = torch.full((4,), 1000.0)
    params = cell.steady.parameters_at(Fz)
    tx, ty = cell.time_constants(torch.full((4,), 20.0), params)
    assert torch.all(tx > 0) and torch.all(ty > 0)


def test_sigma_from_steady_fails_loudly_if_the_steady_model_has_none():
    cell = RelaxationTireCell(EncodedTireNet(), sigma_from_steady=True)
    with pytest.raises(ValueError, match="exposes no sigma"):
        cell.time_constants(torch.ones(2), {})


def test_static_model_rollout_is_the_quasi_static_baseline():
    model = EncodedTireNet()
    a, k, Fz, v = _seq(T=100)
    F = model.rollout(a, k, Fz, v, 2e-3)
    ss = model(a, k, Fz)
    assert torch.allclose(F[..., 1], ss.Fy)


@pytest.mark.parametrize("cls", [GRUTireModel, NeuralODETireModel])
def test_sequence_baselines_share_the_rollout_interface(cls):
    model = cls(context_keys=("vx",))
    a, k, Fz, v = _seq(T=60)
    F = model.rollout(a, k, Fz, v, 2e-3, {"vx": v})
    assert F.shape == (1, 60, 2)
    assert torch.isfinite(F).all()


def test_relaxation_recovers_the_true_relaxation_length_on_synthetic_data():
    """End-to-end: the encoded parameter is identifiable from step-test data."""
    from tire_nn.data import TireDataset, make_synthetic_transient, split_by_group
    from tire_nn.training import TrainConfig, train_model

    df = make_synthetic_transient(n_sequences=10, T=200, sigma_x=0.15, sigma_y=0.30, seed=3)
    tr, va, _ = split_by_group(df)
    mk = lambda d: TireDataset(d, targets=("Fx", "Fy"), context_keys=("vx",), window=60)
    cell = _cell()
    train_model(cell, mk(tr), mk(va),
                TrainConfig(epochs=12, mode="sequence", dt=0.002, targets=("Fx", "Fy"),
                            batch_size=64, lr=0.01),
                verbose=False)
    assert abs(float(cell.sigma_y()) - 0.30) < 0.15
    assert float(cell.sigma_x()) > 0.0
