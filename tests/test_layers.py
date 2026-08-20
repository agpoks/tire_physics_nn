"""Invariant tests for the encoded layers.

Every assertion here must hold for *arbitrary* weights (see ``conftest.randomize_``).
A guarantee that only survives after training is a penalty in disguise.
"""

import pytest
import torch

from tire_nn.layers import (
    DEFAULT_SPECS,
    BoundedParameter,
    BoundedParameterHead,
    FrictionEnvelope,
    OddSymmetricForceField,
    ParamSpec,
    SlipKinematics,
    ellipse_radius,
    project_into_ellipse,
    slip_angle,
    slip_ratio,
    slip_velocity,
    to_bounded,
    to_positive,
)
from conftest import randomize_

N = 257


def _grid():
    alpha = torch.linspace(-0.8, 0.8, N)
    kappa = torch.linspace(-0.9, 0.9, N)
    Fz = torch.linspace(100.0, 4000.0, N)
    return alpha, kappa, Fz


# --- P1: slip kinematics ---------------------------------------------------

def test_slip_kinematics_has_no_learnable_parameters():
    sk = SlipKinematics(R_e=0.3)
    assert list(sk.parameters()) == []


def test_slip_ratio_zero_at_free_rolling_and_signs():
    vx = torch.tensor([10.0, 10.0, 10.0])
    omega = torch.tensor([10.0 / 0.3, 12.0 / 0.3, 8.0 / 0.3])
    k = slip_ratio(omega, vx, 0.3)
    assert abs(k[0].item()) < 1e-6
    assert k[1] > 0 and k[2] < 0


def test_slip_is_finite_at_standstill():
    z = torch.zeros(5)
    assert torch.isfinite(slip_ratio(torch.rand(5), z, 0.3)).all()
    assert torch.isfinite(slip_angle(z, torch.rand(5), 0.1)).all()


def test_slip_velocity_matches_kinematic_definition():
    vsx, vsy = slip_velocity(torch.tensor([10.0]), torch.tensor([0.0]),
                             torch.tensor([30.0]), 0.0, 0.3)
    assert torch.allclose(vsx, torch.tensor([-1.0]), atol=1e-5)
    assert torch.allclose(vsy, torch.tensor([0.0]), atol=1e-5)


# --- P2: symmetry ----------------------------------------------------------

@pytest.mark.parametrize("scale_with_load", [True, False])
def test_zero_slip_gives_exactly_zero_force(scale_with_load):
    field = randomize_(OddSymmetricForceField(scale_with_load=scale_with_load))
    _, _, Fz = _grid()
    zero = torch.zeros(N)
    qx, qy = field(zero, zero, Fz)
    assert torch.equal(qx.abs(), torch.zeros(N))
    assert torch.equal(qy.abs(), torch.zeros(N))


def test_Fy_is_zero_whenever_alpha_is_zero_even_under_longitudinal_slip():
    field = randomize_(OddSymmetricForceField())
    _, kappa, Fz = _grid()
    _, qy = field(torch.zeros(N), kappa, Fz)
    assert torch.equal(qy.abs(), torch.zeros(N))


def test_Fx_is_zero_whenever_kappa_is_zero_even_under_lateral_slip():
    field = randomize_(OddSymmetricForceField())
    alpha, _, Fz = _grid()
    qx, _ = field(alpha, torch.zeros(N), Fz)
    assert torch.equal(qx.abs(), torch.zeros(N))


def test_odd_symmetry_is_exact_for_random_weights():
    field = randomize_(OddSymmetricForceField())
    alpha, kappa, Fz = _grid()
    qx, qy = field(alpha, kappa, Fz)
    qx_m, qy_m = field(-alpha, -kappa, Fz)
    assert torch.equal(qx, -qx_m)
    assert torch.equal(qy, -qy_m)


def test_lateral_symmetry_holds_at_fixed_kappa():
    field = randomize_(OddSymmetricForceField())
    alpha, kappa, Fz = _grid()
    _, qy_p = field(alpha, kappa, Fz)
    _, qy_m = field(-alpha, kappa, Fz)
    assert torch.equal(qy_p, -qy_m)


def test_longitudinal_symmetry_holds_at_fixed_alpha():
    field = randomize_(OddSymmetricForceField())
    alpha, kappa, Fz = _grid()
    qx_p, _ = field(alpha, kappa, Fz)
    qx_m, _ = field(alpha, -kappa, Fz)
    assert torch.equal(qx_p, -qx_m)


def test_force_opposes_slip_direction_dissipativity():
    field = randomize_(OddSymmetricForceField())
    alpha, kappa, Fz = _grid()
    qx, qy = field(alpha, kappa, Fz)
    assert torch.all(qx * kappa >= 0)
    assert torch.all(qy * alpha <= 0)


def test_asymmetry_head_is_separable_and_off_by_default():
    assert OddSymmetricForceField().offset is None
    field = randomize_(OddSymmetricForceField(asymmetry=True))
    alpha, kappa, Fz = _grid()
    qx, qy = field(alpha, kappa, Fz)
    qx_m, qy_m = field(-alpha, -kappa, Fz)
    # The offset is even, so the *average* of the two is exactly the offset term.
    assert torch.all(((qx + qx_m) / 2).abs() > 0)


# --- P3: friction envelope -------------------------------------------------

@pytest.mark.parametrize("mode", ["tanh", "algebraic"])
def test_friction_envelope_never_violated_for_extreme_inputs(mode):
    qx = torch.tensor([0.0, 1e3, -1e6, 1e9, -1e12, 5.0])
    qy = torch.tensor([0.0, -1e4, 1e6, -1e9, 1e12, -7.0])
    Fz = torch.full_like(qx, 800.0)
    mu = torch.full_like(qx, 0.9)
    Fx, Fy = project_into_ellipse(qx, qy, mu, mu, Fz, mode)
    assert torch.all(ellipse_radius(Fx, Fy, mu, mu, Fz) <= 1.0 + 1e-6)


def test_friction_envelope_holds_for_random_network_output():
    field = randomize_(OddSymmetricForceField(), std=8.0)
    env = FrictionEnvelope()
    alpha, kappa, Fz = _grid()
    qx, qy = field(alpha, kappa, Fz)
    mu_x = torch.full_like(Fz, 1.3)
    mu_y = torch.full_like(Fz, 1.1)
    Fx, Fy = env(qx, qy, mu_x, mu_y, Fz)
    assert torch.all(ellipse_radius(Fx, Fy, mu_x, mu_y, Fz) <= 1.0 + 1e-6)


def test_max_utilization_gives_strictly_interior_forces():
    qx = torch.full((16,), 1e9)
    qy = torch.full((16,), -1e9)
    Fz = torch.full((16,), 1000.0)
    mu = torch.ones(16)
    Fx, Fy = project_into_ellipse(qx, qy, mu, mu, Fz, "tanh", max_utilization=0.999)
    assert torch.all(ellipse_radius(Fx, Fy, mu, mu, Fz) < 1.0)


def test_projection_preserves_force_direction():
    qx = torch.tensor([300.0, -1e6])
    qy = torch.tensor([-400.0, 2e6])
    Fz = torch.full_like(qx, 1000.0)
    mu = torch.ones_like(qx)
    Fx, Fy = project_into_ellipse(qx, qy, mu, mu, Fz)
    assert torch.allclose(Fx * qy, Fy * qx, rtol=1e-4)   # cross product == 0


def test_projection_is_identity_like_in_the_linear_range():
    qx = torch.tensor([1.0, 2.0])
    qy = torch.tensor([0.5, -1.0])
    Fz = torch.full_like(qx, 1e6)      # huge limit -> rho ~ 0
    mu = torch.ones_like(qx)
    Fx, Fy = project_into_ellipse(qx, qy, mu, mu, Fz)
    assert torch.allclose(Fx, qx, rtol=1e-6) and torch.allclose(Fy, qy, rtol=1e-6)


def test_projection_is_differentiable_in_saturation():
    qx = torch.tensor([1e5], requires_grad=True)
    qy = torch.tensor([1e5], requires_grad=True)
    Fz = torch.tensor([1000.0])
    mu = torch.ones(1)
    Fx, Fy = project_into_ellipse(qx, qy, mu, mu, Fz)
    (Fx + Fy).sum().backward()
    assert torch.isfinite(qx.grad).all() and torch.isfinite(qy.grad).all()


# --- P4: bounded parameters ------------------------------------------------

def test_bounded_transforms_respect_their_ranges_for_extreme_inputs():
    # Closed-range guarantee (see to_positive/to_bounded docstrings: float32
    # saturation makes the endpoints attainable, so they are declared valid).
    z = torch.linspace(-1e3, 1e3, 1001)
    assert torch.all(to_positive(z, 0.01) >= 0.01)
    b = to_bounded(z, -2.0, 1.0)
    assert torch.all(b >= -2.0) and torch.all(b <= 1.0)
    # In the numerically non-saturated regime the bounds are strict.
    # (float32 sigmoid rounds to exactly 1.0 from |z| ~ 17, softplus to 0 from z ~ -25.)
    z = torch.linspace(-10.0, 10.0, 401)
    assert torch.all(to_positive(z, 0.01) > 0.01)
    b = to_bounded(z, -2.0, 1.0)
    assert torch.all(b > -2.0) and torch.all(b < 1.0)


def test_parameter_head_stays_in_bounds_under_adversarial_features():
    head = randomize_(BoundedParameterHead(6), std=20.0)
    feats = torch.randn(512, 6) * 50.0
    params = head(feats)
    by_name = {s.name: s for s in DEFAULT_SPECS}
    for name, value in params.items():
        spec = by_name[name]
        assert torch.all(value >= spec.lo), f"{name} below lower bound"
        if spec.hi is not None:
            assert torch.all(value <= spec.hi), f"{name} above upper bound"


def test_mu_stays_within_configured_bounds():
    head = randomize_(BoundedParameterHead(3), std=50.0)
    mu = head(torch.randn(1000, 3) * 100.0)["mu"]
    assert torch.all(mu >= 0.05) and torch.all(mu <= 2.5)


def test_relaxation_length_is_strictly_positive():
    spec = ParamSpec("sigma", lo=0.01, hi=None, init=0.3)
    p = BoundedParameter(spec, shape=(4,))
    with torch.no_grad():
        p.raw.copy_(torch.tensor([-1e4, -50.0, 0.0, 1e4]))
    # Strictly positive even when the raw parameter is driven to +-1e4: the declared
    # lower bound is itself a valid relaxation length, so tau = sigma/v stays finite.
    assert torch.all(p() >= 0.01) and torch.all(p() > 0.0)


def test_parameter_head_initialises_at_the_declared_value():
    head = BoundedParameterHead(4)
    params = head(torch.zeros(2, 4))
    assert abs(params["mu"][0].item() - 1.0) < 1e-4
    assert abs(params["B"][0].item() - 10.0) < 1e-3
    assert abs(params["sigma"][0].item() - 0.3) < 1e-4
