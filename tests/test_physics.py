"""Physics module tests: the analytical baselines must themselves be correct."""

import torch

from tire_nn.physics import (
    MFParams,
    MagicFormulaTire,
    VehicleParams,
    brush_combined,
    cornering_stiffness,
    effective_friction,
    graining_rate,
    load_sensitive_mu,
    newton_euler,
    pacejka_lateral,
    pacejka_longitudinal,
    quasi_static_loads,
    slip_power,
    static_loads,
    thermal_rates,
    ThermalParams,
    wear_rate,
    wheel_to_body,
)


def test_magic_formula_zero_slip_zero_force():
    Fz = torch.full((5,), 1000.0)
    zero = torch.zeros(5)
    Fy, _ = pacejka_lateral(zero, Fz, MFParams())
    Fx, _ = pacejka_longitudinal(zero, Fz, MFParams())
    assert torch.allclose(Fy, torch.zeros(5), atol=1e-6)
    assert torch.allclose(Fx, torch.zeros(5), atol=1e-6)


def test_magic_formula_sign_convention_is_sae():
    Fz = torch.full((3,), 1000.0)
    Fy, _ = pacejka_lateral(torch.tensor([0.05, 0.1, 0.2]), Fz, MFParams())
    Fx, _ = pacejka_longitudinal(torch.tensor([0.05, 0.1, 0.2]), Fz, MFParams())
    assert torch.all(Fy < 0), "positive slip angle must give negative Fy (SAE)"
    assert torch.all(Fx > 0), "positive slip ratio must give positive Fx (driving)"


def test_magic_formula_peak_is_bounded_by_mu_Fz():
    Fz = torch.full((201,), 1500.0)
    alpha = torch.linspace(-1.0, 1.0, 201)
    p = MFParams(mu=1.2)
    Fy, mu = pacejka_lateral(alpha, Fz, p)
    assert Fy.abs().max() <= (mu * Fz).max() * 1.0001


def test_combined_slip_stays_inside_friction_circle():
    n = 60
    a = torch.linspace(-0.5, 0.5, n).repeat_interleave(n)
    k = torch.linspace(-0.5, 0.5, n).repeat(n)
    Fz = torch.full_like(a, 1000.0)
    m = MagicFormulaTire(MFParams(mu=1.0), MFParams(mu=1.0))
    Fx, Fy = m(a, k, Fz)
    assert torch.max(torch.sqrt(Fx**2 + Fy**2) / Fz) <= 1.0001


def test_load_sensitivity_decreases_mu():
    Fz = torch.tensor([500.0, 1000.0, 2000.0])
    mu = load_sensitive_mu(Fz, 1.0, k_mu=0.1, Fz0=1000.0)
    assert mu[0] > mu[1] > mu[2]


def test_cornering_stiffness_matches_numerical_derivative():
    Fz = torch.tensor([1000.0])
    p = MFParams(B=9.0, C=1.7, E=0.4, mu=1.1, k_mu=0.0)
    h = 1e-4
    Fy_p, _ = pacejka_lateral(torch.tensor([h]), Fz, p)
    Fy_m, _ = pacejka_lateral(torch.tensor([-h]), Fz, p)
    numeric = -(Fy_p - Fy_m) / (2 * h)          # dFy/dalpha, sign-flipped to positive
    assert torch.allclose(numeric, cornering_stiffness(p, Fz), rtol=1e-3)


def test_brush_saturates_at_mu_Fz():
    Fz = torch.full((50,), 1000.0)
    a = torch.linspace(0.0, 0.6, 50)
    Fx, Fy = brush_combined(a, torch.zeros(50), Fz, 60000.0, 50000.0, 0.9)
    mag = torch.sqrt(Fx**2 + Fy**2)
    assert mag.max() <= 0.9 * 1000.0 * 1.0001
    assert torch.all(Fy <= 1e-6), "positive alpha -> negative Fy"


def test_loads_sum_to_weight_under_load_transfer():
    vp = VehicleParams(m=1200, Iz=1500, lf=1.3, lr=1.4, t_f=1.6, t_r=1.6, h_cg=0.45)
    assert abs(static_loads(vp).sum().item() - vp.m * 9.81) < 1e-3
    ax = torch.tensor([-6.0, 0.0, 4.0])
    ay = torch.tensor([8.0, -3.0, 0.0])
    Fz = quasi_static_loads(ax, ay, vp)
    assert torch.allclose(Fz.sum(-1), torch.full((3,), vp.m * 9.81), rtol=1e-4)
    assert torch.all(Fz > 0)


def test_newton_euler_reference_case():
    """Hand-computed case: 100 N lateral on both front wheels only."""
    vp = VehicleParams(m=1000, Iz=800, lf=1.2, lr=1.4, t_f=1.5, t_r=1.5)
    Fx_b = torch.zeros(1, 4)
    Fy_b = torch.tensor([[100.0, 100.0, 0.0, 0.0]])
    z = torch.zeros(1)
    ax, ay, r_dot = newton_euler(Fx_b, Fy_b, torch.tensor([10.0]), z, z, vp)
    assert torch.allclose(ax, torch.zeros(1))
    assert torch.allclose(ay, torch.tensor([200.0 / 1000.0]))
    assert torch.allclose(r_dot, torch.tensor([1.2 * 200.0 / 800.0]))


def test_newton_euler_longitudinal_asymmetry_yaws_the_car():
    """Left-side traction only must yaw the car to the left (positive r_dot ... check sign)."""
    vp = VehicleParams(m=1000, Iz=800, lf=1.2, lr=1.4, t_f=1.5, t_r=1.5)
    Fx_b = torch.tensor([[200.0, 0.0, 200.0, 0.0]])   # FL, RL only
    Fy_b = torch.zeros(1, 4)
    z = torch.zeros(1)
    _, _, r_dot = newton_euler(Fx_b, Fy_b, torch.tensor([10.0]), z, z, vp)
    # Mz = -sum(y_i Fx_i) = -(0.75*200 + 0.75*200) < 0 -> turns right
    assert r_dot.item() < 0


def test_wheel_to_body_rotation_is_orthogonal():
    Fx = torch.randn(4, 4)
    Fy = torch.randn(4, 4)
    delta = torch.rand(4, 4)
    Fx_b, Fy_b = wheel_to_body(Fx, Fy, delta)
    assert torch.allclose(Fx_b**2 + Fy_b**2, Fx**2 + Fy**2, atol=1e-5)


def test_slip_power_is_non_negative_for_dissipative_tire():
    P = slip_power(torch.tensor([-500.0, 300.0]), torch.tensor([0.0, -200.0]),
                   torch.tensor([2.0, -1.0]), torch.tensor([0.0, 3.0]))
    assert torch.all(P >= 0)


def test_thermal_couples_surface_and_core_conservatively():
    p = ThermalParams()
    Ts, Tc = torch.tensor([350.0]), torch.tensor([320.0])
    env = torch.tensor([300.0])
    dTs, dTc = thermal_rates(Ts, Tc, torch.zeros(1), env, env, p)
    assert dTs < 0 and dTc > 0, "hot surface must cool into a cooler core"
    # With no environment losses the exchanged power cancels exactly.
    p0 = ThermalParams(h_sa=0.0, h_ca=0.0)
    dTs0, dTc0 = thermal_rates(Ts, Tc, torch.zeros(1), env, env, p0)
    assert torch.allclose(dTs0 * p0.Cs, -dTc0 * p0.Cc, atol=1e-4)


def test_wear_rate_never_negative_and_graining_rates_bound_the_state():
    assert torch.all(wear_rate(torch.linspace(-50, 50, 100)) >= 0)
    g0 = torch.zeros(10)
    g1 = torch.ones(10)
    R_form = torch.rand(10) * 5
    R_clean = torch.rand(10) * 5
    assert torch.all(graining_rate(g0, R_form, R_clean) >= 0), "g=0 boundary is not attracting downward"
    assert torch.all(graining_rate(g1, R_form, R_clean) <= 0), "g=1 boundary is not attracting upward"


def test_effective_friction_monotone_in_condition():
    mu = torch.tensor([1.2])
    clean = effective_friction(mu, torch.zeros(1), torch.zeros(1))
    worn = effective_friction(mu, torch.ones(1), torch.zeros(1))
    grained = effective_friction(mu, torch.zeros(1), torch.ones(1) * 0.5)
    assert clean > worn > 0 and clean > grained > 0
