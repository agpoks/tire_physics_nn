"""Contact-patch brush model tests: the PDE/graph form and its learned version."""

import pytest
import torch

from tire_nn.models.patch_brush_net import PatchBrushNet
from tire_nn.physics.brush import brush_combined
from tire_nn.physics.brush_patch import (
    parabolic_pressure,
    patch_coordinates,
    patch_forces,
    pressure_from_logits,
)
from conftest import randomize_

A = 0.06          # contact half-length [m]
KB = 6.0e6        # bristle line stiffness [N/m^2]
C_ALPHA = 2 * A**2 * KB      # implied cornering stiffness, 43.2 kN/rad


def _inputs(n=12, alpha_max=0.08, Fz=1000.0):
    alpha = torch.linspace(0.0, alpha_max, n)
    return (alpha, torch.zeros(n), torch.full((n,), Fz),
            torch.full((n,), A), torch.full((n,), KB), torch.ones(n))


# --- the discretised physics ------------------------------------------------

def test_parabolic_pressure_integrates_to_the_vertical_load():
    Fz = torch.tensor([500.0, 1000.0, 2500.0])
    a = torch.full((3,), A)
    xi, dxi = patch_coordinates(200, a)
    p = parabolic_pressure(xi, a, Fz)
    assert torch.allclose((p * dxi.unsqueeze(-1)).sum(-1), Fz, rtol=1e-3)


def test_softmax_pressure_integrates_to_the_load_for_any_logits():
    """The load balance is exact by construction, not approximately satisfied."""
    Fz = torch.tensor([300.0, 1200.0])
    a = torch.full((2,), A)
    for scale in (0.0, 5.0, 50.0):
        logits = torch.randn(2, 64) * scale
        p = pressure_from_logits(logits, a, Fz, 64)
        _, dxi = patch_coordinates(64, a)
        # Closed bound: softmax underflows to exactly 0 for extreme logits, which just
        # means that element carries no contact. The load balance stays exact.
        assert torch.all(p >= 0)
        assert torch.allclose((p * dxi.unsqueeze(-1)).sum(-1), Fz, rtol=1e-5)


def test_discretised_patch_reproduces_the_closed_form_brush_model():
    alpha, kappa, Fz, a, kb, mu = _inputs()
    out = patch_forces(kappa, torch.tan(alpha), Fz, a, kb, mu, n_elements=512)
    _, Fy_closed = brush_combined(alpha, kappa, Fz, C_ALPHA, C_ALPHA, 1.0)
    assert torch.allclose(out["Fy"], Fy_closed, atol=0.5)


def test_quadrature_converges_at_second_order():
    alpha, kappa, Fz, a, kb, mu = _inputs()
    _, Fy_closed = brush_combined(alpha, kappa, Fz, C_ALPHA, C_ALPHA, 1.0)
    errors = []
    for n_elements in (32, 128):
        out = patch_forces(kappa, torch.tan(alpha), Fz, a, kb, mu, n_elements=n_elements)
        errors.append(float((out["Fy"] - Fy_closed).abs().max()))
    # 4x elements should cut the error by roughly 16x for a second-order rule.
    assert errors[0] / errors[1] > 8.0


def test_patch_force_never_exceeds_the_friction_limit():
    alpha, kappa, Fz, a, kb, mu = _inputs(n=40, alpha_max=1.0)
    out = patch_forces(kappa, torch.tan(alpha), Fz, a, kb, mu, n_elements=128)
    magnitude = torch.sqrt(out["Fx"] ** 2 + out["Fy"] ** 2)
    assert float((magnitude / (mu * Fz)).max()) <= 1.0 + 1e-4


def test_sliding_fraction_grows_from_zero_to_one_with_slip():
    alpha, kappa, Fz, a, kb, mu = _inputs(n=20, alpha_max=0.12)
    out = patch_forces(kappa, torch.tan(alpha), Fz, a, kb, mu, n_elements=256)
    frac = out["sliding_fraction"]
    assert float(frac[0]) == 0.0
    assert float(frac[-1]) > 0.95
    assert torch.all(torch.diff(frac) >= -1e-6), "sliding must grow monotonically with slip"


def test_shear_opposes_the_slip_everywhere():
    n = 16
    alpha = torch.linspace(-0.1, 0.1, n)
    Fz = torch.full((n,), 1000.0)
    out = patch_forces(torch.zeros(n), torch.tan(alpha), Fz, torch.full((n,), A),
                       torch.full((n,), KB), torch.ones(n), n_elements=64)
    assert torch.all(out["Fy"] * alpha <= 1e-6)


# --- the learned model -------------------------------------------------------

def test_learned_patch_keeps_the_load_balance_for_adversarial_weights():
    model = randomize_(PatchBrushNet(n_elements=48), std=10.0)
    Fz = torch.tensor([400.0, 1000.0, 2200.0])
    p = model.pressure_profile(Fz)
    a = model.contact_half_length(Fz)
    _, dxi = patch_coordinates(48, a)
    assert torch.all(p >= 0)
    assert torch.allclose((p * dxi.unsqueeze(-1)).sum(-1), Fz, rtol=1e-4)


def test_learned_patch_is_odd_and_vanishes_at_zero_slip():
    model = randomize_(PatchBrushNet(n_elements=32), std=6.0)
    alpha = torch.linspace(-0.12, 0.12, 25)
    kappa = torch.linspace(-0.1, 0.1, 25)
    Fz = torch.full((25,), 1200.0)
    out = model(alpha, kappa, Fz)
    out_m = model(-alpha, -kappa, Fz)
    assert torch.allclose(out.Fx, -out_m.Fx, atol=1e-4)
    assert torch.allclose(out.Fy, -out_m.Fy, atol=1e-4)
    zero = torch.zeros(25)
    assert torch.allclose(model(zero, zero, Fz).Fy, zero, atol=1e-6)


def test_learned_patch_respects_the_friction_circle():
    model = randomize_(PatchBrushNet(n_elements=32), std=8.0)
    n = 30
    alpha = torch.linspace(-0.6, 0.6, n)
    Fz = torch.full((n,), 900.0)
    out = model(alpha, alpha * 0.5, Fz)
    mu = out.params["mu_x"]
    magnitude = torch.sqrt(out.Fx ** 2 + out.Fy ** 2)
    assert float((magnitude / (mu * Fz)).max()) <= 1.0 + 1e-3


def test_parabolic_mode_matches_the_closed_form():
    model = PatchBrushNet(n_elements=256, learn_pressure=False)
    alpha = torch.linspace(0.0, 0.08, 10)
    Fz = torch.full((10,), 1000.0)
    out = model(alpha, torch.zeros(10), Fz)
    _, Fy_closed = brush_combined(alpha, torch.zeros(10), Fz, C_ALPHA, C_ALPHA, 1.0)
    assert torch.allclose(out.Fy, Fy_closed, atol=1.0)


def test_patch_length_grows_with_load():
    model = PatchBrushNet()
    a = model.contact_half_length(torch.tensor([300.0, 1000.0, 3000.0]))
    assert float(a[0]) < float(a[1]) < float(a[2])


def test_symmetric_pressure_option_is_actually_symmetric():
    model = randomize_(PatchBrushNet(n_elements=32, symmetric_pressure=True), std=5.0)
    p = model.pressure_profile(torch.tensor([1000.0]))[0]
    assert torch.allclose(p, torch.flip(p, dims=[0]), rtol=1e-4)


def test_learned_pressure_recovers_a_non_parabolic_profile():
    """End-to-end: a profile the parabolic assumption cannot represent."""
    torch.manual_seed(0)
    n_el, n = 48, 200
    alpha = torch.linspace(-0.09, 0.09, n)
    Fz = torch.full((n,), 1000.0)
    a = torch.full((n,), A)
    xi, _ = patch_coordinates(n_el, a)
    u = xi / (2 * a.unsqueeze(-1))
    shape = torch.clamp(u, min=1e-6) ** 2.2 * torch.clamp(1 - u, min=1e-6) ** 0.9
    dxi = (2 * a / n_el).unsqueeze(-1)
    p_true = shape / (shape * dxi).sum(-1, keepdim=True) * Fz.unsqueeze(-1)
    target = patch_forces(torch.zeros(n), torch.tan(alpha), Fz, a,
                          torch.full((n,), KB), torch.ones(n),
                          pressure=p_true, n_elements=n_el)["Fy"]

    model = PatchBrushNet(n_elements=n_el, learn_pressure=True)
    opt = torch.optim.Adam(model.parameters(), lr=0.02)
    for _ in range(500):
        opt.zero_grad(set_to_none=True)
        loss = ((model(alpha, torch.zeros(n), Fz).Fy - target) ** 2).mean() / 1e6
        loss.backward()
        opt.step()
    with torch.no_grad():
        rmse = float(((model(alpha, torch.zeros(n), Fz).Fy - target) ** 2).mean().sqrt())
    assert rmse < 20.0, f"learned patch did not fit the non-parabolic tire (RMSE {rmse:.1f} N)"
