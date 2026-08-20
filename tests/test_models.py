"""Model-level invariant tests — every rung of the ablation ladder (PLAN.md §6)."""

import pytest
import torch

from tire_nn.data import TireDataset, make_synthetic, split_by_group
from tire_nn.evaluation import audit
from tire_nn.models import MODEL_REGISTRY, EncodedTireNet, MLPTireModel, ParameterTireNet, build_model
from tire_nn.physics import MagicFormulaTire
from tire_nn.training import TrainConfig, train_model
from conftest import randomize_

ENCODED = ["symmetry", "encoded", "parameter", "residual"]
ENVELOPE_MODELS = ["encoded", "parameter", "residual", "magic_formula"]


def _inputs(n=256):
    alpha = torch.linspace(-0.7, 0.7, n)
    kappa = torch.linspace(-0.8, 0.8, n)
    Fz = torch.linspace(150.0, 3500.0, n)
    return alpha, kappa, Fz


def test_registry_covers_every_documented_rung():
    for name in ("magic_formula", "mlp", "mlp_penalty", "symmetry", "encoded", "parameter", "residual"):
        assert name in MODEL_REGISTRY


@pytest.mark.parametrize("name", ENCODED + ["magic_formula"])
def test_encoded_models_give_zero_force_at_zero_slip(name):
    model = randomize_(build_model(name))
    Fz = torch.linspace(200.0, 3000.0, 64)
    z = torch.zeros_like(Fz)
    out = model(z, z, Fz)
    assert torch.allclose(out.Fx, z, atol=1e-6)
    assert torch.allclose(out.Fy, z, atol=1e-6)


@pytest.mark.parametrize("name", ENCODED + ["magic_formula"])
def test_encoded_models_are_odd_symmetric(name):
    model = randomize_(build_model(name))
    alpha, kappa, Fz = _inputs()
    out = model(alpha, kappa, Fz)
    out_m = model(-alpha, -kappa, Fz)
    assert torch.allclose(out.Fx, -out_m.Fx, atol=1e-5)
    assert torch.allclose(out.Fy, -out_m.Fy, atol=1e-5)


@pytest.mark.parametrize("name", ENCODED)
def test_Fy_vanishes_at_zero_alpha_and_Fx_at_zero_kappa(name):
    model = randomize_(build_model(name))
    alpha, kappa, Fz = _inputs()
    z = torch.zeros_like(alpha)
    assert torch.allclose(model(z, kappa, Fz).Fy, z, atol=1e-6)
    assert torch.allclose(model(alpha, z, Fz).Fx, z, atol=1e-6)


@pytest.mark.parametrize("name", ENVELOPE_MODELS)
def test_friction_envelope_is_never_violated(name):
    model = randomize_(build_model(name), std=6.0)
    report = audit(model, n=2048, alpha_max=1.2, kappa_max=1.2)
    assert report["envelope_violation"] < 1e-5, report
    assert report["envelope_violation_frac"] == 0.0, report


def test_symmetry_only_model_does_violate_the_envelope():
    """The point of the ablation: symmetry alone does not bound the force."""
    model = randomize_(build_model("symmetry"), std=6.0)
    assert audit(model, n=512)["envelope_violation"] > 0.1


def test_plain_mlp_violates_every_prior():
    """Control case — quantifies what the priors buy (PLAN.md §5)."""
    model = randomize_(MLPTireModel(), std=2.0)
    report = audit(model, n=512)
    assert report["zero_slip_force"] > 1e-3
    assert report["sym_violation_y"] > 1e-3


@pytest.mark.parametrize("name", ENCODED)
def test_mu_stays_in_bounds_for_adversarial_weights(name):
    model = randomize_(build_model(name), std=20.0)
    alpha, kappa, Fz = _inputs()
    params = model(alpha, kappa, Fz).params
    if "mu_x" in params:
        assert torch.all(params["mu_x"] >= 0.0) and torch.all(params["mu_x"] <= 3.0)
        assert torch.all(params["mu_y"] >= 0.0) and torch.all(params["mu_y"] <= 3.0)


def test_parameter_model_exposes_interpretable_quantities():
    model = ParameterTireNet()
    alpha, kappa, Fz = _inputs(32)
    params = model(alpha, kappa, Fz).params
    for key in ("mu_x", "mu_y", "B_x", "C_y", "E_y", "sigma_x", "sigma_y", "C_alpha", "C_kappa"):
        assert key in params
    assert torch.all(params["sigma_x"] > 0) and torch.all(params["sigma_y"] > 0)
    assert torch.all(params["C_alpha"] > 0)


def test_parameter_model_coefficients_depend_only_on_condition_not_on_slip():
    """A tire has one set of coefficients per operating point — not one per slip value."""
    model = ParameterTireNet()
    Fz = torch.full((16,), 1200.0)
    p_a = model.parameters_at(Fz)
    out_low = model(torch.full((16,), 0.01), torch.zeros(16), Fz).params
    out_high = model(torch.full((16,), 0.30), torch.zeros(16), Fz).params
    for key in ("B_y", "C_y", "mu_y0"):
        assert torch.allclose(out_low[key], out_high[key])
    assert torch.allclose(p_a["B_y"], out_low["B_y"])


def test_context_is_optional_and_unknown_keys_are_ignored():
    model = EncodedTireNet(context_keys=("vx",))
    alpha, kappa, Fz = _inputs(8)
    a = model(alpha, kappa, Fz)                       # no context at all
    b = model(alpha, kappa, Fz, {"nonsense": torch.ones(8)})
    assert torch.allclose(a.Fy, b.Fy)


def test_context_changes_the_prediction_when_supplied():
    model = randomize_(EncodedTireNet(context_keys=("p",)), std=1.0)
    alpha, kappa, Fz = _inputs(8)
    low = model(alpha, kappa, Fz, {"p": torch.full((8,), 1.5e5)})
    high = model(alpha, kappa, Fz, {"p": torch.full((8,), 2.5e5)})
    assert not torch.allclose(low.Fy, high.Fy)


def test_models_are_batch_shape_agnostic():
    model = EncodedTireNet()
    for shape in [(5,), (3, 4), (2, 3, 4)]:
        alpha = torch.rand(shape) * 0.2
        out = model(alpha, alpha * 0.5, torch.full(shape, 1000.0))
        assert out.Fy.shape == shape


def test_gradients_flow_to_every_active_parameter():
    model = EncodedTireNet(context_keys=("vx",))
    alpha, kappa, Fz = _inputs(64)
    out = model(alpha, kappa, Fz, {"vx": torch.full((64,), 20.0)})
    (out.Fx.sum() + out.Fy.sum()).backward()
    for name, p in model.named_parameters():
        if name == "context.fill":
            # Only used to stand in for *absent* context keys; correctly inactive here.
            assert p.grad is None
            continue
        assert p.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(p.grad).all(), f"{name} has non-finite gradient"


def test_missing_context_activates_the_learned_fill_value():
    """A declared-but-absent key must be handled explicitly, not zero-filled silently."""
    model = EncodedTireNet(context_keys=("Ts",))
    alpha, kappa, Fz = _inputs(32)
    out = model(alpha, kappa, Fz)          # 'Ts' not supplied
    (out.Fx.sum() + out.Fy.sum()).backward()
    fill = dict(model.named_parameters())["context.fill"]
    assert fill.grad is not None and torch.isfinite(fill.grad).all()


def test_training_reduces_loss_and_preserves_invariants():
    df = make_synthetic(600, seed=5)
    tr, va, _ = split_by_group(df)
    mk = lambda d: TireDataset(d, targets=("Fx", "Fy"))
    model = EncodedTireNet()
    hist = train_model(model, mk(tr), mk(va), TrainConfig(epochs=15, patience=15), verbose=False)
    assert hist["train_loss"][-1] < hist["train_loss"][0]
    report = audit(model, n=512)
    assert report["sym_violation_y"] == 0.0
    assert report["zero_slip_force"] == 0.0
    assert report["envelope_violation"] < 1e-5


def test_magic_formula_has_no_learnable_parameters():
    assert list(MagicFormulaTire().parameters()) == []
