"""Name -> model factory, so experiments are fully described by their YAML config."""

from __future__ import annotations

from tire_nn.models.encoded_tire import EncodedTireNet, SymmetryTireNet
from tire_nn.models.mlp_tire import MLPTireModel
from tire_nn.models.parameter_tire import ParameterTireNet
from tire_nn.models.residual_tire import ResidualTireNet
from tire_nn.physics.pacejka import MagicFormulaTire

__all__ = ["MODEL_REGISTRY", "build_model"]

MODEL_REGISTRY = {
    "magic_formula": MagicFormulaTire,
    "mlp": MLPTireModel,
    "mlp_penalty": MLPTireModel,        # identical architecture; the trainer adds the penalty
    "symmetry": SymmetryTireNet,
    "encoded": EncodedTireNet,
    "parameter": ParameterTireNet,
    "residual": ResidualTireNet,
}


def build_model(name: str, **kwargs):
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {name!r}; available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)
