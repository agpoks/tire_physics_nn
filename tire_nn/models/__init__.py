"""Learned tire models — the ablation ladder (PLAN.md §2.3)."""

from tire_nn.models.base import BaseTireModel, ContextEncoder, CONTEXT_KEYS
from tire_nn.models.mlp_tire import MLPTireModel
from tire_nn.models.encoded_tire import EncodedTireNet, SymmetryTireNet
from tire_nn.models.parameter_tire import ParameterTireNet
from tire_nn.models.residual_tire import ResidualTireNet
from tire_nn.models.registry import MODEL_REGISTRY, build_model

__all__ = [
    "BaseTireModel", "ContextEncoder", "CONTEXT_KEYS",
    "MLPTireModel", "SymmetryTireNet", "EncodedTireNet",
    "ParameterTireNet", "ResidualTireNet",
    "MODEL_REGISTRY", "build_model",
]
