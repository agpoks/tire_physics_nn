"""Dataset adapters mapping every source onto one canonical schema (PLAN.md §4)."""

from tire_nn.data.common import (
    CONTEXT_COLUMNS,
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    Normalizer,
    TireDataset,
    flip_sign_convention,
    make_synthetic,
    make_synthetic_transient,
    split_by_condition,
    split_by_group,
    validate_schema,
)

from tire_nn.data import adapters, registry
from tire_nn.data.adapters import ColumnSpec, DatasetNotAvailable
from tire_nn.data.vehicle import VehicleDataset, make_synthetic_vehicle
from tire_nn.data.graining import make_synthetic_graining

__all__ = [
    "adapters", "registry", "ColumnSpec", "DatasetNotAvailable",
    "VehicleDataset", "make_synthetic_vehicle", "make_synthetic_graining",
    "REQUIRED_COLUMNS", "OPTIONAL_COLUMNS", "CONTEXT_COLUMNS",
    "validate_schema", "flip_sign_convention", "make_synthetic", "make_synthetic_transient",
    "TireDataset", "Normalizer", "split_by_group", "split_by_condition",
]
