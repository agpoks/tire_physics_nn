"""Dataset adapters mapping every source onto one canonical schema (PLAN.md §4)."""

from tire_nn.data.common import (
    CONTEXT_COLUMNS,
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    Normalizer,
    TireDataset,
    flip_sign_convention,
    make_synthetic,
    split_by_condition,
    split_by_group,
    validate_schema,
)

__all__ = [
    "REQUIRED_COLUMNS", "OPTIONAL_COLUMNS", "CONTEXT_COLUMNS",
    "validate_schema", "flip_sign_convention", "make_synthetic",
    "TireDataset", "Normalizer", "split_by_group", "split_by_condition",
]
