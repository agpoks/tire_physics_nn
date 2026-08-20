"""Evaluation: plots, extrapolation protocol and physical-consistency audit."""

from tire_nn.evaluation.physical_consistency import audit, audit_table
from tire_nn.evaluation.extrapolation import (
    evaluate_on,
    learning_curve_sizes,
    load_range_holdout,
    slip_range_holdout,
)

__all__ = [
    "audit", "audit_table",
    "slip_range_holdout", "load_range_holdout", "learning_curve_sizes", "evaluate_on",
]
