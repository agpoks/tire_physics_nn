"""Constrained building blocks that encode the physical priors (PLAN.md §3)."""

from tire_nn.layers.slip_kinematics import SlipKinematics, slip_angle, slip_ratio, slip_velocity
from tire_nn.layers.symmetry import OddSymmetricForceField, mlp
from tire_nn.layers.friction_envelope import FrictionEnvelope, ellipse_radius, project_into_ellipse
from tire_nn.layers.bounded_parameters import (
    DEFAULT_SPECS,
    BoundedParameter,
    BoundedParameterHead,
    ParamSpec,
    to_bounded,
    to_positive,
)

__all__ = [
    "SlipKinematics", "slip_ratio", "slip_angle", "slip_velocity",
    "OddSymmetricForceField", "mlp",
    "FrictionEnvelope", "project_into_ellipse", "ellipse_radius",
    "ParamSpec", "DEFAULT_SPECS", "BoundedParameter", "BoundedParameterHead",
    "to_positive", "to_bounded",
]
