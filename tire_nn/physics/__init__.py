"""Pure, differentiable physics. No learnable parameters live in this subpackage."""

from tire_nn.physics.pacejka import (
    MFParams,
    MagicFormulaTire,
    cornering_stiffness,
    load_sensitive_mu,
    magic_formula,
    pacejka_combined,
    pacejka_lateral,
    pacejka_longitudinal,
)
from tire_nn.physics.brush import brush_combined, brush_forces
from tire_nn.physics.simple_models import dugoff_tire, linear_tire
from tire_nn.physics.vehicle_dynamics import (
    VehicleParams,
    corner_positions,
    corner_velocities,
    newton_euler,
    quasi_static_loads,
    static_loads,
    wheel_to_body,
)
from tire_nn.physics.thermal import ThermalParams, slip_power, thermal_rates
from tire_nn.physics.wear import effective_friction, graining_rate, wear_rate

__all__ = [
    "MFParams", "MagicFormulaTire", "magic_formula", "pacejka_lateral",
    "pacejka_longitudinal", "pacejka_combined", "load_sensitive_mu", "cornering_stiffness",
    "brush_combined", "brush_forces", "linear_tire", "dugoff_tire",
    "VehicleParams", "corner_positions", "corner_velocities", "static_loads",
    "quasi_static_loads", "wheel_to_body", "newton_euler",
    "ThermalParams", "slip_power", "thermal_rates",
    "wear_rate", "graining_rate", "effective_friction",
]
