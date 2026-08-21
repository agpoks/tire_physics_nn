# Model catalogue

Every model implements one interface, so they are drop-in interchangeable:

```python
out = model(alpha, kappa, Fz, context=None)   # -> TireForces
Fx, Fy = out                                  # unpacks like a tuple
```

```{figure} ../_static/diagrams/model_map.png
:alt: what each model contains
:width: 100%
```

## The catalogue

| model | category | physics contained | params* | page |
|---|---|---|---|---|
| `MagicFormulaTire` | analytical | MF, load sensitivity, similarity combined slip | 0 | [black-box & analytical](black-box) |
| `MLPTireModel` | black box | none | 4 546 | [black-box & analytical](black-box) |
| `MLPTireModel` + penalty | physics-informed | envelope, softly | 4 546 | [black-box & analytical](black-box) |
| `GRUTireModel` | black box, sequential | none | 3 907 | [black-box & analytical](black-box) |
| `NeuralODETireModel` | black box, continuous-time | none | 1 411 | [black-box & analytical](black-box) |
| `SymmetryTireNet` | physics-encoded | slip kinematics, odd symmetry, dissipativity | 1 250 | [encoded](encoded) |
| `EncodedTireNet` | physics-encoded | + hard friction envelope | 1 254 | [encoded](encoded) |
| `ParameterTireNet` | physics-encoded | + Magic Formula, bounded parameters | 1 483 | [parameter](parameter) |
| `ResidualTireNet` | grey box | analytical prior + bounded symmetric residual | 1 254 | [residual](residual) |
| `RelaxationTireCell` | physics-encoded, dynamic | + relaxation ODE | +2 | [dynamic](dynamic) |
| `ThermoGrainingTire` | physics-encoded, dynamic | + thermal, wear, graining states | +1 k | [dynamic](dynamic) |
| `FourWheelVehicle` | physics-encoded, vehicle | + exact Newton–Euler, one shared tire | shared | [vehicle](vehicle.md) |
| `LapDegradationUDE` | physics-encoded, UDE | additive observation, monotone wear, bounded graining | 1 307 | [degradation](../physics/degradation) |
| `LinearDegradationModel` | analytical baseline | lap time linear in tyre age | 47 | [degradation](../physics/degradation) |
| `BlackBoxDegradationModel` | black box | none | 1 427 | [degradation](../physics/degradation) |

\* at default width, for the configuration used in Experiment 1.

## Composition

The dynamic and vehicle models are *wrappers*: they take a constitutive model and add a
level of the [stack](../physics/index). Each wrapper preserves the guarantees of what it
wraps.

```python
from tire_nn.models import ParameterTireNet
from tire_nn.models.relaxation_tire import RelaxationTireCell
from tire_nn.models.thermo_graining_tire import ThermoGrainingTire
from tire_nn.models.four_wheel_vehicle import FourWheelVehicle
from tire_nn.data.vehicle import DEFAULT_VEHICLE

tire = ParameterTireNet(context_keys=("vx", "p"))
tire = ThermoGrainingTire(tire)                                   # condition states
cell = RelaxationTireCell(tire.steady, sigma_from_steady=True)    # transient
vehicle = FourWheelVehicle(tire, DEFAULT_VEHICLE)                 # rigid body
```

## Building one from a config

Every model is reachable by name, so an experiment is fully described by its YAML:

```python
from tire_nn.models import build_model, MODEL_REGISTRY

sorted(MODEL_REGISTRY)
# ['encoded', 'magic_formula', 'mlp', 'mlp_penalty', 'parameter', 'residual', 'symmetry']

model = build_model("encoded", context_keys=("p",), hidden=(32, 32))
```

## What every model returns

```python
@dataclass
class TireForces:
    Fx: Tensor
    Fy: Tensor
    Mz: Tensor | None = None
    params: dict[str, Tensor] = field(default_factory=dict)
```

`params` carries the physically meaningful quantities the model chooses to expose. It is
fully populated by `ParameterTireNet` ($\mu$, $B$, $C$, $E$, $C_\alpha$, $\sigma$),
partial for the encoded models ($\mu_x$, $\mu_y$, envelope utilisation $\rho$) and empty
for the plain MLP — which is itself the interpretability gradient the catalogue is
meant to expose.
