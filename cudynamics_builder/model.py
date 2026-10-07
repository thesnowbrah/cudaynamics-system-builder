from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class EquationKind(str, Enum):
    STATE = "state"
    ALGEBRAIC = "algebraic"


@dataclass(slots=True)
class Equation:
    target: str
    expression: str
    kind: EquationKind
    source: str = ""


@dataclass(slots=True)
class AttributeDefaults:
    value: float = 0.0
    range_kind: str = "Fixed"
    minimum: float = 0.0
    maximum: float = 1.0
    step: float = 0.01
    count: int = 100
    mean: float = 0.0
    deviation: float = 0.0


METHODS = {
    "ExplicitEuler": "Явный Эйлер",
    "SemiExplicitEuler": "Полуявный Эйлер",
    "ImplicitEuler": "Неявный Эйлер (Ньютон)",
    "ExplicitMidpoint": "Явная средняя точка",
    "ImplicitMidpoint": "Неявная средняя точка (Ньютон)",
    "ExplicitRungeKutta4": "Рунге—Кутта 4",
    "ExplicitDormandPrince8": "Dormand—Prince 8 (фиксированный шаг)",
    "VariableSymmetryCD": "Variable Symmetry CD",
}


@dataclass(slots=True)
class SystemSpec:
    system_id: str = "new_system"
    display_name: str = "New system"
    equations: list[Equation] = field(default_factory=list)
    parameters: list[str] = field(default_factory=list)
    variable_defaults: dict[str, AttributeDefaults] = field(default_factory=dict)
    parameter_defaults: dict[str, AttributeDefaults] = field(default_factory=dict)
    methods: list[str] = field(default_factory=lambda: ["ExplicitEuler", "ExplicitRungeKutta4"])
    step: float = 0.01
    steps: int = 1000
    transient: int = 0
    execute_on_launch: bool = False
    symmetry_default: float = 0.0

    @property
    def states(self) -> list[str]:
        return [e.target for e in self.equations if e.kind == EquationKind.STATE]

    @property
    def algebraic(self) -> list[str]:
        return [e.target for e in self.equations if e.kind == EquationKind.ALGEBRAIC]

    @property
    def variables(self) -> list[str]:
        return self.states + [x for x in self.algebraic if x not in self.states]

    def ensure_defaults(self) -> None:
        for name in self.variables:
            self.variable_defaults.setdefault(name, AttributeDefaults())
        for name in self.parameters:
            self.parameter_defaults.setdefault(name, AttributeDefaults(value=1.0, minimum=1.0))
