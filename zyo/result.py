# 统一不可变结果：保留实际状态、解、界、间隙和元数据，无解时禁止读取变量值。
"""Stable public result schema for every ZYO solver backend."""

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from types import MappingProxyType

from .status import Status


def _deep_freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value):
    if isinstance(value, Mapping):
        return {key: _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, frozenset, set)):
        return [_deep_thaw(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


@dataclass(frozen=True)
class Result:
    status: Status
    solver_name: str
    solver_version: str | None
    objective: float | None
    best_bound: float | None
    mip_gap: float | None
    values: object = field(repr=False)
    runtime: float = 0.0
    node_count: int = 0
    iteration_count: int = 0
    primal_residual: float | None = None
    bound_residual: float | None = None
    integrality_residual: float | None = None
    termination_reason: str = ""
    metadata: object = field(default_factory=lambda: MappingProxyType({}), repr=False)

    def __post_init__(self):
        object.__setattr__(self, "values", _deep_freeze(dict(self.values)))
        object.__setattr__(self, "metadata", _deep_freeze(dict(self.metadata)))

    @property
    def has_solution(self):
        return self.objective is not None

    def to_dict(self):
        return _deep_thaw({
            "status": self.status.value,
            "solver_name": self.solver_name,
            "solver_version": self.solver_version,
            "objective": self.objective,
            "best_bound": self.best_bound,
            "mip_gap": self.mip_gap,
            "values": self.values,
            "runtime": self.runtime,
            "node_count": self.node_count,
            "iteration_count": self.iteration_count,
            "primal_residual": self.primal_residual,
            "bound_residual": self.bound_residual,
            "integrality_residual": self.integrality_residual,
            "termination_reason": self.termination_reason,
            "metadata": self.metadata,
        })

    @classmethod
    def from_legacy(cls, raw):
        residuals = raw.residuals or {}
        status = Status(raw.status) if raw.status in Status._value2member_map_ else Status.UNKNOWN
        return cls(
            status=status,
            solver_name=raw.engine.casefold(),
            solver_version=None,
            objective=raw.objective,
            best_bound=raw.best_bound,
            mip_gap=raw.gap,
            values=dict(raw.values),
            runtime=raw.runtime_seconds,
            node_count=raw.nodes,
            iteration_count=raw.iterations,
            primal_residual=residuals.get("constraint_violation"),
            bound_residual=residuals.get("bound_violation"),
            integrality_residual=residuals.get("integrality_violation"),
            termination_reason=raw.message,
            metadata={"legacy_options": dict(raw.options)},
        )
