# 类型化求解参数及别名转换；记录请求参数不代表每个算法都实际支持。
"""Typed, solver-independent ZYO solve options."""

from dataclasses import dataclass
import math
from numbers import Real

from .errors import InvalidParameterError


_ALIASES = {
    "timelimit": "time_limit",
    "time_limit": "time_limit",
    "nodelimit": "node_limit",
    "node_limit": "node_limit",
    "iterationlimit": "iteration_limit",
    "iteration_limit": "iteration_limit",
    "mipgap": "mip_gap",
    "mip_gap": "mip_gap",
    "miprelgap": "mip_gap",
    "mip_rel_gap": "mip_gap",
    "feasibilitytol": "feasibility_tol",
    "feasibility_tol": "feasibility_tol",
    "integralitytol": "integrality_tol",
    "integrality_tol": "integrality_tol",
    "pivottol": "pivot_tol",
    "pivot_tol": "pivot_tol",
    "objectivetol": "objective_tol",
    "objective_tol": "objective_tol",
    "maxtableaucells": "max_tableau_cells",
    "max_tableau_cells": "max_tableau_cells",
    "threads": "threads",
    "seed": "seed",
    "presolve": "presolve",
}


@dataclass(frozen=True)
class SolveOptions:
    time_limit: float = 60.0
    node_limit: int = 10_000
    iteration_limit: int = 100_000
    mip_gap: float = 0.0
    feasibility_tol: float = 1e-7
    integrality_tol: float = 1e-7
    pivot_tol: float = 1e-10
    objective_tol: float = 1e-8
    max_tableau_cells: int = 2_000_000
    threads: int = 1
    seed: int = 0
    presolve: bool = True

    def __post_init__(self):
        for name in ("time_limit", "mip_gap", "objective_tol"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value < 0
            ):
                raise InvalidParameterError(f"{name} must be finite and nonnegative")

        for name in ("feasibility_tol", "integrality_tol", "pivot_tol"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise InvalidParameterError(f"{name} must be finite and positive")
        if self.integrality_tol >= 0.5:
            raise InvalidParameterError("integrality_tol must be smaller than 0.5")

        for name in ("node_limit", "iteration_limit", "max_tableau_cells", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise InvalidParameterError(f"{name} must be a nonnegative integer")
        if (
            isinstance(self.threads, bool)
            or not isinstance(self.threads, int)
            or self.threads < 1
        ):
            raise InvalidParameterError("threads must be a positive integer")
        if not isinstance(self.presolve, bool):
            raise InvalidParameterError("presolve must be a boolean")

    @classmethod
    def from_kwargs(cls, **kwargs):
        normalized = {}
        for name, value in kwargs.items():
            compact = name.replace("_", "").casefold()
            canonical = _ALIASES.get(compact) or _ALIASES.get(name.casefold())
            if canonical is None:
                raise InvalidParameterError(f"Unknown parameter: {name}")
            if canonical in normalized:
                raise InvalidParameterError(f"Duplicate parameter: {canonical}")
            normalized[canonical] = value
        return cls(**normalized)

    def to_legacy(self):
        from lzyopt.solver import Options

        return Options(
            time_limit=self.time_limit,
            node_limit=self.node_limit,
            iteration_limit=self.iteration_limit,
            mip_rel_gap=self.mip_gap,
            feasibility_tol=self.feasibility_tol,
            integrality_tol=self.integrality_tol,
            pivot_tol=self.pivot_tol,
            objective_tol=self.objective_tol,
            max_tableau_cells=self.max_tableau_cells,
            presolve=self.presolve,
            threads=self.threads,
            seed=self.seed,
        )
