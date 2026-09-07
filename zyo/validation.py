# 独立候选检查器：从原模型重算目标、行残差、边界及整数违反量。
"""Solver-independent candidate validation in original model units."""

from dataclasses import dataclass
import math

from .errors import NumericalError


@dataclass(frozen=True)
class Residuals:
    constraint: float
    bound: float
    integrality: float
    feasibility_tol: float
    integrality_tol: float

    @property
    def is_feasible(self):
        return (
            max(self.constraint, self.bound) <= self.feasibility_tol
            and self.integrality <= self.integrality_tol
        )


def _ordered_values(model, values):
    expected = {variable.name for variable in model.variables}
    actual = set(values)
    if actual != expected:
        raise NumericalError(
            f"Candidate keys differ: expected={sorted(expected)} actual={sorted(actual)}"
        )
    ordered = [float(values[variable.name]) for variable in model.variables]
    if not all(math.isfinite(value) for value in ordered):
        raise NumericalError("Candidate contains a non-finite value")
    return ordered


def evaluate_objective(model, values):
    x = _ordered_values(model, values)
    return float(
        model.objective.constant
        + sum(coefficient * x[index] for index, coefficient in model.objective.terms.items())
    )


def validate_candidate(model, values, feasibility_tol, integrality_tol):
    x = _ordered_values(model, values)
    row_violation = 0.0
    bound_violation = 0.0
    integer_violation = 0.0

    for row in model.constraints:
        activity = row.expression.constant + sum(
            coefficient * x[index]
            for index, coefficient in row.expression.terms.items()
        )
        if row.sense == "==":
            violation = abs(activity)
        elif row.sense == "<=":
            violation = max(0.0, activity)
        else:
            violation = max(0.0, -activity)
        row_violation = max(row_violation, violation)

    for index, variable in enumerate(model.variables):
        bound_violation = max(
            bound_violation,
            variable.lb - x[index],
            x[index] - variable.ub,
        )
        if variable.kind != "C":
            integer_violation = max(
                integer_violation,
                abs(x[index] - round(x[index])),
            )

    return Residuals(
        constraint=float(row_violation),
        bound=float(bound_violation),
        integrality=float(integer_violation),
        feasibility_tol=float(feasibility_tol),
        integrality_tol=float(integrality_tol),
    )
