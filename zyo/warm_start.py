# ZYO-051 热启动与基复用：同一结构模型在界/RHS 变化后复用已有基，并如实校验有效性。
# 只做基层面的复用判定与线性代数，不调用任何优化引擎，也不声称复用一定带来加速。
"""Warm start by basis reuse for the ZYO sparse basis.

A warm start reuses a previously optimal basis after the problem data changes. Reuse is
only meaningful when the basis stays *valid* (non-singular) and *primal feasible* for the
new bounds; this module decides that and reports the decision instead of assuming it:

* ``ReuseDecision`` records whether the same basis is still valid, whether it is still
  primal feasible for the new bounds, the objective at that basis, and the reason;
* ``WarmStart`` packages the reusable basis together with the change it survived.

The caller keeps ownership of algorithm policy: this module never suppresses a cold start
when reuse fails, and it never claims an optimum. It exists so that "reuse the basis" is a
checked decision with evidence, not an assumption.
"""
from dataclasses import dataclass, field
import math

import numpy as np
from scipy.sparse import csc_matrix

from .errors import NumericalError
from .sparse_simplex import SparseBasis


@dataclass
class ReuseDecision:
    """Outcome of asking whether a stored basis can be reused for new problem data."""

    reusable: bool = False
    basis_valid: bool = False
    primal_feasible: bool = False
    objective: float | None = None
    basic_values: list = field(default_factory=list)
    max_primal_violation: float = math.inf
    max_equation_residual: float = math.inf
    reason: str = ''
    checks: dict = field(default_factory=dict)


@dataclass
class WarmStart:
    """A basis that survived a documented change, kept for the next solve."""

    basic: list
    lower: list
    upper: list
    costs: list
    objective: float
    iterations: int = 0
    provenance: str = ''


def evaluate_reuse(matrix, basic, lower, upper, costs, *, rhs=None,
                   feasibility_tolerance=1e-7, residual_tolerance=1e-9):
    """Decide whether ``basic`` is a valid, primal-feasible basis for the NEW data.

    Returns a :class:`ReuseDecision`. ``reusable`` is true only when the basis is
    non-singular, satisfies ``A x = rhs`` to ``residual_tolerance``, and honours every
    finite variable bound to ``feasibility_tolerance``. Anything else is reported with its
    reason so the caller can fall back to a cold start deliberately.
    """
    decision = ReuseDecision()
    matrix = csc_matrix(matrix)
    rows, columns = matrix.shape
    if len(basic) != rows:
        decision.reason = f'basis has {len(basic)} columns but the matrix has {rows} rows'
        return decision
    if len(set(basic)) != rows:
        decision.reason = 'basis contains duplicate columns'
        return decision
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)
    cost = np.asarray(costs, dtype=float).reshape(-1)
    right = np.zeros(rows) if rhs is None else np.asarray(rhs, dtype=float).reshape(-1)
    in_basis = np.zeros(columns, dtype=bool)
    in_basis[list(basic)] = True
    # 非基变量停在有界的一侧；两侧都无界时该基无法给出确定的基解。
    values = np.zeros(columns)
    for j in range(columns):
        if in_basis[j]:
            continue
        if math.isfinite(lower[j]) and math.isfinite(upper[j]):
            values[j] = lower[j]
        elif math.isfinite(lower[j]):
            values[j] = lower[j]
        elif math.isfinite(upper[j]):
            values[j] = upper[j]
        else:
            decision.reason = f'nonbasic variable {j} is free; no determinate basic solution'
            return decision
    try:
        basis = SparseBasis(matrix, basic, residual_tol=residual_tolerance)
    except NumericalError as exc:
        decision.reason = f'basis is singular or unfactorisable: {exc}'
        return decision
    nonbasic = {j: values[j] for j in range(columns) if not in_basis[j]}
    try:
        values[basis.basic] = basis.basic_solution(nonbasic)
    except (NumericalError, ValueError) as exc:
        decision.reason = f'basic solution could not be computed: {exc}'
        return decision
    decision.basis_valid = True
    equation_residual = float(np.max(np.abs(matrix @ values-right), initial=0.0))
    violation = 0.0
    for j in range(columns):
        if math.isfinite(lower[j]):
            violation = max(violation, lower[j]-values[j])
        if math.isfinite(upper[j]):
            violation = max(violation, values[j]-upper[j])
    violation = max(0.0, violation)
    decision.max_equation_residual = equation_residual
    decision.max_primal_violation = violation
    decision.basic_values = [float(values[j]) for j in basic]
    decision.objective = float(cost @ values)
    decision.primal_feasible = (violation <= feasibility_tolerance
                                and equation_residual <= max(residual_tolerance,
                                                             feasibility_tolerance))
    decision.checks = dict(basis_valid=True, equation_residual_ok=equation_residual <=
                           max(residual_tolerance, feasibility_tolerance),
                           bounds_ok=violation <= feasibility_tolerance)
    decision.reusable = bool(decision.basis_valid and decision.primal_feasible)
    if decision.reusable:
        decision.reason = ('stored basis is non-singular and primal feasible for the new '
                           'bounds; warm start is admissible')
    elif not decision.primal_feasible:
        decision.reason = (f'stored basis is not primal feasible for the new bounds '
                           f'(violation {violation:.3e}); a dual/Phase-I step or a cold '
                           f'start is required')
    return decision


def pack(matrix, result, costs, lower, upper, *, provenance=''):
    """Package a solved basis into a :class:`WarmStart` for the next solve."""
    if result.values is None or result.basic is None:
        raise ValueError('Cannot pack a result without a verified basis and values')
    cost = np.asarray(costs, dtype=float).reshape(-1)
    return WarmStart(basic=list(result.basic), lower=list(lower), upper=list(upper),
                     costs=list(cost), objective=float(result.objective),
                     iterations=int(getattr(result, 'iterations', 0) or 0),
                     provenance=str(provenance))
