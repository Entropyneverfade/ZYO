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
    nonbasic_values: dict = field(default_factory=dict)
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


@dataclass
class BasisStart:
    """A checked primal basis-state hint, not a solution or optimality certificate."""

    available: bool = False
    basic: list = field(default_factory=list)
    nonbasic_at_upper: list = field(default_factory=list)
    max_equation_residual: float = math.inf
    max_bound_violation: float = math.inf
    reconstruction_difference: float = math.inf
    reason: str = ''


def evaluate_reuse(matrix, basic, lower, upper, costs, *, rhs=None,
                   nonbasic_at_upper=None,
                   feasibility_tolerance=1e-7, residual_tolerance=1e-9):
    """Decide whether ``basic`` is a valid, primal-feasible basis for the NEW data.

    Returns a :class:`ReuseDecision`. ``reusable`` is true only when the basis is
    non-singular, satisfies ``A x = rhs`` to ``residual_tolerance``, and honours every
    finite variable bound to ``feasibility_tolerance``. ``nonbasic_at_upper`` explicitly
    records which nonbasic variables are at their upper bounds; omitted nonbasic variables
    are placed at their lower bounds. Without it the historical lower-first placement is
    retained. The exact checked nonbasic values are returned for the solver to reuse.
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
    if any(not isinstance(j, (int, np.integer)) or isinstance(j, (bool, np.bool_))
           or j < 0 or j >= columns for j in basic):
        decision.reason = 'basis column index is invalid or outside the model'
        return decision
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)
    cost = np.asarray(costs, dtype=float).reshape(-1)
    right = np.zeros(rows) if rhs is None else np.asarray(rhs, dtype=float).reshape(-1)
    if lower.size != columns or upper.size != columns or cost.size != columns or right.size != rows:
        decision.reason = 'bound, cost or right-hand-side vector length does not match the model'
        return decision
    in_basis = np.zeros(columns, dtype=bool)
    in_basis[list(basic)] = True
    upper_side = None
    if nonbasic_at_upper is not None:
        try:
            requested = list(nonbasic_at_upper)
        except TypeError:
            decision.reason = 'nonbasic upper-side indices must be an iterable'
            return decision
        if any(not isinstance(j, (int, np.integer)) or isinstance(j, (bool, np.bool_))
               or j < 0 or j >= columns for j in requested):
            decision.reason = 'nonbasic upper-side index is invalid or outside the model'
            return decision
        upper_side = set(requested)
        if len(upper_side) != len(requested):
            decision.reason = 'nonbasic upper-side indices contain duplicates'
            return decision
        if any(in_basis[j] for j in upper_side):
            decision.reason = 'nonbasic upper-side list contains a basic column'
            return decision
    # 非基变量停在有界的一侧；两侧都无界时该基无法给出确定的基解。
    values = np.zeros(columns)
    nonbasic = {}
    for j in range(columns):
        if in_basis[j]:
            continue
        if upper_side is not None:
            selected = upper[j] if j in upper_side else lower[j]
            if not math.isfinite(selected):
                decision.reason = ('nonbasic upper side has no finite upper bound' if j in upper_side
                                   else 'nonbasic lower side has no finite lower bound')
                return decision
            values[j] = selected
        elif math.isfinite(lower[j]):
            values[j] = lower[j]
        elif math.isfinite(upper[j]):
            values[j] = upper[j]
        else:
            decision.reason = f'nonbasic variable {j} is free; no determinate basic solution'
            return decision
        nonbasic[j] = float(values[j])
    try:
        basis = SparseBasis(matrix, basic, residual_tol=residual_tolerance)
    except NumericalError as exc:
        decision.reason = f'basis is singular or unfactorisable: {exc}'
        return decision
    try:
        # 一般行右端为 rhs：B x_B = rhs-A_N x_N。旧实现只按齐次方程回代，
        # 即使随后做 rhs 残差检查，也会错误拒绝本来可行的显式界侧起点。
        nonbasic_dense = values.copy()
        values[basis.basic] = basis.ftran(right-matrix @ nonbasic_dense)
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
    decision.nonbasic_values = nonbasic
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


def capture_basis_start(matrix, basic, values, lower, upper, *, rhs=None,
                        feasibility_tolerance=1e-7, residual_tolerance=1e-9):
    """Extract a checked basis and nonbasic bound sides from a primal-feasible point.

    This is a *warm-start hint*: a future solve must validate it again against its
    current model. It never certifies optimality or promises exact continuation of
    LU factors, eta history, pricing history, or iteration counters.
    """
    state = BasisStart()
    matrix = csc_matrix(matrix)
    rows, columns = matrix.shape
    points = np.asarray(values, dtype=float).reshape(-1)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)
    right = np.zeros(rows) if rhs is None else np.asarray(rhs, dtype=float).reshape(-1)
    if points.size != columns or lower.size != columns or upper.size != columns or right.size != rows:
        state.reason = 'basis-start vector length does not match the model'
        return state
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(right)):
        state.reason = 'basis-start point or right-hand side is non-finite'
        return state
    if len(basic) != rows or len(set(basic)) != rows or any(
            not isinstance(j, (int, np.integer)) or isinstance(j, (bool, np.bool_))
            or j < 0 or j >= columns for j in basic):
        state.reason = 'basis-start columns are invalid'
        return state
    state.basic = [int(j) for j in basic]
    residual = matrix @ points-right
    state.max_equation_residual = float(np.max(np.abs(residual), initial=0.0))
    finite_lower = np.isfinite(lower)
    finite_upper = np.isfinite(upper)
    lower_violation = np.max(lower[finite_lower]-points[finite_lower], initial=0.0)
    upper_violation = np.max(points[finite_upper]-upper[finite_upper], initial=0.0)
    state.max_bound_violation = float(max(0.0, lower_violation, upper_violation))
    if (state.max_equation_residual > max(residual_tolerance, feasibility_tolerance)
            or state.max_bound_violation > feasibility_tolerance):
        state.reason = 'basis-start point fails original equation or bound checks'
        return state
    basic_members = set(state.basic)
    upper_side = []
    for j in range(columns):
        if j in basic_members:
            continue
        # 固定变量优先记下界；其余非基变量必须真的贴在某一个有限界上。
        if finite_lower[j] and abs(points[j]-lower[j]) <= feasibility_tolerance:
            continue
        if finite_upper[j] and abs(points[j]-upper[j]) <= feasibility_tolerance:
            upper_side.append(j)
            continue
        state.reason = f'nonbasic variable {j} is at neither bound'
        return state
    # 使用新 LU 从记录的界侧重新算基解，确保“基+侧”足以复建原可行点。
    decision = evaluate_reuse(matrix, state.basic, lower, upper, np.zeros(columns),
                              rhs=right, nonbasic_at_upper=upper_side,
                              feasibility_tolerance=feasibility_tolerance,
                              residual_tolerance=residual_tolerance)
    if not decision.reusable:
        state.reason = 'captured basis did not validate on reconstruction: '+decision.reason
        return state
    difference = np.asarray(decision.basic_values)-points[state.basic]
    state.reconstruction_difference = float(np.max(np.abs(difference), initial=0.0))
    if state.reconstruction_difference > feasibility_tolerance:
        state.reason = 'reconstructed basic point differs from the captured point'
        return state
    state.nonbasic_at_upper = upper_side
    state.available = True
    state.reason = 'primal-feasible basis and all nonbasic bound sides reconstructed'
    return state


def pack(matrix, result, costs, lower, upper, *, provenance=''):
    """Package a solved basis into a :class:`WarmStart` for the next solve."""
    if result.values is None or result.basic is None:
        raise ValueError('Cannot pack a result without a verified basis and values')
    cost = np.asarray(costs, dtype=float).reshape(-1)
    return WarmStart(basic=list(result.basic), lower=list(lower), upper=list(upper),
                     costs=list(cost), objective=float(result.objective),
                     iterations=int(getattr(result, 'iterations', 0) or 0),
                     provenance=str(provenance))
