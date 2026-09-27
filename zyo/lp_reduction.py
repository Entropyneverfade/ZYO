# 一般有界 LP 归约到标准形：把任意变量域与行界改写成 min c'x s.t. Ax=b, x>=0。
# 归约是精确的（双射），解通过映射回代恢复；行界用逻辑列承载，非负变量自带 x>=0。
"""Exact reduction of a bounded LP to standard form.

The verified textbook simplex solves only ``min c'z s.t. Az = b, z >= 0``. This module
reduces a general LP to that form with an *exact* transformation, so the optimum and the
solution map back without approximation:

* ``x_j >= l_j``      -> ``x_j = l_j + z_j``, ``z_j >= 0``
* ``l_j <= x_j <= u_j`` -> ``x_j = l_j + z_j``, ``0 <= z_j <= u_j - l_j`` (upper bound
  becomes an explicit row ``z_j + s = u_j - l_j`` with a slack)
* ``x_j <= u_j``      -> ``x_j = u_j - z_j``, ``z_j >= 0`` (row ``z_j + s = u_j - l_j``)
* ``x_j`` free        -> ``x_j = z_j^+ - z_j^-``, both ``>= 0`` (one may be dropped when it
  is provably unnecessary, but splitting is kept for clarity and exactness)
* row ``sum a x <= b`` -> ``sum a x + s = b``, ``s >= 0`` (``>=`` multiplies the row by -1)

Every bound is represented by a row plus a nonnegative slack, which keeps the standard form
free of variable upper bounds.

Only *inequality* rows receive a slack column. An equality row must NOT receive one: adding
``s >= 0`` turns ``x - y == 4`` into ``x - y <= 4``, which is a strictly weaker feasible set
(measured: the relaxed problem returned objective 0 where the true optimum is 4). Equality
rows are therefore initialised by Phase-I artificial variables, which :func:`two_phase_simplex`
drives out of the basis before Phase-II starts, so no column that could relax the equality
survives into the final basis. Negative right-hand sides are handled inside
:func:`zyo.lp_phase_one.two_phase_simplex`, which negates the row (an exact equivalence for an
equality row) so that the artificial start ``(x = 0, artificials = |b|)`` is feasible.
"""
from dataclasses import dataclass, field
import math

import numpy as np
from scipy.sparse import csc_matrix

from .lp_phase_one import two_phase_simplex


@dataclass
class Reduction:
    """The standard-form counterpart of a bounded LP plus its solution map."""

    matrix: object = None            # 标准形矩阵（稠密或稀疏）
    costs: np.ndarray | None = None
    basic: list = field(default_factory=list)   # 逻辑列下标；等式行为 -1（由 Phase-I 提供人工变量）
    rhs: np.ndarray | None = None
    columns: list = field(default_factory=list)   # 每个原始变量对应的 (base, [(col, sign)])
    artifacts: dict = field(default_factory=dict)


@dataclass
class ReducedResult:
    status: str
    values: np.ndarray | None = None
    objective: float | None = None
    message: str = ''
    standard: dict = field(default_factory=dict)


def reduce_bounded_lp(matrix, costs, lower, upper, rhs, *, maximize=False, sense=None,
                      tolerance=1e-12):
    """Reduce a bounded LP to standard form.

    ``sense`` gives each row's relation to ``rhs``: ``'<='``, ``'>='`` or ``'=='``. When
    omitted every row is treated as an equality. An inequality row gets a nonnegative slack
    (``<=``) or surplus (``>=``) column, exactly as in the textbook standard form.

    Returns a :class:`Reduction`. ``columns[j]`` records how original variable ``j`` is
    rebuilt from the standard-form variables.
    """
    body = np.asarray(matrix, dtype=float)
    cost = np.asarray(costs, dtype=float).reshape(-1)
    lo = np.asarray(lower, dtype=float).reshape(-1)
    hi = np.asarray(upper, dtype=float).reshape(-1)
    right = np.asarray(rhs, dtype=float).reshape(-1)
    rows, columns = body.shape
    direction = -1.0 if maximize else 1.0
    relations = ['==']*rows if sense is None else [str(s) for s in sense]
    if len(relations) != rows or any(s not in ('<=', '>=', '==') for s in relations):
        raise ValueError("sense entries must be '<=', '>=' or '==' and match the row count")
    # '>=' 行先整体取反成 '<='，使符号处理只有一处。
    for row in range(rows):
        if relations[row] == '>=':
            body[row, :] *= -1.0
            right[row] *= -1.0
            relations[row] = '<='

    entries = []      # 标准形列：(原始列索引, 系数) 仅用于装配
    plans = []        # 每个原始变量的重建方案
    upper_rows = []   # 需要加"上界行"的原始变量
    for j in range(columns):
        finite_lo, finite_hi = math.isfinite(lo[j]), math.isfinite(hi[j])
        if finite_lo and finite_hi:
            plans.append(('range', lo[j], [(len(entries), 1.0)]))
            entries.append(('shift', j, 1.0))
            upper_rows.append((j, len(entries)-1, hi[j]-lo[j]))
        elif finite_lo:
            plans.append(('lower', lo[j], [(len(entries), 1.0)]))
            entries.append(('shift', j, 1.0))
        elif finite_hi:
            # x = u - z（z >= 0）本身就已经精确表达了 x <= u，无需再加界行。
            # 曾经这里也追加了一行 z + s = u，那把 z 限制成 z <= u，等价于 0 <= x <= u，
            # 对"只有上界"的变量是**多余的下界**（实测 x,y<=0、x+y>=-5 被压成 x=y=0，
            # 目标从真值 -10 变成 0）。只有 range 型需要显式界行。
            plans.append(('upper', hi[j], [(len(entries), -1.0)]))
            entries.append(('flip', j, 1.0))
        else:
            plus, minus = len(entries), len(entries)+1
            plans.append(('free', 0.0, [(plus, 1.0), (minus, -1.0)]))
            entries.append(('plus', j, 1.0))
            entries.append(('minus', j, 1.0))
    ncols = len(entries)
    nrows = rows+len(upper_rows)
    # 逻辑列只分配给不等式行（含变量界行）。等式行不分配，理由见模块文档：
    # 给等式行加松弛列会把 == 放松成 <=，实测 x-y==4 被放松后最优值 0 而非 4。
    logical_rows = [row for row in range(rows) if relations[row] != '=='] \
        + [rows+offset for offset in range(len(upper_rows))]
    logical = [-1]*nrows
    for offset, row in enumerate(logical_rows):
        logical[row] = ncols+offset
    total = ncols+len(logical_rows)

    dense = np.zeros((nrows, total))
    standard_rhs = np.zeros(nrows)
    standard_rhs[:rows] = right.copy()
    # Σ a_j x_j = Σ a_j (base_j + Σ sign·z_col) = Σ a_j base_j + Σ (a_j sign) z_col。
    # 常数项 Σ a_j base_j 必须移到右端，否则归约后的行与原行不等价
    # （实测：x,y>=1 的行 x+y=4 被写成 z1+z2=4，恢复出 x+y=6）。
    # 注意：'>=' 行在上方已整行取反，因此这里一律用**取反后**的 body 装配系数，
    # 而 shift 也用同一套系数计算，二者必须同源。
    for row in range(rows):
        shift = 0.0
        for j in range(columns):
            a = body[row, j]
            if a == 0.0:
                continue
            kind, base, terms = plans[j]
            shift += a*base
            for column, sign in terms:
                dense[row, column] += a*sign
        standard_rhs[row] = right[row]-shift
    for offset, (j, column, width) in enumerate(upper_rows):
        dense[rows+offset, column] = 1.0
        standard_rhs[rows+offset] = width
    # 不等式行的逻辑列系数恒为 +1：它就是松弛/剩余变量，`Σ a x + s = b`、`s >= 0`
    # 与原不等式完全等价（'>=' 行已整行取反，故此处一律 +1）。
    for row in logical_rows:
        dense[row, logical[row]] = 1.0
    standard_cost = np.zeros(total)
    # 归约只做变量代换，**不乘 direction、也不乘 sign**：standard_cost 就是"用户目标在
    # 归约变量上的系数"。求解方（solve_bounded_lp）再按 maximize 决定是否取反。
    # 先前把 direction 和 sign 一起乘进去，等于把最大化翻了两次、把 upper 型变量也翻了
    # 一次，实测最大化题返回 0、只上界题返回不可行。
    for j in range(columns):
        kind, base, terms = plans[j]
        for column, sign in terms:
            standard_cost[column] += cost[j]*sign
    # 这里**不再**对负右端行做整行取反：取反会把该行逻辑列的 +1 变成 -1，破坏初始基，
    # 而 Phase-I 的人工变量本来就能从任意右端出发（内部会取反并用 |b| 作人工初值）。
    basic = [logical[row] for row in range(nrows)]
    reduction = Reduction(matrix=dense, costs=standard_cost, basic=basic, rhs=standard_rhs,
                          columns=[(kind, base, terms) for kind, base, terms in plans],
                          artifacts=dict(direction=direction, original_columns=columns,
                                         original_rows=rows, upper_rows=upper_rows,
                                         standard_columns=total, logical_columns=logical,
                                         equality_rows=[row for row in range(rows)
                                                        if relations[row] == '=='],
                                         negative_rhs_rows=[row for row in range(nrows)
                                                            if standard_rhs[row] < -tolerance]))
    return reduction


def solve_bounded_lp(matrix, costs, lower, upper, rhs, *, maximize=False, sense=None,
                     iteration_limit=2000):
    """Reduce, solve with the verified standard simplex, and map the solution back."""
    matrix = np.asarray(matrix, dtype=float)
    costs = np.asarray(costs, dtype=float).reshape(-1)
    lower = np.asarray(lower, dtype=float).reshape(-1)
    upper = np.asarray(upper, dtype=float).reshape(-1)
    rhs = np.asarray(rhs, dtype=float).reshape(-1)
    reduction = reduce_bounded_lp(matrix, costs, lower, upper, rhs, maximize=maximize,
                                  sense=sense)
    if reduction.matrix is None:
        return ReducedResult('INFEASIBLE', message=reduction.artifacts.get('reason', ''),
                             standard=dict(reduction.artifacts))
    # 内部统一按最小化求解：最大化时把目标取反，最后再把最优值翻回来。
    internal_costs = -np.asarray(reduction.costs, dtype=float) if maximize \
        else np.asarray(reduction.costs, dtype=float)
    standard_matrix = np.array(reduction.matrix, dtype=float, copy=True)
    standard_rhs = np.array(reduction.rhs, dtype=float, copy=True)
    # 逻辑列列表里的 -1 表示"该等式行没有逻辑列"，其初始基由 Phase-I 的人工变量提供，
    # 因此只有全部行都有逻辑列时才把 basic 传下去（否则相位一自己决定初值）。
    standard_basic = list(reduction.basic)
    if any(column < 0 for column in standard_basic):
        standard_basic = None
    # 负右端行只做记录：修复由 Phase-I 完成（它会整行取反并用 |b| 作人工初值）。
    negative_rows = [row for row in range(standard_matrix.shape[0])
                     if standard_rhs[row] < -1e-12]
    outcome = two_phase_simplex(standard_matrix, internal_costs,
                                standard_rhs, basic=standard_basic,
                                iteration_limit=iteration_limit)
    detail = dict(standard_status=outcome.status,
                  standard_pivots=int(outcome.phase_one.get('pivots', 0)),
                  standard_columns=reduction.artifacts['standard_columns'],
                  standard_rows=int(np.asarray(reduction.rhs).size),
                  negative_rhs_rows=negative_rows,
                  equality_rows=list(reduction.artifacts.get('equality_rows', [])),
                  phase_one=dict(outcome.phase_one))
    if outcome.status != 'OPTIMAL' or outcome.values is None:
        # 归约是双射，因此标准形的状态就是原问题的状态：UNBOUNDED 与 INFEASIBLE
        # 必须原样上报，不能因为"拿到了一个值"就当成最优（实测 min -x 无上界被当成最优）。
        message = outcome.message
        if outcome.status == 'UNBOUNDED':
            message = ('the reduced problem is unbounded; because the reduction is a bijection, '
                       'the original LP is unbounded too')
        elif outcome.status == 'INFEASIBLE':
            message = ('Phase-I proved the reduced rows infeasible; the original LP is '
                       'infeasible')
        return ReducedResult(outcome.status, message=message, standard=detail)
    z = outcome.values[:reduction.artifacts['standard_columns']]
    values = np.zeros(len(reduction.columns))
    for j, (kind, base, terms) in enumerate(reduction.columns):
        values[j] = base+math.fsum(z[column]*sign for column, sign in terms)
    direction = reduction.artifacts['direction']
    objective = float(costs @ values)
    # 独立复核：按**原始行关系**（含 sense）核对，而不是一律当等式。
    # 把 '<=' 行当等式会把 x+y=2 <= 4 误判为违反 2（实测）。
    relations = ['==']*matrix.shape[0] if sense is None else [str(s) for s in sense]
    row_violation = 0.0
    for row in range(matrix.shape[0]):
        activity = math.fsum([0.0, *[a*values[j] for j, a in enumerate(matrix[row])]])
        relation = relations[row]
        if relation == '==':
            violation = abs(activity-rhs[row])
        elif relation == '<=':
            violation = max(0.0, activity-rhs[row])
        else:
            violation = max(0.0, rhs[row]-activity)
        row_violation = max(row_violation, violation)
    bound_violation = 0.0
    for j in range(values.size):
        if math.isfinite(lower[j]):
            bound_violation = max(bound_violation, lower[j]-values[j])
        if math.isfinite(upper[j]):
            bound_violation = max(bound_violation, values[j]-upper[j])
    detail.update(row_violation=row_violation, bound_violation=float(max(0.0, bound_violation)))
    if max(row_violation, bound_violation) > 1e-7:
        return ReducedResult('NUMERICAL_ERROR', values=values, objective=objective,
                             message=f'recovered point violates the original data '
                                     f'(row {row_violation:.3e}, bound {bound_violation:.3e})',
                             standard=detail)
    return ReducedResult(outcome.status, values=values, objective=objective,
                         message='reduced to standard form, solved by two-phase simplex, and '
                                 'mapped back with an original-relation check',
                         standard=detail)
