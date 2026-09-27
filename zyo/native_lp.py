# 自研 LP 的一站式入口：齐次编码 -> 有界变量修正单纯形 -> 独立 KKT 证书 -> 热启动基复用。
# 只使用自研内核与 SciPy 稀疏线性代数；不导入任何外部优化引擎，也不静默回退。
"""Certified native LP solve: one entry point that ends in an independently checked certificate.

The pieces already exist and are separately accepted; this module wires them into a single
auditable path and — importantly — makes the *certificate*, not the solver's own status, the
thing that promotes a candidate to "optimal".

Pipeline
--------
1. **Homogeneous encoding.** Row ``i`` with activity ``a_i'x`` and bounds ``[lo_i, up_i]``
   becomes a logical column ``-e_i`` with the row's own bounds, so the system reads ``A z = 0``
   with ``z = [x ; s]`` and ``l <= z <= u`` (Huangfu & Hall arXiv:1503.01889v1 §2.1). Equality
   rows become fixed logical variables; this encoding is exact in both directions.
2. **Bounded-variable primal revised simplex** (:mod:`zyo.sparse_simplex`): sparse SuperLU
   basis, product-form updates, FTRAN/BTRAN residual gates, Phase-I only when the initial
   basis is not primal feasible, and ``START_INFEASIBLE`` instead of a wrong answer.
3. **KKT certificate** (:mod:`zyo.lp_certificate`): row multipliers ``lambda`` are recovered
   from the homogeneous dual ``y`` (for a ``<=`` row ``lambda = y <= 0``, for a ``>=`` row
   ``lambda = y >= 0``, free for equality rows), and every optimality condition is then
   re-derived in the ORIGINAL units from the model's stored coefficients.
4. **Warm start** (:mod:`zyo.warm_start`): an optional stored basis is only reused when it is
   non-singular and primal feasible for the *new* bounds; a rejection is recorded and the
   solve continues cold.

Contract: ``status`` is copied from the solver, but ``optimal`` is true only when the
certificate verifies. A solve whose candidate fails the certificate is reported as
``CERTIFICATE_FAILED`` with the failed conditions, never as optimal.
"""
from dataclasses import dataclass, field
import math

import numpy as np
from scipy.sparse import csc_matrix, hstack, identity

from .lp_certificate import lp_certificate
from .sparse_simplex import solve_lp
from .warm_start import evaluate_reuse

# 行关系与齐次编码的界：'<=' 取 (-inf, rhs]，'>=' 取 [rhs, inf)，'==' 取 [rhs, rhs]。
ROW_SENSE = ('<=', '>=', '==')
# 行界变量 s_i = a_i'x 的系数是 -e_i，因此 s_i 的简约成本恰为 r_{s_i} = y_i。
# 该恒等式是"齐次对偶 -> 原模型行乘子"映射的依据，也是本模块的自检项。
MULTIPLIER_IDENTITY_TOLERANCE = 1e-7


@dataclass
class CertifiedLPSolution:
    """Outcome of a certified native LP solve; nothing is promoted without evidence."""

    status: str
    optimal: bool = False
    values: np.ndarray | None = None
    objective: float | None = None
    dual: np.ndarray | None = None
    reduced_costs: np.ndarray | None = None
    basic: list = field(default_factory=list)
    certificate: object = None
    iterations: int = 0
    pivots: int = 0
    refactorisations: int = 0
    message: str = ''
    phase_one: dict = field(default_factory=dict)
    record: dict = field(default_factory=dict)


def build_homogeneous(matrix, lower, upper, rhs, sense=None):
    """Encode ``rows (sense, rhs)`` plus variable bounds as ``A z = 0``, ``l <= z <= u``.

    Returns ``(wide, wide_lower, wide_upper, row_lower, row_upper, logical_columns)``.
    ``wide = [body | -I]``; logical column ``n+i`` carries row ``i``'s activity, so its bounds
    are the row bounds and the equation ``(body x)_i - s_i = 0`` is exact.
    """
    body = csc_matrix(matrix, dtype=float)
    rows, columns = body.shape
    right = np.asarray(rhs, dtype=float).reshape(-1)
    lo = np.asarray(lower, dtype=float).reshape(-1)
    hi = np.asarray(upper, dtype=float).reshape(-1)
    if right.size != rows:
        raise ValueError(f'right-hand side has {right.size} entries but the matrix has {rows} rows')
    if lo.size != columns or hi.size != columns:
        raise ValueError('bound vectors must match the number of columns')
    relations = ['==']*rows if sense is None else [str(s) for s in sense]
    if len(relations) != rows or any(s not in ROW_SENSE for s in relations):
        raise ValueError("every sense entry must be '<=', '>=' or '==' and match the row count")
    row_lower = np.full(rows, -math.inf)
    row_upper = np.full(rows, math.inf)
    for index, relation in enumerate(relations):
        if relation == '<=':
            row_upper[index] = right[index]
        elif relation == '>=':
            row_lower[index] = right[index]
        else:
            row_lower[index] = row_upper[index] = right[index]
    # 逻辑块必须**稀疏**构造：`np.eye(rows)` 对 266227 行的实例是 7x10^10 个元素（约 567 GB），
    # `np.hstack([...])` 还会把原始矩阵整体稠密化。改为稀疏拼接后，端到端只按非零元工作，
    # 大行数实例第一次变得可尝试（此前只是被调用方的稠密门挡在门外）。
    logical = -identity(rows, format='csc')
    wide = hstack([body, logical], format='csc')
    wide_lower = np.concatenate([lo, row_lower])
    wide_upper = np.concatenate([hi, row_upper])
    return wide, wide_lower, wide_upper, row_lower, row_upper, list(range(columns, columns+rows))


def build_model(matrix, costs, lower, upper, rhs, sense=None, maximize=False, name='native_lp'):
    """Build a :class:`zyo.Model` mirroring the LP, for certificate verification.

    Rows are stored with the activity ``sum(a_j x_j) - rhs`` as the expression constant, which
    is the repository-wide "activity SENSE 0" convention the certificate relies on.

    The coefficients are read from the sparse rows directly. ``np.asarray`` on a SciPy sparse
    matrix yields a 0-d object array, so densifying through it fails with
    ``TypeError: float() argument must be ... not 'csc_matrix'``; densifying explicitly would
    also cost ``O(m n)`` memory for no benefit.
    """
    from lzyopt.model import Constraint, LinearExpression

    from .model import Model

    body = csc_matrix(matrix, dtype=float).tocsr()
    rows, columns = body.shape
    cost = np.asarray(costs, dtype=float).reshape(-1)
    right = np.asarray(rhs, dtype=float).reshape(-1)
    lo = np.asarray(lower, dtype=float).reshape(-1)
    hi = np.asarray(upper, dtype=float).reshape(-1)
    relations = ['==']*rows if sense is None else [str(s) for s in sense]
    model = Model(name)
    for index in range(columns):
        model.add_var(f'x{index}', lb=float(lo[index]) if math.isfinite(lo[index]) else None,
                      ub=float(hi[index]) if math.isfinite(hi[index]) else None)
    for index in range(rows):
        start, stop = int(body.indptr[index]), int(body.indptr[index+1])
        terms = {int(body.indices[k]): float(body.data[k]) for k in range(start, stop)}
        expression = LinearExpression(model, terms, -float(right[index]))
        model.add_constr(Constraint(expression, relations[index]))
    objective_terms = {j: float(cost[j]) for j in range(columns) if cost[j] != 0.0}
    model.set_objective(LinearExpression(model, objective_terms, 0.0),
                        'max' if maximize else 'min')
    return model


def solve_certified_lp(matrix, costs, lower, upper, rhs, *, sense=None, maximize=False,
                       warm_basis=None, iteration_limit=20000, feasibility_tolerance=1e-7,
                       objective_tolerance=1e-8, **kwargs):
    """Solve a general bounded LP natively and verify a KKT certificate on the result."""
    body = csc_matrix(matrix, dtype=float)
    rows, columns = body.shape
    cost = np.asarray(costs, dtype=float).reshape(-1)
    lo = np.asarray(lower, dtype=float).reshape(-1)
    hi = np.asarray(upper, dtype=float).reshape(-1)
    if cost.size != columns:
        raise ValueError('cost vector must match the number of columns')
    # 自由变量（两侧都无界）无法作为非基变量——有界变量形式要求非基变量停在某一侧界上，
    # 而它没有界可停。此前这里直接报 "Phase-I could not establish a basis"，实测 MIPLIB 的
    # `roll3000`（1 个自由变量）与 `sct2`（141 个）就卡在这一条上，而且被"稠密入口"这层假象
    # 掩盖着（它们先被稠密门记成 SIZE_LIMIT，稀疏化后才走到这一步）。
    #
    # 精确拆分解决：x_j = p_j − n_j，p_j, n_j ≥ 0 且无上界，于是两者都能停在 0 上。
    # * 投影到原变量上是**满射到整个实数轴**，可行集与目标值不变；
    # * 最优性条件也等价：拆分后要求 r⁺ = c_j − λ'a_j ≥ 0 且 r⁻ = −c_j + λ'a_j ≥ 0，
    #   合起来正是自由变量的 r_j = 0，与 lp_certificate 对自由变量的检查一致；
    # * 行方程不变，因此对偶乘子 λ 可直接用于原模型的证书。
    free = [j for j in range(columns)
            if not math.isfinite(lo[j]) and not math.isfinite(hi[j])]
    record_columns = dict(columns=columns, rows=rows, free_variables=len(free))
    if free:
        free_set = set(free)
        keep = [j for j in range(columns) if j not in free_set]
        augmented = hstack([body[:, keep], body[:, free], -body[:, free]], format='csc')
        augmented_lower = np.concatenate([lo[keep], np.zeros(2*len(free))])
        augmented_upper = np.concatenate([hi[keep], np.full(2*len(free), math.inf)])
        augmented_cost = np.concatenate([cost[keep], cost[free], -cost[free]])
        record_columns.update(split_free_variables=len(free), augmented_columns=len(keep)+2*len(free),
                              kept_columns=len(keep))
    else:
        keep = list(range(columns))
        augmented, augmented_lower, augmented_upper = body, lo, hi
        augmented_cost = cost
    wide, wide_lower, wide_upper, row_lower, row_upper, logical = build_homogeneous(
        augmented, augmented_lower, augmented_upper, rhs, sense=sense)
    wide_cost = np.concatenate([augmented_cost, np.zeros(rows)])

    record = dict(record_columns, logical_columns=logical,
                  # 无穷界以 None 表示：±inf 不是合法 JSON（归档时会报
                  # "Out of range float values are not JSON compliant"），而审计需要的是
                  # "该侧无界"这一事实，不是某个具体的浮点值。
                  row_lower=[None if not math.isfinite(v) else float(v) for v in row_lower],
                  row_upper=[None if not math.isfinite(v) else float(v) for v in row_upper])
    reuse = None
    if warm_basis is not None:
        # 只在新数据下**验证过**才复用；拒绝原因同样留痕，不做静默冷启动。
        decision = evaluate_reuse(wide, list(warm_basis), wide_lower, wide_upper, wide_cost,
                                  feasibility_tolerance=feasibility_tolerance)
        reuse = dict(reusable=bool(decision.reusable), reason=decision.reason,
                     basis_valid=bool(decision.basis_valid),
                     primal_feasible=bool(decision.primal_feasible),
                     max_primal_violation=decision.max_primal_violation,
                     max_equation_residual=decision.max_equation_residual)
        record['warm_start'] = reuse

    start = list(warm_basis) if (reuse and reuse['reusable']) else None
    outcome = solve_lp(wide, wide_cost, wide_lower, wide_upper, maximize=maximize,
                       basic=start, feasibility_tolerance=feasibility_tolerance,
                       iteration_limit=iteration_limit, **kwargs)
    record['solver_status'] = outcome.status
    record['solver_message'] = outcome.message
    # 定价规则的实际使用情况随结果留痕：性能改动必须可审计，不能只写在文档里。
    record['pricing'] = dict(getattr(outcome, 'pricing', {}) or {})
    # 基误差和修正证据随认证结果输出，不能用 KKT 成功状态替代中间数值记录。
    record['basis_diagnostics'] = dict(outcome.basis_diagnostics)
    record['iteration_budget'] = dict(outcome.iteration_budget)
    record['bound_flips'] = int(outcome.bound_flips)
    result = CertifiedLPSolution(
        status=outcome.status, values=None, objective=outcome.objective,
        basic=list(outcome.basic), iterations=outcome.iterations, pivots=outcome.pivots,
        refactorisations=outcome.refactorisations, message=outcome.message,
        phase_one=dict(outcome.phase_one or {}), record=record)
    if outcome.status != 'OPTIMAL' or outcome.values is None:
        return result

    # 拆分列必须先投影回原变量：x_keep 直接取用，第 i 个自由变量 x = p_i − n_i。
    solved = np.asarray(outcome.values[:len(keep)+2*len(free)], dtype=float)
    primal = np.zeros(columns)
    primal[keep] = solved[:len(keep)]
    for position, j in enumerate(free):
        primal[j] = solved[len(keep)+position] - solved[len(keep)+len(free)+position]
    dual = np.asarray(outcome.dual, dtype=float).reshape(-1)
    # 齐次对偶 y 就是原模型的行乘子：s_i 的简约成本 r_{s_i} = 0 - (-e_i)'y = y_i，
    # 而原列 j 的简约成本是 r_j = c_j - (body'y)_j，与证书的 r = c - A'lambda 同式。
    # 自由变量拆分会新增列，但**行没有变**，因此 y 仍然就是原模型的行乘子。
    multipliers = dual
    # 自检：用返回的简约成本独立核对上面的映射，而不是假定它成立。
    # `logical` 已经是 wide 中的列下标（build_homogeneous 返回时就带上了增广列的偏移）。
    reduced = np.asarray(outcome.reduced_costs, dtype=float)
    scale = 1.0+float(np.max(np.abs(reduced), initial=0.0))
    identity_gap = float(np.max(np.abs(reduced[logical]-multipliers), initial=0.0)) \
        if rows else 0.0
    record['multiplier_identity_gap'] = identity_gap

    model = build_model(body, cost, lo, hi, rhs, sense=sense, maximize=maximize)
    certificate = lp_certificate(model, primal, multipliers,
                                 feasibility_tolerance=feasibility_tolerance,
                                 objective_tolerance=objective_tolerance,
                                 direction='max' if maximize else 'min')
    result.values = primal
    result.dual = multipliers
    result.reduced_costs = reduced[:columns]
    result.objective = float(cost @ primal)
    result.certificate = certificate
    record['certificate_checks'] = dict(certificate.checks)
    record['certificate_reason'] = certificate.reason
    record['duality_gap'] = certificate.duality_gap
    record['multiplier_scale'] = scale
    if certificate.verified and identity_gap <= MULTIPLIER_IDENTITY_TOLERANCE*scale:
        result.optimal = True
        result.message = ('solved by the ZYO revised simplex and verified by an independent '
                          'KKT certificate in original units')
    else:
        # 求解器自报最优但证书不通过：如实降级，并保留求解器原状态供审计。
        result.optimal = False
        result.status = 'CERTIFICATE_FAILED'
        result.message = (f'the solver reported {outcome.status} but the independent certificate '
                          f'failed: {certificate.reason}'
                          + (f'; multiplier identity gap {identity_gap:.3e}' if rows else ''))
    return result
