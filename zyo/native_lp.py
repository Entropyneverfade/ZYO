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

Contract: ``optimal`` is true only when an original-model certificate verifies. Normally
``status`` follows the solver. A bounded posthoc exception accepts a small basis after the
solver explicitly reports an unpriced reduced-cost stop: exact basis reconstruction and a
complete original-model certificate must both pass; the raw solver status stays in ``record``.
Other numerical stops are never promoted. A solver OPTIMAL candidate whose certificate fails
is reported as ``CERTIFICATE_FAILED``.
"""
from dataclasses import dataclass, field
import math

import numpy as np
from scipy.sparse import csc_matrix, hstack, identity

from .lp_certificate import lp_certificate
from .exact_basis_dual import reconstruct_exact_basis_dual
from .sparse_simplex import solve_lp
from .warm_start import evaluate_reuse, capture_basis_start

# 行关系与齐次编码的界：'<=' 取 (-inf, rhs]，'>=' 取 [rhs, inf)，'==' 取 [rhs, rhs]。
ROW_SENSE = ('<=', '>=', '==')
# 行界变量 s_i = a_i'x 的系数是 -e_i，因此 s_i 的简约成本恰为 r_{s_i} = y_i。
# 该恒等式是"齐次对偶 -> 原模型行乘子"映射的依据，也是本模块的自检项。
MULTIPLIER_IDENTITY_TOLERANCE = 1e-7


def _canonical_binary64_body(matrix):
    """复制并合并稀疏重复项，使求解矩阵与原模型证书使用同一系数。"""
    # SciPy 的 sum_duplicates 原地修改 CSC；必须先复制，不能改变调用方数据。
    # 模型系数定义为重复项按 binary64 聚合后的值；有限项相加也可能上溢。
    body = csc_matrix(matrix, dtype=float).copy()
    with np.errstate(over='ignore', invalid='ignore'):
        body.sum_duplicates()
    if not np.all(np.isfinite(body.data)):
        raise ValueError('matrix coefficients must be finite after duplicate aggregation')
    return body


def _validate_lp_numeric_input(body, lower, upper, rhs, cost=None):
    """在任何齐次编码或证书建模前固定原 LP 的数值定义域。"""
    rows, columns = body.shape
    if lower.size != columns or upper.size != columns:
        raise ValueError('bound vectors must match the number of columns')
    if rhs.size != rows:
        raise ValueError(f'right-hand side has {rhs.size} entries but the matrix has {rows} rows')
    if cost is not None and cost.size != columns:
        raise ValueError('cost vector must match the number of columns')
    # 稀疏矩阵只扫描实际存储的系数；显式 NaN/Inf 也不能在零乘法中被掩盖。
    if not np.all(np.isfinite(body.data)):
        raise ValueError('matrix coefficients must be finite')
    if cost is not None and not np.all(np.isfinite(cost)):
        raise ValueError('cost coefficients must be finite')
    if not np.all(np.isfinite(rhs)):
        raise ValueError('right-hand side must be finite')
    # 合法的开放侧只有下界 -Inf 与上界 +Inf；NaN 不是开放边界。
    if np.any(np.isnan(lower)) or np.any(np.isposinf(lower)):
        raise ValueError('lower bounds must be finite or -Inf')
    if np.any(np.isnan(upper)) or np.any(np.isneginf(upper)):
        raise ValueError('upper bounds must be finite or +Inf')


def _validate_nonnegative_tolerance(name, value):
    """拒绝会让比较恒为假的 NaN、无穷与负容差。"""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be finite and nonnegative')
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'{name} must be finite and nonnegative') from error
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f'{name} must be finite and nonnegative')


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
    restart_candidate: dict | None = None
    primal_feasible_candidate: np.ndarray | None = None
    degenerate_descent_candidate: np.ndarray | None = None
    degenerate_restart_candidate: dict | None = None


def build_homogeneous(matrix, lower, upper, rhs, sense=None):
    """Encode ``rows (sense, rhs)`` plus variable bounds as ``A z = 0``, ``l <= z <= u``.

    Returns ``(wide, wide_lower, wide_upper, row_lower, row_upper, logical_columns)``.
    ``wide = [body | -I]``; logical column ``n+i`` carries row ``i``'s activity, so its bounds
    are the row bounds and the equation ``(body x)_i - s_i = 0`` is exact.
    """
    body = _canonical_binary64_body(matrix)
    rows, columns = body.shape
    right = np.asarray(rhs, dtype=float).reshape(-1)
    lo = np.asarray(lower, dtype=float).reshape(-1)
    hi = np.asarray(upper, dtype=float).reshape(-1)
    if right.size != rows:
        raise ValueError(f'right-hand side has {right.size} entries but the matrix has {rows} rows')
    if lo.size != columns or hi.size != columns:
        raise ValueError('bound vectors must match the number of columns')
    _validate_lp_numeric_input(body, lo, hi, right)
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

    body = _canonical_binary64_body(matrix).tocsr()
    rows, columns = body.shape
    cost = np.asarray(costs, dtype=float).reshape(-1)
    right = np.asarray(rhs, dtype=float).reshape(-1)
    lo = np.asarray(lower, dtype=float).reshape(-1)
    hi = np.asarray(upper, dtype=float).reshape(-1)
    _validate_lp_numeric_input(body, lo, hi, right, cost)
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
                       warm_basis=None, warm_nonbasic_at_upper=None, require_warm_start=False,
                       iteration_limit=20000, feasibility_tolerance=1e-7,
                       objective_tolerance=1e-8, degenerate_rescue_iteration_limit=0,
                       degenerate_basis_bridge=False,
                       **kwargs):
    """Solve a general bounded LP natively and verify a KKT certificate on the result."""
    if warm_nonbasic_at_upper is not None and warm_basis is None:
        raise ValueError('warm_nonbasic_at_upper requires warm_basis')
    for name, value in (('feasibility_tolerance', feasibility_tolerance),
                        ('objective_tolerance', objective_tolerance)):
        _validate_nonnegative_tolerance(name, value)
    if 'pricing_tolerance' in kwargs:
        _validate_nonnegative_tolerance('pricing_tolerance', kwargs['pricing_tolerance'])
    if (isinstance(degenerate_rescue_iteration_limit, bool)
            or not isinstance(degenerate_rescue_iteration_limit, (int, np.integer))
            or degenerate_rescue_iteration_limit < 0):
        raise ValueError('degenerate_rescue_iteration_limit must be a nonnegative integer')
    if not isinstance(degenerate_basis_bridge, bool):
        raise ValueError('degenerate_basis_bridge must be Boolean')
    if degenerate_basis_bridge and degenerate_rescue_iteration_limit == 0:
        raise ValueError('degenerate_basis_bridge requires a positive rescue iteration limit')
    if require_warm_start and warm_basis is None:
        raise ValueError('required warm start needs warm_basis')
    if warm_basis is not None and 'initial' in kwargs:
        raise ValueError('Use warm_nonbasic_at_upper with warm_basis; unchecked initial is ambiguous')
    body = _canonical_binary64_body(matrix)
    rows, columns = body.shape
    cost = np.asarray(costs, dtype=float).reshape(-1)
    lo = np.asarray(lower, dtype=float).reshape(-1)
    hi = np.asarray(upper, dtype=float).reshape(-1)
    if cost.size != columns:
        raise ValueError('cost vector must match the number of columns')
    right = np.asarray(rhs, dtype=float).reshape(-1)
    _validate_lp_numeric_input(body, lo, hi, right, cost)
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

    def project_original(wide_values):
        # 自由变量拆分的点回投影至原变量；限额候选与最优证书共用同一映射。
        solved = np.asarray(wide_values[:len(keep)+2*len(free)], dtype=float)
        primal = np.zeros(columns)
        primal[keep] = solved[:len(keep)]
        for position, j in enumerate(free):
            primal[j] = solved[len(keep)+position]-solved[len(keep)+len(free)+position]
        return primal
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
                                  nonbasic_at_upper=warm_nonbasic_at_upper,
                                  feasibility_tolerance=feasibility_tolerance)
        reuse = dict(reusable=bool(decision.reusable), reason=decision.reason,
                     basis_valid=bool(decision.basis_valid),
                     primal_feasible=bool(decision.primal_feasible),
                     max_primal_violation=decision.max_primal_violation,
                     max_equation_residual=decision.max_equation_residual)
        record['warm_start'] = reuse

    # 冻结分段实验不得在状态无效时暗中冷启动，否则资源和数学轨迹不再对应清单。
    if require_warm_start and not reuse['reusable']:
        raise ValueError(f"required warm start rejected: {reuse['reason']}")

    start = list(warm_basis) if (reuse and reuse['reusable']) else None
    # 复用检查和实际迭代必须采用同一非基界侧；仅传基下标会让内核默认选另一侧。
    forward = dict(kwargs)
    if reuse and reuse['reusable']:
        forward['initial'] = decision.nonbasic_values
    outcome = solve_lp(wide, wide_cost, wide_lower, wide_upper, maximize=maximize,
                       basic=start, feasibility_tolerance=feasibility_tolerance,
                       iteration_limit=iteration_limit, **forward)
    record['solver_status'] = outcome.status
    record['solver_message'] = outcome.message
    # 与后验证书是否成功分开记录底层终止原因；大基拒绝重构时也不能丢失该事实。
    record['unpriced_optimality_stop'] = bool(getattr(outcome, 'unpriced_optimality', False))
    # 定价规则的实际使用情况随结果留痕：性能改动必须可审计，不能只写在文档里。
    record['pricing'] = dict(getattr(outcome, 'pricing', {}) or {})
    # 基误差和修正证据随认证结果输出，不能用 KKT 成功状态替代中间数值记录。
    record['basis_diagnostics'] = dict(outcome.basis_diagnostics)
    record['iteration_budget'] = dict(outcome.iteration_budget)
    record['bound_flips'] = int(outcome.bound_flips)
    if outcome.status in ('OPTIMAL', 'ITERATION_LIMIT') and outcome.values is not None:
        # 这两个数只量当前齐次基的最大界/简约成本符号违反；它们不是
        # 从原 MPS 独立复核的 KKT 证书，限额点更不得因此提升为最优。
        for source, target in (('max_primal_violation', 'internal_max_primal_violation'),
                               ('max_dual_violation', 'internal_max_dual_violation')):
            value = float(getattr(outcome, source))
            record[target] = value if math.isfinite(value) else None
    if outcome.status == 'ITERATION_LIMIT' and outcome.values is not None:
        # 当前齐次基的行乘子仅作候选，必须由原 MPS 独立重算弱对偶界；
        # 不填正式 dual、目标界或证书字段，也不改变求解状态。
        candidate = np.asarray(outcome.dual, dtype=float).reshape(-1)
        record['uncertified_row_multiplier_candidate'] = (
            candidate.tolist() if candidate.size == rows and np.all(np.isfinite(candidate))
            else None)
    result = CertifiedLPSolution(
        status=outcome.status, values=None, objective=outcome.objective,
        basic=list(outcome.basic), iterations=outcome.iterations, pivots=outcome.pivots,
        refactorisations=outcome.refactorisations, message=outcome.message,
        phase_one=dict(outcome.phase_one or {}), record=record)
    if (outcome.status == 'NUMERICAL_ERROR'
            and getattr(outcome, 'unpriced_optimality', False)):
        # 定价阈值只影响选列，不代表最优。仅对明确的未定价停止，尝试从
        # 当前小基精确求 B^T y=c_B，再由原模型证书独立复核全部条件。
        witness = reconstruct_exact_basis_dual(wide, wide_cost, outcome.basic)
        audit = dict(attempted=True, accepted=False,
                     reason=witness.refusal_reason,
                     max_numerator_bits=witness.max_numerator_bits,
                     max_denominator_bits=witness.max_denominator_bits)
        record['posthoc_unpriced_basis_certificate'] = audit
        if witness.multipliers is not None and outcome.values is None:
            audit['reason'] = 'solver returned no candidate point'
        if witness.multipliers is not None and outcome.values is not None:
            try:
                multipliers = np.asarray([float(item) for item in witness.multipliers])
                primal = project_original(outcome.values)
                if not np.all(np.isfinite(primal)) or not np.all(np.isfinite(multipliers)):
                    raise ValueError('nonfinite projected point or multiplier')
                model = build_model(body, cost, lo, hi, rhs, sense=sense,
                                    maximize=maximize)
                certificate = lp_certificate(
                    model, primal, multipliers,
                    feasibility_tolerance=feasibility_tolerance,
                    objective_tolerance=objective_tolerance,
                    direction='max' if maximize else 'min')
            except (ArithmeticError, OverflowError, ValueError) as error:
                audit['reason'] = f'candidate conversion refused: {error}'
            else:
                audit['original_certificate_checks'] = dict(certificate.checks)
                audit['reason'] = certificate.reason
                if certificate.verified:
                    # 原求解器仍为 NUMERICAL_ERROR；只由原单位证书提升公开状态。
                    # 有理行乘子字符串保留精确见证，浮点 dual 仅供接口数值使用。
                    audit['accepted'] = True
                    audit['exact_basis_multipliers'] = [str(item)
                                                        for item in witness.multipliers]
                    record['certificate_checks'] = dict(certificate.checks)
                    record['certificate_reason'] = certificate.reason
                    record['duality_gap'] = certificate.duality_gap
                    result.status = 'OPTIMAL'
                    result.optimal = True
                    result.values = primal
                    result.objective = certificate.objective
                    result.dual = multipliers
                    result.reduced_costs = np.asarray(certificate.reduced_costs)
                    result.certificate = certificate
                    result.message = ('native pricing stopped with unresolved reduced costs; '
                                      'a bounded exact-basis witness and independent original '
                                      'LP certificate verified this candidate')
                    return result
    if (outcome.status in ('ITERATION_LIMIT', 'NUMERICAL_ERROR')
            and outcome.phase_one_candidate is not None):
        # Phase-I 限额点，或 Phase-II 数值失败前的 Phase-I 已验证点，
        # 即使没有可复用的当前结构基，也可能已有原模型可行点。
        # 从原输入重新建模并只读取证书检查器的原始可行性部分；不使用零乘子
        # 所得的对偶/最优字段，不填正式解、目标、最优界或热启动基。
        candidate = project_original(outcome.phase_one_candidate)
        original_model = build_model(body, cost, lo, hi, rhs, sense=sense, maximize=maximize)
        primal_check = lp_certificate(original_model, candidate, np.zeros(rows),
                                      feasibility_tolerance=feasibility_tolerance,
                                      objective_tolerance=objective_tolerance,
                                      direction='max' if maximize else 'min')
        record['phase_one_candidate_check'] = dict(
            source=('stopped Phase-I structural point' if
                    outcome.phase_one.get('status') == 'ITERATION_LIMIT'
                    else 'completed Phase-I structural point before later failure'),
            original_primal_feasible=bool(primal_check.primal_feasible),
            original_row_violation=float(primal_check.row_violation),
            original_bound_violation=float(primal_check.bound_violation),
            original_optimality_certified=False,
            restart_basis_available=False)
        if primal_check.primal_feasible:
            result.primal_feasible_candidate = candidate
    if outcome.status == 'ITERATION_LIMIT' and outcome.values is not None and outcome.basic:
        # 限额点不是正式原模型解；只保存经原方程/界与新 LU 复建的基状态候选。
        start_hint = capture_basis_start(wide, outcome.basic, outcome.values,
                                         wide_lower, wide_upper,
                                         feasibility_tolerance=feasibility_tolerance)
        record['warm_restart'] = dict(available=bool(start_hint.available),
                                      reason=start_hint.reason,
                                      max_equation_residual=start_hint.max_equation_residual,
                                      max_bound_violation=start_hint.max_bound_violation,
                                      reconstruction_difference=start_hint.reconstruction_difference)
        if start_hint.available:
            result.restart_candidate = dict(basic=start_hint.basic,
                                            nonbasic_at_upper=start_hint.nonbasic_at_upper,
                                            **record['warm_restart'])
            # 与正式 values 字段隔离；后验审计要从原始 MPS 独立重算可行性。
            result.primal_feasible_candidate = project_original(outcome.values)
    if outcome.status != 'OPTIMAL' or outcome.values is None:
        if (degenerate_rescue_iteration_limit > 0 and outcome.status == 'ITERATION_LIMIT'
                and outcome.values is not None and result.restart_candidate is not None
                and result.primal_feasible_candidate is not None):
            # 仅在全部原行是等式、原变量界有限时试探；救援失败保留原状态、候选与基。
            relations = ['==']*rows if sense is None else [str(item) for item in sense]
            if len(relations) == rows and all(item == '==' for item in relations) \
                    and np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)):
                try:
                    from .degenerate_direction import attempt_degenerate_descent
                    rescue = attempt_degenerate_descent(
                        body, cost, lo, hi, rhs, result.primal_feasible_candidate,
                        iteration_limit=int(degenerate_rescue_iteration_limit),
                        maximize=maximize,
                        feasibility_tolerance=feasibility_tolerance,
                        objective_tolerance=objective_tolerance)
                except Exception as error:
                    # 隔离辅助分支不能把原生主求解的真实限额状态改写为异常或成功。
                    rescue = dict(attempted=True, accepted=False,
                                  claims_original_optimality=False,
                                  reason=f'auxiliary rescue failed: {type(error).__name__}: {error}')
                record['degenerate_rescue'] = rescue
                if rescue.get('accepted'):
                    result.degenerate_descent_candidate = np.asarray(
                        rescue['point_after'], dtype=float)
                if degenerate_basis_bridge:
                    # 边界点先经原模型行/界/成本复核，再单独重建齐次基；
                    # 任一环节拒绝时只保留原限额结果与原基，不作静默回退或最优宣称。
                    if rescue.get('boundary_available'):
                        try:
                            from .degenerate_basis_bridge import bridge_endpoint_to_basis
                            bridge = bridge_endpoint_to_basis(
                                body, cost, lo, hi, rhs, rescue['boundary_point'],
                                feasibility_tolerance=feasibility_tolerance,
                                objective_tolerance=objective_tolerance)
                        except Exception as error:
                            bridge = dict(available=False, basic=[], nonbasic_at_upper=[],
                                          claims_original_optimality=False,
                                          reason=f'boundary basis reconstruction failed: '
                                                 f'{type(error).__name__}: {error}')
                        if not isinstance(bridge, dict):
                            bridge = dict(available=False, basic=[], nonbasic_at_upper=[],
                                          claims_original_optimality=False,
                                          reason='boundary basis reconstruction returned no report')
                        if bridge.get('available'):
                            # 热启动重建容差不能吞掉原先证实的严格下降。
                            try:
                                before = float(rescue['objective_before'])
                                rebuilt = float(bridge['diagnostics']['objective'])
                                basic = bridge['basic']
                                upper_side = bridge['nonbasic_at_upper']
                                if bridge.get('claims_original_optimality') is not False:
                                    raise ValueError('basis bridge cannot claim original LP optimality')
                                if (not isinstance(basic, list)
                                        or not isinstance(upper_side, list)
                                        or len(basic) != rows):
                                    raise ValueError('bridge omitted a complete basis or bound sides')
                                gain = rebuilt-before if maximize else before-rebuilt
                                gate = objective_tolerance*max(1., abs(before))
                                if (not math.isfinite(gain) or not math.isfinite(gate)
                                        or gain <= gate):
                                    raise ValueError('basis lacks strict original cost improvement')
                            except (KeyError, TypeError, ValueError, OverflowError) as error:
                                bridge.update(available=False, basic=[],
                                              nonbasic_at_upper=[],
                                              reason=f'checked boundary basis refused: {error}')
                            else:
                                bridge['original_cost_improvement'] = float(gain)
                                result.degenerate_restart_candidate = bridge
                    else:
                        bridge = dict(available=False, basic=[], nonbasic_at_upper=[],
                                      claims_original_optimality=False,
                                      reason='auxiliary rescue has no independently checked boundary endpoint')
                    record['degenerate_basis_bridge'] = bridge
        if degenerate_basis_bridge and 'degenerate_basis_bridge' not in record:
            # 显式请求不得无痕略过；保留真实主状态并报告可选路径为何不适用。
            record['degenerate_basis_bridge'] = dict(
                available=False, basic=[], nonbasic_at_upper=[],
                claims_original_optimality=False,
                reason='parent stop, equality-domain, finite-bound, or checked-basis eligibility not met')
        return result

    # 拆分列必须先投影回原变量：x_keep 直接取用，第 i 个自由变量 x = p_i − n_i。
    primal = project_original(outcome.values)
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
        # 对外最优目标采用独立证书按原存储系数核算的值；裸 BLAS 点积在
        # 大数正负抵消时可把真实非零目标舍成 0，与已通过证书自相矛盾。
        result.objective = float(certificate.objective)
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
