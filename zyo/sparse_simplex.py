# 自研稀疏基维护：基矩阵 LU 分解、FTRAN/BTRAN 回代与乘积形式（eta 文件）秩一更新。
# 只使用 SciPy 稀疏线性代数（SuperLU）作为基础分解组件，不调用任何优化引擎。
"""ZYO-authored sparse basis with FTRAN/BTRAN and product-form updates.

Computational form follows Huangfu & Hall (arXiv:1503.01889v1, §2.1): the problem is
``minimize c'x`` subject to ``Ax = 0`` and ``l <= x <= u``, where row bounds are
represented by an identity submatrix inside ``A``. A basis ``B`` is a set of ``m``
linearly independent columns of ``A``; basic variables satisfy ``B x_B = -N x_N``.

Two linear-system operations are required by the revised simplex method:

* FTRAN: solve ``B d = a`` for a column ``a`` (used to form the pivot column).
* BTRAN: solve ``B' y = c_B`` for the basic costs (used to form dual values).

``B`` is refactorised periodically by sparse LU. Between refactorisations the
factorisation is updated through a *product form* (eta file): each pivot appends one
sparse vector and one pivot position, and any solve applies those updates in the
correct order. This replaces the rank-one update of a dense factorisation while
keeping the update algebra explicit and independently checkable.

Numerical safety is not optional here: every solve is validated against the CURRENT
basis (``B_cur x = b``, ``B_cur' y = c``) rather than against the stale factors, and a
bounded residual corrections use the same CURRENT-basis solve. Every correction is
recorded. The default gate preserves the original RHS-relative criterion; a strict
componentwise criterion is available explicitly for research ablations.
"""
from dataclasses import asdict, dataclass, field
import math
import operator

import numpy as np
from scipy.sparse import csc_matrix, diags, hstack, identity
from scipy.sparse.linalg import splu

from .errors import NumericalError

# 默认重分解间隔与 eta 文件规模上限：超限即要求重分解，避免更新链过长导致误差累积。
DEFAULT_REFACTOR_INTERVAL = 100
DEFAULT_MAX_ETA = 200
# 默认保留原 rhs 相对残差门；逐分量后向误差独立记录并可显式消融。
# LP 原尺度可行性/KKT 门保持，不由基误差单独判定最优性。
DEFAULT_RESIDUAL_TOL = 1e-9
DEFAULT_REFINEMENT_STEPS = 2
# 定价规则切换门：先用 Dantzig，枢轴数超过 DEFAULT_BLAND_AFTER 或连续 DEFAULT_STALL_AFTER
# 次枢轴毫无进展时**永久**改用 Bland 规则。Bland 规则的有限终止保证依赖"最终一直用它"，
# 因此切换不可逆；两个门都是"何时开始用它"的触发条件，不改变保证本身。
DEFAULT_BLAND_AFTER = 1000
DEFAULT_STALL_AFTER = 100


@dataclass
class BasisUpdate:
    """One product-form pivot update.

    After the pivot, ``B_new = B_old + (a - B e_p) e_p'`` with ``a`` the entering
    column. Writing ``eta = B_old^{-1} a``, the inverse updates as

        ``B_new^{-1} = (I - (eta - e_p) e_p' / eta_p) B_old^{-1}``,

    so the stored vector is the normalised product-form column
    ``gamma = (eta - e_p) / eta_p`` (with ``gamma_p = (eta_p - 1)/eta_p``).
    Applying one update to a vector ``y`` shifted by the pivot position ``p`` is then
    ``y -= gamma * y[p]``.
    """

    pivot: int          # 换入列在基中的位置（0-based）
    gamma: np.ndarray   # 归一化乘积形式列 (eta - e_p) / eta_p
    eta: np.ndarray     # 原始 FTRAN 列 B_old^{-1} a，保留以备诊断
    entering: int       # 换入列在 A 中的全局列号
    leaving: int        # 换出列在 A 中的全局列号


@dataclass
class BasisState:
    """Result of a basis validity check, kept for evidence rather than discarded."""

    ftran_residual: float = 0.0
    btran_residual: float = 0.0
    eta_count: int = 0
    stale: bool = False
    reason: str = ''
    refactorisations: int = 0
    pivots_since_refactor: int = 0
    ftran_backward_error: float = 0.0
    btran_backward_error: float = 0.0
    solve_checks: int = 0
    refinement_attempts: int = 0
    refinement_improvements: int = 0
    residual_gate_failures: int = 0
    scaled_accepts: int = 0
    max_rhs_residual: float = 0.0
    max_backward_error: float = 0.0


class SparseBasis:
    """Sparse basis with FTRAN/BTRAN, product-form updates and explicit staleness.

    Parameters
    ----------
    matrix:
        The ``m x n`` constraint matrix in CSC form. Columns are all variables,
        including the identity columns representing row bounds.
    basic:
        Global column indices of the initial ``m`` basic columns. Must give a
        non-singular ``m x m`` matrix.
    residual_tol:
        RHS-relative residual gate (or explicit componentwise research gate).
    refinement_steps:
        Maximum CURRENT-basis correction solves per validated operation.
    residual_metric:
        ``rhs`` (default, preserved criterion) or ``componentwise`` (experimental).

    The caller owns the LP semantics; this class only performs basis linear algebra.
    It never decides optimality, feasibility or a pivot choice.
    """

    def __init__(self, matrix, basic, *, residual_tol=DEFAULT_RESIDUAL_TOL,
                 refactor_interval=DEFAULT_REFACTOR_INTERVAL, max_eta=DEFAULT_MAX_ETA,
                 refinement_steps=DEFAULT_REFINEMENT_STEPS, residual_metric='rhs'):
        self.matrix = csc_matrix(matrix)
        self.basic = [int(j) for j in basic]
        if len(self.basic) != self.matrix.shape[0]:
            raise ValueError('Basis size must equal the number of rows')
        if len(set(self.basic)) != len(self.basic):
            raise ValueError('Basis contains duplicate columns')
        self.residual_tol = float(residual_tol)
        if not math.isfinite(self.residual_tol) or self.residual_tol <= 0:
            raise ValueError('residual_tol must be finite and positive')
        if (int(refinement_steps) != refinement_steps
                or not 0 <= refinement_steps <= DEFAULT_REFINEMENT_STEPS):
            raise ValueError('refinement_steps must be an integer between zero and two')
        if residual_metric not in ('rhs', 'componentwise'):
            raise ValueError('residual_metric must be rhs or componentwise')
        self.refinement_steps = int(refinement_steps)
        self.residual_metric = residual_metric
        self.refactor_interval = int(refactor_interval)
        self.max_eta = int(max_eta)
        self.updates = []
        self.state = BasisState()
        self._factorization = None
        self._basis_matrix = None
        self.refactorise()

    # ---------- 基矩阵装配与分解 ----------

    def assemble(self):
        """Current ``B`` as a CSC matrix, columns in the order of ``self.basic``."""
        return csc_matrix(self.matrix[:, self.basic])

    def refactorise(self):
        """Rebuild the LU factors of the current basis and drop the update chain."""
        matrix = self.assemble()
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError('Basis matrix must be square')
        try:
            factorization = splu(matrix.tocsc())
        except RuntimeError as exc:
            raise NumericalError('Basis matrix is singular or unfactorisable: '+str(exc)) from exc
        if not np.all(np.isfinite(factorization.U.data)):
            raise NumericalError('Basis LU factors contain non-finite values')
        self._factorization = factorization
        self._basis_matrix = matrix
        self.updates = []
        self.state.eta_count = 0
        self.state.refactorisations += 1
        self.state.pivots_since_refactor = 0
        self.state.stale = False
        self.state.reason = ''
        return self

    # ---------- 基坐标与全空间坐标 ----------

    def ftran(self, vector, validate=True):
        """Solve ``B d = vector`` for the CURRENT basis (dense RHS accepted)."""
        rhs = np.asarray(vector, dtype=float).reshape(-1)
        if rhs.size != self.matrix.shape[0]:
            raise ValueError('FTRAN right-hand side has the wrong length')
        if not np.all(np.isfinite(rhs)):
            raise NumericalError('FTRAN right-hand side is non-finite')
        solution = self._solve_current(rhs, transpose=False)
        if validate:
            solution = self._check(solution, rhs, transpose=False)
        return solution

    def btran(self, vector, validate=True):
        """Solve ``B' y = vector`` for the CURRENT basis (dense RHS accepted)."""
        rhs = np.asarray(vector, dtype=float).reshape(-1)
        if rhs.size != self.matrix.shape[0]:
            raise ValueError('BTRAN right-hand side has the wrong length')
        if not np.all(np.isfinite(rhs)):
            raise NumericalError('BTRAN right-hand side is non-finite')
        solution = self._solve_current(rhs, transpose=True)
        if validate:
            solution = self._check(solution, rhs, transpose=True)
        return solution

    def _solve_current(self, rhs, *, transpose):
        # 残差修正复用同一个原始算子：必须包括全部 eta，不能只用旧 LU，也不递归触发检查。
        if not transpose:
            solution = self._factorization.solve(rhs)
            for update in self.updates:
                solution -= update.gamma*solution[update.pivot]
            return solution
        solution = rhs.copy()
        # 后向应用同一链的转置：y[p] -= gamma' y，等价于先更新再消去。
        for update in reversed(self.updates):
            solution[update.pivot] -= float(update.gamma @ solution)
        solution = self._factorization.solve(solution, trans='T')
        return solution

    def _check(self, solution, rhs, *, transpose):
        """Return a checked/refined solution, using the CURRENT basis and operator.

        LAPACK 3.12.1 DGERFS defines componentwise backward error using
        ``abs(op(B)) @ abs(y) + abs(rhs)``. This is a linear-algebra check, not
        an LP feasibility/optimality certificate. The former RHS residual is retained.
        """
        self.state.solve_checks += 1
        if not np.all(np.isfinite(solution)):
            self.mark_stale('solve produced non-finite values')
            raise NumericalError('Basis solve produced non-finite values')
        matrix = self.assemble()
        operator = matrix.T if transpose else matrix
        scale = 1.0+float(np.max(np.abs(rhs), initial=0.0))

        def measure(candidate):
            # 按 op(B) 的每个分量计量，防止某个大行掩盖小行；0/0 的精确零方程记零。
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                residual = rhs-operator @ candidate
                denominator = abs(operator) @ np.abs(candidate)+np.abs(rhs)
                ratios = np.divide(np.abs(residual), denominator,
                                   out=np.zeros_like(residual), where=denominator > 0.)
            if (not np.all(np.isfinite(residual))
                    or not np.all(np.isfinite(denominator))
                    or not np.all(np.isfinite(ratios))):
                self.mark_stale('non-finite residual or backward-error scale')
                raise NumericalError('Basis error calculation is non-finite; refactorise')
            if np.any((denominator == 0.) & (residual != 0.)):
                self.mark_stale('nonzero residual has zero backward-error scale')
                raise NumericalError('Basis error scale underflowed; refactorise')
            return residual, float(np.max(np.abs(residual), initial=0.0)), float(
                np.max(ratios, initial=0.0))

        residual, size, backward = measure(solution)
        self.state.max_rhs_residual = max(self.state.max_rhs_residual, size/scale)
        self.state.max_backward_error = max(self.state.max_backward_error, backward)
        # 先改善精度，再判断误差门；已通过两个门的常规回代不增加修正成本。
        for _ in range(self.refinement_steps):
            if max(size/scale, backward) <= self.residual_tol:
                break
            self.state.refinement_attempts += 1
            correction = self._solve_current(residual, transpose=transpose)
            candidate = solution+correction
            if not np.all(np.isfinite(candidate)):
                self.mark_stale('residual correction produced non-finite values')
                raise NumericalError('Basis refinement is non-finite; refactorise')
            new_residual, new_size, new_backward = measure(candidate)
            # 只保留真实降低原方程残差、且不恶化已通过的后向误差门的修正。
            if new_size >= size or new_backward > max(backward, self.residual_tol):
                break
            old_size = size
            solution, residual, size, backward = candidate, new_residual, new_size, new_backward
            self.state.refinement_improvements += 1
            if size > 0.5*old_size:
                break  # 改善不足一半，视为停滞；有界停止而非无限重试。
        relative = size/scale
        if transpose:
            self.state.btran_residual = relative
            self.state.btran_backward_error = backward
        else:
            self.state.ftran_residual = relative
            self.state.ftran_backward_error = backward
        selected = relative if self.residual_metric == 'rhs' else backward
        if selected > self.residual_tol:
            self.state.residual_gate_failures += 1
            self.mark_stale(f'{"BTRAN" if transpose else "FTRAN"} {self.residual_metric} error '
                            f'{selected:.3e} exceeds gate')
            raise NumericalError(f'Basis solve {self.residual_metric} error {selected:.3e} exceeds '
                                 f'the gate (rhs residual {relative:.3e}); refactorise')
        if self.residual_metric == 'componentwise' and relative > self.residual_tol:
            self.state.scaled_accepts += 1
        return solution

    # ---------- 乘积形式更新 ----------

    def update(self, position, entering_column, entering, leaving):
        """Apply one product-form pivot update.

        ``entering_column`` must be ``FTRAN(B_old, A[:, entering])``. The basis index
        at ``position`` becomes ``entering``. The update is only arithmetic: whether
        the pivot is numerically acceptable is decided by the caller's ratio test.
        """
        column = np.asarray(entering_column, dtype=float).reshape(-1)
        if column.size != len(self.basic):
            raise ValueError('Entering column has the wrong length')
        pivot_value = float(column[position])
        if pivot_value == 0.0 or not np.isfinite(pivot_value):
            raise NumericalError('Pivot element is zero or non-finite; basis update rejected')
        if position < 0 or position >= len(self.basic):
            raise ValueError('Pivot position is outside the basis')
        # 归一化乘积形式列 gamma = (eta - e_p)/eta_p，使单个更新成为 y -= gamma*y[p]。
        gamma = column/pivot_value
        gamma[position] -= 1.0/pivot_value
        leaving_column = self.basic[position]
        self.updates.append(BasisUpdate(pivot=int(position), gamma=gamma, eta=column.copy(),
                                        entering=int(entering), leaving=int(leaving_column)))
        self.basic[position] = int(entering)
        self.state.pivots_since_refactor += 1
        self.state.eta_count = len(self.updates)
        return self

    def needs_refactorisation(self):
        """True when the update chain or the pivot count has grown too long."""
        if self.state.stale:
            return True
        if len(self.updates) >= self.max_eta:
            return True
        return self.state.pivots_since_refactor >= self.refactor_interval

    def mark_stale(self, reason):
        """Record that the factors and the current basis no longer agree."""
        self.state.stale = True
        self.state.reason = str(reason)
        return self

    # ---------- 供上层使用的基解与对偶量 ----------

    def basic_solution(self, nonbasic_values):
        """``x_B = -B^{-1} N x_N`` for the given nonbasic values.

        ``nonbasic_values`` may be a mapping ``{index: value}`` or a **dense vector** whose basic
        entries are ignored (the driver passes a zeroed dense copy, which avoids rebuilding a
        seventy-thousand-entry Python dict on every iteration of a large instance).
        """
        if isinstance(nonbasic_values, np.ndarray):
            dense = np.asarray(nonbasic_values, dtype=float).reshape(-1).copy()
            if dense.size != self.matrix.shape[1]:
                raise ValueError('Dense nonbasic vector has the wrong length')
        else:
            dense = np.zeros(self.matrix.shape[1])
            for index, value in dict(nonbasic_values).items():
                dense[int(index)] = float(value)
        return -self.ftran(self.matrix @ dense)

    def reduced_costs(self, costs):
        """``r = c - A' y`` with ``y = BTRAN(c_B)``; returns the full reduced-cost vector."""
        cost = np.asarray(costs, dtype=float).reshape(-1)
        if cost.size != self.matrix.shape[1]:
            raise ValueError('Cost vector has the wrong length')
        dual = self.btran(cost[self.basic])
        return cost - self.matrix.T @ dual

    def dual_values(self, costs):
        """``y = BTRAN(c_B)`` for the current basis."""
        cost = np.asarray(costs, dtype=float).reshape(-1)
        return self.btran(cost[self.basic])


def extract_basis(matrix, *, tolerance=1e-9):
    """Find a full-rank set of ``m`` columns for ``matrix`` (threshold column pivoting).

    Returns ``(basic, dependent_rows)``. ``basic`` holds the indices of linearly
    independent columns; when fewer than ``m`` exist the matrix is rank deficient and
    ``dependent_rows`` lists the trailing pivot columns as evidence. A rank-deficient
    ``A`` is reported rather than silently accepted.
    """
    from scipy.linalg import qr
    dense = csc_matrix(matrix).toarray()
    rows, columns = dense.shape
    if columns < rows:
        raise ValueError('Matrix has fewer columns than rows; a basis cannot exist')
    _, factor, pivots = qr(dense, mode='economic', pivoting=True, check_finite=True)
    diagonal = np.abs(np.diag(factor))
    rank = int(np.sum(diagonal > tolerance))
    if rank < rows:
        return [int(j) for j in pivots[:rank]], [int(j) for j in pivots[rank:]]
    return [int(j) for j in pivots[:rows]], []


# ---------------------------------------------------------------------------
# 有界变量原始修正单纯形主循环
# ---------------------------------------------------------------------------


@dataclass
class SimplexResult:
    """Outcome of one revised-simplex solve; no status is inferred beyond evidence."""

    status: str
    values: np.ndarray | None = None
    objective: float | None = None
    reduced_costs: np.ndarray | None = None
    dual: np.ndarray | None = None
    basic: list = field(default_factory=list)
    iterations: int = 0
    pivots: int = 0
    refactorisations: int = 0
    message: str = ''
    history: list = field(default_factory=list)
    max_primal_violation: float = 0.0
    max_dual_violation: float = 0.0
    phase_one: dict = field(default_factory=dict)
    start_rejected: bool = False
    pricing: dict = field(default_factory=dict)
    refactor_retries: int = 0
    basis_diagnostics: dict = field(default_factory=dict)
    bound_flips: int = 0
    iteration_budget: dict = field(default_factory=dict)


def _iteration_budget(value):
    # 与公开参数一致：允许0用于只检查当前点，拒绝布尔、负数和非整数，先验拒绝而非中途异常。
    try:
        count = operator.index(value)
    except TypeError as error:
        raise ValueError('iteration_limit must be a nonnegative integer') from error
    if isinstance(value, (bool, np.bool_)) or count < 0:
        raise ValueError('iteration_limit must be a nonnegative integer')
    return count


def _budget_record(limit, used=0, *, phase_one=None):
    # 所有真实返回都有同一动作单位；聚合入口再补阶段拆分，元数据不替代状态/证明。
    record = dict(limit=limit, used=used, unit='executed pivot or bound flip')
    if phase_one is not None:
        record.update(phase_one=phase_one, phase_two=used-phase_one)
    return record


def _initial_values(bounds_lower, bounds_upper, basic, size):
    """Start every variable at a finite bound (0 when both bounds are infinite)."""
    values = np.zeros(size)
    for index in range(size):
        lower, upper = bounds_lower[index], bounds_upper[index]
        if math.isfinite(lower):
            values[index] = lower
        elif math.isfinite(upper):
            values[index] = upper
    return values

def _bound_side_after_placement(value, lower, upper):
    """True when a nonbasic variable sits on its UPPER bound after being placed.

    Derived from the placed value rather than from the ratio-test bookkeeping: a step can
    be truncated by the entering variable's own opposite bound, in which case the leaving
    variable ends on the *other* side from the one the ratio test recorded. Writing the
    flag from stale bookkeeping makes the next pricing step choose the wrong direction.
    """
    at_lower = math.isfinite(lower) and abs(value-lower) <= 1e-12
    at_upper = math.isfinite(upper) and abs(value-upper) <= 1e-12
    if at_lower and at_upper:
        return False          # 固定变量：按下界处理，方向由目标符号决定
    if at_upper:
        return True
    if at_lower:
        return False
    # 两侧界都不匹配（数值漂移）：报告而非猜测。
    raise NumericalError(f'Placed value {value} matches neither bound ({lower}, {upper})')


def _nearest_bound(value, lower, upper):
    """Return ``(snapped, at_upper)``: the bound a nonbasic variable must sit on."""
    distance_lower = abs(value-lower) if math.isfinite(lower) else math.inf
    distance_upper = abs(value-upper) if math.isfinite(upper) else math.inf
    if distance_upper <= distance_lower:
        return (upper, True) if math.isfinite(upper) else (lower, False)
    return (lower, False) if math.isfinite(lower) else (upper, True)


def _advance_basic_values(values, basic_indices, moving, step):
    # 前置：SparseBasis已验证基下标唯一，有序下标与独立FTRAN方向一一对应。
    # x_B <- x_B - alpha*d；先乘再减，保持原标量算术顺序，不合并乘加或更新非基值。
    # 重复下标的高级索引赋值不等价于逐次累加，不能将此操作用于未验证的任意索引。
    values[basic_indices] -= step*moving


def revised_simplex(matrix, costs, bounds_lower, bounds_upper, *, basic=None,
                    iteration_limit=20000, pricing_tolerance=1e-9,
                    feasibility_tolerance=1e-7, residual_tol=DEFAULT_RESIDUAL_TOL,
                    refactor_interval=DEFAULT_REFACTOR_INTERVAL, max_eta=DEFAULT_MAX_ETA,
                    objective_offset=0.0, maximize=False, bland_after=DEFAULT_BLAND_AFTER,
                    stall_after=DEFAULT_STALL_AFTER, initial=None,
                    refinement_steps=DEFAULT_REFINEMENT_STEPS, residual_metric='rhs'):
    """Primal revised simplex for ``min c'x`` s.t. ``Ax = 0``, ``l <= x <= u``.

    Computational form follows Huangfu & Hall §2.1 (arXiv:1503.01889v1): row bounds are
    carried inside ``A`` by identity columns, so the system is homogeneous. The method is
    the *bounded-variable* revised simplex: every nonbasic variable sits exactly on one of
    its bounds, and an entering variable may move only away from that bound.

    Robustness choices, each deliberate:

    * **Dantzig pricing with a permanent switch to Bland's rule.** Bland's rule guarantees
      finite termination under degeneracy but prices poorly, so it is entered only when it is
      needed: after ``bland_after`` pivots, or after ``stall_after`` consecutive pivots that
      made no progress (zero step, so no objective change). Once entered the switch is
      permanent, which is what preserves the termination guarantee. An earlier revision
      defaulted ``bland_after`` to 0, i.e. Bland from the very first pivot; the measured cost
      on the public Netlib development set was 3.3x the pivots on ``adlittle`` (558 vs 168,
      0.296 s vs 0.089 s) and 1.7x on ``afiro``, for identical certified optima
      (``docs/research/probe_pricing_rule_ablation.py``).
    * The ratio test accounts for the entering variable's opposite bound. When that bound
      binds first the variable is flipped to it and no basis change occurs.
    * A free variable (both bounds infinite) cannot be nonbasic, because its value would be
      unbounded; such a case is reported instead of being forced onto an artificial bound.
    * ``initial`` supplies values for the **nonbasic** variables, which is what selects the
      side of its bounds each one sits on. Basic values are always recomputed from
      ``x_B = -B^{-1} N x_N``, so ``A x = 0`` holds exactly before the first iteration.

    The returned status is evidence-based: ``OPTIMAL`` requires that the reduced costs
    certify optimality for the reported basis *and* that the primal point respects every
    bound. Nothing is promoted on the strength of iteration counts alone. A starting basis
    that is not primal feasible yields ``START_INFEASIBLE`` rather than a wrong answer, so the
    caller can decide between a cold start and Phase-I.
    """
    iteration_limit = _iteration_budget(iteration_limit)

    def stopped_before_updates(state, **fields):
        # 分解后拒绝起点也有真实工作；成功LU次数来自快照，不把零动作误当零计算。
        fields.setdefault('refactorisations', fields.get('basis_diagnostics', {}).get('refactorisations', 0))
        return SimplexResult(state, iteration_budget=_budget_record(iteration_limit), **fields)

    matrix = csc_matrix(matrix)
    rows, columns = matrix.shape
    cost = np.asarray(costs, dtype=float).reshape(-1)
    lower = np.asarray(bounds_lower, dtype=float).reshape(-1)
    upper = np.asarray(bounds_upper, dtype=float).reshape(-1)
    if cost.size != columns or lower.size != columns or upper.size != columns:
        raise ValueError('Cost and bound vectors must match the number of columns')
    if np.any(lower > upper):
        return stopped_before_updates('INFEASIBLE', message='A variable lower bound exceeds its upper bound')
    direction = -1.0 if maximize else 1.0
    internal_cost = direction*cost

    if basic is None:
        basic, dependent = extract_basis(matrix)
        if dependent or len(basic) != rows:
            return stopped_before_updates('INFEASIBLE',
                                 message='Constraint matrix is rank deficient; a full basis does not exist',
                                 phase_one=dict(dependent_columns=list(dependent)))
    basis = SparseBasis(matrix, basic, residual_tol=residual_tol,
                        refactor_interval=refactor_interval, max_eta=max_eta,
                        refinement_steps=refinement_steps, residual_metric=residual_metric)

    def diagnostics():
        # 包括起始失败的所有基操作路径都输出同一快照，避免只有收尾结果有数值证据。
        return dict(asdict(basis.state), residual_metric=residual_metric,
                    residual_tol=float(residual_tol), refinement_steps=int(refinement_steps))

    # 非基变量精确停在某一侧的界上；free 变量无法满足该不变量，直接报告。
    free_columns = [j for j in range(columns)
                    if not math.isfinite(lower[j]) and not math.isfinite(upper[j])]
    at_upper = np.zeros(columns, dtype=bool)
    values = np.zeros(columns)
    basis_positions = {j: k for k, j in enumerate(basis.basic)}
    # 调用方可给出完整可行初值（含非基变量）：非基变量"停在哪一侧界"必须由它的实际值
    # 判定；用"哪一侧界有限"去猜会让方向反向，产生零步长翻界循环。
    given = dict(initial) if initial is not None else {}
    for index, value in given.items():
        values[int(index)] = float(value)
    for j in range(columns):
        if j in basis_positions:
            continue
        if j in free_columns:
            return stopped_before_updates('NUMERICAL_ERROR',
                                 message=f'Variable {j} is free and cannot be a nonbasic variable '
                                         'in this bounded-variable form',
                                 phase_one=dict(free_columns=free_columns),
                                 basis_diagnostics=diagnostics())
        if j in given:
            at_lower = math.isfinite(lower[j]) and abs(values[j]-lower[j]) <= 1e-12
            at_upper[j] = bool(math.isfinite(upper[j]) and not at_lower
                               and abs(values[j]-upper[j]) <= 1e-12)
            if not at_lower and not at_upper[j]:
                return stopped_before_updates(
                    'NUMERICAL_ERROR',
                    message=f'Nonbasic variable {j} starts strictly inside its bounds '
                            f'({lower[j]}, {upper[j]}); a bounded-variable basis needs a bound',
                    basis_diagnostics=diagnostics())
        elif math.isfinite(upper[j]) and upper[j] != lower[j]:
            values[j], at_upper[j] = upper[j], True
        else:
            values[j], at_upper[j] = lower[j], False
    refactor_retries = 0

    def guarded(function):
        """Run a validated basis operation, refactorising once if the residual gate fires.

        ``SparseBasis`` validates every solve against the CURRENT basis and raises
        ``NumericalError`` with "refactorise" in the message. That exception used to escape the
        driver entirely, so a solve aborted instead of doing what the design asks for. Measured
        on MIPLIB's ``neos-860300`` LP relaxation: ``Basis solve residual 5.681e-08 exceeds the
        gate; refactorise`` ended the whole solve as SOLVER_ERROR. Here the stale factorisation
        is dropped, rebuilt, and the operation retried once; a second failure is reported rather
        than retried forever, so no wrong answer can be produced by looping.
        """
        nonlocal refactor_retries
        try:
            return function()
        except NumericalError:
            refactor_retries += 1
            basis.mark_stale('validated basis solve failed; rebuilding the factorisation')
            basis.refactorise()
            return function()

    # 基变量的值由**非基变量的取值**唯一确定：x_B = −B^{-1} N x_N。因此这里一律重算，
    # 不沿用调用方给出的基变量分量（`initial` 只用于指定非基变量停在哪一侧界）。
    # 这样 A x = 0 在迭代开始前就是精确成立的。
    try:
        values[basis.basic] = guarded(lambda: basis.basic_solution(
            {j: values[j] for j in range(columns) if j not in basis_positions}))
    except NumericalError as error:
        return stopped_before_updates('NUMERICAL_ERROR', basic=list(basis.basic),
                             message=f'the starting basis solve failed even after refactorising: '
                                     f'{error}', refactor_retries=refactor_retries,
                             refactorisations=basis.state.refactorisations,
                             basis_diagnostics=diagnostics())

    # 起始可行性必须**先验证再迭代**：调用方给的基可能不可行，而从不可行点出发的比例
    # 检验没有意义（越界的基变量会被当成可用步长）。如实报告 START_INFEASIBLE，由调用方
    # 决定冷启动还是 Phase-I；这里不静默修补，也不谎称原问题不可行。
    start_violation = _primal_violation(values, lower, upper, feasibility_tolerance)
    start_equation = float(np.max(np.abs(matrix @ values), initial=0.0))
    start_scale = 1.0+float(np.max(np.abs(values), initial=0.0))
    if start_violation > feasibility_tolerance or start_equation > feasibility_tolerance*start_scale:
        return stopped_before_updates(
            'START_INFEASIBLE', values=values, basic=list(basis.basic), iterations=0, pivots=0,
            message=(f'the starting basis is not primal feasible (bound violation '
                     f'{start_violation:.3e}, |A x| {start_equation:.3e}); a cold start or a '
                     f'Phase-I step is required'),
            max_primal_violation=max(start_violation, start_equation), start_rejected=True,
            basis_diagnostics=diagnostics(), refactor_retries=refactor_retries,
            refactorisations=basis.state.refactorisations,
            phase_one=dict(used=False, reason='starting basis rejected before iterating',
                           start_violation=start_violation, start_equation_residual=start_equation))

    history = []
    pivots = 0
    bound_flips = 0
    completed = 0
    stalled = 0
    status = 'ITERATION_LIMIT'
    message = 'Iteration limit reached before optimality was proved'
    for iteration in range(iteration_limit+1):
        completed = iteration
        if basis.needs_refactorisation():
            basis.refactorise()
        # 基变量值必须满足 A x = 0，且在**每次基/非基状态变化之后**重算：非基变量被置于
        # 某一侧界（翻界、换出落界只记状态），沿用旧值会让方程残差停在错误位置，实测
        # 表现为步长恒为 0 的零进展循环。
        # 本轮基更新前不变：只转换一次有序下标，复用于掩码、取值和基值维护。
        basic_indices = np.asarray(basis.basic, dtype=np.intp)
        in_basis = np.zeros(columns, dtype=bool)
        in_basis[basic_indices] = True
        # 基变量值由非基值唯一确定。原先这里每轮构造一个 `{j: values[j] for j in
        # range(columns) if j not in basis_positions}` 的 Python 字典（72747 列的实例每轮要建
        # 一个七万项的字典），换成"整向量清零基变量位置"的 NumPy 拷贝；语义完全相同
        # （基变量位置在 basic_solution 里本来就由方程重算，传入值不被使用）。
        nonbasic_values = values.copy()
        nonbasic_values[basic_indices] = 0.0
        try:
            values[basic_indices] = guarded(lambda: basis.basic_solution(nonbasic_values))
        except NumericalError as error:
            status = 'NUMERICAL_ERROR'
            message = f'refactorisation did not restore the basis: {error}'
            break
        # 每轮由**实际值**重算所有非基变量的所在侧，消除标志与值的漂移。标志若与实际
        # 值不一致，定价会朝不可行方向选变量（实测在已最优点上反复选中不该选的变量）。
        # 这里同样向量化：逐列版本是 `for j in range(columns)` 外加一次 `j in basis_positions`
        # 字典查询，72747 列时每轮就是七万次字典查询——它正是定价循环修好之后暴露出来的
        # 下一个热点（faulthandler 栈停在 sparse_simplex.py 的这一行）。
        # 语义保持逐列版本的三分支：非基且贴在上界 -> True；否则非基且贴在下界 -> False；
        # 基变量与两侧都不贴的变量保持原标志不变。
        snapped_upper = np.isfinite(upper) & (np.abs(values-upper) <= 1e-12)
        snapped_lower = np.isfinite(lower) & (np.abs(values-lower) <= 1e-12)
        at_upper = np.where(~in_basis & snapped_upper, True,
                            np.where(~in_basis & snapped_lower, False, at_upper))
        try:
            # 对偶量只算一次：reduced_costs 内部本来也要做一次 BTRAN，与 dual_values 完全重复。
            # 现在用 r = c − A'y 的定义式直接算简约成本（同一次已验证的 BTRAN + 一次稀疏乘法），
            # 并把"基变量上简约成本必须为零"这条恒等式**显式检查**，比原来重复求解更能发现数值问题。
            dual = guarded(lambda: basis.dual_values(internal_cost))
        except NumericalError as error:
            status = 'NUMERICAL_ERROR'
            message = f'refactorisation did not restore the basis: {error}'
            break
        reduced = internal_cost - matrix.T @ dual
        basic_reduced = float(np.max(np.abs(reduced[basic_indices]), initial=0.0))
        cost_scale = 1.0+float(np.max(np.abs(internal_cost), initial=0.0))
        if basic_reduced > 1e-6*cost_scale:
            # 基变量上的简约成本本应为零；超门说明当前分解/更新链不可信，标记陈旧让下一轮重建。
            basis.mark_stale(f'basic reduced costs are not zero (max {basic_reduced:.3e})')

        # 定价。Dantzig 取最改进者；枢轴数超门或连续多轮无进展后**永久**改用 Bland 最小下标
        # 规则，后者在退化下保证有限终止，因此循环不可能被静默接受。
        # 固定变量（上下界相等）作为非基变量时**不能移动**，其简约成本的符号不构成最优性
        # 条件，因此一律不参与定价；若把它当成可动变量，比例检验会算出零步长并在两个界
        # 之间反复翻转（实测在把等式行写成固定逻辑变量的模型上会空转到迭代上限）。
        #
        # 这里必须是**向量化**定价：原实现是 `for j in range(columns)` 的纯 Python 扫描，
        # 每轮迭代都要走完全部列。实测 `tbfp-network`（72747 列 / 2436 行）在 Phase-I 就被
        # 它主导——`faulthandler` 抓到的栈正停在这个循环上（见
        # docs/research/probe_lp_relaxation_cost.py）。语义必须逐项等价：
        #   * Dantzig 的 `best` 是"改进量绝对值"，从 pricing_tolerance 起严格增大，
        #     因此并列时保留**最先遇到**的下标 -> np.argmax 的首个最大值一致；
        #   * Bland 取最小下标 -> 布尔掩码的 np.argmax 恰好返回第一个 True。
        fixed = lower == upper
        use_bland = (pivots >= bland_after) or (stalled >= stall_after)
        # 变量沿自身可行方向移动时"改进量"：停在上界则下降才有改进，否则上升才有改进。
        improving = np.where(at_upper, reduced, -reduced)
        eligible = (~in_basis) & (~fixed) & (improving > pricing_tolerance)
        entering = -1
        if eligible.any():
            if use_bland:
                entering = int(np.argmax(eligible))          # 第一个 True = 最小下标
            else:
                entering = int(np.argmax(np.where(eligible, improving, -np.inf)))
        if entering < 0:
            violation = _primal_violation(values, lower, upper, feasibility_tolerance)
            if violation <= feasibility_tolerance:
                status, message = 'OPTIMAL', 'Reduced costs proved optimality for the current basis'
            else:
                status = 'NUMERICAL_ERROR'
                message = f'No entering column but primal violation {violation:.3e} remains'
            break

        # 末端可验证当前基是否最优，但不可再执行下一次移动；否则额度N实际执行N+1次。
        if completed >= iteration_limit:
            break

        pivot_column = basis.ftran(matrix[:, entering].toarray().ravel(), validate=False)
        # 沿"变量上升"为正方向；非基变量只能离开它当前所在的界。
        sign = -1.0 if at_upper[entering] else 1.0
        moving = sign*pivot_column
        # 比例检验：先看基本变量能走多远。记录限制变量触到的是哪一侧界——换出后必须
        # 比例检验：先看基本变量能走多远。记录限制变量触到的是哪一侧界——换出后必须
        # 精确落在**实际触到的那个界**上；用"最近界"会把变量放到它并未触到的界上。
        # 关键符号：moving = sign * B^{-1}a 是入基变量沿自身可行方向上升时，基本变量
        # 需要**减去**的变化量（x_B(α) = x_B − moving·α）。故只有 `d = −moving[i] > 0`
        # 的基本变量会下降、才可能先触下界；判据若直接用 moving[i] 会把方向判反，
        # 表现为步长恒为 0。教科书参考实现 `zyo/standard_simplex.py` 用同一符号约定
        # 通过了手算题（两次枢轴、目标 −5）。
        step = math.inf
        limiting = -1
        limiting_upper = False
        for position, j in enumerate(basis.basic):
            d = -moving[position]
            if d > pricing_tolerance and math.isfinite(upper[j]):
                room = (upper[j]-values[j])/d
                # 并列时取变量下标更小者，使退化路径确定（Bland 式 tie-break）。
                if room < step-1e-12 or (abs(room-step) <= 1e-12 and limiting >= 0
                                         and j < basis.basic[limiting]):
                    step, limiting, limiting_upper = room, position, True
            elif d < -pricing_tolerance and math.isfinite(lower[j]):
                room = (lower[j]-values[j])/d
                if room < step-1e-12 or (abs(room-step) <= 1e-12 and limiting >= 0
                                         and j < basis.basic[limiting]):
                    step, limiting, limiting_upper = room, position, False
        # 再看入基变量自身的另一侧界：先触到它即为翻界，不换基。
        flip_room = math.inf
        if sign > 0 and math.isfinite(upper[entering]):
            flip_room = upper[entering]-values[entering]
        elif sign < 0 and math.isfinite(lower[entering]):
            flip_room = values[entering]-lower[entering]
        if not math.isfinite(step) and not math.isfinite(flip_room):
            status = 'UNBOUNDED'
            message = 'No bound limits the step; the objective improves without limit'
            break
        step = max(0.0, min(step, flip_room))

        values[entering] += sign*step
        # 基本变量按 x_B(α) = x_B(0) − moving·α **减去**移动量：由 A x = 0 得
        # x_B = −B^{-1} N x_N，故入基变量上升时基变量必须反向补偿。写成 `+=` 会让本轮
        # 的 values 立刻违反 A x = 0（实测 |A x| = 8），只是下一轮开头重算基变量值把它
        # 掩盖掉；比例检验读到被污染的值仍会算错 room。
        _advance_basic_values(values, basic_indices, moving, step)
        history.append(dict(iteration=iteration, entering=int(entering), step=float(step),
                            limiting_position=int(limiting), reduced=float(reduced[entering]),
                            objective=float(internal_cost @ values),
                            pricing_rule='bland' if use_bland else 'dantzig'))
        # 迭代是实际已执行状态更新，零步长与翻界同样计入，不能只数下一轮检查或换基。
        completed += 1
        # 停滞计数：退化枢轴（步长为零）不改变目标值，长串零步长正是循环的前兆。
        # 它与枢轴数门一起触发"永久改用 Bland"，因此终止性保证不受影响。
        stalled = stalled+1 if step <= pricing_tolerance else 0
        if flip_room <= step+pricing_tolerance and flip_room < math.inf:
            # 翻界：入基变量到达另一侧界，基不变。
            snapped, on_upper = _nearest_bound(values[entering], lower[entering], upper[entering])
            values[entering] = snapped
            at_upper[entering] = on_upper
            bound_flips += 1
            history[-1].update(kind='bound_flip', leaving=None)
            continue

        leaving = basis.basic[limiting]
        history[-1].update(kind='pivot', leaving=int(leaving))
        basis.update(position=limiting, entering_column=pivot_column,
                     entering=entering, leaving=leaving)
        # 换出变量成为非基变量：精确落在比例检验实际触到的那个界上，并由**落位后的值**
        # 反推它停在哪一侧。不用 limiting_upper 直接设标志：步长可能被翻界截断，
        # 此时实际触到的是另一侧界，标志写错会让下一轮定价选错方向（实测选到已最优的
        # 变量并陷入零步长循环）。
        values[leaving] = upper[leaving] if limiting_upper else lower[leaving]
        at_upper[leaving] = _bound_side_after_placement(values[leaving], lower[leaving], upper[leaving])
        at_upper[entering] = False
        pivots += 1
        if not math.isfinite(values[entering]) or abs(values[entering]) > 1e15:
            status = 'NUMERICAL_ERROR'
            message = 'Entering variable diverged after the ratio test'
            break
    else:
        status, message = 'ITERATION_LIMIT', 'Iteration limit reached'

    final_cost = direction*internal_cost
    if status == 'NUMERICAL_ERROR':
        # 循环内已判定该基即使重分解也无法恢复：收尾不再重试一次，否则 refactor_retries
        # 会把同一次失败统计成两次，证据就不再可信。
        reduced = np.zeros(columns)
        dual = np.zeros(rows)
    else:
        try:
            # 收尾重算也必须走同一个重试路径：否则残差门在收尾阶段报陈旧时异常仍会逃出整个
            # 求解（持久失败用例正是抓到这里：循环内已正确转成 NUMERICAL_ERROR，收尾处却直接
            # 抛出 NumericalError）。
            reduced = guarded(lambda: basis.reduced_costs(final_cost))
            dual = guarded(lambda: basis.dual_values(final_cost))
        except NumericalError as error:
            status = 'NUMERICAL_ERROR'
            message = f'refactorisation did not restore the basis: {error}'
            reduced = np.zeros(columns)
            dual = np.zeros(rows)
    primal_violation = _primal_violation(values, lower, upper, feasibility_tolerance)
    dual_violation = _dual_violation(reduced, values, lower, upper, basis.basic, feasibility_tolerance)
    equation_residual = float(np.max(np.abs(matrix @ values), initial=0.0))
    if status == 'OPTIMAL' and max(primal_violation, equation_residual) > feasibility_tolerance:
        status = 'NUMERICAL_ERROR'
        message = (f'Optimality claimed but primal violation {primal_violation:.3e} / '
                   f'equation residual {equation_residual:.3e} exceeds tolerance')
    return SimplexResult(status=status, values=values,
                         # 上报**用户方向**的目标值：最大化时内部按最小化求解，内部目标等于
                         # 用户目标的相反数。早期版本直接返回 internal_cost @ values，
                         # 实测 max 3x+y 返回 -10 而真值是 10。
                         objective=float(cost @ values)+objective_offset,
                         reduced_costs=reduced, dual=dual, basic=list(basis.basic),
                         iterations=completed, pivots=pivots,
                         refactorisations=basis.state.refactorisations, message=message,
                         history=history, max_primal_violation=max(primal_violation, equation_residual),
                         max_dual_violation=dual_violation,
                         pricing=dict(rule='bland' if (pivots >= bland_after
                                                       or stalled >= stall_after) else 'dantzig',
                                      bland_after=int(bland_after), stall_after=int(stall_after),
                                      pivots=int(pivots), stalled_final=int(stalled),
                                      switched=bool(pivots >= bland_after
                                                    or stalled >= stall_after)),
                         refactor_retries=int(refactor_retries),
                         basis_diagnostics=diagnostics(), bound_flips=bound_flips,
                         iteration_budget=_budget_record(iteration_limit, completed))


def _primal_violation(values, lower, upper, tolerance):
    violation = 0.0
    for index in range(values.size):
        if math.isfinite(lower[index]):
            violation = max(violation, lower[index]-values[index])
        if math.isfinite(upper[index]):
            violation = max(violation, values[index]-upper[index])
    return float(max(0.0, violation))


def _dual_violation(reduced, values, lower, upper, basic, tolerance):
    """Optimality violation of the reduced costs against the CURRENT variable states."""
    basic_set = set(basic)
    violation = 0.0
    for index in range(values.size):
        if index in basic_set:
            continue
        # 固定变量（上下界相等）不能移动，其简约成本不受符号约束，不构成违反。
        if lower[index] == upper[index]:
            continue
        r = reduced[index]
        at_lower = math.isfinite(lower[index]) and values[index] <= lower[index]+tolerance
        at_upper = math.isfinite(upper[index]) and values[index] >= upper[index]-tolerance
        if at_lower and r < 0:
            violation = max(violation, -r)
        elif at_upper and r > 0:
            violation = max(violation, r)
        elif not at_lower and not at_upper:
            violation = max(violation, abs(r))
    return float(violation)

def solve_lp(matrix, costs, bounds_lower, bounds_upper, *, maximize=False,
             objective_offset=0.0, feasibility_tolerance=1e-7, basic=None,
             initial=None, **kwargs):
    """Solve ``min c'x`` s.t. ``Ax = 0``, ``l <= x <= u`` using the ZYO revised simplex.

    Three start strategies are tried in a fixed, reported order, and each rejection is
    recorded rather than hidden:

    1. **caller-supplied basis** (``basic``/``initial``): the warm-start entry point. It is
       used only if ``revised_simplex`` accepts it as primal feasible. A rejection is
       recorded in ``phase_one['warm_start_rejected']`` and the solve continues below — the
       caller's hint is a hint, not an assertion, so no wrong answer can be returned.
    2. **logical (unit) columns** of ``A`` when such columns exist: the row-bound identity
       submatrix is already a feasible basis whenever the associated bounds admit zero.
    3. **Phase-I**: an artificial variable is attached to every row and their sum is
       minimised. Only a Phase-I optimum with zero artificial activity is accepted as a
       starting point; a positive sum means the rows are mutually infeasible and is reported
       as such, never papered over.

    ``basic``/``initial`` are keyword-only and are consumed here, so they are never forwarded
    twice into :func:`revised_simplex` (a duplicate-keyword ``TypeError`` in an earlier
    revision made every warm start impossible).
    """
    budget = _iteration_budget(kwargs.get('iteration_limit', 20000))
    rejected_refactors = 0

    def finish(result, *, phase_one_steps=0):
        # 被拒热启动/逻辑起点消耗LU而未移动；汇总成功分解，但不扣动作额度。
        result.refactorisations += rejected_refactors
        result.iteration_budget = _budget_record(budget, result.iterations, phase_one=phase_one_steps)
        result.phase_one['rejected_start_refactorisations'] = rejected_refactors
        return result

    matrix = csc_matrix(matrix)
    rows, columns = matrix.shape
    lower = np.asarray(bounds_lower, dtype=float).reshape(-1)
    upper = np.asarray(bounds_upper, dtype=float).reshape(-1)
    cost = np.asarray(costs, dtype=float).reshape(-1)
    forward = dict(kwargs)
    if initial is not None:
        forward['initial'] = initial

    rejection = {}
    if basic is not None:
        trial = revised_simplex(matrix, cost, lower, upper, basic=basic,
                                maximize=maximize, objective_offset=objective_offset,
                                feasibility_tolerance=feasibility_tolerance, **forward)
        if not trial.start_rejected:
            trial.phase_one = dict(used=False, warm_start=True, accepted=True,
                                   reason='caller-supplied basis was primal feasible')
            return finish(trial)
        rejected_refactors += trial.refactorisations
        rejection = dict(rejected=True, message=trial.message,
                         start_violation=trial.phase_one.get('start_violation'),
                         start_equation_residual=trial.phase_one.get('start_equation_residual'))

    # 只有当单位列真的是 A 的列时才可直接作为初始基；否则走 Phase-I。
    # 探测必须**全程稀疏**：原实现 `matrix.toarray()` 会把整个矩阵稠密化，随后对每行扫描全部列，
    # 既花 O(rows*columns) 时间又花同样量级的内存（2436x75183 的实例就是 1.5 GB 的临时数组，
    # 266227 行的实例根本不可能分配）。这里改为按 CSC 的列指针直接判定"该列是否恰有一个
    # 非零元且等于 +1"，语义与原来逐列检查完全一致。
    unit_columns = []
    csc = matrix.tocsc()
    entries_per_column = np.diff(csc.indptr)
    single = np.flatnonzero(entries_per_column == 1)
    if single.size:
        rows_of = csc.indices[csc.indptr[single]]
        values_of = csc.data[csc.indptr[single]]
        positive = single[np.abs(values_of-1.0) <= 1e-12]
        rows_positive = rows_of[np.abs(values_of-1.0) <= 1e-12]
        chosen = {}
        for column, row in zip(positive.tolist(), rows_positive.tolist()):
            chosen.setdefault(int(row), int(column))
        if len(chosen) == rows:
            unit_columns = [chosen[row] for row in range(rows)]
    if len(unit_columns) == rows and len(set(unit_columns)) == rows:
        trial = revised_simplex(matrix, cost, lower, upper, basic=unit_columns,
                                maximize=maximize, objective_offset=objective_offset,
                                feasibility_tolerance=feasibility_tolerance, **forward)
        if not trial.start_rejected:
            trial.phase_one = dict(used=False, reason='logical basis was primal feasible')
            if rejection:
                trial.phase_one['warm_start_rejected'] = rejection
            return finish(trial)
        rejected_refactors += trial.refactorisations

    phase, phase_record = _phase_one(matrix, lower, upper, rows, columns,
                                     feasibility_tolerance=feasibility_tolerance, **kwargs)
    if rejection:
        phase_record['warm_start_rejected'] = rejection
    if phase is None:
        # 子阶段预算停止不是数值故障；人工目标不能上报成原LP目标。
        state = 'ITERATION_LIMIT' if phase_record.get('status') == 'ITERATION_LIMIT' else 'NUMERICAL_ERROR'
        return finish(SimplexResult(state, message=phase_record.get('reason', 'Phase-I could not establish a basis'),
                             phase_one=phase_record, iterations=phase_record.get('iterations', 0),
                             pivots=phase_record.get('pivots', 0),
                             bound_flips=phase_record.get('bound_flips', 0),
                             refactorisations=phase_record.get('refactorisations', 0)),
                      phase_one_steps=phase_record.get('iterations', 0))
    if phase != 'FEASIBLE':
        return finish(SimplexResult('INFEASIBLE', message='Phase-I proved the rows are infeasible',
                             phase_one=phase_record, iterations=phase_record.get('iterations', 0),
                             pivots=phase_record.get('pivots', 0),
                             bound_flips=phase_record.get('bound_flips', 0),
                             refactorisations=phase_record.get('refactorisations', 0)),
                      phase_one_steps=phase_record.get('iterations', 0))
    # 第二阶段优先使用 Phase-I 结束时那个**已验证可行**的点来摆放非基变量；只有拿不到
    # 它时才退回调用方给的 initial。两者都只是"非基变量停在哪一侧界"的选择。
    phase_two = dict(kwargs)
    phase_two['iteration_limit'] = budget-phase_record['iterations']
    values = phase_record.get('values')
    if values is not None:
        phase_two['initial'] = {j: float(v) for j, v in enumerate(values)
                                if j not in set(phase_record['basis'])}
    elif initial is not None:
        phase_two['initial'] = initial
    result = revised_simplex(matrix, cost, lower, upper, basic=phase_record['basis'],
                             maximize=maximize, objective_offset=objective_offset,
                             feasibility_tolerance=feasibility_tolerance, **phase_two)
    result.phase_one = phase_record
    result.iterations += phase_record['iterations']
    result.pivots += phase_record['pivots']
    result.bound_flips += phase_record['bound_flips']
    result.refactorisations += phase_record['refactorisations']
    return finish(result, phase_one_steps=phase_record['iterations'])


def _crash_start(matrix, lower, upper, placement):
    """Build the "logical column as the row's basic variable" Phase-I start.

    Row ``i`` reads ``(A x)_i - s_i = 0``. A unit column ``c`` (one nonzero of magnitude 1) in
    row ``i`` can take the value ``-others/sign`` that makes its own row hold exactly; when that
    value lies inside ``c``'s own bounds, ``c`` becomes the BASIC variable of row ``i`` and the row
    needs no artificial variable. A unit column touches exactly one row, so fixing its value cannot
    disturb another row — that is what makes the construction well defined.

    A row can carry **several** unit columns (in the homogeneous encoding the logical ``-e_i`` plus,
    for many MIPLIB instances, a structural ``+e_i``), and each has to be tried against its own
    bounds: an earlier prototype kept only the first one, which silently discarded the logical
    whenever the structural column had a smaller index and produced a false "zero benefit" reading
    on ``neos-911970``.

    Returns ``None`` when no row can use a unit column, else a dict with the start values, the basis
    (a list of ``column``/``artificial`` indices for every row) and the resulting artificial count.
    """
    body = csc_matrix(matrix).tocsc()
    rows, columns = body.shape
    counts = np.diff(body.indptr)
    single = np.flatnonzero(counts == 1)
    candidates = {}
    for column, row, value in zip(single.tolist(),
                                  body.indices[body.indptr[single]].tolist(),
                                  body.data[body.indptr[single]].tolist()):
        if abs(abs(value)-1.0) <= 1e-12:
            candidates.setdefault(int(row), []).append((int(column), float(np.sign(value))))
    if not candidates:
        return None
    start = placement.copy()
    chosen = {}
    used_columns = set()
    for row in sorted(candidates):
        coefficients = np.asarray(body[row, :].todense()).ravel()
        for column, sign in candidates[row]:
            if column in used_columns:
                continue
            others = float(coefficients @ start)-coefficients[column]*start[column]
            value = -others/sign
            if lower[column]-1e-12 <= value <= upper[column]+1e-12:
                chosen[row] = (column, float(value))
                used_columns.add(column)
                start[column] = float(value)
                break
    if not chosen:
        return None
    activity = np.asarray(body @ start, dtype=float).reshape(-1)
    artificial_rows = [row for row in range(rows) if row not in chosen]
    basis_offset = []
    for row in range(rows):
        basis_offset.append(chosen[row][0] if row in chosen
                            else columns+artificial_rows.index(row))
    total = float(np.sum(np.abs(activity[artificial_rows]))) if artificial_rows else 0.0
    return dict(initial={j: float(start[j]) for j in range(columns)}, basis_offset=basis_offset,
                artificial_rows=artificial_rows, activity=total, logical_rows=sorted(chosen))


def _phase_one(matrix, lower, upper, rows, columns, *, feasibility_tolerance=1e-7,
               time_limit=None, phase_one_crash=None, **kwargs):
    """Minimise the sum of artificial variables to obtain a feasible basis.

    The auxiliary start is feasible **by construction** for arbitrary bounds:

    1. every structural variable is placed on a finite bound, giving a residual
       ``r = A x0`` (a structural variable with neither bound finite cannot be placed and is
       reported instead of guessed);
    2. row ``i`` receives exactly one artificial column ``sigma_i * e_i`` with bounds
       ``[0, inf)``, so its homogeneous row reads ``(A x)_i + sigma_i t_i = 0``, i.e.
       ``t_i = -sigma_i r_i``;
    3. choosing ``sigma_i = -sign(r_i)`` therefore gives ``t_i = |r_i| >= 0``.

    An earlier revision used ``+e_i`` for every row, which forces ``t_i = -r_i`` and yields a
    NEGATIVE artificial whenever ``r_i > 0``. Measured on ``x - s = 0`` with ``s <= -1`` and
    ``x >= 0`` (genuinely infeasible) that start had ``t = -1``; the start check rejected it
    and the problem was reported as NUMERICAL_ERROR instead of INFEASIBLE.

    Each artificial's finite LOWER bound 0 is what limits the Phase-I ratio test, so Phase-I
    cannot run away along a direction that only makes the artificials negative.

    The starting basis has two variants (see :func:`_crash_start`): "plain" gives every row an
    artificial, and the "logical-first crash" lets a unit column satisfy its own row when it can,
    so those rows need none. ``phase_one_crash`` selects between them: ``None`` (default) takes the
    crash only when it strictly reduces the initial artificial activity, ``True``/``False`` force
    the choice. The auto rule is measured, not assumed — see :func:`_crash_start`.

    Returns ``('FEASIBLE', record)``, ``('INFEASIBLE', record)`` or ``(None, record)``.
    The record always carries the initial basis and the artificial objective so the
    outcome can be audited rather than trusted.
    """
    # ---- 结构变量的落位：每个非基变量必须精确停在某一侧界上 ----
    placement = np.zeros(columns)
    free_columns = []
    for j in range(columns):
        if math.isfinite(lower[j]):
            placement[j] = lower[j]
        elif math.isfinite(upper[j]):
            placement[j] = upper[j]
        else:
            free_columns.append(j)
    if free_columns:
        return None, dict(used=True, basis=None, free_columns=free_columns,
                          reason='structural variables with no finite bound cannot be placed '
                                 'on a bound, so a bounded-variable Phase-I start is unavailable')
    residual = np.asarray(matrix @ placement, dtype=float).reshape(-1)
    # σ_i = -sign(r_i)：使人造变量初值 |r_i| 非负，见上方推导。
    sigma = np.where(residual > 0.0, -1.0, 1.0)
    plain_activity = float(np.sum(np.abs(residual)))
    # ---- 启动方案：逻辑列优先当基（"crash"）与逐行人工变量（plain）择一 ----
    #
    # 行方程是 (A x)_i - s_i = 0。若把某行的单位列取到恰好满足本行的值、且该值落在它自己的
    # 界内，它就能直接当该行的**基本**变量，该行不需要人工变量；基仍是 ±I（每行一个 ±1 单位
    # 列）故非奇异，起点也自动可行（基本列取界内值、其余行人工取 |r_i| ≥ 0、非基列停在界上）。
    #
    # **触发规则来自实测而不是直觉**：只在 crash **严格降低**初始人工不可行量时采用它。
    # 六个实例的对照（docs/research/probe_crash_vs_current.py）显示收益并不一边倒——
    # sct2 由"数值失败、目标 451479"变为目标 68.5，fhnw-binpack4-4 枢轴 948→637，roll3000
    # 目标 12333→10627；但 neos-3381206-awhea 反而从 2035 涨到 2719 枢轴，而它恰好也是
    # 初始不可行量**没有下降**的那一题（1313→1313）；air05 完全无变化（426→426）。
    # 21 个可构造实例的覆盖面测量（probe_crash_trigger_coverage.py）：18 个触发，
    # 触发案例的不可行量降幅中位数 85.3%，其中 neos-2987310-joes 从 1.566e9 降到 0。
    crash = _crash_start(matrix, lower, upper, placement)
    if phase_one_crash is True:
        use_crash = crash is not None
    elif phase_one_crash is False:
        use_crash = False
    else:
        use_crash = crash is not None and crash['activity'] < plain_activity-1e-12
    if crash is None:
        chosen = dict(initial={j: float(placement[j]) for j in range(columns)},
                      basis_offset=None, artificial_rows=list(range(rows)),
                      activity=plain_activity,
                      strategy='plain (no unit column can satisfy its own row inside its bounds)')
    elif use_crash:
        chosen = dict(initial=crash['initial'], basis_offset=crash['basis_offset'],
                      artificial_rows=crash['artificial_rows'], activity=crash['activity'],
                      strategy='logical-first crash (strictly lower initial artificial activity)'
                      if phase_one_crash is None else 'logical-first crash (forced)')
    else:
        chosen = dict(initial={j: float(placement[j]) for j in range(columns)},
                      basis_offset=None, artificial_rows=list(range(rows)),
                      activity=plain_activity,
                      strategy='plain (crash did not reduce the initial artificial activity)'
                      if phase_one_crash is None else 'plain (forced)')
    if chosen['basis_offset'] is None:
        # 人工块与整行拼接都必须稀疏构造：`np.diag(sigma)` 对 17245 行是 2.4 GB 的稠密矩阵，
        # `matrix.toarray()` 对 17245x147912 的实例是 20 GB——两者都只在这里出现，因此大实例
        # 此前根本走不到 Phase-I。稀疏拼接后的方程与符号完全相同。
        artificial_block = diags(sigma, format='csc')
        wide = hstack([csc_matrix(matrix), artificial_block], format='csc')
        wide_lower = np.concatenate([lower, np.zeros(rows)])
        wide_upper = np.concatenate([upper, np.full(rows, math.inf)])
        wide_cost = np.concatenate([np.zeros(columns), np.ones(rows)])
        artificial = list(range(columns, columns+rows))
        start_basis = list(artificial)
    else:
        rows_with_artificial = chosen['artificial_rows']
        count = len(rows_with_artificial)
        artificial_block = csc_matrix(
            (sigma[rows_with_artificial], (rows_with_artificial, range(count))),
            shape=(rows, count))
        wide = hstack([csc_matrix(matrix), artificial_block], format='csc')
        wide_lower = np.concatenate([lower, np.zeros(count)])
        wide_upper = np.concatenate([upper, np.full(count, math.inf)])
        wide_cost = np.concatenate([np.zeros(columns), np.ones(count)])
        artificial = [columns+k for k in range(count)]
        start_basis = list(chosen['basis_offset'])
    placement_initial = chosen['initial']
    status = revised_simplex(wide, wide_cost, wide_lower, wide_upper, basic=start_basis,
                             initial=placement_initial,
                             feasibility_tolerance=feasibility_tolerance, **kwargs)
    record = dict(used=True, status=status.status, artificial_objective=status.objective,
                  artificial_columns=artificial, artificial_signs=[float(s) for s in sigma],
                  start_strategy=chosen['strategy'],
                  start_artificial_rows=len(chosen['artificial_rows']),
                  start_activity=float(chosen['activity']),
                  plain_start_activity=plain_activity,
                  start_residual=[float(v) for v in residual],
                  iterations=status.iterations, pivots=status.pivots, bound_flips=status.bound_flips,
                  refactorisations=status.refactorisations, basis=None, message=status.message,
                  cleanup_refactorisations=0,
                  basis_diagnostics=status.basis_diagnostics,
                  refactor_retries=status.refactor_retries)
    if status.status != 'OPTIMAL':
        record['reason'] = 'Phase-I did not reach optimality'
        return None, record
    total = float(np.sum(status.values[columns:]))
    record['artificial_sum'] = total
    if total > max(feasibility_tolerance, 1e-9)*(1.0+float(np.max(np.abs(status.values), initial=0.0))):
        record['reason'] = 'Phase-I optimum leaves positive artificial activity'
        return 'INFEASIBLE', record
    # Phase-I 最优时人工变量取零，但可能仍留在基里。必须把它们换成结构列，
    # 否则第二阶段拿不到 m 列的合法基。残留且找不到非零枢轴的行即冗余行。
    #
    # 驱除判据是**精确**的：人工列是 σ_i e_i 且位于基的第 position 位，故
    #   B e_position = σ_i e_i  =>  e_position^T B^{-1} = σ_i e_i^T
    #   (B^{-1} A[:, j])_position = e_position^T B^{-1} A[:, j] = σ_i A[i, j]
    # 也就是说"第 i 行结构系数非零"的列恰好能在该位置形成非零枢轴。早期版本改用
    # btran(e_i) 的**第 j 个分量**判据，那对应 B^{-1} 的第 i 行而不是枢轴所在位置，
    # 与所需条件不符（只是碰巧在首列为非零时选中 j=0）。
    sparse_rows = csc_matrix(matrix).tocsr()
    basis = list(status.basic)
    drive_out = []
    guard = 0
    while True:
        guard += 1
        if guard > rows+columns+10:
            record['reason'] = 'artificial drive-out did not terminate'
            return None, record
        position = next((k for k, j in enumerate(basis) if j >= columns), None)
        if position is None:
            break
        artificial_column = basis[position]
        row_index = artificial_column-columns
        coefficients = np.asarray(sparse_rows[row_index, :columns].todense()).ravel()
        chosen = None
        for j in range(columns):
            if j in basis or abs(coefficients[j]) <= 1e-9:
                continue
            chosen = j
            break
        if chosen is None:
            record['reason'] = 'row is redundant; artificial variable cannot be driven out'
            record['redundant_basis_position'] = position
            return None, record
        pivot_value = sigma[row_index]*float(coefficients[chosen])
        if abs(pivot_value) <= 1e-12:
            record['reason'] = 'drive-out pivot is numerically zero'
            return None, record
        current = SparseBasis(wide, basis)
        # 清理每次重建的成功LU也是实际成本，含额度耗尽或零枢轴退出前已完成的工作。
        before_update = current.state.refactorisations
        record['refactorisations'] += before_update
        record['cleanup_refactorisations'] += before_update
        column = current.ftran(wide[:, chosen].toarray().ravel(), validate=False)
        if abs(column[position]) <= 1e-12:
            record['reason'] = 'drive-out pivot is numerically zero'
            return None, record
        # 人工驱除也是实际换基，消耗同一额度；不得在Phase-I用完额度后静默增加工作量。
        if record['iterations'] >= _iteration_budget(kwargs.get('iteration_limit', 20000)):
            record.update(status='ITERATION_LIMIT', reason='Iteration limit reached during artificial cleanup',
                          drive_out=drive_out)
            return None, record
        current.update(position=position, entering_column=column, entering=chosen,
                       leaving=artificial_column)
        extra_refactors = current.state.refactorisations-before_update
        record['refactorisations'] += extra_refactors
        record['cleanup_refactorisations'] += extra_refactors
        record['iterations'] += 1
        record['pivots'] += 1
        basis = list(current.basic)
        drive_out.append(dict(position=position, artificial=int(artificial_column),
                              replaced_by=int(chosen), pivot=float(column[position])))
    record['basis'] = basis
    record['drive_out'] = drive_out
    # Phase-I 结束时人工变量全为 0（否则已判 INFEASIBLE），因此结构部分就是原系统的一个
    # **已验证可行点**，而且非基变量恰好停在它当时的界上。必须把它交给第二阶段：若第二
    # 阶段按"默认落位"（有下界就放上界）重新摆放非基变量，得到的是另一个点，可能越界
    # （实测随机题上第二阶段起点越界 1.68，直接报 START_INFEASIBLE）。
    record['values'] = [float(v) for v in np.asarray(status.values)[:columns]]
    return 'FEASIBLE', record

