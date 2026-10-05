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
update vector and one pivot position, and any solve applies those updates in the
correct order. This replaces the rank-one update of a dense factorisation while
keeping the update algebra explicit and independently checkable.

Numerical safety is not optional here: every solve is validated against the CURRENT
basis (``B_cur x = b``, ``B_cur' y = c``) rather than against the stale factors, and a
bounded residual corrections use the same CURRENT-basis solve. Every correction is
recorded. The default gate preserves the original RHS-relative criterion; a strict
componentwise criterion is available explicitly for research ablations.
"""
from dataclasses import asdict, dataclass, field
from fractions import Fraction
import math
import operator

import numpy as np
from scipy.linalg.blas import daxpy
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
# 次枢轴毫无进展时**永久**改用最小下标入基规则。换出规则与浮点运算
# 尚不满足完整 Bland 定理前提；切换不可逆只保持确定性，不保证有限终止。
DEFAULT_BLAND_AFTER = 1000
DEFAULT_STALL_AFTER = 100


@dataclass
class BasisUpdate:
    """一次乘积形式基更新。

    换基后 ``B_new = B_old + (a - B e_p) e_p'``，其中 ``a`` 是入基列。
    记 ``eta = B_old^{-1} a``，则逆矩阵按下式更新：

        ``B_new^{-1} = (I - (eta - e_p) e_p' / eta_p) B_old^{-1}``,

    归一化列 ``gamma = (eta - e_p) / eta_p`` 仅供已有诊断读取；FTRAN/BTRAN
    先按等价的 eta 主元除法计算，避免主元坐标的大数相减。
    """

    pivot: int          # 换入列在基中的位置（0-based）
    gamma: np.ndarray   # 归一化列只供现有剖析读数；求解运算改用稳定 eta 式
    eta: np.ndarray     # 原始 FTRAN 列 B_old^{-1} a，供稳定更新及诊断
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
        # SciPy 用 Python 列表切取十万级基列时，每次都会逐项解析索引。
        # 基索引数组与列表只在枢轴处同步，供装配和目标系数切片重复使用。
        self._basic_indices = np.asarray(self.basic, dtype=np.intp)
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
        # 残差复核在相邻枢轴间使用同一个 B；缓存只服务线性代数，不能改变换基决策。
        self._residual_basis_matrix = None
        self._residual_abs_basis_matrix = None
        self.refactorise()

    # ---------- 基矩阵装配与分解 ----------

    def assemble(self):
        """Current ``B`` as a CSC matrix, columns in the order of ``self.basic``."""
        return csc_matrix(self.matrix[:, self._basic_indices])

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
        self._residual_basis_matrix = matrix
        # LAPACK DGERFS 的分量后向误差分母为 |op(B)| |y|+|rhs|；
        # 同一基的正反回代可复用 |B|，转置时仅取其转置，换基后必须失效。
        self._residual_abs_basis_matrix = abs(matrix)
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
                # E 的主元列为 eta。先直接除得新主元，再对其余行做 DAXPY；
                # 不用 y[p]-(1-1/eta[p])*y[p] 的灾难性相消式。
                pivot_value = float(solution[update.pivot]/update.eta[update.pivot])
                solution = daxpy(update.eta, solution, a=-pivot_value)
                solution[update.pivot] = pivot_value
            return solution
        solution = rhs.copy()
        # E^{-T} 的主元为 (y[p]-sum_{i != p} eta[i]*y[i])/eta[p]；
        # 排除主元后分别点积，避免先形成含主元的大和再相减。
        for update in reversed(self.updates):
            pivot = update.pivot
            off_pivot = (float(update.eta[:pivot] @ solution[:pivot])
                         + float(update.eta[pivot+1:] @ solution[pivot+1:]))
            solution[pivot] = (solution[pivot]-off_pivot)/update.eta[pivot]
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
        # 每次换基后仅重建一次实际 B；FTRAN/BTRAN 的残差必须始终对当前基检验。
        if self._residual_basis_matrix is None:
            self._residual_basis_matrix = self.assemble()
            self._residual_abs_basis_matrix = abs(self._residual_basis_matrix)
        matrix = self._residual_basis_matrix
        operator = matrix.T if transpose else matrix
        absolute_operator = (self._residual_abs_basis_matrix.T if transpose
                             else self._residual_abs_basis_matrix)
        scale = 1.0+float(np.max(np.abs(rhs), initial=0.0))

        def measure(candidate):
            # 按 op(B) 的每个分量计量，防止某个大行掩盖小行；0/0 的精确零方程记零。
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                residual = rhs-operator @ candidate
                denominator = absolute_operator @ np.abs(candidate)+np.abs(rhs)
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
        self._basic_indices[position] = int(entering)
        self._residual_basis_matrix = None
        self._residual_abs_basis_matrix = None
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
        dual = self.btran(cost[self._basic_indices])
        return cost - self.matrix.T @ dual

    def dual_values(self, costs):
        """``y = BTRAN(c_B)`` for the current basis."""
        cost = np.asarray(costs, dtype=float).reshape(-1)
        return self.btran(cost[self._basic_indices])


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
    phase_one_candidate: np.ndarray | None = None
    unpriced_optimality: bool = False


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
    # 极窄盒两端都可能落入同一个绝对贴界容差。精确落界先于近似判定，
    # 否则精确下界会被错标为上界（反之亦然），导致定价方向翻转。
    if math.isfinite(lower) and value == lower:
        return False
    if math.isfinite(upper) and value == upper:
        return True
    at_lower = math.isfinite(lower) and abs(value-lower) <= 1e-12
    at_upper = math.isfinite(upper) and abs(value-upper) <= 1e-12
    if at_lower and at_upper:
        # 同时近界但均非精确界时，无法从值确定非基状态；不可任意猜一侧。
        raise NumericalError(f'Placed value {value} ambiguously matches both bounds '
                             f'({lower}, {upper})')
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


def _finite_bound_room(bound, current, direction):
    """通常走 binary64；有限数相减上溢时按原 binary64 值精确求商。"""
    bound, current, direction = float(bound), float(current), float(direction)
    if not all(math.isfinite(value) for value in (bound, current, direction)) or direction == 0:
        raise NumericalError('Basic blocking ratio has non-finite input or zero direction')
    span = bound-current
    if math.isfinite(span):
        return span/direction
    # 异常路径很少触发：不能把两个有限数的上溢差误当作无穷阻挡。
    exact = ((Fraction.from_float(bound)-Fraction.from_float(current))
             /Fraction.from_float(direction))
    try:
        return float(exact)
    except OverflowError:
        return math.inf if exact > 0 else -math.inf


def _ratio_limit_scalar(basic, moving, values, lower, upper, pricing_tolerance,
                        *, feasibility_tolerance=None):
    """取严格最小比例；显式提供可行容差时审查负比例的退化路径。"""
    step = math.inf
    limiting = -1
    limiting_upper = False
    unrepresentable_positive_blocker = False
    negative_room = []
    for position, j in enumerate(basic):
        d = -moving[position]
        # 无论 d 多小，只要非零且沿该方向移动，足够大的步长仍会触及有限界。
        # 定价阈值仅决定是否入基；比例检验必须按真实方向维护原始可行性。
        if d > 0.0 and math.isfinite(upper[j]):
            room = _finite_bound_room(upper[j], values[j], d)
            if not math.isfinite(room):
                # 正溢出比任何可表示的有限候选都晚；若没有别的候选则不能宣告无界。
                if room > 0:
                    unrepresentable_positive_blocker = True
                    continue
                raise NumericalError('Finite basic upper blocker has an invalid ratio')
            if room < 0.0 and feasibility_tolerance is not None:
                negative_room.append((j, d, upper[j]))
                continue
            # 比例差不能直接作为行可行容差：1e-12 的步长差乘以 1e12
            # 的方向分量，会使其他基本变量越界 1 个原模型单位。
            if room < step or (room == step and limiting >= 0
                               and j < basic[limiting]):
                step, limiting, limiting_upper = room, position, True
        elif d < 0.0 and math.isfinite(lower[j]):
            room = _finite_bound_room(lower[j], values[j], d)
            if not math.isfinite(room):
                if room > 0:
                    unrepresentable_positive_blocker = True
                    continue
                raise NumericalError('Finite basic lower blocker has an invalid ratio')
            if room < 0.0 and feasibility_tolerance is not None:
                negative_room.append((j, d, lower[j]))
                continue
            if room < step or (room == step and limiting >= 0
                               and j < basic[limiting]):
                step, limiting, limiting_upper = room, position, False
    if limiting < 0 and unrepresentable_positive_blocker:
        raise NumericalError('Finite basic blocker ratio exceeds binary64 range')
    if negative_room:
        # 负比例来自当前基本量微越界，不存在合法的负步长。若直接截为零却保留
        # 离基位置，近零方向会成为病态主元。只在另一个真实有限阻挡点已确定、
        # 且起点和该点上的微越界都未超过既定原可行容差时，才忽略伪阻挡。
        # 无下一阻挡点时不能借此声称改善射线；真实正比值的小方向始终参与。
        if limiting < 0 or not math.isfinite(step):
            raise NumericalError('Negative basic ratio has no checked finite alternative')
        for j, d, bound in negative_room:
            proposed = values[j]+d*step
            if d > 0.0:
                start_error = values[j]-bound
                end_error = proposed-bound
            else:
                start_error = bound-values[j]
                end_error = bound-proposed
            if (not math.isfinite(proposed) or not math.isfinite(start_error)
                    or not math.isfinite(end_error)
                    or start_error > feasibility_tolerance
                    or end_error > feasibility_tolerance):
                raise NumericalError('Negative basic ratio cannot be covered by the '
                                     'existing primal feasibility tolerance')
    return step, limiting, limiting_upper


def _ratio_limit_vectorized(basic, moving, values, lower, upper, pricing_tolerance,
                            *, feasibility_tolerance=None):
    """非零方向均参与阻挡；默认直调保留历史纯比例口径。"""
    indices = np.asarray(basic, dtype=np.intp)
    direction = -np.asarray(moving, dtype=float)
    bound_upper = upper[indices]
    bound_lower = lower[indices]
    current = values[indices]
    # 与标量回退保持同一物理判据；不复用简约成本的 pricing_tolerance。
    upper_candidate = (direction > 0.0) & np.isfinite(bound_upper)
    lower_candidate = (direction < 0.0) & np.isfinite(bound_lower)
    candidate = upper_candidate | lower_candidate
    if not np.any(candidate):
        return math.inf, -1, False
    rooms = np.full(indices.size, math.inf)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        rooms[upper_candidate] = ((bound_upper[upper_candidate]-current[upper_candidate])
                                  /direction[upper_candidate])
        rooms[lower_candidate] = ((bound_lower[lower_candidate]-current[lower_candidate])
                                  /direction[lower_candidate])
    finite_rooms = rooms[candidate]
    if not np.all(np.isfinite(finite_rooms)):
        return _ratio_limit_scalar(basic, moving, values, lower, upper, pricing_tolerance,
                                   feasibility_tolerance=feasibility_tolerance)
    if np.any(finite_rooms < 0.0):
        return _ratio_limit_scalar(basic, moving, values, lower, upper, pricing_tolerance,
                                   feasibility_tolerance=feasibility_tolerance)
    if finite_rooms.size > 1:
        two_smallest = np.partition(finite_rooms, 1)[:2]
        if two_smallest[1]-two_smallest[0] <= 1e-12:
            return _ratio_limit_scalar(basic, moving, values, lower, upper, pricing_tolerance,
                                       feasibility_tolerance=feasibility_tolerance)
    position = int(np.argmin(rooms))
    return float(rooms[position]), position, bool(upper_candidate[position])


def _weak_pivot_refine_needed(moving, limiting, threshold):
    """仅判断已选离基主元相对 FTRAN 最大分量是否弱；不改变比例规则。"""
    direction = np.asarray(moving, dtype=float)
    if limiting < 0 or limiting >= direction.size or not np.all(np.isfinite(direction)):
        return False
    scale = float(np.max(np.abs(direction), initial=0.))
    return bool(scale > 0 and abs(float(direction[limiting]))/scale < threshold)


def _stable_zero_ratio_choice(basic, moving, values, lower, upper, selected,
                              feasibility_tolerance, pricing_tolerance, *, use_bland):
    """仅在舍入级零步长并列中选择较大主元；不允许正步跨过任何边界。"""
    step, old_position, old_upper = selected
    if use_bland or old_position < 0 or step >= 0:
        return selected
    indices = np.asarray(basic, dtype=np.intp)
    direction = -np.asarray(moving, dtype=float)
    current = np.asarray(values, dtype=float)[indices]
    lower_values = np.asarray(lower, dtype=float)[indices]
    upper_values = np.asarray(upper, dtype=float)[indices]
    machine_epsilon = np.finfo(float).eps

    def boundary(position):
        d = float(direction[position])
        if d > pricing_tolerance and math.isfinite(upper_values[position]):
            return float(upper_values[position]), True
        if d < -pricing_tolerance and math.isfinite(lower_values[position]):
            return float(lower_values[position]), False
        return None, False

    old_bound, _ = boundary(old_position)
    if old_bound is None:
        return selected
    old_gap = abs(old_bound-current[old_position])
    old_roundoff = min(feasibility_tolerance,
                       8*machine_epsilon*max(1.0, abs(old_bound),
                                              abs(float(current[old_position]))))
    if old_gap > old_roundoff:
        return selected
    # 已观测的 ratio<=0 在外层本就裁为零；这里所有候选都只执行零步换基。
    # 使用绝对边界差而非 room，可避免 1 ULP 除以微小方向后伪装成远离零的最小比值。
    best = old_position
    best_magnitude = abs(float(direction[old_position]))
    best_upper = old_upper
    for position in range(indices.size):
        bound, on_upper = boundary(position)
        if bound is None:
            continue
        roundoff = min(feasibility_tolerance,
                       8*machine_epsilon*max(1.0, abs(bound), abs(float(current[position]))))
        if abs(bound-current[position]) > roundoff:
            continue
        magnitude = abs(float(direction[position]))
        if (magnitude > best_magnitude
                or (magnitude == best_magnitude and indices[position] < indices[best])):
            best, best_magnitude, best_upper = position, magnitude, on_upper
    return (0.0, int(best), bool(best_upper))


def revised_simplex(matrix, costs, bounds_lower, bounds_upper, *, basic=None,
                    iteration_limit=20000, pricing_tolerance=1e-9,
                    feasibility_tolerance=1e-7, residual_tol=DEFAULT_RESIDUAL_TOL,
                    refactor_interval=DEFAULT_REFACTOR_INTERVAL, max_eta=DEFAULT_MAX_ETA,
                    objective_offset=0.0, maximize=False, bland_after=DEFAULT_BLAND_AFTER,
                    stall_after=DEFAULT_STALL_AFTER, initial=None,
                    refinement_steps=DEFAULT_REFINEMENT_STEPS, residual_metric='rhs',
                    _phase_one_structural_columns=None, stable_zero_pivot=False,
                    integer_zero_refine=False, integer_zero_refine_weak_threshold=None,
                    integer_exact_snap=False,
                    positive_step_pricing=False,
                    positive_step_scan_after=100, positive_step_scan_every=100):
    """Primal revised simplex for ``min c'x`` s.t. ``Ax = 0``, ``l <= x <= u``.

    Computational form follows Huangfu & Hall §2.1 (arXiv:1503.01889v1): row bounds are
    carried inside ``A`` by identity columns, so the system is homogeneous. The method is
    the *bounded-variable* revised simplex: every nonbasic variable sits exactly on one of
    its bounds, and an entering variable may move only away from that bound.

    Robustness choices, each deliberate:

    * **Dantzig pricing with a permanent switch to Bland-style least-index pricing.**
      Exact Bland entering/leaving rules are finite in exact arithmetic. This numerical
      implementation uses floating tolerances in the ratio test, so that theorem is not
      claimed as a finite-runtime guarantee here. The switch happens after
      ``bland_after`` pivots or ``stall_after`` consecutive zero steps and stays active.
      An earlier revision
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
    if integer_exact_snap and not integer_zero_refine:
        raise ValueError('integer_exact_snap requires integer_zero_refine')
    if integer_zero_refine_weak_threshold is not None and not integer_zero_refine:
        raise ValueError('integer_zero_refine_weak_threshold requires integer_zero_refine')
    if (integer_zero_refine_weak_threshold is not None
            and (isinstance(integer_zero_refine_weak_threshold, bool)
                 or not isinstance(integer_zero_refine_weak_threshold, (int, float))
                 or not math.isfinite(integer_zero_refine_weak_threshold)
                 or not 0 < integer_zero_refine_weak_threshold <= 1)):
        raise ValueError('weak pivot threshold must be finite and in (0, 1]')
    for label, value, minimum in (
            ('positive_step_scan_after', positive_step_scan_after, 0),
            ('positive_step_scan_every', positive_step_scan_every, 1)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f'{label} must be an integer at least {minimum}')

    def stopped_before_updates(state, **fields):
        # 分解后拒绝起点也有真实工作；成功LU次数来自快照，不把零动作误当零计算。
        fields.setdefault('refactorisations', fields.get('basis_diagnostics', {}).get('refactorisations', 0))
        return SimplexResult(state, iteration_budget=_budget_record(iteration_limit), **fields)

    matrix = csc_matrix(matrix)
    rows, columns = matrix.shape
    # NaN/Inf 系数不能参与最优性/无界性证明；先拒绝非法输入，
    # 否则 NaN 的比较结果为假，可能把零动作点误晋升为 OPTIMAL。
    if not np.all(np.isfinite(matrix.data)):
        raise ValueError('finite matrix coefficients required')
    if _phase_one_structural_columns is not None:
        if (isinstance(_phase_one_structural_columns, bool)
                or not isinstance(_phase_one_structural_columns, (int, np.integer))
                or not 0 < _phase_one_structural_columns < columns):
            raise ValueError('Phase-I structural column count must be inside the matrix')
        phase_original = matrix[:, :_phase_one_structural_columns]
    else:
        phase_original = None
    cost = np.asarray(costs, dtype=float).reshape(-1)
    lower = np.asarray(bounds_lower, dtype=float).reshape(-1)
    upper = np.asarray(bounds_upper, dtype=float).reshape(-1)
    if cost.size != columns or lower.size != columns or upper.size != columns:
        raise ValueError('Cost and bound vectors must match the number of columns')
    if not np.all(np.isfinite(cost)):
        raise ValueError('finite cost coefficients required')
    if not math.isfinite(objective_offset):
        raise ValueError('finite objective offset required')
    if (np.any(np.isnan(lower)) or np.any(np.isnan(upper))
            or np.any(np.isposinf(lower)) or np.any(np.isneginf(upper))):
        raise ValueError('invalid variable bound: NaN or wrong-side infinity')
    for label, tolerance in (('pricing_tolerance', pricing_tolerance),
                             ('feasibility_tolerance', feasibility_tolerance)):
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError(f'finite nonnegative tolerance required: {label}')
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
            try:
                at_upper[j] = _bound_side_after_placement(values[j], lower[j], upper[j])
            except NumericalError:
                return stopped_before_updates(
                    'NUMERICAL_ERROR',
                    message=f'Nonbasic variable {j} does not select an unambiguous bound '
                            f'({lower[j]}, {upper[j]}); a bounded-variable basis needs a bound',
                    basis_diagnostics=diagnostics())
            # 调用方给出近界值时落实为真实界，维持非基变量精确贴界的不变量。
            values[j] = upper[j] if at_upper[j] else lower[j]
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
    stable_zero_choices = 0
    integer_refine_attempts = 0
    integer_refine_choices = 0
    integer_refine_weak_skips = 0
    integer_refine_last_rejection = None
    integer_snap_attempts = 0
    integer_snap_choices = 0
    integer_snap_last_rejection = None
    positive_step_scans = 0
    positive_step_scanned_columns = 0
    positive_step_choices = 0
    positive_step_last_rejection = None
    status = 'ITERATION_LIMIT'
    message = 'Iteration limit reached before optimality was proved'
    unpriced_optimality = False
    for iteration in range(iteration_limit+1):
        completed = iteration
        if basis.needs_refactorisation():
            basis.refactorise()
        # 基变量值必须满足 A x = 0，且在**每次基/非基状态变化之后**重算：非基变量被置于
        # 某一侧界（翻界、换出落界只记状态），沿用旧值会让方程残差停在错误位置，实测
        # 表现为步长恒为 0 的零进展循环。
        # SparseBasis 在成功换基时同步维护有序 intp 下标；本轮直接复用，
        # 避免每次把大基列表重新解析为数组。换基后的下一轮会读取已更新的缓存。
        basic_indices = basis._basic_indices
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
        if phase_original is not None:
            # Phase-I 只需找到原模型可行点；当每个人工量已接近零时才做较贵的
            # 原结构行/界复核。通过后保留专用状态，由调用方驱除残余人工基。
            artificial = values[_phase_one_structural_columns:]
            if np.all(np.isfinite(artificial)) and float(np.max(np.abs(artificial), initial=0.0)) \
                    <= feasibility_tolerance:
                check = _phase_one_primal_check(matrix, phase_original, values, lower,
                                                upper, feasibility_tolerance)
                if check['internal_primal_feasible']:
                    status = 'PHASE_ONE_FEASIBLE'
                    message = 'Original structural point passed Phase-I feasibility gates'
                    break
        # 每轮由**实际值**重算所有非基变量的所在侧，消除标志与值的漂移。标志若与实际
        # 值不一致，定价会朝不可行方向选变量（实测在已最优点上反复选中不该选的变量）。
        # 这里同样向量化：逐列版本是 `for j in range(columns)` 外加一次 `j in basis_positions`
        # 字典查询，72747 列时每轮就是七万次字典查询——它正是定价循环修好之后暴露出来的
        # 下一个热点（faulthandler 栈停在 sparse_simplex.py 的这一行）。
        # 语义保持逐列版本的三分支：非基且贴在上界 -> True；否则非基且贴在下界 -> False；
        # 基变量与两侧都不贴的变量保持原标志不变。
        exact_lower = np.isfinite(lower) & (values == lower)
        exact_upper = np.isfinite(upper) & (values == upper)
        snapped_upper = np.isfinite(upper) & (np.abs(values-upper) <= 1e-12)
        snapped_lower = np.isfinite(lower) & (np.abs(values-lower) <= 1e-12)
        ambiguous = (~in_basis) & (~exact_lower) & (~exact_upper) & snapped_lower & snapped_upper
        if np.any(ambiguous):
            status = 'NUMERICAL_ERROR'
            message = 'A nonbasic value ambiguously matches both bounds'
            break
        # 精确下界/上界优先；只在精确值缺失时才容忍单侧的近界舍入。
        at_upper = np.where(~in_basis & exact_lower, False,
                            np.where(~in_basis & exact_upper, True,
                                     np.where(~in_basis & snapped_upper, True,
                                              np.where(~in_basis & snapped_lower, False, at_upper))))
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

        # 定价。Dantzig 取最改进者；枢轴数超门或连续多轮无进展后**永久**改用最小下标。
        # 精确 Bland 规则有有限性定理，但当前浮点容差与近并列处理不能直接继承该证明；
        # 达到迭代额度仍如实返回 ITERATION_LIMIT，绝不据切换本身推断最优。
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
                # 定价门是选列阈值，不是严格最优证书：微小简约成本可乘以巨大
                # 可行步长，形成显著目标改进；无穷界时甚至可能无界。没有额外
                # 证书就保守保留当前候选，不将其升级为 OPTIMAL。
                unpriced = (~in_basis) & (~fixed) & (improving > 0.0)
                if np.any(unpriced):
                    status = 'NUMERICAL_ERROR'
                    # 仅此分支可由原域独立证书尝试恢复；其他数值故障不能
                    # 依赖消息文本相似性而被事后误认成可认证的停止。
                    unpriced_optimality = True
                    message = (f'{int(np.count_nonzero(unpriced))} improving nonbasic '
                               'reduced costs were below the pricing threshold; '
                               'the current basis does not certify optimality')
                else:
                    status, message = 'OPTIMAL', 'Reduced costs proved optimality for the current basis'
            else:
                status = 'NUMERICAL_ERROR'
                message = f'No entering column but primal violation {violation:.3e} remains'
            break

        # 末端可验证当前基是否最优，但不可再执行下一次移动；否则额度N实际执行N+1次。
        if completed >= iteration_limit:
            break

        try:
            # 入基方向直接决定比例检验和新基；必须先对当前 B 验证，失败时只重分解一次。
            # 跳过此门会把含显著残差的 eta_p 当主元，使本来满秩的旧基换成数值奇异基。
            entering_rhs = matrix[:, entering].toarray().ravel()
            pivot_column = guarded(lambda: basis.ftran(entering_rhs, validate=True))
        except NumericalError as error:
            status = 'NUMERICAL_ERROR'
            message = f'validated pivot direction did not recover after refactorisation: {error}'
            break
        # 沿"变量上升"为正方向；非基变量只能离开它当前所在的界。
        sign = -1.0 if at_upper[entering] else 1.0
        moving = sign*pivot_column
        # 比例检验：先看基本变量能走多远。记录限制变量触到的是哪一侧界——换出后必须
        # 精确落在**实际触到的那个界**上；用"最近界"会把变量放到它并未触到的界上。
        # 关键符号：moving = sign * B^{-1}a 是入基变量沿自身可行方向上升时，基本变量
        # 需要**减去**的变化量（x_B(α) = x_B − moving·α）。故只有 `d = −moving[i] > 0`
        # 的基本变量会下降、才可能先触下界；判据若直接用 moving[i] 会把方向判反，
        # 表现为步长恒为 0。教科书参考实现 `zyo/standard_simplex.py` 用同一符号约定
        # 通过了手算题（两次枢轴、目标 −5）。
        # 明显唯一的最小比例走 NumPy 快路径；退化近并列回退严格最小比的标量规则。
        try:
            step, limiting, limiting_upper = _ratio_limit_vectorized(
                basic_indices, moving, values, lower, upper, pricing_tolerance)
        except NumericalError as error:
            status = 'NUMERICAL_ERROR'
            message = f'physical ratio test failed before any state update: {error}'
            break
        positive_choice = None
        if (positive_step_pricing and phase_original is None
                and step <= pricing_tolerance and stalled >= positive_step_scan_after
                and (stalled-positive_step_scan_after) % positive_step_scan_every == 0):
            # Positive Edge 文献强调非退化改进的价值；这里是代价更高的确定性筛选，
            # 先用新 LU 估步，再用现有内核 FTRAN/比例门复核，不能把估计直接当求解动作。
            from .positive_step_pricing import screen_positive_step
            positive_step_scans += 1
            scan = screen_positive_step(
                matrix=matrix, basic=basic_indices, eligible=eligible,
                improving=improving, at_upper=at_upper, values=values,
                lower=lower, upper=upper, pricing_tolerance=pricing_tolerance,
                minimum_step=feasibility_tolerance)
            positive_step_scanned_columns += scan['scanned_columns']
            if scan['accepted']:
                proposed = scan['column']
                try:
                    proposed_rhs = matrix[:, proposed].toarray().ravel()
                    proposed_pivot = guarded(lambda: basis.ftran(proposed_rhs, validate=True))
                except NumericalError as error:
                    # 备选列失败不等于已验证的普通列失败；重分解后重新验证普通方向。
                    positive_step_last_rejection = f'alternative FTRAN failed: {error}'
                    try:
                        pivot_column = guarded(lambda: basis.ftran(entering_rhs, validate=True))
                    except NumericalError as ordinary_error:
                        status = 'NUMERICAL_ERROR'
                        message = f'ordinary direction also failed after alternative rejection: {ordinary_error}'
                        break
                    moving = sign*pivot_column
                    try:
                        step, limiting, limiting_upper = _ratio_limit_vectorized(
                            basic_indices, moving, values, lower, upper, pricing_tolerance)
                    except NumericalError as ordinary_error:
                        status = 'NUMERICAL_ERROR'
                        message = (f'ordinary physical ratio failed after alternative '
                                   f'rejection: {ordinary_error}')
                        break
                else:
                    proposed_sign = -1.0 if at_upper[proposed] else 1.0
                    proposed_moving = proposed_sign*proposed_pivot
                    try:
                        proposed_step, proposed_limit, proposed_upper = _ratio_limit_vectorized(
                            basic_indices, proposed_moving, values, lower, upper,
                            pricing_tolerance)
                    except NumericalError as error:
                        # 备选列的算术失败只否决该提议；普通列的已核比例保持不变。
                        positive_step_last_rejection = f'alternative physical ratio failed: {error}'
                    else:
                        proposed_room = (values[proposed]-lower[proposed] if proposed_sign < 0
                                         else upper[proposed]-values[proposed])
                        if min(proposed_step, proposed_room) > feasibility_tolerance:
                            entering, entering_rhs, pivot_column = proposed, proposed_rhs, proposed_pivot
                            sign, moving = proposed_sign, proposed_moving
                            step, limiting, limiting_upper = (proposed_step, proposed_limit,
                                                              proposed_upper)
                            positive_choice = scan
                            positive_step_choices += 1
                        else:
                            positive_step_last_rejection = 'fresh-LU positive estimate failed native FTRAN/ratio gate'
            else:
                positive_step_last_rejection = scan['reason']
        integer_choice = None
        used_integer_snap = False
        refine_this_pivot = (integer_zero_refine and phase_original is None
                             and step <= pricing_tolerance)
        if (refine_this_pivot and integer_zero_refine_weak_threshold is not None
                and not _weak_pivot_refine_needed(
                    moving, limiting, integer_zero_refine_weak_threshold)):
            # 阈值仅减少可选高精度试验的触发次数；未触发时严格沿用原比值结果。
            integer_refine_weak_skips += 1
            refine_this_pivot = False
        if refine_this_pivot:
            # 只在整数系数及非基整数边界可证明生成精确整数 RHS 时采用额外精度；
            # 普通 LP/Phase-I 不变，所选离基值还须几乎精确在原界上，不能暗中扩可行域。
            from .primal_integer_zero_refine import (refine_integer_basic,
                                                     strong_zero_leaving)
            integer_refine_attempts += 1
            integer_snap_attempts += int(integer_exact_snap)
            try:
                refined = refine_integer_basic(
                    matrix, basic_indices, nonbasic_values,
                    exact_integer_snap=integer_exact_snap,
                    basic_lower=lower[basic_indices] if integer_exact_snap else None,
                    basic_upper=upper[basic_indices] if integer_exact_snap else None,
                    feasibility_tolerance=feasibility_tolerance if integer_exact_snap else None)
                if refined['accepted']:
                    snapped = bool(refined.get('integer_exact_snap_accepted'))
                    if integer_exact_snap and not snapped:
                        integer_snap_last_rejection = refined['integer_exact_snap_reason']
                    # 整数点通过 Bq=b 后仍需原行、原界和原 FTRAN 方向门；
                    # 若新点被任一旧门拒绝，保留原额外精度候选的既有路径。
                    candidates = [(snapped, refined['values'])]
                    if snapped:
                        candidates.append((False, refined['unsnapped_values']))
                    for using_snap, candidate_values in candidates:
                        candidate = np.asarray(candidate_values, dtype=float)
                        trial_values = values.copy()
                        trial_values[basic_indices] = candidate
                        row_error = float(np.max(np.abs(matrix @ trial_values), initial=0.))
                        row_gate = feasibility_tolerance*(1.+float(np.max(np.abs(trial_values), initial=0.)))
                        if row_error <= row_gate:
                            integer_choice = strong_zero_leaving(
                                basic=basic_indices, moving=moving, values=candidate,
                                lower=lower[basic_indices], upper=upper[basic_indices],
                                feasibility_tolerance=feasibility_tolerance,
                                pricing_tolerance=pricing_tolerance)
                            if integer_choice['accepted']:
                                values[basic_indices] = candidate
                                step = integer_choice['step']
                                limiting = integer_choice['position']
                                limiting_upper = integer_choice['leaving_upper']
                                integer_refine_choices += 1
                                integer_snap_choices += int(using_snap)
                                used_integer_snap = using_snap
                                break
                            rejection = integer_choice['reason']
                        else:
                            rejection = 'refined basic point failed original equation gate'
                        integer_refine_last_rejection = rejection
                        if using_snap:
                            integer_snap_last_rejection = rejection
                else:
                    integer_refine_last_rejection = refined['reason']
                    if integer_exact_snap:
                        integer_snap_last_rejection = refined['reason']
            except (ArithmeticError, ValueError, RuntimeError) as error:
                # 算术失败不晋升状态；实验臂回到原比值并在末端保留拒绝原因。
                integer_refine_last_rejection = f'{type(error).__name__}: {error}'
                if integer_exact_snap:
                    integer_snap_last_rejection = integer_refine_last_rejection
        if stable_zero_pivot and phase_original is None and step < 0 and not use_bland:
            ordinary_choice = (step, limiting, limiting_upper)
            step, limiting, limiting_upper = _stable_zero_ratio_choice(
                basic_indices, moving, values, lower, upper, ordinary_choice,
                feasibility_tolerance, pricing_tolerance, use_bland=use_bland)
            stable_zero_choices += int(limiting != ordinary_choice[1])
        if step < 0.0:
            # 可选整数精确基点门先有机会把舍入残差修正为真实零步；若仍为
            # 负比例，则必须在任何状态更新之前核对另一阻挡点与既定可行容差。
            try:
                step, limiting, limiting_upper = _ratio_limit_scalar(
                    basic_indices, moving, values, lower, upper, pricing_tolerance,
                    feasibility_tolerance=feasibility_tolerance)
            except NumericalError as error:
                status = 'NUMERICAL_ERROR'
                message = f'physical ratio test failed before any state update: {error}'
                break
        # 再看入基变量自身的另一侧界：先触到它即为翻界，不换基。
        flip_room = math.inf
        finite_flip_bound = False
        if sign > 0 and math.isfinite(upper[entering]):
            finite_flip_bound = True
            flip_room = _finite_bound_room(upper[entering], values[entering], 1.0)
        elif sign < 0 and math.isfinite(lower[entering]):
            finite_flip_bound = True
            flip_room = _finite_bound_room(lower[entering], values[entering], -1.0)
        if not math.isfinite(step) and not math.isfinite(flip_room):
            # 两个有限端点的跨度也可能在 binary64 上溢；这种情形没有无界射线证书。
            status = 'NUMERICAL_ERROR' if finite_flip_bound else 'UNBOUNDED'
            message = ('Finite entering bound step exceeds binary64 range'
                       if finite_flip_bound else
                       'No bound limits the step; the objective improves without limit')
            break
        step = max(0.0, min(step, flip_room))

        # 动作前只复核会变化的基本量与入基量；比例检验或备选策略若漏掉
        # 微小但非零的方向，不允许先污染状态、再指望终点证书发现越界。
        # 此检查使用原变量的绝对可行容差，不把定价阈值当作物理界容差。
        basic_array = np.asarray(basic_indices, dtype=np.intp)
        with np.errstate(over='ignore', invalid='ignore'):
            proposed_basic = values[basic_array]-step*moving
            proposed_entering = values[entering]+sign*step
            basic_violation = np.maximum(lower[basic_array]-proposed_basic,
                                         proposed_basic-upper[basic_array])
        if (not math.isfinite(step) or not math.isfinite(proposed_entering)
                or not np.all(np.isfinite(proposed_basic))
                or float(np.max(basic_violation, initial=0.0)) > feasibility_tolerance
                or (math.isfinite(lower[entering])
                    and lower[entering]-proposed_entering > feasibility_tolerance)
                or (math.isfinite(upper[entering])
                    and proposed_entering-upper[entering] > feasibility_tolerance)):
            status = 'NUMERICAL_ERROR'
            message = 'Proposed simplex step violates a finite variable bound'
            break

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
        if positive_choice is not None:
            history[-1].update(positive_step_pricing=True,
                               screened_columns=positive_choice['scanned_columns'])
        if integer_zero_refine and integer_choice and integer_choice.get('accepted'):
            history[-1].update(integer_zero_refine=True,
                               selected_pivot_magnitude=integer_choice['pivot_magnitude'],
                               bound_snap_change=integer_choice['snap_change'])
            if used_integer_snap:
                history[-1]['integer_exact_snap'] = True
        # 迭代是实际已执行状态更新，零步长与翻界同样计入，不能只数下一轮检查或换基。
        completed += 1
        # 停滞计数：退化枢轴（步长为零）不改变目标值，长串零步长正是循环的前兆。
        # 这只改变入基选择；离基及浮点并列未满足完整 Bland 定理前提，
        # 因此不得据此声称有限终止。
        stalled = stalled+1 if step <= pricing_tolerance else 0
        # 翻界只在入基变量确实先触界（或严格同一步长）时成立。
        # 若仅因定价容差把稍大的 flip_room 当并列，强行落界的偏差会经
        # 大矩阵系数放大，导致原行或基本变量界严重失准。
        if flip_room <= step and flip_room < math.inf:
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
                                      stable_zero_pivot=bool(stable_zero_pivot),
                                      stable_zero_choices=int(stable_zero_choices),
                                      **(dict(positive_step_pricing=True,
                                              positive_step_scans=int(positive_step_scans),
                                              positive_step_scanned_columns=int(positive_step_scanned_columns),
                                              positive_step_choices=int(positive_step_choices),
                                              positive_step_last_rejection=positive_step_last_rejection)
                                         if positive_step_pricing else {}),
                                      **(dict(integer_zero_refine=True,
                                              integer_zero_refine_weak_threshold=(
                                                  None if integer_zero_refine_weak_threshold is None
                                                  else float(integer_zero_refine_weak_threshold)),
                                              integer_zero_refine_weak_skips=int(integer_refine_weak_skips),
                                              integer_zero_refine_attempts=int(integer_refine_attempts),
                                              integer_zero_refine_choices=int(integer_refine_choices),
                                              integer_zero_refine_last_rejection=integer_refine_last_rejection)
                                         if integer_zero_refine else {}),
                                      **(dict(integer_exact_snap=True,
                                              integer_exact_snap_attempts=int(integer_snap_attempts),
                                              integer_exact_snap_choices=int(integer_snap_choices),
                                              integer_exact_snap_last_rejection=integer_snap_last_rejection)
                                         if integer_exact_snap else {}),
                                      switched=bool(pivots >= bland_after
                                                    or stalled >= stall_after)),
                         refactor_retries=int(refactor_retries),
                         basis_diagnostics=diagnostics(), bound_flips=bound_flips,
                         iteration_budget=_budget_record(iteration_limit, completed),
                         unpriced_optimality=bool(status == 'NUMERICAL_ERROR'
                                                  and unpriced_optimality))


def _primal_violation(values, lower, upper, tolerance):
    violation = 0.0
    for index in range(values.size):
        if math.isfinite(lower[index]):
            violation = max(violation, lower[index]-values[index])
        if math.isfinite(upper[index]):
            violation = max(violation, values[index]-upper[index])
    return float(max(0.0, violation))


def _phase_one_primal_check(wide, original, values, lower, upper, tolerance):
    """独立于辅助目标检查原结构行、界、人工量及扩展方程。"""
    point = np.asarray(values, dtype=float).reshape(-1)
    structural_columns = original.shape[1]
    if point.size != wide.shape[1] or not np.all(np.isfinite(point)):
        return dict(internal_primal_feasible=False, reason='nonfinite or mismatched point')
    structural = point[:structural_columns]
    artificial = point[structural_columns:]
    row_error = float(np.max(np.abs(original @ structural), initial=0.0))
    augmented_error = float(np.max(np.abs(wide @ point), initial=0.0))
    bound_error = _primal_violation(structural, lower[:structural_columns],
                                    upper[:structural_columns], tolerance)
    artificial_error = float(np.max(np.abs(artificial), initial=0.0))
    accepted = max(row_error, augmented_error, bound_error,
                   artificial_error) <= tolerance
    return dict(internal_primal_feasible=bool(accepted),
                structural_row_residual=row_error, augmented_row_residual=augmented_error,
                structural_bound_violation=bound_error, artificial_max_abs=artificial_error,
                tolerance=float(tolerance), original_domain_optimality_certified=False)


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


def _phase_two_nonbasic_initial(values, basis):
    """从 Phase-I 可行点提取非基界位；基集合只构造一次。"""
    # 旧写法在字典推导式条件里调用 set(basis)，每处理一列都重建约 m 项集合。
    # 对 n 列、m 行是 O(nm) 的纯 Python 过渡成本；预构造后为 O(n+m)，
    # 下标遍历顺序和每个浮点值均与原写法保持一致，不改变单纯形决策。
    basic_members = set(basis)
    return {j: float(value) for j, value in enumerate(values) if j not in basic_members}

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
        # 只传递经过原齐次行/界复核的中间点；不伪造一个无人工列的基，也不改变停止状态。
        phase_candidate = phase_record.pop('stopped_feasible_candidate', None)
        return finish(SimplexResult(state, message=phase_record.get('reason', 'Phase-I could not establish a basis'),
                             phase_one=phase_record, iterations=phase_record.get('iterations', 0),
                             pivots=phase_record.get('pivots', 0),
                             bound_flips=phase_record.get('bound_flips', 0),
                             refactorisations=phase_record.get('refactorisations', 0),
                             phase_one_candidate=(None if phase_candidate is None else
                                                  np.asarray(phase_candidate, dtype=float))),
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
        phase_two['initial'] = _phase_two_nonbasic_initial(values, phase_record['basis'])
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
    if (result.status != 'OPTIMAL'
            and phase_record.get('original_feasible_check', {}).get('internal_primal_feasible')
            and values is not None):
        # 第二阶段数值失败/限额不得抹去已核验的第一阶段可行点；与正式解、
        # 第二阶段当前基及最优性证书分开保存，后续仍须回原输入再验。
        result.phase_one_candidate = np.asarray(values, dtype=float)
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
            candidates.setdefault(int(row), []).append((int(column), float(value), float(np.sign(value))))
    if not candidates:
        return None
    start = placement.copy()
    # 单位列只影响自己的唯一非零行；前面已选列不会改变后续候选行的活动量。
    # 一次稀疏乘法得到原始行活动量，避免逐候选行切片及稠密化。
    original_activity = np.asarray(body @ placement, dtype=float).reshape(-1)
    chosen = {}
    used_columns = set()
    for row in sorted(candidates):
        for column, coefficient, sign in candidates[row]:
            if column in used_columns:
                continue
            others = float(original_activity[row]-coefficient*placement[column])
            value = -others/sign
            if lower[column]-1e-12 <= value <= upper[column]+1e-12:
                chosen[row] = (column, float(value))
                used_columns.add(column)
                start[column] = float(value)
                break
    if not chosen:
        return None
    activity = np.asarray(body @ start, dtype=float).reshape(-1)
    artificial_rows = []
    basis_offset = []
    for row in range(rows):
        if row in chosen:
            basis_offset.append(chosen[row][0])
        else:
            # 人工列按出现次序编号，避免对人工行列表重复线性搜索。
            basis_offset.append(columns+len(artificial_rows))
            artificial_rows.append(row)
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
    crash when it strictly reduces the initial artificial activity; at equal activity it also
    selects crash when the artificial columns avoided exceed the update budget. ``True``/``False``
    force the choice. The equal-activity rule is a measured budget heuristic, not a convergence proof.

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
    # **触发规则来自实测而不是直觉**：优先按初始人工目标严格下降选 crash。
    # 六个实例的对照（docs/research/probe_crash_vs_current.py）显示收益并不一边倒——
    # sct2 由"数值失败、目标 451479"变为目标 68.5，fhnw-binpack4-4 枢轴 948→637，roll3000
    # 目标 12333→10627；但 neos-3381206-awhea 反而从 2035 涨到 2719 枢轴，而它恰好也是
    # 初始不可行量**没有下降**的那一题（1313→1313）；air05 完全无变化（426→426）。
    # 人工目标相等时只在省去的人工基列数超过本次全部换基额度时选 crash：大题 plain 起点
    # 的 106954 个人工基列使前 3000 步全部零步，而 crash 仅需 232 个、1531 步找到可行点；
    # awhea 的 479/4 列虽同目标，但省去 475 小于 5000 额度，继续保留较快的 plain。
    # 这是资源相关启发式而非普遍速度保证；其他实例仍需同预算消融与独立检查。
    # 21 个可构造实例的覆盖面测量（probe_crash_trigger_coverage.py）：18 个触发，
    # 触发案例的不可行量降幅中位数 85.3%，其中 neos-2987310-joes 从 1.566e9 降到 0。
    crash = _crash_start(matrix, lower, upper, placement)
    if phase_one_crash is True and crash is None:
        # 显式消融不能把不可用的 crash 静默替换为 plain，否则结果标签会误导比较。
        raise ValueError('Forced Phase-I logical-first crash is unavailable for this matrix')
    budget = _iteration_budget(kwargs.get('iteration_limit', 20000))
    budget_tie = (crash is not None
                  and abs(crash['activity']-plain_activity) <= 1e-12
                  and rows-len(crash['artificial_rows']) > budget)
    if phase_one_crash is True:
        use_crash = crash is not None
    elif phase_one_crash is False:
        use_crash = False
    else:
        use_crash = crash is not None and (crash['activity'] < plain_activity-1e-12 or budget_tie)
    if crash is None:
        chosen = dict(initial={j: float(placement[j]) for j in range(columns)},
                      basis_offset=None, artificial_rows=list(range(rows)),
                      activity=plain_activity,
                      strategy='plain (no unit column can satisfy its own row inside its bounds)')
    elif use_crash:
        chosen = dict(initial=crash['initial'], basis_offset=crash['basis_offset'],
                      artificial_rows=crash['artificial_rows'], activity=crash['activity'],
                      strategy=('logical-first crash (budget-aware equal artificial activity)'
                                if budget_tie and phase_one_crash is None else
                                'logical-first crash (strictly lower initial artificial activity)')
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
                             feasibility_tolerance=feasibility_tolerance,
                             _phase_one_structural_columns=(columns if artificial else None), **kwargs)
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
    record['feasibility_found_before_auxiliary_optimum'] = status.status == 'PHASE_ONE_FEASIBLE'
    if status.status not in ('OPTIMAL', 'PHASE_ONE_FEASIBLE'):
        if status.status == 'ITERATION_LIMIT' and status.values is not None:
            # 辅助问题尚未证明最优，仍可独立检查其结构部分是否已满足原齐次行和界。
            # 数值异常状态不走此路径，人工目标近零也不单独作为可行依据。
            check = _phase_one_primal_check(wide, matrix, status.values,
                                            wide_lower, wide_upper, feasibility_tolerance)
            record['stopped_candidate_check'] = check
            if check['internal_primal_feasible']:
                record['stopped_feasible_candidate'] = np.asarray(status.values[:columns]).tolist()
        record['reason'] = 'Phase-I did not reach optimality'
        return None, record
    total = float(np.sum(status.values[columns:]))
    record['artificial_sum'] = total
    if (status.status == 'OPTIMAL'
            and total > max(feasibility_tolerance, 1e-9)*(
                1.0+float(np.max(np.abs(status.values), initial=0.0)))):
        record['reason'] = 'Phase-I optimum leaves positive artificial activity'
        return 'INFEASIBLE', record
    # Phase-I 最优时人工变量取零，但可能仍留在基里。必须把它们换成结构列，
    # 否则第二阶段拿不到 m 列的合法基。残留且找不到非零枢轴的行即冗余行。
    #
    # 真正的驱除枢轴是 e_position^T B^{-1} A[:,j]。
    # B 的人工列为 σ_i e_i 只说明 B^{-1}e_i=σ_i e_position（逆矩阵的列），
    # 并不能推出 e_position^T B^{-1}=σ_i e_i^T（逆矩阵的行）。
    # 因此先 BTRAN 得到逆基第 position 行，再乘原结构矩阵并剔除现有基列。
    structural = csc_matrix(matrix)
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
        current = SparseBasis(wide, basis)
        # 清理每次重建的成功LU也是实际成本，含额度耗尽或零枢轴退出前已完成的工作。
        before_update = current.state.refactorisations
        record['refactorisations'] += before_update
        record['cleanup_refactorisations'] += before_update
        selector = np.zeros(rows, dtype=float)
        selector[position] = 1.0
        try:
            multiplier = current.btran(selector)
        except NumericalError as error:
            record['reason'] = f'drive-out transpose basis solve failed: {error}'
            return None, record
        transformed = np.asarray(structural.T @ multiplier, dtype=float).reshape(-1)
        eligible = np.isfinite(transformed) & (np.abs(transformed) > 1e-9)
        eligible[np.asarray([j for j in basis if j < columns], dtype=int)] = False
        if not np.any(eligible):
            record['reason'] = 'row is redundant; artificial variable cannot be driven out'
            record['redundant_basis_position'] = position
            return None, record
        scores = np.where(eligible, np.abs(transformed), -np.inf)
        chosen = int(np.argmax(scores))
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
    check = _phase_one_primal_check(wide, matrix, status.values,
                                    wide_lower, wide_upper, feasibility_tolerance)
    record['original_feasible_check'] = check
    if not check['internal_primal_feasible']:
        record['reason'] = 'Phase-I exit point failed the original structural gates'
        return None, record
    record['basis'] = basis
    record['drive_out'] = drive_out
    # Phase-I 结束时人工变量全为 0（否则已判 INFEASIBLE），因此结构部分就是原系统的一个
    # **已验证可行点**，而且非基变量恰好停在它当时的界上。必须把它交给第二阶段：若第二
    # 阶段按"默认落位"（有下界就放上界）重新摆放非基变量，得到的是另一个点，可能越界
    # （实测随机题上第二阶段起点越界 1.68，直接报 START_INFEASIBLE）。
    record['values'] = [float(v) for v in np.asarray(status.values)[:columns]]
    return 'FEASIBLE', record

