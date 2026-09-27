# 稠密参考单纯形：只用于交叉验证稀疏驱动，不进入生产求解路径。
# 目的不是性能，而是提供一个可逐步对照的"可信参照"：基、方向、步长与简约成本。
"""Dense reference bounded-variable primal simplex.

This module exists so the sparse driver can be compared pivot by pivot against an
implementation whose linear algebra is trivially checkable. It is deliberately dense
(``O(m^2)`` storage) and must never be used for production solves.

Form: ``min c'x`` subject to ``Ax = 0`` and ``l <= x <= u`` (Huangfu & Hall §2.1).
Every nonbasic variable sits exactly on one of its bounds; that invariant is enforced,
not assumed, and is the property the sparse driver must reproduce.
"""
from dataclasses import dataclass, field
import math

import numpy as np


@dataclass
class DenseSimplexResult:
    status: str
    values: np.ndarray | None = None
    objective: float | None = None
    basic: list = field(default_factory=list)
    reduced_costs: np.ndarray | None = None
    dual: np.ndarray | None = None
    iterations: int = 0
    pivots: int = 0
    traffic: list = field(default_factory=list)   # 逐枢轴记录，供与稀疏驱动逐步对照
    message: str = ''
    max_primal_violation: float = 0.0


def dense_simplex(matrix, costs, lower, upper, *, basic, rhs=None, maximize=False,
                  iteration_limit=10000, tolerance=1e-7, initial=None, debug=False):
    """Bounded-variable primal simplex with Bland's rule.

    ``basic`` must be a valid basis (a list of column indices). The constraint system is
    ``A x = rhs`` (``rhs`` defaults to zero, the homogeneous Huangfu & Hall form). The
    routine keeps the invariant that each nonbasic variable equals one of its bounds
    exactly, and returns an explicit status instead of a best-effort point.

    ``initial`` optionally gives starting values for the basic variables; it must describe
    a primal FEASIBLE basis, because this routine performs no Phase-I of its own and will
    report rather than repair an infeasible start.
    """
    matrix = np.asarray(matrix, dtype=float)
    rows, columns = matrix.shape
    cost = np.asarray(costs, dtype=float).reshape(-1).copy()
    lo = np.asarray(lower, dtype=float).reshape(-1).copy()
    hi = np.asarray(upper, dtype=float).reshape(-1).copy()
    right = np.zeros(rows) if rhs is None else np.asarray(rhs, dtype=float).reshape(-1).copy()
    if right.size != rows:
        return DenseSimplexResult('INFEASIBLE', message='Right-hand side has the wrong length')
    if maximize:
        cost = -cost
    if len(basic) != rows:
        return DenseSimplexResult('INFEASIBLE', message='Basis size does not match the row count')
    basis = [int(j) for j in basic]
    if len(set(basis)) != rows:
        return DenseSimplexResult('INFEASIBLE', message='Basis contains duplicate columns')

    values = np.zeros(columns)
    at_upper = np.zeros(columns, dtype=bool)
    in_basis = np.zeros(columns, dtype=bool)
    in_basis[basis] = True
    # 显式给定的基变量初值先落位，因为非基变量的"停在哪一侧界"要由它的实际值判断。
    if initial is not None:
        for index, value in dict(initial).items():
            values[int(index)] = float(value)
    for j in range(columns):
        if in_basis[j]:
            continue
        if not math.isfinite(lo[j]) and not math.isfinite(hi[j]):
            return DenseSimplexResult(
                'NUMERICAL_ERROR',
                message=f'Variable {j} is free and cannot be a nonbasic variable')
        if initial is not None and j in dict(initial):
            # 调用方给了非基变量的初值：按它究竟等于哪一侧界来判定所在侧，
            # 不能用"哪侧界有限"猜（猜错会让方向反向，产生零步长翻界死循环）。
            value = values[j]
            at_lower = math.isfinite(lo[j]) and abs(value-lo[j]) <= 1e-12
            at_upper[j] = bool(math.isfinite(hi[j]) and (not at_lower) and abs(value-hi[j]) <= 1e-12)
            if not at_lower and not at_upper[j]:
                return DenseSimplexResult(
                    'NUMERICAL_ERROR',
                    message=f'Nonbasic variable {j} starts strictly inside its bounds '
                            f'({lo[j]}, {hi[j]}); a bounded-variable basis requires a bound')
        elif math.isfinite(lo[j]):
            values[j], at_upper[j] = lo[j], False
        else:
            values[j], at_upper[j] = hi[j], True

    # 起始可行性先验证再迭代：本函数不做 Phase-I，从不可行点出发的比例检验没有意义。
    # 基变量值按 x_B = B^{-1}(rhs - N x_N) 计算（调用方给的基变量分量只作参考）。
    start = values.copy()
    if len(basis) == rows:
        try:
            nonbasic = [j for j in range(columns) if not in_basis[j]]
            start[basis] = _solve(_factor(matrix[:, basis]),
                                  right - matrix[:, nonbasic] @ start[nonbasic])
        except np.linalg.LinAlgError:
            return DenseSimplexResult('NUMERICAL_ERROR', message='Basis is singular')
    start_violation = _violation(start, lo, hi)
    start_residual = float(np.max(np.abs(matrix @ start-right), initial=0.0))
    start_scale = 1.0+float(np.max(np.abs(start), initial=0.0))
    if start_violation > tolerance or start_residual > tolerance*start_scale:
        return DenseSimplexResult(
            'START_INFEASIBLE', values=start, basic=list(basis),
            message=(f'the starting basis is not primal feasible (bound violation '
                     f'{start_violation:.3e}, |A x - rhs| {start_residual:.3e}); this routine '
                     f'does not repair an infeasible start'),
            max_primal_violation=max(start_violation, start_residual))

    traffic = []
    blocked = -1   # 上一轮刚翻界的变量：本轮定价跳过它，避免零步长两点循环
    for iteration in range(iteration_limit+1):
        matrix_b = matrix[:, basis]
        try:
            lu = _factor(matrix_b)
        except np.linalg.LinAlgError:
            return DenseSimplexResult('NUMERICAL_ERROR', message='Basis became singular')
        nonbasic = [j for j in range(columns) if not in_basis[j]]
        # 基变量值**增量维护**：每次枢轴已按 x := x + step*moving 更新过，用
        # x_B = B^{-1}(rhs - N x_N) 重算会把刚做的移动抹掉（实测 values 恒等于初值、
        # 目标恒定不变）。仅在缺少可行初值的首轮由方程建立基值。
        if iteration == 0 and initial is None:
            residual = right - matrix[:, nonbasic] @ values[nonbasic]
            values[basis] = _solve(lu, residual)
        if debug:
            equation_residual = float(np.max(np.abs(matrix @ values-right), initial=0.0))
            if equation_residual > 1e-6:
                print(f'[dr] it{iteration} equation residual {equation_residual:.3e}')
        dual = _solve(lu, cost[basis], transpose=True)
        reduced = cost - matrix.T @ dual

        # Bland 规则：取最小下标且能改进的变量，保证退化下有限终止。
        # 刚发生过翻界的变量本轮不再参与定价：它已经贴在新的一侧界上，立刻再次入选会
        # 产生零步长的两点循环；跳过它让后续变量取得进展。
        if debug:
            print(f'[dr] it{iteration} basis={list(basis)} nonbasic={nonbasic} '
                  f'values={np.round(values, 6)} at_upper={at_upper.astype(int)}')
        entering = -1
        for j in nonbasic:
            if j == blocked:
                continue
            if at_upper[j]:
                if reduced[j] > tolerance:
                    entering = j
                    break
            elif reduced[j] < -tolerance:
                entering = j
                break
        violation = _violation(values, lo, hi)
        residual_norm = float(np.max(np.abs(matrix @ values-right), initial=0.0))
        if entering < 0:
            if max(violation, residual_norm) <= 1e-7:
                return DenseSimplexResult(
                    'OPTIMAL', values=values,
                    objective=float(((-cost) if maximize else cost) @ values),
                    basic=list(basis), reduced_costs=reduced, dual=dual,
                    iterations=iteration, pivots=len(traffic), traffic=traffic,
                    message='Reduced costs proved optimality', max_primal_violation=violation)
            return DenseSimplexResult('NUMERICAL_ERROR', values=values, basic=list(basis),
                                      iterations=iteration, pivots=len(traffic), traffic=traffic,
                                      message=(f'No entering column but violation {violation:.3e} / '
                                               f'residual {residual_norm:.3e} remains'),
                                      max_primal_violation=max(violation, residual_norm))

        direction_plus = _solve(lu, matrix[:, entering])
        sign = -1.0 if at_upper[entering] else 1.0
        moving = sign*direction_plus

        # 比例检验：判据是**方向是否朝向该界**，而不是"变量是否已经在另一侧界上"。
        # 符号约定由方程 A x = rhs 唯一确定：x_B = B^{-1}(rhs − N x_N)，所以入基变量沿
        # 自身可行方向走 α 时 x_B(α) = x_B(0) − moving·α。令 d = −moving：
        #   d_j > 0 -> 基变量 j 上升，只有**上界**能限制步长；
        #   d_j < 0 -> 基变量 j 下降，只有**下界**能限制步长。
        # 早期版本用 d = +moving 配 `values += step*moving`，两处符号互相抵消、
        # 看起来自洽，但整体与 A x = rhs 相反：手算题 x+y<=4, x+3y<=6 上它把
        # s0 = x+y 判成"下降且无下界"，直接误报 UNBOUNDED（实测）。
        # 变量恰好停在目标界上时 room = 0，这是退化枢轴（步长为零），不是"无界"。
        step = math.inf
        limiting = -1
        limiting_upper = False
        for position, j in enumerate(basis):
            d = -moving[position]
            if d > tolerance and math.isfinite(hi[j]):
                room = (hi[j]-values[j])/d
                if room < step-1e-12:
                    step, limiting, limiting_upper = room, position, True
            elif d < -tolerance and math.isfinite(lo[j]):
                room = (lo[j]-values[j])/d
                if room < step-1e-12:
                    step, limiting, limiting_upper = room, position, False
        # 入基变量自身的另一侧界；先触到即翻界，不换基。
        flip_room = math.inf
        if sign > 0 and math.isfinite(hi[entering]):
            flip_room = hi[entering]-values[entering]
        elif sign < 0 and math.isfinite(lo[entering]):
            flip_room = values[entering]-lo[entering]
        if not math.isfinite(step) and not math.isfinite(flip_room):
            if debug:
                print(f'[dr] UNBOUNDED at it{iteration}: entering={entering} sign={sign} '
                      f'moving={np.round(moving, 6)} values={np.round(values, 6)} '
                      f'lo={lo} hi={hi}')
            return DenseSimplexResult('UNBOUNDED', values=values, basic=list(basis),
                                      iterations=iteration, pivots=len(traffic), traffic=traffic,
                                      message='Objective improves without limit')
        flip = flip_room < step-1e-12
        step = max(0.0, min(step, flip_room))

        # 所有变量（含入基变量）走一步。基本变量按 x_B(α) = x_B(0) − moving·α 减去移动
        # 量；这正是 A x = rhs 要求的补偿方向（写成 `+=` 会让 |A x| 立刻变成 8）。
        # 换出变量落在 position=limiting 上，`−moving·step` 恰好把它送到刚触到的界：
        # step = (hi−x)/d 且 d = −moving，故 x + d·step = hi。
        values[entering] += sign*step
        for position, j in enumerate(basis):
            values[j] -= step*moving[position]
        traffic.append(dict(iteration=iteration, entering=int(entering), sign=float(sign),
                            step=float(step), limiting=int(limiting), flip=bool(flip),
                            reduced=float(reduced[entering]), pivot_column=direction_plus.copy(),
                            objective=float(cost @ values)))
        if flip:
            # 翻界后该变量改停另一侧界，基不变；本轮之后跳过它一次。
            if sign > 0:
                values[entering] = hi[entering]
                at_upper[entering] = True
            else:
                values[entering] = lo[entering]
                at_upper[entering] = False
            blocked = entering
            continue

        leaving = basis[limiting]
        # 换出变量精确落在它触到的那个界上，并据此记录所在侧。
        values[leaving] = hi[leaving] if limiting_upper else lo[leaving]
        at_upper[leaving] = limiting_upper
        at_upper[entering] = False
        in_basis[leaving] = False
        in_basis[entering] = True
        basis[limiting] = entering
        blocked = -1
    return DenseSimplexResult('ITERATION_LIMIT', values=values, basic=list(basis),
                              iterations=iteration_limit, pivots=len(traffic), traffic=traffic,
                              message='Iteration limit reached')


def _factor(matrix):
    from scipy.linalg import lu_factor
    return lu_factor(matrix)


def _solve(lu, rhs, *, transpose=False):
    from scipy.linalg import lu_solve
    return lu_solve(lu, rhs, trans=1 if transpose else 0)


def _violation(values, lo, hi):
    worst = 0.0
    for index in range(values.size):
        if math.isfinite(lo[index]):
            worst = max(worst, lo[index]-values[index])
        if math.isfinite(hi[index]):
            worst = max(worst, values[index]-hi[index])
    return float(max(0.0, worst))
