# 标准形单纯形（教科书形式，唯一约定）：min c'x s.t. Ax=b, x>=0。
# 只做最基础的一步：非基变量均为 0，比例检验取最先降到 0 的基本变量，Bland 规则定价。
"""Textbook standard-form simplex: ``min c'x`` s.t. ``Ax = b``, ``x >= 0``.

Deliberately minimal and unambiguous. There are no variable upper bounds here, so the only
complication is the ratio test ``alpha = min{x_B[i] / (-d_B[i]) : d_B[i] < 0}``. Callers must
supply a primal-feasible basis (``x_B = B^{-1}b >= 0``); the routine performs no Phase-I and
reports an infeasible start instead of repairing it. Bland's rule guarantees termination.
"""
from dataclasses import dataclass, field
import math

import numpy as np


@dataclass
class StdResult:
    status: str
    values: np.ndarray | None = None
    objective: float | None = None
    basic: list = field(default_factory=list)
    pivots: list = field(default_factory=list)
    message: str = ''


def standard_simplex(matrix, costs, basic, rhs, *, iteration_limit=1000, tolerance=1e-9):
    """Solve from a primal-feasible basis of a standard-form LP."""
    matrix = np.asarray(matrix, dtype=float)
    cost = np.asarray(costs, dtype=float).reshape(-1)
    right = np.asarray(rhs, dtype=float).reshape(-1)
    rows, columns = matrix.shape
    if cost.size != columns:
        return StdResult('NUMERICAL_ERROR',
                         message=f'cost vector has {cost.size} entries but the matrix has '
                                 f'{columns} columns')
    if len(basic) != rows or right.size != rows:
        return StdResult('INFEASIBLE', message='basis size or rhs length mismatch')
    basis = [int(j) for j in basic]
    values = np.zeros(columns)
    in_basis = np.zeros(columns, dtype=bool)
    in_basis[basis] = True

    pivots = []
    for _ in range(iteration_limit):
        square = matrix[:, basis]
        # scipy 的 lu_factor 对奇异矩阵可能不抛异常而返回含 0 主元的分解，
        # 因此必须显式检查 U 的对角线，不能依赖异常。
        try:
            lu = _factor(square)
        except Exception:  # noqa: BLE001
            return StdResult('NUMERICAL_ERROR', values=values, basic=list(basis), pivots=pivots,
                             message='basis is singular')
        diagonal = np.abs(np.diag(lu[0]))
        scale = max(1.0, float(np.max(np.abs(square), initial=0.0)))
        # 注意：np.min(arr, initial=0.0) 会把最小值钳到 0，导致一切都被判为奇异；
        # 空数组的情形已被上面的尺寸检查排除，这里直接用数组自身的最小值。
        if diagonal.size == 0 or float(np.min(diagonal)) <= 1e-12*scale:
            return StdResult('NUMERICAL_ERROR', values=values, basic=list(basis), pivots=pivots,
                             message='basis is singular (zero pivot in the LU factorization)')
        basic_values = _solve(lu, right)
        if np.min(basic_values) < -1e-7:
            return StdResult('INFEASIBLE', basic=list(basis), pivots=pivots,
                             message=f'supplied basis is not primal feasible '
                                     f'(min x_B = {np.min(basic_values):.3e})')
        values[:] = 0.0
        for position, j in enumerate(basis):
            values[j] = basic_values[position]
        dual = _solve(lu, cost[basis], transpose=True)
        reduced = cost - matrix.T @ dual

        entering = -1
        for j in range(columns):
            if in_basis[j]:
                continue
            if reduced[j] < -tolerance:      # Bland：取最小下标
                entering = j
                break
        if entering < 0:
            return StdResult('OPTIMAL', values=values, objective=float(cost @ values),
                             basic=list(basis), pivots=pivots,
                             message='all reduced costs nonnegative for this basis')

        direction = _solve(lu, matrix[:, entering])
        # x_B(alpha) = B^{-1}(b - a*alpha) = x_B(0) - d*alpha，故 x_e 增大时 x_B 按 -d 变化。
        # 只有当 -d[i] < 0（即 d[i] > 0）时该基本变量才会下降并可能先触零，
        # 此时 alpha = x_B[i]/d[i]。写成 d[i] < 0 会把方向判反、一步都走不动。
        step = math.inf
        leaving_position = -1
        for position in range(rows):
            if direction[position] > tolerance:
                room = basic_values[position]/direction[position]
                if room < step-1e-12 or (abs(room-step) <= 1e-12 and leaving_position >= 0
                                         and basis[position] < basis[leaving_position]):
                    step, leaving_position = room, position
        if leaving_position < 0:
            return StdResult('UNBOUNDED', values=values, basic=list(basis), pivots=pivots,
                             message='no basic variable decreases; the objective is unbounded')
        step = max(0.0, step)
        # 推进：基变量按方向走，入基变量取步长（非基变量起始为 0）。
        basic_values = basic_values - step*direction
        values[:] = 0.0
        for position, j in enumerate(basis):
            values[j] = basic_values[position]
        values[entering] = step
        pivots.append(dict(entering=int(entering), step=float(step),
                           leaving_position=int(leaving_position),
                           reduced=float(reduced[entering]), objective=float(cost @ values)))
        leaving = basis[leaving_position]
        in_basis[leaving] = False
        in_basis[entering] = True
        basis[leaving_position] = entering
    return StdResult('ITERATION_LIMIT', basic=list(basis), pivots=pivots,
                     message='iteration limit reached')


def _factor(matrix):
    from scipy.linalg import lu_factor
    return lu_factor(matrix)


def _solve(lu, rhs, *, transpose=False):
    from scipy.linalg import lu_solve
    return lu_solve(lu, rhs, trans=1 if transpose else 0)
