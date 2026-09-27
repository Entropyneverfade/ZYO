# 两阶段单纯形：先用人工变量求可行基（Phase-I），再用已验证的单纯形求最优（Phase-II）。
# 只使用自研 standard_simplex，不调用任何优化引擎；Phase-I 最优值不为零即判不可行。
"""Two-phase simplex for ``min c'x`` s.t. ``Ax = b``, ``x >= 0`` with general ``b``.

Phase-I attaches one artificial variable per row with unit cost and minimises their sum.
The auxiliary problem always has the feasible start ``(x=0, artificials=b)`` when every
right-hand side is non-negative, so the logical basis is used directly. A Phase-I optimum
with a positive objective proves the original system infeasible; a zero objective yields a
feasible basis for Phase-II after any artificial columns still basic are pivoted out.

Nothing is assumed: the Phase-I objective is compared against an explicit tolerance that
scales with the data, artificial drive-out failures are reported rather than patched, and
the returned point is re-checked against the ORIGINAL ``A x = b`` and ``x >= 0``.

Reduction strategy for negative right-hand sides: a row with ``b_i < 0`` is negated (which
is an exact equivalence for an equality row) so the logical start stays feasible. When the
caller supplies logical unit columns for the rows that were *not* negated, that mixed basis is
still the identity and is reused; the choice is validated by an explicit identity check, so a
feasible start is constructed rather than assumed.
"""
from dataclasses import dataclass, field
import math

import numpy as np

from .standard_simplex import standard_simplex


@dataclass
class TwoPhaseResult:
    status: str
    values: np.ndarray | None = None
    objective: float | None = None
    basic: list = field(default_factory=list)
    phase_one: dict = field(default_factory=dict)
    message: str = ''
    max_row_residual: float = math.inf
    max_negativity: float = math.inf


def two_phase_simplex(matrix, costs, rhs, *, basic=None, iteration_limit=2000,
                      artificial_tolerance=None):
    """Solve a standard-form LP from a possibly infeasible logical start.

    ``basic`` optionally names the initial basis columns; when omitted the last ``m``
    columns are used, which is the usual slack/artificial layout.
    """
    matrix = np.asarray(matrix, dtype=float)
    cost = np.asarray(costs, dtype=float).reshape(-1)
    right = np.asarray(rhs, dtype=float).reshape(-1)
    rows, columns = matrix.shape
    if cost.size != columns or right.size != rows:
        return TwoPhaseResult('NUMERICAL_ERROR',
                              message='cost or right-hand side length mismatch')
    # 负右端行取反：对等式行是精确等价，使人工初值 |b| 非负。
    # 取反是就地写入，因此必须先复制，不能改到调用方的矩阵（实测会把归约结果污染）。
    matrix = np.array(matrix, dtype=float, copy=True)
    right = np.array(right, dtype=float, copy=True)
    flipped = []
    for row in range(rows):
        if right[row] < 0:
            matrix[row, :] *= -1.0
            right[row] *= -1.0
            flipped.append(row)
    flipped_set = set(flipped)
    scale = max(1.0, float(np.max(np.abs(right), initial=0.0)))
    if artificial_tolerance is None:
        artificial_tolerance = 1e-7*scale

    # ---- Phase-I：人工变量一行一个，目标为它们的和 ----
    artificial = np.eye(rows)
    auxiliary = np.hstack([matrix, artificial])
    auxiliary_cost = np.concatenate([np.zeros(columns), np.ones(rows)])
    start = [columns+row for row in range(rows)]
    # 若调用方给出的逻辑列在**未被取反**的行上恰好构成单位阵，就直接复用它作为初始基：
    # 此时基矩阵是 I、x_B = rhs >= 0，天然可行，可以少引入一批人工变量。
    # 只有通过单位阵校验才复用，这样"初始基可行且非奇异"是**构造性**保证，而不是假设。
    if basic is not None and len(basic) == rows:
        candidate = list(start)
        for row in range(rows):
            column = int(basic[row])
            if row in flipped_set or not 0 <= column < columns:
                continue
            candidate[row] = column
        if np.allclose(auxiliary[:, candidate], np.eye(rows), atol=1e-9):
            start = candidate
    phase_one = standard_simplex(auxiliary, auxiliary_cost, start, right,
                                 iteration_limit=iteration_limit)
    record = dict(status=phase_one.status, objective=phase_one.objective,
                  pivots=len(phase_one.pivots), artificial_columns=list(start),
                  tolerance=artificial_tolerance, flipped_rows=flipped)
    if phase_one.status != 'OPTIMAL' or phase_one.values is None:
        return TwoPhaseResult(phase_one.status, phase_one=record,
                              message='Phase-I did not reach optimality: '+phase_one.message)
    artificial_sum = float(np.sum(phase_one.values[columns:]))
    record['artificial_sum'] = artificial_sum
    if artificial_sum > artificial_tolerance:
        return TwoPhaseResult('INFEASIBLE', phase_one=record,
                              message=f'Phase-I optimum has positive artificial activity '
                                      f'{artificial_sum:.3e}; the rows are infeasible')

    # ---- 驱除基中残留的人工变量，得到第二阶段可用的基 ----
    basis = list(phase_one.basic)
    drive_out = []
    guard = 0
    while True:
        guard += 1
        if guard > rows+columns+10:
            return TwoPhaseResult('NUMERICAL_ERROR', phase_one=record,
                                  message='artificial drive-out did not terminate')
        position = next((k for k, j in enumerate(basis) if j >= columns), None)
        if position is None:
            break
        artificial_column = basis[position]
        # 该行在结构列上的系数：B^{-T} e_(artificial) 给出 B^{-1} 对应行。
        square = auxiliary[:, basis]
        unit = np.zeros(rows)
        unit[artificial_column-columns] = 1.0
        try:
            row_vector = np.linalg.solve(square.T, unit)
        except np.linalg.LinAlgError:
            return TwoPhaseResult('NUMERICAL_ERROR', phase_one=record,
                                  message='basis became singular during artificial drive-out')
        chosen = None
        for j in range(columns):
            if j in basis:
                continue
            if abs(row_vector[j]) > 1e-9:
                chosen = j
                break
        if chosen is None:
            record['redundant_row_position'] = position
            return TwoPhaseResult('NUMERICAL_ERROR', phase_one=record,
                                  message='a row is redundant: the artificial variable cannot '
                                          'be driven out; drop the row explicitly')
        # 用 chosen 替换该人工变量，并保持基矩阵非奇异。
        trial = list(basis)
        trial[position] = chosen
        candidate_square = auxiliary[:, trial]
        if abs(np.linalg.det(candidate_square)) <= 1e-12*max(
                1.0, float(np.max(np.abs(candidate_square), initial=0.0))):
            # 该列无法在此位置形成枢轴，换下一列。
            found = False
            for alternative in range(columns):
                if alternative in basis:
                    continue
                trial[position] = alternative
                candidate_square = auxiliary[:, trial]
                if abs(np.linalg.det(candidate_square)) > 1e-12*max(
                        1.0, float(np.max(np.abs(candidate_square), initial=0.0))):
                    chosen, found = alternative, True
                    break
            if not found:
                record['redundant_row_position'] = position
                return TwoPhaseResult('NUMERICAL_ERROR', phase_one=record,
                                      message='a row is redundant: no structural column can '
                                              'replace the artificial variable')
        drive_out.append(dict(position=position, artificial=int(artificial_column),
                              replaced_by=int(chosen)))
        basis = trial
    record['drive_out'] = drive_out
    record['phase_two_basis'] = list(basis)

    # ---- Phase-II：在原问题上从可行基求最优 ----
    outcome = standard_simplex(matrix, cost, basis, right, iteration_limit=iteration_limit)
    residual = float(np.max(np.abs(matrix @ outcome.values-right), initial=0.0)) \
        if outcome.values is not None else math.inf
    negativity = float(max(0.0, -np.min(outcome.values))) if outcome.values is not None else math.inf
    result = TwoPhaseResult(outcome.status, values=outcome.values, objective=outcome.objective,
                            basic=list(outcome.basic), phase_one=record,
                            message=outcome.message, max_row_residual=residual,
                            max_negativity=negativity)
    if outcome.status == 'OPTIMAL' and (residual > 1e-7 or negativity > 1e-7):
        result.status = 'NUMERICAL_ERROR'
        result.message = (f'Phase-II claims optimality but the point violates the original '
                          f'system (residual {residual:.3e}, negativity {negativity:.3e})')
    return result
