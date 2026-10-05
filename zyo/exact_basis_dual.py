"""小规模候选基的事后精确行乘子见证，不参与原生单纯形迭代。"""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
from scipy.sparse import csc_matrix


MAX_ROWS = 8
MAX_COLUMNS = 64
MAX_BITS = 4096


@dataclass(frozen=True)
class ExactBasisDualWitness:
    """只有 ``multipliers`` 非空时才表示精确满足所给基列方程。"""

    multipliers: tuple[Fraction, ...] | None
    refusal_reason: str | None
    max_numerator_bits: int = 0
    max_denominator_bits: int = 0


class _BitBudgetExceeded(Exception):
    """内部有理数超过事后见证的固定资源预算。"""


def reconstruct_exact_basis_dual(matrix, costs, basic, *, max_bits=MAX_BITS):
    """以输入 binary64 的精确值求 ``B.T @ y = c_B``，失败时只返回原因。

    仅允许至多 8 行、64 列；见证既不是原 LP 的完整 KKT，也不改变求解状态。
    此处的精确值是已存储的 binary64，而非原始 MPS 十进制文字。
    """
    shape = getattr(matrix, 'shape', None)
    if not isinstance(shape, tuple) or len(shape) != 2:
        return ExactBasisDualWitness(None, 'invalid_shape')
    rows, columns = shape
    if rows < 1 or columns < rows:
        return ExactBasisDualWitness(None, 'invalid_shape')
    # 先查形状，避免意外把大型题稠密化或复制；此帮助器不承担一般大基分解。
    if rows > MAX_ROWS:
        return ExactBasisDualWitness(None, 'rows_out_of_scope')
    if columns > MAX_COLUMNS:
        return ExactBasisDualWitness(None, 'columns_out_of_scope')
    if isinstance(max_bits, bool) or not isinstance(max_bits, int) or not 1 <= max_bits <= MAX_BITS:
        return ExactBasisDualWitness(None, 'invalid_budget')
    try:
        basic_columns = list(basic)
    except TypeError:
        return ExactBasisDualWitness(None, 'invalid_basis')
    if (len(basic_columns) != rows
            or any(isinstance(index, (bool, np.bool_))
                   or not isinstance(index, (int, np.integer))
                   or index < 0 or index >= columns for index in basic_columns)
            or len(set(basic_columns)) != rows):
        return ExactBasisDualWitness(None, 'invalid_basis')
    try:
        raw_cost = np.asarray(costs)
        if np.iscomplexobj(raw_cost) or np.iscomplexobj(matrix):
            return ExactBasisDualWitness(None, 'nonfinite_input')
        cost = np.asarray(costs, dtype=float).reshape(-1)
        if cost.size != columns:
            return ExactBasisDualWitness(None, 'invalid_shape')
        body = csc_matrix(matrix, dtype=float).copy()
        body.sum_duplicates()
    except (TypeError, ValueError, OverflowError):
        return ExactBasisDualWitness(None, 'nonfinite_input')
    if not np.all(np.isfinite(body.data)) or not np.all(np.isfinite(cost)):
        return ExactBasisDualWitness(None, 'nonfinite_input')

    max_num = 0
    max_den = 0

    def checked(value):
        nonlocal max_num, max_den
        num_bits = value.numerator.bit_length()
        den_bits = value.denominator.bit_length()
        max_num = max(max_num, num_bits)
        max_den = max(max_den, den_bits)
        if num_bits > max_bits or den_bits > max_bits:
            raise _BitBudgetExceeded
        return value

    def refused(reason):
        return ExactBasisDualWitness(None, reason, max_num, max_den)

    try:
        # 按 CSC 的所给基本列构成 B^T；每个非零系数用其 binary64 精确有理值。
        transposed = [[Fraction(0) for _ in range(rows)] for _ in range(rows)]
        for position, column in enumerate(basic_columns):
            for pointer in range(body.indptr[column], body.indptr[column+1]):
                row = int(body.indices[pointer])
                transposed[position][row] = checked(
                    Fraction.from_float(float(body.data[pointer])))
        rhs = [checked(Fraction.from_float(float(cost[column])))
               for column in basic_columns]
        original = [line[:] for line in transposed]
        # 小基才允许逐项有理消元；中途每一步均检查分子/分母位长。
        for pivot_index in range(rows):
            pivot_row = next((row for row in range(pivot_index, rows)
                              if transposed[row][pivot_index]), None)
            if pivot_row is None:
                return refused('singular_basis')
            if pivot_row != pivot_index:
                transposed[pivot_index], transposed[pivot_row] = (
                    transposed[pivot_row], transposed[pivot_index])
                rhs[pivot_index], rhs[pivot_row] = rhs[pivot_row], rhs[pivot_index]
            pivot = transposed[pivot_index][pivot_index]
            for row in range(pivot_index+1, rows):
                if not transposed[row][pivot_index]:
                    continue
                factor = checked(transposed[row][pivot_index]/pivot)
                transposed[row][pivot_index] = Fraction(0)
                for column in range(pivot_index+1, rows):
                    transposed[row][column] = checked(
                        transposed[row][column]-factor*transposed[pivot_index][column])
                rhs[row] = checked(rhs[row]-factor*rhs[pivot_index])
        solution = [Fraction(0) for _ in range(rows)]
        for row in range(rows-1, -1, -1):
            tail = sum((transposed[row][column]*solution[column]
                        for column in range(row+1, rows)), Fraction(0))
            solution[row] = checked((rhs[row]-tail)/transposed[row][row])
        if any(sum((coefficient*value for coefficient, value in zip(line, solution)),
                   Fraction(0)) != target
               for line, target in zip(original,
                                       [Fraction.from_float(float(cost[column]))
                                        for column in basic_columns])):
            return refused('exact_check_failed')
    except _BitBudgetExceeded:
        return refused('bit_budget')
    return ExactBasisDualWitness(tuple(solution), None, max_num, max_den)
