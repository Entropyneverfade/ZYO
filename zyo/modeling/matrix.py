# 线性模型矩阵装配：保留原变量、行关系、目标常数及 MW/MWh 等调用方单位。
# 默认稀疏装配只使用 SciPy 的矩阵容器，不导入 scipy.optimize 或任何外部求解器。
"""Convert a ZYO linear model to dense reference or sparse solver arrays."""

import math

import numpy as np
from scipy.sparse import csc_matrix


# 仅用于显式稠密参考入口，防止行×列内存分配抢先导致进程无记录退出。
DEFAULT_DENSE_ENTRY_GATE = 2_000_000


def _vectors(model):
    """从原模型一次提取目标、界、右端项与行关系，避免两种表示口径漂移。"""
    columns = len(model.variables)
    rows = len(model.constraints)
    rhs = np.zeros(rows)
    sense = []
    for index, row in enumerate(model.constraints):
        rhs[index] = -row.expression.constant
        sense.append(row.sense)
    cost = np.zeros(columns)
    for index, coefficient in model.objective.terms.items():
        cost[int(index)] = float(coefficient)
    lower = np.array([v.lb if math.isfinite(v.lb) else -math.inf for v in model.variables])
    upper = np.array([v.ub if math.isfinite(v.ub) else math.inf for v in model.variables])
    return cost, lower, upper, rhs, sense, float(model.objective.constant)


def model_arrays(model, max_dense_entries=DEFAULT_DENSE_ENTRY_GATE):
    """显式稠密参考装配；调用前先按行×列检查分配规模。"""
    columns = len(model.variables)
    rows = len(model.constraints)
    if max_dense_entries is not None and rows*columns > max_dense_entries:
        raise MemoryError(
            f'the dense tableau needs {rows*columns} entries ({rows*columns*8/1e9:.2f} GB); '
            f'the predeclared gate is {max_dense_entries} entries '
            f'({max_dense_entries*8/1e6:.1f} MB)')
    matrix = np.zeros((rows, columns))
    for index, row in enumerate(model.constraints):
        for j, coefficient in row.expression.terms.items():
            matrix[index, int(j)] = float(coefficient)
    return (matrix, *_vectors(model))


def model_sparse_arrays(model):
    """按实际非零元装配 CSC；返回值与显式稠密参考入口同序。"""
    columns = len(model.variables)
    rows = len(model.constraints)
    row_index, column_index, entries = [], [], []
    for index, row in enumerate(model.constraints):
        for j, coefficient in row.expression.terms.items():
            row_index.append(index)
            column_index.append(int(j))
            entries.append(float(coefficient))
    # COO 坐标形式便于逐行装配；CSC 保留稀疏列运算能力，且重复坐标按 SciPy 规则求和。
    matrix = csc_matrix((np.asarray(entries, dtype=float),
                         (np.asarray(row_index, dtype=np.int64),
                          np.asarray(column_index, dtype=np.int64))),
                        shape=(rows, columns))
    return (matrix, *_vectors(model))
