# 自主密集两阶段单纯形：仅使用 NumPy；规模保护、数值失败和时间限制均保留真实状态。
"""Independent NumPy-only two-phase tableau simplex. No external solver calls.

This dense, cold-start implementation favors inspection over performance.
No cuts, presolve, revised simplex, warm starts or exact certificates.
"""
from dataclasses import dataclass
import time
import numpy as np


@dataclass
class LPResult:
    status: str
    x: object = None
    objective: object = None
    iterations: int = 0
    message: str = ""


def _leaving_row(column, rhs, basis, pivot_tol, feasibility_tol):
    # 固定比值绝对误差无法约束松弛误差：1e-12 步长乘 1e13 列系数即可越界 10。
    # 仅在实际最小非负比值相同的行间择优，等步长优先大主元，最后按基编号确定顺序。
    leaving=np.flatnonzero(column>pivot_tol)
    if not len(leaving): return None
    ratios=np.maximum(0,rhs[leaving])/column[leaving]
    ties=leaving[ratios==np.min(ratios)]
    for row in sorted((int(i) for i in ties),key=lambda i:(-column[i],basis[i])):
        # 比值检验容许微负RHS，但消元使用真实RHS；负的实际步长可能被其他列系数放大。
        # 逐候选检查真实换基后RHS，不裁剪、不改方程；没有安全主元时保留数值失败。
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            step=rhs[row]/column[row]
            predicted=rhs-column*step
        predicted[row]=step
        if np.all(np.isfinite(predicted)) and np.min(predicted)>=-feasibility_tol:
            return row
    return None


def relaxation(model, lower, upper, options, deadline):
    # 求解当前分支节点的 LP 松弛；lower/upper 是节点边界，不能被全局变量边界替代。
    n = len(lower)
    if np.any(lower > upper):
        return LPResult("INFEASIBLE", message="Contradictory variable bounds")
    # 变量变换 x = offset + Tz、z>=0：有限下界平移，仅上界反向，自由变量拆成正负两项。
    offset = np.zeros(n)
    transforms = [[] for _ in range(n)]
    width = 0
    bound_rows = []
    for i, (lo, hi) in enumerate(zip(lower, upper)):
        if np.isfinite(lo):
            offset[i] = lo
            if hi == lo:
                continue
            transforms[i] = [(width, 1.0)]
            if np.isfinite(hi):
                bound_rows.append((width, hi - lo))
            width += 1
        elif np.isfinite(hi):
            offset[i] = hi
            transforms[i] = [(width, -1.0)]
            width += 1
        else:
            transforms[i] = [(width, 1.0), (width + 1, -1.0)]
            width += 2
    # 分配密集单纯形表之前做规模保护，估算包含人工变量；触发保护不表示模型不可行。
    m_est = len(model.constraints) + len(bound_rows)
    if (m_est + 1) * (width + 2 * m_est + 1) > options.max_tableau_cells:
        return LPResult("SIZE_LIMIT", message="Native dense tableau size limit; select engine='highs'")
    rows, rhs, senses = [], [], []
    for constraint in model.constraints:
        row = np.zeros(width)
        b = -constraint.expression.constant
        for i, a in constraint.expression.terms.items():
            b -= a * offset[i]
            for j, sign in transforms[i]:
                row[j] += a * sign
        rows.append(row)
        rhs.append(b)
        senses.append(constraint.sense)
    for j, b in bound_rows:
        row = np.zeros(width)
        row[j] = 1
        rows.append(row)
        rhs.append(b)
        senses.append("<=")
    kept_rows, kept_rhs, kept_senses = [], [], []
    for a, b, sense in zip(rows, rhs, senses):
        scale = float(np.max(np.abs(a))) if width else 0.0
        if scale == 0:
            violation = abs(b) if sense == "==" else (max(0, -b) if sense == "<=" else max(0, b))
            if violation > options.feasibility_tol:
                return LPResult("INFEASIBLE", message="Inconsistent constant row")
            continue
        a, b = a / scale, b / scale
        if b < 0:
            a, b = -a, -b
            sense = {"<=": ">=", ">=": "<=", "==": "=="}[sense]
        kept_rows.append(a)
        kept_rhs.append(b)
        kept_senses.append(sense)
    m = len(kept_rows)
    slack_count = sum(s != "==" for s in kept_senses)
    artificial_count = sum(s != "<=" for s in kept_senses)
    real_columns = width + slack_count
    total = real_columns + artificial_count
    tab = np.zeros((m + 1, total + 1))
    basis = []
    slack, artificial = width, real_columns
    for i, (a, b, sense) in enumerate(zip(kept_rows, kept_rhs, kept_senses)):
        tab[i, :width] = a
        tab[i, -1] = b
        if sense != "==":
            tab[i, slack] = 1 if sense == "<=" else -1
            if sense == "<=":
                basis.append(slack)
            slack += 1
        if sense != "<=":
            tab[i, artificial] = 1
            basis.append(artificial)
            artificial += 1
    iterations = 0

    def pivot(row, column):
        # 高斯消元更新整张表，同时替换基变量；迭代次数包含第一阶段人工变量退出。
        nonlocal iterations
        tab[row] /= tab[row, column]
        factors = tab[:, column].copy()
        factors[row] = 0
        tab[:] -= factors[:, None] * tab[row][None, :]
        tab[:, column] = 0.0
        tab[row, column] = 1.0
        basis[row] = column
        iterations += 1

    def set_cost(cost):
        # 按当前基消去目标行中的基变量系数，以得到约化成本。
        tab[-1] = 0
        tab[-1, :-1] = cost
        for i, j in enumerate(basis):
            tab[-1] -= cost[j] * tab[i]

    def simplex():
        # 入基按最小编号；等步长离基优先大主元以抑制退化消元增长，再按基编号确定顺序。
        # 该混合规则不是完整 Bland 防循环定理，循环仍由迭代/时间限制如实终止。
        while True:
            if time.perf_counter() >= deadline:
                return "TIME_LIMIT"
            if not np.all(np.isfinite(tab)):
                return "NUMERICAL_ERROR"
            if len(basis) and np.min(tab[:-1, -1]) < -options.feasibility_tol:
                return "NUMERICAL_ERROR"
            eligible = np.flatnonzero(tab[-1, :-1] < -options.pivot_tol)
            if not len(eligible):
                return "OPTIMAL"
            if iterations >= options.iteration_limit:
                return "ITERATION_LIMIT"
            entering = int(eligible[0])  # Bland's entering rule.
            leaving = np.flatnonzero(tab[:-1, entering] > options.pivot_tol)
            if not len(leaving):
                return "UNBOUNDED"
            row = _leaving_row(tab[:-1, entering],tab[:-1,-1],basis,
                               options.pivot_tol,options.feasibility_tol)
            if row is None:
                return "NUMERICAL_ERROR"
            pivot(row, entering)

    if artificial_count:
        # 第一阶段最小化人工变量之和；只有该阶段成功且人工目标足够小才进入原目标。
        phase1 = np.zeros(total)
        phase1[real_columns:] = 1.0
        set_cost(phase1)
        status = simplex()
        if status != "OPTIMAL":
            return LPResult("NUMERICAL_ERROR" if status == "UNBOUNDED" else status,
                            iterations=iterations, message="Phase I did not finish")
        if -tab[-1, -1] > options.feasibility_tol:
            return LPResult("INFEASIBLE", iterations=iterations, message="Phase I artificial objective positive")
        # 零值人工基变量退出；无真实列可换入时删除对应冗余行，不删除原模型约束。
        row = 0
        while row < len(basis):
            if basis[row] >= real_columns:
                choices = [int(j) for j in np.flatnonzero(np.abs(tab[row, :real_columns]) > options.pivot_tol)
                           if j not in basis]
                if choices:
                    if time.perf_counter() >= deadline:
                        return LPResult("TIME_LIMIT", iterations=iterations)
                    if iterations >= options.iteration_limit:
                        return LPResult("ITERATION_LIMIT", iterations=iterations)
                    pivot(row, choices[0])
                else:
                    tab = np.delete(tab, row, axis=0)
                    basis.pop(row)
                    continue
            row += 1
        tab = np.concatenate((tab[:, :real_columns], tab[:, -1:]), axis=1)
    cost = np.zeros(real_columns)
    direction = 1 if model.sense == "min" else -1
    for i, a in model.objective.terms.items():
        for j, sign in transforms[i]:
            cost[j] += direction * a * sign
    # 第二阶段缩放目标，防止单位过小使约化成本全部落入绝对 pivot 阈值；输出恢复原单位。
    objective_scale = float(np.max(np.abs(cost), initial=0))
    if objective_scale:
        cost /= objective_scale
    set_cost(cost)
    status = simplex()
    if status != "OPTIMAL":
        return LPResult(status, iterations=iterations, message="Phase II did not finish")
    z = np.zeros(real_columns)
    for row, j in enumerate(basis):
        z[j] = tab[row, -1]
    x = offset.copy()
    # 映射回原变量并计入目标常数，统一返回最小化方向目标；由调用层恢复用户的最大化方向。
    for i, parts in enumerate(transforms):
        x[i] += sum(sign * z[j] for j, sign in parts)
    objective = direction * (model.objective.constant + sum(a * x[i] for i, a in model.objective.terms.items()))
    return LPResult("OPTIMAL", x, float(objective), iterations)
