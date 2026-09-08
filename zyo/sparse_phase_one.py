# 原生弹性 Phase-I：只生成原始行乘子候选，最终证明由独立盒证书检查器负责。
"""Finite-box elastic feasibility model, with no optimizer dispatch."""
import math
import time

import numpy as np
from lzyopt.model import Model, LinearExpression, Constraint


def build_phase_one(model, lower, upper, deadline):
    """Return a data-only auxiliary LP and its original-row multiplier map."""
    with np.errstate(all='raise'):
        lo = np.asarray(lower, dtype=float)
        hi = np.asarray(upper, dtype=float)
        if (lo.shape != (len(model.variables),) or hi.shape != lo.shape
                or not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi))
                or np.any(lo > hi)):
            raise ValueError('Phase-I requires a valid finite node box')
        width = hi-lo
        active = np.flatnonzero(width > 0)
        indices = {int(i): j for j, i in enumerate(active)}
        auxiliary = Model('zyo_native_phase_one')
        for j in range(len(active)):
            if time.perf_counter() >= deadline:
                raise TimeoutError('Phase-I construction time limit')
            auxiliary.add_var(f'z{j}', lb=0, ub=1)
        # 归一化行在整个单位盒上的绝对值不超过 2；tau=2.5 严格满足所有弹性行。
        # 3 只约束辅助松弛，不改变原始变量边界；原模型可行当且仅当最优 tau=0。
        tau = auxiliary.add_var('tau', lb=0, ub=3)
        mapping = []
        for j, row in enumerate(model.constraints):
            if time.perf_counter() >= deadline:
                raise TimeoutError('Phase-I construction time limit')
            shifted = math.fsum([row.expression.constant,
                                *[a*lo[i] for i,a in row.expression.terms.items()]])
            terms = {indices[i]: a*width[i] for i,a in row.expression.terms.items() if i in indices}
            scale = max(1., abs(shifted), math.fsum(abs(a) for a in terms.values()))
            if not math.isfinite(scale):
                raise ValueError('Nonfinite Phase-I row scale')
            # 统一成 <=；等式用正反两行，回映射时相减，保留等式乘子的自由符号。
            signs = (1., -1.) if row.sense == '==' else (1.,) if row.sense == '<=' else (-1.,)
            for sign in signs:
                factor = np.float64(sign)/scale
                coefficients = {i: a*factor for i,a in terms.items()}
                coefficients[tau.index] = -1.
                expression = LinearExpression(auxiliary, coefficients, shifted*factor)
                auxiliary.add_constr(Constraint(expression, '<='))
                mapping.append((j, float(factor)))
        auxiliary.set_objective(tau, sense='min')
        return auxiliary, mapping


def original_multipliers(model, mapping, multipliers):
    # 任意有限乘子只作为候选；缩放下溢或维度错误必须拒绝，不能制造虚假分离。
    with np.errstate(all='raise'):
        values = np.asarray(multipliers, dtype=float)
        if values.shape != (len(mapping),) or not np.all(np.isfinite(values)):
            raise ValueError('Invalid Phase-I multipliers')
        grouped = [[] for _ in model.constraints]
        for (original, factor), value in zip(mapping, values):
            grouped[original].append(value*factor)
        result = np.array([math.fsum(v) for v in grouped])
        for j, row in enumerate(model.constraints):
            if row.sense == '<=':
                result[j] = min(0., result[j])
            elif row.sense == '>=':
                result[j] = max(0., result[j])
        return result
