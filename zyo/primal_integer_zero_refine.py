# 整数结构 LP 的零步专用数值实验；只使用稀疏 LU 基础线性代数，不调用外部优化器。
"""Opt-in bounded-primal integer-basis refinement and guarded strong-pivot choice."""
from decimal import Decimal, localcontext
import math

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu


def _certify_integer_basic_point(basis, rhs, values, lower, upper, tolerance):
    """仅在整条近整基向量精确满足整数方程和原界时替换浮点残值。"""
    current = np.asarray(values, dtype=float).reshape(-1)
    lo = np.asarray(lower, dtype=float).reshape(-1)
    hi = np.asarray(upper, dtype=float).reshape(-1)
    if (current.size != basis.shape[1] or lo.size != current.size
            or hi.size != current.size or not math.isfinite(tolerance)
            or tolerance <= 0 or not np.all(np.isfinite(current))
            or np.any(np.isnan(lo)) or np.any(np.isnan(hi))
            or np.any(lo > hi)):
        return dict(accepted=False, reason='invalid integer-snap dimensions or values')
    if (basis.shape[0] != basis.shape[1] or basis.nnz > 2**20
            or not np.all(np.isfinite(rhs))
            or np.max(np.abs(rhs), initial=0.) > 2**52
            or not np.all(np.isfinite(basis.data))
            or np.max(np.abs(basis.data), initial=0.) > 2**53):
        return dict(accepted=False, reason='basis or RHS exceeds exact integer data range')
    rounded = np.rint(current)
    if (not np.all(np.isfinite(rounded))
            or np.max(np.abs(rounded), initial=0.) > 2**52):
        return dict(accepted=False, reason='rounded basic integer exceeds safe range')
    # 近整仅负责筛选候选，不是正确性证据；最终必须以 Python-int 精确逐行核验。
    near = min(1e-12, tolerance*1e-3)
    if np.any(np.abs(current-rounded) > near):
        return dict(accepted=False, reason='a basic coordinate is not near an integer')
    integers = [int(value) for value in rounded]
    for j, value in enumerate(integers):
        # Python 内置 int 与 float 按精确数值比较；原界保持 binary64 原值，
        # 不作整数化或放宽。先过滤无穷界，继续拒绝上面的 NaN 输入。
        if ((math.isfinite(lo[j]) and value < float(lo[j]))
                or (math.isfinite(hi[j]) and value > float(hi[j]))):
            return dict(accepted=False, reason='rounded basic point violates an original bound')
    if (not np.all(np.isfinite(rhs)) or not np.all(rhs == np.rint(rhs))
            or not np.all(np.isfinite(basis.data))
            or not np.all(basis.data == np.rint(basis.data))):
        return dict(accepted=False, reason='basis or RHS is not exact integer data')
    exact_rhs = [int(value) for value in rhs]
    exact_rows = [0]*basis.shape[0]
    for column, value in enumerate(integers):
        for k in range(basis.indptr[column], basis.indptr[column+1]):
            exact_rows[int(basis.indices[k])] += int(basis.data[k])*value
    if exact_rows != exact_rhs:
        return dict(accepted=False, reason='rounded basic point fails exact integer equations')
    return dict(accepted=True, reason='entire basic point satisfies exact integer equations and bounds',
                values=[float(value) for value in integers], rows_checked=len(exact_rows))


def refine_integer_basic(matrix, basic, nonbasic_values, *, corrections=4,
                         exact_integer_snap=False, basic_lower=None, basic_upper=None,
                         feasibility_tolerance=None):
    """仅在精确整数点积条件下重建 Bx_B=-Nx_N，额外精度计算原方程残差。"""
    A = csc_matrix(matrix, dtype=float)
    indices = np.asarray(basic, dtype=np.intp).reshape(-1)
    nonbasic = np.asarray(nonbasic_values, dtype=float).reshape(-1)
    if (len(indices) != A.shape[0] or len(set(indices.tolist())) != len(indices)
            or nonbasic.size != A.shape[1] or np.any(indices < 0)
            or np.any(indices >= A.shape[1])):
        return dict(accepted=False, reason='invalid basis/nonbasic dimensions')
    if np.any(nonbasic[indices] != 0):
        return dict(accepted=False, reason='basic entries in nonbasic vector are nonzero')
    if (not np.all(np.isfinite(A.data)) or not np.all(A.data == np.rint(A.data))
            or np.max(np.abs(A.data), initial=0) > 2**53):
        return dict(accepted=False, reason='matrix coefficients are not exact binary64 integers')
    if (not np.all(np.isfinite(nonbasic))
            or not np.all(nonbasic == np.rint(nonbasic))
            or np.max(np.abs(nonbasic), initial=0) > 2**53):
        return dict(accepted=False, reason='nonbasic values are not exact binary64 integers')
    # 浮点绝对和在 2^53 边界可把 2^53+1 舍入为 2^53；留一位安全余量，
    # 并限制加法次数，使真实正项和仍严格小于 2^53，随后每项/部分和都是精确整数。
    if A.nnz > 2**20:
        return dict(accepted=False, reason='integer row sum operation count exceeds safe range')
    absolute_sum = abs(A) @ np.abs(nonbasic)
    if (not np.all(np.isfinite(absolute_sum))
            or np.max(absolute_sum, initial=0) > 2**52):
        return dict(accepted=False, reason='integer row sum exceeds exact binary64 range')
    rhs = -(A @ nonbasic)
    if not np.all(rhs == np.rint(rhs)):
        return dict(accepted=False, reason='integer right-hand side identity failed')
    B = csc_matrix(A[:, indices])
    try:
        factor = splu(B)
        values = factor.solve(rhs)
    except RuntimeError as exc:
        return dict(accepted=False, reason='integer basis factorization failed: '+str(exc))
    initial = values.copy()
    residuals = []
    with localcontext() as context:
        context.prec = 80
        # Decimal.from_float 精确转换固定 binary64 值，与 80 位运算上下文无关；
        # B 和 rhs 在本次修正中不变，只构造一次，逐轮变化的基值仍逐轮转换。
        exact_rhs = [Decimal.from_float(float(value)) for value in rhs]
        exact_data = [Decimal.from_float(float(value)) for value in B.data]
        for iteration in range(corrections+1):
            residual = exact_rhs.copy()
            for column in range(B.shape[1]):
                value = Decimal.from_float(float(values[column]))
                for k in range(B.indptr[column], B.indptr[column+1]):
                    row = int(B.indices[k])
                    residual[row] -= exact_data[k]*value
            error = float(max((abs(value) for value in residual), default=Decimal(0)))
            residuals.append(error)
            if iteration < corrections:
                values += factor.solve(np.asarray([float(value) for value in residual]))
            if not np.all(np.isfinite(values)):
                return dict(accepted=False, reason='integer refinement became nonfinite')
    if residuals[-1] > min(residuals[0], 1e-12):
        return dict(accepted=False, reason='extra-precision original residual did not pass',
                    residuals=residuals)
    snap = (None if not exact_integer_snap else _certify_integer_basic_point(
        B, rhs, values, basic_lower, basic_upper, feasibility_tolerance))
    return dict(accepted=True, reason='exact integer RHS and high-precision residual accepted',
                values=(snap['values'] if snap is not None and snap['accepted']
                        else values.tolist()), initial_values=initial.tolist(),
                exact_rhs_error=residuals[-1], residuals=residuals,
                rhs_max_abs=float(np.max(np.abs(rhs), initial=0)),
                **(dict(integer_exact_snap_accepted=bool(snap['accepted']),
                        integer_exact_snap_reason=snap['reason'],
                        integer_exact_snap_rows=snap.get('rows_checked', 0),
                        unsnapped_values=values.tolist())
                   if snap is not None else {}))


def strong_zero_leaving(*, basic, moving, values, lower, upper,
                        feasibility_tolerance, pricing_tolerance):
    """Gill 等两遍择行的零步特例；离基值必须几乎精确在原边界，方可落界。"""
    indices = np.asarray(basic, dtype=np.intp)
    direction = -np.asarray(moving, dtype=float)
    current = np.asarray(values, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if (any(vector.size != indices.size for vector in (direction, current, lo, hi))
            or not np.all(np.isfinite(direction)) or not np.all(np.isfinite(current))
            or not math.isfinite(feasibility_tolerance) or feasibility_tolerance <= 0
            or not math.isfinite(pricing_tolerance) or pricing_tolerance <= 0):
        return dict(accepted=False, reason='invalid bounded-primal inputs')
    strict = min(1e-12, feasibility_tolerance*1e-3)
    violation = float(max(np.max(lo-current, initial=0.),
                          np.max(current-hi, initial=0.), 0.))
    if violation > strict:
        return dict(accepted=False, reason='refined basic values remain off their original bounds',
                    max_bound_violation=violation)
    candidates = []
    for position, derivative in enumerate(direction):
        if derivative > pricing_tolerance and math.isfinite(hi[position]):
            bound, side = hi[position], True
            relaxed = (bound+feasibility_tolerance-current[position])/derivative
        elif derivative < -pricing_tolerance and math.isfinite(lo[position]):
            bound, side = lo[position], False
            relaxed = (bound-feasibility_tolerance-current[position])/derivative
        else:
            continue
        exact = (bound-current[position])/derivative
        if math.isfinite(exact) and math.isfinite(relaxed):
            candidates.append((max(0., relaxed), exact, abs(derivative),
                               int(indices[position]), position, side, bound))
    if not candidates:
        return dict(accepted=False, reason='no bounded basic row limits the step')
    first = min(item[0] for item in candidates)
    eligible = [item for item in candidates if item[1] <= first]
    if not eligible:
        return dict(accepted=False, reason='no second-pass blocker')
    chosen = max(eligible, key=lambda item: (item[2], -item[3]))
    step = max(0., chosen[1])
    leaving_value = current[chosen[4]]+step*direction[chosen[4]]
    snap = abs(float(chosen[6]-leaving_value))
    if snap > strict:
        return dict(accepted=False, reason='strong pivot requires a genuine off-bound state',
                    snap_change=snap)
    trial = current+step*direction
    trial[chosen[4]] = chosen[6]
    trial_violation = float(max(np.max(lo-trial, initial=0.),
                                np.max(trial-hi, initial=0.), 0.))
    if trial_violation > feasibility_tolerance:
        return dict(accepted=False, reason='strong pivot violates a nonleaving bound',
                    max_bound_violation=trial_violation)
    return dict(accepted=True, position=chosen[4], leaving_upper=chosen[5],
                step=float(step), snap_change=snap,
                pivot_magnitude=float(chosen[2]), eligible_rows=len(eligible),
                reason='refined integer basic point permits near-exact strong pivot')
