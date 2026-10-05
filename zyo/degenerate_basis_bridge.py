# 全比率原域端点的基桥接：独立核内点列秩、非基界侧和齐次基复建，不签发最优证书。
"""Reconstruct a checked homogeneous warm basis from a bounded equality-LP endpoint."""

import math

import numpy as np
from scipy.linalg import qr
from scipy.sparse import csc_matrix, hstack, identity

from .warm_start import capture_basis_start, evaluate_reuse


def bridge_endpoint_to_basis(matrix, costs, lower, upper, rhs, endpoint, *,
                             feasibility_tolerance=1e-7,
                             residual_tolerance=1e-9,
                             rank_tolerance=1e-10,
                             objective_tolerance=1e-8):
    """Return a checked warm-basis hint or a reasoned refusal for ``A x = b``.

    The caller must supply the *full-ratio boundary endpoint*, not an interior
    trial point. This routine checks its own input but does not prove that the
    endpoint arose from a valid descent, nor that it is optimal.
    """
    result = dict(available=False, basic=[], nonbasic_at_upper=[],
                  claims_original_optimality=False, reason='', diagnostics={})
    diagnostics = result['diagnostics']
    tolerances = (feasibility_tolerance, residual_tolerance,
                  rank_tolerance, objective_tolerance)
    if any(not isinstance(value, (int, float, np.integer, np.floating))
           or not math.isfinite(value) or value <= 0 for value in tolerances):
        result['reason'] = 'all bridge tolerances must be positive and finite'
        return result
    try:
        body = csc_matrix(matrix, dtype=float)
        body.sum_duplicates()
        row_count, column_count = body.shape
        cost = np.asarray(costs, dtype=float).reshape(-1)
        lo = np.asarray(lower, dtype=float).reshape(-1)
        hi = np.asarray(upper, dtype=float).reshape(-1)
        right = np.asarray(rhs, dtype=float).reshape(-1)
        point = np.asarray(endpoint, dtype=float).reshape(-1)
    except (TypeError, ValueError, OverflowError) as exc:
        result['reason'] = f'bridge input cannot be converted: {exc}'
        return result
    if row_count < 1 or column_count < 1:
        result['reason'] = 'bridge requires at least one row and one structural column'
        return result
    if (cost.size != column_count or lo.size != column_count
            or hi.size != column_count or point.size != column_count
            or right.size != row_count):
        result['reason'] = 'matrix, cost, bounds, right-hand side or endpoint dimensions disagree'
        return result
    if (not np.all(np.isfinite(body.data)) or not np.all(np.isfinite(cost))
            or not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi))
            or not np.all(np.isfinite(right)) or not np.all(np.isfinite(point))):
        result['reason'] = 'bridge input contains a non-finite coefficient, bound or point'
        return result
    if np.any(lo > hi):
        result['reason'] = 'a structural lower bound exceeds its upper bound'
        return result
    # 原域先验门：可行端点和目标均由本模块重算，不能仅相信上游的成功标志。
    with np.errstate(over='ignore', invalid='ignore'):
        row_error = body @ point - right
    if not np.all(np.isfinite(row_error)):
        result['reason'] = 'original row activity overflowed'
        return result
    diagnostics['endpoint_row_residual'] = float(np.max(np.abs(row_error), initial=0.0))
    with np.errstate(over='ignore', invalid='ignore'):
        diagnostics['endpoint_bound_violation'] = float(max(
            0.0, np.max(lo-point, initial=0.0), np.max(point-hi, initial=0.0)))
        endpoint_objective = float(cost @ point)
    if not math.isfinite(endpoint_objective):
        result['reason'] = 'original endpoint objective is non-finite'
        return result
    diagnostics['endpoint_objective'] = endpoint_objective
    if (diagnostics['endpoint_row_residual'] > feasibility_tolerance
            or diagnostics['endpoint_bound_violation'] > feasibility_tolerance):
        result['reason'] = 'endpoint fails original row or bound feasibility'
        return result

    free = []
    upper_side = []
    near_bound_candidates = []
    # 真正贴界仅允许浮点舍入量；介于舍入量和可行容差之间的点无法定界侧。
    for j in range(column_count):
        with np.errstate(over='ignore', invalid='ignore'):
            width = hi[j]-lo[j]
        if not math.isfinite(width):
            result['reason'] = f'bound width is non-finite at column {j}'
            return result
        rounding = 32*np.finfo(float).eps*max(1.0, abs(lo[j]), abs(hi[j]))
        if width == 0:
            if abs(point[j]-lo[j]) > rounding:
                result['reason'] = f'fixed column {j} is not at its bound'
                return result
            continue
        if width <= feasibility_tolerance:
            result['reason'] = f'column {j} has ambiguous narrow distinct bounds'
            return result
        # 大偏移窄区间中，纯相对舍入窗可能覆盖整个区间并错认另一侧。
        # 原端点由辅助 LP 浮点运算形成，稍超舍入窗的候选只在原有残差门内
        # 暂定界侧；随后必须从界侧重新解基并逐项核对原点、行、界及目标。
        # 不能像未经重建的宽松比率检验那样直接移动非基变量。
        rounding = min(rounding, 0.25*width, feasibility_tolerance)
        side_window = max(rounding, min(residual_tolerance,
                                        feasibility_tolerance, 0.25*width))
        from_lower = point[j]-lo[j]
        from_upper = hi[j]-point[j]
        if abs(from_lower) <= side_window:
            if abs(from_lower) > rounding:
                near_bound_candidates.append((j, 'lower', abs(float(from_lower))))
            continue
        if abs(from_upper) <= side_window:
            if abs(from_upper) > rounding:
                near_bound_candidates.append((j, 'upper', abs(float(from_upper))))
            upper_side.append(j)
            continue
        if from_lower > feasibility_tolerance and from_upper > feasibility_tolerance:
            free.append(j)
            continue
        result['reason'] = f'column {j} is ambiguously near a bound'
        return result
    diagnostics['interior_columns'] = list(free)
    diagnostics['near_bound_candidates'] = near_bound_candidates
    diagnostics['near_bound_max_offset'] = max(
        (distance for _, _, distance in near_bound_candidates), default=0.0)
    if len(free) > row_count:
        result['reason'] = 'more strictly interior columns than equality rows'
        return result
    if row_count*len(free) > 5_000_000:
        result['reason'] = 'interior rank check exceeds the bounded dense workspace'
        return result

    selected_rows = []
    completed_condition = 1.0
    if free:
        # 对 A_F^T 做带主元 QR 选独立原行；补上其余 -I 列后基维数恰为 m。
        try:
            _, _, permutation = qr(body[:, free].T.toarray(), pivoting=True,
                                   mode='economic', check_finite=False)
            selected_rows = sorted(int(i) for i in permutation[:len(free)])
            square = body[selected_rows, :][:, free].toarray()
            singular_values = np.linalg.svd(square, compute_uv=False)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            result['reason'] = f'interior rank check failed: {exc}'
            return result
        if not np.all(np.isfinite(singular_values)) or singular_values[0] <= 0:
            result['reason'] = 'interior columns have non-finite or zero numerical rank'
            return result
        rank_ratio = float(singular_values[-1]/singular_values[0])
        diagnostics['interior_rank_ratio'] = rank_ratio
        if rank_ratio <= rank_tolerance:
            result['reason'] = 'interior columns are dependent or near singular'
            return result
        # 局部 S=A[P,F] 满秩不代表补全的 B=[A_F,-I_Z] 条件良好。
        # 按块逆 B^-1=[[S^-1,0],[A[Z,F]S^-1,-I]] 求完整基的无穷范数条件，
        # 只需内点列数阶的小矩阵求解，不稠密化原问题所有行列。
        complement = [i for i in range(row_count) if i not in set(selected_rows)]
        try:
            inverse = np.linalg.solve(square, np.eye(len(free)))
            remainder = body[complement, :][:, free].toarray()
            inverse_coupling = remainder @ inverse
            basis_norm = max(
                float(np.max(np.sum(np.abs(square), axis=1), initial=0.0)),
                float(np.max(1.0+np.sum(np.abs(remainder), axis=1), initial=0.0)))
            inverse_norm = max(
                float(np.max(np.sum(np.abs(inverse), axis=1), initial=0.0)),
                float(np.max(1.0+np.sum(np.abs(inverse_coupling), axis=1), initial=0.0)))
            completed_condition = basis_norm*inverse_norm
        except (ValueError, np.linalg.LinAlgError, FloatingPointError, OverflowError) as exc:
            result['reason'] = f'completed basis condition check failed: {exc}'
            return result
        if not math.isfinite(completed_condition) or completed_condition >= 1.0/rank_tolerance:
            result['reason'] = 'completed basis condition exceeds the rank safety gate'
            diagnostics['full_basis_condition_inf'] = float(completed_condition)
            return result
    diagnostics['full_basis_condition_inf'] = float(completed_condition)
    diagnostics['independent_rows'] = list(selected_rows)
    selected = set(selected_rows)
    basic = list(free)+[column_count+i for i in range(row_count) if i not in selected]
    wide = hstack([body, -identity(row_count, format='csc')], format='csc')
    wide_lower = np.concatenate([lo, right])
    wide_upper = np.concatenate([hi, right])
    wide_point = np.concatenate([point, right])
    wide_cost = np.concatenate([cost, np.zeros(row_count)])
    try:
        captured = capture_basis_start(
            wide, basic, wide_point, wide_lower, wide_upper,
            feasibility_tolerance=feasibility_tolerance,
            residual_tolerance=residual_tolerance)
        if not captured.available:
            result['reason'] = f'homogeneous basis capture refused: {captured.reason}'
            return result
        # 界侧必须与原点的严格分类一致，尤其不能把上界误标成下界。
        if captured.nonbasic_at_upper != upper_side:
            result['reason'] = 'captured nonbasic bound sides differ from original endpoint'
            return result
        decision = evaluate_reuse(
            wide, basic, wide_lower, wide_upper, wide_cost,
            nonbasic_at_upper=upper_side,
            feasibility_tolerance=feasibility_tolerance,
            residual_tolerance=residual_tolerance)
    except (ValueError, TypeError, ArithmeticError) as exc:
        result['reason'] = f'homogeneous reconstruction failed: {exc}'
        return result
    if not decision.reusable:
        result['reason'] = f'homogeneous reuse refused: {decision.reason}'
        return result
    rebuilt = np.zeros(column_count+row_count)
    for j, value in decision.nonbasic_values.items():
        rebuilt[j] = value
    for j, value in zip(basic, decision.basic_values):
        rebuilt[j] = value
    if not np.all(np.isfinite(rebuilt)):
        result['reason'] = 'reconstructed basis point is non-finite'
        return result
    diagnostics['reconstruction_difference'] = float(np.max(
        np.abs(rebuilt-wide_point), initial=0.0))
    original_rebuilt = rebuilt[:column_count]
    with np.errstate(over='ignore', invalid='ignore'):
        rebuilt_rows = body @ original_rebuilt-right
    diagnostics['row_residual'] = float(np.max(np.abs(rebuilt_rows), initial=0.0))
    with np.errstate(over='ignore', invalid='ignore'):
        diagnostics['bound_violation'] = float(max(
            0.0, np.max(lo-original_rebuilt, initial=0.0),
            np.max(original_rebuilt-hi, initial=0.0)))
        diagnostics['objective'] = float(cost @ original_rebuilt)
    diagnostics['objective_difference'] = abs(diagnostics['objective']-endpoint_objective)
    if (not np.all(np.isfinite(rebuilt_rows))
            or not all(math.isfinite(diagnostics[key]) for key in (
                'row_residual', 'bound_violation', 'objective', 'objective_difference'))
            or diagnostics['reconstruction_difference'] > feasibility_tolerance
            or diagnostics['row_residual'] > feasibility_tolerance
            or diagnostics['bound_violation'] > feasibility_tolerance
            or diagnostics['objective_difference'] >
            objective_tolerance*max(1.0, abs(endpoint_objective))):
        result['reason'] = 'reconstructed basis differs in original point, row, bounds or objective'
        return result
    result.update(available=True, basic=basic, nonbasic_at_upper=upper_side,
                  reason='original-feasible endpoint reconstructed by a checked warm basis')
    return result
