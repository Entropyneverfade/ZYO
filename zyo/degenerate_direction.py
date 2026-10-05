# 全退化原生 LP 的兼容方向候选：辅助 LP 只提出方向，原模型复核决定能否另存下降点。
"""Find a separately checked descent from an all-bound feasible LP point."""
import math
import time

import numpy as np
from scipy.sparse import csc_matrix, diags, hstack

from .lp_certificate import lp_certificate


def attempt_degenerate_descent(matrix, costs, lower, upper, rhs, point, *,
                               iteration_limit, maximize=False, feasibility_tolerance=1e-7,
                               objective_tolerance=1e-8):
    """求全退化 IPS 兼容方向；只返回原模型可行下降候选，不证明原 LP 最优。"""
    from .native_lp import build_model, solve_certified_lp

    report = dict(attempted=False, accepted=False, claims_original_optimality=False,
                  auxiliary_status=None, auxiliary_iterations=0, auxiliary_seconds=0.,
                  reason='eligibility not checked')
    # 规范化重复坐标，确保稀疏乘法与原模型逐行证书使用同一组系数。
    body = csc_matrix(matrix, dtype=float).copy()
    body.sum_duplicates()
    rows, columns = body.shape
    cost = np.asarray(costs, dtype=float).reshape(-1)
    lo = np.asarray(lower, dtype=float).reshape(-1)
    hi = np.asarray(upper, dtype=float).reshape(-1)
    right = np.asarray(rhs, dtype=float).reshape(-1)
    start = np.asarray(point, dtype=float).reshape(-1)
    if (isinstance(iteration_limit, bool) or not isinstance(iteration_limit, (int, np.integer))
            or iteration_limit < 0):
        raise ValueError('auxiliary iteration limit must be a nonnegative integer')
    if (any(vector.size != columns for vector in (cost, lo, hi, start))
            or right.size != rows or np.any(lo > hi)
            or not np.all(np.isfinite(body.data)) or not np.all(np.isfinite(cost))
            or not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi))
            or not np.all(np.isfinite(right)) or not np.all(np.isfinite(start))):
        return dict(report, reason='invalid dimensions, bounds, or nonfinite original data')
    if (not math.isfinite(feasibility_tolerance) or feasibility_tolerance <= 0
            or not math.isfinite(objective_tolerance) or objective_tolerance <= 0):
        raise ValueError('audit tolerances must be positive and finite')

    original = build_model(body, cost, lo, hi, right, sense=['==']*rows,
                           maximize=maximize)
    zero_multipliers = np.zeros(rows)
    before = lp_certificate(original, start, zero_multipliers,
                            feasibility_tolerance=feasibility_tolerance,
                            objective_tolerance=objective_tolerance)
    if not before.primal_feasible:
        return dict(report, reason='original starting point failed row or bound gate',
                    starting_row_violation=before.row_violation,
                    starting_bound_violation=before.bound_violation)
    if before.objective is None or not math.isfinite(before.objective):
        # 原始点目标溢出时，浮点的 inf-inf 会成为 NaN；不得把 NaN 当成严格下降。
        return dict(report, reason='original starting objective is nonfinite')

    movable = np.flatnonzero(lo < hi)
    if not movable.size:
        return dict(report, reason='no movable bound variable')
    # 界侧只用原可行容差识别；同时接近两侧的小区间无法可靠定向，直接拒绝。
    near_lower = np.abs(start[movable]-lo[movable]) <= feasibility_tolerance
    near_upper = np.abs(start[movable]-hi[movable]) <= feasibility_tolerance
    if np.any(~(near_lower | near_upper)) or np.any(near_lower & near_upper):
        return dict(report, reason='not a uniquely oriented all-bound point')
    signs = np.where(near_lower, 1., -1.)
    oriented = body[:, movable] @ diags(signs, format='csc')
    # 最大化先对目标取负；辅助定价始终求有效成本的负斜率。
    oriented_cost = (-1. if maximize else 1.)*cost[movable]*signs
    # 原论文全退化定价 P: min g'y, Cy=0, 1'y=1, y>=0。求其对偶以获得可行零起点。
    # μ=μ0+v，μ0=min(g)，则 π=0、v=0 满足 C'π+v<=g-μ0。
    offset = float(np.min(oriented_cost))
    auxiliary_matrix = hstack((oriented.T,
                               csc_matrix(np.ones((movable.size, 1)))), format='csc')
    # 墙钟时间只覆盖递归定价调用；主 LP 的迭代和时间不与辅助预算混算。
    auxiliary_started = time.perf_counter()
    auxiliary = solve_certified_lp(
        auxiliary_matrix, np.r_[np.zeros(rows), 1.],
        np.r_[np.full(rows, -math.inf), 0.], np.full(rows+1, math.inf),
        oriented_cost-offset, sense=['<=']*movable.size, maximize=True,
        iteration_limit=int(iteration_limit),
        feasibility_tolerance=feasibility_tolerance,
        objective_tolerance=objective_tolerance,
        degenerate_rescue_iteration_limit=0)
    auxiliary_seconds = time.perf_counter()-auxiliary_started
    report.update(attempted=True, auxiliary_status=auxiliary.status,
                  auxiliary_iterations=int(auxiliary.iterations),
                  auxiliary_seconds=float(auxiliary_seconds),
                  auxiliary_optimal=bool(auxiliary.optimal),
                  pricing_offset=offset)
    if (auxiliary.status != 'OPTIMAL' or not auxiliary.optimal
            or auxiliary.certificate is None or not auxiliary.certificate.verified
            or auxiliary.dual is None):
        return dict(report, reason='auxiliary pricing lacks a certified primal direction')
    y = np.asarray(auxiliary.dual, dtype=float).reshape(-1)
    if (y.size != movable.size or not np.all(np.isfinite(y))
            or np.min(y, initial=0.) < -feasibility_tolerance):
        return dict(report, reason='auxiliary row multipliers are not nonnegative pricing weights')
    # 浮点对偶乘子的舍入级负量只作为候选投影；随后重新核验每个原方程与目标。
    y = np.maximum(y, 0.)
    total = float(math.fsum(float(value) for value in y))
    if not math.isfinite(total) or total <= 0:
        return dict(report, reason='auxiliary pricing weights have no positive mass')
    y /= total
    pricing_row_residual = float(np.max(np.abs(oriented @ y), initial=0.))
    pricing_slope = float(math.fsum(float(a*b) for a, b in zip(oriented_cost, y)))
    if (not math.isfinite(pricing_row_residual) or not math.isfinite(pricing_slope)
            or abs(float(math.fsum(float(value) for value in y))-1.) > feasibility_tolerance
            or pricing_row_residual > feasibility_tolerance
            or pricing_slope >= -objective_tolerance):
        return dict(report, reason='pricing direction failed equation or decrease gate',
                    pricing_row_residual=pricing_row_residual,
                    pricing_slope=pricing_slope)

    direction = np.zeros(columns)
    direction[movable] = signs*y
    active = np.flatnonzero(direction)
    rooms = np.where(direction[active] > 0,
                     hi[active]-start[active], start[active]-lo[active])
    if np.any(rooms <= 0):
        return dict(report, reason='oriented direction has no positive room')
    maximal_step = float(np.min(rooms/np.abs(direction[active])))
    # 半步避开浮点恰好触另一侧界；这只是可行候选，不直接更新旧基或限额状态。
    step = 0.5*maximal_step
    if not math.isfinite(step) or step <= feasibility_tolerance:
        return dict(report, reason='compatible direction has no robust positive step',
                    maximal_step=maximal_step)
    after_point = start+step*direction
    after = lp_certificate(original, after_point, zero_multipliers,
                           feasibility_tolerance=feasibility_tolerance,
                           objective_tolerance=objective_tolerance)
    if after.objective is None or not math.isfinite(after.objective):
        return dict(report, reason='original candidate objective is nonfinite')
    improvement = float((after.objective-before.objective) if maximize else
                        (before.objective-after.objective))
    threshold = objective_tolerance*max(1., abs(before.objective))
    if (not after.primal_feasible or not math.isfinite(improvement)
            or not math.isfinite(threshold) or improvement <= threshold):
        return dict(report, reason='candidate failed original row, bound, or cost gate',
                    candidate_row_violation=after.row_violation,
                    candidate_bound_violation=after.bound_violation,
                    objective_before=before.objective,
                    objective_after=after.objective)
    # 最大比率边界点另行复核：半步候选本身通常不能构成单纯形基，
    # 触界点也只有在内点列可补成非奇异基时才可能用于后续热启动。
    boundary = dict(boundary_available=False, maximal_step=maximal_step)
    boundary_point = start+maximal_step*direction
    if np.all(np.isfinite(boundary_point)):
        edge = lp_certificate(original, boundary_point, zero_multipliers,
                              feasibility_tolerance=feasibility_tolerance,
                              objective_tolerance=objective_tolerance)
        edge_objective = edge.objective
        edge_improvement = (float(edge_objective-before.objective)
                            if maximize else float(before.objective-edge_objective)) \
            if edge_objective is not None and math.isfinite(edge_objective) else math.nan
        if (edge.primal_feasible and math.isfinite(edge_improvement)
                and edge_improvement > threshold):
            boundary.update(boundary_available=True,
                            boundary_point=boundary_point.tolist(),
                            boundary_objective=float(edge_objective),
                            boundary_row_violation=float(edge.row_violation),
                            boundary_bound_violation=float(edge.bound_violation))
        else:
            boundary['boundary_reason'] = 'maximal-ratio point failed original feasibility or cost gate'
    else:
        boundary['boundary_reason'] = 'maximal-ratio point is nonfinite'
    return dict(report, accepted=True, **boundary,
                reason='independently checked original feasible descent; no optimum claim',
                pricing_row_residual=pricing_row_residual,
                pricing_slope=pricing_slope, step=step,
                direction_support=[int(j) for j in active],
                objective_before=float(before.objective),
                objective_after=float(after.objective),
                candidate_row_violation=float(after.row_violation),
                candidate_bound_violation=float(after.bound_violation),
                point_after=after_point.tolist())
