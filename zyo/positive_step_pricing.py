# 退化时的正步入基筛选：参考 Positive Edge 的研究动机，但执行精确列 FTRAN 而非其概率相容性判别。
"""Optional read-only full-column positive-step screening for a primal basis."""
import math

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu


def screen_positive_step(*, matrix, basic, eligible, improving, at_upper,
                         values, lower, upper, pricing_tolerance,
                         minimum_step, max_columns=5000):
    """仅返回预计可走严格正步的改善列；原生主循环仍须重新 FTRAN/比例验证。"""
    body = csc_matrix(matrix, dtype=float)
    rows, columns = body.shape
    basis = np.asarray(basic, dtype=np.intp).reshape(-1)
    mask = np.asarray(eligible, dtype=bool).reshape(-1)
    gain = np.asarray(improving, dtype=float).reshape(-1)
    upper_side = np.asarray(at_upper, dtype=bool).reshape(-1)
    point = np.asarray(values, dtype=float).reshape(-1)
    lo = np.asarray(lower, dtype=float).reshape(-1)
    hi = np.asarray(upper, dtype=float).reshape(-1)
    if (basis.size != rows or len(set(basis.tolist())) != rows
            or any(vector.size != columns for vector in (mask, gain, upper_side, point, lo, hi))
            or np.any(basis < 0) or np.any(basis >= columns)
            or not np.all(np.isfinite(point)) or not np.all(np.isfinite(gain[mask]))
            or not math.isfinite(pricing_tolerance) or pricing_tolerance <= 0
            or not math.isfinite(minimum_step) or minimum_step <= 0
            or isinstance(max_columns, bool)
            or not isinstance(max_columns, (int, np.integer)) or max_columns < 1):
        raise ValueError('Invalid positive-step screening state or budget')
    if np.any(mask[basis]):
        raise ValueError('A basic column cannot enter through screening')
    basic_point = point[basis]
    if max(float(np.max(lo[basis]-basic_point, initial=0.)),
           float(np.max(basic_point-hi[basis], initial=0.))) > minimum_step:
        return dict(accepted=False, reason='starting basis exceeds positive-step feasibility gate',
                    scanned_columns=0, eligible_columns=int(np.count_nonzero(mask)))
    candidates = np.flatnonzero(mask)
    if not candidates.size:
        return dict(accepted=False, reason='no improving candidate',
                    scanned_columns=0, eligible_columns=0)
    # 改进量大者优先；同值取小下标。扫描预算截断必须显式记录，不能当作遍历全部列。
    order = candidates[np.lexsort((candidates, -gain[candidates]))]
    try:
        factor = splu(body[:, basis].tocsc())
    except RuntimeError as error:
        return dict(accepted=False, reason='fresh basis LU failed: '+str(error),
                    scanned_columns=0, eligible_columns=int(candidates.size))
    scanned = 0
    for column in order[:max_columns]:
        j = int(column)
        scanned += 1
        sign = -1. if upper_side[j] else 1.
        flip_room = (point[j]-lo[j]) if sign < 0 else (hi[j]-point[j])
        if flip_room <= minimum_step:
            continue
        move = -sign*factor.solve(body[:, j].toarray().ravel())
        if not np.all(np.isfinite(move)):
            continue
        step = float(flip_room)
        for position, derivative in enumerate(move):
            index = int(basis[position])
            if derivative > pricing_tolerance and math.isfinite(float(hi[index])):
                step = min(step, float((hi[index]-basic_point[position])/derivative))
            elif derivative < -pricing_tolerance and math.isfinite(float(lo[index])):
                step = min(step, float((lo[index]-basic_point[position])/derivative))
            if step <= minimum_step:
                break
        if math.isfinite(step) and step > minimum_step:
            return dict(accepted=True, column=j, estimated_step=step,
                        estimated_gain=float(gain[j]*step), scanned_columns=scanned,
                        eligible_columns=int(candidates.size),
                        reason='improving column has a positive fresh-LU ratio estimate')
    return dict(accepted=False, reason='no robust positive step in scan budget',
                scanned_columns=scanned, eligible_columns=int(candidates.size),
                scan_truncated=bool(scanned < candidates.size))
