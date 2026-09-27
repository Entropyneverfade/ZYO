# LP 对偶/简约成本/KKT 证书导出与独立验证（ZYO-033）。
# 只做"检查并如实报告"，不修改候选、不放宽容差、不把残差小当作解准确。
"""Export and independently verify an LP optimality certificate.

Given a candidate ``x`` and row multipliers ``lambda`` in ORIGINAL model units, this
module re-derives every optimality condition from the model's stored coefficients:

* primal feasibility — row activity, variable bounds and integrality;
* dual feasibility — reduced costs ``r = c - A'lambda`` compatible with each variable's
  domain (``r_j >= 0`` for a lower-bounded variable, ``r_j <= 0`` for an upper-bounded
  one, ``r_j == 0`` numerically for a free one), plus the row sign convention;
* complementary slackness — a nonbasic-at-bound variable must have the matching reduced
  cost sign, and a variable strictly inside its bounds must have ``r_j == 0``;
* strong duality — the primal objective against the Lagrangian dual value.

Nothing here trusts the solver that produced the candidate. All quantities are recomputed
with ``math.fsum`` and a conservative roundoff allowance, and a certificate is only
reported ``verified`` when every condition passes.
"""
from dataclasses import dataclass, field
import math

import numpy as np

# 简约成本与域的相容性判据所用的相对尺度阈值。
DOMAIN_TOLERANCE = 1e-7


@dataclass
class LPCertificate:
    """Structured certificate; every field is recomputed, none is copied from a solver."""

    verified: bool = False
    reason: str = ''
    primal_feasible: bool = False
    dual_feasible: bool = False
    complementary: bool = False
    strongly_dual: bool = False
    objective: float | None = None
    dual_objective: float | None = None
    duality_gap: float | None = None
    relative_gap: float | None = None
    row_violation: float = math.inf
    bound_violation: float = math.inf
    integrality_violation: float = math.inf
    dual_violation: float = math.inf
    complementarity_violation: float = math.inf
    row_sign_violation: float = 0.0
    multiplier_scale: float = 0.0
    reduced_costs: list = field(default_factory=list)
    row_activities: list = field(default_factory=list)
    slack_lower: list = field(default_factory=list)
    slack_upper: list = field(default_factory=list)
    roundoff_allowance: float = 0.0
    checks: dict = field(default_factory=dict)
    scope: str = ('floating-point KKT certificate in original model units; not exact '
                  'rational arithmetic and not a general infeasibility/unboundedness proof')


def _activity(expression, point):
    return math.fsum([expression.constant,
                      *[a*point[i] for i, a in expression.terms.items()]])


def lp_certificate(model, values, multipliers, *, feasibility_tolerance=1e-7,
                   objective_tolerance=1e-8, direction=None):
    """Build and verify a KKT/duality certificate for ``model``.

    ``multipliers`` must be given in the model's original row units and the model's own
    sense (``min`` unless ``direction`` overrides it). Signed convention: for a
    minimization problem ``lambda <= 0`` on ``<=`` rows, ``lambda >= 0`` on ``>=`` rows
    and free on ``==`` rows; for maximization the sign flips with the direction.
    """
    certificate = LPCertificate()
    point = np.asarray(values, dtype=float).reshape(-1)
    lam = np.asarray(multipliers, dtype=float).reshape(-1)
    count = len(model.variables)
    rows = len(model.constraints)
    if point.size != count:
        certificate.reason = f'candidate has {point.size} values but the model has {count} variables'
        return certificate
    if lam.size != rows:
        certificate.reason = f'multiplier vector has {lam.size} entries but the model has {rows} rows'
        return certificate
    if not np.all(np.isfinite(point)):
        certificate.reason = 'candidate contains non-finite values'
        return certificate
    if not np.all(np.isfinite(lam)):
        certificate.reason = 'multiplier vector contains non-finite values'
        return certificate

    sign = 1.0 if (direction or model.sense) == 'min' else -1.0
    objective = _activity(model.objective, point)

    # ---- 原始可行性 ----
    row_violation = 0.0
    activities = []
    for row in model.constraints:
        value = _activity(row.expression, point)
        activities.append(value)
        violation = abs(value) if row.sense == '==' else max(0.0, value if row.sense == '<=' else -value)
        row_violation = max(row_violation, violation)
    bound_violation = 0.0
    integrality_violation = 0.0
    for index, var in enumerate(model.variables):
        if math.isfinite(var.lb):
            bound_violation = max(bound_violation, var.lb-point[index])
        if math.isfinite(var.ub):
            bound_violation = max(bound_violation, point[index]-var.ub)
        if var.kind != 'C':
            integrality_violation = max(integrality_violation, abs(point[index]-round(point[index])))
    bound_violation = max(0.0, bound_violation)
    certificate.row_violation = float(row_violation)
    certificate.bound_violation = float(bound_violation)
    certificate.integrality_violation = float(integrality_violation)
    certificate.row_activities = [float(v) for v in activities]
    certificate.primal_feasible = max(row_violation, bound_violation,
                                      integrality_violation) <= feasibility_tolerance

    # ---- 对偶可行性：行符号约定与简约成本-域相容性 ----
    # 符号检验必须带**与其它判据一致的容差**：乘子由 `B'y = c_B` 解出，行在最优解上恰好取等号
    # 时该分量的精确值为 0，浮点结果是 ±eps 量级（实测 afiro：19 个 `<=` 行里有 2 行的有符号
    # 乘子为 +7.0e-18 与 +2.3e-16，而乘子尺度为 3.29，行活动量与右端之差为 0 与 1.8e-14）。
    # 早期版本用**精确** `value > 0` 判失败，于是把目标值已与公开参考值一致到 8.6e-16 的
    # 真最优判为证书不通过。容差按乘子尺度缩放，符号真错（如 λ=±1）仍然会被拒。
    row_sign_violation = 0.0
    lam_scale = 1.0+float(np.max(np.abs(lam), initial=0.0)) if lam.size else 1.0
    for j, row in enumerate(model.constraints):
        value = sign*lam[j]
        if row.sense == '<=':
            row_sign_violation = max(row_sign_violation, value)
        elif row.sense == '>=':
            row_sign_violation = max(row_sign_violation, -value)
    row_sign_ok = row_sign_violation <= DOMAIN_TOLERANCE*lam_scale
    certificate.row_sign_violation = float(max(0.0, row_sign_violation))
    certificate.multiplier_scale = float(lam_scale)
    reduced = np.zeros(count)
    for index, coefficient in model.objective.terms.items():
        reduced[index] = sign*coefficient
    for j, row in enumerate(model.constraints):
        for index, coefficient in row.expression.terms.items():
            reduced[index] -= coefficient*sign*lam[j]
    reference = 1.0+float(np.max(np.abs(reduced), initial=0.0))
    domain_ok = True
    dual_violation = 0.0
    for index, var in enumerate(model.variables):
        r = float(reduced[index])
        lower, upper = math.isfinite(var.lb), math.isfinite(var.ub)
        if lower and upper:
            continue
        if lower:
            if r < -DOMAIN_TOLERANCE*reference:
                domain_ok = False
                dual_violation = max(dual_violation, -r)
        elif upper:
            if r > DOMAIN_TOLERANCE*reference:
                domain_ok = False
                dual_violation = max(dual_violation, r)
        elif abs(r) > DOMAIN_TOLERANCE*reference:
            domain_ok = False
            dual_violation = max(dual_violation, abs(r))
    certificate.dual_violation = float(dual_violation)
    certificate.reduced_costs = [float(v) for v in reduced]
    certificate.dual_feasible = bool(row_sign_ok and domain_ok)

    # ---- 互补松弛 ----
    # 双边有界变量的 KKT 条件只要求**单向蕴含**：r_j > 0 则 x_j 必须停在下界，r_j < 0 则
    # 必须停在上界；x_j = l_j 时另一端的乘积并不要求为零（此时 μ_j = r_j > 0、ν_j = 0，
    # 而 u_j - l_j > 0）。早期版本同时要求 r_j(x_j-l_j) = 0 **且** r_j(u_j-x_j) = 0，
    # 连最简单的 `min -x, 0 <= x <= 1`（最优点 x=1、r=-1）都判为失败——
    # 实测 complementarity_violation = 1.0、verified = False，而该点确为真最优。
    # 违反量按"简约成本 × 错误一侧的位移"计量：停在正确一侧时为 0。
    slack_lower = []
    slack_upper = []
    complementarity = 0.0
    for index, var in enumerate(model.variables):
        r = float(reduced[index])
        lower, upper = math.isfinite(var.lb), math.isfinite(var.ub)
        slack_lower.append(float(point[index]-var.lb) if lower else None)
        slack_upper.append(float(var.ub-point[index]) if upper else None)
        if r > DOMAIN_TOLERANCE*reference:
            # 简约成本为正：变量必须在下界。下界不存在时该方向无法被界住，直接计为违反。
            displacement = max(0.0, float(point[index]-var.lb)) if lower else 1.0
            complementarity = max(complementarity, r*displacement)
        elif r < -DOMAIN_TOLERANCE*reference:
            displacement = max(0.0, float(var.ub-point[index])) if upper else 1.0
            complementarity = max(complementarity, -r*displacement)
    certificate.slack_lower = slack_lower
    certificate.slack_upper = slack_upper
    certificate.complementarity_violation = float(complementarity)
    certificate.complementary = complementarity <= max(feasibility_tolerance,
                                                        objective_tolerance)*reference

    # ---- 强对偶 ----
    # 存储约定为 "活动量 SENSE 0"：行的活动量是 Σ a_j x_j - rhs，而目标的常数项就是它
    # 本身（目标 = Σ c_j x_j + constant）。因此对偶项用真实右端 rhs = -constant，而
    # 常数项直接用目标常数；两者都随方向 sign 缩放。
    objective_constant = model.objective.constant
    dual_terms = [sign*objective_constant]
    magnitude = abs(dual_terms[0])
    for j, row in enumerate(model.constraints):
        row_rhs = -row.expression.constant
        term = row_rhs*sign*lam[j]
        dual_terms.append(term)
        magnitude += abs(term)
    box = 0.0
    for index, var in enumerate(model.variables):
        r = float(reduced[index])
        lower, upper = math.isfinite(var.lb), math.isfinite(var.ub)
        if lower and upper:
            contribution = min(r*var.lb, r*var.ub)
        elif lower:
            contribution = r*var.lb
        elif upper:
            contribution = r*var.ub
        else:
            contribution = 0.0
        box += contribution
        magnitude += abs(contribution)
    raw_dual = math.fsum([*dual_terms, box])
    allowance = 64*np.finfo(float).eps*(1.0+magnitude+abs(raw_dual))
    dual_value = float(np.nextafter(raw_dual-allowance, -math.inf))
    certificate.dual_objective = dual_value
    certificate.objective = float(objective)
    certificate.roundoff_allowance = float(allowance)
    # 内部统一按最小化方向比较：signed_objective = sign * 原目标。
    # 最小化时界是下界（gap = obj - bound）；最大化时 sign=-1，同一式子给出 max - 上界。
    signed_objective = sign*float(objective)
    gap = signed_objective-dual_value
    certificate.duality_gap = gap
    scale = max(1.0, abs(objective))
    certificate.relative_gap = gap/scale
    certificate.strongly_dual = (-allowance) <= gap <= objective_tolerance*scale+allowance

    certificate.checks = dict(
        row_sign_convention=row_sign_ok,
        reduced_cost_domain_compatible=domain_ok,
        primal_within_tolerance=certificate.primal_feasible,
        dual_within_tolerance=certificate.dual_feasible,
        complementarity_within_tolerance=certificate.complementary,
        strong_duality_within_tolerance=certificate.strongly_dual)
    certificate.verified = all(certificate.checks.values())
    if certificate.verified:
        certificate.reason = 'all KKT and duality conditions verified in original units'
    else:
        failed = [name for name, ok in certificate.checks.items() if not ok]
        certificate.reason = 'failed conditions: '+', '.join(failed)
    return certificate
