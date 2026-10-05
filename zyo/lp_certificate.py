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

Nothing here trusts the solver that produced the candidate. The preliminary KKT check
uses ``math.fsum`` and a conservative roundoff allowance. Before a candidate is
reported ``verified``, its original row activities are also recomputed exactly from
the stored binary64 coefficients and candidate values.
"""
from dataclasses import dataclass, field
from fractions import Fraction
import math

import numpy as np

# 简约成本与域的相容性判据所用的相对尺度阈值。
DOMAIN_TOLERANCE = 1e-7
# 小模型才试探有理乘子见证；候选必须通过完整原模型精确复核。
RATIONAL_WITNESS_MAX_ROWS = 16
RATIONAL_WITNESS_MAX_VARIABLES = 32
RATIONAL_WITNESS_MAX_DENOMINATOR = 1_000_000
RATIONAL_WITNESS_PROXIMITY = Fraction(1, 10_000_000_000)


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
    witness_multipliers: list = field(default_factory=list)
    reconstruction_used: bool = False
    reconstruction_attempted: bool = False
    checks: dict = field(default_factory=dict)
    scope: str = ('floating-point KKT screening in original model units with exact '
                  'stored-binary64 row, reduced-cost and primal-dual recheck on accepted '
                  'candidates; small-model rational multiplier witnesses are explicit; '
                  'tolerance-based, not a formal exact optimality or general '
                  'infeasibility/unboundedness proof')


def _activity(expression, point):
    return math.fsum([expression.constant,
                      *[a*point[i] for i, a in expression.terms.items()]])


def _exact_stored_row_activity(expression, point):
    """按原输入的二进制浮点数精确计算一行；不先舍入各个乘积。"""
    activity = Fraction.from_float(float(expression.constant))
    for index, coefficient in expression.terms.items():
        activity += (Fraction.from_float(float(coefficient))
                     * Fraction.from_float(float(point[index])))
    return activity


def _exact_stored_kkt(model, point, lam, sign, feasibility_tolerance,
                      objective_tolerance, roundoff_allowance):
    """按存储的二进制浮点输入精确重算拟接受证书的目标、简约成本和对偶差。"""
    def f(value):
        return value if isinstance(value, Fraction) else Fraction.from_float(float(value))
    exact_point = [f(value) for value in point]
    exact_lam = [f(value) for value in lam]
    exact_sign = Fraction(int(sign))
    reduced = [Fraction(0) for _ in model.variables]
    objective = f(model.objective.constant)
    for index, coefficient in model.objective.terms.items():
        exact_cost = f(coefficient)
        objective += exact_cost*exact_point[index]
        reduced[index] += exact_sign*exact_cost
    dual = exact_sign*f(model.objective.constant)
    row_sign_ok = True
    row_sign_violation = Fraction(0)
    for j, row in enumerate(model.constraints):
        signed_lam = exact_sign*exact_lam[j]
        if row.sense == '<=':
            row_sign_violation = max(row_sign_violation, signed_lam)
        elif row.sense == '>=':
            row_sign_violation = max(row_sign_violation, -signed_lam)
        dual += -f(row.expression.constant)*signed_lam
        for index, coefficient in row.expression.terms.items():
            reduced[index] -= f(coefficient)*signed_lam

    reference = Fraction(1)+max((abs(r) for r in reduced), default=Fraction(0))
    domain_tolerance = f(DOMAIN_TOLERANCE)*reference
    complementarity_tolerance = max(f(feasibility_tolerance), f(objective_tolerance))*reference
    domain_ok = True
    dual_violation = Fraction(0)
    complementarity_violation = Fraction(0)
    for index, var in enumerate(model.variables):
        r = reduced[index]
        lower, upper = math.isfinite(var.lb), math.isfinite(var.ub)
        if lower and upper:
            lb, ub = f(var.lb), f(var.ub)
            dual += min(r*lb, r*ub)
        elif lower:
            lb = f(var.lb)
            dual += r*lb
            # 无上界时，任意严格负简约成本都使该方向的对偶下界为 -∞；
            # 数值容差可用于近似驻点诊断，不能证明有限的全局最优界。
            if r < 0:
                domain_ok = False
                dual_violation = max(dual_violation, -r)
        elif upper:
            ub = f(var.ub)
            dual += r*ub
            if r > 0:
                domain_ok = False
                dual_violation = max(dual_violation, r)
        elif r != 0:
            domain_ok = False
            dual_violation = max(dual_violation, abs(r))
        if r > domain_tolerance:
            displacement = max(Fraction(0), exact_point[index]-f(var.lb)) if lower else Fraction(1)
            complementarity_violation = max(complementarity_violation, r*displacement)
        elif r < -domain_tolerance:
            displacement = max(Fraction(0), f(var.ub)-exact_point[index]) if upper else Fraction(1)
            complementarity_violation = max(complementarity_violation, -r*displacement)

    allowance = f(roundoff_allowance)
    scale = max(Fraction(1), abs(objective))
    gap = exact_sign*objective-dual
    reported_dual = dual-allowance
    # 精确原式的负 Gap 不能被浮点求和余量掩盖：小原行误差乘以巨大乘子
    # 可产生很大的负目标偏差。负侧限额取既有余量与用户目标容差的较小值，
    # 正侧沿用用户目标容差；两侧都比旧判据更保守。
    objective_budget = f(objective_tolerance)*scale
    negative_budget = min(allowance, objective_budget)
    strongly_dual = -negative_budget <= gap <= objective_budget
    row_sign_ok = row_sign_violation == 0
    return dict(reduced=reduced, row_sign_ok=row_sign_ok,
                row_sign_violation=row_sign_violation,
                domain_ok=domain_ok, dual_violation=dual_violation,
                complementarity_violation=complementarity_violation,
                complementary=complementarity_violation <= complementarity_tolerance,
                objective=objective, reported_dual=reported_dual,
                reported_gap=gap+allowance, scale=scale, strongly_dual=strongly_dual)


def _rational_multiplier_candidate(model, lam):
    """从浮点乘子提议邻近的低分母有理数；这里仅生成候选，不发放证书。"""
    # Applegate–Cook–Dash–Espinoza (2007) §3：先试有理近似，随后完整有理验算。
    if (len(model.constraints) > RATIONAL_WITNESS_MAX_ROWS
            or len(model.variables) > RATIONAL_WITNESS_MAX_VARIABLES):
        return None
    expressions = [model.objective, *(row.expression for row in model.constraints)]
    if any(not math.isfinite(float(value))
           for expression in expressions
           for value in (expression.constant, *expression.terms.values())):
        return None
    proposed = []
    changed = False
    for value in lam:
        if not math.isfinite(float(value)):
            return None
        original = Fraction.from_float(float(value))
        candidate = original.limit_denominator(RATIONAL_WITNESS_MAX_DENOMINATOR)
        if abs(candidate-original) > RATIONAL_WITNESS_PROXIMITY*max(Fraction(1), abs(original)):
            return None
        proposed.append(candidate)
        changed |= candidate != original
    return proposed if changed else None


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

    # 原输入虽全有限，乘积本身仍可能上溢成 ±Inf；fsum 在相消时会抛错。
    # 精确原式也许可计算，但该证书入口此处不能安全报告活动量，故结构化拒绝。
    try:
        with np.errstate(over='ignore', invalid='ignore'):
            objective = _activity(model.objective, point)
            if not math.isfinite(objective):
                raise ArithmeticError('objective activity is non-finite')
            row_violation = 0.0
            activities = []
            for row in model.constraints:
                value = _activity(row.expression, point)
                if not math.isfinite(value):
                    raise ArithmeticError('row activity is non-finite')
                activities.append(value)
                violation = (abs(value) if row.sense == '==' else
                             max(0.0, value if row.sense == '<=' else -value))
                row_violation = max(row_violation, violation)
    except (ArithmeticError, ValueError, OverflowError):
        certificate.reason = 'original activity cannot be represented safely in binary64'
        certificate.checks['finite_original_activity'] = False
        return certificate

    # ---- 原始可行性 ----
    # 目标属于已核的原始候选测量：后续对偶端点上溢时仍应留存它，
    # 只拒绝最优性证书，避免辅助模块误认原始目标本身缺失。
    certificate.objective = float(objective)
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

    # fsum 只能补偿已舍入浮点乘积的加和，无法恢复乘法丢失的低位。
    # 只要将要报告原始可行，便对存储的二进制浮点原行做一次精确点积复核；
    # Phase-I 候选也会读取 primal_feasible，不能只在最终 KKT 通过后审计。
    exact_rows_checked = bool(certificate.primal_feasible)
    exact_ok = False
    if exact_rows_checked:
        try:
            exact_tolerance = Fraction.from_float(float(feasibility_tolerance))
            exact_violation = Fraction(0)
            exact_activities = []
            for row in model.constraints:
                activity = _exact_stored_row_activity(row.expression, point)
                exact_activities.append(float(activity))
                if row.sense == '==':
                    violation = abs(activity)
                elif row.sense == '<=':
                    violation = max(Fraction(0), activity)
                else:
                    violation = max(Fraction(0), -activity)
                exact_violation = max(exact_violation, violation)
            certificate.row_violation = float(exact_violation)
            certificate.row_activities = exact_activities
            exact_ok = exact_violation <= exact_tolerance
        except (OverflowError, ValueError):
            # 原行不可精确核算或精确值无法报告时保守拒绝可行性声明。
            certificate.row_violation = math.inf
            certificate.row_activities = []
        certificate.primal_feasible = bool(exact_ok and bound_violation <= feasibility_tolerance
                                           and integrality_violation <= feasibility_tolerance)

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
    try:
        with np.errstate(over='ignore', invalid='ignore'):
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
        if not all(math.isfinite(value) for value in
                   (raw_dual, magnitude, allowance, dual_value, box, *dual_terms)):
            raise ArithmeticError('non-finite dual contribution')
    except (ArithmeticError, ValueError, OverflowError):
        # 有限输入的中间 RHS×乘子或盒贡献仍会溢出；不能给出有限对偶界。
        certificate.reason = 'dual activity cannot be represented safely in binary64'
        certificate.checks['finite_dual_activity'] = False
        return certificate
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
    if exact_rows_checked:
        certificate.checks['original_rows_exact'] = bool(exact_ok)
    if all(certificate.checks.values()):
        # 目标与 A'lambda 也会发生"乘积先舍入"的抵消；最终发放证书前，
        # 对同一存储模型精确重算这些点积；浮点乘子失败后，仅小模型可
        # 提议邻近低分母有理乘子，且必须重新通过全套原模型条件。
        try:
            exact_kkt = _exact_stored_kkt(
                model, point, lam, sign, feasibility_tolerance,
                objective_tolerance, certificate.roundoff_allowance)
            reconstructed_witness = None
            exact_valid = all((exact_kkt['row_sign_ok'], exact_kkt['domain_ok'],
                               exact_kkt['complementary'], exact_kkt['strongly_dual']))
            if not exact_valid:
                proposed = _rational_multiplier_candidate(model, lam)
                if proposed is not None:
                    certificate.reconstruction_attempted = True
                    reconstructed = _exact_stored_kkt(
                        model, point, proposed, sign, feasibility_tolerance,
                        objective_tolerance, certificate.roundoff_allowance)
                    reconstructed_valid = all((reconstructed['row_sign_ok'],
                                               reconstructed['domain_ok'],
                                               reconstructed['complementary'],
                                               reconstructed['strongly_dual']))
                    if reconstructed_valid:
                        exact_kkt = reconstructed
                        reconstructed_witness = [str(value) for value in proposed]
            reduced_values = [float(value) for value in exact_kkt['reduced']]
            objective_value = float(exact_kkt['objective'])
            conservative_dual = float(np.nextafter(float(exact_kkt['reported_dual']), -math.inf))
            gap_value = float(exact_kkt['reported_gap'])
            relative_gap = float(exact_kkt['reported_gap']/exact_kkt['scale'])
            dual_violation = float(exact_kkt['dual_violation'])
            complementarity_violation = float(exact_kkt['complementarity_violation'])
            if not all(map(math.isfinite, (objective_value, conservative_dual,
                                           gap_value, relative_gap, dual_violation,
                                           complementarity_violation, *reduced_values))):
                raise OverflowError('exact KKT values cannot be reported as finite binary64')
        except (OverflowError, ValueError):
            # 精确审计无法完成时也不能沿用浮点初筛的成功状态。
            certificate.checks['stored_binary64_optimality'] = False
        else:
            if reconstructed_witness is not None:
                certificate.reconstruction_used = True
                certificate.witness_multipliers = reconstructed_witness
            certificate.reduced_costs = reduced_values
            certificate.objective = objective_value
            certificate.dual_objective = conservative_dual
            certificate.duality_gap = gap_value
            certificate.relative_gap = relative_gap
            certificate.dual_violation = dual_violation
            certificate.complementarity_violation = complementarity_violation
            certificate.row_sign_violation = float(exact_kkt['row_sign_violation'])
            certificate.dual_feasible = bool(exact_kkt['row_sign_ok'] and exact_kkt['domain_ok'])
            certificate.complementary = bool(exact_kkt['complementary'])
            certificate.strongly_dual = bool(exact_kkt['strongly_dual'])
            certificate.checks['row_sign_convention'] = bool(exact_kkt['row_sign_ok'])
            certificate.checks['reduced_cost_domain_compatible'] = bool(exact_kkt['domain_ok'])
            certificate.checks['dual_within_tolerance'] = certificate.dual_feasible
            certificate.checks['complementarity_within_tolerance'] = certificate.complementary
            certificate.checks['strong_duality_within_tolerance'] = certificate.strongly_dual
            certificate.checks['stored_binary64_optimality'] = bool(
                certificate.dual_feasible and certificate.complementary and certificate.strongly_dual)
    certificate.verified = all(certificate.checks.values())
    if certificate.verified:
        certificate.reason = ('all KKT and duality conditions verified in original units'
                              + (' using reconstructed rational row multipliers'
                                 if certificate.reconstruction_used else ''))
    else:
        failed = [name for name, ok in certificate.checks.items() if not ok]
        certificate.reason = 'failed conditions: '+', '.join(failed)
    return certificate
