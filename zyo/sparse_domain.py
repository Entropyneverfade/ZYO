# 稀疏内核对非有限变量域的显式处理：精确代换表达式，但人造宽度会截断变量域，
# 并把人造上界的使用、失效检测和原域证书回拉全部留痕，绝不静默裁剪变量域。
"""Explicit finite-box approximation and original-domain checks.

The sparse interior-point core works on a finite box. This module builds an
finite-box working problem for models whose variables are free or
one-sided bounded:

* ``lo <= x``        -> ``x = lo + z``,  ``0 <= z <= artificial``
* ``x <= up``        -> ``x = up - z``,  ``0 <= z <= artificial``
* ``x`` free         -> ``x = p - n``,   ``0 <= p, n <= artificial``
* ``lo <= x <= up``  -> ``x = lo + z``,  ``0 <= z <= up - lo``  (unchanged width)

The affine expressions preserve objective and row evaluation for reconstructed
points, but artificial finite widths truncate an originally unbounded domain.
The free-variable split is not one-to-one. Only finite original boxes retain
their exact domains. Artificial widths and attempts are recorded; an ACTIVE
artificial bound invalidates optimality because the candidate may be clipped.

The shifted/split variables are nonnegative, so a floating-point weak-duality
bound on the transformed problem pulls back to the ORIGINAL domain analytically:
the transformed box term is replaced by the original-domain one. Acceptance
depends on original-domain primal/dual and gap checks, not on success in the
artificial box alone. Free variables whose split parts both carry non-negligible
reduced costs yield no bound. Candidate multipliers start as binary64 values;
accepted row signs, domain recession signs and bounds are checked with exact
rational arithmetic on the parsed binary64 model.
"""
import math
from fractions import Fraction

import numpy as np
from lzyopt.model import Constraint, LinearExpression

# 人造上界阶梯：由小到大逐个尝试。过大的人造宽度会恶化内点法条件数（实测 1e4
# 以上出现相对间隙劣化和线性方程残差失败），过小则可能裁剪真实最优。
ARTIFICIAL_LADDER = (1.0e3, 1.0e4, 1.0e5)


def domain_kinds(model):
    """Per-variable domain kind: finite, lower, upper or free."""
    kinds = []
    for var in model.variables:
        lower = math.isfinite(var.lb)
        upper = math.isfinite(var.ub)
        kinds.append('finite' if lower and upper else
                     'lower' if lower else 'upper' if upper else 'free')
    return kinds


def needs_domain_transform(model):
    """True only when at least one variable lacks a finite box."""
    return any(kind != 'finite' for kind in domain_kinds(model))


def build_transformed(model, artificial):
    """Build the finite-box working model and reconstruction expressions."""
    from .model import Model

    columns = []
    out = Model()
    for var in model.variables:
        lower = var.lb if math.isfinite(var.lb) else None
        upper = var.ub if math.isfinite(var.ub) else None
        if lower is not None and upper is not None:
            probe = out.add_var(f'{var.name}#d', lb=0.0, ub=float(upper-lower))
            column = dict(kind='finite', base=float(lower), artificial=None, entries=[(probe, 1.0)])
        elif lower is not None:
            probe = out.add_var(f'{var.name}#l', lb=0.0, ub=float(artificial))
            column = dict(kind='lower', base=float(lower), artificial=float(artificial), entries=[(probe, 1.0)])
        elif upper is not None:
            probe = out.add_var(f'{var.name}#u', lb=0.0, ub=float(artificial))
            column = dict(kind='upper', base=float(upper), artificial=float(artificial), entries=[(probe, -1.0)])
        else:
            positive = out.add_var(f'{var.name}#p', lb=0.0, ub=float(artificial))
            negative = out.add_var(f'{var.name}#n', lb=0.0, ub=float(artificial))
            column = dict(kind='free', base=0.0, artificial=float(artificial),
                          entries=[(positive, 1.0), (negative, -1.0)])
        column['name'] = var.name
        columns.append(column)

    def rewrite(expression):
        constant = expression.constant
        terms = {}
        for index, coefficient in expression.terms.items():
            column = columns[index]
            constant += coefficient*column['base']
            for var, sign in column['entries']:
                terms[var.index] = terms.get(var.index, 0.0)+coefficient*sign
        built = LinearExpression(out, {k: v for k, v in terms.items() if v != 0.0}, constant)
        return built

    for row in model.constraints:
        out.add_constr(Constraint(rewrite(row.expression), row.sense))
    out.set_objective(rewrite(model.objective), model.sense)
    return out, columns


def recover_primal(columns, values):
    """Map transformed values back to the original variables (exact inverse)."""
    return np.array([column['base']+math.fsum(sign*float(values[var.name])
                                              for var, sign in column['entries'])
                     for column in columns])


def active_artificial(columns, values, feasibility_tol):
    """Names of variables whose ARTIFICIAL upper bound is reached.

    Only widths that we invented count. An active artificial bound means the
    candidate may have been clipped by a device of our own making, so optimality
    must not be claimed by the caller.
    """
    active = []
    for column in columns:
        width = column['artificial']
        if width is None:
            continue
        used = max(abs(float(values[var.name])) for var, _ in column['entries'])
        if used >= width*(1.0-feasibility_tol):
            active.append(column['name'])
    return active


def original_multipliers(certificate):
    """Original-model row multipliers carried by a certificate.

    ``box_bound`` records the multipliers it actually evaluated in original model
    rows and units, so those are authoritative. Re-deriving them from the
    standardized solution would require undoing the per-row scaling of the
    standardized problem, and getting that scale wrong silently multiplies the
    multipliers (and therefore the bound) by the box width.
    """
    multipliers = certificate.get('multipliers')
    if multipliers is None:
        return None
    return np.asarray(multipliers, dtype=float)


def domain_bound(model, multipliers):
    """Weak-duality bound over the ORIGINAL (possibly unbounded) domains.

    For minimization with ``lambda <= 0`` on ``<=`` rows, ``lambda >= 0`` on ``>=``
    rows and free equality multipliers, the Lagrangian dual function at ``lambda``
    is ``c0 + lambda'b + sum_j min_{x in D_j} r_j x`` with ``r = c - A'lambda``.

    The in-domain minimum is finite only when the reduced cost is compatible with
    the variable's domain: ``r >= 0`` for a lower-bounded variable, ``r <= 0`` for
    an upper-bounded one, and ``r == 0`` exactly for a free one. An infinite
    Lagrangian infimum for one multiplier does NOT prove primal unboundedness:
    the original rows may block that direction. All signs and terms below use
    exact rationals of the parsed binary64 inputs; the float bound rounds down.
    """
    lam = np.asarray(multipliers, dtype=float)
    if lam.shape != (len(model.constraints),) or not np.all(np.isfinite(lam)):
        return dict(available=False, reason='multiplier count or magnitude is invalid')
    for row, value in zip(model.constraints, lam):
        if (row.sense == '<=' and value > 0) or (row.sense == '>=' and value < 0):
            return dict(available=False, reason='multiplier sign is incompatible with the original row')
    direction = 1 if model.sense == 'min' else -1
    def rational(value):
        return Fraction.from_float(float(value))

    def evaluate(weights):
        # 恢复乘子后重新逐项精确计算，不能把近零浮点简约成本直接截成零。
        reduced = [Fraction(0) for _ in model.variables]
        for index, coefficient in model.objective.terms.items():
            reduced[index] = direction*rational(coefficient)
        rhs_terms = []
        for row, weight in zip(model.constraints, weights):
            if (row.sense == '<=' and weight > 0) or (row.sense == '>=' and weight < 0):
                return None, [], ['row multiplier sign']
            rhs_terms.append(-rational(row.expression.constant)*weight)
            for index, coefficient in row.expression.terms.items():
                reduced[index] -= rational(coefficient)*weight
        box = []
        invalid = []
        for index, var in enumerate(model.variables):
            cost = reduced[index]
            lower, upper = math.isfinite(var.lb), math.isfinite(var.ub)
            if lower and upper:
                box.append(min(cost*rational(var.lb), cost*rational(var.ub)))
            elif lower:
                if cost < 0:
                    invalid.append(var.name)
                else:
                    box.append(cost*rational(var.lb))
            elif upper:
                if cost > 0:
                    invalid.append(var.name)
                else:
                    box.append(cost*rational(var.ub))
            elif cost != 0:
                invalid.append(var.name)
            else:
                box.append(Fraction(0))
        if invalid:
            return None, [], invalid
        return direction*rational(model.objective.constant)+sum(rhs_terms)+sum(box), box, []

    try:
        original = [rational(value) for value in lam]
        # 小分母有理数恢复只提出另一份候选乘子；每一项仍经过完整原域精确核验。
        # 恢复失败时保持 UNKNOWN，不改变求解容差或把近零数直接当作数学零。
        reconstructed = [value.limit_denominator(1_000_000) for value in original]
        candidates = [('original_binary64', original)]
        if reconstructed != original:
            candidates.append(('rational_reconstruction_1e6', reconstructed))
        valid = []
        invalid = []
        for source, weights in candidates:
            exact, box, failed = evaluate(weights)
            if exact is None:
                invalid.extend(failed)
            else:
                valid.append((exact, box, source, weights))
        if not valid:
            # 某组乘子的拉格朗日下确界为 -∞，只说明它不给出有效界；
            # 原约束未必允许沿该列移动，因此绝不能据此报告原 LP 无界。
            return dict(available=False, can_improve_without_limit=False,
                        unbounded_lagrangian=True, variables=sorted(set(invalid)),
                        reason='Proposed multipliers have no finite original-domain '
                               'Lagrangian bound; no primal unbounded ray is proved')
        exact, box, source, weights = max(valid, key=lambda item: item[0])
        raw = float(exact)
        if not math.isfinite(raw):
            raise OverflowError('Exact bound lies outside finite binary64 range')
        bound = raw
        if rational(bound) > exact:
            bound = math.nextafter(bound, -math.inf)
        if not math.isfinite(bound):
            raise OverflowError('Rounded bound lies outside finite binary64 range')
        terms = [float(value) for value in box]
        allowance = float(exact-rational(bound))
    except (ValueError, OverflowError, ZeroDivisionError) as exc:
        return dict(available=False, reason='Exact original-domain bound arithmetic failed: '+str(exc))
    return dict(available=True, bound=bound, raw_bound=raw, roundoff_allowance=allowance,
                domain_box_term=terms, multipliers=[float(value) for value in weights],
                multipliers_exact=[str(value) for value in weights], multiplier_source=source,
                scope='exact-rational normalized-min weak duality on parsed binary64 data; '
                      'normalized bound rounds down and an original max bound rounds up')
