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
reduced costs yield no bound. These are floating-point checks, not exact proofs.
"""
import math

import numpy as np
from lzyopt.model import Constraint, LinearExpression

# 人造上界阶梯：由小到大逐个尝试。过大的人造宽度会恶化内点法条件数（实测 1e4
# 以上出现相对间隙劣化和线性方程残差失败），过小则可能裁剪真实最优。
ARTIFICIAL_LADDER = (1.0e3, 1.0e4, 1.0e5)

# 自由变量拆分中"可忽略的简约成本"判定：两分量都不可忽略时原域下确界为 -∞。
FREE_SPLIT_REDUCED_TOL = 1e-6


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


def _scale_of(model, reduced):
    """Magnitude reference for judging whether a reduced cost is numerically zero."""
    total = abs(model.objective.constant) if hasattr(model.objective, 'constant') else 0.0
    for coefficient in model.objective.terms.values():
        total += abs(coefficient)
    return 1.0+total+float(np.max(np.abs(reduced), initial=0.0))


def domain_bound(model, multipliers):
    """Weak-duality bound over the ORIGINAL (possibly unbounded) domains.

    For minimization with ``lambda <= 0`` on ``<=`` rows, ``lambda >= 0`` on ``>=``
    rows and free equality multipliers, the Lagrangian dual function at ``lambda``
    is ``c0 + lambda'b + sum_j min_{x in D_j} r_j x`` with ``r = c - A'lambda``.

    The in-domain minimum is finite only when the reduced cost is compatible with
    the variable's domain: ``r >= 0`` for a lower-bounded variable, ``r <= 0`` for
    an upper-bounded one, and ``r == 0`` (numerically) for a free one. When that
    fails along a variable that has no bound in the improving direction, the
    objective is unbounded below, which is reported as ``improving_ray`` instead of
    silently returning -inf or a clipped optimum. Returns ``available=False`` with
    a reason when no finite bound can be justified.
    """
    lam = np.asarray(multipliers, dtype=float)
    if lam.shape != (len(model.constraints),) or not np.all(np.isfinite(lam)):
        return dict(available=False, reason='multiplier count or magnitude is invalid')
    for row, value in zip(model.constraints, lam):
        if (row.sense == '<=' and value > 0) or (row.sense == '>=' and value < 0):
            return dict(available=False, reason='multiplier sign is incompatible with the original row')
    direction = 1.0 if model.sense == 'min' else -1.0
    reduced = np.zeros(len(model.variables))
    for index, coefficient in model.objective.terms.items():
        reduced[index] = direction*coefficient
    magnitude = abs(direction*model.objective.constant)
    dual_terms = []
    for row, value in zip(model.constraints, lam):
        dual_terms.append(-row.expression.constant*value)
        magnitude += abs(row.expression.constant*value)
        for i, coefficient in row.expression.terms.items():
            reduced[i] -= coefficient*value
    reference = _scale_of(model, reduced)
    # 域可行性（fail-closed）：简约成本与变量域不相容时，下确界不是有限值。
    # 对下界型变量要求 r>=0；上界型要求 r<=0；自由变量要求 r 数值为零。
    box = np.zeros(len(model.variables))
    improving = []
    for index, var in enumerate(model.variables):
        r = reduced[index]
        zero = abs(r) <= FREE_SPLIT_REDUCED_TOL*reference
        if math.isfinite(var.lb) and math.isfinite(var.ub):
            box[index] = min(r*var.lb, r*var.ub)
        elif math.isfinite(var.lb):
            if r < -FREE_SPLIT_REDUCED_TOL*reference:
                improving.append(var.name)
                continue
            box[index] = r*var.lb
        elif math.isfinite(var.ub):
            if r > FREE_SPLIT_REDUCED_TOL*reference:
                improving.append(var.name)
                continue
            box[index] = r*var.ub
        else:
            if not zero:
                improving.append(var.name)
                continue
            box[index] = 0.0
        magnitude += abs(box[index])
    if improving:
        return dict(available=False, can_improve_without_limit=True, variables=improving,
                    reason='reduced cost is incompatible with the variable domain, so the '
                           'objective improves without limit along that direction',
                    reduced_costs=reduced.tolist())
    raw = math.fsum([direction*model.objective.constant, *dual_terms, *box])
    operations = 64+max((len(row.expression.terms) for row in model.constraints), default=0)
    allowance = operations*np.finfo(float).eps*(1.0+magnitude+abs(raw))
    bound = float(np.nextafter(raw-allowance, -np.inf))
    if not math.isfinite(bound):
        return dict(available=False, reason='original-domain bound arithmetic is non-finite')
    return dict(available=True, bound=bound, raw_bound=raw, roundoff_allowance=allowance,
                domain_box_term=box.tolist(), multipliers=lam.tolist(),
                scope='floating-point original-domain weak-duality bound over free/one-sided variables')
