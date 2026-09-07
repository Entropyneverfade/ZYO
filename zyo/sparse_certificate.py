# 原始有限盒上的弱对偶界与充分不可行证书；使用浮点保护，不是精确算术证明器。
"""Original-unit Lagrangian box bounds, independent of the sparse LP algorithm.

For minimization: lambda <= 0 for <= rows, >= 0 for >= rows; equality
multipliers are free. Any such lambda gives a bound even with nonzero reduced
costs. All variable bounds must be finite. This is floating-point verification,
with a conservative arithmetic allowance, not exact rational certification.
"""
import math
import numpy as np


def objective_value(model,x):
    # 使用原目标系数及常数复算，不采用标准化问题的目标值。
    return math.fsum([model.objective.constant,*[a*x[i] for i,a in model.objective.terms.items()]])


def primal_check(model,x):
    # 残差按原始单位计算；返回有限容差意义下的违反量，不等于最优性证明。
    x=np.asarray(x,dtype=float)
    if x.shape!=(len(model.variables),) or not np.all(np.isfinite(x)):
        return dict(constraint_violation=math.inf,bound_violation=math.inf,integrality_violation=math.inf)
    constraint=0.; bound=0.; integer=0.
    for row in model.constraints:
        activity=math.fsum([row.expression.constant,*[a*x[i] for i,a in row.expression.terms.items()]])
        violation=abs(activity) if row.sense=='==' else max(0.,activity if row.sense=='<=' else -activity)
        constraint=max(constraint,violation)
    for i,v in enumerate(model.variables):
        bound=max(bound,v.lb-x[i],x[i]-v.ub)
        if v.kind!='C': integer=max(integer,abs(x[i]-np.rint(x[i])))
    return dict(constraint_violation=constraint,bound_violation=bound,integrality_violation=integer)


@np.errstate(over='raise',invalid='raise',divide='raise',under='raise')
def box_bound(model, multipliers, lower=None, upper=None):
    # 统一最小化下界：d*c0 + λ'b + Σ min((d*c-A'λ)*lo, (d*c-A'λ)*hi)。
    # <= 行要求 λ<=0，>= 行要求 λ>=0，等式乘子自由；扣除舍入保护后才可作为节点界。
    lam=np.asarray(multipliers,dtype=float)
    lo=np.array([v.lb for v in model.variables]) if lower is None else np.asarray(lower,dtype=float)
    hi=np.array([v.ub for v in model.variables]) if upper is None else np.asarray(upper,dtype=float)
    if (lam.shape!=(len(model.constraints),) or lo.shape!=(len(model.variables),)
            or hi.shape!=lo.shape or not np.all(np.isfinite(lam))
            or not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)) or np.any(lo>hi)):
        raise ValueError('Finite multipliers and valid finite variable boxes required')
    direction=1. if model.sense=='min' else -1.
    reduced=np.zeros(len(lo))
    for i,c in model.objective.terms.items():
        reduced[i]=direction*c
    box=np.maximum(np.abs(lo),np.abs(hi))
    magnitude=float(np.abs(reduced)@box)+abs(model.objective.constant)
    if not np.any(lam):
        # 零乘子时没有 A'λ 相减误差，只保护实际选取的端点乘积和求和；仍需保守舍入。
        magnitude=float(np.sum(np.abs(np.minimum(reduced*lo,reduced*hi))))+abs(model.objective.constant)
    rhs_terms=[]
    for j,row in enumerate(model.constraints):
        # 保留 NumPy 标量乘法以捕捉下溢；极小系数乘巨大端点可能产生不可忽略影响。
        value=lam[j]
        if (row.sense=='<=' and value>0) or (row.sense=='>=' and value<0):
            raise ValueError('Multiplier sign incompatible with original row')
        rhs_terms.append(-row.expression.constant*value)
        magnitude+=abs(row.expression.constant*value)
        for i,a in row.expression.terms.items():
            reduced[i]-=a*value
            magnitude+=abs(a*value)*box[i]
    raw=math.fsum([direction*model.objective.constant,*rhs_terms,
                   *np.minimum(reduced*lo,reduced*hi)])
    # 误差与每列累加深度有关；fsum 控制最终求和，额外保护覆盖乘法和 A'λ 相减。
    column_counts=np.zeros(len(lo),dtype=int)
    for row in model.constraints:
        for i in row.expression.terms:
            column_counts[i]+=1
    operations=max(64,int(np.max(column_counts,initial=0))+8)
    allowance=operations*np.finfo(float).eps*(1.+magnitude+abs(raw))
    bound=float(np.nextafter(raw-allowance,-np.inf))
    if not math.isfinite(bound):
        raise ValueError('Bound arithmetic is non-finite')
    return dict(bound=bound,raw_bound=raw,roundoff_allowance=allowance,
                direction=direction,scope='floating-point original-box weak-duality bound')


@np.errstate(over='raise',invalid='raise',divide='raise',under='raise')
def box_infeasibility(model, multipliers, lower=None, upper=None, *, feasibility_tol=1e-7):
    """Check a Farkas separation witness against original rows and a finite box.

    Row signs follow box_bound, but the OBJECTIVE IS NEVER USED. Any feasible
    x must satisfy lambda.T A x >= lambda.T b. If even the maximum over the
    box is smaller, the box has no feasible point. Normalize the proposed ray
    before arithmetic; subtract roundoff and account for the same primal
    tolerance on rows AND variable bounds. A non-certificate is not evidence
    of feasibility. This is a sufficient floating-point check, not exact math.
    Underflow must fail closed too: a tiny A'lambda product can have a large
    effect after multiplication by a huge box endpoint. Relative roundoff
    allowances alone cannot protect an underflowed coefficient.
    """
    lam=np.asarray(multipliers,dtype=float).copy()
    # 不可行证书不读取目标函数；归一化候选射线，再验证当前有限盒与原约束是否分离。
    lo=np.array([v.lb for v in model.variables]) if lower is None else np.asarray(lower,dtype=float)
    hi=np.array([v.ub for v in model.variables]) if upper is None else np.asarray(upper,dtype=float)
    if (lam.shape!=(len(model.constraints),) or lo.shape!=(len(model.variables),)
            or hi.shape!=lo.shape or not np.all(np.isfinite(lam))
            or not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)) or np.any(lo>hi)
            or not math.isfinite(feasibility_tol) or feasibility_tol<0):
        raise ValueError('Finite multipliers, finite valid boxes and nonnegative finite tolerance required')
    for value,row in zip(lam,model.constraints):
        if (row.sense=='<=' and value>0) or (row.sense=='>=' and value<0):
            raise ValueError('Multiplier sign incompatible with original row')
    norm=float(np.max(np.abs(lam),initial=0))
    if norm:
        lam/=norm
    terms=[[] for _ in lo]
    rhs=[]; magnitude_terms=[]
    # 同时考虑行容差和边界容差扩张；只有分离裕度超过容差量与舍入保护才可拒绝该节点。
    box=np.maximum(np.abs(lo),np.abs(hi))+feasibility_tol
    for value,row in zip(lam,model.constraints):
        rhs.append(-row.expression.constant*value)
        magnitude_terms.append(abs(rhs[-1]))
        for i,a in row.expression.terms.items():
            product=-a*value
            terms[i].append(product)
            magnitude_terms.append(abs(product)*box[i])
    reduced=np.array([math.fsum(column) for column in terms])
    endpoints=np.minimum(reduced*lo,reduced*hi)
    raw=math.fsum([*rhs,*endpoints])
    tolerance=feasibility_tol*math.fsum([*[abs(v) for v in lam],*[abs(v) for v in reduced]])
    operations=max(64,max((len(column) for column in terms),default=0)+8)
    allowance=operations*np.finfo(float).eps*(1.+math.fsum(magnitude_terms)+abs(raw)+tolerance)
    margin=float(np.nextafter(raw-allowance,-np.inf))
    if not all(math.isfinite(v) for v in (raw,tolerance,allowance,margin)):
        raise ValueError('Infeasibility certificate arithmetic is non-finite')
    return dict(kind='original_box_farkas',verified=bool(margin>tolerance),
                raw_margin=raw,margin=margin,roundoff_allowance=allowance,
                tolerance_allowance=tolerance,feasibility_tol=feasibility_tol,
                multipliers=lam.tolist(),lower=lo.tolist(),upper=hi.tolist(),
                scope='sufficient floating-point original-box separation; no objective or optimizer used')
