# 自编不可行初值原始—对偶预测校正内点法；迭代失败本身不能证明不可行或无界。
"""ZYO-authored sparse infeasible-start primal-dual predictor/corrector LP.

Only NumPy and SciPy sparse/SuperLU linear algebra are used, no optimizer.
Finite variable boxes are required in this experimental path. Infeasible-start
iteration failure is NOT an infeasibility or unboundedness certificate.
"""
from dataclasses import dataclass, field, replace
import math
import time
import numpy as np
from scipy.sparse import coo_matrix, diags, eye, bmat
from scipy.sparse.linalg import splu

from .sparse_certificate import box_bound, box_infeasibility, primal_check, objective_value


@dataclass
class SparseLPResult:
    status: str
    x: object=None
    objective: float | None=None  # normalized minimization direction
    bound: float | None=None
    multipliers: object=None  # original constraint rows, minimization direction
    iterations: int=0
    message: str=''
    history: list=field(default_factory=list)
    certificate: object=None
    infeasibility_certificate: object=None
    feasibility_recovery: object=None  # 辅助问题的真实预算、状态和乘子来源。


def _standard_form(model,lo,hi):
    # 以 x=lo+(hi-lo)*z 映射有限盒至 [0,1]，固定变量消去；不等式加入带正确符号的松弛。
    """Finite-box affine map and explicit nonnegative inequality slacks."""
    width=hi-lo
    active=np.flatnonzero(width>0)
    index=np.full(len(lo),-1,dtype=int)
    index[active]=np.arange(len(active))
    rows=[]; cols=[]; data=[]; rhs=[]; origins=[]; scales=[]
    slack_rows=[]
    for original,row in enumerate(model.constraints):
        b=-math.fsum([row.expression.constant,*[a*lo[i] for i,a in row.expression.terms.items()]])
        terms=[(index[i],a*width[i]) for i,a in row.expression.terms.items() if index[i]>=0]
        scale=max([abs(a) for _,a in terms],default=0.)
        if scale==0:
            violation=abs(b) if row.sense=='==' else max(0.,-b if row.sense=='<=' else b)
            if violation>0:
                return None,'Exact constant row contradiction'
            continue
        j=len(rhs)
        for i,a in terms:
            rows.append(j); cols.append(i); data.append(a/scale)
        rhs.append(b/scale); origins.append(original); scales.append(scale)
        if row.sense!='==':
            slack_rows.append((j,1. if row.sense=='<=' else -1.))
    for i in range(len(active)):
        j=len(rhs)
        rows.append(j); cols.append(i); data.append(1.)
        rhs.append(1.); origins.append(-1); scales.append(1.)
        slack_rows.append((j,1.))
    for k,(j,sign) in enumerate(slack_rows):
        rows.append(j); cols.append(len(active)+k); data.append(sign)
    matrix=coo_matrix((data,(rows,cols)),shape=(len(rhs),len(active)+len(slack_rows))).tocsc()
    cost=np.zeros(matrix.shape[1])
    direction=1. if model.sense=='min' else -1.
    for i,c in model.objective.terms.items():
        if index[i]>=0:
            cost[index[i]]=direction*c*width[i]
    cost_scale=max(1.,float(np.max(np.abs(cost),initial=0)))
    return dict(A=matrix,b=np.asarray(rhs),c=cost/cost_scale,cost_scale=cost_scale,
                active=active,width=width,origins=np.asarray(origins),scales=np.asarray(scales)),None


def _step(x,dx,fraction=1.):
    # 计算保持正变量不越界的最大步长；校正步使用小于 1 的比例留在严格内部。
    negative=dx<0
    return min(1.,fraction*float(np.min(-x[negative]/dx[negative]))) if np.any(negative) else 1.


def _normal_solver(A,ratio):
    # 正规方程 A*diag(x/s)*A' 的缩放与分解；正则化仅帮助解线性方程，不改变优化模型。
    matrix=(A.multiply(ratio)@A.T).tocsc()
    scale=1./np.sqrt(np.maximum(matrix.diagonal(),1e-30))
    equilibrated=(diags(scale)@matrix@diags(scale)).tocsc()
    factors={}
    statistics={'regularization':0.}
    def solve(rhs):
        for ridge in (0.,1e-12,1e-10,1e-8):
            if ridge not in factors:
                try:
                    factors[ridge]=splu(equilibrated+ridge*eye(A.shape[0],format='csc'),permc_spec='MMD_AT_PLUS_A')
                except RuntimeError:
                    factors[ridge]=None
            lu=factors[ridge]
            if lu is None:
                continue
            value=scale*lu.solve(scale*rhs)
            # 迭代改进始终针对未正则化方程；不能通过较大正则项掩盖原方程残差。
            for _ in range(3):
                residual=rhs-matrix@value
                if np.max(np.abs(residual),initial=0)<=1e-8*(1+np.max(np.abs(rhs),initial=0)):
                    statistics['regularization']=ridge
                    return value
                value+=scale*lu.solve(scale*residual)
            residual=rhs-matrix@value
            if np.max(np.abs(residual),initial=0)>1e-6*(1+np.max(np.abs(rhs),initial=0)):
                continue
            statistics['regularization']=ridge
            return value
        raise ArithmeticError('Sparse Newton equations failed residual check for all factorizations')
    return solve,statistics


def _augmented_solver(A,x,s,rp,rd,statistics=None,stabilize=None):
    """Solve the unsquared Newton system with symmetric diagonal scaling.

Normal equations can lose rank numerically as x/s separates near a vertex.
This augmented system avoids forming A*diag(x/s)*A', and checks the original
Newton equations. A positive dual block stabilizes the factorization even
when rows are dependent; refinement and acceptance use the UNREGULARIZED
equations. No regularization changes the optimization model.
"""
    n=len(x)
    # 增广系统避免直接形成正规方程造成条件数平方；方向还须通过原始牛顿方程检查。
    scale_x=np.sqrt(x/s)
    column_scaled=A.multiply(scale_x)
    scale_rows=1./np.sqrt(np.maximum(np.asarray(column_scaled.power(2).sum(axis=1)).ravel(),1e-30))
    C=(diags(scale_rows)@column_scaled).tocsc()
    K=bmat([[-eye(n,format='csc'),C.T],[C,None]],format='csc')
    statistics={} if statistics is None else statistics
    lu=None
    if stabilize is None or not stabilize:
        try:
            lu=splu(K,permc_spec='COLAMD')
        except RuntimeError:
            if stabilize is False:
                raise
    if stabilize is None:
        # 仅在LP初始 x=s=1 时判断分解风险，避免将临近顶点的正常尺度分离误当成初始秩亏。
        # U主元比例只是选择数值路径的启发式，不是删行依据，也不声明精确矩阵秩。
        pivots=np.abs(lu.U.diagonal()) if lu is not None else np.array([0.])
        pivot_ratio=float(np.min(pivots,initial=1.)/max(1.,float(np.max(pivots,initial=0))))
        stabilize=lu is None or pivot_ratio<=64*np.finfo(float).eps
        statistics['initial_pivot_ratio']=pivot_ratio
    statistics['stabilized']=bool(stabilize)
    # K 的右下零块在等式相关时导致奇异或巨大零空间乘子。正对角块使 Schur 补
    # C*C' + ridge*I 正定；这是线性方程预条件分解，不是给原约束/目标加惩罚。
    # 单位行范数缩放后使用 sqrt(eps) 平衡舍入放大与扰动，随后用原 K 消除扰动。
    ridge=math.sqrt(np.finfo(float).eps) if stabilize else 0.
    if stabilize:
        regularized=K+diags(np.r_[np.zeros(n),np.full(A.shape[0],ridge)])
        lu=splu(regularized.tocsc(),permc_spec='COLAMD')
    unregularized=None
    unregularized_statistics={}
    def solve(rc):
        nonlocal unregularized
        rhs=np.r_[scale_x*(rd-rc/x),scale_rows*rp]
        step=lu.solve(rhs)
        original_refinements=0
        if not stabilize:
            # 初始分解没有秩亏风险时，保留原来的数值轨迹，避免无条件扰动正常LP/不可行分支。
            for _ in range(3):
                defect=rhs-K@step
                if np.max(np.abs(defect),initial=0)<=1e-11*(1+np.max(np.abs(rhs),initial=0)):
                    break
                step+=lu.solve(defect)
                original_refinements+=1
        for refinement in range(9 if stabilize else 1):
            dx=scale_x*step[:n]; dy=scale_rows*step[n:]
            ds=rd-A.T@dy
            primal=float(np.max(np.abs(A@dx-rp),initial=0))
            complement=float(np.max(np.abs(s*dx+x*ds-rc),initial=0))
            # 不能只看缩放/正则化系统残差：还原方向后必须满足原有两项1e-9门。
            if (all(np.all(np.isfinite(v)) for v in (dx,dy,ds))
                    and primal<=1e-9*(1+np.max(np.abs(rp),initial=0))
                    and complement<=1e-9*(1+np.max(np.abs(rc),initial=0))):
                statistics.update(regularization=ridge,refinement_steps=original_refinements+refinement,
                                  unregularized_fallback=False)
                return dx,dy,ds
            if stabilize and refinement<8:
                step+=lu.solve(rhs-K@step)
        if stabilize:
            # 主元启发式也会选中近相关的满秩矩阵；若稳定化改进受限，仍尝试原分解，
            # 并复用相同的原方程门。两种分解均为基础线性代数，实际采用路径写入日志。
            try:
                if unregularized is None:
                    unregularized=_augmented_solver(A,x,s,rp,rd,unregularized_statistics,False)
                step=unregularized(rc)
            except RuntimeError as exc:
                raise ArithmeticError('Augmented factorization alternatives exhausted') from exc
            statistics.update(unregularized_statistics,unregularized_fallback=True)
            return step
        # 不一致的相关行不能靠正则化通过；拒绝方向，由原有失败/证书逻辑处理。
        raise ArithmeticError('Augmented Newton equations failed original residual check')
    return solve


def solve_relaxation(model,lower,upper,options,deadline):
    # 算术异常只返回数值未决；不可行判断需要单独的充分证书。
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            result = _solve_relaxation(model,lower,upper,options,deadline)
    except (ArithmeticError,ValueError) as exc:
        result = SparseLPResult('NUMERICAL_ERROR',message='LP arithmetic unresolved: '+str(exc))
    if result.status == 'NUMERICAL_ERROR':
        return _recover_infeasibility(model,lower,upper,options,deadline,result)
    return result


def _recover_infeasibility(model,lower,upper,options,deadline,result):
    # 仅在旧射线检查未成功的数值失败后尝试一次；直调同一内核避免辅助问题递归恢复。
    # 辅助解、目标和界均不能替代原 LP；只有原始模型充分证书可以关闭节点。
    from .sparse_phase_one import build_phase_one, original_multipliers
    started = time.perf_counter()
    record = dict(method='native_elastic_phase_one',trigger_status=result.status,
                  trigger_reason=result.message,original_iterations=result.iterations,
                  iteration_budget=max(0,options.iteration_limit-result.iterations),
                  auxiliary_status=None,auxiliary_iterations=0,certificate_verified=False)
    result.feasibility_recovery = record
    try:
        if started >= deadline:
            raise TimeoutError('Phase-I shared time budget exhausted')
        if record['iteration_budget'] == 0:
            record['skipped'] = 'shared iteration budget exhausted'
            return result
        auxiliary, mapping = build_phase_one(model,lower,upper,deadline)
        record.update(variables=len(auxiliary.variables),constraints=len(auxiliary.constraints))
        aux_options = replace(options,iteration_limit=record['iteration_budget'])
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            phase = _solve_relaxation(auxiliary,[v.lb for v in auxiliary.variables],
                                      [v.ub for v in auxiliary.variables],aux_options,deadline)
        result.iterations += phase.iterations
        record.update(auxiliary_status=phase.status,auxiliary_iterations=phase.iterations,
                      auxiliary_reason=phase.message,auxiliary_objective=phase.objective,
                      auxiliary_history=phase.history)
        if phase.status == 'TIME_LIMIT' or time.perf_counter() >= deadline:
            raise TimeoutError('Phase-I shared time budget exhausted')
        if phase.status == 'ITERATION_LIMIT':
            result.status = 'ITERATION_LIMIT'
            result.message = 'Shared LP/Phase-I iteration limit; original node unresolved'
        elif phase.status == 'OPTIMAL' and phase.multipliers is not None:
            lam = original_multipliers(model,mapping,phase.multipliers)
            proof = box_infeasibility(model,lam,lower,upper,feasibility_tol=options.feasibility_tol)
            record['certificate_verified'] = proof['verified']
            if proof['verified']:
                proof.update(trigger_status=record['trigger_status'],trigger_reason=record['trigger_reason'],
                             candidate_source='native_elastic_phase_one')
                result = SparseLPResult('INFEASIBLE',iterations=result.iterations,history=result.history,
                                        message='Native Phase-I original-box Farkas separation verified',
                                        infeasibility_certificate=proof,feasibility_recovery=record)
    except TimeoutError as exc:
        result.status = 'TIME_LIMIT'
        result.message = str(exc)
        record['recovery_error'] = str(exc)
    except (ArithmeticError,ValueError,RuntimeError) as exc:
        record['recovery_error'] = str(exc)
    finally:
        finished = time.perf_counter()
        record['runtime_seconds'] = finished-started
        # 回映射、原始证书扫描和异常处理也耗时；越过共同截止时刻不能关闭节点。
        if finished >= deadline:
            result.status = 'TIME_LIMIT'
            result.message = 'Phase-I shared time budget exhausted'
            result.infeasibility_certificate = None
            record['recovery_error'] = result.message
        record['certificate_accepted'] = result.status == 'INFEASIBLE'
    return result


def _solve_relaxation(model,lower,upper,options,deadline):
    lo=np.asarray(lower,dtype=float); hi=np.asarray(upper,dtype=float)
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        return SparseLPResult('UNKNOWN',message='native_sparse requires finite lower and upper bounds')
    if np.any(lo>hi):
        return SparseLPResult('INFEASIBLE',message='Contradictory node bounds')
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            for row in model.constraints:
                low_terms=[row.expression.constant,*[a*(lo[i] if a>=0 else hi[i]) for i,a in row.expression.terms.items()]]
                high_terms=[row.expression.constant,*[a*(hi[i] if a>=0 else lo[i]) for i,a in row.expression.terms.items()]]
                minimum=math.fsum(low_terms); maximum=math.fsum(high_terms)
                error_allowance=16*np.finfo(float).eps*(1+math.fsum(abs(v) for v in low_terms+high_terms))
                if ((row.sense in ('<=','==') and minimum-error_allowance>options.feasibility_tol)
                        or (row.sense in ('>=','==') and maximum+error_allowance < -options.feasibility_tol)):
                    return SparseLPResult('INFEASIBLE',message='Original row outward interval excludes feasibility')
            standard,error=_standard_form(model,lo,hi)
    except (ArithmeticError,ValueError) as exc:
        return SparseLPResult('NUMERICAL_ERROR',message='Preprocessing arithmetic: '+str(exc))
    if error:
        return SparseLPResult('NUMERICAL_ERROR',message=error)
    A,b,c=standard['A'],standard['b'],standard['c']
    n=A.shape[1]
    history=[]
    direction=1. if model.sense=='min' else -1.
    if n==0:
        objective=direction*objective_value(model,lo)
        lam=np.zeros(len(model.constraints))
        certificate=box_bound(model,lam,lo,hi)
        checked=primal_check(model,lo)
        gap=(objective-certificate['bound'])/max(1.,abs(objective))
        status='OPTIMAL' if max(checked.values())<=options.feasibility_tol and gap<=options.objective_tol else 'NUMERICAL_ERROR'
        return SparseLPResult(status,lo.copy(),objective,certificate['bound'],lam,
                              message='Fixed-box primal and dual arithmetic checked',certificate=certificate)
    x=np.ones(n); s=np.ones(n); y=np.zeros(len(b))
    stabilize_augmented=None
    # 允许初始点原始/对偶不可行，但 x、s 保持正；rp、rd、mu 分别为两类残差和平均互补量。
    zero_multipliers=np.zeros(len(model.constraints))
    simple_certificate=box_bound(model,zero_multipliers,lo,hi)
    failure='ITERATION_LIMIT'; message='LP iteration limit; optimality not established'
    last=None
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            for iteration in range(options.iteration_limit+1):
                if time.perf_counter()>=deadline:
                    failure='TIME_LIMIT'; message='LP time limit'; break
                rp=b-A@x; rd=c-A.T@y-s
                mu=float(x@s/n)
                primal=float(np.max(np.abs(rp),initial=0))
                dual=float(np.max(np.abs(rd),initial=0))
                history.append(dict(iteration=iteration,primal=primal,dual=dual,mu=mu))
                if not math.isfinite(mu) or mu>1e30 or np.max(np.abs(x),initial=0)>1e30:
                    failure='NUMERICAL_ERROR'; message='Divergent IPM; feasibility unresolved'; break
                if primal<=1e-8 and dual<=1e-8 and mu<=1e-8:
                    # 标准化残差足够小后，仍须恢复原变量并检查原始可行性与独立有限盒对偶界。
                    original=lo.copy()
                    original[standard['active']]+=standard['width'][standard['active']]*x[:len(standard['active'])]
                    lam=np.zeros(len(model.constraints))
                    for j,origin in enumerate(standard['origins']):
                        if origin>=0:
                            lam[origin]=y[j]*standard['cost_scale']/standard['scales'][j]
                    for j,row in enumerate(model.constraints):
                        if row.sense=='<=': lam[j]=min(0.,lam[j])
                        if row.sense=='>=': lam[j]=max(0.,lam[j])
                    certificate=box_bound(model,lam,lo,hi)
                    if simple_certificate['bound']>certificate['bound']:
                        certificate=simple_certificate
                        lam=zero_multipliers.copy()
                    checked=primal_check(model,original)
                    objective=direction*objective_value(model,original)
                    gap=(objective-certificate['bound'])/max(1.,abs(objective))
                    history[-1].update(original_primal=checked['constraint_violation'],objective=objective,bound=certificate['bound'],gap=gap)
                    last=SparseLPResult('ITERATION_LIMIT',original,objective,certificate['bound'],lam,
                                        iteration,history=history,certificate=certificate)
                    if (max(checked['constraint_violation'],checked['bound_violation'],
                            float(np.max(lo-original,initial=0)),float(np.max(original-hi,initial=0)))<=options.feasibility_tol
                            and 0<=gap<=min(options.objective_tol,1e-9)):
                        last.status='OPTIMAL'; last.message='Original primal and box-dual gap checks passed'
                        return last
                if iteration==options.iteration_limit:
                    break
                try:
                    augmented_statistics={}
                    augmented=_augmented_solver(A,x,s,rp,rd,augmented_statistics,stabilize_augmented)
                    stabilize_augmented=augmented_statistics['stabilized']
                    if 'initial_pivot_ratio' in augmented_statistics:
                        history[-1]['initial_augmented_pivot_ratio']=augmented_statistics['initial_pivot_ratio']
                except RuntimeError:
                    augmented=None  # Redundant equality rows can be rank deficient.
                normal=None
                linear_methods=[]
                linear_ridge=0.
                linear_refinements=0
                linear_factorization_fallbacks=0
                def direction_step(rc):
                    nonlocal normal,linear_ridge,linear_refinements,linear_factorization_fallbacks
                    if augmented is not None:
                        try:
                            step=augmented(rc)
                            linear_methods.append('scaled_augmented_refinement' if augmented_statistics['regularization'] else 'scaled_augmented')
                            linear_ridge=max(linear_ridge,augmented_statistics['regularization'])
                            linear_refinements=max(linear_refinements,augmented_statistics['refinement_steps'])
                            linear_factorization_fallbacks+=int(augmented_statistics['unregularized_fallback'])
                            return step
                        except ArithmeticError:
                            pass
                    if normal is None:
                        normal=_normal_solver(A,x/s)
                    normal_solve,statistics=normal
                    dy=normal_solve(rp-A@((rc-x*rd)/s))
                    ds=rd-A.T@dy
                    dx=(rc-x*ds)/s
                    if np.max(np.abs(A@dx-rp),initial=0)>1e-9*(1+np.max(np.abs(rp),initial=0)):
                        raise ArithmeticError('Refined Newton direction failed original residual check')
                    linear_methods.append('normal_equations_refinement')
                    linear_ridge=max(linear_ridge,statistics['regularization'])
                    return dx,dy,ds
                dx_a,dy_a,ds_a=direction_step(-x*s)
                # 仿射预测步估计互补量，再用中心化及二阶修正项构造校正方向。
                alpha_x=_step(x,dx_a); alpha_s=_step(s,ds_a)
                mu_aff=float((x+alpha_x*dx_a)@(s+alpha_s*ds_a)/n)
                sigma=min(1.,max(0.,(mu_aff/mu)**3))
                dx,dy,ds=direction_step(sigma*mu-x*s-dx_a*ds_a)
                alpha_x=_step(x,dx,.995); alpha_s=_step(s,ds,.995)
                x+=alpha_x*dx; y+=alpha_s*dy; s+=alpha_s*ds
                history[-1].update(step_primal=alpha_x,step_dual=alpha_s,
                                   linear_systems_used=linear_methods,regularization=linear_ridge,
                                   augmented_refinement_steps=linear_refinements,
                                   augmented_unregularized_fallbacks=linear_factorization_fallbacks)
    except (ArithmeticError,FloatingPointError,RuntimeError,ValueError) as exc:
        failure='NUMERICAL_ERROR'; message=str(exc)
    if failure=='NUMERICAL_ERROR':
        # 失败迭代中的对偶量只作为候选分离射线；必须在原始行和当前节点盒上独立复核。
        # 失败本身不是不可行证明，无有效证书时保留 NUMERICAL_ERROR。
        try:
            lam=np.zeros(len(model.constraints))
            for j,origin in enumerate(standard['origins']):
                if origin>=0:
                    lam[origin]=y[j]/standard['scales'][j]
            for j,row in enumerate(model.constraints):
                if row.sense=='<=': lam[j]=min(0.,lam[j])
                if row.sense=='>=': lam[j]=max(0.,lam[j])
            proof=box_infeasibility(model,lam,lo,hi,feasibility_tol=options.feasibility_tol)
            if proof['verified']:
                proof.update(trigger_status=failure,trigger_reason=message)
                return SparseLPResult('INFEASIBLE',iterations=max(0,len(history)-1),
                                      message='Original-box Farkas separation verified',history=history,
                                      infeasibility_certificate=proof)
        except (ArithmeticError,ValueError):
            pass  # 射线算术异常或无法判定时，保留原来的数值失败，禁止错误剪枝。
    if last is not None:
        last.status=failure; last.message=message
        last.iterations=max(0,len(history)-1)
        return last
    return SparseLPResult(failure,iterations=max(0,len(history)-1),message=message,history=history)
