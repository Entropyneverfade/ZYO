# 通用有限盒分支定界：候选目标与节点下界严格分离，不使用电力模型特例。
"""Generic finite-box MILP search using ZYO sparse LP lower bounds.

Candidates and lower bounds are separate: an LP objective is never a node bound.
No power-system variables, special constraints, or external optimizer calls.
"""
from dataclasses import asdict
import heapq
import math
import time
import numpy as np
from scipy.sparse import coo_matrix

from .sparse_certificate import (primal_check as check_solution, objective_value as _objective,
                                 original_domain_infeasibility)
from ._version import __version__
from .result import Result
from .status import Status
from .sparse_lp import solve_relaxation


def _integer_candidate(model,x,integers,lo,hi,tol):
    # 逐个固定整数变量时利用其所在约束维护允许区间；这只是候选启发式，之后仍须全量复核。
    candidate=x.copy()
    rows=[];cols=[];data=[]
    for j,row in enumerate(model.constraints):
        for i,a in row.expression.terms.items():
            rows.append(j);cols.append(i);data.append(a)
    A=coo_matrix((data,(rows,cols)),shape=(len(model.constraints),len(lo))).tocsc()
    activity=A@candidate+np.array([r.expression.constant for r in model.constraints])
    for i in integers:
        low=lo[i]; high=hi[i]
        for k in range(A.indptr[i],A.indptr[i+1]):
            j=A.indices[k]; a=A.data[k]
            other=activity[j]-a*candidate[i]
            sense=model.constraints[j].sense
            if sense in ('<=','=='):
                value=(tol-other)/a
                if a>0: high=min(high,value)
                else: low=max(low,value)
            if sense in ('>=','=='):
                value=(-tol-other)/a
                if a>0: low=max(low,value)
                else: high=min(high,value)
        low=math.ceil(low); high=math.floor(high)
        if low>high:
            return None
        chosen=min(high,max(low,round(float(candidate[i]))))
        delta=chosen-candidate[i]
        candidate[i]=chosen
        for k in range(A.indptr[i],A.indptr[i+1]):
            activity[A.indices[k]]+=A.data[k]*delta
    return candidate


def solve(model,options):
    """Solve with the sparse kernel, resolving non-finite variable domains first.

    A model with free or one-sided variables is solved through an exact affine
    reformulation onto a finite box. The artificial width used by that
    reformulation is a device of our own making, so every width tried is recorded
    and an ACTIVE artificial bound at the candidate forbids an optimality claim by
    the box bound alone: the width is escalated, and the candidate is accepted only
    when the ORIGINAL-domain weak-duality bound closes the gap. An infinite
    Lagrangian infimum for a chosen multiplier is not a primal unbounded ray;
    without a checked original-domain proof the result stays unresolved.
    """
    from .sparse_domain import needs_domain_transform
    from .sparse_lp import _canonical_model
    # 统一在所有阶段之前把 >= 行改写成等价 <= 行，使对偶乘子符号约定唯一；
    # 变量名、含义、目标和可行域不变，所以返回的解仍可直接对应原模型。
    canonical=_canonical_model(model)
    if not needs_domain_transform(canonical):
        return _solve_box(canonical,options)
    return _solve_domains(canonical,options,original_model=model)


def _solve_domains(model,options,*,original_model=None):
    """Escalating artificial-width driver for models without a finite box."""
    from .sparse_domain import (ARTIFICIAL_LADDER, active_artificial, build_transformed,
                                domain_bound, original_multipliers, recover_primal)
    started=time.perf_counter()
    original_model=model if original_model is None else original_model
    attempts=[]
    best=None
    last=None
    for artificial in ARTIFICIAL_LADDER:
        if time.perf_counter()-started>=options.time_limit:
            break
        transformed,columns=build_transformed(model,artificial)
        inner=_solve_box(transformed,options)
        attempt=dict(artificial=artificial,status=str(inner.status),reason=inner.termination_reason,
                     transformed_variables=len(transformed.variables),nodes=inner.node_count,
                     iterations=inner.iteration_count)
        attempts.append(attempt)
        if inner.status==Status.INFEASIBLE:
            # 内层有限人造盒不可行只提供候选射线；必须在无穷原域按精确系数重新验算。
            root=inner.metadata.get('root_relaxation')
            candidate=root.get('infeasibility_certificate') if hasattr(root,'get') else None
            if (root is not None and root.get('status')=='INFEASIBLE'
                    and hasattr(candidate,'get') and candidate.get('verified')
                    and candidate.get('multipliers') is not None):
                try:
                    # 规范化曾翻转 >= 行；乘子逐行反向回拉后在调用方原行上验算。
                    multipliers=[-value if row.sense=='>=' else value
                                 for row,value in zip(original_model.constraints,
                                                      candidate['multipliers'])]
                    proof=original_domain_infeasibility(
                        original_model,multipliers,feasibility_tol=options.feasibility_tol)
                except (ArithmeticError,TypeError,ValueError,OverflowError) as exc:
                    attempt['original_domain_certificate_reason']=str(exc)
                else:
                    attempt['original_domain_certificate_verified']=proof['verified']
                    attempt['original_domain_certificate_reason']=proof.get('reason')
                    if proof['verified']:
                        # 传入真实内层结果以保留节点、迭代及有限盒诊断，但采纳的是独立原域证明。
                        return _domain_result(
                            model,Status.INFEASIBLE,None,None,None,None,inner,started,
                            'Original-domain Farkas separation verified from transformed root LP',
                            dict(attempts=attempts,certified_width=artificial,
                                 original_domain_infeasibility_certificate=proof))
        if inner.status!=Status.OPTIMAL or not inner.values:
            last=inner
            if inner.status==Status.TIME_LIMIT:
                break
            continue
        original=recover_primal(columns,inner.values)
        active=active_artificial(columns,inner.values,options.feasibility_tol)
        attempt['active_artificial']=active
        check=check_solution(model,original)
        direction=1. if model.sense=='min' else -1.
        objective=_objective(model,original)
        attempt['objective']=objective
        if max(check['constraint_violation'],check['bound_violation'],check['integrality_violation'])>options.feasibility_tol:
            attempt['rejected']='recovered candidate is not feasible in original units'
            last=inner
            continue
        # 原域弱对偶界是采纳候选的唯一依据：只有界有限、间隙闭合才声明最优。
        certificate=_accepted_certificate(inner)
        bound=None; lam=None; candidate=None
        if certificate is not None:
            lam=original_multipliers(certificate)
            # 规范化把 >= 行翻转过；回拉到调用方原行后再做精确弱对偶核验。
            if lam is not None and len(lam)==len(original_model.constraints):
                original_lam=[-float(value) if row.sense=='>=' else float(value)
                              for row,value in zip(original_model.constraints,lam)]
                candidate=domain_bound(original_model,original_lam)
            else:
                candidate=dict(available=False,reason='Original row multiplier count differs')
            attempt['bound_available']=candidate['available']
            attempt['bound_reason']=None if candidate['available'] else candidate['reason']
            if candidate['available']:
                bound=candidate['bound']
        record=dict(artificial=artificial,variables=[column['kind'] for column in columns])
        if bound is None:
            # 无有限原域对偶界不等于原问题无界；继续尝试，仍无证明则保留未决。
            attempt['rejected']='no valid original-domain bound'
            last=inner
            continue
        # 内部统一按最小化方向比较：界与目标都必须先转到该方向，否则最大化会得到负间隙。
        gap=(direction*objective-bound)/max(1.,abs(objective))
        attempt['gap']=gap
        if active and gap>min(options.objective_tol,1e-8):
            # 人造界活跃且间隙未闭合：当前候选可能是被自己造的界裁出来的，放大后重解。
            attempt['rejected']='artificial bound active with an unclosed original-domain gap'
            last=inner
            continue
        if not (0<=gap<=min(options.objective_tol,1e-8)):
            attempt['rejected']='original-domain gap exceeds the declared tolerance'
            last=inner
            continue
        # 证书内部按最小化方向保存弱界；公开结果与证书主字段必须恢复调用方目标方向。
        # 最小化下界向下取整后取负，恰是最大化目标的保守上界。
        original_bound=direction*bound
        attempt['bound_certificate']=dict(
            bound=original_bound,normalized_min_bound=bound,
            raw_bound=direction*candidate['raw_bound'],
            roundoff_allowance=candidate['roundoff_allowance'],
            multipliers_exact=candidate['multipliers_exact'],
            multiplier_source=candidate['multiplier_source'],
            objective_sense=model.sense,scope=candidate['scope'])
        result=_domain_result(model,inner.status,objective,original_bound,gap,original,inner,started,
                              'Original-domain primal, integrality and weak-duality gap checks passed',
                              dict(record,attempts=attempts))
        if best is None or gap<best[0]:
            best=(gap,result,original,inner)
            if gap<=min(options.objective_tol,1e-9):
                return result
    if last is not None and best is None:
        # 内层在变换后的有限盒上"最优"不等于原域最优：没有可回拉的原域证书时
        # 一律降级为未决，绝不把内层状态直接当成原问题的结论。
        return _domain_result(model,Status.UNKNOWN,None,None,None,None,None,started,
                              'Unresolved in the original domain: no valid original-domain bound; '
                              'inner status='+str(last.status)+'; reason='+str(last.termination_reason),
                              dict(attempts=attempts,final_inner_status=str(last.status)))
    if best is not None:
        # 阶梯中已获得闭合的界，但更宽的尝试失败；返回已验证过的最紧候选。
        gap,result,original,selected_inner=best
        # 早期已认证档次被保留时，用它的实际根求解信息重建；尝试表则取全阶梯现场。
        # Result 会深冻结元数据，不能复用早期快照而遗漏后续档次失败。
        return _domain_result(model,result.status,result.objective,result.best_bound,gap,
                              original,selected_inner,started,result.termination_reason,
                              dict(result.metadata['domain_transform'],
                                   attempts=attempts,
                                   selected_nodes=selected_inner.node_count,
                                   selected_iterations=selected_inner.iteration_count,
                                   selected_by='tightest closed original-domain gap'))
    if last is None:
        return _domain_result(model,Status.UNKNOWN,None,None,None,None,None,started,
                              'No artificial width produced a verified candidate',
                              dict(attempts=attempts))
    return _domain_result(model,Status.UNKNOWN,None,None,None,None,None,started,
                          'Unresolved: no artificial width yielded a closed original-domain bound; '
                          'reason='+str(last.termination_reason),
                          dict(attempts=attempts,final_inner_status=str(last.status)))


def _accepted_certificate(inner):
    """Certificate of the accepted root relaxation, if the driver exposed one.

    ``Result.metadata`` is exposed as a read-only mapping, so this must not test
    for ``dict`` specifically.
    """
    metadata=getattr(inner,'metadata',None)
    root=metadata.get('root_relaxation') if hasattr(metadata,'get') else None
    certificate=root.get('certificate') if hasattr(root,'get') else None
    if hasattr(certificate,'get') and certificate.get('scaled_multipliers') is not None:
        return certificate
    return None


def _domain_result(model,status,objective,bound,gap,original,inner,started,message,record):
    """Assemble a Result in ORIGINAL units for the domain-reformulation path."""
    feasibility=None; bound_residual=None; integrality=None
    if original is not None:
        check=check_solution(model,original)
        feasibility=check.get('constraint_violation'); bound_residual=check.get('bound_violation')
        integrality=check.get('integrality_violation')
    metadata=dict(inner.metadata) if inner is not None and hasattr(inner,'metadata') else {}
    metadata.update(domain_transform=record,fallback_requested=False,fallback_used=False,
                    scope='experimental sparse LP/MILP; non-finite domains resolved by exact affine reformulation')
    # 多档人造盒是连续的真实求解调用；公开总计数需累计全部尝试，选中档另记在元数据。
    attempts=record.get('attempts') if hasattr(record,'get') else None
    nodes=(sum(int(attempt['nodes']) for attempt in attempts) if attempts is not None
           else inner.node_count if inner is not None else 0)
    iterations=(sum(int(attempt['iterations']) for attempt in attempts) if attempts is not None
                else inner.iteration_count if inner is not None else 0)
    return Result(status=status,solver_name='native_sparse',solver_version=__version__,
                  objective=objective,best_bound=bound,mip_gap=gap,
                  values={v.name:float(original[i]) for i,v in enumerate(model.variables)} if original is not None else {},
                  runtime=time.perf_counter()-started,
                  node_count=nodes,iteration_count=iterations,
                  primal_residual=feasibility,bound_residual=bound_residual,integrality_residual=integrality,
                  termination_reason=message,metadata=metadata)


def _solve_box(model,options):
    # direction 将最大化翻成最小化；所有搜索界先按最小化存储，最终再恢复用户目标方向。
    started=time.perf_counter(); deadline=started+options.time_limit
    lo=np.array([v.lb for v in model.variables]); hi=np.array([v.ub for v in model.variables])
    integers=np.array([i for i,v in enumerate(model.variables) if v.kind!='C'],dtype=int)
    direction=1. if model.sense=='min' else -1.
    incumbent=math.inf; best=None; closed_bound=math.inf
    nodes=0; iterations=0; logs=[]; root=None
    queue=[]; serial=1
    # 队列中的节点即未关闭的证明义务，不能因 LP 数值失败而丢弃。
    reason='UNKNOWN'; message='No sparse solve performed'
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        message='native_sparse requires finite lower and upper bounds; no implicit fallback'
    elif options.time_limit==0:
        reason='TIME_LIMIT'; message='Zero time budget'
    else:
        lo[integers]=np.ceil(lo[integers]); hi[integers]=np.floor(hi[integers])
        if np.any(lo>hi):
            reason='INFEASIBLE'; message='No feasible value inside variable bounds'
        else:
            queue=[(-math.inf,0,lo,hi)]
            reason='OPTIMAL'; message='Search closed by checked box-dual bounds and integer candidates'
    def feasible(candidate,left,right):
        check=check_solution(model,candidate)
        return (max(check['constraint_violation'],check['bound_violation'],
                    float(np.max(left-candidate,initial=0)),float(np.max(candidate-right,initial=0)))<=options.feasibility_tol
                and check['integrality_violation']<=options.integrality_tol)
    def gap_value(bound):
        return max(0.,incumbent-bound)/max(1.,abs(incumbent)) if math.isfinite(incumbent) and math.isfinite(bound) else None
    while queue:
        if time.perf_counter()>=deadline:
            reason='TIME_LIMIT'; message='Sparse search time limit'; break
        if nodes>=options.node_limit:
            reason='NODE_LIMIT'; message='Sparse search node limit'; break
        inherited,number,left,right=heapq.heappop(queue)
        numerical=options.objective_tol*max(1.,abs(incumbent)) if best is not None else 0.
        if best is not None and inherited>=incumbent-numerical:
            closed_bound=min(closed_bound,inherited)
            continue
        lp=solve_relaxation(model,left,right,options,deadline)
        nodes+=1; iterations+=lp.iterations
        bound=max(inherited,lp.bound) if lp.bound is not None else inherited
        logs.append(dict(node=number,status=lp.status,bound=bound if math.isfinite(bound) else None,
                         objective=lp.objective,iterations=lp.iterations,message=lp.message,
                         infeasibility_certificate=lp.infeasibility_certificate,
                         feasibility_recovery=lp.feasibility_recovery))
        if root is None:
            root=dict(status=lp.status,bound=lp.bound,objective=lp.objective,history=lp.history,
                      multipliers=lp.multipliers.tolist() if lp.multipliers is not None else None,
                      certificate=lp.certificate,infeasibility_certificate=lp.infeasibility_certificate,
                      feasibility_recovery=lp.feasibility_recovery)
        if lp.status=='INFEASIBLE':
            continue
        if lp.x is not None:
            candidate=_integer_candidate(model,lp.x,integers,left,right,options.feasibility_tol)
            if candidate is not None and feasible(candidate,left,right):
                value=direction*_objective(model,candidate)
                if value<incumbent:
                    incumbent=value; best=candidate
        if lp.status!='OPTIMAL':
            # 未证节点放回队列；即使有可行 incumbent，也不能将本次终止报告为最优。
            heapq.heappush(queue,(bound,number,left,right))
            reason=lp.status; message=lp.message; break
        numerical=options.objective_tol*max(1.,abs(incumbent)) if best is not None else 0.
        if best is not None and bound>=incumbent-numerical:
            closed_bound=min(closed_bound,bound)
        else:
            fractions=np.abs(lp.x[integers]-np.rint(lp.x[integers]))
            if not len(integers) or float(np.max(fractions,initial=0))<=options.integrality_tol:
                heapq.heappush(queue,(bound,number,left,right))
                reason='NUMERICAL_ERROR'; message='Candidate and dual bound did not close within tolerance'; break
            i=int(integers[np.argmax(fractions)])
            split=math.floor(float(lp.x[i]))
            left_hi=right.copy(); left_hi[i]=min(right[i],split)
            right_lo=left.copy(); right_lo[i]=max(left[i],split+1)
            if left[i]<=left_hi[i]:
                heapq.heappush(queue,(bound,serial,left.copy(),left_hi)); serial+=1
            if right_lo[i]<=right[i]:
                heapq.heappush(queue,(bound,serial,right_lo,right.copy())); serial+=1
        global_bound=min(incumbent,closed_bound,queue[0][0] if queue else math.inf)
        # 全局界保留被容差关闭的节点界和未决队列界，不用可行解目标替换它们制造零 Gap。
        gap=gap_value(global_bound)
        if queue and options.mip_gap>0 and gap is not None and gap<=options.mip_gap:
            reason='GAP_LIMIT'; message='Requested gap reached; remaining nodes retained'; break
    if not queue and best is None and reason=='OPTIMAL':
        reason='INFEASIBLE'; message='All nodes rejected by checked box/row contradictions or Farkas separation'
    bound=min(incumbent,closed_bound,queue[0][0] if queue else math.inf)
    check=check_solution(model,best) if best is not None else {}
    unmapped=['pivot_tol','max_tableau_cells','threads','seed','presolve']
    metadata=dict(backend_kind='self-developed',algorithm='sparse_primal_dual_predictor_corrector+box_bound_branch_and_bound',
                  dependency_boundary='NumPy and SciPy sparse/SuperLU linear algebra only; no optimization engine',
                  scope='experimental finite-box LP/MILP; sufficient Farkas witnesses, no general detection or unboundedness certificate',
                  fallback_requested=False,fallback_used=False,
                  parameters_requested=asdict(options),
                  parameters_applied={k:v for k,v in asdict(options).items() if k not in unmapped},
                  parameters_unmapped=unmapped,numerical_gap_definition='(incumbent-bound)/max(1,abs(incumbent)) <= objective_tol',
                  validation_parameters=dict(feasibility_tol=options.feasibility_tol,integrality_tol=options.integrality_tol,objective_tol=options.objective_tol),
                  root_relaxation=root,node_log=logs,remaining_nodes=len(queue))
    return Result(status=Status(reason),solver_name='native_sparse',solver_version=__version__,
                  objective=direction*incumbent if best is not None else None,
                  best_bound=direction*bound if math.isfinite(bound) else None,mip_gap=gap_value(bound),
                  values={v.name:float(best[i]) for i,v in enumerate(model.variables)} if best is not None else {},
                  runtime=time.perf_counter()-started,node_count=nodes,iteration_count=iterations,
                  primal_residual=check.get('constraint_violation'),bound_residual=check.get('bound_violation'),
                  integrality_residual=check.get('integrality_violation'),termination_reason=message,metadata=metadata)
