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

from .sparse_certificate import primal_check as check_solution, objective_value as _objective
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
                         infeasibility_certificate=lp.infeasibility_certificate))
        if root is None:
            root=dict(status=lp.status,bound=lp.bound,objective=lp.objective,history=lp.history,
                      multipliers=lp.multipliers.tolist() if lp.multipliers is not None else None,
                      certificate=lp.certificate,infeasibility_certificate=lp.infeasibility_certificate)
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
