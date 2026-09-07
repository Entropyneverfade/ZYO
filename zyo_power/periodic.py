# 周期稳态迭代：连续传递上一轮末态，并额外确认重复；首末闭合不代表没有缺电。
"""Empirical periodic-state iteration, independent of the dispatch algorithm."""
import time
import traceback
import numpy as np
from .validation import validate_dispatch


def _repeat_error(first, second, case):
    a,b=first['trace'],second['trace']
    energy=np.max(np.abs(np.asarray(a['energy'])-np.asarray(b['energy'])),axis=0)
    power=max(float(np.max(np.abs(np.asarray(a[k])-np.asarray(b[k])),initial=0.))
              for k in ['charge','discharge','thermal','curtailment','shed'])
    return energy,power


def iterate_periodic(case, policy, initial_energy, *, max_cycles=30, atol_energy=1e-7,
                     rtol_energy=1e-9, atol_power=1e-7, metric_atol=1e-6,
                     metric_rtol=1e-8, cycle_memory=8, strict=False):
    """Propagate energy unchanged; a closed candidate requires one more full run.

    Only storage energy is state here. Strategies with other memory (commitment,
    ramps, controllers) require an extended state contract before being used.
    """
    if isinstance(max_cycles,bool) or not isinstance(max_cycles,int) or max_cycles<1:
        raise ValueError('max_cycles must be positive integer')
    if not isinstance(cycle_memory,int) or cycle_memory<2:
        raise ValueError('cycle_memory must be at least two')
    tolerances=np.array([atol_energy,rtol_energy,atol_power,metric_atol,metric_rtol])
    if not np.isfinite(tolerances).all() or np.any(tolerances<0):
        raise ValueError('Tolerances must be finite and nonnegative')
    start=time.perf_counter()
    energy=case.initial(initial_energy,atol_energy=atol_energy,rtol_energy=rtol_energy)
    initial=energy.copy()
    tolerance=atol_energy+rtol_energy*np.array([b.energy_max for b in case.batteries])
    states=[energy.copy()]
    history=[]
    candidate=None
    output=dict(status='NOT_CONVERGED',formal=None,formal_audit=None,initial_energy=initial.tolist(),
                history=history,energy_tolerance_mwh=tolerance.tolist(),
                state_scope='storage energy only; capacities/efficiencies frozen; no ageing during virtual cycles',
                confirmation_required=True)
    for cycle in range(1,max_cycles+1):
        try:
            result=policy(energy.copy())
        except Exception as exc:
            output.update(status='SUBPROBLEM_ERROR',error_type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
            break
        status=result.get('status','UNKNOWN')
        if status not in ['COMPLETED','OPTIMAL']:
            output['status']='SUBPROBLEM_INFEASIBLE' if status=='INFEASIBLE' else 'NUMERICAL_FAILURE' if 'NUMER' in status else 'SUBPROBLEM_LIMIT' if 'LIMIT' in status else 'SUBPROBLEM_ERROR'
            output['failed_dispatch']=result
            break
        audit=validate_dispatch(case,result,initial_energy=energy,strict=strict,
                                atol_power=atol_power,atol_energy=atol_energy,rtol_energy=rtol_energy)
        if not audit['physical_pass']:
            output.update(status='NUMERICAL_FAILURE',failed_dispatch=result,failed_audit=audit)
            break
        trace=result['trace']
        end=np.asarray(trace['energy'][-1],dtype=float).copy()
        residual=np.abs(end-energy)
        entry=dict(cycle=cycle,start_energy_mwh=energy.tolist(),end_energy_mwh=end.tolist(),
                   residual_mwh=residual.tolist(),role='warmup',audit=audit,trace=trace,
                   dispatch_status=status,engine=result.get('engine'),metadata=result.get('metadata',{}))
        history.append(entry)
        closed=bool(np.all(residual<=tolerance))
        if candidate is not None and closed:
            # 初次首末闭合只是候选，还需下一完整轮的轨迹与指标均重复才能记录正式周期。
            e_error,p_error=_repeat_error(candidate[0],result,case)
            metric_errors={key:abs(value-candidate[1]['metrics'][key]) for key,value in audit['metrics'].items()}
            metrics_repeat=all(diff<=metric_atol+metric_rtol*max(1.,abs(audit['metrics'][key]),abs(candidate[1]['metrics'][key]))
                               for key,diff in metric_errors.items())
            entry['repeat_energy_error_mwh']=e_error.tolist()
            entry['repeat_power_error_mw']=p_error
            entry['repeat_metric_errors']=metric_errors
            if np.all(e_error<=tolerance) and p_error<=atol_power and metrics_repeat:
                entry['role']='formal_confirmation'
                output.update(status='CONVERGED',formal=result,formal_audit=audit,
                              confirmed_cycle=cycle,preheat_cycles=cycle-1)
                break
        candidate=(result,audit) if closed else None
        if not closed:
            for index,previous in enumerate(states[:-1]):
                if np.all(np.abs(end-previous)<=tolerance):
                    output.update(status='SUSPECTED_MULTI_PERIOD_CYCLE',suspected_periods=len(states)-index,
                                  message='Repeated boundary state observed, not a proof of a unique long-period orbit')
                    break
            if output['status']=='SUSPECTED_MULTI_PERIOD_CYCLE':
                break
        energy=end
        # 原值传递上一轮末态，预热不重置电量，也不因虚拟轮次累计容量老化。
        states.append(end.copy())
        states=states[-cycle_memory:]
    output['cycles_executed']=len(history)
    output['runtime_seconds']=time.perf_counter()-start
    return output


def initialization_study(case, policy, **kwargs):
    low=np.array([b.energy_min for b in case.batteries])
    high=np.array([b.energy_max for b in case.batteries])
    runs=[iterate_periodic(case,policy,initial,**kwargs) for initial in [low,(low+high)/2,high]]
    status='INITIALIZATION_DEPENDENCE_UNRESOLVED'
    if all(run['status']=='CONVERGED' for run in runs):
        status='SAME_OBSERVED_CYCLE'
        for run in runs[1:]:
            e_error,p_error=_repeat_error(runs[0]['formal'],run['formal'],case)
            if np.any(e_error>np.array(run['energy_tolerance_mwh'])) or p_error>kwargs.get('atol_power',1e-7):
                status='MULTIPLE_OBSERVED_CYCLES'
    return dict(initialization_status=status,runs=runs,
                interpretation='Three initializations only; no general convergence or uniqueness proof')
