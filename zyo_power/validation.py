# 从输入与输出独立重算供需、损耗、SOC、互斥和成本；物理可行不单独证明最优。
"""Independent original-input physical and accounting audit.

No imports from model builders, dispatch policies, or solver residual code.
"""
import numpy as np


def validate_dispatch(case, result, *, initial_energy=None, strict=False,
                      atol_power=1e-7, atol_energy=1e-7, rtol_energy=1e-9,
                      require_exclusivity=True):
    if min(atol_power, atol_energy, rtol_energy) < 0 or not np.isfinite([atol_power,atol_energy,rtol_energy]).all():
        raise ValueError('Validation tolerances must be finite and nonnegative')
    t_count, s_count = case.periods, len(case.batteries)
    trace = result.get('trace')
    if trace is None:
        return dict(physical_pass=False, reason='missing trace', metrics=None)
    arrays = {}
    for key in ['energy','charge','discharge','thermal','curtailment','shed']:
        shape = (t_count+1,s_count) if key=='energy' else (t_count,s_count) if key in ['charge','discharge'] else (t_count,)
        try:
            array = np.asarray(trace[key],dtype=float)
        except (KeyError,TypeError,ValueError):
            return dict(physical_pass=False, reason=f'missing/invalid {key}', metrics=None)
        if array.shape != shape or not np.isfinite(array).all():
            return dict(physical_pass=False, reason=f'nonfinite/wrong shape {key}', metrics=None)
        arrays[key] = array
    e,ch,dis,th,curt,shed = (arrays[k] for k in ['energy','charge','discharge','thermal','curtailment','shed'])
    maxabs=lambda x:float(np.max(np.abs(x),initial=0.))
    violation=lambda x:float(np.max(x,initial=0.))
    eta_c=np.array([b.eta_charge for b in case.batteries])
    eta_d=np.array([b.eta_discharge for b in case.batteries])
    emin=np.array([b.energy_min for b in case.batteries])
    emax=np.array([b.energy_max for b in case.batteries])
    dt=case.dt[:,None]
    energy_tol=atol_energy+rtol_energy*emax
    residual=e[1:]-case.retention*e[:-1]-eta_c*ch*dt+dis*dt/eta_d
    energy_violation=np.maximum(np.max(emin-e,axis=0),np.max(e-emax,axis=0))
    power_balance=case.renewable-curt+th+dis.sum(axis=1)+shed-case.load-ch.sum(axis=1)
    limits=max(violation(-ch),violation(ch-np.array([b.charge_max for b in case.batteries])),
               violation(-dis),violation(dis-np.array([b.discharge_max for b in case.batteries])),
               violation(-th),violation(th-case.thermal_capacity),violation(-curt),violation(curt-case.renewable),
               violation(-shed),violation(shed-case.load))
    simultaneous=violation(np.minimum(ch,dis))
    source_violation=violation(ch.sum(axis=1)-(case.renewable-curt)) if case.charging_source=='renewable_only' else 0.
    initial_residual=np.zeros(s_count) if initial_energy is None else np.abs(e[0]-np.asarray(initial_energy,dtype=float))
    if initial_residual.shape!=(s_count,) or not np.isfinite(initial_residual).all():
        return dict(physical_pass=False,reason='invalid expected initial energy',metrics=None)
    energy_residual=np.max(np.abs(residual),axis=0)
    cycle_residual=np.abs(e[-1]-e[0])
    standing=(1-case.retention)*e[:-1]
    charging_loss=(1-eta_c)*ch*dt
    discharging_loss=(1/eta_d-1)*dis*dt
    operating_cost=float(np.dot(case.thermal_cost*th,case.dt)+sum(b.throughput_cost*np.dot(ch[:,s]+dis[:,s],case.dt) for s,b in enumerate(case.batteries)))
    ens=float(np.dot(shed,case.dt))
    loss=float((standing+charging_loss+discharging_loss).sum())
    charge=float((ch*dt).sum()); discharge=float((dis*dt).sum())
    net_inventory=float((e[-1]-e[0]).sum())
    metrics=dict(charge_mwh=charge,discharge_mwh=discharge,ens_mwh=ens,
                 curtailment_mwh=float(np.dot(curt,case.dt)),thermal_mwh=float(np.dot(th,case.dt)),
                 standing_loss_mwh=float(standing.sum()),conversion_loss_mwh=float((charging_loss+discharging_loss).sum()),
                 total_storage_loss_mwh=loss,inventory_change_mwh=net_inventory,
                 operating_cost=operating_cost,shortage_cost=ens*case.shed_penalty,
                 cost_with_ens_penalty=operating_cost+ens*case.shed_penalty,
                 global_energy_accounting_residual_mwh=abs(net_inventory-charge+discharge+loss))
    physical=bool(maxabs(power_balance)<=atol_power and limits<=atol_power and source_violation<=atol_power
                  and np.all(energy_residual<=energy_tol) and np.all(energy_violation<=energy_tol)
                  and np.all(initial_residual<=energy_tol) and (not require_exclusivity or simultaneous<=atol_power)
                  and (not strict or violation(shed)<=atol_power))
    return dict(physical_pass=physical, cycle_closed=bool(np.all(cycle_residual<=energy_tol)),
                supply_adequate=bool(violation(shed)<=atol_power), strict_supply_pass=bool(violation(shed)<=atol_power),
                max_power_balance_residual_mw=maxabs(power_balance),max_power_bound_violation_mw=limits,
                max_energy_residual_mwh=maxabs(residual),max_energy_bound_violation_mwh=max(0.,violation(energy_violation)),
                mutual_exclusion_violation_mw=simultaneous,charge_source_violation_mw=source_violation,
                initial_residual_mwh=initial_residual.tolist(),cycle_residual_mwh=cycle_residual.tolist(),
                energy_tolerance_mwh=energy_tol.tolist(),metrics=metrics,
                scope='deterministic single node; ENS, not EENS/LOLE; closure and adequacy are separate')
