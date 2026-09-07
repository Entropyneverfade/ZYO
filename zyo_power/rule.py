# 规则仿真：富余充电、短缺放电；先约束可执行功率，禁止事后截断能量掩盖错误。
"""Deterministic surplus-charge/deficit-discharge policy; no optimizer imports."""
import time
import numpy as np


def rule_dispatch(case, initial_energy, *, strict=False):
    start = time.perf_counter()
    t_count, s_count = case.periods, len(case.batteries)
    e = np.zeros((t_count + 1, s_count))
    e[0] = case.initial(initial_energy)
    charge, discharge = np.zeros((t_count,s_count)), np.zeros((t_count,s_count))
    thermal, curtailment, shed = (np.zeros(t_count) for _ in range(3))
    bounds_ok = True
    for t in range(t_count):
        available = case.retention[t] * e[t]
        net = case.renewable[t] - case.load[t]
        for s, battery in enumerate(case.batteries):
            if net > 0:
                room = max(0., battery.energy_max - available[s])
                charge[t,s] = min(net, battery.charge_max, room/(battery.eta_charge*case.dt[t]))
                net -= charge[t,s]
            elif net < 0:
                usable = max(0., available[s] - battery.energy_min)
                discharge[t,s] = min(-net, battery.discharge_max, usable*battery.eta_discharge/case.dt[t])
                net += discharge[t,s]
            # No post-hoc clipping: select feasible power, then apply the energy equation.
            e[t+1,s] = available[s] + battery.eta_charge*charge[t,s]*case.dt[t] - discharge[t,s]*case.dt[t]/battery.eta_discharge
            bounds_ok &= battery.energy_min-1e-10 <= e[t+1,s] <= battery.energy_max+1e-10
        if net < 0:
            thermal[t] = min(-net, case.thermal_capacity[t])
            shed[t] = -net - thermal[t]
        else:
            curtailment[t] = net
    status = 'COMPLETED'
    if not bounds_ok or (strict and np.any(shed > 1e-7)):
        status = 'INFEASIBLE'
    return dict(status=status, mode='rule', engine='rule',
                trace=dict(energy=e,charge=charge,discharge=discharge,thermal=thermal,curtailment=curtailment,shed=shed),
                metadata=dict(runtime_seconds=time.perf_counter()-start, strict=strict,
                              exclusivity='rule_by_net_sign', policy='renewables, then surplus charge / deficit discharge, then thermal',
                              charge_source='renewable_surplus_only', fallback_used=False))
