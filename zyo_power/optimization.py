# 构造固定容量线性或互斥 MILP 调度模型；供需平衡、能量递推与周期条件明确区分。
"""Fixed-capacity dispatch using explicitly selected ZYO solver backends.

Power is MW, energy MWh and interval duration hours. Terminal value is an
optimization incentive only; it is never an operating-cost credit.
"""
from dataclasses import asdict
import hashlib
import json
import math

import numpy as np
from zyo import Model, SolveOptions, quicksum


def build_dispatch_model(case, *, initial_energy=None, cyclic=True, strict=False,
                         exclusivity='milp', terminal_value=0.):
    """Return a model and grouped object arrays of named ZYO variables.

    ``mode[t,s]`` is one for charging in MILP, continuous in the explicitly
    requested LP relaxation. An unspecified initial state is allowed only for
    a cyclic model, where the initial and final bounded states are equal.
    """
    if exclusivity not in ('milp', 'relaxed'):
        raise ValueError("exclusivity must be 'milp' or 'relaxed'")
    terminal_value = float(terminal_value)
    if not math.isfinite(terminal_value):
        raise ValueError('terminal_value must be finite')
    if cyclic and terminal_value != 0:
        raise ValueError('terminal_value cannot be used in a cyclic model')
    if initial_energy is None and not cyclic:
        raise ValueError('A noncyclic model requires initial_energy')
    initial = None if initial_energy is None else case.initial(initial_energy)
    T, S = case.periods, len(case.batteries)
    # T 个运行区间对应 T+1 个能量时点；未指定初态时仅循环模型允许把它作为有界决策变量。
    model = Model('storage:' + case.name)
    variables = {key: np.empty((T + (key == 'energy'), S), dtype=object)
                 for key in ('energy', 'charge', 'discharge', 'mode')}
    variables.update({key: np.empty(T, dtype=object)
                      for key in ('thermal', 'curtailment', 'shed')})
    for s, battery in enumerate(case.batteries):
        for t in range(T + 1):
            lb, ub = battery.energy_min, battery.energy_max
            if t == 0 and initial is not None:
                lb = ub = float(initial[s])
            variables['energy'][t, s] = model.add_var(f'energy[{t},{s}]', lb=lb, ub=ub)
        for t in range(T):
            for key, bound in [('charge', battery.charge_max), ('discharge', battery.discharge_max)]:
                variables[key][t, s] = model.add_var(f'{key}[{t},{s}]', ub=bound)
            variables['mode'][t, s] = model.add_var(f'mode[{t},{s}]', ub=1,
                                        vtype='B' if exclusivity == 'milp' else 'C')
    for t in range(T):
        for key, bound in [('thermal', case.thermal_capacity[t]),
                           ('curtailment', case.renewable[t]),
                           ('shed', 0 if strict else case.load[t])]:
            variables[key][t] = model.add_var(f'{key}[{t}]', ub=float(bound))
    energy, charge, discharge, mode = (variables[k] for k in ('energy', 'charge', 'discharge', 'mode'))
    for t in range(T):
        charging = quicksum(charge[t])
        # 充电作为需求，放电作为供给；shed 单独表示缺供，不能用周期闭合掩盖它。
        supply = (float(case.renewable[t]) - variables['curtailment'][t]
                  + variables['thermal'][t] + quicksum(discharge[t]) + variables['shed'][t])
        model.add_constr(supply == float(case.load[t]) + charging, name=f'balance[{t}]')
        if case.charging_source == 'renewable_only':
            model.add_constr(charging <= float(case.renewable[t]) - variables['curtailment'][t],
                             name=f'renewable_charging[{t}]')
        for s, battery in enumerate(case.batteries):
            # 能量递推：保留率*期初能量 + 充电效率*功率*小时 - 放电功率*小时/放电效率。
            model.add_constr(energy[t + 1, s] == float(case.retention[t, s]) * energy[t, s]
                + battery.eta_charge * float(case.dt[t]) * charge[t, s]
                - float(case.dt[t]) / battery.eta_discharge * discharge[t, s], name=f'energy_balance[{t},{s}]')
            model.add_constr(charge[t, s] <= battery.charge_max * mode[t, s], name=f'charge_mode[{t},{s}]')
            model.add_constr(discharge[t, s] <= battery.discharge_max * (1 - mode[t, s]),
                             name=f'discharge_mode[{t},{s}]')
    if cyclic:
        # 仅完整周期闭合，不给滚动模型增加每日 SOC 相等；能量相等不要求充放电量相等。
        for s in range(S):
            model.add_constr(energy[T, s] == energy[0, s], name=f'cyclic[{s}]')
    model.minimize(quicksum(float(case.dt[t]) * (
        float(case.thermal_cost[t]) * variables['thermal'][t]
        + case.shed_penalty * variables['shed'][t]
        + quicksum(b.throughput_cost * (charge[t, s] + discharge[t, s])
                   for s, b in enumerate(case.batteries))) for t in range(T))
        - terminal_value * quicksum(energy[T]))
    return model, variables


def operating_costs(case, trace):
    # 只重算实际执行的运行费用；末端能量价值是优化激励，不是实际收入。
    """Compute actual executed operating costs, with no terminal-state credit."""
    thermal = float(np.sum(case.dt * case.thermal_cost * trace['thermal']))
    ens = float(np.sum(case.dt * case.shed_penalty * trace['shed']))
    throughput = float(sum(b.throughput_cost * np.sum(case.dt *
        (trace['charge'][:, s] + trace['discharge'][:, s])) for s, b in enumerate(case.batteries)))
    return {'thermal_cost': thermal, 'ens_cost': ens, 'shortage_cost': ens,
            'throughput_cost': throughput, 'operating_cost': thermal + throughput,
            'cost_with_ens_penalty': thermal + ens + throughput}


def optimize_dispatch(case, *, engine='native', options=None, initial_energy=None,
                      cyclic=True, strict=False, exclusivity='milp', terminal_value=0.):
    """Solve without backend fallback or an implicit integrality relaxation.

    Failed/invalid solver candidates have no dispatch trace. A valid incumbent
    at a solver limit retains its limit status and is available for inspection.
    """
    if options is None:
        options = SolveOptions()
    elif isinstance(options, dict):
        options = SolveOptions.from_kwargs(**options)
    if not isinstance(options, SolveOptions):
        raise TypeError('options must be SolveOptions or a parameter dictionary')
    model, variables = build_dispatch_model(case, initial_energy=initial_energy,
        cyclic=cyclic, strict=strict, exclusivity=exclusivity, terminal_value=terminal_value)
    specification = model.to_dict()
    digest = hashlib.sha256(json.dumps(specification, sort_keys=True, separators=(',', ':'),
                                      allow_nan=False).encode('utf-8')).hexdigest()
    metadata = {
        'model': specification if case.periods <= 168 else None,
        'model_retention': 'full' if case.periods <= 168 else 'hash_and_scale_only',
        'model_hash': digest,
        'model_scale': {'variables': len(model.variables), 'constraints': len(model.constraints),
                        'binary_variables': sum(v.kind == 'B' for v in model.variables),
                        'nonzeros': sum(len(r.expression.terms) for r in model.constraints),
                        'periods': case.periods, 'batteries': len(case.batteries)},
        'options': asdict(options), 'requested_engine': engine, 'fallback_used': False,
        'cyclic': bool(cyclic), 'strict': bool(strict), 'exclusivity': exclusivity,
        'lp_relaxation': exclusivity == 'relaxed', 'charging_source': case.charging_source,
        'terminal_value': float(terminal_value), 'solver_result': None,
        'initial_energy': None if initial_energy is None else case.initial(initial_energy).tolist(),
        'battery_names': [b.name for b in case.batteries], 'operating_cost': None,
        'cost_with_ens_penalty': None,
        'terminal_objective_term': None,
    }
    output = {'status': 'SOLVER_ERROR', 'mode': 'optimization', 'engine': engine,
              'trace': None, 'metadata': metadata}
    try:
        result = model.solve(solver=engine, options=options)
    except Exception as exc:
        # Solver availability, licensing and native errors remain inspectable.
        # Model/input validation above deliberately propagates to the caller.
        metadata['solver_exception'] = {'type': type(exc).__name__, 'message': str(exc)}
        return output
    metadata['solver_result'] = result.to_dict()
    output.update(status=result.status.value, engine=result.solver_name)
    if result.has_solution and result.status.value not in (
            'NUMERICAL_ERROR', 'SOLVER_ERROR', 'INFEASIBLE', 'UNBOUNDED', 'INF_OR_UNBD', 'UNKNOWN'):
        trace = {key: np.array([result.values[v.name] for v in array.flat], dtype=float).reshape(array.shape)
                 for key, array in variables.items() if key != 'mode'}
        output['trace'] = trace
        metadata.update(operating_costs(case, trace))
        metadata['terminal_objective_term'] = -float(terminal_value) * float(np.sum(trace['energy'][-1]))
    return output
