# 有限前瞻调度：只执行并计费前段，传递已执行末态；历史默认 HiGHS 仅用于比较复现。
"""Hourly finite-lookahead dispatch with exact executed-state continuity."""
from numbers import Integral

import numpy as np

from .optimization import optimize_dispatch, operating_costs


def rolling_dispatch(case, initial_energy, *, lookahead=48, execute=24, engine='highs',
                     options=None, strict=False, exclusivity='milp', terminal_value=0.):
    """Execute one data period using full, periodically wrapped forecast windows.

    Forecast states beyond each executed segment are discarded. There are no
    daily SOC constraints, and no terminal-value credits in the executed cost.
    A failed window stops execution; any preceding partial trace is explicitly
    retained in metadata, never returned as a completed period.
    """
    if (isinstance(lookahead, bool) or isinstance(execute, bool)
            or not isinstance(lookahead, Integral) or not isinstance(execute, Integral)
            or not lookahead >= execute > 0):
        raise ValueError('Require integer lookahead >= execute > 0')
    if not np.all(case.dt == 1):
        raise ValueError('Rolling dispatch supports only hourly dt == 1')
    lookahead, execute = int(lookahead), int(execute)
    current = case.initial(initial_energy)
    pieces = {k: [] for k in ('energy', 'charge', 'discharge', 'thermal', 'curtailment', 'shed')}
    pieces['energy'].append(current[None, :])
    metadata = {'lookahead': lookahead, 'execute': execute, 'windows': [],
                'proof_scope': 'window subproblems only; no full-period optimality certificate',
                'forecast': 'finite lookahead; perfect within window',
                'executed_periods': 0, 'initial_energy': current.tolist(),
                'charging_source': case.charging_source, 'exclusivity': exclusivity,
                'lp_relaxation': exclusivity == 'relaxed', 'strict': bool(strict),
                'terminal_value': float(terminal_value), 'requested_engine': engine,
                'fallback_used': False, 'thermal_cost': 0., 'ens_cost': 0.,
                'shortage_cost': 0., 'throughput_cost': 0., 'operating_cost': 0.,
                'cost_with_ens_penalty': 0.}
    output = {'status': 'OPTIMAL', 'mode': 'rolling', 'engine': engine,
              'trace': None, 'metadata': metadata}
    for offset in range(0, case.periods, execute):
        count = min(execute, case.periods - offset)
        forecast = case.window(offset, lookahead)
        # 前瞻跨周期末尾时按基础时序取模延伸，不缩短窗口；窗口内采用准确未来输入。
        result = optimize_dispatch(forecast, initial_energy=current, cyclic=False,
            engine=engine, options=options, strict=strict, exclusivity=exclusivity,
            terminal_value=terminal_value)
        solver_result = result['metadata']['solver_result']
        window = {'offset': offset, 'forecast_length': lookahead,
                  'forecast_indices': ((offset + np.arange(lookahead)) % case.periods).tolist(),
                  'forecast_absolute_indices': list(range(offset, offset + lookahead)),
                  'planned_execution_indices': list(range(offset, offset + count)),
                  'execution_indices': [], 'executed_length': 0,
                  'initial_energy': current.tolist(), 'executed_final_energy': None,
                  'status': result['status'], 'engine': result['engine'],
                  'operating_cost': 0.,
                  'forecast_operating_cost': result['metadata']['operating_cost'],
                  'forecast_objective': None if solver_result is None else solver_result['objective'],
                  'optimization_metadata': result['metadata']}
        metadata['windows'].append(window)
        output['engine'] = result['engine']
        if result['trace'] is None:
            output['status'] = result['status']
            metadata['failed_window'] = len(metadata['windows']) - 1
            if metadata['executed_periods']:
                metadata['partial_trace'] = {k: np.concatenate(v, axis=0) for k, v in pieces.items()}
            return output
        if result['status'] != 'OPTIMAL' and output['status'] == 'OPTIMAL':
            # Executing a valid incumbent does not upgrade its proof status.
            output['status'] = result['status']
        trace = result['trace']
        executed = {k: v[:count + (k == 'energy')] for k, v in trace.items()}
        # 丢弃未执行预测段，只计费本段；下一窗口初态取已执行末态，不取整窗末态。
        costs = operating_costs(case.window(offset, count), executed)
        window.update(costs)
        window['execution_indices'] = list(range(offset, offset + count))
        window['executed_length'] = count
        current = executed['energy'][-1].copy()
        window['executed_final_energy'] = current.tolist()
        for key in pieces:
            pieces[key].append(executed[key][1:] if key == 'energy' else executed[key])
        for key, value in costs.items():
            metadata[key] += value
        metadata['executed_periods'] += count
    output['trace'] = {k: np.concatenate(v, axis=0) for k, v in pieces.items()}
    metadata['final_energy'] = current.tolist()
    return output
