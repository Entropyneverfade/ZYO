# 单节点小时级机组组合：原始输入、通用建模和独立物理检查；不导入外部优化引擎。
"""Fixed-capacity hourly UC; finite horizon with explicit residual obligations."""
import math
from numbers import Real


def validate_uc_input(case):
    """验证 MW、MWh、小时和货币单位一致的输入；拒绝悄悄忽略的字段。"""
    required = {'name', 'units', 'demand', 'wind', 'solar', 'reserve', 'strict', 'ens_cost'}
    if not isinstance(case, dict) or not required <= case.keys() or set(case)-required-{'source'}:
        raise ValueError('UC 输入字段不完整或含未知字段')
    if not isinstance(case['name'], str) or not case['name'] or type(case['strict']) is not bool:
        raise ValueError('UC 名称和 strict 类型错误')

    def nonnegative(value):
        return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value) and value >= 0

    periods = len(case['demand'])
    if periods < 1 or not nonnegative(case['ens_cost']):
        raise ValueError('至少一个时段；ENS 惩罚须非负有限')
    for key in ('demand', 'wind', 'solar', 'reserve'):
        if len(case[key]) != periods or not all(nonnegative(x) for x in case[key]):
            raise ValueError('时序长度或数值错误：' + key)
    if not case['units']:
        raise ValueError('至少一台常规机组')
    names = set()
    numeric = ('p_min', 'p_max', 'marginal_cost', 'no_load_cost', 'startup_cost',
               'shutdown_cost', 'ramp_up', 'ramp_down', 'startup_ramp', 'shutdown_ramp', 'initial_power')
    for unit in case['units']:
        expected = set(numeric) | {'name', 'min_up', 'min_down', 'initial_on', 'initial_duration'}
        if set(unit) != expected or not isinstance(unit['name'], str) or not unit['name'] or unit['name'] in names:
            raise ValueError('机组字段或唯一名称错误')
        names.add(unit['name'])
        if not all(nonnegative(unit[k]) for k in numeric) or not 0 <= unit['p_min'] <= unit['p_max'] or unit['p_max'] == 0:
            raise ValueError('机组容量、费用或爬坡数值错误')
        for key in ('min_up', 'min_down', 'initial_duration'):
            if type(unit[key]) is not int or unit[key] < (0 if key == 'initial_duration' else 1):
                raise ValueError('小时数必须为整数：' + key)
        if type(unit['initial_on']) is not int or unit['initial_on'] not in (0, 1):
            raise ValueError('initial_on 必须为 0 或 1')
        low, high = unit['p_min']*unit['initial_on'], unit['p_max']*unit['initial_on']
        if not low <= unit['initial_power'] <= high:
            raise ValueError('初始出力不满足初始开停机状态')
    return periods


def build_uc_model(case):
    """构造一次相同数学模型；求解调用由原生入口或隔离比较进程负责。"""
    from zyo import Model, quicksum
    periods = validate_uc_input(case)
    model = Model(case['name'])
    fields = {key: [] for key in ('power', 'on', 'start', 'stop')}
    cost = 0
    for g, unit in enumerate(case['units']):
        power = [model.add_var(f'p_{g}_{t}', ub=unit['p_max']) for t in range(periods)]
        on = [model.add_var(f'u_{g}_{t}', vtype='B') for t in range(periods)]
        # 启停量可以连续：由二元状态、转换等式和两个上界严格确定为 0/1。
        start = [model.add_var(f'su_{g}_{t}', ub=1) for t in range(periods)]
        stop = [model.add_var(f'sd_{g}_{t}', ub=1) for t in range(periods)]
        for key, array in zip(fields, (power, on, start, stop)):
            fields[key].append(array)
        remain = max(0, unit['min_up' if unit['initial_on'] else 'min_down'] - unit['initial_duration'])
        for t in range(periods):
            prev_on = on[t-1] if t else unit['initial_on']
            prev_p = power[t-1] if t else unit['initial_power']
            model.add_constr(power[t] >= unit['p_min']*on[t], name=f'pmin_{g}_{t}')
            model.add_constr(power[t] <= unit['p_max']*on[t], name=f'pmax_{g}_{t}')
            model.add_constr(on[t]-prev_on == start[t]-stop[t], name=f'transition_{g}_{t}')
            model.add_constr(start[t] <= on[t], name=f'start_on_{g}_{t}')
            model.add_constr(start[t] <= 1-prev_on, name=f'start_off_{g}_{t}')
            model.add_constr(power[t]-prev_p <= unit['ramp_up']*prev_on + unit['startup_ramp']*start[t], name=f'ru_{g}_{t}')
            model.add_constr(prev_p-power[t] <= unit['ramp_down']*on[t] + unit['shutdown_ramp']*stop[t], name=f'rd_{g}_{t}')
            # 后向启停计数：最近 L 小时启动则当前必须在线，停机的对偶约束同理。
            model.add_constr(quicksum(start[k] for k in range(max(0,t-unit['min_up']+1),t+1)) <= on[t], name=f'min_up_{g}_{t}')
            model.add_constr(quicksum(stop[k] for k in range(max(0,t-unit['min_down']+1),t+1)) <= 1-on[t], name=f'min_down_{g}_{t}')
            if t < remain:
                model.add_constr(on[t] == unit['initial_on'], name=f'initial_{g}_{t}')
            cost += (unit['marginal_cost']*power[t] + unit['no_load_cost']*on[t]
                     + unit['startup_cost']*start[t] + unit['shutdown_cost']*stop[t])
    for key in ('wind', 'solar', 'ens'):
        upper = case[key] if key != 'ens' else ([0]*periods if case['strict'] else case['demand'])
        fields[key] = [model.add_var(f'{key}_{t}', ub=upper[t]) for t in range(periods)]
    for t in range(periods):
        thermal = quicksum(p[t] for p in fields['power'])
        model.add_constr(thermal+fields['wind'][t]+fields['solar'][t]+fields['ens'][t] == case['demand'][t], name=f'balance_{t}')
        # 备用定义为在线容量裕度；没有宣称此裕度具备更短时间尺度的爬坡可交付性。
        headroom = quicksum(unit['p_max']*fields['on'][g][t]-fields['power'][g][t] for g,unit in enumerate(case['units']))
        model.add_constr(headroom >= case['reserve'][t], name=f'reserve_{t}')
        cost += case['ens_cost']*fields['ens'][t]
    model.minimize(cost)
    return model, fields


def extract_uc_trace(mapping, values):
    """只按变量名称恢复逐时输出；不在此处修正、取整或截断候选。"""
    return {key: ([[float(values[v.name]) for v in row] for row in array]
                  if key in ('power','on','start','stop') else [float(values[v.name]) for v in array])
            for key,array in mapping.items()}


def validate_uc(case, trace, *, feasibility_tol=1e-7, integrality_tol=1e-7):
    """独立按物理时序扫描复算，不调用建模器、不使用它的约束对象或成功标志。"""
    periods = validate_uc_input(case)
    for value in (feasibility_tol, integrality_tol):
        if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError('检查容差必须为正有限数')
    if integrality_tol >= .5:
        raise ValueError('整数容差必须小于 0.5')
    expected = {'power','on','start','stop','wind','solar','ens'}
    if set(trace) != expected:
        raise ValueError('输出字段不匹配')
    for key, array in trace.items():
        rows = array if key in ('power','on','start','stop') else [array]
        if key in ('power','on','start','stop') and len(rows) != len(case['units']):
            raise ValueError('机组数量不匹配')
        if any(len(row) != periods or any(not isinstance(x,Real) or isinstance(x,bool) or not math.isfinite(x) for x in row) for row in rows):
            raise ValueError('输出维度或数值非法')
    residual = dict(balance=0., bound=0., integrality=0., transition=0., ramp=0.,
                    min_up_down=0., initial_state=0., reserve=0.)
    components = dict(fuel=0., no_load=0., startup=0., shutdown=0., ens_penalty=0.)
    terminal = []
    for g, unit in enumerate(case['units']):
        prior = unit['initial_on']
        previous_power = unit['initial_power']
        duration = unit['initial_duration']
        remain = max(0, unit['min_up' if prior else 'min_down']-duration)
        for t in range(periods):
            p, on, start, stop = (trace[k][g][t] for k in ('power','on','start','stop'))
            for x in (on,start,stop):
                residual['integrality'] = max(residual['integrality'], abs(x-round(x)))
                residual['bound'] = max(residual['bound'], -x, x-1)
            state = int(round(on))
            # 原模型还具有独立的出力盒界；近整数状态的误差不能被容量放大后掩盖越界。
            residual['bound'] = max(residual['bound'], -p, p-unit['p_max'],
                                    unit['p_min']*on-p, p-unit['p_max']*on)
            # 扫描实际开停机片段，独立于建模器的后向线性不等式。
            if state != prior:
                needed = unit['min_up' if prior else 'min_down']
                residual['min_up_down'] = max(residual['min_up_down'], needed-duration)
                duration = 0
            if t < remain:
                residual['initial_state'] = max(residual['initial_state'], abs(on-unit['initial_on']))
            wanted_start, wanted_stop = max(0,state-prior), max(0,prior-state)
            residual['transition'] = max(residual['transition'], abs(start-wanted_start), abs(stop-wanted_stop))
            up = unit['startup_ramp'] if wanted_start else unit['ramp_up']*prior
            down = unit['shutdown_ramp'] if wanted_stop else unit['ramp_down']*state
            residual['ramp'] = max(residual['ramp'], p-previous_power-up, previous_power-p-down)
            for name, value, rate in [('fuel',p,'marginal_cost'), ('no_load',on,'no_load_cost'),
                                      ('startup',start,'startup_cost'), ('shutdown',stop,'shutdown_cost')]:
                components[name] += value*unit[rate]
            prior, previous_power, duration = state, p, duration+1
        terminal.append(dict(name=unit['name'], on=prior, power=previous_power, duration=duration,
                             remaining_obligation_hours=max(0,unit['min_up' if prior else 'min_down']-duration)))
    for t in range(periods):
        supplied = sum(row[t] for row in trace['power'])+trace['wind'][t]+trace['solar'][t]+trace['ens'][t]
        residual['balance'] = max(residual['balance'], abs(supplied-case['demand'][t]))
        for key in ('wind','solar','ens'):
            upper = case[key][t] if key != 'ens' else (0 if case['strict'] else case['demand'][t])
            residual['bound'] = max(residual['bound'], -trace[key][t], trace[key][t]-upper)
        available = sum(unit['p_max']*trace['on'][g][t]-trace['power'][g][t] for g,unit in enumerate(case['units']))
        residual['reserve'] = max(residual['reserve'], case['reserve'][t]-available)
        components['ens_penalty'] += trace['ens'][t]*case['ens_cost']
    cost = sum(components.values())
    # 有限输入的乘积或求和仍可能溢出；拒绝非有限派生成本，避免返回可验收标志。
    if not all(math.isfinite(value) for value in [*components.values(),cost]):
        raise ValueError('派生成本非有限：cost overflow')
    physical = max(v for k,v in residual.items() if k != 'integrality') <= feasibility_tol and residual['integrality'] <= integrality_tol
    return dict(physical_pass=physical, supply_satisfied=physical and max(trace['ens']) <= feasibility_tol,
                residuals=residual, max_physical_residual=max(residual.values()),
                cost=cost, cost_components=components, ens_mwh=sum(trace['ens']),
                curtailment_mwh=sum(case[k][t]-trace[k][t] for k in ('wind','solar') for t in range(periods)),
                terminal_state=terminal, terminal_policy='finite horizon; carry residual obligations to continuation',
                reserve_definition='online capacity headroom MW; no sub-hour ramp delivery guarantee',
                feasibility_tol=feasibility_tol, integrality_tol=integrality_tol)
