# 储能实验管理及逐时归档；历史综合入口含外部比较，不作为自主计算默认流程。
"""Local, append-only reproducible storage experiments (no solver fallback)."""
import csv
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time

import numpy as np
from .data import StorageCase
from .examples import teaching_case, tiny_case, synthetic_case
from .periodic import initialization_study
from .rule import rule_dispatch
from .validation import validate_dispatch


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)  # Explicit nonfinite diagnostic, never silently zero.
    return value


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(_jsonable(value), stream, ensure_ascii=False, indent=2, allow_nan=False)


def _csv(path, header, rows):
    with path.open('x', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _measure(call):
    """Observed process RSS, not an allocator exact maximum or total child memory."""
    try:
        import psutil
    except ImportError:
        return call(), {'rss_measurement':'unavailable: optional psutil not installed'}
    process=psutil.Process()
    baseline=process.memory_info().rss
    peak=[baseline]
    stop=threading.Event()
    def sample():
        while not stop.wait(.02):
            peak[0]=max(peak[0],process.memory_info().rss)
    worker=threading.Thread(target=sample,daemon=True)
    worker.start()
    try:
        result=call()
    finally:
        stop.set()
        worker.join()
        peak[0]=max(peak[0],process.memory_info().rss)
    return result,dict(rss_measurement='same process sampled every 20ms; includes Python and retained experiment data',
                       baseline_process_rss_bytes=baseline,peak_observed_process_rss_bytes=peak[0])


def _provenance():
    root = Path(__file__).resolve().parent.parent
    def git(*args):
        result = subprocess.run(['git', '-c', f'safe.directory={root.as_posix()}', *args],
                                cwd=root, text=True, capture_output=True, encoding='utf-8', errors='replace')
        return result.stdout.strip() if result.returncode == 0 else None
    versions = {}
    for name in ['ZYO', 'numpy', 'scipy', 'matplotlib', 'psutil']:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    try:
        head, dirty = git('rev-parse', 'HEAD'), git('status', '--porcelain')
    except FileNotFoundError:
        head, dirty = None, None
    return dict(utc=datetime.now(timezone.utc).isoformat(), python=sys.version,
                executable=sys.executable, platform=platform.platform(), processor=platform.processor(),
                logical_cpus=os.cpu_count(), versions=versions, git_head=head, git_status=dirty,
                source_root=str(root), source_hashes={p.name:_sha(p) for p in sorted((root/'zyo_power').glob('*.py'))},
                thread_environment={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},
                thread_policy='solver threads=1; BLAS thread settings are recorded, not silently altered',
                dependencies_boundary='rule/iteration: NumPy; native: ZYO+NumPy; native_sparse: ZYO+NumPy+SciPy sparse/SuperLU; external optimizers isolated comparisons only',
                scientific_scope='deterministic ENS, not EENS/LOLE; virtual cycles are not independent samples',
                checkpoint_scope='completed experiment artifacts only; no solver-state restart')


class Archive:
    def __init__(self, root, plots):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)
        self.plots = plots
        self.inputs = {}
        provenance=_provenance()
        _write_json(self.root/'environment.json', provenance)
        source_root=Path(__file__).resolve().parent.parent
        for package in ['zyo_power','zyo','lzyopt','tests']:
            for path in (source_root/package).rglob('*.py'):
                target=self.root/'source'/path.relative_to(source_root)
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(path,target)
        for path in (source_root/'tests'/'data').rglob('*.json'):
            target=self.root/'source'/path.relative_to(source_root)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,target)
        for name in ['pyproject.toml','_zyo_version.py','ZYO.py']:
            path=source_root/name
            if path.is_file():
                shutil.copy2(path,self.root/'source'/name)
        with (self.root/'dependencies.txt').open('x',encoding='utf-8') as stream:
            for name,version in sorted({d.metadata['Name']:d.version for d in metadata.distributions() if d.metadata['Name']}.items()):
                stream.write(f'{name}=={version}\n')
        if provenance['git_head']:
            diff=subprocess.run(['git','-c',f'safe.directory={source_root.as_posix()}',
                                 'diff','--binary','HEAD'],cwd=source_root,capture_output=True,check=True)
            with (self.root/'tracked-changes.patch').open('xb') as stream:
                stream.write(diff.stdout)

    def input(self, key, case, parts=None):
        self.inputs[case.name]=f'inputs/{key}.json'
        _write_json(self.root/'inputs'/f'{key}.json', case.to_dict())
        columns = {'load_mw':case.load, 'renewable_mw':case.renewable,
                   'thermal_capacity_mw':case.thermal_capacity, 'thermal_cost_per_mwh':case.thermal_cost,
                   'dt_hours':case.dt}
        columns.update({k+'_mw':v for k,v in (parts or {}).items()})
        _csv(self.root/'inputs'/f'{key}.csv', ['interval']+list(columns),
             ([t]+[v[t] for v in columns.values()] for t in range(case.periods)))

    def _models(self, value):
        # Deduplicate complete serialized models, not the actual solutions/logs.
        if isinstance(value, dict):
            output = {}
            for key, item in value.items():
                if key == 'model' and isinstance(item, dict) and 'model_hash' in value:
                    name = f"models/{value['model_hash']}.json"
                    path = self.root/name
                    if not path.exists():
                        _write_json(path, item)
                    output['model_file'] = name
                else:
                    output[key] = self._models(item)
            return output
        if isinstance(value, list):
            return [self._models(v) for v in value]
        return value

    def dispatch(self, name, case, result):
        directory = self.root/name
        directory.mkdir(exist_ok=True)
        _write_json(directory/'dispatch.json', self._models(result))
        trace = result.get('trace')
        if trace is None:
            return
        _write_json(directory/'verification.json',dict(input_file=self.inputs[case.name],
                    status=result['status'],engine=result['engine'],mode=result['mode'],
                    strict=result.get('metadata',{}).get('strict',False),
                    proof_scope=result.get('metadata',{}).get('proof_scope','solver status only; independent physical checks are not optimality proofs')))
        header = ['interval','elapsed_hours','load_mw','renewable_mw','thermal_mw','curtailment_mw','shed_mw']
        for b in case.batteries:
            header += [b.name+'_charge_mw', b.name+'_discharge_mw']
        elapsed = np.r_[0, np.cumsum(case.dt)]
        _csv(directory/'hourly.csv', header, ([t,elapsed[t],case.load[t],case.renewable[t],
             trace['thermal'][t],trace['curtailment'][t],trace['shed'][t]]+
             [trace[k][t,s] for s in range(len(case.batteries)) for k in ['charge','discharge']]
             for t in range(case.periods)))
        _csv(directory/'states.csv', ['state_timepoint','elapsed_hours']+
             [b.name+'_energy_mwh' for b in case.batteries]+
             [b.name+'_soc_fraction' for b in case.batteries],
             ([t,elapsed[t]]+list(trace['energy'][t])+
              [trace['energy'][t,s]/b.energy_max if b.energy_max else 0 for s,b in enumerate(case.batteries)]
              for t in range(case.periods+1)))
        if self.plots:
            from .plotting import plot_dispatch
            plot_dispatch(case,result,directory/'dispatch.png')

    def finish(self, summary):
        _write_json(self.root/'summary.json', summary)
        manifest = {p.relative_to(self.root).as_posix():dict(sha256=_sha(p),bytes=p.stat().st_size)
                    for p in sorted(self.root.rglob('*')) if p.is_file()}
        _write_json(self.root/'manifest.json', dict(schema='zyo-storage-artifacts-1',files=manifest))


def _study(archive, name, case, policy, max_cycles, strict=False):
    begin=time.perf_counter()
    result,memory=_measure(lambda:initialization_study(case,policy,max_cycles=max_cycles,strict=strict))
    _write_json(archive.root/(name+'.json'),archive._models(result))
    output=dict(status=result['initialization_status'],runs={},runtime_seconds=time.perf_counter()-begin,memory=memory)
    for label,run in zip(['low','mid','high'],result['runs']):
        output['runs'][label]={k:v for k,v in run.items() if k not in ['history','formal','failed_dispatch','traceback']}
        output['runs'][label]['boundary_history']=[{k:h[k] for k in ['cycle','start_energy_mwh','end_energy_mwh','residual_mwh','role']} for h in run['history']]
        if run['formal'] is not None:
            archive.dispatch(name+'_'+label,case,run['formal'])
    if archive.plots:
        from .plotting import plot_convergence
        plot_convergence(case,result,archive.root/(name+'_convergence.png'))
    print(name,output['status'],[r['status'] for r in result['runs']],flush=True)
    return output


def _optimization(archive,name,case,engine,settings,strict=False):
    from .optimization import optimize_dispatch
    begin=time.perf_counter()
    result,memory=_measure(lambda:optimize_dispatch(case,engine=engine,options=settings,cyclic=True,strict=strict))
    audit=validate_dispatch(case,result,strict=strict)
    archive.dispatch(name,case,result)
    meta=result['metadata']
    objective_check=None
    if audit['physical_pass'] and meta['solver_result'] is not None:
        objective=meta['solver_result']['objective']
        expected=audit['metrics']['cost_with_ens_penalty']
        objective_check=objective is not None and abs(objective-expected)<=1e-6+1e-7*max(abs(objective),abs(expected))
    output=dict(status=result['status'],actual_engine=result['engine'],audit=audit,
                model_hash=meta['model_hash'],scale=meta['model_scale'],options=meta['options'],
                solver_result=meta['solver_result'],runtime_seconds=time.perf_counter()-begin,
                fallback_used=meta['fallback_used'],memory=memory,objective_recalculation_pass=objective_check)
    print(name,result['status'],'physical_pass=',audit['physical_pass'],flush=True)
    return output


def run_suite(output, *, suite='all', plots=True, max_cycles=8, seed=20260907):
    if suite not in ['teaching','small','week','year-rule','year-rolling','all']:
        raise ValueError('Unknown suite')
    archive=Archive(output,plots)
    started=time.perf_counter()
    summary=dict(schema='zyo-storage-experiment-1',suite=suite,seed=seed,
                 settings=dict(max_cycles=max_cycles,atol_power_mw=1e-7,atol_energy_mwh=1e-7,
                               rtol_energy=1e-9,objective_atol=1e-6,objective_rtol=1e-7,
                               model_exclusivity='MILP',ens_penalty_per_mwh=10000),
                 validation_population='constructed/synthetic development cases; not frozen holdout acceptance')
    teaching=teaching_case()
    archive.input('teaching',teaching)
    summary['teaching_rule']=_study(archive,'teaching_rule',teaching,
                                   lambda energy:rule_dispatch(teaching,energy),max_cycles)
    if summary['teaching_rule']['status']!='SAME_OBSERVED_CYCLE':
        summary['expansion_gate']='BLOCKED: teaching rule did not confirm'
        archive.finish(summary)
        return summary
    observed=summary['teaching_rule']['runs']['low']
    expected_end=[7.7052631579,14.7368421053,14.7368421053]
    history=observed['boundary_history']
    reference_ok=len(history)>=4 and all(abs(history[i]['end_energy_mwh'][0]-e)<=1e-7 for i,e in enumerate(expected_end))
    for key,value in [('charge_mwh',39.8891966759),('discharge_mwh',36.),('curtailment_mwh',8.1108033241),('ens_mwh',0.)]:
        reference_ok=reference_ok and abs(observed['formal_audit']['metrics'][key]-value)<=1e-7
    summary['teaching_reference_pass']=reference_ok
    if not reference_ok:
        summary['expansion_gate']='BLOCKED: deterministic teaching reference mismatch'
        archive.finish(summary)
        return summary
    if suite in ['small','week','year-rolling','all']:
        # Equal resource budgets for same-model comparisons, no reruns with looser tolerances.
        settings=dict(time_limit=3.,node_limit=100,threads=1,seed=seed,mip_gap=0.)
        tiny=tiny_case()
        archive.input('tiny',tiny)
        for label,case in [('tiny',tiny),('teaching',teaching)]:
            for engine in ['native','highs']:
                summary[label+'_'+engine]=_optimization(archive,label+'_'+engine,case,engine,settings)
        gate=True
        for label,target in [('tiny',4.),('teaching',0.)]:
            for engine in ['native','highs']:
                result=summary[label+'_'+engine]
                # This bounded development suite requires both engines on these tiny gates.
                gate=gate and result['status']=='OPTIMAL' and result['audit']['physical_pass'] and result['audit'].get('cycle_closed',False)
                gate=gate and result.get('objective_recalculation_pass',False)
                if gate:
                    gate=abs(result['solver_result']['objective']-target)<=1e-6
            gate=gate and summary[label+'_native'].get('model_hash')==summary[label+'_highs'].get('model_hash')
        summary['expansion_gate']='PASS: tiny/teaching optimization and independent references' if gate else 'BLOCKED: small optimization/reference gate failed'
        if not gate:
            archive.finish(summary)
            return summary
    if suite in ['week','year-rolling','all']:
        week,parts=synthetic_case(168,seed)
        archive.input('week',week,parts)
        summary['week_rule']=_study(archive,'week_rule',week,lambda e:rule_dispatch(week,e),max_cycles)
        settings=dict(time_limit=10.,node_limit=1000,threads=1,seed=seed,mip_gap=0.)
        for engine in ['native','highs']:
            summary['week_'+engine]=_optimization(archive,'week_'+engine,week,engine,settings)
        if not (summary['week_highs']['status']=='OPTIMAL' and summary['week_highs']['audit']['physical_pass']
                and summary['week_highs']['audit']['cycle_closed'] and summary['week_highs']['objective_recalculation_pass']):
            summary['expansion_gate']='BLOCKED: weekly full-cycle reference validation failed'
            archive.finish(summary)
            return summary
        from .rolling import rolling_dispatch
        for horizon,value in [(48,0.),(48,25.),(48,50.),(72,25.)]:
            name=f'week_rolling_{horizon}_value{int(value)}'
            summary[name]=_study(archive,name,week,
                 lambda e,h=horizon,v=value:rolling_dispatch(week,e,engine='highs',
                    options=settings,lookahead=h,execute=24,terminal_value=v),max_cycles)
    if suite in ['year-rule','year-rolling','all']:
        annual,parts=synthetic_case(8760,seed)
        archive.input('year',annual,parts)
        summary['year_rule']=_study(archive,'year_rule',annual,lambda e:rule_dispatch(annual,e),max_cycles)
        summary['annual_optimization']='NOT_RUN: annual rule physics only; no annual MILP/native capability claim'
        if suite=='year-rolling':
            if summary['week_rolling_48_value25']['status']=='SAME_OBSERVED_CYCLE':
                summary['year_rolling_48_value25']=_study(archive,'year_rolling_48_value25',annual,
                     lambda e:rolling_dispatch(annual,e,engine='highs',options=settings,
                                                lookahead=48,execute=24,terminal_value=25.),min(max_cycles,4))
                summary['annual_optimization']='HiGHS rolling 48/24 only; full-year global optimization NOT_RUN; native NOT_RUN'
            else:
                summary['annual_optimization']='BLOCKED: weekly rolling gate did not pass'
    comparisons={}
    for label in ['tiny','teaching','week']:
        a,b=summary.get(label+'_native'),summary.get(label+'_highs')
        if a is None or b is None:
            continue
        same=a['model_hash']==b['model_hash']
        proved=all(r['status']=='OPTIMAL' and r['audit']['physical_pass'] and r['audit']['cycle_closed'] for r in [a,b])
        difference=None
        passed=None
        if proved:
            x,y=a['solver_result']['objective'],b['solver_result']['objective']
            difference=abs(x-y)
            passed=same and difference<=1e-6+1e-7*max(abs(x),abs(y))
        comparisons[label]=dict(same_model_hash=same,both_proved_optimal=proved,
                               objective_absolute_difference=difference,comparison_pass=passed,
                               explanation='No objective comparison pass is inferred from an absent native solution')
    summary['native_highs_comparisons']=comparisons
    summary['runtime_seconds']=time.perf_counter()-started
    archive.finish(summary)
    return summary


def run_config(config,output,*,mode='rule',engine='native',plots=True,lookahead=48,terminal_value=25.,max_cycles=30,strict=False):
    with Path(config).open(encoding='utf-8-sig') as stream:
        case=StorageCase.from_dict(json.load(stream))
    archive=Archive(output,plots)
    archive.input('case',case)
    if mode=='rule':
        summary=_study(archive,'rule',case,lambda e:rule_dispatch(case,e,strict=strict),max_cycles,strict=strict)
    elif mode=='full':
        summary=_optimization(archive,'full',case,engine,dict(time_limit=60.,threads=1),strict=strict)
    elif mode=='rolling':
        from .rolling import rolling_dispatch
        summary=_study(archive,'rolling',case,lambda e:rolling_dispatch(case,e,engine=engine,
                       lookahead=lookahead,terminal_value=terminal_value,strict=strict),max_cycles,strict=strict)
    else:
        raise ValueError('Unknown mode')
    archive.finish(summary)
    return summary
