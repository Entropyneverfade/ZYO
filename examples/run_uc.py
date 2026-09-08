# 安装包可运行的原生机组组合示例；外部优化器禁导入，无自动回退。
"""Native UC example: python -I run_uc.py --input uc_24h.json --output fresh-dir --plot."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.abc
import json
import math
import os
from pathlib import Path
import sys
import time

FORBIDDEN = ('scipy.optimize','highspy','gurobipy','coptpy',
             'zyo.solvers.highs','zyo.solvers.gurobi','zyo.solvers.copt')


class NativeOnly(importlib.abc.MetaPathFinder):
    def __init__(self, engine='native_sparse'):
        # 稀疏路径不得加载密集旧求解链；显式密集原生路径保留其合法依赖。
        self.names = FORBIDDEN + (('lzyopt.solver',) if engine=='native_sparse' else ())

    def find_spec(self, fullname, path=None, target=None):
        if any(fullname==p or fullname.startswith(p+'.') for p in self.names):
            raise ImportError('Native UC embargo: '+fullname)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--engine',choices=['native','native_sparse'],default='native_sparse')
    parser.add_argument('--plot',action='store_true')
    args=parser.parse_args()
    guard=NativeOnly(args.engine)
    # 必须在导入数值包之前固定线程数，并拒绝已经污染的解释器。
    def loaded():
        return [n for n in sys.modules if any(n==p or n.startswith(p+'.') for p in guard.names)]
    if loaded(): raise RuntimeError('External optimizer already loaded')
    sys.meta_path.insert(0,guard)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[name]='1'
    from zyo import SolveOptions
    from zyo.validation import validate_candidate, evaluate_objective
    from zyo_power.unit_commitment import build_uc_model, extract_uc_trace, validate_uc
    from zyo_power.uc_plot import save_json, export_trace, plot_dispatch
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=False)
    original=args.input.read_bytes(); case=json.loads(original.decode('utf-8-sig'))
    (output/'input.json').write_bytes(original)
    options=SolveOptions(time_limit=30,threads=1,seed=20260908,mip_gap=0,feasibility_tol=1e-7,integrality_tol=1e-7)
    start=time.perf_counter(); model,mapping=build_uc_model(case); build_seconds=time.perf_counter()-start
    spec=model.to_dict(); save_json(output/'model.json',spec)
    start=time.perf_counter(); result=model.solve(args.engine,options=options); call_seconds=time.perf_counter()-start
    raw=result.to_dict(); audit={}; trace=None; residual=None; objective=None
    if result.has_solution:
        trace=extract_uc_trace(mapping,result.values); audit=validate_uc(case,trace)
        residual=validate_candidate(model,result.values,1e-7,1e-7)
        objective=evaluate_objective(model,result.values)
        export_trace(case,trace,output/'trace.csv')
        save_json(output/'trace.json',trace)
    finite=all(type(x) in (float,int) and math.isfinite(x) for x in (result.objective,result.best_bound,objective,audit.get('cost')))
    accepted=bool(finite and result.status.value=='OPTIMAL' and residual.is_feasible and audit['physical_pass']
        and abs(result.objective-result.best_bound)/max(1,abs(result.objective))<=1e-7
        and abs(result.objective-objective)<=1e-6+1e-7*max(1,abs(result.objective))
        and abs(result.objective-audit['cost'])<=1e-6+1e-7*max(1,abs(result.objective)))
    if loaded() or args.input.read_bytes()!=original: raise RuntimeError('Import or input integrity failure')
    save_json(output/'result.json',dict(result=raw,audit=audit,accepted_optimal=accepted,
        raw_residuals=asdict(residual) if residual else None,raw_objective=objective,options=asdict(options),
        input_sha256=hashlib.sha256(original).hexdigest(),
        model_sha256=hashlib.sha256(json.dumps(spec,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        build_seconds=build_seconds,solve_call_seconds=call_seconds,forbidden_optimizer_modules_loaded=loaded(),
        native_guard_installed=True,fallback_used=False,scope='synthetic teaching/development'))
    if args.plot and trace is not None and audit.get('physical_pass'):
        plot_dispatch(case,trace,output/'dispatch',args.engine,result.status.value)
    print(json.dumps(dict(status=result.status.value,objective=result.objective,accepted_optimal=accepted)))
    return 0 if accepted else 1


if __name__=='__main__': raise SystemExit(main())
