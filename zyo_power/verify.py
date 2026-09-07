# 只读复核存档哈希与原始 CSV 物理指标；路径越界或缺失数据不能计为通过。
"""Read-only SHA256 and raw exported CSV physical validation."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .data import StorageCase
from .validation import validate_dispatch


def _load(path):
    with path.open(encoding='utf-8-sig') as stream:
        return json.load(stream)


def _inside(root,name):
    path=(root/name).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Manifest path escapes archive')
    return path


def verify_archive(root):
    root=Path(root).resolve()
    manifest=_load(root/'manifest.json')
    mismatches=[]
    for name,evidence in manifest['files'].items():
        path=_inside(root,name)
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=evidence['sha256']:
            mismatches.append(name)
    output=dict(hashes_pass=not mismatches,mismatches=mismatches,checked_files=len(manifest['files']),
                checked_traces=0,all_exported_traces_physical_pass=False,checks={},
                scope='Raw CSV physics only; missing/limited solver results are NOT upgraded to optimal or successful')
    if mismatches:
        return output
    for info_path in sorted(root.glob('*/verification.json')):
        info=_load(info_path)
        case=StorageCase.from_dict(_load(_inside(root,info['input_file'])))
        directory=info_path.parent
        with (directory/'hourly.csv').open(encoding='utf-8-sig',newline='') as stream:
            rows=list(csv.DictReader(stream))
        with (directory/'states.csv').open(encoding='utf-8-sig',newline='') as stream:
            states=list(csv.DictReader(stream))
        try:
            if [int(r['interval']) for r in rows]!=list(range(case.periods)):
                raise ValueError('Hourly interval IDs not complete and unique')
            if [int(r['state_timepoint']) for r in states]!=list(range(case.periods+1)):
                raise ValueError('Energy timepoint IDs not complete and unique')
            elapsed=np.r_[0,np.cumsum(case.dt)]
            if not np.allclose([float(r['elapsed_hours']) for r in states],elapsed,atol=1e-9,rtol=0):
                raise ValueError('State elapsed time mismatch')
            trace={key:np.array([float(r[column]) for r in rows]) for key,column in
                   [('thermal','thermal_mw'),('curtailment','curtailment_mw'),('shed','shed_mw')]}
            for key in ['charge','discharge']:
                trace[key]=np.array([[float(r[b.name+'_'+key+'_mw']) for b in case.batteries] for r in rows])
            trace['energy']=np.array([[float(r[b.name+'_energy_mwh']) for b in case.batteries] for r in states])
            check=validate_dispatch(case,{'trace':trace},strict=info['strict'])
            check.update(actual_engine=info['engine'],reported_status=info['status'],proof_scope=info['proof_scope'])
        except (KeyError,ValueError,TypeError) as exc:
            check=dict(physical_pass=False,error=str(exc))
        output['checks'][directory.name]=check
    output['checked_traces']=len(output['checks'])
    output['all_exported_traces_physical_pass']=bool(output['checks']) and all(c['physical_pass'] for c in output['checks'].values())
    return output


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive')
    args=parser.parse_args()
    report=verify_archive(args.archive)
    print(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False))
    raise SystemExit(0 if report['hashes_pass'] and report['all_exported_traces_physical_pass'] else 1)
