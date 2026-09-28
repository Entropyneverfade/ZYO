# 公共 LP 装配存储量测量：同一原模型分别进入稠密参考和 CSC 容器，只比矩阵缓冲区。
"""复现公开稀疏 LP 入口的人工对角算例和矩阵缓冲区存储量。"""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import scipy

from zyo import Model, Status
from zyo.modeling.matrix import model_arrays, model_sparse_arrays


SIZES = (200, 600, 1000, 1420)
FORBIDDEN = ('scipy.optimize', 'highspy', 'gurobipy', 'coptpy',
             'zyo.solvers.highs', 'zyo.solvers.gurobi', 'zyo.solvers.copt')


def build_case(size):
    """构造可手算的无量纲稀疏 LP：目标 0，所有行仅一个非零元。"""
    if not isinstance(size, int) or size < 1:
        raise ValueError('size must be a positive integer')
    model = Model(f'diagonal_lp_{size}')
    variables = [model.add_var(f'x{i}', lb=0, ub=1) for i in range(size)]
    for variable in variables:
        model.add_constr(variable <= 1)
    model.minimize(variables[0])
    return model


def measure_case(size, solve=False):
    """逐元素核对原矩阵，再量取真实 ndarray/CSC 三个缓冲区的字节。"""
    model = build_case(size)
    dense, *_ = model_arrays(model, max_dense_entries=None)
    sparse, *_ = model_sparse_arrays(model)
    row = {
        'rows': size, 'columns': size, 'nonzeros': int(sparse.nnz),
        'dense_bytes': int(dense.nbytes),
        'csc_bytes': int(sparse.data.nbytes+sparse.indices.nbytes+sparse.indptr.nbytes),
        'matrices_equal': bool(np.array_equal(dense, sparse.toarray())),
        'model_definition': 'min x0; 0<=xi<=1; xi<=1 for every i; dimensionless',
    }
    if not row['matrices_equal']:
        raise ArithmeticError('dense and CSC representations differ')
    if solve:
        # 公共入口求解与装配量分开记录；不把存储量变化解释成求解加速。
        result = model.solve('native_simplex')
        verified = (result.metadata.get('certificate') or {}).get('verified') is True
        if (result.status is not Status.OPTIMAL or result.objective is None
                or abs(result.objective) > 1e-9 or not verified
                or result.metadata.get('fallback_used') is not False
                or result.solver_name != 'native_simplex'):
            raise RuntimeError('public native LP solve did not meet the preregistered optimum/certificate gate')
        row.update(status=result.status.value, objective=result.objective,
                   certificate_verified=verified,
                   fallback_used=result.metadata['fallback_used'],
                   solver_name=result.solver_name, solver_version=result.solver_version,
                   solve_seconds=result.runtime,
                   assembly_seconds=result.metadata.get('assembly_seconds'))
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path,
                        help='new output directory for measured JSON and CSV')
    args = parser.parse_args(argv)
    output = args.output.resolve()
    rows = [measure_case(size, solve=size == 1420) for size in SIZES]
    loaded = [name for name in sys.modules if any(name == item or name.startswith(item+'.')
                                                  for item in FORBIDDEN)]
    if loaded:
        raise RuntimeError(f'external optimizer modules loaded: {loaded}')
    # 全部数值/引擎门通过后才创建发布目录；失败不留下看似完整的结果。
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'kind': 'synthetic diagonal LP matrix storage, not solver speed',
                'numpy': np.__version__, 'scipy': scipy.__version__,
                'forbidden_modules_loaded': loaded, 'rows': rows}
    with (output/'measurements.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
    with (output/'matrix_storage.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('rows', 'dense_bytes', 'csc_bytes',
                                                     'nonzeros', 'matrices_equal'), lineterminator='\n')
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in rows)
    print(json.dumps({'output': str(output), 'status_1420': rows[-1].get('status'),
                      'csc_bytes_1420': rows[-1]['csc_bytes']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
