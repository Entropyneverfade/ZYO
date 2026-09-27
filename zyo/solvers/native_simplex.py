# 自研认证修正单纯形后端：齐次编码 + 有界变量修正单纯形 + 独立 KKT 证书。
# 只声明 LP 能力（有整数变量时明确拒绝，不静默当作连续松弛求解）；不导入任何外部优化引擎。
"""Self-developed certified revised-simplex backend.

The public entry point is :func:`zyo.native_lp.solve_certified_lp`. This adapter exists so the
same path is reachable through the normal ``Model.solve('native_simplex')`` API and returns the
project's standard :class:`zyo.Result`.

Two deliberate contracts:

* **LP only.** A model with integer variables is refused with ``SOLVER_ERROR`` and an explicit
  reason rather than silently solved as a relaxation — reporting a fractional point as the
  answer to a MILP is exactly the kind of silent substitution this project forbids.
* **The certificate decides.** When the solver reports ``OPTIMAL`` but the independent KKT
  certificate does not verify, the public status is ``NUMERICAL_ERROR`` and the failed
  conditions are recorded in the metadata; the raw solver status is preserved alongside so the
  disagreement is auditable instead of hidden.
"""
import math
import time
from dataclasses import asdict

import numpy as np
from scipy.sparse import csc_matrix

from zyo._version import __version__
from zyo.capabilities import LP
from zyo.result import Result
from zyo.status import Status
from zyo.validation import evaluate_objective, validate_candidate

from .base import BackendInfo

# 稠密装配的预登记规模门（行×列 的 double 个数）。这个后端本来就是**稠密表**实现，因此
# 超过门的模型必须在分配之前就知道自己进不去，否则进程会先申请 rows×columns×8 字节，
# 被系统 OOM 杀掉、连一条结论都留不下。实测 `neos-4722843-widden`（113555 行 × 77723 列）
# 要求 70.6 GB —— 一个 12.6 MB 的 MPS 文件；冻结批次 -07 里 38/48 个 worker 就是这样丢的。
# 门值取 2_000_000（16 MB），并按 `DEFAULT_DENSE_ENTRY_GATE` 暴露给调用方复用。
DEFAULT_DENSE_ENTRY_GATE = 2_000_000

# 求解器状态 -> 公共状态。OPTIMAL 不在此表中：它必须由证书确认后才允许出现。
_STATUS_MAP = {
    'INFEASIBLE': Status.INFEASIBLE,
    'UNBOUNDED': Status.UNBOUNDED,
    'UNBOUNDED_OR_INFEASIBLE': Status.INF_OR_UNBD,
    'ITERATION_LIMIT': Status.ITERATION_LIMIT,
    'NUMERICAL_ERROR': Status.NUMERICAL_ERROR,
    'START_INFEASIBLE': Status.NUMERICAL_ERROR,
    'CERTIFICATE_FAILED': Status.NUMERICAL_ERROR,
}


def model_arrays(model, max_dense_entries=DEFAULT_DENSE_ENTRY_GATE):
    """Dense rows, cost, bounds, rhs and senses of a continuous ``zyo.Model``.

    Rows keep the repository-wide "activity SENSE 0" convention: the stored expression constant
    is ``-rhs``, so the real right-hand side is ``-constant``.

    ``max_dense_entries`` gates the *allocation*, not just the use: the dense matrix costs
    ``8·rows·columns`` bytes, and on large sparse models that dwarfs the model itself (a 12.6 MB MPS
    with 113555×77723 shape asks for 70.6 GB). Exceeding the gate raises ``MemoryError`` with the
    requested size, so a caller can report an honest size limit instead of being OOM-killed with no
    record. Pass ``None`` to allocate unconditionally (only for small models and tests).
    """
    columns = len(model.variables)
    rows = len(model.constraints)
    if max_dense_entries is not None and rows*columns > max_dense_entries:
        raise MemoryError(
            f'the dense tableau needs {rows*columns} entries ({rows*columns*8/1e9:.2f} GB); '
            f'the predeclared gate is {max_dense_entries} entries '
            f'({max_dense_entries*8/1e6:.1f} MB)')
    matrix = np.zeros((rows, columns))
    rhs = np.zeros(rows)
    sense = []
    for index, row in enumerate(model.constraints):
        rhs[index] = -row.expression.constant
        for j, coefficient in row.expression.terms.items():
            matrix[index, j] = coefficient
        sense.append(row.sense)
    cost = np.zeros(columns)
    for j, coefficient in model.objective.terms.items():
        cost[j] = coefficient
    lower = np.array([v.lb if math.isfinite(v.lb) else -math.inf for v in model.variables])
    upper = np.array([v.ub if math.isfinite(v.ub) else math.inf for v in model.variables])
    return matrix, cost, lower, upper, rhs, sense, float(model.objective.constant)


def model_sparse_arrays(model):
    """Sparse rows plus cost, bounds, rhs and senses — same semantics as :func:`model_arrays`.

    The certified path consumes ``csc_matrix`` throughout (``build_homogeneous`` and
    ``solve_certified_lp`` both call ``csc_matrix(matrix)``), so the dense intermediate was never
    required by the solver: it existed only because this adapter allocated one. Building the CSC
    structure directly means the memory cost is ``O(nonzeros)`` instead of ``O(rows·columns)``, which
    is what lets the large sparse MIPLIB instances reach the certified relaxation at all (the dense
    form asks for 70.6 GB on `neos-4722843-widden`).

    Rows keep the repository-wide "activity SENSE 0" convention (``-rhs`` is stored as the constant).
    """
    import scipy.sparse as sp

    columns = len(model.variables)
    rows = len(model.constraints)
    row_index, column_index, entries = [], [], []
    rhs = np.zeros(rows)
    sense = []
    for index, row in enumerate(model.constraints):
        rhs[index] = -row.expression.constant
        for j, coefficient in row.expression.terms.items():
            row_index.append(index)
            column_index.append(int(j))
            entries.append(float(coefficient))
        sense.append(row.sense)
    matrix = sp.csc_matrix((np.asarray(entries, dtype=float),
                            (np.asarray(row_index, dtype=np.int64),
                             np.asarray(column_index, dtype=np.int64))),
                           shape=(rows, columns))
    cost = np.zeros(columns)
    for j, coefficient in model.objective.terms.items():
        cost[int(j)] = float(coefficient)
    lower = np.array([v.lb if math.isfinite(v.lb) else -math.inf for v in model.variables])
    upper = np.array([v.ub if math.isfinite(v.ub) else math.inf for v in model.variables])
    return matrix, cost, lower, upper, rhs, sense, float(model.objective.constant)


class NativeSimplexBackend:
    def info(self):
        return BackendInfo(
            name='native_simplex', version=__version__,
            available=True, capabilities=frozenset({LP}), license_mode='self-developed',
            message=('ZYO-authored bounded-variable revised simplex with product-form basis '
                     'updates and an independent KKT certificate; LP only'))

    def solve(self, model, options):
        from zyo.native_lp import solve_certified_lp

        info = self.info()
        requested = asdict(options)
        base_metadata = {
            'backend_kind': 'self-developed',
            'license_mode': info.license_mode,
            'algorithm': 'homogeneous row-bound encoding + bounded-variable primal revised simplex',
            'parameters': requested,
            'parameters_requested': requested,
            'validation_parameters': {
                'feasibility_tol': options.feasibility_tol,
                'integrality_tol': options.integrality_tol,
                'objective_tol': options.objective_tol,
            },
            'fallback_requested': False,
            'fallback_used': False,
        }
        integer = [v.name for v in model.variables if v.kind != 'C']
        if integer:
            return Result(
                status=Status.SOLVER_ERROR, solver_name=info.name, solver_version=info.version,
                objective=None, best_bound=None, mip_gap=None, values={}, runtime=0.0,
                node_count=0, iteration_count=0,
                termination_reason=(f'this backend solves LP relaxations only; the model has '
                                    f'{len(integer)} integer variable(s): {integer[:5]}'),
                metadata=dict(base_metadata, capability='LP', integer_variables=len(integer)))
        if options.mip_gap not in (None, 0.0):
            return Result(
                status=Status.SOLVER_ERROR, solver_name=info.name, solver_version=info.version,
                objective=None, best_bound=None, mip_gap=None, values={}, runtime=0.0,
                node_count=0, iteration_count=0,
                termination_reason='mip_gap is not applicable to this LP backend',
                metadata=dict(base_metadata, capability='LP'))

        try:
            matrix, cost, lower, upper, rhs, sense, constant = model_arrays(model)
        except MemoryError as error:
            # 规模门：如实上报 SIZE_LIMIT 与实测维度/字节数，而不是让进程撞内存墙后什么都不留。
            columns, rows = len(model.variables), len(model.constraints)
            return Result(
                status=Status.SIZE_LIMIT, solver_name=info.name, solver_version=info.version,
                objective=None, best_bound=None, mip_gap=None, values={}, runtime=0.0,
                node_count=0, iteration_count=0,
                termination_reason=(f'the dense tableau is {rows} x {columns} '
                                    f'({rows*columns*8/1e9:.2f} GB), above this backend\'s '
                                    f'predeclared gate of {DEFAULT_DENSE_ENTRY_GATE} entries; '
                                    f'use the sparse backend for this size ({error})'),
                metadata=dict(base_metadata, capability='LP', dense_entries=rows*columns,
                              dense_entry_gate=DEFAULT_DENSE_ENTRY_GATE,
                              dense_bytes=rows*columns*8))
        started = time.perf_counter()
        outcome = solve_certified_lp(matrix, cost, lower, upper, rhs, sense=sense,
                                    maximize=model.sense == 'max',
                                    iteration_limit=int(options.iteration_limit),
                                    feasibility_tolerance=float(options.feasibility_tol),
                                    objective_tolerance=float(options.objective_tol))
        runtime = time.perf_counter()-started
        certificate = outcome.certificate
        metadata = dict(
            base_metadata, capability='LP', raw_status=outcome.status,
            raw_message=outcome.message, raw_gap=None if certificate is None
            else certificate.duality_gap,
            raw_objective=outcome.objective, pivots=int(outcome.pivots),
            refactorisations=int(outcome.refactorisations),
            phase_one_used=bool((outcome.phase_one or {}).get('used')),
            basis_diagnostics=dict(outcome.record.get('basis_diagnostics') or {}),
            phase_one_diagnostics=dict((outcome.phase_one or {}).get('basis_diagnostics') or {}),
            iteration_budget=dict(outcome.record.get('iteration_budget') or {}),
            bound_flips=int(outcome.record.get('bound_flips', 0)),
            warm_start=dict(outcome.record.get('warm_start') or {}),
            certificate=None if certificate is None else dict(
                certificate.checks, verified=bool(certificate.verified),
                reason=certificate.reason, duality_gap=certificate.duality_gap,
                relative_gap=certificate.relative_gap,
                row_sign_violation=certificate.row_sign_violation,
                multiplier_scale=certificate.multiplier_scale),
        )
        if not outcome.optimal:
            return Result(
                status=_STATUS_MAP.get(outcome.status, Status.UNKNOWN), solver_name=info.name,
                solver_version=info.version, objective=None, best_bound=None, mip_gap=None,
                values={}, runtime=runtime, node_count=0,
                iteration_count=int(outcome.iterations), termination_reason=outcome.message,
                metadata=metadata)

        values = {model.variables[index].name: float(value)
                  for index, value in enumerate(outcome.values)}
        residuals = validate_candidate(model, values, options.feasibility_tol,
                                       options.integrality_tol)
        objective = evaluate_objective(model, values)
        if not residuals.is_feasible:
            return Result(
                status=Status.NUMERICAL_ERROR, solver_name=info.name,
                solver_version=info.version, objective=objective, best_bound=None, mip_gap=None,
                values=values, runtime=runtime, node_count=0,
                iteration_count=int(outcome.iterations),
                primal_residual=residuals.constraint, bound_residual=residuals.bound,
                integrality_residual=residuals.integrality,
                termination_reason=('the KKT certificate verified but the independent candidate '
                                    'validation in ORIGINAL model units failed'),
                metadata=metadata)
        return Result(
            status=Status.OPTIMAL, solver_name=info.name, solver_version=info.version,
            objective=objective, best_bound=objective, mip_gap=0.0, values=values,
            runtime=runtime, node_count=0, iteration_count=int(outcome.iterations),
            primal_residual=residuals.constraint, bound_residual=residuals.bound,
            integrality_residual=residuals.integrality,
            termination_reason=('optimal basis with reduced costs verified by an independent '
                                'KKT certificate in original units'),
            metadata=metadata)


BACKEND = NativeSimplexBackend()
