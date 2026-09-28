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
import time
from dataclasses import asdict

from zyo._version import __version__
from zyo.capabilities import LP
from zyo.modeling.matrix import DEFAULT_DENSE_ENTRY_GATE, model_arrays, model_sparse_arrays
from zyo.result import Result
from zyo.status import Status
from zyo.validation import evaluate_objective, validate_candidate

from .base import BackendInfo

# 稠密参考入口保留旧名称与规模门；公共求解器仅走同语义的稀疏装配。

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

        assembly_started = time.perf_counter()
        try:
            matrix, cost, lower, upper, rhs, sense, constant = model_sparse_arrays(model)
        except MemoryError as error:
            # 稀疏装配若仍遇到资源不足，保留原状态，不转用第三方引擎或稠密路径。
            columns, rows = len(model.variables), len(model.constraints)
            return Result(
                status=Status.SIZE_LIMIT, solver_name=info.name, solver_version=info.version,
                objective=None, best_bound=None, mip_gap=None, values={}, runtime=0.0,
                node_count=0, iteration_count=0,
                termination_reason=(f'sparse model assembly failed for {rows} x {columns}: {error}'),
                metadata=dict(base_metadata, capability='LP', matrix_format='csc',
                              model_rows=rows, model_columns=columns,
                              assembly_seconds=time.perf_counter()-assembly_started,
                              resource_failure_stage='assembly'))
        # 与旧 Result.runtime 的“内核求解秒”口径分列，供端到端成本剖析使用。
        base_metadata['assembly_seconds'] = time.perf_counter()-assembly_started
        started = time.perf_counter()
        try:
            outcome = solve_certified_lp(matrix, cost, lower, upper, rhs, sense=sense,
                                        maximize=model.sense == 'max',
                                        iteration_limit=int(options.iteration_limit),
                                        feasibility_tolerance=float(options.feasibility_tol),
                                        objective_tolerance=float(options.objective_tol))
        except MemoryError as error:
            # 稀疏装配之外的齐次编码、基LU或证书也可能分配失败；保留实际阶段与规模。
            return Result(
                status=Status.SIZE_LIMIT, solver_name=info.name, solver_version=info.version,
                objective=None, best_bound=None, mip_gap=None, values={},
                runtime=time.perf_counter()-started, node_count=0, iteration_count=0,
                termination_reason=f'native sparse LP solve allocation failed: {error}',
                metadata=dict(base_metadata, capability='LP', matrix_format='csc',
                              model_rows=int(matrix.shape[0]), model_columns=int(matrix.shape[1]),
                              model_nonzeros=int(matrix.nnz), resource_failure_stage='solve'))
        runtime = time.perf_counter()-started
        try:
            certificate = outcome.certificate
            metadata = dict(
                base_metadata, capability='LP', matrix_format='csc', model_nonzeros=int(matrix.nnz),
                model_rows=int(matrix.shape[0]), model_columns=int(matrix.shape[1]),
                raw_status=outcome.status,
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
        except MemoryError as error:
            # 结果映射与独立复核阶段失败时，不把求解器的候选状态升级为最优。
            return Result(
                status=Status.SIZE_LIMIT, solver_name=info.name, solver_version=info.version,
                objective=None, best_bound=None, mip_gap=None, values={}, runtime=runtime,
                node_count=0, iteration_count=int(outcome.iterations),
                termination_reason=f'native LP result processing allocation failed: {error}',
                metadata=dict(base_metadata, capability='LP', matrix_format='csc',
                              model_rows=int(matrix.shape[0]), model_columns=int(matrix.shape[1]),
                              model_nonzeros=int(matrix.nnz), raw_status=outcome.status,
                              resource_failure_stage='result_processing'))


BACKEND = NativeSimplexBackend()
