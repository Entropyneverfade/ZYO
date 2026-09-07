# 外部比较结果转换：原始单位下重新检查候选，不直接相信外部成功标志。
"""Build a validated public result from a direct third-party adapter."""

from dataclasses import asdict, replace
import math
from zyo.result import Result
from zyo.status import Status
from zyo.validation import evaluate_objective, validate_candidate


def make_result(
    model,
    options,
    info,
    *,
    status,
    has_solution,
    values,
    reported_objective,
    best_bound,
    raw_gap,
    runtime,
    node_count,
    iteration_count,
    termination_reason,
    metadata,
):
    values = dict(values or {})
    requested = asdict(options)
    raw_best_bound = best_bound
    public_metadata = {
        "backend_kind": "third-party-comparison",
        "license_mode": info.license_mode,
        "raw_objective": (
            float(reported_objective)
            if reported_objective is not None
            and math.isfinite(float(reported_objective))
            else None
        ),
        "raw_gap": (
            float(raw_gap)
            if raw_gap is not None and math.isfinite(float(raw_gap))
            else None
        ),
        "raw_best_bound": (
            float(raw_best_bound)
            if raw_best_bound is not None
            and math.isfinite(float(raw_best_bound))
            else None
        ),
        "parameters": requested,
        "parameters_requested": requested,
        "validation_parameters": {
            "feasibility_tol": options.feasibility_tol,
            "integrality_tol": options.integrality_tol,
            "objective_tol": options.objective_tol,
        },
        "fallback_requested": False,
        "fallback_used": False,
        **metadata,
    }
    public_metadata.setdefault("parameters_applied", {})
    public_metadata.setdefault("parameters_unmapped", [])
    if best_bound is not None:
        best_bound = float(best_bound)
        if not math.isfinite(best_bound):
            best_bound = None
            public_metadata["best_bound_nonfinite_dropped"] = True
    result = Result(
        status=status,
        solver_name=info.name,
        solver_version=info.version,
        objective=None,
        best_bound=best_bound,
        mip_gap=None,
        values=values,
        runtime=float(runtime),
        node_count=int(node_count),
        iteration_count=int(iteration_count),
        termination_reason=termination_reason,
        metadata=public_metadata,
    )
    if not has_solution:
        return result

    residuals = validate_candidate(
        model,
        values,
        options.feasibility_tol,
        options.integrality_tol,
    )
    objective = evaluate_objective(model, values)
    mismatch = (
        reported_objective is None
        or not math.isfinite(float(reported_objective))
        or abs(objective - float(reported_objective))
        > max(
            options.objective_tol,
            options.feasibility_tol * max(1.0, abs(objective)),
        )
    )
    public_gap = None
    bound_mismatch = False
    if best_bound is not None:
        bound_tol = max(
            options.objective_tol,
            options.feasibility_tol * max(1.0, abs(objective)),
        )
        bound_mismatch = (
            best_bound > objective + bound_tol
            if model.sense == "min"
            else best_bound < objective - bound_tol
        )
    if best_bound is not None and not bound_mismatch:
        public_gap = abs(objective - float(best_bound)) / max(1.0, abs(objective))
    checked_status = status
    checked_reason = termination_reason
    if not residuals.is_feasible or mismatch or bound_mismatch:
        checked_status = Status.NUMERICAL_ERROR
        checked_reason = "Independent candidate validation failed"
        if mismatch:
            checked_reason += "; reported and recomputed objectives differ"
        if bound_mismatch:
            checked_reason += "; reported bound has the wrong direction"
    return replace(
        result,
        status=checked_status,
        objective=objective,
        mip_gap=public_gap,
        primal_residual=residuals.constraint,
        bound_residual=residuals.bound,
        integrality_residual=residuals.integrality,
        termination_reason=checked_reason,
    )
