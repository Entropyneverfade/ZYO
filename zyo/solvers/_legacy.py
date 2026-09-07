# 将旧结果字段映射到 ZYO 公共契约，并保留算法、参数和回退信息。
"""Shared conversion from the retained 0.1 engine result to ZYO contracts."""

from dataclasses import asdict, replace

from zyo.result import Result
from zyo.status import Status
from zyo.validation import evaluate_objective, validate_candidate


def convert_legacy_result(
    model,
    options,
    raw,
    info,
    backend_kind,
    *,
    parameters_applied=None,
    parameters_unmapped=None,
    validation_parameters=None,
):
    result = Result.from_legacy(raw)
    requested = asdict(options)
    metadata = {
        "backend_kind": backend_kind,
        "license_mode": info.license_mode,
        "raw_status": raw.status,
        "raw_message": raw.message,
        "raw_objective": raw.objective,
        "raw_gap": raw.gap,
        "parameters": requested,
        "parameters_requested": requested,
        "parameters_applied": parameters_applied or {},
        "parameters_unmapped": parameters_unmapped or [],
        "validation_parameters": validation_parameters
        or {
            "feasibility_tol": options.feasibility_tol,
            "integrality_tol": options.integrality_tol,
            "objective_tol": options.objective_tol,
        },
        "fallback_requested": False,
        "fallback_used": False,
    }
    result = replace(
        result,
        solver_name=info.name,
        solver_version=info.version,
        metadata=metadata,
    )
    if not raw.has_solution:
        return result

    residuals = validate_candidate(
        model,
        raw.values,
        options.feasibility_tol,
        options.integrality_tol,
    )
    objective = evaluate_objective(model, raw.values)
    mismatch = (
        raw.objective is None
        or abs(objective - raw.objective)
        > max(
            options.objective_tol,
            options.feasibility_tol * max(1.0, abs(objective)),
        )
    )
    if not residuals.is_feasible or mismatch:
        reason = "Independent candidate validation failed"
        if mismatch:
            reason += "; reported and recomputed objectives differ"
        return replace(
            result,
            status=Status.NUMERICAL_ERROR,
            objective=objective,
            primal_residual=residuals.constraint,
            bound_residual=residuals.bound,
            integrality_residual=residuals.integrality,
            termination_reason=reason,
        )
    return replace(
        result,
        objective=objective,
        primal_residual=residuals.constraint,
        bound_residual=residuals.bound,
        integrality_residual=residuals.integrality,
    )
