# 仅供隔离比较的 SciPy 内嵌 HiGHS 适配器；实际版本不是独立 highspy 包版本。
"""Explicit SciPy/HiGHS comparison backend."""

from zyo.capabilities import LP, MILP
from zyo.errors import InvalidParameterError, SolverUnavailableError

from ._legacy import convert_legacy_result
from .base import BackendInfo


def _embedded_highs_version():
    from scipy.optimize._highspy import _core

    return ".".join(
        str(value)
        for value in (
            _core.HIGHS_VERSION_MAJOR,
            _core.HIGHS_VERSION_MINOR,
            _core.HIGHS_VERSION_PATCH,
        )
    )


class HighsBackend:
    def info(self):
        try:
            import scipy

            version = _embedded_highs_version()
        except ImportError as exc:
            return BackendInfo(
                name="highs",
                version=None,
                available=False,
                capabilities=frozenset({LP, MILP}),
                license_mode="unavailable",
                message=str(exc),
            )
        return BackendInfo(
            name="highs",
            version=version,
            available=True,
            capabilities=frozenset({LP, MILP}),
            license_mode="open-source",
            message=f"HiGHS {version} through SciPy {scipy.__version__}",
        )

    def solve(self, model, options):
        from lzyopt.solver import solve

        if options.threads != 1:
            raise InvalidParameterError(
                "The SciPy/HiGHS backend currently supports exactly one thread; "
                "SciPy's embedded HiGHS scheduler is process-global and cannot "
                "reliably change thread counts between solves."
            )
        info = self.info()
        if not info.available:
            raise SolverUnavailableError(f"highs unavailable: {info.message}")
        raw = solve(model, engine="highs", options=options.to_legacy())
        mip = any(variable.kind != "C" for variable in model.variables)
        if mip:
            applied = {
                "time_limit": options.time_limit,
                "mip_max_nodes": options.node_limit,
                "mip_rel_gap": options.mip_gap,
                "mip_feasibility_tolerance": max(
                    1e-10,
                    min(options.feasibility_tol, options.integrality_tol),
                ),
                "presolve": options.presolve,
                "threads": options.threads,
                "random_seed": options.seed,
            }
            unmapped = ["iteration_limit", "pivot_tol", "max_tableau_cells"]
        else:
            applied = {
                "time_limit": options.time_limit,
                "simplex_iteration_limit": options.iteration_limit,
                "primal_feasibility_tolerance": max(
                    1e-10,
                    options.feasibility_tol,
                ),
                "presolve": options.presolve,
                "threads": options.threads,
                "random_seed": options.seed,
            }
            unmapped = [
                "node_limit",
                "mip_gap",
                "integrality_tol",
                "pivot_tol",
                "max_tableau_cells",
            ]
        result = convert_legacy_result(
            model,
            options,
            raw,
            info,
            backend_kind="third-party-comparison",
            parameters_applied=applied,
            parameters_unmapped=unmapped,
            validation_parameters={
                "feasibility_tol": options.feasibility_tol,
                "integrality_tol": options.integrality_tol,
                "objective_tol": options.objective_tol,
            },
        )
        return result


BACKEND = HighsBackend()
