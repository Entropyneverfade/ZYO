# 仅供隔离比较的 COPT 适配器；记录许可证与实际状态，不参与自主算法。
"""License-aware optional COPT comparison backend."""

from functools import lru_cache
import math
import os
import sys
import threading

from zyo.capabilities import LP, MILP
from zyo.errors import SolverExecutionError, SolverUnavailableError
from zyo.status import Status

from ._direct import make_result
from .base import BackendInfo


_ENVIRONMENT_OUTPUT_LOCK = threading.Lock()


def _create_quiet_environment(cp):
    """Create COPT without allowing native license diagnostics to corrupt JSON."""
    config = cp.EnvrConfig()
    config.set("nobanner", "1")
    with _ENVIRONMENT_OUTPUT_LOCK:
        sys.stdout.flush()
        sys.stderr.flush()
        null_fd = os.open(os.devnull, os.O_WRONLY)
        stdout_fd = os.dup(1)
        stderr_fd = os.dup(2)
        try:
            os.dup2(null_fd, 1)
            os.dup2(null_fd, 2)
            return cp.Envr(config)
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)
            os.close(null_fd)


def _version(cp):
    return ".".join(
        str(part)
        for part in (
            cp.COPT.VERSION_MAJOR,
            cp.COPT.VERSION_MINOR,
            cp.COPT.VERSION_TECHNICAL,
        )
    )


@lru_cache(maxsize=1)
def _info():
    try:
        import coptpy as cp
    except ImportError as exc:
        return BackendInfo(
            "copt", None, False, frozenset({LP, MILP}), "unavailable", str(exc)
        )
    try:
        environment = _create_quiet_environment(cp)
        environment.close()
    except cp.CoptError as exc:
        return BackendInfo(
            "copt",
            _version(cp),
            False,
            frozenset({LP, MILP}),
            "unavailable",
            f"COPT initialization failed: {exc}",
        )
    return BackendInfo(
        "copt",
        _version(cp),
        True,
        frozenset({LP, MILP}),
        "limited-license",
        "License tier is not exposed by coptpy; conservatively treated as limited",
    )


class CoptBackend:
    def info(self):
        return _info()

    def solve(self, model, options):
        info = self.info()
        if not info.available:
            raise SolverUnavailableError(f"copt unavailable: {info.message}")
        import coptpy as cp

        status_map = {
            cp.COPT.OPTIMAL: Status.OPTIMAL,
            cp.COPT.INFEASIBLE: Status.INFEASIBLE,
            cp.COPT.UNBOUNDED: Status.UNBOUNDED,
            cp.COPT.INF_OR_UNB: Status.INF_OR_UNBD,
            cp.COPT.NUMERICAL: Status.NUMERICAL_ERROR,
            cp.COPT.NODELIMIT: Status.NODE_LIMIT,
            cp.COPT.TIMEOUT: Status.TIME_LIMIT,
            cp.COPT.ITERLIMIT: Status.ITERATION_LIMIT,
        }
        status_names = {
            value: name
            for name, value in vars(cp.COPT).items()
            if name
            in {
                "OPTIMAL",
                "INFEASIBLE",
                "UNBOUNDED",
                "INF_OR_UNB",
                "NUMERICAL",
                "NODELIMIT",
                "IMPRECISE",
                "TIMEOUT",
                "UNFINISHED",
                "INTERRUPTED",
                "ITERLIMIT",
            }
        }
        environment = None
        try:
            environment = _create_quiet_environment(cp)
            target = environment.createModel(model.name)
            target.setParam(cp.COPT.Param.Logging, 0)
            settings = {
                "TimeLimit": (cp.COPT.Param.TimeLimit, options.time_limit),
                "NodeLimit": (cp.COPT.Param.NodeLimit, options.node_limit),
                "RelGap": (cp.COPT.Param.RelGap, options.mip_gap),
                "FeasTol": (cp.COPT.Param.FeasTol, options.feasibility_tol),
                "IntTol": (cp.COPT.Param.IntTol, options.integrality_tol),
                "Threads": (cp.COPT.Param.Threads, options.threads),
                "Presolve": (cp.COPT.Param.Presolve, 1 if options.presolve else 0),
            }
            for _, (parameter, value) in settings.items():
                target.setParam(parameter, value)

            kinds = {
                "C": cp.COPT.CONTINUOUS,
                "I": cp.COPT.INTEGER,
                "B": cp.COPT.BINARY,
            }
            variables = []
            for variable in model.variables:
                lower = variable.lb if math.isfinite(variable.lb) else -cp.COPT.INFINITY
                upper = variable.ub if math.isfinite(variable.ub) else cp.COPT.INFINITY
                variables.append(
                    target.addVar(
                        lb=lower,
                        ub=upper,
                        vtype=kinds[variable.kind],
                        name=variable.name,
                    )
                )

            for row in model.constraints:
                expression = cp.quicksum(
                    coefficient * variables[index]
                    for index, coefficient in row.expression.terms.items()
                ) + row.expression.constant
                comparison = {
                    "<=": expression <= 0,
                    ">=": expression >= 0,
                    "==": expression == 0,
                }[row.sense]
                target.addConstr(comparison, name=row.name)

            objective = cp.quicksum(
                coefficient * variables[index]
                for index, coefficient in model.objective.terms.items()
            ) + model.objective.constant
            target.setObjective(
                objective,
                cp.COPT.MINIMIZE if model.sense == "min" else cp.COPT.MAXIMIZE,
            )
            target.solve()

            raw_status = int(target.getAttr(cp.COPT.Attr.Status))
            has_solution = bool(target.getAttr(cp.COPT.Attr.HasSol))
            values = (
                {variable.name: float(mapped.x) for variable, mapped in zip(model.variables, variables)}
                if has_solution
                else {}
            )
            reported_objective = float(target.getAttr(cp.COPT.Attr.ObjVal)) if has_solution else None
            raw_best_bound = float(target.getAttr(cp.COPT.Attr.ObjBound))
            best_bound = (
                None
                if raw_status
                in {cp.COPT.INFEASIBLE, cp.COPT.UNBOUNDED, cp.COPT.INF_OR_UNB}
                or not math.isfinite(raw_best_bound)
                or abs(raw_best_bound) >= float(cp.COPT.INFINITY)
                else raw_best_bound
            )
            raw_gap = float(target.getAttr(cp.COPT.Attr.BestGap)) if has_solution else None
            applied = {
                name: target.getParam(parameter)
                for name, (parameter, _) in settings.items()
            }
            return make_result(
                model,
                options,
                info,
                status=status_map.get(raw_status, Status.UNKNOWN),
                has_solution=has_solution,
                values=values,
                reported_objective=reported_objective,
                best_bound=best_bound,
                raw_gap=raw_gap,
                runtime=float(target.getAttr(cp.COPT.Attr.SolvingTime)),
                node_count=int(target.getAttr(cp.COPT.Attr.NodeCnt)),
                iteration_count=int(target.getAttr(cp.COPT.Attr.SimplexIter)),
                termination_reason=f"COPT {status_names.get(raw_status, raw_status)}",
                metadata={
                    "raw_status": raw_status,
                    "raw_best_bound": (
                        raw_best_bound
                        if math.isfinite(raw_best_bound)
                        else None
                    ),
                    "best_bound_sentinel_dropped": best_bound is None
                    and raw_best_bound is not None,
                    "parameters_applied": applied,
                    "parameters_unmapped": [
                        "iteration_limit",
                        "objective_tol",
                        "pivot_tol",
                        "max_tableau_cells",
                        "seed",
                    ],
                    "license_message": info.message,
                },
            )
        except cp.CoptError as exc:
            raise SolverExecutionError(f"COPT execution failed: {exc}") from exc
        finally:
            if environment is not None:
                environment.close()


BACKEND = CoptBackend()
