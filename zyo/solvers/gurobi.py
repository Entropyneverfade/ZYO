# 仅供隔离比较的 Gurobi 适配器；许可失败不等于求解算法能力失败。
"""License-aware optional Gurobi comparison backend."""

from functools import lru_cache
import math

from zyo.capabilities import LP, MILP
from zyo.errors import SolverExecutionError, SolverUnavailableError
from zyo.status import Status

from ._direct import make_result
from .base import BackendInfo


@lru_cache(maxsize=1)
def _info():
    try:
        import gurobipy as gp
    except ImportError as exc:
        return BackendInfo(
            "gurobi", None, False, frozenset({LP, MILP}), "unavailable", str(exc)
        )
    version = ".".join(str(part) for part in gp.gurobi.version())
    try:
        with gp.Env(params={"OutputFlag": 0}) as environment:
            with gp.Model(env=environment) as model:
                expiration = int(model.LicenseExpiration)
    except gp.GurobiError as exc:
        return BackendInfo(
            "gurobi",
            version,
            False,
            frozenset({LP, MILP}),
            "unavailable",
            f"Gurobi license initialization failed: {exc}",
        )
    return BackendInfo(
        "gurobi",
        version,
        True,
        frozenset({LP, MILP}),
        "restricted",
        (
            "Available license conservatively labeled restricted; "
            f"reported expiration={expiration}"
        ),
    )


class GurobiBackend:
    def info(self):
        return _info()

    def solve(self, model, options):
        info = self.info()
        if not info.available:
            raise SolverUnavailableError(f"gurobi unavailable: {info.message}")
        import gurobipy as gp

        status_map = {
            gp.GRB.OPTIMAL: Status.OPTIMAL,
            gp.GRB.INFEASIBLE: Status.INFEASIBLE,
            gp.GRB.UNBOUNDED: Status.UNBOUNDED,
            gp.GRB.INF_OR_UNBD: Status.INF_OR_UNBD,
            gp.GRB.TIME_LIMIT: Status.TIME_LIMIT,
            gp.GRB.NODE_LIMIT: Status.NODE_LIMIT,
            gp.GRB.ITERATION_LIMIT: Status.ITERATION_LIMIT,
            gp.GRB.NUMERIC: Status.NUMERICAL_ERROR,
        }
        status_names = {
            value: name
            for name, value in vars(gp.GRB).items()
            if name
            in {
                "OPTIMAL",
                "INFEASIBLE",
                "UNBOUNDED",
                "INF_OR_UNBD",
                "TIME_LIMIT",
                "NODE_LIMIT",
                "ITERATION_LIMIT",
                "NUMERIC",
                "SUBOPTIMAL",
                "INTERRUPTED",
                "WORK_LIMIT",
                "MEM_LIMIT",
            }
        }
        try:
            with gp.Env(params={"OutputFlag": 0}) as environment:
                with gp.Model(model.name, env=environment) as target:
                    target.Params.OutputFlag = 0
                    target.Params.TimeLimit = options.time_limit
                    target.Params.NodeLimit = options.node_limit
                    target.Params.IterationLimit = options.iteration_limit
                    target.Params.MIPGap = options.mip_gap
                    target.Params.FeasibilityTol = options.feasibility_tol
                    target.Params.IntFeasTol = options.integrality_tol
                    target.Params.OptimalityTol = options.objective_tol
                    target.Params.Threads = options.threads
                    target.Params.Seed = options.seed
                    target.Params.Presolve = -1 if options.presolve else 0

                    kinds = {
                        "C": gp.GRB.CONTINUOUS,
                        "I": gp.GRB.INTEGER,
                        "B": gp.GRB.BINARY,
                    }
                    variables = []
                    for variable in model.variables:
                        lower = variable.lb if math.isfinite(variable.lb) else -gp.GRB.INFINITY
                        upper = variable.ub if math.isfinite(variable.ub) else gp.GRB.INFINITY
                        variables.append(
                            target.addVar(
                                lb=lower,
                                ub=upper,
                                vtype=kinds[variable.kind],
                                name=variable.name,
                            )
                        )
                    target.update()

                    for row in model.constraints:
                        expression = gp.LinExpr(row.expression.constant)
                        expression.addTerms(
                            list(row.expression.terms.values()),
                            [variables[index] for index in row.expression.terms],
                        )
                        comparison = {
                            "<=": expression <= 0,
                            ">=": expression >= 0,
                            "==": expression == 0,
                        }[row.sense]
                        target.addConstr(comparison, name=row.name)

                    objective = gp.LinExpr(model.objective.constant)
                    objective.addTerms(
                        list(model.objective.terms.values()),
                        [variables[index] for index in model.objective.terms],
                    )
                    target.setObjective(
                        objective,
                        gp.GRB.MINIMIZE if model.sense == "min" else gp.GRB.MAXIMIZE,
                    )
                    target.optimize()

                    raw_status = int(target.Status)
                    has_solution = int(target.SolCount) > 0
                    values = (
                        {variable.name: float(mapped.X) for variable, mapped in zip(model.variables, variables)}
                        if has_solution
                        else {}
                    )
                    reported_objective = float(target.ObjVal) if has_solution else None
                    best_bound = (
                        float(target.ObjBound)
                        if raw_status
                        not in {gp.GRB.INFEASIBLE, gp.GRB.UNBOUNDED, gp.GRB.INF_OR_UNBD}
                        else None
                    )
                    raw_gap = (
                        float(target.MIPGap)
                        if has_solution and bool(target.IsMIP)
                        else (0.0 if raw_status == gp.GRB.OPTIMAL else None)
                    )
                    applied = {
                        "TimeLimit": float(target.Params.TimeLimit),
                        "NodeLimit": float(target.Params.NodeLimit),
                        "IterationLimit": float(target.Params.IterationLimit),
                        "MIPGap": float(target.Params.MIPGap),
                        "FeasibilityTol": float(target.Params.FeasibilityTol),
                        "IntFeasTol": float(target.Params.IntFeasTol),
                        "OptimalityTol": float(target.Params.OptimalityTol),
                        "Threads": int(target.Params.Threads),
                        "Seed": int(target.Params.Seed),
                        "Presolve": int(target.Params.Presolve),
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
                        runtime=float(target.Runtime),
                        node_count=int(target.NodeCount),
                        iteration_count=int(target.IterCount),
                        termination_reason=f"Gurobi {status_names.get(raw_status, raw_status)}",
                        metadata={
                            "raw_status": raw_status,
                            "parameters_applied": applied,
                            "parameters_unmapped": ["pivot_tol", "max_tableau_cells"],
                            "license_message": info.message,
                        },
                    )
        except gp.GurobiError as exc:
            raise SolverExecutionError(f"Gurobi execution failed: {exc}") from exc


BACKEND = GurobiBackend()
