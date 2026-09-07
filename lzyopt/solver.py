# 历史兼容实现：自主分支定界和显式 HiGHS 比较入口共存；自主分支不能调用外部入口。
"""Status-aware branch-and-bound, HiGHS adapter and primal checks."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import heapq
import math
import time
import warnings
import numpy as np
from .native import relaxation


@dataclass
class Options:
    time_limit: float = 60.0
    node_limit: int = 10000
    iteration_limit: int = 100000  # per native LP relaxation
    mip_rel_gap: float = 0.0
    feasibility_tol: float = 1e-7
    integrality_tol: float = 1e-7
    pivot_tol: float = 1e-10
    objective_tol: float = 1e-8
    max_tableau_cells: int = 2000000
    presolve: bool = True
    threads: int = 1
    seed: int = 0

    def validate(self):
        for name in ("time_limit", "mip_rel_gap", "objective_tol"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in ("feasibility_tol", "integrality_tol", "pivot_tol"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.integrality_tol >= 0.5:
            raise ValueError("integrality_tol must be smaller than 0.5")
        for name in ("node_limit", "iteration_limit", "max_tableau_cells"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not isinstance(self.presolve, bool):
            raise ValueError("presolve must be a boolean")
        if isinstance(self.threads, bool) or not isinstance(self.threads, int) or self.threads < 1:
            raise ValueError("threads must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")


@dataclass
class Result:
    status: str
    engine: str
    objective: float | None = None
    best_bound: float | None = None
    gap: float | None = None
    values: dict = field(default_factory=dict)
    nodes: int = 0
    iterations: int = 0
    runtime_seconds: float = 0.0
    residuals: dict = field(default_factory=dict)
    message: str = ""
    options: dict = field(default_factory=dict)

    @property
    def has_solution(self):
        return self.objective is not None

    def to_dict(self):
        def safe(value):
            if isinstance(value, dict):
                return {k: safe(v) for k, v in value.items()}
            if isinstance(value, float) and not math.isfinite(value):
                return None
            return value
        return safe(asdict(self))


def check_solution(model, x):
    # 从原模型重算行、边界及整数残差；求解器状态不能替代此项检查。
    x = np.asarray(x, dtype=float)
    if x.shape != (len(model.variables),) or not np.all(np.isfinite(x)):
        return {"constraint_violation": math.inf, "bound_violation": math.inf,
                "integrality_violation": math.inf}
    row_violation, bound_violation, integer_violation = 0.0, 0.0, 0.0
    for row in model.constraints:
        activity = row.expression.constant + sum(a * x[i] for i, a in row.expression.terms.items())
        violation = abs(activity) if row.sense == "==" else max(0, activity if row.sense == "<=" else -activity)
        row_violation = max(row_violation, float(violation))
    for i, v in enumerate(model.variables):
        bound_violation = max(bound_violation, v.lb - x[i], x[i] - v.ub)
        if v.kind != "C":
            integer_violation = max(integer_violation, abs(x[i] - np.rint(x[i])))
    return {"constraint_violation": float(row_violation), "bound_violation": float(bound_violation),
            "integrality_violation": float(integer_violation)}


def _feasible(residuals, options, integer=True):
    return (max(residuals["constraint_violation"], residuals["bound_violation"]) <= options.feasibility_tol
            and (not integer or residuals["integrality_violation"] <= options.integrality_tol))


def _objective(model, x):
    return float(model.objective.constant + sum(a * x[i] for i, a in model.objective.terms.items()))


def _gap(incumbent, bound):
    if not math.isfinite(incumbent) or not math.isfinite(bound):
        return None
    return max(0.0, incumbent - bound) / max(1.0, abs(incumbent))


def _native(model, options, start, lower, upper):
    # 内部统一最小化：incumbent 是已验可行解目标，队列保存尚未关闭节点及其继承界。
    direction = 1 if model.sense == "min" else -1
    integers = np.array([i for i, v in enumerate(model.variables) if v.kind != "C"], dtype=int)
    deadline = start + options.time_limit
    result = Result("UNKNOWN", "native")
    if not len(integers):
        lp = relaxation(model, lower, upper, options, deadline)
        result.status, result.message, result.iterations = lp.status, lp.message, lp.iterations
        if lp.x is not None:
            result.residuals = check_solution(model, lp.x)
            if not _feasible(result.residuals, options):
                result.status, result.message = "NUMERICAL_ERROR", "LP failed original-unit primal feasibility check"
            else:
                result.objective = direction * lp.objective
                result.best_bound, result.gap = result.objective, 0.0
                result.values = {v.name: float(lp.x[i]) for i, v in enumerate(model.variables)}
        return result
    # 仅整数变量的下界上取整、上界下取整，不改变其合法整数集合。
    lower[integers] = np.ceil(lower[integers])
    upper[integers] = np.floor(upper[integers])
    if np.any(lower > upper):
        return Result("INFEASIBLE", "native", message="No integer value lies within a variable's bounds")
    queue = [(-math.inf, 0, lower.copy(), upper.copy())]
    serial = 1
    incumbent, best_x = math.inf, None
    reason = "OPTIMAL"
    message = "Search exhausted within objective/feasibility tolerances"
    while queue:
        if time.perf_counter() >= deadline:
            reason, message = "TIME_LIMIT", "Time limit reached; optimality not established"
            break
        if result.nodes >= options.node_limit:
            reason, message = "NODE_LIMIT", "Node limit reached; optimality not established"
            break
        bound, number, lo, hi = heapq.heappop(queue)
        if bound >= incumbent - options.objective_tol:
            continue
        lp = relaxation(model, lo, hi, options, deadline)
        result.nodes += 1
        result.iterations += lp.iterations
        if lp.status == "INFEASIBLE":
            continue
        if lp.status != "OPTIMAL":
            # LP 松弛无界不能推出 MILP 无界；把未决节点放回队列，保留真实未决状态。
            reason = "RELAXATION_UNBOUNDED" if lp.status == "UNBOUNDED" else lp.status
            message = ("LP relaxation unbounded; MILP infeasibility/unboundedness is unresolved"
                       if lp.status == "UNBOUNDED" else lp.message)
            heapq.heappush(queue, (bound, number, lo, hi))
            break
        residuals = check_solution(model, lp.x)
        node_bound_violation = max(float(np.max(lo - lp.x, initial=0)), float(np.max(lp.x - hi, initial=0)))
        if not _feasible(residuals, options, integer=False) or node_bound_violation > options.feasibility_tol:
            reason, message = "NUMERICAL_ERROR", "Node LP failed original-unit feasibility check"
            heapq.heappush(queue, (bound, number, lo, hi))
            break
        value = lp.objective
        if value >= incumbent - options.objective_tol:
            continue
        fractions = np.abs(lp.x[integers] - np.rint(lp.x[integers]))
        if np.max(fractions, initial=0) <= options.integrality_tol:
            # 近整数点取整后必须重新验算所有原始约束，不能直接当成整数可行解。
            candidate = lp.x.copy()
            candidate[integers] = np.rint(candidate[integers])
            checked = check_solution(model, candidate)
            if not _feasible(checked, options):
                reason, message = "NUMERICAL_ERROR", "Rounded integer candidate failed feasibility check"
                heapq.heappush(queue, (value, number, lo, hi))
                break
            incumbent, best_x = direction * _objective(model, candidate), candidate
        else:
            j = int(integers[int(np.argmax(fractions))])
            split = math.floor(lp.x[j])
            # 两个子节点继承父 LP 界；未求解的子节点也必须计入全局界。
            left_hi = hi.copy()
            left_hi[j] = min(left_hi[j], split)
            right_lo = lo.copy()
            right_lo[j] = max(right_lo[j], split + 1)
            if lo[j] <= left_hi[j]:
                heapq.heappush(queue, (value, serial, lo.copy(), left_hi))
                serial += 1
            if right_lo[j] <= hi[j]:
                heapq.heappush(queue, (value, serial, right_lo, hi.copy()))
                serial += 1
        if best_x is not None and queue:
            gap = _gap(incumbent, min(incumbent, queue[0][0]))
            if options.mip_rel_gap > 0 and gap is not None and gap <= options.mip_rel_gap:
                reason, message = "GAP_LIMIT", "Requested gap reached; incumbent and bound reported"
                break
    result.status = reason if queue or best_x is not None else "INFEASIBLE"
    result.message = message if result.status != "INFEASIBLE" else "All nodes proved infeasible"
    bound = min(incumbent, queue[0][0]) if queue else incumbent
    if math.isfinite(bound):
        result.best_bound = direction * bound
    if best_x is not None:
        result.objective = direction * incumbent
        result.gap = _gap(incumbent, bound)
        result.values = {v.name: float(best_x[i]) for i, v in enumerate(model.variables)}
        result.residuals = check_solution(model, best_x)
    return result


def _highs(model, options, lower, upper):
    # 仅用于历史显式比较；_native 不得调用此函数，不能将其结果回传帮助原生搜索。
    try:
        from scipy.optimize import milp, linprog, Bounds, LinearConstraint
        from scipy.sparse import coo_matrix
    except ImportError as exc:
        raise ImportError("HiGHS adapter requires SciPy: pip install 'lzyopt[research]' from this source package") from exc
    n, m = len(model.variables), len(model.constraints)
    row_ids, col_ids, data = [], [], []
    row_lo, row_hi = np.full(m, -np.inf), np.full(m, np.inf)
    for j, row in enumerate(model.constraints):
        for i, a in row.expression.terms.items():
            row_ids.append(j)
            col_ids.append(i)
            data.append(a)
        rhs = -row.expression.constant
        if row.sense in (">=", "=="):
            row_lo[j] = rhs
        if row.sense in ("<=", "=="):
            row_hi[j] = rhs
    matrix = coo_matrix((data, (row_ids, col_ids)), shape=(m, n)).tocsc()
    direction = 1 if model.sense == "min" else -1
    cost = np.zeros(n)
    for i, a in model.objective.terms.items():
        cost[i] = direction * a
    integrality = np.array([int(v.kind != "C") for v in model.variables])
    mip = bool(np.any(integrality))
    if mip:
        solver_options = {
            "time_limit": options.time_limit,
            "node_limit": options.node_limit,
            "mip_rel_gap": options.mip_rel_gap,
            "mip_feasibility_tolerance": max(
                1e-10,
                min(options.feasibility_tol, options.integrality_tol),
            ),
            "presolve": options.presolve,
            "threads": options.threads,
            "random_seed": options.seed,
        }
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Unrecognized options detected:.*",
                category=Warning,
            )
            output = milp(cost, integrality=integrality, bounds=Bounds(lower, upper),
                          constraints=LinearConstraint(matrix, row_lo, row_hi),
                          options=solver_options)
    else:
        from scipy.sparse import vstack
        eq = row_lo == row_hi
        up = np.isfinite(row_hi) & ~eq
        down = np.isfinite(row_lo) & ~eq
        aub = vstack([matrix[up], -matrix[down]], format="csc")
        bub = np.concatenate([row_hi[up], -row_lo[down]])
        solver_options = {
            "time_limit": options.time_limit,
            "maxiter": options.iteration_limit,
            "primal_feasibility_tolerance": max(1e-10, options.feasibility_tol),
            "presolve": options.presolve,
            "threads": options.threads,
            "random_seed": options.seed,
        }
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Unrecognized options detected:.*",
                category=Warning,
            )
            output = linprog(cost, A_ub=aub, b_ub=bub, A_eq=matrix[eq], b_eq=row_hi[eq],
                             bounds=list(zip(lower, upper)), method="highs",
                             options=solver_options)
    status = {0: "OPTIMAL", 1: "LIMIT_REACHED", 2: "INFEASIBLE", 3: "UNBOUNDED", 4: "SOLVER_ERROR"}[output.status]
    result = Result(status, "highs", message=str(output.message),
                    nodes=int(getattr(output, "mip_node_count", 0) or 0),
                    iterations=int(getattr(output, "nit", 0) or 0))
    dual = getattr(output, "mip_dual_bound", None)
    if dual is not None and math.isfinite(dual):
        result.best_bound = direction * float(dual) + model.objective.constant
    if output.x is not None and output.status in (0, 1):
        residuals = check_solution(model, output.x)
        result.residuals = residuals
        if not _feasible(residuals, options):
            result.status, result.message = "NUMERICAL_ERROR", "HiGHS candidate failed configured primal checks"
            return result
        result.objective = _objective(model, output.x)
        result.values = {v.name: float(output.x[i]) for i, v in enumerate(model.variables)}
        if not mip and status == "OPTIMAL":
            result.best_bound = result.objective
        if result.best_bound is not None:
            result.gap = _gap(direction * result.objective, direction * result.best_bound)
        if mip and status == "OPTIMAL" and result.gap is not None and result.gap > options.objective_tol:
            result.status = "GAP_LIMIT"
    return result


def solve(model, engine="native", options=None):
    options = options or Options()
    options.validate()
    if engine not in ("native", "highs"):
        raise ValueError("engine must be 'native' or 'highs'; no silent fallback")
    start = time.perf_counter()
    lower = np.array([v.lb for v in model.variables], dtype=float)
    upper = np.array([v.ub for v in model.variables], dtype=float)
    if np.any(lower > upper):
        result = Result("INFEASIBLE", engine, message="Contradictory bounds")
    elif not len(lower):
        residuals = check_solution(model, [])
        result = Result("OPTIMAL" if _feasible(residuals, options) else "INFEASIBLE", engine, residuals=residuals)
        if result.status == "OPTIMAL":
            result.objective = model.objective.constant
            result.best_bound, result.gap = result.objective, 0.0
    elif options.time_limit == 0:
        result = Result("TIME_LIMIT", engine, message="Zero time budget")
    else:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                result = (_native(model, options, start, lower, upper) if engine == "native"
                          else _highs(model, options, lower, upper))
            except FloatingPointError as exc:
                result = Result("NUMERICAL_ERROR", engine, message=str(exc))
    result.options = asdict(options)
    result.runtime_seconds = time.perf_counter() - start
    return result
