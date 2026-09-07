# 科研建模外观接口：复用既有线性模型，提供直观别名与显式求解配置。
"""Public ZYO modeling facade."""

from lzyopt.model import Model as LegacyModel

from .constants import CONTINUOUS, normalize_sense, normalize_vtype
from .vardict import VarDict


class Model(LegacyModel):
    """Linear/MILP model with concise and compatibility-oriented helpers."""

    def add_var(self, name=None, lb=0.0, ub=None, vtype=CONTINUOUS, kind=None):
        resolved = normalize_vtype(kind if kind is not None else vtype)
        return super().add_var(name=name, lb=lb, ub=ub, kind=resolved)

    def add_vars(self, keys, *, name="x", **kwargs):
        result = VarDict()
        for key in keys:
            if key in result:
                raise ValueError(f"Duplicate variable key: {key!r}")
            result[key] = self.add_var(name=f"{name}[{key}]", **kwargs)
        return result

    def add_constrs(self, constraints, *, name="c"):
        return [
            self.add_constr(row, name=f"{name}[{index}]")
            for index, row in enumerate(constraints)
        ]

    def set_objective(self, expression, sense="min"):
        return super().set_objective(expression, normalize_sense(sense))

    def minimize(self, expression):
        self.set_objective(expression, "min")

    def maximize(self, expression):
        self.set_objective(expression, "max")

    def solve(self, solver="native", options=None, **kwargs):
        from .errors import SolverUnavailableError
        from .options import SolveOptions
        from .solvers import get_backend

        if options is not None and kwargs:
            raise ValueError("Pass either options or keyword parameters, not both")
        public_options = options or SolveOptions.from_kwargs(**kwargs)
        if not isinstance(public_options, SolveOptions):
            raise TypeError("options must be a zyo.SolveOptions instance")
        backend = get_backend(solver)
        info = backend.info()
        if not info.available:
            raise SolverUnavailableError(f"{info.name} unavailable: {info.message}")
        self._last_result = backend.solve(self, public_options)
        return self._last_result

    @property
    def obj_value(self):
        result = getattr(self, "_last_result", None)
        return result.objective if result is not None else None

    @property
    def status(self):
        result = getattr(self, "_last_result", None)
        return result.status if result is not None else None

    @property
    def best_bound(self):
        result = getattr(self, "_last_result", None)
        return result.best_bound if result is not None else None

    @property
    def mip_gap(self):
        result = getattr(self, "_last_result", None)
        return result.mip_gap if result is not None else None

    addVar = add_var
    addVars = add_vars
    addConstr = LegacyModel.add_constr
    addConstrs = add_constrs
    setObjective = set_objective
    optimize = solve
