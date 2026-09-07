# 线性表达式、变量、约束与严格 JSON 模型表示；约束统一存为表达式与零比较。
"""Sparse linear modeling and a portable, strict JSON representation."""
from __future__ import annotations
from dataclasses import dataclass
from numbers import Real
import json
import math
from pathlib import Path


def finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("Coefficients and constants must be finite")
    return value


class LinearExpression:
    def __init__(self, model=None, terms=None, constant=0.0):
        self.model = model
        self.terms = {int(k): finite(v) for k, v in (terms or {}).items() if v != 0}
        self.constant = finite(constant)

    @staticmethod
    def cast(value):
        if isinstance(value, LinearExpression):
            return value
        if isinstance(value, Real):
            return LinearExpression(constant=value)
        raise TypeError("Only linear expressions and real numbers are supported")

    def __add__(self, other):
        other = self.cast(other)
        if self.model is not None and other.model is not None and self.model is not other.model:
            raise ValueError("Cannot mix variables from different models")
        terms = self.terms.copy()
        for k, v in other.terms.items():
            terms[k] = terms.get(k, 0.0) + v
        return LinearExpression(self.model if self.model is not None else other.model,
                                terms, self.constant + other.constant)

    __radd__ = __add__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + -self.cast(other)

    def __rsub__(self, other):
        return self.cast(other) + -self

    def __mul__(self, scalar):
        if not isinstance(scalar, Real):
            raise TypeError("Variable products/nonlinear expressions are not supported")
        scalar = finite(scalar)
        return LinearExpression(self.model, {k: v * scalar for k, v in self.terms.items()},
                                self.constant * scalar)

    __rmul__ = __mul__

    def __truediv__(self, scalar):
        return self * (1.0 / finite(scalar))

    def __le__(self, other):
        return Constraint(self - other, "<=")

    def __ge__(self, other):
        return Constraint(self - other, ">=")

    def __eq__(self, other):
        return Constraint(self - other, "==")

    def __bool__(self):
        raise TypeError("A symbolic expression has no truth value")


class Variable(LinearExpression):
    def __init__(self, model, index, name, lb, ub, kind):
        super().__init__(model, {index: 1.0})
        self.index, self.name = index, name
        self.lb, self.ub, self.kind = lb, ub, kind

    @property
    def x(self):
        result = getattr(self.model, "_last_result", None)
        if result is None or self.name not in result.values:
            from zyo.errors import SolutionUnavailableError

            raise SolutionUnavailableError(
                f"No solution value is available for {self.name!r}"
            )
        return result.values[self.name]


@dataclass
class Constraint:
    expression: LinearExpression
    sense: str
    name: str = ""

    def __bool__(self):
        raise TypeError("Chained comparisons are unsupported; add each constraint separately")


def quicksum(values):
    # Linear-time assembly; repeated expression addition can be quadratic.
    owner, terms, constant = None, {}, 0.0
    for value in values:
        expr = LinearExpression.cast(value)
        if owner is not None and expr.model is not None and owner is not expr.model:
            raise ValueError("Cannot mix variables from different models")
        if expr.model is not None:
            owner = expr.model
        constant += expr.constant
        for i, v in expr.terms.items():
            terms[i] = terms.get(i, 0.0) + v
    return LinearExpression(owner, terms, constant)


class Model:
    def __init__(self, name="model"):
        self.name = str(name)
        self.variables = []
        self.constraints = []
        self.objective = LinearExpression()
        self.sense = "min"

    def add_var(self, name=None, lb=0.0, ub=None, kind="C"):
        """kind C/I/B; None lower/upper bounds mean -infinity/+infinity."""
        if kind not in ("C", "I", "B"):
            raise ValueError("kind must be C, I or B")
        lb = -math.inf if lb is None else float(lb)
        ub = math.inf if ub is None else float(ub)
        if math.isnan(lb) or math.isnan(ub) or lb == math.inf or ub == -math.inf:
            raise ValueError("Invalid variable bounds")
        if kind == "B":
            lb, ub = max(0.0, lb), min(1.0, ub)
        name = str(name if name is not None else f"x{len(self.variables)}")
        if not name or name in {v.name for v in self.variables}:
            raise ValueError("Variable names must be nonempty and unique")
        variable = Variable(self, len(self.variables), name, lb, ub, kind)
        self.variables.append(variable)
        return variable

    def _check(self, expression):
        if expression.model is not None and expression.model is not self:
            raise ValueError("Expression belongs to another model")
        if any(k < 0 or k >= len(self.variables) for k in expression.terms):
            raise ValueError("Invalid variable index")

    def add_constr(self, constraint, name=None):
        if not isinstance(constraint, Constraint) or constraint.sense not in ("<=", ">=", "=="):
            raise TypeError("Expected a linear comparison, e.g. x + y <= 3")
        self._check(constraint.expression)
        row = Constraint(LinearExpression(self, constraint.expression.terms,
                                         constraint.expression.constant), constraint.sense,
                         str(name if name is not None else f"c{len(self.constraints)}"))
        self.constraints.append(row)
        return row

    def set_objective(self, expression, sense="min"):
        if sense not in ("min", "max"):
            raise ValueError("sense must be min or max")
        expression = LinearExpression.cast(expression)
        self._check(expression)
        self.objective = LinearExpression(self, expression.terms, expression.constant)
        self.sense = sense

    def optimize(self, engine="native", options=None):
        from .solver import solve
        return solve(self, engine=engine, options=options)

    def to_dict(self):
        def expression(e):
            return {"terms": {self.variables[k].name: v for k, v in e.terms.items()},
                    "constant": e.constant}
        return {"format": "lzyopt-1", "name": self.name, "sense": self.sense,
                "variables": [{"name": v.name, "lb": v.lb if math.isfinite(v.lb) else None,
                               "ub": v.ub if math.isfinite(v.ub) else None, "kind": v.kind}
                              for v in self.variables],
                "objective": expression(self.objective),
                "constraints": [{"name": r.name, "sense": r.sense,
                                 **expression(r.expression)} for r in self.constraints]}

    @classmethod
    def from_dict(cls, data):
        if data.get("format") != "lzyopt-1":
            raise ValueError("Unsupported model format; expected lzyopt-1")
        model = cls(data.get("name", "model"))
        variables = {v["name"]: model.add_var(**v) for v in data["variables"]}

        def expression(d):
            return quicksum(finite(a) * variables[n] for n, a in d.get("terms", {}).items()) + finite(d.get("constant", 0))

        model.set_objective(expression(data["objective"]), data["sense"])
        for row in data["constraints"]:
            model.add_constr(Constraint(expression(row), row["sense"]), row["name"])
        return model

    def write(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def read(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8-sig")))
