# 公共 Python 接口汇总；导入时不提前加载第三方优化引擎。
"""ZYO: an inspectable general-purpose mathematical optimizer."""

from ._version import __version__
from .constants import BINARY, CONTINUOUS, INTEGER, MAXIMIZE, MINIMIZE
from .status import Status
from .model import Model
from .options import SolveOptions
from .result import Result
from .vardict import VarDict
from .validation import Residuals, evaluate_objective, validate_candidate
from .solvers import BackendInfo, available_solvers, get_backend
from .errors import (
    InvalidParameterError,
    ModelingError,
    NumericalError,
    SolutionUnavailableError,
    SolverExecutionError,
    SolverUnavailableError,
    UnsupportedFeatureError,
    ZYOError,
)
from lzyopt import LinearExpression, Variable as Var, quicksum

Options = SolveOptions

__all__ = [
    "BINARY",
    "BackendInfo",
    "CONTINUOUS",
    "INTEGER",
    "MAXIMIZE",
    "MINIMIZE",
    "InvalidParameterError",
    "LinearExpression",
    "Model",
    "ModelingError",
    "NumericalError",
    "Options",
    "Result",
    "Residuals",
    "SolveOptions",
    "SolutionUnavailableError",
    "SolverExecutionError",
    "SolverUnavailableError",
    "Status",
    "UnsupportedFeatureError",
    "Var",
    "VarDict",
    "ZYOError",
    "available_solvers",
    "evaluate_objective",
    "get_backend",
    "quicksum",
    "validate_candidate",
]
