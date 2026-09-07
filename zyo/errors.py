# 公共异常类别：区分建模、参数、依赖、数值及无可用解等失败原因。
"""Public ZYO exception hierarchy."""


class ZYOError(Exception):
    """Base class for user-facing ZYO errors."""


class ModelingError(ZYOError):
    """The mathematical model is invalid."""


class InvalidParameterError(ZYOError):
    """A solver option is unknown, duplicated, or out of range."""


class UnsupportedFeatureError(ZYOError):
    """The selected solver cannot represent a requested feature."""


class SolverUnavailableError(ZYOError):
    """A requested solver, binary, or license is unavailable."""


class SolverExecutionError(ZYOError):
    """A solver failed before producing a trustworthy result."""


class NumericalError(ZYOError):
    """A candidate or certificate failed numerical validation."""


class SolutionUnavailableError(ZYOError):
    """A solution-only attribute was requested without a solution."""
