# 大小写兼容的惰性注册表；未知名称明确报错，不猜测或回退引擎。
"""Case-insensitive lazy registry for ZYO solver backends."""

from importlib import import_module

from zyo.errors import SolverUnavailableError

from .base import BackendInfo


_MODULES = {
    "native": "zyo.solvers.native",
    "native_sparse": "zyo.solvers.native_sparse",
    "highs": "zyo.solvers.highs",
    "gurobi": "zyo.solvers.gurobi",
    "copt": "zyo.solvers.copt",
}
_CACHE = {}


def normalize_solver_name(name):
    return str(name).strip().casefold()


def get_backend(name):
    key = normalize_solver_name(name)
    if key not in _MODULES:
        raise SolverUnavailableError(f"Unknown solver: {name}")
    if key not in _CACHE:
        try:
            _CACHE[key] = import_module(_MODULES[key]).BACKEND
        except ModuleNotFoundError as exc:
            if exc.name == _MODULES[key]:
                raise SolverUnavailableError(
                    f"Solver backend is not implemented: {key}"
                ) from exc
            raise
    return _CACHE[key]


def available_solvers():
    discovered = {}
    for name in _MODULES:
        try:
            discovered[name] = get_backend(name).info()
        except SolverUnavailableError as exc:
            discovered[name] = BackendInfo(
                name=name,
                version=None,
                available=False,
                capabilities=frozenset(),
                license_mode="unavailable",
                message=str(exc),
            )
    return discovered
