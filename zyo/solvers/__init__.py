# 显式后端注册入口；不因导入注册表而加载外部求解器。
"""Explicit solver registry without eager third-party imports."""

from .base import Backend, BackendInfo
from .registry import available_solvers, get_backend, normalize_solver_name

__all__ = [
    "Backend",
    "BackendInfo",
    "available_solvers",
    "get_backend",
    "normalize_solver_name",
]
