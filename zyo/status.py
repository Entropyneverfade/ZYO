# 统一停止状态；时间、节点和数值限制不能升级为已证明最优。
"""Solver-independent ZYO termination states."""

from enum import Enum


class Status(str, Enum):
    OPTIMAL = "OPTIMAL"
    INFEASIBLE = "INFEASIBLE"
    UNBOUNDED = "UNBOUNDED"
    INF_OR_UNBD = "INF_OR_UNBD"
    RELAXATION_UNBOUNDED = "RELAXATION_UNBOUNDED"
    TIME_LIMIT = "TIME_LIMIT"
    NODE_LIMIT = "NODE_LIMIT"
    ITERATION_LIMIT = "ITERATION_LIMIT"
    SIZE_LIMIT = "SIZE_LIMIT"
    GAP_LIMIT = "GAP_LIMIT"
    LIMIT_REACHED = "LIMIT_REACHED"
    NUMERICAL_ERROR = "NUMERICAL_ERROR"
    SOLVER_ERROR = "SOLVER_ERROR"
    UNKNOWN = "UNKNOWN"
