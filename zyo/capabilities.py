# 能力名称枚举供接口声明使用；有名称不等于实现或验收完成。
"""Backend-independent solver capability names."""

LP = "LP"
MILP = "MILP"
MPS_READ = "MPS_READ"
MPS_WRITE = "MPS_WRITE"
WARM_START = "WARM_START"

__all__ = ["LP", "MILP", "MPS_READ", "MPS_WRITE", "WARM_START"]
