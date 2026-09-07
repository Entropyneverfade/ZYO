# 旧名称兼容层：保留历史导入语法，实际发行身份统一为 ZYO。
"""Legacy LZYOpt compatibility API backed by the ZYO distribution."""
from _zyo_version import __version__
from .model import Model, Variable, LinearExpression, quicksum


def __getattr__(name):
    # Basic modeling must not load the old mixed-engine compatibility module.
    # Its legacy names remain available when explicitly requested by old code.
    if name in ('Options', 'Result', 'solve'):
        from importlib import import_module
        value=getattr(import_module('.solver',__name__),name)
        globals()[name]=value
        return value
    raise AttributeError(name)

__all__ = ["Model", "Variable", "LinearExpression", "quicksum", "Options", "Result", "solve"]
