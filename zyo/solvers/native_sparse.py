# 实验性自主稀疏内核适配器；SciPy 仅提供基础稀疏线性代数，不调用其优化器。
"""Explicit optional self-developed sparse kernel (not an external engine)."""
from importlib.util import find_spec
from zyo._version import __version__
from zyo.capabilities import LP, MILP
from .base import BackendInfo


class NativeSparseBackend:
    def info(self):
        return BackendInfo(name='native_sparse',version=__version__,available=find_spec('scipy') is not None,
                           capabilities=frozenset({LP,MILP}),license_mode='self-developed',
                           message='Experimental ZYO finite-box LP/MILP; SciPy sparse linear algebra only, no optimizer')

    def solve(self,model,options):
        from zyo.sparse_mip import solve
        return solve(model,options)


BACKEND=NativeSparseBackend()
