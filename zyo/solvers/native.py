# 自主密集内核适配器：调用既有单纯形与分支定界，不自动换外部引擎。
"""Discovery metadata for the self-developed ZYO backend."""

from dataclasses import asdict

from zyo._version import __version__
from zyo.capabilities import LP, MILP

from .base import BackendInfo
from ._legacy import convert_legacy_result


class NativeBackend:
    def info(self):
        return BackendInfo(
            name="native",
            version=__version__,
            available=True,
            capabilities=frozenset({LP, MILP}),
            license_mode="self-developed",
            message="NumPy-only ZYO dense LP/MILP research kernel",
        )

    def solve(self, model, options):
        from lzyopt.solver import solve

        info = self.info()
        raw = solve(model, engine="native", options=options.to_legacy())
        requested = asdict(options)
        return convert_legacy_result(
            model,
            options,
            raw,
            info,
            backend_kind="self-developed",
            parameters_applied={
                name: value
                for name, value in requested.items()
                if name not in {"threads", "seed", "presolve"}
            },
            parameters_unmapped=["threads", "seed", "presolve"],
        )


BACKEND = NativeBackend()
