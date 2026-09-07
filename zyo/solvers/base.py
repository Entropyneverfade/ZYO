# 后端协议和元数据结构；区分自主与外部比较能力。
"""Common contracts for explicit ZYO solver backends."""

from dataclasses import dataclass
from typing import FrozenSet, Protocol


@dataclass(frozen=True)
class BackendInfo:
    name: str
    version: str | None
    available: bool
    capabilities: FrozenSet[str]
    license_mode: str
    message: str = ""


class Backend(Protocol):
    def info(self) -> BackendInfo:
        ...

    def solve(self, model, options):
        ...
