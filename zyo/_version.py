# 导出统一版本号，避免公共包与兼容层各自维护不同版本。
"""Public version export."""

from _zyo_version import __version__

__all__ = ["__version__"]
