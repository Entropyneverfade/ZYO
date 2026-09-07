# 大小写兼容入口：复用 zyo 公共接口，不维护第二份求解实现。
"""Case-compatible import shim for ``import ZYO``."""

from zyo import *
from zyo import __version__
