# 变量类型和目标方向的统一常量；兼容大小写与约定别名，非法输入明确拒绝。
"""Canonical ZYO modeling constants and input normalization."""

CONTINUOUS = "C"
INTEGER = "I"
BINARY = "B"

MINIMIZE = "min"
MAXIMIZE = "max"

_VTYPES = {
    "c": CONTINUOUS,
    "continuous": CONTINUOUS,
    "i": INTEGER,
    "int": INTEGER,
    "integer": INTEGER,
    "b": BINARY,
    "bin": BINARY,
    "binary": BINARY,
}

_SENSES = {
    "min": MINIMIZE,
    "minimum": MINIMIZE,
    "minimize": MINIMIZE,
    "max": MAXIMIZE,
    "maximum": MAXIMIZE,
    "maximize": MAXIMIZE,
}


def _token(value):
    if not isinstance(value, str):
        raise ValueError("Expected a string token")
    return value.strip().casefold()


def normalize_vtype(value):
    try:
        return _VTYPES[_token(value)]
    except KeyError as exc:
        raise ValueError(f"Unsupported variable type: {value!r}") from exc


def normalize_sense(value):
    try:
        return _SENSES[_token(value)]
    except KeyError as exc:
        raise ValueError(f"Unsupported objective sense: {value!r}") from exc
