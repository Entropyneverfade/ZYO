# 带索引的变量容器，支持批量建模；不改变变量本身的数学含义。
"""Containers for indexed ZYO variables."""


class VarDict(dict):
    """Insertion-ordered indexed variables with explicit selection helpers."""

    def select(self, predicate=None):
        if predicate is None:
            return list(self.values())
        return [value for key, value in self.items() if predicate(key)]
