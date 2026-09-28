# 建模与输入标准化边界；这里只转换模型数据，不选择或调用求解引擎。
"""Solver-independent modeling data transformations."""

from .matrix import model_arrays, model_sparse_arrays

__all__ = ['model_arrays', 'model_sparse_arrays']
