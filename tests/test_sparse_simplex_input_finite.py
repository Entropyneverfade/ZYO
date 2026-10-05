"""原生稀疏内核的输入域门不能把 NaN/Inf 晋升为最优或无界证书。"""

import math
import unittest

from scipy.sparse import csc_matrix

from zyo.sparse_simplex import revised_simplex


class SparseSimplexFiniteInputTests(unittest.TestCase):
    def setUp(self):
        self.matrix = csc_matrix([[1.0, 0.0]])
        self.kwargs = dict(basic=[0], initial={1: 0.0}, iteration_limit=1)

    def test_rejects_nonfinite_costs_before_optimality_check(self):
        # NaN 的比较为假；若不先拦截，零步就会假称 OPTIMAL。
        for wrong in (math.nan, math.inf, -math.inf):
            with self.subTest(wrong=wrong):
                with self.assertRaisesRegex(ValueError, 'finite.*cost'):
                    revised_simplex(self.matrix, [0.0, wrong], [0.0, 0.0],
                                    [0.0, 1.0], **self.kwargs)

    def test_rejects_nonfinite_objective_offset(self):
        for wrong in (math.nan, math.inf, -math.inf):
            with self.subTest(wrong=wrong):
                with self.assertRaisesRegex(ValueError, 'finite.*objective'):
                    revised_simplex(self.matrix, [0.0, -1.0], [0.0, 0.0],
                                    [0.0, 1.0], objective_offset=wrong, **self.kwargs)

    def test_rejects_nan_and_wrong_side_infinite_bounds(self):
        bad = (([math.nan, 0.0], [0.0, 1.0]),
               ([0.0, 0.0], [0.0, math.nan]),
               ([math.inf, 0.0], [math.inf, 1.0]),
               ([0.0, 0.0], [0.0, -math.inf]))
        for lower, upper in bad:
            with self.subTest(lower=lower, upper=upper):
                with self.assertRaisesRegex(ValueError, 'invalid.*bound'):
                    revised_simplex(self.matrix, [0.0, -1.0], lower, upper,
                                    **self.kwargs)

    def test_rejects_nonfinite_matrix_coefficients(self):
        for wrong in (math.nan, math.inf, -math.inf):
            with self.subTest(wrong=wrong):
                with self.assertRaisesRegex(ValueError, 'finite.*matrix'):
                    revised_simplex(csc_matrix([[1.0, wrong]]), [0.0, -1.0],
                                    [0.0, 0.0], [0.0, 1.0], **self.kwargs)

    def test_rejects_nonfinite_or_negative_solve_tolerances(self):
        # 无限/NaN 定价阈值会使一列改善方向完全消失，假报 OPTIMAL。
        for name in ('pricing_tolerance', 'feasibility_tolerance'):
            for wrong in (math.nan, math.inf, -math.inf, -1.0):
                with self.subTest(name=name, wrong=wrong):
                    with self.assertRaisesRegex(ValueError, 'finite.*tolerance'):
                        revised_simplex(self.matrix, [0.0, -1.0], [0.0, 0.0],
                                        [0.0, 1.0], **self.kwargs, **{name: wrong})


if __name__ == '__main__':
    unittest.main()
