"""非规范稀疏输入中的重复项必须在自研求解和原模型证书中同义。"""

import unittest

import numpy as np
from scipy.sparse import csc_matrix

from zyo.native_lp import build_homogeneous, build_model, solve_certified_lp


class NativeLPDuplicateEntriesTests(unittest.TestCase):
    def test_duplicate_row_entry_cannot_falsely_certify_warm_start(self):
        # 真实聚合 binary64 行为 (-2e-10-1e-10)x+y<=1；x=1e12,y=300
        # 严格可行且目标约 100。旧证书字典仅保留末项 -1e-10，误报 (0,1) 最优。
        for shape in ('csc', 'csr'):
            with self.subTest(shape=shape):
                matrix = csc_matrix(
                    (np.array([-2e-10, -1e-10, 1.0]),
                     np.array([0, 0, 0]), np.array([0, 2, 3])), shape=(1, 2))
                if shape == 'csr':
                    matrix = matrix.tocsr()
                original = (matrix.data.copy(), matrix.indices.copy(), matrix.indptr.copy())
                result = solve_certified_lp(
                    matrix, [-2e-10, 1.0], [0.0, 0.0], [1e12, np.inf], [1.0],
                    sense=['<='], maximize=True, warm_basis=[1],
                    warm_nonbasic_at_upper=[2], require_warm_start=True,
                    iteration_limit=20)
                if result.optimal:
                    # 仅当取得不低于独立可行见证的真实最优目标，才容许声称最优。
                    self.assertGreaterEqual(result.objective, 99.0,
                                            (result.status, result.record.get('solver_status'), result.record))
                self.assertTrue(np.array_equal(matrix.data, original[0]))
                self.assertTrue(np.array_equal(matrix.indices, original[1]))
                self.assertTrue(np.array_equal(matrix.indptr, original[2]))

    def test_direct_model_and_homogeneous_entries_sum_duplicates_without_mutation(self):
        # 证书建模与齐次编码两个可独立调用的入口也必须使用同一个 3.0。
        for shape in ('csc', 'csr'):
            with self.subTest(shape=shape):
                matrix = csc_matrix(
                    (np.array([1.0, 2.0]), np.array([0, 0]),
                     np.array([0, 2])), shape=(1, 1))
                if shape == 'csr':
                    matrix = matrix.tocsr()
                original = (matrix.data.copy(), matrix.indices.copy(), matrix.indptr.copy())
                model = build_model(matrix, [1.0], [0.0], [10.0], [3.0], sense=['>='])
                wide = build_homogeneous(matrix, [0.0], [10.0], [3.0], sense=['>='])[0]
                self.assertEqual(model.constraints[0].expression.terms[0], 3.0)
                self.assertEqual(wide[0, 0], 3.0)
                self.assertTrue(np.array_equal(matrix.data, original[0]))
                self.assertTrue(np.array_equal(matrix.indices, original[1]))
                self.assertTrue(np.array_equal(matrix.indptr, original[2]))

    def test_nonfinite_sum_of_finite_duplicates_is_rejected_before_solve(self):
        matrix = csc_matrix(
            (np.array([1e308, 1e308]), np.array([0, 0]),
             np.array([0, 2])), shape=(1, 1))
        original = (matrix.data.copy(), matrix.indices.copy(), matrix.indptr.copy())
        with self.assertRaisesRegex(ValueError, 'finite'):
            solve_certified_lp(matrix, [1.0], [0.0], [1.0], [1.0], sense=['=='])
        with self.assertRaisesRegex(ValueError, 'finite'):
            build_model(matrix, [1.0], [0.0], [1.0], [1.0], sense=['=='])
        with self.assertRaisesRegex(ValueError, 'finite'):
            build_homogeneous(matrix, [0.0], [1.0], [1.0], sense=['=='])
        self.assertTrue(np.array_equal(matrix.data, original[0]))
        self.assertTrue(np.array_equal(matrix.indices, original[1]))
        self.assertTrue(np.array_equal(matrix.indptr, original[2]))


if __name__ == '__main__':
    unittest.main()
