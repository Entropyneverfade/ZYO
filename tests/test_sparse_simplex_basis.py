# 稀疏基维护的独立验收：FTRAN/BTRAN 与乘积形式更新必须与稠密直接解一致。
# 预期全部由 numpy 稠密解或手算给出，不调用被测模块的分解/更新函数。
import unittest
from unittest.mock import patch

import numpy as np
from scipy.linalg.blas import daxpy as reference_daxpy
from scipy.sparse import csc_matrix

from zyo.errors import NumericalError
from zyo.sparse_simplex import SparseBasis, extract_basis


def dense_matrix(rows):
    return csc_matrix(np.array(rows, dtype=float))


# 一个 4x6 的稀疏矩阵：前 4 列构成单位阵（对应行界逻辑列），后 2 列是结构列。
A_ROWS = [
    [1.0, 0.0, 0.0, 0.0, 2.0, -1.0],
    [0.0, 1.0, 0.0, 0.0, 1.0, 3.0],
    [0.0, 0.0, 1.0, 0.0, -1.0, 2.0],
    [0.0, 0.0, 0.0, 1.0, 4.0, 1.0],
]


class SparseBasisTests(unittest.TestCase):
    def setUp(self):
        self.matrix = dense_matrix(A_ROWS)
        self.basic = [0, 1, 2, 3]
        self.basis = SparseBasis(self.matrix, self.basic)

    # ---------- 基本分解与残差 ----------

    def test_identity_basis_solves_exactly(self):
        rhs = np.array([3.0, -2.0, 5.0, 1.0])
        np.testing.assert_allclose(self.basis.ftran(rhs), rhs, atol=1e-12)
        np.testing.assert_allclose(self.basis.btran(rhs), rhs, atol=1e-12)

    def test_ftran_matches_dense_solve_on_a_nonsymmetric_basis(self):
        # 非对称、非单位基，直接与 numpy 稠密解比较。
        basic = [0, 4, 2, 5]
        basis = SparseBasis(self.matrix, basic)
        square = np.array(A_ROWS)[:, basic]
        rhs = np.array([1.5, -0.5, 2.25, 0.75])
        expected = np.linalg.solve(square, rhs)
        np.testing.assert_allclose(basis.ftran(rhs), expected, rtol=1e-10, atol=1e-12)
        # BTRAN 解的是 B' y = c，须与稠密转置解一致。
        other = np.array([-1.0, 2.0, 0.5, 3.0])
        expected_t = np.linalg.solve(square.T, other)
        np.testing.assert_allclose(basis.btran(other), expected_t, rtol=1e-10, atol=1e-12)

    def test_ftran_and_btran_are_exact_inverses(self):
        basic = [3, 4, 1, 5]
        basis = SparseBasis(self.matrix, basic)
        rng = np.random.default_rng(20260910)
        for _ in range(5):
            rhs = rng.normal(size=4)
            # B(B^{-1} b) = b 与 B'(B^{-T} c) = c 必须同时成立。
            np.testing.assert_allclose(basis.matrix[:, basic] @ basis.ftran(rhs), rhs, atol=1e-10)
            np.testing.assert_allclose(
                basis.matrix[:, basic].T @ basis.btran(rhs), rhs, atol=1e-10)

    # ---------- 乘积形式更新 ----------

    def test_product_form_update_matches_reassembled_basis(self):
        # 单个枢轴：换入第 4 列、换出第 1 个基位（全局列 0）。
        entering = 4
        column = self.basis.ftran(self.matrix[:, entering].toarray().ravel())
        self.basis.update(position=0, entering_column=column, entering=entering, leaving=0)
        new_basic = [4, 1, 2, 3]
        square = np.array(A_ROWS)[:, new_basic]
        rhs = np.array([2.0, -1.0, 0.5, 4.0])
        np.testing.assert_allclose(self.basis.basic, new_basic)
        np.testing.assert_allclose(self.basis.ftran(rhs), np.linalg.solve(square, rhs),
                                   rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(self.basis.btran(rhs), np.linalg.solve(square.T, rhs),
                                   rtol=1e-9, atol=1e-11)

    def test_large_eta_pivot_does_not_cancel_the_pivot_coordinate(self):
        # B0=[1] 换入 [1e12] 后，B1=[1e12]；直接除法给 FTRAN(1e12)=1。
        # 若先算 gamma≈1 再做 1e12-gamma*1e12，会丢失约 5e-10 的解精度，
        # 放回原方程就产生约 500 的绝对残差。
        matrix = dense_matrix([[1., 1e12]])
        basis = SparseBasis(matrix, [0])
        direction = basis.ftran(matrix[:, 1].toarray().ravel())
        basis.update(0, direction, 1, 0)
        primal = basis.ftran(np.array([1e12]))
        dual = basis.btran(np.array([-1.]))
        self.assertLessEqual(abs(primal[0]-1.), 1e-12)
        self.assertLessEqual(abs(float((matrix[:, basis.basic] @ primal)[0])-1e12), 1e-7)
        self.assertLessEqual(abs(dual[0]+1e-12), 1e-20)

    def test_product_form_ftran_uses_inplace_axpy_and_dense_reference(self):
        # 单个 eta 更新仍使用 BLAS DAXPY 处理非主元，并与稠密直接解一致。
        entering = 4
        column = self.basis.ftran(self.matrix[:, entering].toarray().ravel())
        self.basis.update(0, column, entering, 0)
        rhs = np.array([2., -1., 0.5, 4.])
        square = np.array(A_ROWS)[:, self.basis.basic]
        with patch('zyo.sparse_simplex.daxpy', wraps=reference_daxpy) as axpy:
            actual = self.basis.ftran(rhs)
        self.assertGreaterEqual(axpy.call_count, 1)
        np.testing.assert_allclose(actual, np.linalg.solve(square, rhs),
                                   rtol=1e-9, atol=1e-11)

    def test_cached_basis_indices_follow_every_pivot_and_rebuild(self):
        # 大规模稀疏切列反复解析 Python 列表会耗时；索引缓存必须与真实基同步，
        # 否则残差门检查的是旧基，即使回代看似正常也不能视作正确解。
        np.testing.assert_array_equal(self.basis._basic_indices, [0, 1, 2, 3])
        for position, entering in ((0, 4), (2, 5)):
            leaving = self.basis.basic[position]
            column = self.basis.ftran(self.matrix[:, entering].toarray().ravel())
            self.basis.update(position, column, entering, leaving)
            np.testing.assert_array_equal(self.basis._basic_indices, self.basis.basic)
            np.testing.assert_array_equal(self.basis.assemble().toarray(),
                                          np.array(A_ROWS)[:, self.basis.basic])
        self.basis.refactorise()
        np.testing.assert_array_equal(self.basis._basic_indices, self.basis.basic)
        rhs = np.array([1., -2., 3., 0.5])
        expected = np.linalg.solve(np.array(A_ROWS)[:, self.basis.basic].T, rhs)
        np.testing.assert_allclose(self.basis.btran(rhs), expected, rtol=1e-10, atol=1e-12)

    def test_residual_operator_reuses_current_basis_until_pivot(self):
        # 先验证红灯：同一基上的 FTRAN/BTRAN 不应为残差复核重复切列；
        # 换基后首个复核必须切取新基，随后再次复用，重分解亦须刷新。
        rhs = np.array([1., -2., 3., 0.5])
        with patch.object(self.basis, 'assemble', wraps=self.basis.assemble) as assemble:
            np.testing.assert_allclose(self.basis.ftran(rhs), rhs)
            np.testing.assert_allclose(self.basis.btran(rhs), rhs)
            self.assertEqual(assemble.call_count, 0)
            entering = 4
            column = self.basis.ftran(self.matrix[:, entering].toarray().ravel())
            self.basis.update(0, column, entering, 0)
            square = np.array(A_ROWS)[:, self.basis.basic]
            np.testing.assert_allclose(self.basis.ftran(rhs), np.linalg.solve(square, rhs),
                                       rtol=1e-9, atol=1e-11)
            np.testing.assert_allclose(self.basis.btran(rhs), np.linalg.solve(square.T, rhs),
                                       rtol=1e-9, atol=1e-11)
            self.assertEqual(assemble.call_count, 1)
            self.basis.refactorise()
            self.assertEqual(assemble.call_count, 2)
            np.testing.assert_allclose(self.basis.btran(rhs), np.linalg.solve(square.T, rhs),
                                       rtol=1e-9, atol=1e-11)
            self.assertEqual(assemble.call_count, 2)

    def test_absolute_residual_operator_is_cached_only_for_current_basis(self):
        # LAPACK 后向误差分母需要 |B|；同一基上的正反回代复用，换基则必须失效。
        rhs = np.array([1., -2., 3., 0.5])
        initial = self.basis._residual_abs_basis_matrix
        np.testing.assert_allclose(initial.toarray(), np.eye(4))
        self.basis.ftran(rhs)
        self.basis.btran(rhs)
        self.assertIs(self.basis._residual_abs_basis_matrix, initial)
        entering = 4
        column = self.basis.ftran(self.matrix[:, entering].toarray().ravel())
        self.basis.update(0, column, entering, 0)
        self.assertIsNone(self.basis._residual_abs_basis_matrix)
        square = np.array(A_ROWS)[:, self.basis.basic]
        np.testing.assert_allclose(self.basis.ftran(rhs), np.linalg.solve(square, rhs), atol=1e-10)
        np.testing.assert_allclose(self.basis.btran(rhs), np.linalg.solve(square.T, rhs), atol=1e-10)
        refreshed = self.basis._residual_abs_basis_matrix
        self.assertIsNot(refreshed, initial)
        np.testing.assert_allclose(refreshed.toarray(), np.abs(square))
        self.basis.refactorise()
        self.assertIsNot(self.basis._residual_abs_basis_matrix, refreshed)
        np.testing.assert_allclose(self.basis._residual_abs_basis_matrix.toarray(), np.abs(square))

    def test_repeated_updates_stay_consistent_without_refactorisation(self):
        # 连续 6 次枢轴，更新链全程不重分解；每次都与当前基的稠密解比较。
        rng = np.random.default_rng(20260911)
        for step in range(6):
            position = step % 4
            leaving = self.basis.basic[position]
            candidates = [j for j in range(6) if j not in self.basis.basic]
            entering = candidates[step % len(candidates)]
            column = self.basis.ftran(self.matrix[:, entering].toarray().ravel())
            self.basis.update(position=position, entering_column=column,
                              entering=entering, leaving=leaving)
            square = self.basis.assemble().toarray()
            rhs = rng.normal(size=4)
            np.testing.assert_allclose(self.basis.ftran(rhs), np.linalg.solve(square, rhs),
                                       rtol=1e-8, atol=1e-10)
            np.testing.assert_allclose(self.basis.btran(rhs), np.linalg.solve(square.T, rhs),
                                       rtol=1e-8, atol=1e-10)
        self.assertGreaterEqual(len(self.basis.updates), 1)

    def test_refactorisation_clears_updates_and_keeps_the_same_basis(self):
        entering = 5
        column = self.basis.ftran(self.matrix[:, entering].toarray().ravel())
        self.basis.update(position=2, entering_column=column, entering=entering, leaving=2)
        snapshot = list(self.basis.basic)
        self.basis.refactorise()
        self.assertEqual(self.basis.basic, snapshot)
        self.assertEqual(self.basis.updates, [])
        rhs = np.array([1.0, 1.0, 1.0, 1.0])
        square = np.array(A_ROWS)[:, snapshot]
        np.testing.assert_allclose(self.basis.ftran(rhs), np.linalg.solve(square, rhs), rtol=1e-10)

    def test_zero_pivot_is_rejected_instead_of_corrupting_the_basis(self):
        before = list(self.basis.basic)
        column = np.array([0.0, 1.0, 0.0, 0.0])
        with self.assertRaises(NumericalError):
            self.basis.update(position=0, entering_column=column, entering=4, leaving=0)
        self.assertEqual(self.basis.basic, before)

    # ---------- 规模与陈旧标记 ----------

    def test_update_chain_length_triggers_refactorisation_request(self):
        basis = SparseBasis(self.matrix, self.basic, max_eta=2)
        self.assertFalse(basis.needs_refactorisation())
        for step in range(2):
            position = step
            entering = 4+step
            column = basis.ftran(self.matrix[:, entering].toarray().ravel())
            basis.update(position=position, entering_column=column,
                         entering=entering, leaving=position)
        self.assertTrue(basis.needs_refactorisation())

    def test_stale_marking_is_explicit_and_survives_until_refactorisation(self):
        self.basis.mark_stale('test reason')
        self.assertTrue(self.basis.needs_refactorisation())
        self.assertEqual(self.basis.state.reason, 'test reason')
        self.basis.refactorise()
        self.assertFalse(self.basis.needs_refactorisation())
        self.assertEqual(self.basis.state.reason, '')

    def test_singular_initial_basis_is_rejected(self):
        with self.assertRaises(NumericalError):
            SparseBasis(dense_matrix([[1.0, 2.0], [2.0, 4.0]]), [0, 1])

    # ---------- 对偶量与基解 ----------

    def test_reduced_costs_vanish_on_basic_columns(self):
        # 基列上的简约成本必须为（近）零：r = c - A'y 且 y = B^{-1}c_B。
        basis = SparseBasis(self.matrix, [4, 1, 2, 5])
        costs = np.array([1.0, -2.0, 3.0, 0.5, 4.0, -1.0])
        reduced = basis.reduced_costs(costs)
        for column in basis.basic:
            self.assertAlmostEqual(reduced[column], 0.0, places=9)
        # 非基列的简约成本须与稠密公式一致。
        square = basis.assemble().toarray()
        dual = np.linalg.solve(square.T, costs[basis.basic])
        expected = costs - np.array(A_ROWS).T @ dual
        np.testing.assert_allclose(reduced, expected, rtol=1e-9, atol=1e-11)

    def test_basic_solution_satisfies_the_basis_equations(self):
        basis = SparseBasis(self.matrix, self.basic)
        nonbasic = {4: 1.5, 5: -0.5}
        x_basic = basis.basic_solution(nonbasic)
        # 手算：B 为单位阵，故 x_B = -(A[:,4]*1.5 + A[:,5]*(-0.5))。
        expected = -1.5*np.array(A_ROWS)[:, 4]+0.5*np.array(A_ROWS)[:, 5]
        np.testing.assert_allclose(x_basic, expected, atol=1e-12)

    # ---------- 初始基提取 ----------

    def test_extract_basis_finds_a_full_rank_set(self):
        basic, dependent = extract_basis(self.matrix)
        self.assertEqual(len(basic), 4)
        self.assertEqual(dependent, [])
        square = np.array(A_ROWS)[:, basic]
        self.assertGreater(abs(np.linalg.det(square)), 1e-12)

    def test_extract_basis_reports_dependent_rows_for_duplicate_columns(self):
        matrix = dense_matrix([[1.0, 2.0, 2.0], [1.0, 1.0, 1.0], [2.0, 3.0, 3.0]])
        basic, dependent = extract_basis(matrix)
        self.assertEqual(len(basic), 2)
        self.assertTrue(dependent)


if __name__ == '__main__':
    unittest.main()
