# 稀疏基数值修正的独立反例：使用手算线性方程与真实 LU/eta 链，不借助优化器。
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csc_matrix

from zyo.errors import NumericalError
from zyo.sparse_simplex import SparseBasis, revised_simplex


class BasisRefinementTests(unittest.TestCase):
    def test_btran_returns_the_corrected_solution_not_the_original_candidate(self):
        # B^T y=(1,0) 的精确解为 (1,-10^8)；候选的第二行残差为 1。
        basis = SparseBasis(csc_matrix([[1., 1e8], [0., 1.]]), [0, 1])
        corrected = basis._check(np.array([1., -1e8+1.]), np.array([1., 0.]),
                                 transpose=True)
        np.testing.assert_array_equal(corrected, [1., -1e8])
        self.assertFalse(basis.state.stale)

    def test_refinement_solves_against_the_current_eta_updated_basis(self):
        # 初始 LU 是 I，换基后非对称 B=[[2,1],[3,4]]；两个方向的右端项手算不同。
        matrix = csc_matrix([[1., 0., 2., 1.], [0., 1., 3., 4.]])
        for transpose, rhs in ((False, [4., 11.]), (True, [8., 9.])):
            with self.subTest(transpose=transpose):
                basis = SparseBasis(matrix, [0, 1])
                for position, entering in ((0, 2), (1, 3)):
                    column = basis.ftran(matrix[:, entering].toarray().ravel())
                    basis.update(position, column, entering, basis.basic[position])
                corrected = basis._check(np.array([1.01, 1.99]), np.array(rhs),
                                         transpose=transpose)
                np.testing.assert_allclose(corrected, [1., 2.], atol=1e-12, rtol=0.)
                self.assertEqual(basis.state.refactorisations, 1)

    def test_componentwise_gate_catches_a_bad_small_row_hidden_by_a_large_rhs(self):
        # 大行不应替小行担保：小行的相对后向误差是 1/3，而原 rhs 门为约 10^-20。
        basis = SparseBasis(csc_matrix([[1e12, 0.], [0., 1e-8]]), [0, 1],
                            refinement_steps=0, residual_metric='componentwise')
        with self.assertRaises(NumericalError):
            basis._check(np.array([1., 2.]), np.array([1e12, 1e-8]), transpose=False)
        self.assertTrue(basis.state.stale)

    def test_componentwise_gate_uses_the_transposed_operator_and_not_column_norms(self):
        # B^T y=(1,0)，第二分量的分母为 2*10^8+0.01，后向误差约 5*10^-11。
        basis = SparseBasis(csc_matrix([[1., 1e8], [0., 1.]]), [0, 1],
                            refinement_steps=0, residual_metric='componentwise')
        candidate = np.array([1., -1e8+0.01])
        corrected = basis._check(candidate, np.array([1., 0.]), transpose=True)
        np.testing.assert_array_equal(corrected, candidate)
        self.assertGreater(basis.state.btran_residual, 1e-3)
        self.assertLess(basis.state.btran_backward_error, 1e-9)

    def test_nonfinite_candidate_is_rejected_before_any_refinement(self):
        basis = SparseBasis(csc_matrix(np.eye(2)), [0, 1])
        with self.assertRaises(NumericalError):
            basis._check(np.array([np.nan, 1.]), np.ones(2), transpose=False)
        self.assertTrue(basis.state.stale)

    def test_overflow_in_the_backward_error_scale_is_not_an_acceptance(self):
        # 若分母溢出，残差/inf 变零也不能当作证据。
        basis = SparseBasis(csc_matrix([[1e308, 0.], [0., 1.]]), [0, 1],
                            refinement_steps=0)
        with np.errstate(over='ignore', invalid='ignore'):
            with self.assertRaises(NumericalError):
                basis._check(np.array([2., 1.]), np.ones(2), transpose=False)

    def test_zero_rhs_and_exact_zero_solution_have_zero_backward_error(self):
        basis = SparseBasis(csc_matrix(np.eye(2)), [0, 1])
        np.testing.assert_array_equal(basis.ftran(np.zeros(2)), [0., 0.])
        self.assertEqual(basis.state.ftran_backward_error, 0.)

    def test_validated_ftran_and_btran_propagate_the_refined_vector_to_the_caller(self):
        # 故障注入只在一次回代加入已知误差，修正仍用真实分解；检查调用者得到的物理方程。
        for transpose, rhs in ((False, [4., 11.]), (True, [8., 9.])):
            with self.subTest(transpose=transpose):
                basis = SparseBasis(csc_matrix([[2., 1.], [3., 4.]]), [0, 1])
                original = basis._solve_current
                first = [True]

                def perturbed(vector, *, transpose):
                    result = original(vector, transpose=transpose)
                    if first[0]:
                        first[0] = False
                        result += [0.01, -0.01]
                    return result

                basis._solve_current = perturbed
                actual = basis.btran(rhs) if transpose else basis.ftran(rhs)
                np.testing.assert_allclose(actual, [1., 2.], atol=1e-12, rtol=0.)
                self.assertGreater(basis.state.refinement_improvements, 0)

    def test_stagnant_corrections_stop_and_reject_instead_of_looping(self):
        basis = SparseBasis(csc_matrix(np.eye(2)), [0, 1])
        # 模拟分解已坏且无法修正，0 修正不能被视为进步或通过。
        basis._solve_current = lambda rhs, *, transpose: np.zeros(2)
        with self.assertRaises(NumericalError):
            basis._check(np.array([2., 1.]), np.ones(2), transpose=False)
        self.assertEqual(basis.state.refinement_attempts, 1)
        self.assertEqual(basis.state.refinement_improvements, 0)

    def test_refinement_budget_stops_an_improving_but_inaccurate_solver(self):
        basis = SparseBasis(csc_matrix(np.eye(2)), [0, 1], refinement_steps=2)
        # 每次仅修正残差的 3/4，两次后仍大于门，必须拒绝。
        basis._solve_current = lambda rhs, *, transpose: 0.75*rhs
        with self.assertRaises(NumericalError):
            basis._check(np.array([2., 1.]), np.ones(2), transpose=False)
        self.assertEqual(basis.state.refinement_attempts, 2)

    def test_legacy_rhs_gate_is_available_only_as_an_explicit_measured_alternative(self):
        basis = SparseBasis(csc_matrix([[1., 1e8], [0., 1.]]), [0, 1],
                            refinement_steps=0, residual_metric='rhs')
        with self.assertRaises(NumericalError):
            basis._check(np.array([1., -1e8+0.01]), np.array([1., 0.]), transpose=True)

    def test_invalid_accuracy_settings_are_rejected(self):
        for options in ({'residual_tol': 0.}, {'residual_tol': np.nan},
                        {'refinement_steps': -1}, {'refinement_steps': 1.5},
                        {'residual_metric': 'column_l2'}):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    SparseBasis(csc_matrix(np.eye(2)), [0, 1], **options)

    def test_default_gate_preserves_the_absolute_roundoff_allowance_near_a_zero_component(self):
        # 零右端项的微小分量不具相对精度承诺：绝对残差 10^-16 可通过原有门。
        basis = SparseBasis(csc_matrix(np.eye(2)), [0, 1], refinement_steps=0)
        actual = basis._check(np.array([1., 1e-16]), np.array([1., 0.]), transpose=False)
        np.testing.assert_array_equal(actual, [1., 1e-16])

    def test_default_gate_still_rejects_large_absolute_error_with_tiny_backward_error(self):
        # 文献意义的后向稳定不能代替原 rhs 门：B^T y 的绝对误差为 0.01，需显式修正。
        basis = SparseBasis(csc_matrix([[1., 1e8], [0., 1.]]), [0, 1], refinement_steps=0)
        with self.assertRaises(NumericalError):
            basis._check(np.array([1., -1e8+0.01]), np.array([1., 0.]), transpose=True)

    def test_more_than_two_corrections_cannot_bypass_the_refinement_budget(self):
        with self.assertRaises(ValueError):
            SparseBasis(csc_matrix(np.eye(2)), [0, 1], refinement_steps=3)

    def test_starting_solve_failure_exports_the_actual_check_and_refactor_evidence(self):
        # 故障注入持续坏回代，真实检查/驱动/重分解照常运行；不能只给状态而丢失诊断。
        with patch.object(SparseBasis, '_solve_current',
                          side_effect=lambda self_rhs, **kwargs: np.full(2, np.nan)):
            result = revised_simplex(csc_matrix(np.eye(2)), np.zeros(2),
                                     np.zeros(2), np.ones(2), basic=[0, 1])
        self.assertEqual(result.status, 'NUMERICAL_ERROR')
        self.assertEqual(result.basis_diagnostics['solve_checks'], 2)
        self.assertEqual(result.basis_diagnostics['refactorisations'], 2)
        self.assertTrue(result.basis_diagnostics['stale'])
        self.assertIn('non-finite', result.basis_diagnostics['reason'])
        self.assertEqual(result.refactor_retries, 1)


if __name__ == '__main__':
    unittest.main()
