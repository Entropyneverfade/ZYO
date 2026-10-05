# 认证修正单纯形后端的独立验收：公共 API 必须返回与手算一致的结果，且证书必须随结果导出。
# 整数模型必须被明确拒绝，而不是静默当作松弛求解。
import math
import unittest

import numpy as np

from zyo import Model
from zyo.solvers import available_solvers, get_backend
from zyo.status import Status

INF = math.inf


def build_lp():
    """min -x-2y ; x+y <= 4 ; x+3y <= 6 ; x,y >= 0 -> (3,1)，目标 -5（手算）。"""
    model = Model('simplex_lp')
    x = model.add_var('x', lb=0, ub=None)
    y = model.add_var('y', lb=0, ub=None)
    model.add_constr(x + y <= 4)
    model.add_constr(x + 3*y <= 6)
    model.minimize(-x - 2*y)
    return model


def _extra_row(model):
    """A second row so the incidence structure is not just the two original constraints."""
    x, y = model.variables[0], model.variables[1]
    return x + y <= 10


class NativeSimplexBackendTests(unittest.TestCase):
    def test_backend_is_registered_and_declares_lp_only(self):
        info = get_backend('native_simplex').info()
        self.assertEqual(info.name, 'native_simplex')
        self.assertTrue(info.available)
        self.assertEqual(info.license_mode, 'self-developed')
        self.assertIn('LP', info.capabilities)
        self.assertNotIn('MILP', info.capabilities)
        self.assertIn('native_simplex', available_solvers())

    def test_name_is_case_insensitive(self):
        self.assertIs(get_backend('Native_Simplex'), get_backend('native_simplex'))

    def test_public_api_matches_the_hand_solution(self):
        result = build_lp().solve('native_simplex')
        self.assertEqual(result.status, Status.OPTIMAL, result.termination_reason)
        self.assertAlmostEqual(result.objective, -5.0, places=8)
        self.assertAlmostEqual(result.values['x'], 3.0, places=7)
        self.assertAlmostEqual(result.values['y'], 1.0, places=7)
        self.assertLessEqual(result.primal_residual, 1e-7)
        self.assertLessEqual(result.bound_residual, 1e-7)

    def test_certificate_is_exported_with_the_result(self):
        result = build_lp().solve('native_simplex')
        certificate = result.metadata['certificate']
        self.assertTrue(certificate['verified'], certificate['reason'])
        for key in ('row_sign_convention', 'reduced_cost_domain_compatible',
                    'primal_within_tolerance', 'dual_within_tolerance',
                    'complementarity_within_tolerance', 'strong_duality_within_tolerance'):
            self.assertTrue(certificate[key], key)
        self.assertIsNotNone(certificate['duality_gap'])
        self.assertFalse(result.metadata['fallback_used'])
        self.assertEqual(result.metadata['backend_kind'], 'self-developed')

    def test_maximization_returns_the_user_direction_objective(self):
        # max x+2y ; x+y <= 4 ; x+3y <= 6 -> (3,1)，目标 5（网格枚举）。
        model = Model('simplex_max')
        x = model.add_var('x', lb=0, ub=None)
        y = model.add_var('y', lb=0, ub=None)
        model.add_constr(x + y <= 4)
        model.add_constr(x + 3*y <= 6)
        model.maximize(x + 2*y)
        result = model.solve('native_simplex')
        self.assertEqual(result.status, Status.OPTIMAL, result.termination_reason)
        self.assertAlmostEqual(result.objective, 5.0, places=8)

    def test_bounded_variables_are_respected(self):
        # x,y <= 2 时最优 (2, 4/3)，目标 -14/3（网格枚举确认）。
        model = Model('simplex_box')
        x = model.add_var('x', lb=0, ub=2)
        y = model.add_var('y', lb=0, ub=2)
        model.add_constr(x + y <= 4)
        model.add_constr(x + 3*y <= 6)
        model.minimize(-x - 2*y)
        result = model.solve('native_simplex')
        self.assertEqual(result.status, Status.OPTIMAL, result.termination_reason)
        self.assertAlmostEqual(result.objective, -14.0/3.0, places=8)

    def test_infeasible_model_is_reported(self):
        model = Model('simplex_infeasible')
        x = model.add_var('x', lb=0, ub=None)
        model.add_constr(x <= -1)
        model.minimize(x)
        result = model.solve('native_simplex')
        self.assertEqual(result.status, Status.INFEASIBLE, result.termination_reason)
        self.assertIsNone(result.objective)

    def test_integer_variables_are_refused_not_relaxed(self):
        # 有整数变量时必须明确拒绝：把分数解当成 MILP 答案正是本项目禁止的静默替换。
        model = Model('simplex_int')
        z = model.add_var('z', lb=0, ub=10, vtype='I')
        model.minimize(z)
        result = model.solve('native_simplex')
        self.assertEqual(result.status, Status.SOLVER_ERROR, result.termination_reason)
        self.assertIn('LP relaxations only', result.termination_reason)
        self.assertEqual(result.metadata['integer_variables'], 1)

    def test_equality_rows_are_not_relaxed(self):
        # x-y == 4 ; x,y >= 0 ; min x+y -> (4,0)，目标 4。
        model = Model('simplex_eq')
        x = model.add_var('x', lb=0, ub=None)
        y = model.add_var('y', lb=0, ub=None)
        model.add_constr(x - y == 4)
        model.minimize(x + y)
        result = model.solve('native_simplex')
        self.assertEqual(result.status, Status.OPTIMAL, result.termination_reason)
        self.assertAlmostEqual(result.objective, 4.0, places=8)

    def test_sparse_assembly_matches_the_dense_one(self):
        # 认证路径内部全程 csc_matrix，稠密中间表示从来不是求解器要求的：`model_sparse_arrays`
        # 按 O(nonzeros) 装配，语义必须与 `model_arrays` 逐项一致（否则大小实例会算出不同结果）。
        from zyo.solvers.native_simplex import model_arrays, model_sparse_arrays

        model = build_lp()
        model.add_constr(_extra_row(model))
        dense = model_arrays(model, max_dense_entries=None)
        sparse = model_sparse_arrays(model)
        self.assertTrue(np.allclose(np.asarray(sparse[0].todense()), dense[0]))
        for index in (1, 2, 3, 4):
            self.assertTrue(np.allclose(sparse[index], dense[index]), index)
        self.assertEqual(list(sparse[5]), list(dense[5]))
        self.assertEqual(sparse[6], dense[6])

    def test_modeling_boundary_preserves_original_row_and_objective_units(self):
        # 装配属于建模边界；右端项、目标常数与系数都必须在原单位保留。
        from zyo.modeling.matrix import model_sparse_arrays

        model = Model('matrix_boundary')
        x = model.add_var('x', lb=-2, ub=4)
        y = model.add_var('y', lb=0, ub=None)
        model.add_constr(2*x - 3*y >= -5)
        model.minimize(7*x + 11)
        matrix, cost, lower, upper, rhs, sense, constant = model_sparse_arrays(model)
        self.assertEqual(matrix.shape, (1, 2))
        self.assertEqual(matrix[0, 0], 2)
        self.assertEqual(matrix[0, 1], -3)
        self.assertEqual(cost.tolist(), [7, 0])
        self.assertEqual(lower.tolist(), [-2, 0])
        self.assertEqual(upper[0], 4)
        self.assertTrue(math.isinf(upper[1]))
        self.assertEqual(rhs.tolist(), [-5])
        self.assertEqual(sense, ['>='])
        self.assertEqual(constant, 11)

    def test_public_api_accepts_sparse_lp_above_former_dense_gate(self):
        # 1420×1420 已超过旧稠密门，但仅 1420 个非零元；原生入口必须按稀疏量装配。
        model = Model('sparse_public_entry')
        variables = [model.add_var(f'x{i}', lb=0, ub=1) for i in range(1420)]
        for variable in variables:
            model.add_constr(variable <= 1)
        model.minimize(variables[0])
        result = model.solve('native_simplex')
        self.assertEqual(result.status, Status.OPTIMAL, result.termination_reason)
        self.assertAlmostEqual(result.objective, 0.0)
        self.assertEqual(result.metadata['model_nonzeros'], 1420)
        self.assertEqual(result.metadata['matrix_format'], 'csc')
        self.assertGreaterEqual(result.metadata['assembly_seconds'], 0.0)

    def test_status_enum_is_used_not_a_bare_string(self):
        result = build_lp().solve('native_simplex')
        self.assertIsInstance(result.status, Status)
        self.assertEqual(result.status.value, 'OPTIMAL')

    def test_dense_gate_refuses_to_allocate_instead_of_being_killed(self):
        # 规模门必须在**分配之前**拒绝：稠密表要 rows*columns 个 double，实测 MIPLIB 的
        # neos-4722843-widden（113555 x 77723，源文件只有 12.6 MB）要求 70.6 GB，冻结批次 -07 里
        # 38/48 个 worker 就是在这次分配里撞上 RSS 上限被杀、什么记录都没留下的。
        from zyo.solvers.native_simplex import DEFAULT_DENSE_ENTRY_GATE, model_arrays

        class _Size:
            def __init__(self, count):
                self._count = count

            def __len__(self):
                return self._count

        class _Huge:
            variables = _Size(77_723)
            constraints = _Size(113_555)

        with self.assertRaises(MemoryError) as caught:
            model_arrays(_Huge())
        message = str(caught.exception)
        self.assertIn('70.61 GB', message)
        self.assertIn(str(DEFAULT_DENSE_ENTRY_GATE), message)
        # 小模型不受影响，且显式放开时仍可分配（只有测试与小模型会这么做）。
        model = build_lp()
        matrix, *_rest = model_arrays(model)
        self.assertEqual(matrix.shape, (2, 2))
        self.assertEqual(DEFAULT_DENSE_ENTRY_GATE, 2_000_000)

    def test_sparse_allocation_failure_reports_size_limit_through_the_public_api(self):
        # 注入资源失败时仍返回真实状态；公共入口不能静默切换到外部引擎。
        import zyo.solvers.native_simplex as backend
        import time

        original = backend.model_sparse_arrays

        def _refuse(model):
            # Windows 的 sleep(0.01) 可能提前返回；用同一单调时钟保证装配计时门的下界。
            deadline = time.perf_counter() + 0.012
            while time.perf_counter() < deadline:
                pass
            raise MemoryError('sparse allocation refused')

        backend.model_sparse_arrays = _refuse
        try:
            result = build_lp().solve('native_simplex')
        finally:
            backend.model_sparse_arrays = original
        self.assertEqual(result.status, Status.SIZE_LIMIT, result.termination_reason)
        self.assertIn('sparse model assembly failed', result.termination_reason)
        self.assertEqual(result.metadata['matrix_format'], 'csc')
        self.assertFalse(result.metadata['fallback_used'])
        self.assertGreaterEqual(result.metadata['assembly_seconds'], 0.009)
        self.assertEqual(result.metadata['resource_failure_stage'], 'assembly')

    def test_sparse_solver_allocation_failure_reports_size_limit(self):
        # 矩阵装配成功后的内部资源失败也必须留状态，不能冒充数值失败或触发外部回退。
        from unittest.mock import patch

        with patch('zyo.native_lp.solve_certified_lp', side_effect=MemoryError('basis allocation refused')):
            result = build_lp().solve('native_simplex')
        self.assertEqual(result.status, Status.SIZE_LIMIT, result.termination_reason)
        self.assertEqual(result.metadata['model_nonzeros'], 4)
        self.assertFalse(result.metadata['fallback_used'])

    def test_result_validation_allocation_failure_reports_size_limit(self):
        # 求解已返回后，原模型独立检查仍可能申请内存；失败必须带阶段和原生状态。
        from unittest.mock import patch

        with patch('zyo.solvers.native_simplex.validate_candidate',
                   side_effect=MemoryError('postprocess allocation refused')):
            result = build_lp().solve('native_simplex')
        self.assertEqual(result.status, Status.SIZE_LIMIT, result.termination_reason)
        self.assertEqual(result.metadata['resource_failure_stage'], 'result_processing')
        self.assertEqual(result.metadata['model_nonzeros'], 4)
        self.assertFalse(result.metadata['fallback_used'])


if __name__ == '__main__':
    unittest.main()
