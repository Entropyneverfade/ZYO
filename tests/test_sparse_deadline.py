# 时间预算在**牛顿步内部**耗尽时的行为验收。
# 背景：IPM 只在两轮迭代之间检查 deadline，而单次增广系统稀疏 LU 分解可以远超整个时间预算。
# 实测 MIPLIB neos-4763324-toguru（106954 行 / 53593 列）在 time_limit=60 下于 splu 内停留
# 1000 秒以上、RSS 13.6 GB（栈由 faulthandler 捕获，见 research/probe_sparse_deadline_stack.py）。
import time
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csc_matrix

from zyo.model import Model
from zyo.options import SolveOptions
from zyo.sparse_lp import (_augmented_solver, _normal_solver, _standard_form, DeadlineExceeded,
                           solve_relaxation)


def tiny_model():
    """min -x - 2y ; x + y <= 4 ; x + 3y <= 6 ; 0 <= x,y <= 10 -> (3,1)，目标 -5。

    注意不是 `min x+y`：那个题在 (0,0) 取到 0（写测试时曾把它误当成 (4,0)，实测求解器给出的
    5.8e-11 才是对的）。
    """
    model = Model('deadline')
    x = model.add_var('x', lb=0.0, ub=10.0)
    y = model.add_var('y', lb=0.0, ub=10.0)
    model.add_constr(x + y <= 4)
    model.add_constr(x + 3*y <= 6)
    model.minimize(-x - 2*y)
    return model, [0.0, 0.0], [10.0, 10.0]


class SparseDeadlineTests(unittest.TestCase):
    def test_augmented_solver_raises_when_the_budget_is_already_gone(self):
        matrix = csc_matrix(np.array([[1.0, 1.0], [1.0, 3.0]]))
        x = np.ones(2)
        s = np.ones(2)
        rp = np.zeros(2)
        rd = np.zeros(2)
        with self.assertRaises(DeadlineExceeded):
            _augmented_solver(matrix, x, s, rp, rd, deadline=time.perf_counter()-1.0)

    def test_augmented_solver_still_works_with_a_future_deadline(self):
        matrix = csc_matrix(np.array([[1.0, 1.0], [1.0, 3.0]]))
        x = np.ones(2)
        s = np.ones(2)
        statistics = {}
        solve = _augmented_solver(matrix, x, s, np.zeros(2), np.zeros(2), statistics,
                                  deadline=time.perf_counter()+60.0)
        step = solve(-x*s)
        self.assertEqual(len(step), 3)
        self.assertTrue(np.all(np.isfinite(np.concatenate([np.atleast_1d(part) for part in step]))))

    def test_normal_solver_raises_when_the_budget_is_already_gone(self):
        matrix = csc_matrix(np.array([[1.0, 1.0], [1.0, 3.0]]))
        solve, _ = _normal_solver(matrix, np.ones(2), time.perf_counter()-1.0)
        with self.assertRaises(DeadlineExceeded):
            solve(np.zeros(2))

    def test_relaxation_reports_time_limit_when_a_newton_step_runs_out_of_budget(self):
        # 直接验证"牛顿步内耗尽 -> TIME_LIMIT"这条映射，不依赖真实分解的偶然耗时。
        model, lower, upper = tiny_model()
        options = SolveOptions(time_limit=60., iteration_limit=200)
        started = time.perf_counter()
        with patch('zyo.sparse_lp._augmented_solver', side_effect=DeadlineExceeded()):
            result = solve_relaxation(model, lower, upper, options, started+60.0)
        self.assertEqual(result.status, 'TIME_LIMIT', result.message)
        # 必须真的在预算附近返回，而不是继续迭代到上限。
        self.assertLess(time.perf_counter()-started, 30.0)

    def test_relaxation_still_solves_normally(self):
        model, lower, upper = tiny_model()
        options = SolveOptions(time_limit=60., iteration_limit=200)
        started = time.perf_counter()
        result = solve_relaxation(model, lower, upper, options, started+60.0)
        self.assertEqual(result.status, 'OPTIMAL', result.message)
        # 手算：min -x-2y s.t. x+y<=4, x+3y<=6 -> (3,1)，目标 -5。
        self.assertAlmostEqual(result.objective, -5.0, places=6)

    def test_expired_budget_returns_time_limit_at_the_iteration_checkpoint(self):
        model, lower, upper = tiny_model()
        options = SolveOptions(time_limit=0., iteration_limit=200)
        started = time.perf_counter()
        result = solve_relaxation(model, lower, upper, options, started)
        self.assertEqual(result.status, 'TIME_LIMIT', result.message)

    def test_standard_form_is_reachable_for_the_tiny_model(self):
        # 保证上面用到的 _standard_form 没有被改动成不可用（装配阶段是检查点之前的必经步骤）。
        model, lower, upper = tiny_model()
        standard, error = _standard_form(model, np.asarray(lower, dtype=float),
                                        np.asarray(upper, dtype=float))
        self.assertIsNone(error)
        self.assertEqual(standard['A'].shape[0], 4)


if __name__ == '__main__':
    unittest.main()
