# Phase-I 启动方案的独立验收：逻辑列优先启动（crash）必须与逐行人工变量（plain）给出同一个
# 最优值，并且只在**严格降低**初始人工不可行量时才被自动采用——这条触发规则来自 MIPLIB 六题
# 实测（neos-3381206-awhea 是唯一被 crash 弄差的实例，而它恰好也是不可行量没有下降的那一题）。
import math
import unittest

import numpy as np
from scipy.sparse import csc_matrix

from zyo.native_lp import build_homogeneous, solve_certified_lp
from zyo.sparse_simplex import _crash_start, _phase_one

INF = math.inf


def homogeneous(rows, rhs, sense, lower, upper):
    body = csc_matrix(np.array(rows, dtype=float))
    return build_homogeneous(body, np.array(lower, dtype=float), np.array(upper, dtype=float),
                             np.array(rhs, dtype=float), sense=sense)


class PhaseOneCrashTests(unittest.TestCase):
    def test_crash_is_taken_when_it_strictly_reduces_the_artificial_activity(self):
        # min x ; 0 <= x <= 5。齐次行是 x - s = 0，s ∈ (-inf, 5]；结构列 x 自己就是 +e_0，
        # 于是它能取到恰好满足本行的值 5（在 [0, 5] 内），该行不需要人工变量。
        # 逐行人工变量方案的初值是 s 落在上界 5 时产生的 5，所以 5 -> 0 是严格下降。
        wide, lower, upper, _, _, _ = homogeneous([[1.0]], [5.0], ['<='], [0.0], [5.0])
        phase, record = _phase_one(wide, lower, upper, 1, wide.shape[1])
        self.assertEqual(phase, 'FEASIBLE', record.get('reason'))
        self.assertIn('logical-first crash', record['start_strategy'])
        self.assertEqual(record['start_activity'], 0.0)
        self.assertLess(record['start_activity'], record['plain_start_activity'])

    def test_forcing_each_strategy_yields_the_same_phase_one_outcome(self):
        # 强制 crash 与强制 plain 都必须到达人工目标 0（Phase-I 可行），即启动方案不改变结论。
        wide, lower, upper, _, _, _ = homogeneous([[1.0, 1.0], [1.0, 3.0]], [4.0, 6.0],
                                                  ['<=', '<='], [0.0, 0.0], [10.0, 10.0])
        outcomes = {}
        for label, forced in (('crash', True), ('plain', False)):
            phase, record = _phase_one(wide, lower, upper, 2, wide.shape[1],
                                       phase_one_crash=forced)
            self.assertEqual(phase, 'FEASIBLE', f'{label}: {record.get("reason")}')
            self.assertLessEqual(abs(record.get('artificial_sum') or 0.0), 1e-7, label)
            outcomes[label] = record
        self.assertIn('forced', outcomes['crash']['start_strategy'])
        self.assertIn('forced', outcomes['plain']['start_strategy'])
        # plain 一定给每行都加人工变量；crash 的人工变量数不超过它。
        self.assertEqual(outcomes['plain']['start_artificial_rows'], 2)
        self.assertLessEqual(outcomes['crash']['start_artificial_rows'], 2)

    def test_crash_never_changes_the_certified_optimum(self):
        # 这一题必须**真的走 Phase-I**：min -x-2y ; x+y<=4 ; x+3y<=6 ; 0<=x,y<=10。
        # 齐次编码的列是 [1,1,-1,0] / [1,3,0,-1]，没有 +1 单位列，因此 solve_lp 的逻辑基
        # 探测不会命中，一定进入 Phase-I（用单行题做这个断言是错的：那一题的 +1 结构列本身
        # 就是可行的单位基，根本不会进 Phase-I）。
        result = solve_certified_lp(matrix=np.array([[1.0, 1.0], [1.0, 3.0]]),
                                    costs=np.array([-1.0, -2.0]), lower=np.zeros(2),
                                    upper=np.array([10.0, 10.0]), rhs=np.array([4.0, 6.0]),
                                    sense=['<=', '<='])
        self.assertTrue(result.optimal, result.message)
        self.assertAlmostEqual(result.objective, -5.0, places=8)
        self.assertTrue(result.certificate.verified, result.certificate.reason)
        phase = result.phase_one or {}
        self.assertTrue(phase.get('used'), phase)
        for key in ('start_strategy', 'start_activity', 'plain_start_activity'):
            self.assertIn(key, phase)

    def test_a_feasible_unit_basis_still_skips_phase_one(self):
        # 单行题 min -x ; 0<=x<=5 ; x<=5：结构列 x 自己就是 +1 单位列，取到 5 即满足本行，
        # 因此 solve_lp 直接采纳该基、**不进 Phase-I**——启动方案对这类题无关。
        result = solve_certified_lp(matrix=np.array([[1.0]]), costs=np.array([-1.0]),
                                    lower=np.array([0.0]), upper=np.array([5.0]),
                                    rhs=np.array([5.0]), sense=['<='])
        self.assertTrue(result.optimal, result.message)
        self.assertAlmostEqual(result.objective, -5.0, places=8)
        self.assertFalse((result.phase_one or {}).get('used'))

    def test_crash_start_reports_a_basis_for_every_row(self):
        # 基必须覆盖每一行：直接用作基的列或该行的人工列，且互不重复。
        wide, lower, upper, _, _, _ = homogeneous([[1.0, 1.0], [1.0, 3.0]], [4.0, 6.0],
                                                  ['<=', '<='], [0.0, 0.0], [10.0, 10.0])
        crash = _crash_start(wide, lower, upper, np.array([0.0, 0.0, 4.0, 6.0]))
        self.assertIsNotNone(crash)
        self.assertEqual(len(crash['basis_offset']), wide.shape[0])
        self.assertEqual(len(set(crash['basis_offset'])), wide.shape[0])
        # 直接取到基的列（非人工列）必须真的落在自己的界内。
        for column in crash['basis_offset']:
            if column >= wide.shape[1]:
                continue
            value = crash['initial'][column]
            self.assertGreaterEqual(value, lower[column]-1e-9)
            self.assertLessEqual(value, upper[column]+1e-9)

    def test_crash_start_is_none_when_no_unit_column_can_help(self):
        # 没有单位列可用时返回 None，由调用方退回 plain。
        wide = csc_matrix(np.array([[2.0, 3.0]]))
        crash = _crash_start(wide, np.array([-INF, -INF]), np.array([INF, INF]),
                             np.array([0.0, 0.0]))
        self.assertIsNone(crash)

    def test_strategy_is_recorded_for_audit(self):
        result = solve_certified_lp(matrix=np.array([[1.0, 1.0]]), costs=np.array([1.0, 1.0]),
                                    lower=np.zeros(2), upper=np.array([10.0, 10.0]),
                                    rhs=np.array([2.0]), sense=['>='])
        self.assertTrue(result.optimal, result.message)
        phase = result.phase_one or {}
        for key in ('start_strategy', 'start_activity', 'plain_start_activity',
                    'start_artificial_rows'):
            self.assertIn(key, phase)

    def test_free_variables_still_work_with_the_crash(self):
        # 拆分与启动方案必须能共存：x,y 自由 ; x+y>=5 ; x-y==0 ; min x+y -> 5。
        result = solve_certified_lp(
            matrix=np.array([[1.0, 1.0], [1.0, -1.0]]), costs=np.array([1.0, 1.0]),
            lower=np.array([-INF, -INF]), upper=np.array([INF, INF]),
            rhs=np.array([5.0, 0.0]), sense=['>=', '=='])
        self.assertTrue(result.optimal, result.message)
        self.assertAlmostEqual(result.objective, 5.0, places=8)
        self.assertTrue(result.certificate.verified, result.certificate.reason)


if __name__ == '__main__':
    unittest.main()
