# Phase-I 启动方案的独立验收：逻辑列优先启动（crash）必须与逐行人工变量（plain）给出同一个
# 最优值；等人工目标时仅在节省的人工基列超过当前更新额度时优先 crash，
# 避免让大题的整个额度耗在零值人工列的逐一退出，也保留小题原来的 plain 选择。
import math
import inspect
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
    def test_forced_crash_reports_unavailable_instead_of_silent_plain_fallback(self):
        # 两行共享同一结构列，没有任一行的单位列；强制策略应明确失败。
        wide = csc_matrix([[1.], [1.]])
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            _phase_one(wide, np.zeros(1), np.full(1, INF), 2, 1,
                       phase_one_crash=True, iteration_limit=10)

    def test_equal_activity_uses_crash_if_it_saves_more_artificial_rows_than_budget(self):
        # 手算：3个单位行、零残差。两个起点人工目标都为0；plain有3个人工基列，
        # 仅2次额度不足以全部驱除，crash直接用单位列为基，不占清理额度。
        wide = csc_matrix(np.eye(3))
        lower = np.zeros(3)
        upper = np.full(3, INF)
        phase, record = _phase_one(wide, lower, upper, 3, 3, iteration_limit=2)
        self.assertEqual(phase, 'FEASIBLE', record.get('reason'))
        self.assertIn('budget-aware', record['start_strategy'])
        self.assertEqual(record['start_artificial_rows'], 0)

    def test_equal_activity_keeps_plain_when_budget_covers_original_artificial_rows(self):
        # 小题若额度已能覆盖初始人工基列，则不因“更少列”而改变默认路径。
        wide = csc_matrix(np.eye(3))
        phase, record = _phase_one(wide, np.zeros(3), np.full(3, INF),
                                   3, 3, iteration_limit=4)
        self.assertEqual(phase, 'FEASIBLE', record.get('reason'))
        self.assertIn('plain', record['start_strategy'])

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

    def test_nonconsecutive_artificial_rows_keep_compact_column_numbers(self):
        # 两个逻辑基行夹着三个需人工列的行；人工列编号必须按其出现次序连续分配。
        body = csc_matrix(np.array([[1., 0., 0.], [0., 0., 2.], [0., 0., 3.],
                                    [0., 1., 0.], [0., 0., 4.]]))
        crash = _crash_start(body, np.zeros(3), np.full(3, INF), np.zeros(3))
        self.assertEqual(crash['artificial_rows'], [1, 2, 4])
        self.assertEqual(crash['basis_offset'], [0, 3, 4, 1, 5])

    def test_second_unit_candidate_uses_signed_original_row_activity(self):
        # 第一候选 +e0 需取 -2 而越界；第二候选 -e0 取 2 合法。
        # 多行列固定在 1，会贡献行活动量 (2,3,4)，故仅后两行需人工变量 3+4=7。
        body = csc_matrix(np.array([[1., -1., 0., 2., 0.],
                                    [0., 0., 1., 3., 0.],
                                    [0., 0., 0., 4., 1.]]))
        lower = np.array([0., 0., 0., 1., 0.])
        upper = np.array([10., 10., 10., 1., 10.])
        crash = _crash_start(body, lower, upper, lower.copy())
        self.assertEqual(crash['logical_rows'], [0])
        self.assertEqual(crash['basis_offset'], [1, 5, 6])
        self.assertEqual(crash['artificial_rows'], [1, 2])
        self.assertEqual(crash['initial'][1], 2.)
        self.assertEqual(crash['activity'], 7.)

    def test_sparse_crash_matches_independent_dense_small_row_oracle(self):
        # 小矩阵才允许稠密参考；直接按行定义计算候选，不调用求解器生成预期值。
        rng = np.random.default_rng(20260929)
        for trial in range(30):
            body = rng.integers(-2, 3, size=(5, 8)).astype(float)
            body[:, :5] = np.diag(rng.choice([-1., 1.], size=5))
            lower = rng.integers(-2, 2, size=8).astype(float)
            upper = lower+rng.integers(0, 5, size=8)
            placement = lower.copy()
            candidates = {}
            for column in range(body.shape[1]):
                nonzero = np.flatnonzero(body[:, column])
                if len(nonzero) == 1 and abs(abs(body[nonzero[0], column])-1.) <= 1e-12:
                    row = int(nonzero[0])
                    candidates.setdefault(row, []).append((column, float(np.sign(body[row, column]))))
            expected = placement.copy()
            chosen = {}
            for row in sorted(candidates):
                for column, sign in candidates[row]:
                    others = float(body[row] @ expected)-body[row, column]*expected[column]
                    value = -others/sign
                    if lower[column]-1e-12 <= value <= upper[column]+1e-12:
                        chosen[row] = column
                        expected[column] = value
                        break
            actual = _crash_start(csc_matrix(body), lower, upper, placement)
            if not chosen:
                self.assertIsNone(actual, trial)
                continue
            artificial_rows = [row for row in range(5) if row not in chosen]
            basis = [chosen[row] if row in chosen else 8+artificial_rows.index(row)
                     for row in range(5)]
            with self.subTest(trial=trial):
                self.assertEqual(actual['logical_rows'], sorted(chosen))
                self.assertEqual(actual['basis_offset'], basis)
                self.assertEqual(actual['artificial_rows'], artificial_rows)
                np.testing.assert_allclose([actual['initial'][j] for j in range(8)], expected,
                                           rtol=0, atol=1e-12)
                self.assertAlmostEqual(actual['activity'],
                                       float(np.sum(np.abs((body @ expected)[artificial_rows]))), places=10)

    def test_crash_mapping_does_not_scan_artificial_row_list_for_every_row(self):
        # 大矩阵启动频繁发生；逐行 list.index 会把 O(m) 编号退化为 O(m²)。
        self.assertNotIn('artificial_rows.index(', inspect.getsource(_crash_start))

    def test_crash_does_not_materialize_each_candidate_row(self):
        # 单位列仅作用本行；逐行稀疏切片再稠密化会令大题启动成本随候选行数爆炸。
        self.assertNotIn('body[row, :].todense()', inspect.getsource(_crash_start))

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
