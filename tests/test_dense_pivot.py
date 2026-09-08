# 密集内核反例：原始24小时UC在第487次换基后失去可行性；目标由独立对照冻结。
import json
from pathlib import Path
import unittest
from zyo import SolveOptions
from zyo.validation import validate_candidate
from zyo_power.unit_commitment import build_uc_model, extract_uc_trace, validate_uc
import numpy as np
from lzyopt.native import _leaving_row


class DensePivotTests(unittest.TestCase):
    def test_near_zero_ratio_is_not_a_tie_in_slack_units(self):
        # 选第一行会使第二行松弛 -10；真正的零步长阻塞行是第二行。
        row=_leaving_row(np.array([1.,1e13]),np.array([1e-12,0.]),[1,2],1e-10,1e-7)
        self.assertEqual(row,1)

    def test_equal_pivots_use_basis_index_deterministically(self):
        # 步长和主元均相同才以基编号打破平局，不依赖约束行号。
        self.assertEqual(_leaving_row(np.ones(2),np.zeros(2),[8,3],1e-10,1e-7),1)

    def test_exact_degenerate_tie_prefers_stable_pivot(self):
        # 两行都只允许零步长；选择 1e-9 主元会把另一列放大到 1e9。
        row=_leaving_row(np.array([1e-9,1.]),np.zeros(2),[2,3],1e-10,1e-7)
        self.assertEqual(row,1)

    def test_small_negative_rhs_cannot_be_amplified_by_negative_step(self):
        # 负RHS在容差内不等于可安全倒退：第二行负步长会使第三行越界 -0.005。
        row=_leaving_row(np.array([1.,2.,-1e6]),np.array([0.,-1e-8,0.]),[1,2,3],1e-10,1e-7)
        self.assertEqual(row,0)

    def test_no_safe_pivot_is_reported_without_clipping(self):
        rhs=np.array([-1e-8,0.]); before=rhs.copy()
        row=_leaving_row(np.array([1.,-1e6]),rhs,[1,2],1e-10,1e-7)
        self.assertIsNone(row)
        np.testing.assert_array_equal(rhs,before)

    def test_full_uc_keeps_primal_feasibility_through_phase_one(self):
        # 不删原始约束、不改成本和容差；旧的近似比值平局会返回NUMERICAL_ERROR。
        case = json.loads((Path(__file__).resolve().parents[1] / 'examples/uc_24h.json').read_text(encoding='utf-8'))
        model, mapping = build_uc_model(case)
        result = model.solve('native', options=SolveOptions(time_limit=30, mip_gap=0,
            feasibility_tol=1e-7, integrality_tol=1e-7))
        self.assertEqual(result.status.value, 'OPTIMAL', result.termination_reason)
        self.assertAlmostEqual(result.objective, 28625, places=5)
        self.assertTrue(validate_candidate(model, result.values, 1e-7, 1e-7).is_feasible)
        physical = validate_uc(case, extract_uc_trace(mapping, result.values))
        self.assertTrue(physical['physical_pass'])
        self.assertAlmostEqual(physical['cost'], 28625, places=5)


if __name__ == '__main__':
    unittest.main()
