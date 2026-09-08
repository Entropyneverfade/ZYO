# 机组组合独立预期：手算成本与主动破坏物理约束，避免只检查求解器状态。
import copy
import importlib.util
import unittest


class UnitCommitmentTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('zyo_power.unit_commitment'),
                             '缺少独立机组组合模块')
        from zyo_power.unit_commitment import build_uc_model, extract_uc_trace, validate_uc, validate_uc_input
        self.build, self.extract, self.check, self.input_check = build_uc_model, extract_uc_trace, validate_uc, validate_uc_input

    def case(self):
        # 四小时手算：出力 3,5,5,3；一次启动 7、燃料 32、空载 4，总成本 43。
        return dict(name='hand_uc', demand=[3, 5, 5, 3], wind=[0]*4, solar=[0]*4,
                    reserve=[0]*4, strict=True, ens_cost=10000,
                    units=[dict(name='G1', p_min=2, p_max=6, marginal_cost=2,
                                no_load_cost=1, startup_cost=7, shutdown_cost=3,
                                ramp_up=2, ramp_down=2, startup_ramp=3, shutdown_ramp=3,
                                min_up=3, min_down=2, initial_on=0, initial_duration=2,
                                initial_power=0)])

    def test_hand_cost_and_initial_start(self):
        case = self.case()
        model, mapping = self.build(case)
        result = model.solve(solver='native', time_limit=10)
        self.assertEqual(result.status.value, 'OPTIMAL')
        self.assertAlmostEqual(result.objective, 43)
        trace = self.extract(mapping, result.values)
        audit = self.check(case, trace)
        self.assertTrue(audit['physical_pass'], audit)
        self.assertAlmostEqual(audit['cost'], 43)
        for observed, expected in zip(trace['start'][0], [1, 0, 0, 0]):
            self.assertAlmostEqual(observed, expected, places=7)

    def test_checker_rejects_balance_transition_and_nonfinite(self):
        case = self.case()
        trace = dict(power=[[3,5,5,3]], on=[[1]*4], start=[[1,0,0,0]],
                     stop=[[0]*4], wind=[0]*4, solar=[0]*4, ens=[0]*4)
        self.assertTrue(self.check(case, trace)['physical_pass'])
        for field, index, value in [('power', 1, 6), ('start', 1, 1), ('on', 1, .5)]:
            bad = copy.deepcopy(trace)
            bad[field][0][index] = value
            self.assertFalse(self.check(case, bad)['physical_pass'])
        trace['power'][0][0] = float('nan')
        with self.assertRaises(ValueError):
            self.check(case, trace)

    def test_initial_min_down_enforced_and_diagnostic_ens(self):
        case = self.case()
        case['units'][0]['initial_duration'] = 1
        model, _ = self.build(case)
        self.assertEqual(model.solve(solver='native').status.value, 'INFEASIBLE')
        case['strict'] = False
        model, mapping = self.build(case)
        result = model.solve(solver='native')
        self.assertEqual(result.status.value, 'OPTIMAL')
        audit = self.check(case, self.extract(mapping, result.values))
        self.assertTrue(audit['physical_pass'])
        self.assertGreater(audit['ens_mwh'], 0)
        self.assertFalse(audit['supply_satisfied'])

    def test_checker_min_up_reserve_and_ramps(self):
        case = self.case()
        # 启动一小时后就停机：功率平衡满足，但违反最小开机时间。
        case['demand'] = [3, 0, 0, 0]
        trace = dict(power=[[3,0,0,0]], on=[[1,0,0,0]], start=[[1,0,0,0]],
                     stop=[[0,1,0,0]], wind=[0]*4, solar=[0]*4, ens=[0]*4)
        audit = self.check(case, trace)
        self.assertGreater(audit['residuals']['min_up_down'], 0)
        case = self.case()
        trace.update(power=[[3,5,5,3]], on=[[1]*4], stop=[[0]*4])
        case['reserve'] = [4]*4
        self.assertGreater(self.check(case, trace)['residuals']['reserve'], 0)
        case['reserve'] = [0]*4
        case['units'][0]['ramp_up'] = 1
        self.assertGreater(self.check(case, trace)['residuals']['ramp'], 0)

    def test_shutdown_min_down_and_initial_on_checks(self):
        case = self.case()
        unit = case['units'][0]
        unit.update(initial_on=1, initial_power=3, initial_duration=1)
        case['demand'] = [0, 3, 3, 3]
        trace = dict(power=[[0,3,3,3]], on=[[0,1,1,1]], start=[[0,1,0,0]],
                     stop=[[1,0,0,0]], wind=[0]*4, solar=[0]*4, ens=[0]*4)
        audit = self.check(case, trace)
        self.assertGreater(audit['residuals']['initial_state'], 0)
        self.assertGreater(audit['residuals']['min_up_down'], 0)

    def test_invalid_input_rejected(self):
        for change in [dict(min_up=1.5), dict(p_min=7), dict(initial_power=1),
                       dict(ramp_up=float('inf')), dict(initial_duration=-1)]:
            case = self.case()
            case['units'][0].update(change)
            with self.assertRaises(ValueError):
                self.input_check(case)
        case = self.case()
        case['wind'] = [0]
        with self.assertRaises(ValueError):
            self.input_check(case)

    def one_hour(self):
        case = self.case()
        case.update(demand=[100], wind=[0], solar=[0], reserve=[0])
        case['units'][0].update(p_min=100, p_max=100, initial_on=1, initial_power=100,
                                initial_duration=1, min_up=1, min_down=1, shutdown_ramp=200)
        return case

    def test_absolute_negative_power_bound_not_scaled_by_near_binary(self):
        case = self.one_hour(); case.update(demand=[1],wind=[2])
        trace = dict(power=[[-5e-6]],on=[[-5e-8]],start=[[0]],stop=[[1]],
                     wind=[1.000005],solar=[0],ens=[0])
        audit = self.check(case,trace)
        self.assertFalse(audit['physical_pass'])
        self.assertAlmostEqual(audit['residuals']['bound'],5e-6)

    def test_absolute_upper_power_bound_not_scaled_by_near_binary(self):
        case = self.one_hour(); case['demand']=[100.000005]
        trace = dict(power=[[100.000005]],on=[[1.00000005]],start=[[0]],stop=[[0]],
                     wind=[0],solar=[0],ens=[0])
        audit = self.check(case,trace)
        self.assertFalse(audit['physical_pass'])
        self.assertAlmostEqual(audit['residuals']['bound'],5e-6)

    def test_finite_inputs_with_overflowing_derived_cost_rejected(self):
        case = self.one_hour(); case['demand']=[10]
        case['units'][0].update(p_min=0,initial_power=10,marginal_cost=1e308)
        trace = dict(power=[[10]],on=[[1]],start=[[0]],stop=[[0]],wind=[0],solar=[0],ens=[0])
        with self.assertRaisesRegex(ValueError,'成本|cost'):
            self.check(case,trace)

    def test_late_start_carries_remaining_minimum_up_time(self):
        case = self.case(); case['demand']=[0,0,0,3]
        trace = dict(power=[[0,0,0,3]],on=[[0,0,0,1]],start=[[0,0,0,1]],stop=[[0]*4],
                     wind=[0]*4,solar=[0]*4,ens=[0]*4)
        audit = self.check(case,trace)
        self.assertTrue(audit['physical_pass'])
        self.assertEqual(audit['terminal_state'],[dict(name='G1',on=1,power=3,duration=1,remaining_obligation_hours=2)])

    def test_feasible_initial_on_needs_no_new_startup(self):
        case = self.case(); case['units'][0].update(initial_on=1,initial_power=3,initial_duration=1)
        trace = dict(power=[[3,5,5,3]],on=[[1]*4],start=[[0]*4],stop=[[0]*4],
                     wind=[0]*4,solar=[0]*4,ens=[0]*4)
        audit = self.check(case,trace)
        self.assertTrue(audit['physical_pass'])
        self.assertEqual(audit['cost'],36)


if __name__ == '__main__':
    unittest.main()
