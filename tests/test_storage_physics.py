# 回归测试：用确定性教学预期与反例验证能量守恒、缺电和自放电。
import unittest
import numpy as np


class StoragePhysicsTests(unittest.TestCase):
    def test_teaching_rule_first_three_periods(self):
        from zyo_power.data import Battery, StorageCase
        from zyo_power.rule import rule_dispatch
        from zyo_power.validation import validate_dispatch
        net=np.repeat([5.,-2.,3.,-4.],6)
        case=StorageCase('teaching', np.maximum(-net,0),np.maximum(net,0),[Battery('battery',40,5,5)])
        e=[0.]
        for expected in [7.7052631579,14.7368421053,14.7368421053,14.7368421053]:
            result=rule_dispatch(case,e)
            audit=validate_dispatch(case,result,initial_energy=e)
            self.assertTrue(audit['physical_pass'])
            e=result['trace']['energy'][-1]
            self.assertAlmostEqual(e[0],expected,places=8)
        metrics=audit['metrics']
        self.assertAlmostEqual(metrics['charge_mwh'],39.8891966759,places=8)
        self.assertAlmostEqual(metrics['discharge_mwh'],36.,places=8)
        self.assertAlmostEqual(metrics['curtailment_mwh'],8.1108033241,places=8)
        self.assertAlmostEqual(metrics['ens_mwh'],0.,places=8)

    def test_independent_audit_catches_fake_energy_and_nan(self):
        from zyo_power.data import Battery, StorageCase
        from zyo_power.rule import rule_dispatch
        from zyo_power.validation import validate_dispatch
        case=StorageCase('tiny',[0,1],[1,0],[Battery('b',1,1,1,1,1)])
        result=rule_dispatch(case,[0])
        result['trace']['energy'][1,0]=0.5
        self.assertFalse(validate_dispatch(case,result)['physical_pass'])
        result['trace']['charge'][0,0]=np.nan
        self.assertFalse(validate_dispatch(case,result)['physical_pass'])

    def test_self_discharge_is_not_clipping(self):
        from zyo_power.data import Battery, StorageCase
        from zyo_power.rule import rule_dispatch
        from zyo_power.validation import validate_dispatch
        case=StorageCase('loss',[0,0],[0,0],[Battery('b',40,5,5)],retention=.9,dt=2.)
        result=rule_dispatch(case,[20.])
        np.testing.assert_allclose(result['trace']['energy'][:,0],[20,18,16.2])
        audit=validate_dispatch(case,result)
        self.assertTrue(audit['physical_pass'])
        self.assertAlmostEqual(audit['metrics']['standing_loss_mwh'],3.8)

    def test_inputs_reject_nonfinite_and_window_wraps(self):
        from zyo_power.data import Battery, StorageCase
        battery=Battery('b',4,1,1)
        with self.assertRaises(ValueError): StorageCase('bad',[1,np.nan],[1,2],[battery])
        case=StorageCase('wrap',[1,2,3],[0,0,0],[battery])
        np.testing.assert_equal(case.window(2,5).load,[3,1,2,3,1])

    def test_ample_total_energy_can_fail_power_or_capacity_limits(self):
        from zyo_power.data import Battery, StorageCase
        from zyo_power.rule import rule_dispatch
        from zyo_power.validation import validate_dispatch
        for battery in [Battery('power_limited',40,1,5,1,1),
                        Battery('energy_limited',1,5,5,1,1)]:
            case=StorageCase('counterexample',[0,4],[10,0],[battery])
            audit=validate_dispatch(case,rule_dispatch(case,[0]))
            self.assertTrue(audit['physical_pass'])
            self.assertTrue(audit['cycle_closed'])
            self.assertFalse(audit['supply_adequate'])
            self.assertAlmostEqual(audit['metrics']['ens_mwh'],3.)

    def test_nonunit_duration_and_two_batteries_energy_accounting(self):
        from zyo_power.data import Battery, StorageCase
        from zyo_power.rule import rule_dispatch
        from zyo_power.validation import validate_dispatch
        case=StorageCase('two',[0,4],[8,0],
                         [Battery('a',3,2,2,.8,.9),Battery('b',4,3,3,.9,.8)],dt=[.5,1.5])
        result=rule_dispatch(case,[0,0])
        audit=validate_dispatch(case,result)
        self.assertTrue(audit['physical_pass'])
        np.testing.assert_allclose(result['trace']['energy'][1],[.8,1.35])
        self.assertAlmostEqual(audit['metrics']['global_energy_accounting_residual_mwh'],0.)


if __name__=='__main__': unittest.main()
