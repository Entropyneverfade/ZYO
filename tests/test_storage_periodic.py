# 回归测试：检查状态传递、闭合确认、初始化影响及不收敛反例。
import unittest
import numpy as np


class PeriodicTests(unittest.TestCase):
    def test_float_endpoint_is_carried_without_clipping_or_spurious_failure(self):
        from zyo_power.data import Battery, StorageCase
        from zyo_power.rule import rule_dispatch
        from zyo_power.periodic import iterate_periodic
        case=StorageCase('endpoint',[2],[0],[Battery('b',1,2,2,.95,.6598923271591431)])
        result=iterate_periodic(case,lambda e:rule_dispatch(case,e),[.2431612927331097])
        self.assertEqual(result['status'],'CONVERGED')
        first_end=result['history'][0]['end_energy_mwh'][0]
        self.assertLess(first_end,0)
        self.assertEqual(result['history'][1]['start_energy_mwh'][0],first_end)
        self.assertFalse(result['formal_audit']['supply_adequate'])

    def test_teaching_initializations_and_confirmation(self):
        from zyo_power.data import Battery,StorageCase
        from zyo_power.rule import rule_dispatch
        from zyo_power.periodic import iterate_periodic,initialization_study
        net=np.repeat([5.,-2.,3.,-4.],6)
        case=StorageCase('teaching',np.maximum(-net,0),np.maximum(net,0),[Battery('b',40,5,5)])
        study=initialization_study(case,lambda e:rule_dispatch(case,e))
        self.assertEqual(study['initialization_status'],'SAME_OBSERVED_CYCLE')
        for run in study['runs']:
            self.assertEqual(run['status'],'CONVERGED')
            self.assertAlmostEqual(run['formal']['trace']['energy'][-1,0],14.7368421053,places=8)
        self.assertEqual(len(study['runs'][0]['history']),4)

    def test_closed_empty_battery_is_not_adequacy(self):
        from zyo_power.data import Battery,StorageCase
        from zyo_power.rule import rule_dispatch
        from zyo_power.periodic import iterate_periodic
        case=StorageCase('short',[1]*24,[0]*24,[Battery('b',40,5,5)])
        result=iterate_periodic(case,lambda e:rule_dispatch(case,e),[0])
        self.assertEqual(result['status'],'CONVERGED')
        self.assertFalse(result['formal_audit']['supply_adequate'])
        self.assertEqual(result['formal_audit']['metrics']['ens_mwh'],24.)
        declining=iterate_periodic(case,lambda e:rule_dispatch(case,e),[40],max_cycles=1)
        self.assertEqual(declining['status'],'NOT_CONVERGED')
        strict=iterate_periodic(case,lambda e:rule_dispatch(case,e,strict=True),[0],strict=True)
        self.assertEqual(strict['status'],'SUBPROBLEM_INFEASIBLE')

    def test_multiple_fixed_points_are_reported(self):
        from zyo_power.data import Battery,StorageCase
        from zyo_power.rule import rule_dispatch
        from zyo_power.periodic import initialization_study
        case=StorageCase('idle',[0],[0],[Battery('b',40,5,5)])
        study=initialization_study(case,lambda e:rule_dispatch(case,e))
        self.assertEqual(study['initialization_status'],'MULTIPLE_OBSERVED_CYCLES')

    def test_suspected_two_cycle_uses_real_feasible_policy(self):
        from zyo_power.data import Battery,StorageCase
        from zyo_power.periodic import iterate_periodic
        case=StorageCase('two-cycle',[1],[1],[Battery('b',1,1,1,1,1)])
        def policy(e):
            charge=float(e[0]<.5); discharge=1-charge
            return dict(status='COMPLETED',engine='test_rule',mode='test',metadata={},trace=dict(
                energy=np.array([[e[0]],[1-e[0]]]),charge=np.array([[charge]]),discharge=np.array([[discharge]]),
                thermal=np.array([charge]),curtailment=np.array([discharge]),shed=np.zeros(1)))
        # Conventional 1 MW makes both states physically feasible.
        case=StorageCase('two-cycle',[1],[1],case.batteries,thermal_capacity=1)
        result=iterate_periodic(case,policy,[0])
        self.assertEqual(result['status'],'SUSPECTED_MULTI_PERIOD_CYCLE')


if __name__=='__main__': unittest.main()
