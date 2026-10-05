"""公开手算反例：微小负比值不能诱导奇异的零步换基。"""

import math
import unittest

import numpy as np

from zyo.sparse_simplex import NumericalError, _ratio_limit_scalar, _ratio_limit_vectorized


class PublicNegativeRatioTests(unittest.TestCase):
    def setUp(self):
        # x0 略低于零界且方向极小；x1 上升，在步长 2 触及其上界。
        self.basic = [0, 1]
        self.values = np.array([-1e-30, 0.0])
        self.lower = np.array([0.0, -math.inf])
        self.upper = np.array([math.inf, 2.0])
        self.moving = np.array([1e-17, -1.0])

    def test_zero_feasibility_tolerance_rejects_negative_room(self):
        for routine in (_ratio_limit_scalar, _ratio_limit_vectorized):
            with self.subTest(routine=routine.__name__), self.assertRaises(NumericalError):
                routine(self.basic, self.moving, self.values, self.lower,
                        self.upper, 1e-9, feasibility_tolerance=0.0)

    def test_next_finite_blocker_is_safe_only_within_original_tolerance(self):
        # alpha=2 时 x0=-1e-30-2e-17，仍小于独立设定的 1e-7 可行门。
        for routine in (_ratio_limit_scalar, _ratio_limit_vectorized):
            with self.subTest(routine=routine.__name__):
                self.assertEqual(
                    routine(self.basic, self.moving, self.values, self.lower,
                            self.upper, 1e-9, feasibility_tolerance=1e-7),
                    (2.0, 1, True))

    def test_bad_start_cannot_be_masked(self):
        values = np.array([-1e-4, 0.0])
        for routine in (_ratio_limit_scalar, _ratio_limit_vectorized):
            with self.subTest(routine=routine.__name__), self.assertRaises(NumericalError):
                routine(self.basic, self.moving, values, self.lower,
                        self.upper, 1e-9, feasibility_tolerance=1e-7)

    def test_material_violation_at_next_blocker_is_rejected(self):
        # 方向 0.01 将使 x0 在 alpha=2 时越界 0.02，不能视为舍入噪声。
        moving = np.array([0.01, -1.0])
        for routine in (_ratio_limit_scalar, _ratio_limit_vectorized):
            with self.subTest(routine=routine.__name__), self.assertRaises(NumericalError):
                routine(self.basic, moving, self.values, self.lower,
                        self.upper, 1e-9, feasibility_tolerance=1e-7)

    def test_no_other_finite_blocker_does_not_prove_a_ray(self):
        upper = np.array([math.inf, math.inf])
        for routine in (_ratio_limit_scalar, _ratio_limit_vectorized):
            with self.subTest(routine=routine.__name__), self.assertRaises(NumericalError):
                routine(self.basic, self.moving, self.values, self.lower,
                        upper, 1e-9, feasibility_tolerance=1e-7)


if __name__ == '__main__':
    unittest.main()
