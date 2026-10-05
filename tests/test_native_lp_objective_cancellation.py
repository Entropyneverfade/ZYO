"""公共最优目标必须与原存储数的已核证目标一致。"""

from fractions import Fraction
import unittest

from scipy.sparse import csc_matrix

from zyo.native_lp import solve_certified_lp


class NativeLPObjectiveCancellationTests(unittest.TestCase):
    def test_verified_optimum_uses_independently_recomputed_objective(self):
        # 两个变量固定，模型的唯一可行点必为最优；二进制浮点点积先舍入成 0。
        first = 1151660734277.0
        second = -1421803362854.3792
        point = [1.23456789012345, 1.0]
        expected = (Fraction.from_float(first)*Fraction.from_float(point[0])
                    + Fraction.from_float(second)*Fraction.from_float(point[1]))
        self.assertNotEqual(expected, 0)
        result = solve_certified_lp(csc_matrix((0, 2)), [first, second],
                                    point, point, [])
        self.assertEqual(result.status, 'OPTIMAL')
        self.assertTrue(result.optimal)
        self.assertTrue(result.certificate.verified)
        self.assertEqual(result.objective, float(expected))
        self.assertEqual(result.objective, result.certificate.objective)


if __name__ == '__main__':
    unittest.main()
