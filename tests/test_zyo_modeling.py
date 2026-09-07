# 回归测试：验证公共建模别名、批量变量和表达式数学含义。
import unittest

import zyo as zo


class ZYOModelingTests(unittest.TestCase):
    def test_canonical_and_compatibility_methods_build_same_model(self):
        m = zo.Model("api")
        x = m.add_var("x", lb=0, vtype="continuous")
        y = m.addVar(name="y", lb=0, vtype="BIN")
        m.add_constr(x + y <= 3, name="c0")
        m.addConstr(x >= 1, name="c1")
        m.maximize(2 * x + y)

        self.assertEqual([v.kind for v in m.variables], ["C", "B"])
        self.assertEqual(m.sense, "max")
        self.assertEqual([c.name for c in m.constraints], ["c0", "c1"])

    def test_add_vars_and_add_constrs_preserve_keys(self):
        m = zo.Model("indexed")
        x = m.add_vars(["a", "b"], lb=0, vtype=zo.INTEGER, name="x")
        rows = m.add_constrs((x[k] <= i + 1 for i, k in enumerate(x)), name="ub")
        m.minimize(zo.quicksum(x.values()))

        self.assertEqual(list(x), ["a", "b"])
        self.assertEqual([v.name for v in x.values()], ["x[a]", "x[b]"])
        self.assertEqual([r.name for r in rows], ["ub[0]", "ub[1]"])

    def test_invalid_type_is_rejected_before_mutating_model(self):
        m = zo.Model()

        with self.assertRaisesRegex(ValueError, "Unsupported variable type"):
            m.add_var(vtype="choice")

        self.assertEqual(m.variables, [])


if __name__ == "__main__":
    unittest.main()
