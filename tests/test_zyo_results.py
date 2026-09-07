# 回归测试：检查不可变结果、字段映射和无解时的访问保护。
from dataclasses import FrozenInstanceError
import unittest

import zyo as zo


class ZYOResultTests(unittest.TestCase):
    def test_solve_accepts_casefolded_solver_and_parameter_aliases(self):
        m = zo.Model("result")
        x = m.add_var("x", ub=2)
        m.maximize(x)

        result = m.solve(solver="NATIVE", TimeLimit=5, MIPGap=0)

        self.assertEqual(result.status, zo.Status.OPTIMAL)
        self.assertEqual(result.objective, 2.0)
        self.assertEqual(result.mip_gap, 0.0)
        self.assertEqual(result.solver_name, "native")
        self.assertEqual(x.x, 2.0)
        self.assertEqual(m.obj_value, 2.0)

    def test_optimize_is_an_alias_for_solve(self):
        m = zo.Model()
        x = m.add_var(ub=1)
        m.minimize(-x)

        result = m.optimize()

        self.assertEqual(result.objective, -1.0)
        self.assertIs(result.status, zo.Status.OPTIMAL)

    def test_solution_value_is_unavailable_before_or_without_solution(self):
        m = zo.Model()
        x = m.add_var()

        with self.assertRaises(zo.SolutionUnavailableError):
            _ = x.x

        m.add_constr(x >= 2)
        m.add_constr(x <= 1)
        m.solve()
        with self.assertRaises(zo.SolutionUnavailableError):
            _ = x.x

    def test_unknown_parameter_is_rejected(self):
        with self.assertRaises(zo.InvalidParameterError):
            zo.SolveOptions.from_kwargs(MagicSpeed=10)

    def test_invalid_option_values_are_rejected_at_the_public_boundary(self):
        invalid = [
            ({"time_limit": -1}, "time_limit"),
            ({"feasibility_tol": 0}, "feasibility_tol"),
            ({"node_limit": 1.5}, "node_limit"),
            ({"threads": 0}, "threads"),
            ({"presolve": 1}, "presolve"),
        ]
        for kwargs, field_name in invalid:
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(zo.InvalidParameterError, field_name):
                    zo.SolveOptions(**kwargs)

    def test_result_and_solution_values_are_immutable(self):
        m = zo.Model()
        x = m.add_var(ub=1)
        m.maximize(x)
        result = m.solve()

        with self.assertRaises(FrozenInstanceError):
            result.objective = 0
        with self.assertRaises(TypeError):
            result.values["x0"] = 0
        with self.assertRaises(TypeError):
            result.metadata["parameters"]["time_limit"] = 0


if __name__ == "__main__":
    unittest.main()
