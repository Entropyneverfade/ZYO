# GridMind 调用 ZYO 的公共契约与年度状态路径最小例测试。
import importlib.util
import json
from pathlib import Path
import unittest

import zyo as zo
from zyo import ModelContract, ResultContract, engine_contract
from zyo.contract import engine_support


def annual_example():
    path = Path(__file__).resolve().parents[1] / "examples" / "annual_state_path.py"
    spec = importlib.util.spec_from_file_location("zyo_annual_state_path", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PublicSolverContractTests(unittest.TestCase):
    def test_model_contract_is_serializable_and_survives_json_roundtrip(self):
        model = zo.Model("contract-roundtrip")
        x = model.add_var("x", lb=0.0, ub=2.0)
        model.add_constr(x >= 1.0, name="lower")
        model.minimize(3.0 * x + 4.0)
        metadata = {"units": {"x": "MW", "objective": "CNY"}, "source": "hand-written"}
        contract = ModelContract.from_model(model, metadata)
        self.assertEqual(contract.schema, "zyo-model-contract-1")
        self.assertEqual(len(contract.fingerprint), 64)
        payload = json.dumps(contract.to_dict(), ensure_ascii=False, allow_nan=False)
        restored = zo.Model.from_dict(json.loads(payload)["model"])
        self.assertEqual(ModelContract.from_model(restored, metadata).fingerprint,
                         contract.fingerprint)

    def test_metadata_and_engine_requirements_change_model_identity_or_support(self):
        model = zo.Model("integer")
        model.add_var("x", lb=0.0, ub=1.0, vtype="B")
        self.assertFalse(engine_support(model, "native_simplex")["supported"])
        self.assertTrue(engine_support(model, "native_sparse")["supported"])
        contract = ModelContract.from_model(model, {"case": 1})
        changed = ModelContract.from_model(model, {"case": 2})
        self.assertNotEqual(contract.fingerprint, changed.fingerprint)

        free = zo.Model("free")
        free.add_var("x", lb=None, ub=1.0)
        sparse = engine_support(free, "native_sparse")
        self.assertFalse(sparse["supported"])
        self.assertIn("finite lower and upper bounds", "; ".join(sparse["reasons"]))
        self.assertTrue(engine_support(free, "native_simplex")["supported"])
        self.assertIn("finite-box", engine_contract("native_sparse")["model_type"])

    def test_annual_state_path_matches_independent_enumeration_and_contract(self):
        example = annual_example()
        payload = example.example_payload()
        model, result, model_receipt, result_receipt, mapping = example.solve(payload)
        expected, expected_path = example.enumerate_best_path(payload)
        self.assertEqual(expected_path, (0, 0, 3))
        self.assertEqual(example.selected_states(payload, result), expected_path)
        self.assertEqual(result.status, zo.Status.OPTIMAL)
        self.assertLess(abs(result.objective - expected), 1e-8)
        self.assertLess(abs(result.best_bound - expected), 1e-8)
        self.assertLess(result.mip_gap, 1e-9)
        audit = example.audit_result(payload, model, result, model_receipt)
        self.assertTrue(audit["objective_matches_enumeration"])
        self.assertEqual(audit["max_onehot_violation"], 0.0)
        self.assertEqual(audit["max_monotonic_violation"], 0.0)
        self.assertEqual(audit["bound_direction_error_yuan"], 0.0)
        interpretation = result_receipt.to_dict()["interpretation"]
        self.assertTrue(interpretation["proved_optimal"])
        self.assertTrue(interpretation["global_bound_available"])
        self.assertTrue(interpretation["candidate_residuals_available"])
        self.assertEqual(result_receipt.model_fingerprint, model_receipt.fingerprint)

    def test_state_path_mapping_and_effective_parameters_are_reported(self):
        example = annual_example()
        payload = example.example_payload()
        _, result, model_receipt, result_receipt, mapping = example.solve(payload)
        self.assertEqual(mapping["schema"], "zyo-representation-mapping-1")
        self.assertEqual(mapping["source_model_sha256"], example.payload_sha256(payload))
        self.assertEqual(mapping["zyo_model_fingerprint"], model_receipt.fingerprint)
        self.assertEqual(mapping["matrix"]["rows"], 7)
        self.assertEqual(mapping["matrix"]["columns"], 12)
        self.assertEqual(mapping["matrix"]["constraint_nonzeros"], 28)
        self.assertEqual(mapping["matrix"]["binary_columns"], 12)
        receipt = result_receipt.to_dict()
        self.assertIn("time_limit", receipt["applied_options"])
        self.assertIn("threads", receipt["unmapped_options"])
        self.assertIn("seed", receipt["unmapped_options"])
        self.assertIn("parameters_applied", receipt["interpretation"]["parameter_semantics"])

    def test_infeasible_and_node_limit_results_stay_unproved(self):
        for name, kwargs in (("infeasible", {}), ("node-limit", {"NodeLimit": 0})):
            with self.subTest(name=name):
                model = zo.Model(name)
                x = model.add_var("x", lb=0.0, ub=1.0, vtype="B")
                model.add_constr(x >= 1.0, name="force")
                model.add_constr(x <= 0.0, name="block")
                model.minimize(x)
                result = model.solve("native_sparse", TimeLimit=3, **kwargs)
                receipt = ResultContract.from_result(
                    result, ModelContract.from_model(model, {"case": name}))
                payload = receipt.to_dict()
                self.assertIn(result.status, (zo.Status.INFEASIBLE, zo.Status.NODE_LIMIT))
                self.assertFalse(payload["interpretation"]["proved_optimal"])
                self.assertFalse(payload["interpretation"]["global_bound_available"])
                self.assertIsNone(payload["result"]["objective"])
                self.assertIsNone(payload["result"]["best_bound"])
                self.assertIsNone(payload["result"]["mip_gap"])

    def test_result_contract_rejects_ambiguous_fingerprint(self):
        model = zo.Model("result-contract")
        model.add_var("x", lb=0.0, ub=1.0)
        model.minimize(model.variables[0])
        result = model.solve("native_simplex", TimeLimit=3)
        with self.assertRaisesRegex(TypeError, "fingerprint"):
            ResultContract.from_result(result, "not-a-fingerprint")


if __name__ == "__main__":
    unittest.main()
