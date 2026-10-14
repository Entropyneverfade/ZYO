"""按 GRIDMIND_STATE_PATH_MILP_V1 演示 ZYO 通用契约的纯合成示例。

领域费用表契约由 GridMind 维护；本文件只原样携带一份 V1 payload，翻译成
同一变量/行列命名，并输出 ZYO 模型指纹、矩阵映射和参数实际应用回执。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import zyo as zo
from zyo import ModelContract, ResultContract, quicksum, representation_mapping
from zyo.contract import engine_support


def example_payload():
    """固定纯合成 V1 费用表；无私有项目数据。"""
    return {
        "schema_version": "GRIDMIND_STATE_PATH_MILP_V1",
        "candidate_count": 2,
        "discount_rate": 0.10,
        "stages": [
            {"year": 2030, "states": [
                {"state": 0, "cost_yuan": 10.0, "feasible": True},
                {"state": 1, "cost_yuan": 20.0, "feasible": True},
                {"state": 2, "cost_yuan": 16.0, "feasible": True},
                {"state": 3, "cost_yuan": 24.0, "feasible": True},
            ]},
            {"year": 2031, "states": [
                {"state": 0, "cost_yuan": 12.0, "feasible": True},
                {"state": 1, "cost_yuan": 14.0, "feasible": True},
                {"state": 2, "cost_yuan": 15.0, "feasible": True},
                {"state": 3, "cost_yuan": 20.0, "feasible": True},
            ]},
            {"year": 2032, "states": [
                {"state": 0, "cost_yuan": 20.0, "feasible": True},
                {"state": 1, "cost_yuan": 10.0, "feasible": True},
                {"state": 2, "cost_yuan": 16.0, "feasible": True},
                {"state": 3, "cost_yuan": 8.0, "feasible": True},
            ]},
        ],
    }


def payload_sha256(payload):
    # 与 GridMind V1 相同的排序键、紧凑分隔符、UTF-8 和非有限数拒绝。
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _minimal_v1_guard(payload):
    # 完整字段/位序/费用校验由 GridMind state_path_contract.validate_spec 独占。
    # 这里只防止翻译器在未收到 V1 时猜模型。
    if (not isinstance(payload, dict)
            or payload.get("schema_version") != "GRIDMIND_STATE_PATH_MILP_V1"):
        raise ValueError("Expected GRIDMIND_STATE_PATH_MILP_V1 payload")
    if not isinstance(payload.get("stages"), list) or len(payload["stages"]) < 2:
        raise ValueError("State-path payload has no stages to translate")


def build_model(payload):
    _minimal_v1_guard(payload)
    model = zo.Model("gridmind_state_path")
    variables = {}
    for index, stage in enumerate(payload["stages"]):
        for row in stage["states"]:
            if row["feasible"]:
                variables[(index, row["state"])] = model.add_var(
                    f"z_{index}_{row['state']}", lb=0.0, ub=1.0, vtype="B"
                )
    for index, stage in enumerate(payload["stages"]):
        model.add_constr(
            quicksum(variables[(index, row["state"])]
                     for row in stage["states"] if row["feasible"]) == 1.0,
            name=f"one_state_{index}",
        )
    for index in range(1, len(payload["stages"])):
        previous = [row["state"] for row in payload["stages"][index - 1]["states"]
                    if row["feasible"]]
        current = [row["state"] for row in payload["stages"][index]["states"]
                   if row["feasible"]]
        for bit in range(payload["candidate_count"]):
            before = quicksum(variables[(index - 1, state)]
                              for state in previous if state & (1 << bit))
            after = quicksum(variables[(index, state)]
                             for state in current if state & (1 << bit))
            model.add_constr(before <= after, name=f"monotonic_{index}_{bit}")
    model.minimize(quicksum(
        row["cost_yuan"] / (1.0 + payload["discount_rate"]) ** index
        * variables[(index, row["state"])]
        for index, stage in enumerate(payload["stages"])
        for row in stage["states"] if row["feasible"]
    ))
    return model, variables


def enumerate_best_path(payload):
    rows = []

    def visit(index, previous, path, cost):
        if index == len(payload["stages"]):
            rows.append((cost, tuple(path)))
            return
        for row in payload["stages"][index]["states"]:
            state = row["state"]
            if row["feasible"] and (previous & state) == previous:
                factor = (1.0 + payload["discount_rate"]) ** (-index)
                visit(index + 1, state, path + [state],
                      cost + factor * row["cost_yuan"])

    visit(0, 0, [], 0.0)
    if not rows:
        raise ValueError("State-path payload has no feasible monotonic path")
    return min(rows, key=lambda item: (item[0], item[1]))


def selected_states(payload, result):
    states = []
    for index, stage in enumerate(payload["stages"]):
        selected = [
            row["state"] for row in stage["states"]
            if row["feasible"]
            and result.values[f"z_{index}_{row['state']}"] > 0.5
        ]
        if len(selected) != 1:
            raise RuntimeError(f"stage {index} does not select exactly one state")
        states.append(selected[0])
    return tuple(states)


def discounted_cost(payload, path):
    return math.fsum(
        row["cost_yuan"] / (1.0 + payload["discount_rate"]) ** index
        for index, (stage, state) in enumerate(zip(payload["stages"], path))
        for row in stage["states"] if row["state"] == state
    )


def audit_result(payload, model, result, contract):
    expected, expected_path = enumerate_best_path(payload)
    path = selected_states(payload, result)
    objective = discounted_cost(payload, path)
    onehot = max(
        abs(sum(result.values[f"z_{index}_{row['state']}"]
                for row in stage["states"] if row["feasible"]) - 1.0)
        for index, stage in enumerate(payload["stages"])
    )
    monotonic = max(
        0.0 if (previous & state) == previous else 1.0
        for previous, state in zip(path, path[1:])
    )
    return {
        "source_model_sha256": payload_sha256(payload),
        "zyo_model_fingerprint": contract.fingerprint,
        "selected_states": list(path),
        "enumeration_selected_states": list(expected_path),
        "recomputed_objective_yuan": objective,
        "enumeration_objective_yuan": expected,
        "max_onehot_violation": onehot,
        "max_monotonic_violation": monotonic,
        "bound_direction_error_yuan": max(0.0, result.best_bound - objective),
        "objective_matches_enumeration": abs(objective - expected) <= 1e-8,
    }


def solve(payload=None, engine="native_sparse"):
    payload = example_payload() if payload is None else payload
    model, _ = build_model(payload)
    support = engine_support(model, engine)
    if not support["supported"]:
        raise ValueError("; ".join(support["reasons"]))
    contract = ModelContract.from_model(model, payload)
    result = model.solve(engine, time_limit=10.0, node_limit=10_000,
                         iteration_limit=100_000, mip_gap=0.0, threads=1)
    receipt = ResultContract.from_result(result, contract)
    mapping = representation_mapping(contract, payload_sha256(payload))
    return model, result, contract, receipt, mapping


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("native_sparse",), default="native_sparse")
    parser.add_argument("--output", type=Path, help="optional JSON receipt path")
    args = parser.parse_args(argv)
    payload = example_payload()
    model, result, contract, receipt, mapping = solve(payload, args.engine)
    payload_out = {
        "source_model": payload,
        "model_contract": contract.to_dict(),
        "representation_mapping": mapping,
        "result_contract": receipt.to_dict(),
        "audit": audit_result(payload, model, result, contract),
    }
    text = json.dumps(payload_out, ensure_ascii=False, indent=2,
                      allow_nan=False) + "\n"
    if args.output is None:
        sys.stdout.write(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
