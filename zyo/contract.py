"""GridMind 调用 ZYO 的版本化模型/结果契约。

契约只做身份、序列化和语义解释，不参与建模、求解或证书生成。模型指纹是对
规范 JSON 字节的 SHA-256：同一变量顺序、约束顺序、系数、边界和来源元数据得到
同一身份；代数等价但序列不同的模型不承诺同指纹。
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MappingProxyType

from .options import SolveOptions
from .status import Status


MODEL_CONTRACT_SCHEMA = "zyo-model-contract-1"
RESULT_CONTRACT_SCHEMA = "zyo-result-contract-1"
FINGERPRINT_ALGORITHM = "sha256"


ENGINE_CONTRACTS = {
    # 连续 LP 路径：整数模型必须显式拒绝；证书和预算语义来自 Result 元数据。
    "native_simplex": {
        "model_type": "LP",
        "integer_variables": False,
        "integration_requires_finite_bounds": False,
        "objective": "continuous original-model optimum when status is OPTIMAL",
        "bound_and_gap": "LP dual certificate; MIP best_bound/mip_gap are not applicable",
    },
    # GridMind 集成入口：本契约要求所有变量有有限上下界。求解器内部的非有限域
    # 试验是另一条能力路线，不能在上层悄悄加人造界来冒充原模型。
    "native_sparse": {
        "model_type": "finite-box LP/MILP",
        "integer_variables": True,
        "integration_requires_finite_bounds": True,
        "objective": "original finite-box objective when status is OPTIMAL",
        "bound_and_gap": "best_bound is a checked box-dual bound; gap uses max(1,abs(objective))",
    },
}


def _freeze(value):
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


def _thaw(value):
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in value]
    return value


def _canonical_bytes(value):
    return json.dumps(_thaw(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha256(value):
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


@dataclass(frozen=True)
class ModelContract:
    """规范模型、应用来源元数据和指纹的不可变包装。"""

    schema: str
    model: Mapping
    metadata: Mapping
    fingerprint: str

    def __post_init__(self):
        object.__setattr__(self, "model", _freeze(self.model))
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    @classmethod
    def from_model(cls, model, metadata=None):
        payload = {
            "schema": MODEL_CONTRACT_SCHEMA,
            "model": model.to_dict(),
            "metadata": {} if metadata is None else dict(metadata),
        }
        digest = _sha256(payload)
        return cls(payload["schema"], payload["model"], payload["metadata"], digest)

    def to_dict(self):
        return {
            "schema": self.schema,
            "model": _thaw(self.model),
            "metadata": _thaw(self.metadata),
            "fingerprint": self.fingerprint,
            "fingerprint_algorithm": FINGERPRINT_ALGORITHM,
        }


def model_contract(model, metadata=None):
    return ModelContract.from_model(model, metadata)


def model_fingerprint(model, metadata=None):
    return ModelContract.from_model(model, metadata).fingerprint


def matrix_identity(model):
    """返回求解器矩阵的维度与结构摘要，用于同模翻译映射核对。

    该摘要不是完整数学模型身份；完整身份仍由 ModelContract.fingerprint 和
    上游领域模型指纹共同组成。这里显式给出行列、非零、整数与边界结构，便于
    GridMind 在翻译前发现行列错位、漏行或 silently 改变量类型。
    """

    variables = list(model.variables)
    constraints = list(model.constraints)
    payload = {
        "rows": len(constraints),
        "columns": len(variables),
        "constraint_nonzeros": sum(len(row.expression.terms) for row in constraints),
        "objective_nonzeros": len(model.objective.terms),
        "objective_constant": float(model.objective.constant),
        "objective_sense": model.sense,
        "integer_columns": sum(variable.kind != "C" for variable in variables),
        "binary_columns": sum(variable.kind == "B" for variable in variables),
        "finite_lower_columns": sum(math.isfinite(float(variable.lb)) for variable in variables),
        "finite_upper_columns": sum(math.isfinite(float(variable.ub)) for variable in variables),
    }
    return {**payload, "matrix_fingerprint": _sha256(payload)}


def representation_mapping(model_contract, source_model_sha256):
    """把上游领域模型指纹、ZYO 完整模型指纹和矩阵摘要放在同一张回执。"""

    if not isinstance(model_contract, ModelContract):
        raise TypeError("model_contract must be a zyo.ModelContract")
    if (not isinstance(source_model_sha256, str)
            or len(source_model_sha256) != 64):
        raise ValueError("source_model_sha256 must be a 64-character SHA-256")
    # ModelContract 的 model 是已冻结映射；从零解冻后按确定变量顺序重建矩阵摘要。
    from .model import Model

    model = Model.from_dict(_thaw(model_contract.model))
    return {
        "schema": "zyo-representation-mapping-1",
        "source_model_sha256": source_model_sha256,
        "zyo_model_fingerprint": model_contract.fingerprint,
        "matrix": matrix_identity(model),
    }


def engine_contract(engine):
    name = str(engine).strip().lower()
    if name not in ENGINE_CONTRACTS:
        supported = ", ".join(sorted(ENGINE_CONTRACTS))
        raise ValueError(f"Contract supports engines {supported}; received {engine!r}")
    return dict(ENGINE_CONTRACTS[name])


def engine_support(model, engine):
    """返回本集成契约下的支持状态；False 不表示求解器全局无该能力。"""

    contract = engine_contract(engine)
    integer = any(variable.kind != "C" for variable in model.variables)
    finite_bounds = all(
        math.isfinite(float(variable.lb)) and math.isfinite(float(variable.ub))
        for variable in model.variables
    )
    reasons = []
    if integer and not contract["integer_variables"]:
        reasons.append("integer variables are outside the LP engine contract")
    if contract["integration_requires_finite_bounds"] and not finite_bounds:
        reasons.append("all variables need finite lower and upper bounds")
    return {"supported": not reasons, "engine": str(engine).strip().lower(),
            "contract": contract, "reasons": reasons}


def _requested_options(options, result):
    if options is None:
        return _thaw(dict(getattr(result, "metadata", {}).get("parameters_requested", {})))
    if isinstance(options, SolveOptions):
        return {
            "time_limit": options.time_limit,
            "node_limit": options.node_limit,
            "iteration_limit": options.iteration_limit,
            "mip_gap": options.mip_gap,
            "feasibility_tol": options.feasibility_tol,
            "integrality_tol": options.integrality_tol,
            "pivot_tol": options.pivot_tol,
            "objective_tol": options.objective_tol,
            "max_tableau_cells": options.max_tableau_cells,
            "threads": options.threads,
            "seed": options.seed,
            "presolve": options.presolve,
        }
    if isinstance(options, Mapping):
        return _thaw(dict(options))
    raise TypeError("options must be SolveOptions, a mapping, or None")


@dataclass(frozen=True)
class ResultContract:
    """保留原始 Result 字段并补充跨进程解释字段。"""

    schema: str
    model_fingerprint: str
    result: Mapping
    requested_options: Mapping
    applied_options: Mapping
    unmapped_options: object
    interpretation: Mapping = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self):
        object.__setattr__(self, "result", _freeze(self.result))
        object.__setattr__(self, "requested_options", _freeze(self.requested_options))
        object.__setattr__(self, "applied_options", _freeze(self.applied_options))
        object.__setattr__(self, "unmapped_options", _freeze(self.unmapped_options))
        object.__setattr__(self, "interpretation", _freeze(self.interpretation))

    @classmethod
    def from_result(cls, result, contract, options=None):
        if isinstance(contract, ModelContract):
            fingerprint = contract.fingerprint
        elif isinstance(contract, str) and len(contract) == 64:
            fingerprint = contract
        else:
            raise TypeError("contract must be ModelContract or a 64-character fingerprint")
        raw = result.to_dict()
        metadata = dict(getattr(result, "metadata", {}))
        bound_available = result.best_bound is not None and math.isfinite(result.best_bound)
        residuals = [
            value for value in (result.primal_residual, result.bound_residual,
                                result.integrality_residual)
            if value is not None
        ]
        interpretation = {
            "has_solution": bool(result.has_solution),
            "proved_optimal": result.status == Status.OPTIMAL,
            "global_bound_available": bound_available,
            "candidate_residuals_available": bool(residuals) and all(
                math.isfinite(value) for value in residuals
            ),
            "status_semantics": (
                "Only OPTIMAL means proved optimum in the engine's stated scope; "
                "limits and NUMERICAL_ERROR must be propagated unchanged"
            ),
            "gap_semantics": (
                "mip_gap=(objective-best_bound)/max(1,abs(objective)); "
                "null whenever objective or best_bound is unavailable"
            ),
            "parameter_semantics": (
                "parameters_requested records user input; only parameters_applied "
                "may be claimed effective, and parameters_unmapped stays excluded"
            ),
        }
        return cls(RESULT_CONTRACT_SCHEMA, fingerprint, raw,
                   _requested_options(options, result),
                   _thaw(dict(metadata.get("parameters_applied", {}))),
                   metadata.get("parameters_unmapped", []), interpretation)

    def to_dict(self):
        return {
            "schema": self.schema,
            "model_fingerprint": self.model_fingerprint,
            "result": _thaw(self.result),
            "requested_options": _thaw(self.requested_options),
            "applied_options": _thaw(self.applied_options),
            "unmapped_options": _thaw(self.unmapped_options),
            "interpretation": _thaw(self.interpretation),
        }


def result_contract(result, contract, options=None):
    return ResultContract.from_result(result, contract, options)
