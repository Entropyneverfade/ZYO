# GRIDMIND 调用 ZYO 的公共契约 / Public GRIDMIND-ZYO contract

## 适用范围

`zyo-model-contract-1` 包装现有 `lzyopt-1` 模型和应用元数据，用规范 JSON 的 SHA-256 作为 `fingerprint`。指纹覆盖变量顺序、名称、类型、上下界、目标、约束和上游应用元数据；代数等价但表示不同的模型不保证同指纹。GridMind 年度状态路径领域模型由 `GRIDMIND_STATE_PATH_MILP_V1` 独占定义；ZYO 契约不重新解释其费用表和位序。

`zyo-result-contract-1` 原样保存 `zyo.Result` 的状态、实际后端/版本、候选值、目标、全局界、Gap、耗时、节点/迭代、三类残差、终止原因和元数据。`interpretation.proved_optimal` 只有在 `status == OPTIMAL` 时为真；`TIME_LIMIT`、`NODE_LIMIT`、`ITERATION_LIMIT`、`NUMERICAL_ERROR` 和其他状态不得改写。没有目标或有效界时 Gap 必须为空。

结果契约同时分开记录 `requested_options`、`applied_options` 和 `unmapped_options`。`threads`、`seed`、`presolve` 等若出现在未映射列表，只代表用户请求值，不得声称已控制实际执行。矩阵翻译用 `zyo-representation-mapping-1` 同时记录上游领域模型 SHA-256、ZYO 完整模型指纹、行列、约束非零、目标非零、整数列和有限边界计数；摘要用于发现翻译错位，完整模型身份仍由两个指纹共同决定。

## 引擎边界

- `native_simplex`：本契约只接受连续 LP，含整数变量即不支持。最优性依赖原尺度 KKT/证书元数据，MILP 的 `best_bound`/`mip_gap` 不适用。
- `native_sparse`：本集成契约要求所有变量具有有限上下界，支持有限盒 LP/MILP。`best_bound` 是已检查的 box-dual 全局界，Gap 定义为 `(objective-best_bound)/max(1,abs(objective))`。非有限域须走另行声明和验收的原域路径，不允许悄悄添加活跃人造界。

## 最小年度状态路径示例

[examples/annual_state_path.py](../examples/annual_state_path.py) 原样携带一份 2030-2032、两个候选位的 `GRIDMIND_STATE_PATH_MILP_V1` 纯合成 payload，并按 GridMind 桥接器的变量命名翻译。独立枚举给出路径 `(0,0,3)` 和目标 `27.52066115702479` 元；当前 `native_sparse` 单次本机结果给出同一目标、界 `27.52066115694064`、Gap 约 `3.06e-12`，并复核 one-hot、单调性、目标方向和界方向。矩阵映射为 7 行、12 列、28 个约束非零、12 个二元列。

该示例只证明冻结年度状态费用表上的路径最优，不代表全年 8760 小时机组组合、物理运行模拟或概率可靠性指标已由 ZYO 证明。物理结论仍须由 GRIDMIND 原运行器复核。

## 使用片段

```python
from zyo import ModelContract, ResultContract

contract = ModelContract.from_model(model, source_metadata)
receipt = ResultContract.from_result(result, contract)
assert result.status == "OPTIMAL"
assert receipt.to_dict()["interpretation"]["proved_optimal"]
```

跨进程传输时使用 `ModelContract.to_dict()` 和 `ResultContract.to_dict()`；两侧分别重算模型指纹并要求一致后，才可称为同一次同模求解。
