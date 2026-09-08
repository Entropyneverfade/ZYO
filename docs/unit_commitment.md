# 机组组合与结果图 / Unit commitment and result plots

ZYO 0.3.0 提供单节点、固定容量、1 小时步长的线性成本机组组合 MILP，位于独立应用包 `zyo_power`。求解器仍是通用的 ZYO 内核。

## 安装与运行 / Install and run

在克隆的仓库根目录创建并激活 Python 3.10+ 虚拟环境，然后执行：

```bash
python -m pip install ".[storage]"
python -I examples/run_uc.py --input examples/uc_24h.json --output uc-results --plot
python -m unittest discover -s tests -v
```

`--output` 指向尚不存在的目录；重复实验使用新的目录名。默认 `native_sparse`，也可以显式 `--engine native`。入口启用外部优化模块导入禁令，保存原始输入、模型、完整结果、独立检查、CSV、PNG、SVG 及绘图数据。退出码 0 表示通过原始约束、物理/成本复核和最优状态/界验收。时限、数值失败和不可行会保留实际状态。

English: create and activate a Python environment, install the storage extra, then run the native example above. Use a fresh output directory for each run. The default engine is `native_sparse`; external optimizers are blocked in this entry point. Outputs retain model/input hashes, parameters, the complete result, independently checked physics/cost, CSV and plots. Exit code zero requires optimal status, valid bounds and independent checks.

## Python 调用 / Python API

```python
import json
import zyo
from zyo_power.unit_commitment import build_uc_model, extract_uc_trace, validate_uc

# 同一份输入可反复调整参数；不要将外部比较结果送入自主求解过程。
with open("examples/uc_24h.json", encoding="utf-8") as stream:
    case = json.load(stream)
model, variables = build_uc_model(case)
result = model.solve("native_sparse", time_limit=30, mip_gap=0)
print(zyo.__version__, result.status, result.objective, result.best_bound)
if result.has_solution:
    trace = extract_uc_trace(variables, result.values)
    print(validate_uc(case, trace))
```

## 数学与物理口径 / Formulation and assumptions

- 功率 MW，逐时能量 MWh，时间步长固定 1 h，费用使用输入指定的统一货币单位。`marginal_cost` 是每 MWh，`no_load_cost` 是在线每小时，启停费用是每次事件。
- 每台机组的开机量 `on` 为二元；连续 `start/stop` 通过转换等式与上界严格确定为 0/1，不依赖正成本来排除虚假启停。
- 功率满足 `p_min*on <= p <= p_max*on`，还具有原始绝对功率界。逐时常规电源、实际风光利用量与 ENS 共同进入负荷平衡；风光允许弃电，边际成本取 0。
- 初始 `initial_on`、`initial_power`、`initial_duration` 描述第一个区间之前的状态，首小时也检查爬坡；连续开停机时长不足时先履行剩余约束。
- 常规爬坡约束为 `p[t]-p[t-1] <= ramp_up*on[t-1]+startup_ramp*start[t]`，向下约束对称。`startup_ramp/shutdown_ramp` 表示启动/停机时段可达到/退出的 MW，不是持续爬坡速率。
- 最小开停机约束在给定时域内执行；期末允许存在延续义务，`terminal_state` 返回状态、功率、已持续小时和剩余义务。接续运行应传递这些状态，不能自行重置；本例不是周期 UC。
- 备用定义为在线容量裕度 `sum(p_max*on-p) >= reserve`，不包含网络可达性、短时间爬坡可交付性或 N−1 安全分析。
- `strict=true` 禁止切负荷；诊断模式的缺电量单列 ENS，费用加 `ens_cost*ENS`，是显式加权目标而非字典序保供。

English: hourly single-node UC includes binary commitment, exact transition indicators, initial conditions, minimum up/down durations, startup/shutdown ramps, thermal limits, curtailable renewable supply and online-capacity headroom reserve. Costs are linear energy, online-hour and event costs. Terminal obligations are reported for continuation. Strict mode fixes ENS to zero; diagnostic mode uses an explicit weighted ENS penalty. Network/security deliverability and periodic continuation are separate models.

定义参照并核对 [PyPSA v1.1.1 机组组合文档](https://docs.pypsa.org/v1.1.1/user-guide/optimization/unit-commitment/)。ZYO 自行实现上述显式启停及独立时序扫描检查；未调用 PyPSA。相对于该参考，ZYO 明确要求初始出力并检查首小时爬坡，费用和 MW 输入直接给出。

## 合成 24 小时开发例 / Synthetic 24-hour development case

`examples/uc_24h.json` 包含两台机组与风光负荷，264 变量、48 二进制变量、480 行约束。逐机状态、出力、备用、功率平衡和费用均从输入/输出独立复算。单条确定性时序使用 ENS。

运行与对照图、逐路径实测口径见 [本版本实测](uc_results.md)。这些是教学开发数据；Python 接口与实现语言策略见 [结果导向的实现语言](implementation_languages.md)。
