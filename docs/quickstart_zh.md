# 安装、建模与储能 / Installation, modeling and storage

## 1. 获取和安装

0.3.4受限新增内容安装/运行前，先阅读[LICENSE](../LICENSE)及[范围](../LICENSE-SCOPE.json)，在[用途申请](https://github.com/Entropyneverfade/ZYO/issues/new?template=zyo-test.yml)只填用途，获账号/申请批准并接受条款后在期限内测试（默认90日）。仅使用独立历史MIT部分依原许可，无需新申请。商用、生产、再分发及服务另批；API权限独立管理。

Before installing/running restricted0.3.4 additions, read the license/scope, submit only purpose, obtain account/request approval and assent, then test within the term (default90 days). Independent historical MIT portions retain original rights without a new request. Commercial/production use, redistribution and services need separate approval; API rights are separate.

从仓库页面选择 **Code → Download ZIP**，解压后进入含 `pyproject.toml` 的目录；或者按 README 使用 Git 克隆。以下命令以 Windows PowerShell 为例：

```powershell
python -m venv .venv
$py = '.\.venv\Scripts\python.exe'
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
& $py -m pip install -e ".[native-sparse]"
& $py -m pip check
& $py -m zyo --version
& $py -m unittest discover -s tests -v
```

可编辑安装会直接使用当前源码。修改代码后重启 Python；不要在同一环境混装多个源码目录。仅 NumPy 安装将跳过依赖 SciPy 的测试，必须报告跳过数量，不能声称全项验证。

Linux/macOS 用 `python3 -m venv .venv`、`.venv/bin/python` 替代上述解释器路径；这些平台尚需用户自行实测。

## 2. 建模与结果

变量使用 `add_var(name, lb, ub, vtype)`，支持连续 `C`、整数 `I` 和二进制 `B`。约束通过表达式 `<=`、`>=`、`==` 添加；目标用 `minimize` 或 `maximize`。常见别名 `addVar`、`addConstr`、`optimize` 可用，但不是 gurobipy/coptpy 的完整兼容实现。

```python
import zyo as zo

m = zo.Model('production')
x = m.add_var('x', lb=0, ub=10)
y = m.add_var('y', vtype='B')
m.add_constr(x + 2*y >= 4)
m.minimize(3*x + 2*y)
r = m.solve('native_sparse', TimeLimit=10, MIPGap=0)

# 不要把限制终止或只有可行解的结果写成已证明最优。
print(r.status, r.termination_reason)
print(r.objective, r.best_bound, r.mip_gap)
if r.has_solution:
    print(x.x, y.x)

# JSON 导出保留原数学模型；目录需要事先存在。
m.write('model.json')
```

变量边界应来自实际数学/物理模型，稀疏路径要求显式有限上下界。已交付[受控free MPS读取](mps.md)，支持范围和格式检查见专门说明。Bounds should follow the mathematical or physical model; the sparse path requires finite bounds. Controlled free-MPS import is available with the documented feature checks.

## 3. 不调用优化器的储能教学例

```python
import numpy as np
from zyo_power.examples import teaching_case
from zyo_power.rule import rule_dispatch
from zyo_power.periodic import initialization_study

# 单节点固定容量：功率 MW、能量 MWh，T 区间使用 T+1 能量状态。
case = teaching_case()
study = initialization_study(case, lambda e: rule_dispatch(case, e), max_cycles=10)
print(study['initialization_status'])
for run in study['runs']:
    print(run['status'], run.get('confirmed_cycle'))
    if run['formal_audit'] is not None:
        print(run['formal_audit'])
```

前三轮空电池期末能量约为 7.7052631579、14.7368421053、14.7368421053 MWh；还需第四轮确认重复。含损耗时充电量不等于放电量。闭合、物理可行和是否存在 ENS 必须分别报告。这里是确定性周期稳态迭代，不是概率 EENS/LOLE 计算。

## 4. 自主全周期优化

```python
from zyo import SolveOptions
from zyo_power.examples import tiny_case
from zyo_power.optimization import optimize_dispatch
from zyo_power.validation import validate_dispatch

# 可手算的两小时 MILP，保留严格互斥与循环能量约束。
case = tiny_case()
result = optimize_dispatch(case, engine='native_sparse', cyclic=True,
    exclusivity='milp', options=SolveOptions(time_limit=10))
audit = validate_dispatch(case, result, require_exclusivity=True)
print(result['status'], audit)
```

扩展到 168/8760 小时前，先看独立原始残差、真实停止原因和资源开销。滚动模型必须显式指定 `engine='native_sparse'`，其历史函数默认值保留了外部比较行为，不推荐直接使用默认值。不要运行旧综合比较入口并把外部成绩标成自主结果。

## 5. 报错怎么处理

`SIZE_LIMIT` 是密集表规模保护；`UNKNOWN` 可能是当前路径不支持无限边界；`NUMERICAL_ERROR` 表示数值问题未解决，不自动意味着不可行。保留原模型、版本、选用内核、参数与完整异常，在 Issues 提交一个可公开的小型复现。不要为了通过而删约束、改目标或无依据调宽容差。

---

## English tutorial

### Install and verify

Download **Code → Download ZIP** or clone the repository. Enter the directory containing `pyproject.toml`, create a virtual environment, and run the commands in section 1. Set UTF-8 mode on Windows to preserve Chinese paths and logs. Editable installation uses your checkout directly; restart Python after editing. Do not install multiple source checkouts into one environment. NumPy-only mode skips SciPy-dependent tests: report skips rather than claiming complete validation.

### Model and inspect results

Use `add_var(name, lb, ub, vtype)` with `C`, `I` or `B`; add linear relations with `add_constr`, and use `minimize`/`maximize`. Common aliases are supported, but this is not a full gurobipy/coptpy compatibility layer. Section 2 provides a runnable example, explicit `native_sparse` selection, result inspection and JSON export. An unspecified infinite bound must not be replaced with an arbitrary large number merely to pass a test. General MPS/LP import is not yet delivered.

### Storage rule simulation

Section 3 runs deterministic surplus-charge/deficit-discharge simulation from low, middle and high initial energy. Units are MW, MWh and hours, with T+1 energy states for T intervals. Starting empty, the first three final energies are approximately 7.7052631579, 14.7368421053 and 14.7368421053 MWh; a fourth complete cycle confirms repetition. Charge and discharge energies differ when efficiency is below one. Physical feasibility, cycle closure and ENS are separate checks. This is empirical periodic-state iteration, not probabilistic EENS/LOLE estimation.

### Native cyclic optimization

Section 4 solves a hand-checkable two-hour MILP with strict charging/discharging exclusivity, explicitly using the independent sparse path, and recomputes the physical checks from inputs and outputs. Inspect the real termination status and residuals before increasing scale. For rolling dispatch, explicitly pass `engine='native_sparse'`: the legacy function default retains external-comparison behavior. Window optimality is not a full-year optimality certificate.

### Troubleshoot

`SIZE_LIMIT` indicates the dense allocation guard; `UNKNOWN` may indicate unsupported infinite bounds; `NUMERICAL_ERROR` means unresolved numerical difficulty, not proven infeasibility. Retain version, input, parameters, selected engine and complete errors. Do not remove constraints, change objectives, or loosen tolerances merely to obtain a passing result.
