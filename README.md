# ZYO 0.2.0 · 自主 LP/MILP 研究求解器 / Independent LP/MILP research solver

ZYO 是可本地运行、可修改算法的通用优化研究项目，提供 Python 线性建模接口、自主求解内核和单节点储能教学应用。本次为 **研究开发版**，欢迎下载和交流。本项目基于ChatGPT 6 开发，作者本人也在学习中，欢迎相关专业同好交流

[中英安装与使用 / Bilingual quickstart](docs/quickstart_zh.md) · [能力与限制 / Limitations](docs/limitations.md) · [参与开发 / Contributing](CONTRIBUTING.md) · [版本记录 / Changelog](CHANGELOG.md) · [依赖 / Dependencies](THIRD_PARTY.md)

## 自主求解边界

| 路径 | 算法 | 依赖与范围 |
|---|---|---|
| `native` | ZYO 自编密集两阶段单纯形＋基础分支定界 | NumPy；有密集表规模保护 |
| `native_sparse` | ZYO 自编稀疏原始—对偶内点法＋有限盒界＋分支定界 | NumPy、SciPy 稀疏线性代数/SuperLU；变量必须有有限上下界，实验性 |
| 外部适配器 | HiGHS/Gurobi/COPT | 仅供显式、隔离的独立比较 |

自主路径不调用外部优化引擎，也不自动回退。SciPy 的矩阵/线性方程组件不是优化求解器调用。历史 `lzyopt` 包名为兼容建模与密集内核保留，推荐使用 `import zyo`，也支持 `import ZYO`。

## 安装

需要 64 位 Python 3.10+；目前已直接验证的环境为 Windows/Python 3.10.19。其他平台请运行测试确认，不能从 Python 代码可导入推断全部算法已通过。

```bash
git clone https://github.com/Entropyneverfade/ZYO.git
cd ZYO
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[native-sparse]"
.\.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v
```

Linux/macOS 对应解释器为 `.venv/bin/python`。只用密集自主内核可安装 `pip install -e .`；初次安装依赖需网络，安装完成后本地求解无需网络。当前不是 PyPI 发布，请不要把同名第三方包当作本项目安装。

## 第一个模型

```python
import zyo as zo

# 两类资源满足需求；y 是二进制开关，x 是连续量。
m = zo.Model('first_model')
x = m.add_var('x', lb=0, ub=10)
y = m.add_var('y', vtype='B')
m.add_constr(x + 2*y >= 4, name='demand')
m.minimize(3*x + 2*y)

# 手算最优 x=2、y=1，目标为 8；明确选取自编算法。
r = m.solve('native')
print(r.status, r.objective, r.best_bound, r.mip_gap)
if r.has_solution:
    print(r.values)
```

有解不等于已证明最优。请同时检查实际状态、界、Gap 和原始约束残差；时间/节点/数值限制不能被描述成最优。

## 测试与规模声明

公开包提供独立预期的核心/储能测试及 60 个冻结小整数开发回归。

已有一个 8760 小时合成储能循环 MILP 的自主运行证据，但其搜索仅一个根节点，详见能力说明。

许可证为 MIT，保留原始版权声明。

---

## English

ZYO is a locally runnable, editable, general-purpose **research prototype** for linear and mixed-integer linear optimization. It includes a Python modeling API, independent algorithms, and small single-node storage applications. It is **not an industrial replacement for Gurobi, COPT or HiGHS**; independently verify results before using them in research or decisions.

- `native`: ZYO's dense two-phase tableau simplex and basic branch-and-bound; requires NumPy and retains a dense-size guard.
- `native_sparse`: ZYO's experimental sparse primal-dual predictor/corrector method, finite-box bounds and branch-and-bound. It requires explicit finite variable bounds. NumPy and SciPy sparse/SuperLU supply numerical linear algebra, **not optimization engines**.
- Legacy HiGHS/Gurobi/COPT adapters are retained for explicitly selected, isolated comparisons only. External solutions, bases and cuts must not assist the native solver. No silent fallback is performed.

Use the installation commands above with 64-bit Python 3.10+. On Linux/macOS replace `.venv\Scripts\python.exe` with `.venv/bin/python`; those platforms still require direct testing. The directly tested development platform is Windows/Python 3.10.19. Install `-e .` for NumPy-only dense mode, or `-e ".[native-sparse]"` for sparse mode. Initial dependency installation needs network access; subsequent solving can run offline. **This is a GitHub source release, not a PyPI release.**

The example above minimizes `3*x+2*y` subject to `x+2*y>=4`, with bounded continuous `x` and binary `y`. Its hand-checked optimum is `x=2, y=1, objective=8`. Always inspect status, incumbent, bound, gap and original-model residuals: a feasible point or a time limit does not prove optimality.

The public tests include independently specified examples and 60 frozen small integer development regressions. They are **not an unseen holdout set or an international ranking**. A single synthetic 8760-hour cyclic storage MILP has been solved by the native sparse path, but closed at the root node; it does not establish general hard-MILP or year-long unit-commitment capability. Numerical robustness and infeasibility detection remain limited.

This screened export retains the existing algorithm files without solver rewrites. Private local experiments and complete workstation history are not published. Python identifiers remain English; Chinese maintenance comments and English module descriptions support bilingual development. Distributed under the existing MIT license. Please submit small, shareable reproductions without credentials or unauthorized research data.
