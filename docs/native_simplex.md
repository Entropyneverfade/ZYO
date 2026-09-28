# 原生认证LP / Native certified LP

## 模型与算法 / Model and algorithm

`native_simplex`是ZYO自研有界变量原始稀疏修正单纯形LP路径。输入为线性目标、
等式/不等式和变量界；自由变量通过精确拆分映射。行活动变量将标准化模型写为

\[
\min c^Tx+c_0,\quad Ax=0,\quad l\le x\le u.
\]

最小化/最大化和目标常数回映射到用户原模型。逻辑起点不可行时执行人工变量Phase-I；
基变量满足 $Bx_B=-Nx_N$，入列方向 $d=B^{-1}a_q$，移动后
$x_B\leftarrow x_B-\alpha\,s_qd$。有界变量可翻界而不换基。
NumPy数组、SciPy稀疏矩阵/SuperLU是基础数值组件；ZYO维护eta更新、定价、比例检验与状态。

`native_simplex` implements ZYO's primal bounded-variable sparse revised simplex for LPs.
Exact free-variable splitting and homogeneous row encoding preserve the original model.
ZYO owns the optimization logic; NumPy and SciPy/SuperLU provide basic numerical operations.
The original objective direction, constant and variable mapping are restored in the result.

## 安装后手算复现 / Installed hand-checkable example

安装 `.[native-sparse]`，再运行 / Install `.[native-sparse]`, then run:

```python
import zyo

m = zyo.Model('certified-lp')
x = m.add_var('x', lb=0, ub=None)
y = m.add_var('y', lb=0, ub=None)
m.add_constr(x + y <= 4)
m.add_constr(x + 3*y <= 6)
m.minimize(-x - 2*y)
r = m.solve('native_simplex', iteration_limit=100)
assert r.status == zyo.Status.OPTIMAL
assert abs(r.objective + 5) < 1e-7
assert r.metadata['certificate']['verified']
assert not r.metadata['fallback_used']
print(r.values, r.primal_residual, r.metadata['iteration_budget'])
```

手算交点 $(x,y)=(3,1)$、目标 $-5$。独立检查原约束、变量界、对偶符号、简约成本、
互补与强对偶；仅证书通过才从后端报告OPTIMAL。证书失败保留原状态并返回数值错误。
Feasibility alone is not optimality: the adapter verifies six independent KKT conditions before
reporting OPTIMAL. The analytical optimum is $(3,1)$ with objective $-5$.

## 预算、输出和范围 / Budget, output and scope

- Phase-I、人工驱除与Phase-II共享`iteration_limit`。每次换基、零步长动作和翻界各算一步。
  `metadata.iteration_budget`记录额度、已用量、各阶段与单位；`refactorisations`计全部成功LU工作，
  包括被拒起点及人工清理，不是仅接受路径或失败尝试数。
- LP结果保留状态、目标、实际引擎/版本、迭代、原尺度残差、原始/对偶值及证书。
  资源限制退出保留真实状态；多解不要求轨迹相同。
- 此路径仅LP；MILP使用现有`native`或有限盒`native_sparse`明确选择。对偶修正单纯形、
  大规模困难整数搜索和所有公开题覆盖分别有后续验证门。
- 0.3.6公共模型适配器按CSC装配输入，`metadata.matrix_format='csc'`、`model_nonzeros`和
  `assembly_seconds`可用于追踪表示与装配成本；显式稠密参考入口仍保留2,000,000元素分配门。
  1420阶对角教学LP已通过公共入口，但底层稀疏Phase-I大题实验不等于通用大题验收。

English: all three stages share one executed-update budget. Successful factorisations include
rejected starts and artificial cleanup. Real stopping states, engine identity and independent
certificates accompany results. This backend is LP-only; native MILP paths are separately selected
and validated. Dual revised simplex and broad industrial-scale coverage remain distinct milestones.
Since0.3.6 the public adapter assembles CSC input and reports format, nonzeros and assembly time.
The explicit dense reference still gates allocations at2,000,000 row-column entries. A solved
1420-dimensional diagonal tutorial is distinct from broad large-scale LP acceptance.

本地Python求解不需要API密钥；受限新增内容的使用许可仍按用途申请管理。
未来远程API授权与本地源码许可分别管理，不会将第三方引擎或公网服务作为原生计算的依赖。
Local Python solving needs no API key. Permission for restricted additions follows purpose-only
testing grants; prospective remote API authorization is separate from local-source licensing.

针对性复现 / Targeted reproduction:

```powershell
python -B -X utf8 -m unittest discover -s tests -p "test_simplex*.py" -v
python -B -X utf8 -m unittest discover -s tests -p "test_native_lp.py" -v
python -B -X utf8 -m unittest discover -s tests -v
```

来源 / Sources: [Huangfu & Hall, 2018](https://doi.org/10.1007/s12532-017-0130-5)的基方程与操作定义；
[NumPy2.2索引赋值](https://numpy.org/doc/2.2/user/basics.indexing.html#assigning-values-to-indexed-arrays)
用于唯一基下标更新；[Gurobi公开日志定义](https://docs.gurobi.com/projects/optimizer/en/current/concepts/logging/simplex.html)
用于工作量/状态分离。它们是公开参考，不是ZYO调用外部优化器的实现。
