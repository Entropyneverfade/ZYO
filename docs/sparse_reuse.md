# 0.3.3 稀疏结构复用 / Sparse structural reuse

本片在每个LP内部缓存增广矩阵 `[-I C'; C 0]` 的CSC装配映射。shape、indptr或indices改变即重建；每轮重新填入当轮数值并执行LU，返回独立存储。一般模型、节点LP与Phase-I各自创建缓存。Python API和数学验收门保持稳定。

The increment caches only the CSC assembly map of `[-I C'; C 0]` within each LP. Any shape/pointer/index change invalidates the cache. Values and LU factors are computed afresh; each returned matrix owns independent storage. This is structural assembly reuse, not numerical-factor reuse or basis warm starting.

## 依据 / References

- [Gondzio, EJOR218 (2012), §5](https://doi.org/10.1016/j.ejor.2011.09.017)：区分牛顿系统结构与变化的对角数值，并考虑不定分解的主元依赖。Separate Newton structure from changing diagonal values and account for numerical pivoting.
- [Gurobi13.0 Barrier preprocessing](https://docs.gurobi.com/projects/optimizer/en/current/concepts/logging/barrier.html)：公开日志区分结构排序与后续分解工作，用于指导剖析。Public guidance motivates separating structural work from numerical factorization.
- [SciPy1.15.3 splu](https://docs.scipy.org/doc/scipy-1.15.3/reference/generated/scipy.sparse.linalg.splu.html)：使用CSC与原COLAMD/LU路径。Uses CSC and the existing COLAMD/LU interface.

## 开发集实测 / Development-set measurement

2026-09-08同机20题×两方法×三次新进程，1线程，求解10秒/监督45秒/内存4GiB；线性代数已导入、模型冷求解，进程启动另记。两方法使用同一自主内核和输入，只切换结构缓存。外部优化器导入被封锁。

All120 calls retain identical status, objective, best bound, gap, variable values, node count and iteration count within each case. The comparison isolates cache use in the same native code and frozen inputs; external optimizers are blocked. Three repetitions measure implementation behavior, not a global ranking or a statistical significance guarantee.

| 指标 / Metric | 无缓存 / Uncached | 局部缓存 / Cached |
|---|---:|---:|
| 60次调用的结构重建 / Structural builds over60 solves | 17,394 | 1,965 |
| 结构复用 / Structural reuses | 0 | 15,429 |
| 20题逐题中位耗时之和 / Sum of per-case median seconds | 9.527917 | 9.241643 |
| 168h储能中位调用秒 / Storage-week median seconds | 0.072894 | 0.069028 |
| 8760h储能中位调用秒 / Storage-year median seconds | 6.003369 | 5.931489 |
| 8760h储能峰值RSS中位MiB / Storage-year median peak RSS | 366.47 | 377.07 |
| joint-infeasible 中位秒 / Median seconds | 0.004023 | 0.004515 |
| unbounded 中位秒 / Median seconds | 0.000828 | 0.000844 |

总计减少3.00%；上述两个小例变慢，额外O(nnz+n+m)缓存增加内存。所有状态保留：每方法每重复17 OPTIMAL、1 INFEASIBLE、2 UNKNOWN。Windows CPU计时在部分小例量化为零，保留原值。全部20题的逐题调用耗时在下表，失败也计入分母；同一最优目标不要求不同引擎变量轨迹相同。

The aggregate reduction is3.00%; the two small cases shown above are slower, and the O(nnz+n+m) cache trades memory for assembly work. Every method/repetition retains17 OPTIMAL,1 INFEASIBLE and2 UNKNOWN results. Quantized zero CPU measurements remain zero. This narrowly measured scope is distinct from the earlier UC comparison.

| 题目 / Case | 无缓存秒 / Uncached s | 缓存秒 / Cached s |
|---|---:|---:|
| mknap1-01 | 0.043811 | 0.041438 |
| mknap1-02 | 0.114315 | 0.105215 |
| mknap1-03 | 0.438974 | 0.411106 |
| mknap1-04 | 0.183502 | 0.170940 |
| mknap1-05 | 0.442011 | 0.411868 |
| mknap1-06 | 1.237679 | 1.144974 |
| mknap1-07 | 0.629795 | 0.601788 |
| transport-10 | 0.012567 | 0.012231 |
| transport-30 | 0.067349 | 0.066397 |
| sparse-lp-100 | 0.016828 | 0.016255 |
| sparse-lp-500 | 0.183816 | 0.182073 |
| joint-infeasible | 0.004023 | 0.004515 |
| unbounded | 0.000828 | 0.000844 |
| free-variable | 0.000833 | 0.000830 |
| redundant-equality | 0.005076 | 0.004916 |
| seed-641927-084 | 0.024448 | 0.022723 |
| seed-883109-075 | 0.031334 | 0.028926 |
| storage-teaching | 0.014464 | 0.014088 |
| storage-week | 0.072894 | 0.069028 |
| storage-year | 6.003369 | 5.931489 |

233项研发测试通过；5项新测试含独立稠密块定义、同nnz异结构、重复/空矩阵、输出污染和RHS/盒改变。378个精确开发检查继续单列两近可行NUMERICAL_ERROR；[Gurobi数值边界说明](https://docs.gurobi.com/projects/optimizer/en/current/concepts/numericguide/tolerances_scaling.html)用于解释精确与容差预期的差别，原容差/状态门保持。

Passed233 development tests, including five new independent tests. The378-case exact development audit retains two near-feasible NUMERICAL_ERROR cases separately; documented numerical-boundary semantics guide diagnosis without changing the frozen acceptance policy.
