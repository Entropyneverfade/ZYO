# 24h 机组组合实测 / Measured 24-hour unit commitment — 0.3.0

2026-09-08，合成教学/开发输入 `examples/uc_24h.json`：24×1h、2台固定容量机组、264变量、48二进制、480约束、1537非零元。模型为单节点有限时域 MILP，完整假设见 [建模与复现教程](unit_commitment.md)。开发实例与保留测试集分开管理。

ZYO 稀疏原生三次均返回 OPTIMAL 并通过独立原始约束、整数、物理、成本和界检查：目标 `28625.000009868018`，最优界 `28624.99999612956`，归一化 Gap `4.79946e-10`，最大物理残差 `2.17679e-8`，ENS `0 MWh`，21节点/214迭代。三种隔离比较引擎均返回目标 `28625`。不同最优轨迹均可接受。

| 实际路径 / Actual engine | 通过 / Accepted | 实际状态 / Status | 冷调用中位 s / Median call s | 采样峰值中位 MiB / Median peak RSS |
|---|---:|---|---:|---:|
| ZYO native 0.3.0 | 0/3 | NUMERICAL_ERROR | 0.5842625 | 50.93 |
| ZYO native_sparse 0.3.0 | 3/3 | OPTIMAL | 0.8262802 | 57.98 |
| HiGHS 1.8.0 | 3/3 | OPTIMAL | 0.0082488 | 56.40 |
| Gurobi 13.0.2 | 3/3 | OPTIMAL | 0.0103207 | 36.58 |
| COPT 8.0.6 | 3/3 | OPTIMAL | 0.0183972 | 44.39 |

密集原生真实失败为 `Phase I did not finish`，根节点487迭代，无候选解；表中其耗时是失败尝试耗时。独立4h手算题目标43，密集/稀疏两种自主路径均通过。24h原生稀疏本例调用约为 Gurobi 的80倍；这只描述一个开发实例，不能作为通用性能倍率或全球排名。

![运行轨迹 / Dispatch](figures/uc_dispatch.svg)

![隔离比较 / Isolated comparison](figures/uc_comparison.svg)

[绘图数据 / Plot data CSV](figures/uc_comparison.csv)

## 统一资源与验收口径 / Protocol

- 同机 Windows、AMD Ryzen 9 9950X（16核32逻辑处理器），Python3.10.19、NumPy2.2.6、SciPy1.15.3。实际比较每进程请求1线程，数值库线程环境均1；完整结果单列各引擎实际接受及未映射参数。
- 每路径3个新进程，随机顺序种子20260908；无预热，冷 `model.solve` 调用计时与监督进程墙钟分开。建模一次约0.01403s；进程墙钟中位依次0.76201/0.99280/0.49371/0.25448/0.25493s，包含启动、导入、检查与求解。
- 请求求解30s、监督60s、4GiB；节点10000、迭代100000；可行性/整数容差1e-7、目标参数1e-8、请求mip_gap0。实际接受门另以原始候选、独立物理/成本和归一化界差检查；浮点Gap并非精确零。
- 成本比较与已知28625一致；Gap定义为 `abs(objective-bound)/max(1,abs(objective))`。ENS与约束可行性分开。失败计时保留、缺失值不记零；RSS为0.1秒间隔的进程树采样峰值，可能漏掉短峰。
- 所有路径读取相同冻结未求解模型；原生进程封锁外部优化模块，比较结果不回传自主求解。输入SHA256 `4bcda8ca3325b0fec02fa65c29550762f6bf98c9a396eb6e6f05aad7ce547abf`；数学指纹 `bcd11bfb8b748ce83075adcc388290510fe141ffcc5a3754028165e26fe0be85`。

## English

The synthetic development fixture has 24 hourly periods, two fixed-capacity units, 264 variables, 48 binaries, 480 constraints and 1537 nonzeros. All three sparse-native attempts pass independent original-model, integer, physical, cost and bound checks at the objective, bound and residual listed above. ENS is zero. The three isolated reference engines independently obtain 28625. Dense native retains its actual root Phase-I numerical failure; its time is a failed-attempt time. Both native paths pass a separate hand-checked four-hour cost-43 fixture.

Measurements use the same Windows/Ryzen 9 9950X host, Python 3.10.19, NumPy 2.2.6 and SciPy 1.15.3. Three fresh processes per engine run in seeded randomized order with one requested thread, 30-second solve / 60-second process budgets and 4 GiB RSS supervision. No warmup is used. Solve-call time and complete process wall time are separate; RSS samples are taken every 0.1 seconds. Failures stay in the table. Requested feasibility/integer tolerances are 1e-7 and mip_gap is zero; returned floating-point gaps are checked explicitly using the stated formula, not rounded into exact proof.

Every engine receives the identical frozen unsolved model. External results never enter the native computation. This one-case sparse-native call ratio of roughly 80× Gurobi motivates profiling and algorithm work; it is not a general performance ratio or global rank. Network SCUC, annual UC and probabilistic adequacy have separate validation milestones. Use the linked tutorial to reproduce the native example with your own measured runtime.
