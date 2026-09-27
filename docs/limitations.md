# 能力和限制 / Capabilities and limitations

## 当前原生路径 / Current native paths

| 引擎 / Engine | 已验证范围 / Verified scope | 能力边界 / Boundary |
|---|---|---|
| `native_simplex` | 原始稀疏修正单纯形LP；自由/单边变量、标准化与独立KKT；Netlib4×3通过 | LP专用；公开装配门行×变量≤2,000,000；困难大题底层Phase-I有退化/额度限制；对偶驱动尚未验收 |
| `native_sparse` | 自研有限盒内点LP/基本分支定界；非有限域显式人造盒阶梯及原域检查 | 人造截断本身不等价于原域；仅原域检查通过的结果可接受，MILP难搜索仍受限 |
| `native` | 自研密集两阶段单纯形/基本分支定界，原有小题/UC检查 | 密集表规模门；其他数值边界继续独立测试 |

原生计算不调用外部优化器，无静默回退。NumPy/SciPy/SuperLU用于基础数值运算。
可行性本身不证明最优；限制状态、大题失败和不同模型范围分别保存。
[认证LP教程](native_simplex.md)及[0.3.5实测](releases/0.3.5.md)提供当前方法、证据与复现入口。

English: the primal sparse revised-simplex LP engine, finite-box interior-point MILP engine and dense
MILP engine have separate validation scopes. Netlib4×3 independently passes on the certified LP path;
large degenerate Phase-I and the unaccepted dual driver retain distinct limitations. Native work uses
basic linear algebra only, without optimization-engine calls or silent fallback. Status and independent
certificates determine scientific interpretation; feasibility alone does not prove optimality.
The public LP adapter has a2,000,000 row-column allocation gate for its current dense input
assembly. The large development benchmark uses the lower-level sparse Phase-I driver instead.

当前发布工程验收：459研发回归、283独立非editable安装测试（0跳过），安装教程目标−5且证书通过；
全部安装验收封锁外部优化模块。小题数、内部大题Phase-I和成熟求解器覆盖分别评价。
Release engineering acceptance:459 development regressions and283 independently installed public
tests pass with zero skips under an external-optimizer embargo; the installed analytical guide
returns objective−5 and a verified certificate. These scopes are evaluated separately.

## 0.3.3历史能力与实测 / Historical 0.3.3 scope and measurements

0.3.3增加每个LP局部的CSC装配结构复用；233项研发回归、120次同核结构消融数值同一。当前自主数学支持范围沿用下表，缓存换取少量时间收益并增加结构存储；完整逐题比较见[结构复用](sparse_reuse.md)。

Version0.3.3 adds per-LP CSC assembly reuse, verified by233 development regressions and120 numerically identical ablation calls. Mathematical scope remains as listed below; structural memory is traded for modest measured runtime gains. See the linked per-case comparison.

0.3.2历史增量：稀疏原方程Newton残差校正、类型安全CSC行缩放及独立解析反例，219项研发回归通过。原生受控free MPS读取见[格式说明](mps.md)。24h/2机组UC的`native`和`native_sparse`接受原始约束、物理、成本及界检查，完整图表及协议见[UC实测](uc_results.md)。密集算法仍有其他数值失败，退化换基规则不具备完整Bland防循环保证；时间与迭代限制继续有效。

The historical 0.3.2 increment adds original-equation Newton refinement and type-safe CSC row scaling, with analytical regressions and219 passing development tests. Both native engines are checked against the original UC constraints, physics, costs and bounds. Other numerical failures remain; the dense stable-tie rule is not the complete Bland anti-cycling rule, and resource limits remain enforced. See the linked protocol and figures.

- LP/MILP：自主密集两阶段单纯形/基本分支定界；实验性自主有限盒稀疏内点法/分支定界。不是稀疏修正单纯形，也没有成熟的热启动、割平面、伪成本或并行搜索体系。
- 稀疏路径要求所有变量有显式有限上下界。盒对偶界及不可行分离采用保守浮点检查，不是一般精确算术证书；无法构造充分射线时仍可能数值未决。
- 2026-09-08 Phase-I开发集复验：原60与另外300普通小整数题全部OPTIMAL；18边界反例为9 OPTIMAL / 7 INFEASIBLE / 2 NUMERICAL_ERROR。168份分离证书通过Fraction精确复算；174项完整研发回归通过。两近可行未决如实保留，公开测试仍属于开发回归。
- 本地20题×5路径×3次最终测评中，稀疏原生对18有限题三次稳定通过17题，自由变量题UNKNOWN；另2道状态题为INFEASIBLE、UNKNOWN。密集原生14/18。原两个整数失败各三次OPTIMAL；10/30阶运输LP仍三次OPTIMAL，目标182/172。一般自由/无穷变量及无界证书继续独立开发。
- 单个合成年度储能 MILP 自主通过，规模为 61,321 变量、8,760 二进制、35,041 约束、113,882 非零元；根节点即完成，不代表一般难整数搜索性能，也不代表论文实验完成。
- 储能支持单节点固定容量规则、周期迭代与调度模型。周期闭合与 ENS 分别报告。新增有限时域、固定容量单节点机组组合；网络 SCUC、全年 UC 和概率充裕度尚待验收。
- 原始凸 QP、一般非线性/MINLP、非凸 AC-OPF 尚未验收；MPS 写出、完整 fixed MPS/LP 语法、完整 LP 原始—对偶/基本解证书和大规模公开基准仍在计划中。
- 外部适配器保留历史显式比较用途，不能计为自主算法；没有静默回退。共享小模型目标相同只是交叉佐证，不是完备证明。

尚无可信“全球第几”结论。目前定位为可修改、可追溯的自主研究原型，未达到成熟工业通用求解器水平。国际测试将按问题类别、固定公开实例、同机资源、正确性门、解决率和失败计入的统计开展，不能从有利单例推算排名。

## English

- The independent engines implement dense two-phase simplex/basic branch-and-bound and an experimental finite-box sparse interior-point/branch-and-bound path. They do not yet provide mature revised simplex, warm starts, cutting planes, pseudocost branching or parallel search.
- The sparse path requires explicit finite bounds. Box bounds and sufficient separation witnesses use protected floating-point arithmetic, not a general exact certification system. Failure to construct a witness can leave a numerical failure unresolved.
- The 2026-09-08 Phase-I replay solves all360 ordinary small development MILPs. The18 adversarial cases return9 OPTIMAL /7 INFEASIBLE /2 NUMERICAL_ERROR. All168 emitted certificates pass exact Fraction recomputation and all174 full development tests pass. The two near-feasible numerical failures remain explicit; public tests are development regressions.
- In the final300-run assessment, native sparse passes all three repeats on17/18 finite instances and dense native on14/18. The remaining sparse finite case has free variables and returns UNKNOWN; separate status cases return INFEASIBLE and UNKNOWN. Both repaired integer cases and both10/30-order transportation LPs pass three repeats. Free/infinite-variable support and general unboundedness certificates retain separate validation milestones.
- One synthetic annual storage MILP had 61,321 variables, 8,760 binaries, 35,041 constraints and 113,882 nonzeros, and closed at the root node. This is not evidence of difficult combinatorial search, general yearly unit commitment, or completed paper experiments.
- Storage examples cover fixed-capacity single-node rules, periodic iteration and optimization, reporting deterministic ENS separately from cycle closure. A finite-horizon single-node UC model is now validated in the stated scope. Network SCUC, annual UC and probabilistic adequacy remain separate milestones.
- Original convex QP, general nonlinear/MINLP and nonconvex AC-OPF are not validated. MPS writing, complete fixed-MPS/LP syntax, full LP primal-dual/basic-solution certificates and broad public benchmarks remain future work.
- External adapters are for isolated comparisons only; no silent fallback. Agreement with another optimizer is corroboration, not a complete proof.

There is no defensible global rank. ZYO is currently an inspectable independent research prototype, not an industrial-class general-purpose solver. Future comparisons must freeze instance sets, resource budgets, tolerances and correctness gates, and report solve rates and failures rather than selecting only favorable examples.
