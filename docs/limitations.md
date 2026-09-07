# 能力和限制 / Capabilities and limitations — 0.2.0 research build

- LP/MILP：自主密集两阶段单纯形/基本分支定界；实验性自主有限盒稀疏内点法/分支定界。不是稀疏修正单纯形，也没有成熟的热启动、割平面、伪成本或并行搜索体系。
- 稀疏路径要求所有变量有显式有限上下界。盒对偶界及不可行分离采用保守浮点检查，不是一般精确算术证书；无法构造充分射线时仍可能数值未决。
- 开发证据：原先 60 个小整数题中的 22 个失败已修复并通过独立穷举；另 300 个开发题仍有 2 个未决，18 个针对性反例仍有 7 个未决。公开单元测试不是未见保留集。
- 本地 20 题隔离测评中，18 个有限最优问题三次均通过的题数为密集原生 14、稀疏原生 13；两个原生路径不能取并集冒充一个完整求解器。运输 LP 与旧整数困难例仍有失败。
- 单个合成年度储能 MILP 自主通过，规模为 61,321 变量、8,760 二进制、35,041 约束、113,882 非零元；根节点即完成，不代表一般难整数搜索性能，也不代表论文实验完成。
- 储能支持单节点固定容量规则、周期迭代与调度模型。周期闭合不代表供电充裕；缺供单列 ENS。无网络、完整机组组合或概率充裕度验收。
- 原始凸 QP、一般非线性/MINLP、非凸 AC-OPF 尚未验收；受控 MPS/LP 标准格式、完整 LP 原始—对偶/基本解证书和大规模公开基准仍在计划中。
- 外部适配器保留历史显式比较用途，不能计为自主算法；没有静默回退。共享小模型目标相同只是交叉佐证，不是完备证明。

尚无可信“全球第几”结论。目前定位为可修改、可追溯的自主研究原型，未达到成熟工业通用求解器水平。国际测试将按问题类别、固定公开实例、同机资源、正确性门、解决率和失败计入的统计开展，不能从有利单例推算排名。

## English

- The independent engines implement dense two-phase simplex/basic branch-and-bound and an experimental finite-box sparse interior-point/branch-and-bound path. They do not yet provide mature revised simplex, warm starts, cutting planes, pseudocost branching or parallel search.
- The sparse path requires explicit finite bounds. Box bounds and sufficient separation witnesses use protected floating-point arithmetic, not a general exact certification system. Failure to construct a witness can leave a numerical failure unresolved.
- The original 22 failures among 60 small integer development cases were repaired and checked by independent enumeration. A separate 300-case development audit still had two unresolved cases; seven of 18 targeted adversarial cases remained unresolved. Public tests are not an unseen holdout set.
- In a local 20-instance isolated comparison, the dense and sparse native paths passed all three repeats on 14 and 13 of the 18 finite-optimum instances respectively. Their successes cannot be pooled to pretend one engine solves everything. Transportation LP and difficult integer examples still fail.
- One synthetic annual storage MILP had 61,321 variables, 8,760 binaries, 35,041 constraints and 113,882 nonzeros, and closed at the root node. This is not evidence of difficult combinatorial search, general yearly unit commitment, or completed paper experiments.
- Storage examples cover fixed-capacity single-node rules, periodic iteration and optimization. Report deterministic ENS separately from cycle closure. Network, complete unit commitment and probabilistic adequacy validation are not completed.
- Original convex QP, general nonlinear/MINLP and nonconvex AC-OPF are not validated. General controlled MPS/LP exchange, complete LP primal-dual/basic-solution certificates and broad public benchmarks remain future work.
- External adapters are for isolated comparisons only; no silent fallback. Agreement with another optimizer is corroboration, not a complete proof.

There is no defensible global rank. ZYO is currently an inspectable independent research prototype, not an industrial-class general-purpose solver. Future comparisons must freeze instance sets, resource budgets, tolerances and correctness gates, and report solve rates and failures rather than selecting only favorable examples.
