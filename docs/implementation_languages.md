# 结果导向的实现语言策略 / Outcome-driven implementation languages

更新：2026-09-08。Python 科研接口 `import zyo` 保持稳定；实现语言服务于正确性、性能、维护和可部署性，不以某种语言的代码占比验收。

## 成熟项目的可借鉴结构

- [HiGHS 官方工作坊接口说明](https://workshop24.highs.dev/tutorial)：C++/C 内核与 Python 接口共存，并提供 C API。借鉴接口分层和批量数据传递；不将其代码、求解结果、基或割接入 ZYO 自主链。
- [SCIP 官方接口文档](https://scipopt.org/doc/html/INTERFACES.php)：基础求解器与 Python 等语言接口分离，Python 可以扩展插件。借鉴清晰的回调契约和扩展边界；ZYO 的插件能力须逐项自主实现和验证。
- [VS Code 官方 GCC/GDB 配置指南](https://code.visualstudio.com/docs/cpp/config-mingw)：编译任务、调试配置和 IntelliSense 分离，作为本机工具配置依据。

这是公开架构资料学习，不是复制外部求解框架。引用资料与自行实现的验证证据分别存档。

## 迁移决策门

2026-09-08本机盘点：RTX5060Ti、16311MiB显存，NVIDIA驱动610.88。当前内核为CPU实现；本轮先优化已经剖析到的CSC组装开销。GPU候选先评估稀疏矩阵传输/组装、线性求解或一阶算法，分别记录端到端时间、CPU时间、RSS、显存与原始残差；基础数值库可用，外部优化器仍仅隔离比较。GPU kernel availability and speed require separate validation; hardware inventory is not solver capability.

1. 同一模型分别测建模、标准化、节点 LP、稀疏线性代数、搜索管理和结果检查；区分 Python 开销与已经在 BLAS/SuperLU 内执行的计算。
2. 先测试算法、矩阵复用或数据结构改进。只有实现语言确为瓶颈时才形成 C++ 候选。
3. 候选使用清晰的数组/稀疏矩阵接口；不要逐变量跨语言往返。原始数学、参数、停止状态、原生引擎来源和异常保持可追踪。
4. 同预算开发集核对正确性、端到端时间、内存及部署；保留失败和变慢实例。对调优用例与冻结保留集分别管理。
5. 只有收益足以覆盖构建/ABI/跨平台维护成本时才替换；NumPy、基础 C/C++ 数值库、Python 和后续语言均按此门评价。

English: retain a stable Python API and select implementation languages by measured end-to-end value. Profile before migration, compare identical models and tolerances, and validate packaging and error semantics. External optimizers remain isolated reference processes. This increment installs a C++ experimentation toolchain while keeping the existing optimization kernels unchanged.
