# 结果导向的实现语言策略 / Outcome-driven implementation languages

Python 科研接口 `import zyo` 保持稳定；实现语言服务于正确性、性能、维护和部署，不以某种语言的代码占比验收。

## 可借鉴的公开结构 / Public architecture references

- [HiGHS 官方接口教程](https://workshop24.highs.dev/tutorial)展示 C++/C 内核与 Python 接口的分层。ZYO 仅借鉴接口边界，不调用其优化引擎参与自主求解。
- [SCIP 官方接口文档](https://scipopt.org/doc/html/INTERFACES.php)展示基础求解器与语言接口分离。ZYO 的算法和扩展点逐项自主实现、验证。
- [VS Code GCC/GDB 配置指南](https://code.visualstudio.com/docs/cpp/config-mingw)说明编译、调试与 IntelliSense 配置。

English: keep the Python research API stable. Public solver documentation informs interface boundaries; third-party optimization engines remain isolated comparison references.

## 迁移决策门 / Migration decision gate

1. 对同一模型分段测建模、标准化、节点 LP、稀疏线性代数、搜索管理和结果检查；区分 Python 开销与 BLAS/SuperLU 内执行的计算。
2. 优先测算法、矩阵复用和数据结构。只有语言层确为瓶颈，才提出 C++ 候选。
3. 候选采用批量数组或稀疏矩阵接口，保持数学模型、参数、状态及异常可追踪。
4. 在同预算开发集与冻结保留集核对正确性、端到端时间、内存和部署，保留变慢与失败实例。
5. GPU 候选另测数据传输、组装、线性求解、原始残差与端到端收益，不根据硬件存在推断求解能力。

English: profile before moving a hotspot. Compare identical models, tolerances, full runtime and memory; preserve failures. CPU, C++ and future GPU work each require independently validated end-to-end benefit.
