# 依赖边界 / Dependency boundary

ZYO 的自主算法代码在本项目中维护，内部保留旧 `lzyopt` 名称。独立求解并不意味着完全没有第三方基础数值库。

- `native`：NumPy 数组与基础浮点运算。
- `native_sparse`：NumPy、SciPy 稀疏矩阵与 `sparse.linalg.splu` / SuperLU，用于线性方程组；不使用 `scipy.optimize` 或其内嵌 HiGHS。
- 储能规则和物理检查使用 NumPy；可选 Matplotlib 仅绘图。
- 历史 HiGHS/Gurobi/COPT 适配器只保留显式、隔离比较用途，不能提供自主求解的解、基、割、界或回退。商业求解器与许可证不随本源码分发。

安装依赖仍受各自许可约束；本源码包不捆绑第三方二进制。再分发包含依赖的产品时须保留对应版权和许可。

## English

ZYO's independent algorithms are maintained in this project, retaining the legacy internal `lzyopt` name. Independent optimization does not mean dependency-free numerical computing.

- `native` uses NumPy arrays and floating-point operations.
- `native_sparse` uses NumPy, SciPy sparse matrices and `sparse.linalg.splu` / SuperLU for linear systems, not `scipy.optimize` or its embedded HiGHS engine.
- Storage rules and physical checks use NumPy. Optional Matplotlib is for plotting only.
- Legacy HiGHS/Gurobi/COPT adapters are exclusively for explicitly selected, isolated comparisons. They must not supply solutions, bases, cuts, bounds or fallback to native solving. Commercial optimizers and licenses are not distributed here.

Dependencies retain their own licenses. This source package does not bundle third-party binaries; redistributing a product with those dependencies requires preserving their notices and licenses.
