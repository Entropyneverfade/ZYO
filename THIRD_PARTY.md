# 第三方与历史许可 / Third-party and historical notices

ZYO0.3.4采用分范围许可：受限新增内容依根LICENSE；历史MIT全文见LICENSES/MIT-legacy.txt，逐文件公开基线见LICENSES/MIT-BASELINE.json，范围见LICENSE-SCOPE.json。既有求解/应用内核保留MIT；原署名LZYOpt contributors继续保留。新条款不主张第三方代码或数据归Ziyuan Li独占。
ZYO0.3.4 uses scoped terms: root LICENSE for eligible additions, LICENSES/MIT-legacy.txt for inherited text, LICENSES/MIT-BASELINE.json for baseline identity and LICENSE-SCOPE.json for scope. Existing solver/application kernels retain MIT and the LZYOpt contributors attribution. New terms do not claim exclusive ownership of third-party code or data.

| 组件 / Component | 用途和许可边界 / Use and terms |
|---|---|
| NumPy | native数组/浮点；BSD-3-Clause，具体发行包所含库另按其声明 / arrays/floating point; BSD-3-Clause plus notices for components in the actual distribution |
| SciPy / SuperLU / BLAS / LAPACK | 稀疏矩阵与线性方程组，不调用scipy.optimize；SciPy BSD-3-Clause及各数值库自身声明 / sparse linear algebra, not scipy.optimize; SciPy BSD-3-Clause and component notices |
| Matplotlib / fonts | 可选绘图；Matplotlib及所用字体独立许可 / optional plotting under Matplotlib and individual font terms |
| HiGHS / Gurobi / COPT | 仅隔离比较；各自许可，不转授商用key / isolated comparison only; separate licenses, no commercial keys sublicensed |
| setuptools | 构建工具；自身及vendored组件声明 / build tool with its own and vendored notices |
| RTS24 / MATPOWER / UW / 论文与手册 | 原数据、图表、文献各按来源权利；MATPOWER代码许可不自动覆盖case数据 / data, figures and literature retain source-specific rights; MATPOWER code licensing does not automatically cover case data |

本wheel和sdist只分发ZYO源码与声明，不捆绑上述依赖、优化引擎、字体二进制、第三方原始算例或文献。pip安装依赖由其发行物携带各自许可；本表不是离线环境的完整SBOM。今后分发容器、离线环境或C++/GPU二进制时需按实际内容补齐全部版权和许可全文。
The wheel and sdist contain ZYO source and notices, not bundled dependencies, optimization engines, font binaries, third-party raw cases or literature. Dependencies installed by pip carry their own notices. This table is not a complete offline-environment SBOM; containers, offline environments and C++/GPU binaries require artifact-specific complete notices.

旧公开图和CSV按原版本原样保留，保留来源引用；本次许可切换不将它们纳入新增独占范围，也不扩大原来未确定的数据/字体再分发许可。纯依赖不等于版权转让。
Existing public figures/CSVs remain byte-identical with source references. This transition neither places them within newly claimed restricted ownership nor broadens unresolved data/font redistribution rights. Dependency use is not copyright assignment.

来源 / Sources: [NumPy](https://numpy.org/doc/stable/license.html), [SciPy](https://github.com/scipy/scipy/blob/v1.15.3/LICENSE.txt), [Matplotlib](https://matplotlib.org/stable/project/license.html), [MATPOWER8.1](https://github.com/MATPOWER/matpower/blob/8.1/LICENSE).
