# 版本记录 / Changelog

## 0.2.0 — 2026-09-07 · 研究开发版 / Research development build

这是当前代码的首次 GitHub 公开源码整理，不是一个新内核版本或商业级稳定版。

- 统一 `zyo`/`ZYO` Python API，保留内部 `lzyopt` 兼容层。
- 自编密集单纯形/分支定界，及实验性有限盒稀疏内点法/分支定界。
- 单节点储能规则、周期迭代、调度建模及独立物理检查。
- 中英文 README、安装教程、限制说明、依赖说明和贡献指南；自有代码中文维护注释。
- 公开 44 项测试方法，其中一个方法覆盖 60 个独立穷举预期的小整数开发模型；不是 104 个互不重叠测试方法，也不是未见保留集。
- 发布整理沿用原有算法，未通过修改模型、容差或引擎来提高测试成绩。

验证环境：Windows、Python 3.10.19、NumPy 2.2.6、SciPy 1.15.3。已用非可编辑安装在源码目录外运行公开测试，44 通过、0 跳过。更广泛数值问题、一般不可行检测和规模限制见 [能力说明](docs/limitations.md)。

### English

This is the first screened GitHub source publication of the current code, not a new kernel version or an industrial stable release.

- Unified `zyo`/`ZYO` Python API, retaining the internal `lzyopt` compatibility layer.
- Independent dense simplex/branch-and-bound and experimental finite-box sparse interior-point/branch-and-bound.
- Single-node storage rules, periodic iteration, optimization modeling and independent physical checks.
- Bilingual README, installation tutorial, limitations, dependency notes and contributing guide; Chinese maintenance comments in owned code.
- 44 public test methods; one covers 60 small integer development models with independently enumerated expected values. Do not count this as 104 disjoint test methods or an unseen holdout set.
- Publication retains the existing algorithms, models, tolerances and explicit engine selection.

Verified with Windows, Python 3.10.19, NumPy 2.2.6 and SciPy 1.15.3: a non-editable installation ran the public suite outside the source directory, with 44 passes and no skips. See [limitations](docs/limitations.md) for unresolved numerical issues and unsupported scope.
