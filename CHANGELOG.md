# 版本记录 / Changelog

## 0.2.0 — 2026-09-08 · 运输 LP 数值修复 / Transportation LP numerical fix

- 自主稀疏内核增加初始分解风险筛选、稳定化增广分解及原方程迭代校正；近相关满秩问题保留原分解候选并逐项核验方向。
- 记录实际线性系统、正则量、校正次数及原分解候选使用次数。
- 新增 4 项独立回归：最优值 88 的运输 LP、一般相关行、一致性拒绝、近相关满秩方向。
- 10×10、30×30 运输 LP 各三次 OPTIMAL，目标 182、172；原始最大行残差分别约 1.55e-15、1.91e-14。求解调用中位耗时 0.01446、0.07103 秒。
- 完成 20 题×3 次自主重测；18 个有限最优题三次稳定通过数由 13 增至 15。24/168/8760 小时合成储能各三次求解及独立逐时物理、成本复核通过。
- 完成原 60 个整数题和另外 318 个开发反例的精确预期检查，验证 161 份分离证书。完整研发回归 147 项通过。
- 合并作者在 GitHub 的项目介绍修改，同步维护中英文说明、安装教程及公开测试清单。
- 独立非可编辑安装的公开 48 项测试、4 段教程、解析 LP/整数分支/两小时储能自主检查通过。

环境：Windows、Python 3.10.19、NumPy 2.2.6、SciPy 1.15.3，数值库单线程。测评使用原 10 秒求解预算、45 秒进程预算、4 GiB 采样内存上限；模型、参数、三次重复和原始结果逐项归档。

### English

- Added initial factorization-risk screening, stabilized augmented factorization and original-equation refinement to the native sparse kernel. Nearly dependent full-rank systems retain an original-factorization candidate checked against the same direction equations.
- Recorded the actual linear system, regularization, refinement count and use of the original-factorization candidate.
- Added four independent regressions: the objective-88 transportation LP, general dependent rows, inconsistent equations and a nearly dependent full-rank direction.
- Both transportation LPs pass all three repeats: 10×10 at objective 182 and 30×30 at 172. Maximum original row residuals are approximately 1.55e-15 and 1.91e-14; median solve-call times are 0.01446 and 0.07103 seconds.
- Replayed 20 instances three times with the native sparse kernel. Three-repeat successes on 18 finite-optimum instances increase from 13 to 15. The synthetic 24/168/8760-hour storage cases each pass three solves and independent hourly physical/cost checks.
- Checked the original 60 integer cases and 318 additional development/adversarial cases against exact expectations; verified 161 separation certificates. The full development suite passes 147 tests.
- Merged the author's GitHub introduction edits and maintained bilingual documentation, installation instructions and the public test manifest.
- Verified a non-editable public installation with 48 public tests, four tutorial blocks and native analytical LP/integer-branch/two-hour storage checks.

Environment: Windows, Python 3.10.19, NumPy 2.2.6, SciPy 1.15.3, single-thread numerical libraries. The assessment retains its 10-second solver budget, 45-second process budget and 4 GiB sampled memory limit; inputs, parameters, repeats and raw outputs are archived.

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
