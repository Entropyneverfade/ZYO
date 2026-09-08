# 版本记录 / Changelog

## 0.3.4 — 2026-09-08 · 正式许可切换 / Formal licensing transition

- 采用ZYO非商业测试许可1.0.0，登记许可人Ziyuan Li、日期、适用范围和商业化保留。
- 保存历史MIT全文、104文件公开基线及第三方边界；同步运行版本、SPDX混合元数据、安装教程、贡献规则和独立API条款。
- 将申请统一为单项用途摘要，按账号/申请批准，默认90日软件测试；增加正式许可导出和损坏产物负例检查。
- 本版交付与维护说明见[0.3.4小结](docs/releases/0.3.4.md)。

English: adopted scoped Noncommercial Testing License1.0.0 with licensor/date/commercialization reservation; retained MIT text, the104-file baseline and third-party boundaries; synchronized version, SPDX metadata, installation/contribution guidance and API terms; added purpose-only90-day account/request trials and formal-license export/negative checks. See the release summary.

## 2026-09-08 · 授权说明与用途申请 / Licensing guidance and purpose-only requests

- 明确Ziyuan Li对自有或有权授权内容保留商业许可、收费软件、API/SaaS等未来商业化安排。
- 发布中英文后续非商业测试/API方案、历史MIT与第三方权利说明及版本适用流程。
- 添加仅填写用途的申请表；使用平台账号或申请编号衔接审批和后续授权。
- 增加四文件政策增量导出及两项隔离导出/申请字段回归检查，软件版本沿用0.3.3。

English: documented the licensor's future commercial options and prospective testing/API policy with inherited-rights boundaries; added a purpose-only form and account/request-based approval references; added a four-file policy export with two isolation/form regressions, retaining software version0.3.3.

## 0.3.3 — 2026-09-08 · 局部稀疏结构复用 / Local sparse structural reuse

- 完成合成风光RTS24的24/168/8760小时原生DC/PWL调度与逐时检查；年度循环183.01秒，风光实发/弃电单列。
- 交付[本版九幅图及分类电量](docs/releases/0.3.3.md)，按版本保留原总量/节点图、类型图及新增风光堆叠图。
- Completed native24/168/8760-hour RTS24 DC/PWL dispatch with synthetic wind/PV and hourly checks; the annual loop takes183.01 seconds with explicit renewable generation and curtailment.
- Delivered[nine release-specific figures and type-level energy data](docs/releases/0.3.3.md), retaining earlier views alongside the new wind/PV stacks.

- 增加每个LP独立的增广CSC装配缓存，精确比对形状、列指针和行索引，逐轮刷新数值并保护输入/输出存储。
- 增加5项独立结构、数值和生命周期反例，完成233项研发回归及378题精确开发检查。
- 完成20题×2方法×3次结构复用消融，120次输出数值一致；结构重建17,394→1,965次，逐题中位时间之和减少3.00%。
- 补齐消融装配次数统计，并保持历史装配/行缩放比较的原操作定义。
- 更新双语贡献规范，将“遇到问题先查原始文献和成熟求解器公开方法”设为开发基本原则。

English:

- Added a per-LP augmented CSC assembly cache with exact shape/pointer/index invalidation, fresh values and isolated input/output storage.
- Added five independent structural, numerical and lifecycle regressions; passed233 development tests and378 exact development checks.
- Completed120 numerically identical calls across20 cases and two methods; reduced structural builds from17,394 to1,965 and the sum of per-case median times by3.00%.
- Recorded assembly counts while preserving historical assembly/row-scaling ablation definitions.
- Updated bilingual contribution guidelines with mandatory literature-first diagnosis using primary research and public commercial-solver methods.

## 0.3.2 — 2026-09-08 · 原方程校正与行缩放 / Original-equation refinement and row scaling

- 增加由原Newton方程残差触发的同LU有界校正，使用二进制精确解析方向构造回归，恢复原RTS24失败输入的自主求解。
- 实现直接CSC行缩放与混合类型精度提升，补充矩形、空维度、重复项、输入不变、整数及float32极小值测试。
- 消融报告绑定实际操作、题目、方法与代码身份，并加入标签篡改检测回归。
- 统一机组组合运行图的资源语义颜色与纹理，更新独立比较图和中英文结果说明。
- 通过219项完整研发回归，完成360普通小整数最优解和168份证书的独立精确核验。

English:

- Added bounded same-LU correction triggered by the original Newton equations, a binary-exact analytical regression, and native recovery of the retained RTS24 failure.
- Implemented direct CSC row scaling with promoted precision and independent shape, duplicate, input-preservation, integer and tiny-float32 checks.
- Bound ablation reports to the actual operation, case, method and code identity, with a tampered-label regression.
- Applied semantic resource colors and hatches to UC figures and refreshed the independently checked bilingual comparison.
- Passed219 development regressions and independently verified360 ordinary small integer optima and168 exact witnesses.

## 0.3.1 — 2026-09-08 · 密集换基与稀疏装配 / Dense pivots and sparse assembly

- 密集内核采用实际最小比值、大主元退化平局及真实换基RHS检查，增加五个解析机制反例与原24h UC回归。
- 同一24h UC模型密集/稀疏及三种隔离对照各三次OPTIMAL，密集成本28625、ENS0，完整物理与原始约束检查通过。
- 稀疏内核直接构造CSC增广矩阵，增加矩形/空维度/显式零/重复项/输入不变性测试。
- 完成20题×2方法×3次消融，状态、解、目标、界、节点和迭代逐项一致；七个背包MILP中位调用时间减少19%–23%。
- 原60题和另外300小整数题全部OPTIMAL，168份不可行证书完成Fraction精确复核；同步双语实测图表与公开测试清单。

English:

- Added exact-minimum ratio selection, stable degenerate ties and actual post-pivot RHS checks, with five analytical regressions and the unchanged 24-hour UC regression.
- Passed all three UC attempts for each of two native engines and three isolated references; dense cost28625 and ENS0 pass independent physical and original-model checks.
- Constructed the native augmented matrix directly in CSC, with rectangular, empty, explicit-zero, duplicate-entry and input-preservation tests.
- Completed120 controlled ablation runs with identical statuses, solutions, objectives, bounds, nodes and iterations. Median calls on seven knapsack MILPs decrease by19%–23%.
- Solved all360 ordinary small integer cases and exactly rechecked168 infeasibility witnesses; refreshed bilingual figures and the public test manifest.

## 0.3.0 — 2026-09-08 · 原生输入与机组组合 / Native input and unit commitment

- 增加自有 free MPS 受控解析器、Model.read 分派、目标常数和整数/边界处理，以及含极端指数的格式反例测试。
- 增加固定容量机组组合模型，覆盖逐时平衡、风光、启停、最小开停机、初始状态、爬坡和在线容量备用；独立复算原始功率界、成本及末端携带状态。
- 交付原生运行入口、24h 合成输入、逐时 CSV、双语运行图和五路径隔离比较图；15 次同模尝试和全部状态归档。
- ZYO 稀疏原生三次 OPTIMAL，目标约 28625.000009868，最大物理残差 2.177e-8，ENS 0；4h 手算题目标 43 自主验证通过。
- 完成 196 项研发回归，按功能里程碑同步 0.3.0 包版本、安装元数据与公开清单版本；维护双语教程和结果导向的语言策略。
- 独立非可编辑安装的74项公开测试、24h稀疏原生与4h密集原生运行/出图通过验收。

English:

- Added an owned controlled free-MPS parser, Model.read dispatch, objective constants, integer/bound semantics and extreme-exponent regressions.
- Added fixed-capacity UC with hourly balance, renewables, commitment transitions, minimum up/down times, initial state, ramping and online-capacity reserve; independently checked raw power bounds, cost and terminal carry-over state.
- Delivered a native-only runner, synthetic 24-hour input, hourly CSV, bilingual dispatch and isolated five-path comparison figures, with all 15 attempted runs archived.
- Sparse native passes all three attempts at objective approximately 28625.000009868, maximum physical residual 2.177e-8 and ENS 0; the four-hour hand fixture reaches cost 43 with native algorithms.
- Passed all 196 development regressions and synchronized package, installed metadata and export versions to 0.3.0; maintained bilingual tutorials and outcome-driven language policy.
- Passed 74 non-editably installed public tests and validated installed 24-hour sparse-native and four-hour dense-native examples with figures.

## 0.2.0 — 2026-09-08 · 原生 Phase-I 证书恢复 / Native Phase-I certificate recovery

- 自主稀疏LP增加有限盒弹性Phase-I、原始行乘子回映射和原始盒证书验收；原LP与辅助LP共享节点迭代预算及总截止时刻。
- 保存辅助实际状态、迭代轨迹、耗时、原失败原因及证书验证/采纳标志；补充证书检查超时的终止门。
- 增加9项回归，覆盖两个历史整数反例、等式/固定量映射、预算、正松弛反例、下溢和可选依赖；完整174项研发测试通过。
- 原60与另外300普通小整数题全部OPTIMAL，168份不可行证书经精确有理数复核。完成378题开启/关闭策略消融，逐题保存状态、时间、节点和迭代。
- 完成20题×5路径×3次同机比较，自主稀疏18有限题稳定通过17题；两个修复题各三次OPTIMAL。24/168/8760h合成储能各三次通过独立物理与成本检查。
- 同步公开测试白名单、中英文能力记录、原始比较日志字节捕获与本地复现文档。

English:

- Added native finite-box elastic Phase-I recovery, original-row multiplier mapping and original-box certificate validation, sharing the existing per-node iteration budget and total deadline.
- Recorded auxiliary status/history/runtime, original failure cause and separate certificate verification/acceptance; enforced the deadline after certificate checks.
- Added nine regressions covering retained integer failures, equality/fixed-variable mapping, budgets, positive-slack counterexamples, underflow and optional dependencies. All174 development tests pass.
- Solved all360 ordinary small development MILPs and exactly rechecked168 emitted certificates. Completed enabled/disabled audits on all378 prior cases with per-case status/time/node/iteration records.
- Completed a300-run controlled comparison: native sparse reaches17/18 stable finite-instance successes. Both repaired cases pass three repeats; synthetic24/168/8760-hour storage passes independent physical/cost checks.
- Maintained the public test manifest, bilingual capability notes, byte-preserving comparison-log capture and local reproduction documentation.

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
