# 参与开发 / Contributing to ZYO

欢迎提交小规模、可独立复核的错误报告、测试和算法增量。当前为研究开发版，不以接口数量或代码量衡量成熟度。

开发基本原则：遇到正确性、数值或性能问题，首先检索原始文献与成熟商业求解器的公开方法，再提出实现假设。记录来源版本/章节、数学条件、与 ZYO 的差异及独立检查；以公开说明为依据区分已知行为与商业内部实现。

1. 从可解析问题、独立穷举或有来源的公开参考构造失败测试，先确认旧实现确实失败。
2. 说明数学适用条件、变量单位、容差、数值风险及反例，最小修改现有实现。
3. 自主求解不得调用外部优化引擎，也不得消费外部解、基、割或预处理结果。NumPy/SciPy 稀疏线性代数可用，依赖用途要记录。
4. 运行公开测试和针对性独立验证；报告全部失败、时间、节点、界和残差。限制状态不能冒充最优。
5. 自有代码写中文注释，重点说明公式、有效条件和失败处理；不要只逐句翻译代码。
6. 提交问题时注明源码提交、Python/依赖版本、算法路径、参数和最小输入。不要上传密钥、许可证或未经许可的研究数据。

公开包的测试是从本地开发集中筛选出的可公开子集，不等于完整内部回归，更不是最终保留测试集。新读过或调优过的实例必须归入开发集。

修改版本号时同时维护 `_zyo_version.py` 与 `pyproject.toml`；安装元数据尚未自动从单一变量读取。不要只改其中一个造成结果版本与安装版本不一致。

发布图按版本归档：每次新增图附在对应的 `docs/releases/<版本>.md` 小结，图放 `docs/figures/<版本>/`。旧小结与旧图保留原内容。更正图使用新的修订编号，并链接原图、说明更正依据。

## English

Literature-first diagnosis is required: consult primary research and public guidance from mature commercial solvers when a correctness, numerical or performance problem appears, before proposing implementation changes. Record source versions/sections, assumptions, applicability to ZYO and independent tests; distinguish documented behavior from undisclosed internals.

Start with an independent, reproducible failing example, then implement the smallest justified change. Explain mathematical validity, units, tolerances, numerical risks and counterexamples. Native algorithms must not invoke external optimization engines or consume their solutions, bases, cuts or presolve results. Basic NumPy/SciPy sparse linear algebra is allowed with a documented dependency boundary.

Run public tests and independent checks; retain all failures, timings, nodes, bounds and residuals. Resource-limit termination is not proven optimality. Write Chinese maintenance comments with English API names and module explanations for bilingual development. Include commit, Python/dependency versions, engine, parameters and a minimal public input in bug reports; never upload secrets or unauthorized research data.

The public tests are a publishable subset of local development tests, not the entire internal suite or a final holdout set. Once inspected or tuned against, an instance belongs to development data. Update both `_zyo_version.py` and `pyproject.toml` when changing versions: installation metadata is not yet derived automatically from the runtime version constant.

Archive each new figure under `docs/figures/<version>/` and include it in that release's `docs/releases/<version>.md` summary. Retain previous summaries and figures unchanged. Publish corrections under a new revision identifier with a link to the original and the reason for correction.
