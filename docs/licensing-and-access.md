# ZYO 授权与用途申请 / Licensing and purpose-only requests

2026-09-08 · 政策说明 D2 / Policy statement D2

## 当前版本与后续方案 / Current release and prospective policy

当前公开 **ZYO 0.3.3** 的正式许可证仍为 [MIT](../LICENSE)，依该许可使用当前公开材料不需要申请。下述受控测试及API安排是后续方案，正式生效前将固定适用版本、提交、日期和内容范围，并完成权属、法律及安装包一致性审核。申请是用途登记，不自动产生新许可、API访问或交付承诺。

Public **ZYO0.3.3** remains under [MIT](../LICENSE); use of current public material under that license does not require an application. The controlled-testing/API arrangements below are prospective. Before activation, identify the covered version, commit, date and material and complete provenance, legal and package-consistency checks. A request records intended use; it does not automatically create a grant, API access or a delivery commitment.

## 许可人保留未来商业化 / Licensor's future commercial rights

**Ziyuan Li（个人）**保留对其自有或有权授权内容自行使用、开发、许可和商业化的权利，包括销售软件或订阅、商业产品集成、收费求解/API/SaaS、咨询服务，以及分别向不同使用者提供商业许可和非商业测试许可。给某位使用者的非商业测试权限不限制许可人自己的商业化，也不自动让其他使用者获得同等商业权限。

**Ziyuan Li, an individual**, reserves the right to use, develop, license and commercialize material he owns or is authorized to license, including software/subscription sales, commercial integration, paid solving/API/SaaS, consulting and separate commercial or noncommercial-test grants for different recipients. A recipient's noncommercial test grant does not limit the licensor's own commercialization or automatically extend commercial rights to other recipients.

上述保留不追溯收回历史MIT权利，不主张独占第三方代码、数据或已有授权内容。混合版本将保留来源及必要声明，新增受限范围单独识别；不能用整个新包的“MIT OR 测试许可”标签意外为全部新增内容提供MIT选项。后续代码贡献须确认许可人有权按计划分发，提交反馈或PR本身不自动转让版权。[MIT原文](https://opensource.org/license/mit)及[许可变更指南](https://opensource.guide/legal/#what-if-i-want-to-change-the-license-of-my-project)。

These reservations do not retract historical MIT grants or claim exclusive ownership of third-party or already-licensed material. Mixed releases preserve provenance/notices and identify restricted additions separately; a blanket “MIT OR testing license” label would unintentionally offer an MIT alternative for all additions. Future contributions require appropriate distribution rights; feedback or a PR does not automatically assign copyright.

## 申请只需说明用途 / Only intended use is requested

[打开用途申请表 / Open the purpose-only form](https://github.com/Entropyneverfade/ZYO/issues/new?template=zyo-test.yml)

只写一段概括用途即可，例如：“希望用公开或合成小模型学习LP/MILP，并比较求解结果，属于非商业测试。”若希望申请未来API试用，可在同一段用途内说明。科研可概括计算任务，不必提交论文、项目真实名称、合同或数据。

Write one short purpose summary, for example: “I would like to learn LP/MILP using public or synthetic small models and compare solutions for noncommercial testing.” Mention prospective API use in that same summary if relevant. Research purposes may be described generically; no paper, identifying project name, contract or dataset is needed.

- ZYO申请表不索取真实姓名、邮箱、电话、地址、身份证件、单位/学校、雇主或证明材料，也不要求追加联系方式。
- 审批使用申请所在平台账号（可用昵称）及申请编号衔接；这些标识用于区别申请，不作实名核验。不把昵称宣称为法律身份已认证。
- 公开Issue会显示GitHub账号、时间及提交内容；GitHub自身的数据处理由其平台政策决定。请勿填写隐私、研究秘密、密码或API密钥。平台标识本身也按最少必要原则处理，不宣称全匿名。
- 申请时只登记用途；实际授权的版本、期限、条款和接受记录由后续审批流程补齐，不让申请人填写额外隐私表格。未来密钥交付渠道另行安全设计，密钥不在公开Issue发送。

English: the ZYO form requests no real name, email, phone, address, identity documents, affiliation, employer or supporting records. Use the existing platform account/pseudonym and request number to distinguish requests, not to claim verified legal identity. GitHub publicly shows the account, time and submitted text and handles platform data under its own policies; do not submit sensitive data or credentials. Keep platform references minimal rather than claiming complete anonymity. Version, term, terms and assent are handled later without an additional privacy questionnaire. Future credential delivery requires a separately secured channel, never a public issue.

## 后续默认规则 / Prospective defaults

| 项目 / Item | 安排 / Arrangement |
|---|---|
| 下载测试 / Software trial | 按申请或账号审批；个人学习、功能评估、性能测试；默认90日，可申请续期 / request/account approval for learning, evaluation and benchmarking; default90 days, renewal by request |
| 非商业科研 / Noncommercial research | 明确批准概括用途和可识别申请；不要求提交隐私项目材料 / explicit approval of the summarized purpose and identifiable request, without private project records |
| 本地修改 / Local modification | 获批测试范围内允许研究修改和必要备份，保留来源及修改说明 / approved local research changes and necessary backups with notices |
| 商业及边界用途 / Commercial and boundary uses | 获授权人商用、广告/引流、商业配套、赞助/横向科研、收费教学及企业评估另批；申请只说明用途性质 / recipient commercial, advertising/acquisition, ancillary, sponsored/commissioned, paid-teaching and employer use requires separate approval; describe only the purpose category |
| 再分发与第三方服务 / Redistribution and services | 对受限新增内容另行书面授权；继承MIT/第三方部分依其原许可 / separate written permission for restricted additions; inherited licenses govern their own portions |
| API / Hosted API | 独立申请/账号授权；规划30日、可撤销；禁止共享密钥或转售；额度及资源在服务验证后确定 / separate request/account grant, planned30 days, revocable, no key sharing/resale; limits finalized after service validation |

这些管理默认不是离线到期锁，也不构成正在运行的API服务。客观测试结果可按获准用途发表，包含负面结果；许可人不因求解或反馈自动取得他人输入、输出或修改的所有权。

These defaults are not an offline expiry lock or a currently operating API service. Objective results, including negative findings, may be published for the approved purpose; solving or feedback does not automatically transfer ownership of another party's inputs, outputs or modifications.

本页用于说明政策与申请流程，不替代正式许可证；自定义条款需要专业法律审核，尤其是匿名/昵称授权的接受证据、适用范围、终止与责任条款。后续采用“非商业测试许可 / 源码可用”名称，适用版本与条款将单独公开。

This policy/application guide is not a replacement license. Custom terms require qualified legal review, particularly assent evidence for pseudonymous requests, scope, termination and liability. Future restricted offerings will be described as noncommercial testing / source-available with their exact applicable versions and terms.
