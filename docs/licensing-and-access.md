# ZYO 授权与用途申请 / Licensing and purpose-only requests

2026-09-08 · 软件0.3.4 / Software0.3.4 · 正式条款1.0.0 / Formal terms1.0.0

## 正式许可 / Formal license

自0.3.4起，根[LICENSE](../LICENSE)正式采用“ZYO非商业测试许可1.0.0”，许可人Ziyuan Li（个人），采用日期2026-09-08（Asia/Shanghai）。这是源码可用、受控非商业测试方案，不是标准开源许可。[范围清单](../LICENSE-SCOPE.json)登记受限新增表达及继承边界。专业法律审核尚未完成，特别是双语解释、接受证据、专利、限责和终止条款，建议由合格法律专业人士复核。

From0.3.4 the root [LICENSE](../LICENSE) formally adopts ZYO Noncommercial Testing License1.0.0, licensor Ziyuan Li (individual), adopted2026-09-08 Asia/Shanghai. It is source-available controlled noncommercial testing, not standard open source. The scope schedule identifies eligible additions and inherited boundaries. Qualified legal review remains outstanding, especially language, assent, patents, liability and termination.

## 你可以如何测试 / How to request testing

[用途申请 / Purpose-only request](https://github.com/Entropyneverfade/ZYO/issues/new?template=zyo-test.yml)

只写概括用途，例如：“用公开/合成小模型学习LP/MILP并比较结果，属于非商业测试。”无需实名、联系方式、单位、真实项目名、资助方或合同/证件/数据证明。研究和边界用途只说明性质。平台账号或申请编号（允许昵称）用于衔接审批；公开Issue会显示平台账号、时间和内容，请勿填写隐私或密钥，不宣称完全匿名。

Write only a purpose summary, such as learning LP/MILP and comparing public/synthetic models for noncommercial testing. No identity, contact, affiliation, identifying project/sponsor names, contracts, documents or datasets are requested. Describe only the research/boundary-use category. Existing platform accounts/pseudonyms or request IDs connect approvals. Public issues show account/time/content; do not submit private data or keys, and do not assume complete anonymity.

许可人回复批准范围、版本、条款和起止时间，申请人在原渠道明确接受后，才开始运行受限新增内容。默认90日，可申请续期；期限及授权记录由管理员填写，不追加隐私问卷。仅查看、下载、fork或发起申请不自动取得运行/生产/商用权限。GitHub适用条款实际授予的平台查看/fork权利保留。

The licensor replies with scope, version, terms and start/end times; explicitly assent through the same channel before running restricted additions. Default90 days, renewable by approval; administrators complete the grant without another privacy questionnaire. Viewing, downloading, forking or applying alone does not confer execution/production/commercial rights. Applicable GitHub viewing/forking permissions remain preserved.

| 用途 / Purpose | 授权方式 / Permission |
|---|---|
| 个人学习、功能评估、性能测试 / Learning, evaluation, benchmarking | 账号/申请批准后在非生产环境测试，默认90日 / approved account/request, non-production, default90 days |
| 非商业科研 / Noncommercial research | 对概括科研用途明确批准，不索取真实项目资料 / explicit approval of generic research purpose, no identifying records |
| 商用、生产、广告引流、商业配套 / Commercial, production, advertising/acquisition, ancillary | 另行书面许可 / separate written license |
| 赞助/委托/混合资助、收费教学、雇主评估 / Sponsored/commissioned/mixed-funded, paid education, employer evaluation | 按用途性质个案批准 / case-specific purpose approval |
| 本地研究修改 / Local research modification | 获批范围内允许，保留声明和修改记录 / allowed within grant, retain notices/change record |
| 受限修改版再分发、转授权、第三方服务 / Restricted redistribution, sublicensing, services | 另行书面许可 / separate written permission |
| API / Hosted access | [独立条款](API-TERMS.md)，规划30日；实际服务/安全/资源验收及授权后发钥 / separate terms, planned30 days; keys follow service/security/resource validation and approval |

## 许可人商业化与历史MIT / Commercialization and inherited MIT

Ziyuan Li保留自有或有权授权内容的商业许可、软件/订阅销售、商业集成、收费求解/API/SaaS与咨询。接收者的非商业测试限制不约束许可人自己的商业化；也不自动向别人授予同等商业权限。

Ziyuan Li retains commercial licensing, software/subscription sales, integration, paid solving/API/SaaS and consulting for owned/authorized material. Recipient test restrictions do not limit the licensor's own commercialization or grant equivalent commercial rights to others.

历史0.3.3及此前MIT材料继续依[原MIT](../LICENSES/MIT-legacy.txt)，无需新测试申请；包括既有求解内核。清单中已公开的104个文件身份见[基线](../LICENSES/MIT-BASELINE.json)。新条款只覆盖有权授权的新增表达，不能禁止他人按原MIT使用旧代码。第三方原权利保留。当前包的“新许可 AND MIT”表示各部分分别遵守，不是给受限新增提供MIT任选，也不是给旧MIT叠加限制。

Historical0.3.3 and earlier MIT material, including existing solver kernels, retains the original MIT without a new application. The baseline records104 prior public file identities. New terms cover eligible new expression only and do not prohibit use of old code under MIT; third-party rights remain. The package's “new license AND MIT” identifies respective component terms, not an MIT alternative for restricted additions or extra restrictions on old MIT.

成果和负面评价可按获批用途发表，需保留合理版本/引擎/数据来源说明。许可人不因运行或反馈取得他人输入、输出、修改版权。参见[正式条款](../LICENSE)、[第三方说明](../THIRD_PARTY.md)、[贡献规则](../CONTRIBUTING.md)及[本版更新](releases/0.3.4.md)。

Results and negative evaluations may be published for approved purposes with reasonable version/engine/source attribution. Execution or feedback does not assign others' inputs, outputs or modifications. Consult the formal terms, third-party notices, contribution rules and release summary.

参考 / References: [MIT](https://opensource.org/license/mit), [许可变更指南 / license changes](https://opensource.guide/legal/#what-if-i-want-to-change-the-license-of-my-project).
