# ZYO API 使用条款 1.0.0 / API Access Terms 1.0.0

2026-09-08 · 许可人 / Licensor: Ziyuan Li（个人 / individual）

## 独立授权 / Separate authorization

软件许可与托管API访问分开。API权限仅在服务可用、资源与安全验收完成、许可人批准并由使用者接受具体授权记录后产生；源码或测试资格不自动产生密钥。目前这份条款定义将来的授权条件，服务上线及发钥需单独批准。

Software licensing and hosted API access are separate. API rights arise only after service readiness, resource/security validation, licensor approval and recipient acceptance of a specific grant. Source or software test eligibility does not create a key. These terms define future access conditions; deployment/key issuance requires separate approval.

## 申请与授权记录 / Request and grant

申请只填写概括用途。以既有平台账号/昵称或申请编号联系审批，不索取实名、邮箱、电话、单位、合同或证明材料。批准记录列用途、条款版本、起止时间/时区，默认30日并可申请续期。管理员填写调用额度、最大并发、模型/内存/运行时间限额及允许端点，用户接受后才可发钥；这些数值依实测容量确定，不以空值或猜测默认开放无限资源。

Requests contain only a purpose summary and use existing account/pseudonym or request references without real names, email, phone, affiliation, contracts or evidence. Grants specify purpose, terms version and start/end times/time zone, defaulting to30 days with renewal by approval. Administrators set call quotas, concurrency, model/memory/runtime limits and endpoints from measured capacity before assent and key issuance; missing limits are not unlimited access.

## 使用边界与撤销 / Boundaries and revocation

获批测试仅限授权账号和用途，禁止共享密钥、转售权限、代第三方提供服务或规避额度。商业、赞助科研、收费教学、生产及其他边界用途按软件LICENSE第3条另批；免费不自动属于非商业。许可人保留自有或有权授权服务的未来收费与商业化。许可人可按授权记录撤销访问或暂停异常任务，通知原申请渠道；安全事件、密钥泄露和持续滥用可即时停用。终止后不再接收新任务，现有任务、结果提取窗口按事先约定处理，不能承诺永久保留。

Approved access is limited to the account/purpose; no key sharing, resale, third-party services or limit circumvention. Commercial, sponsored, paid-teaching, production and boundary uses require separate approval under software LICENSE clause3; free does not automatically mean noncommercial. The licensor reserves commercialization of owned/authorized services. Access may be revoked or abnormal jobs suspended under the grant with notice through the request channel; security incidents, key leakage or ongoing abuse may require immediate suspension. No new jobs follow termination; existing jobs and result retrieval follow pre-agreed windows, without permanent-retention promises.

## 安全和数据 / Security and data

密钥按账号独立生成、仅安全渠道交付，服务端仅存校验所需哈希/标识，禁止公开Issue发钥或日志记录原始密钥。上线前确定TLS、认证、撤销、配额、隔离、输入限制和数据删除机制。安全交付渠道及备份/删除周期必须先验证，不能因免采集联系方式而将密钥公开。

Keys must be account-specific, delivered securely, stored server-side only as necessary hashes/identifiers and excluded from public issues and raw logs. Before deployment verify TLS, authentication, revocation, quotas, isolation, input limits and deletion. Validate credential delivery and retention/deletion periods first; avoiding contact collection does not justify public keys.

输入输出所有权不因调用自动转移给许可人；不把模型或结果默认用于训练。只保存执行、安全和计量所需的最少数据及请求标识，保留期和删除说明须在服务启用前展示。使用者不提交其无权交付的数据或密钥。授权及计量标识可能仍属个人信息，不宣称完全匿名。

Calls do not transfer ownership of inputs/outputs or authorize default training use. Retain only minimal execution/security/metering data and request references, and disclose retention/deletion before activation. Recipients must not submit unauthorized data or secrets. Authorization/metering references may still be personal data; complete anonymity is not claimed.

## 结果与责任 / Results and responsibility

结果记录实际引擎、版本、参数、终止状态、解/界/残差及资源。API封装不提升原生算法的数值保证；使用者独立核验结果。获准成果发表、免责、不可排除责任、历史权利及解释按软件LICENSE第1、5、8、9条，另行有效服务协议优先于其明确变更的部分。定制条款尚待独立专业法律审核。

Results record actual engine/version/parameters, termination, solution/bounds/residuals and resources. API packaging does not increase native numerical guarantees; recipients independently validate results. Software LICENSE clauses1,5,8,9 govern inherited rights, publication, disclaimers, non-excludable liability and interpretation, subject to expressly varied terms in a valid separate service agreement. Independent qualified legal review remains outstanding.
