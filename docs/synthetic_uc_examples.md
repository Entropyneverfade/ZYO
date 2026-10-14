# 独立生成的 24 小时机组组合算例 / Independently generated 24-hour UC cases

## 来源与用途 / Provenance and purpose

`examples/generate_synthetic_uc.py` 仅使用代码内写明的常数、三角函数和 Python `random.Random` 的固定种子生成输入；不读取真实系统、论文算例或私人文件。随附的 `balanced` 和 `tight_reserve` JSON 均由种子 `20261014` 生成，测试会逐字段核对生成器输出。它们是单节点、两台合成火电、风电、光伏、1 小时步长的**教学与数值边界算例**，不代表真实电网，也没有随机充裕度概率意义。

The generator reads no external case or private file. Its explicit constants, trigonometric profiles and fixed `random.Random` seed produce two reproducible teaching inputs. They describe a synthetic single-node, two-thermal-unit system with wind, solar and 24 hourly periods, not a real network or probabilistic adequacy study.

第 $t=0,\ldots,23$ 小时的需求和可用风光先按下式生成，再保留三位小数：

$$
\begin{aligned}
D_t&=75+14\cos\!\left(\frac{2\pi(t-19)}{24}\right)+3(U_{t,1}-1/2),\\
W_t&=\max\!\left\{0,17+5\cos\!\left(\frac{2\pi(t+3)}{24}\right)+4(U_{t,2}-1/2)\right\},\\
S_t&=24\max\!\left\{0,\sin\!\left(\frac{\pi(t-6)}{12}\right)\right\},\\
R_t&=5+0.06D_t.
\end{aligned}
$$

$U_{t,1},U_{t,2}$ 是该种子顺序产生的独立伪随机数。`tight_reserve` 在舍入前把 $D_t$ 增加 5 MW、把 $W_t$ 乘以 0.8，并把 $R_t$ 增加 3 MW。功率单位为 MW，时段为 1 h，因此各时段 MW 数值对应 MWh 电量；费用是统一的输入货币单位。原始数据、公式、舍入顺序和生成器代码共同定义算例，跨平台逐字节复现以随附 JSON 为准。

For the tight-reserve profile, the unrounded demand rises by 5 MW, available wind is scaled by 0.8, and required reserve rises by 3 MW. The displayed equations, seed, rounding order and source code define the example; the bundled JSON pins its exact values across environments.

## 优化模型 / Optimization model

两台机组的出力 $p_{g,t}$、开机 $u_{g,t}\in\{0,1\}$、启动 $v_{g,t}$、停机 $w_{g,t}$，以及实际利用风光 $P^W_t,P^S_t$ 共同满足：

$$
\begin{aligned}
\min\;&\sum_{t,g}(c_gp_{g,t}+f_gu_{g,t}+s_gv_{g,t}+h_gw_{g,t})+\sum_t C_{\rm ENS}\,ENS_t,\\
&\sum_g p_{g,t}+P^W_t+P^S_t+ENS_t=D_t,\qquad ENS_t=0,\\
&\underline P_g u_{g,t}\le p_{g,t}\le\overline P_g u_{g,t},\\
&0\le P^W_t\le W_t,\quad 0\le P^S_t\le S_t,\\
&\sum_g(\overline P_g u_{g,t}-p_{g,t})\ge R_t.
\end{aligned}
$$

还包含启停状态转换、首小时初始状态与出力、爬坡和最小开停机时间约束。`strict=true` 把缺电固定为零；风光可弃。备用只是在线容量裕度，不包括网络输送、亚小时爬坡可交付性或 N−1 安全。两个输入使用相同机组参数，因此紧备用案例集中检验更吃紧的负荷、风电与备用组合。详细约束语义见[机组组合说明](unit_commitment.md)。

The objective combines thermal energy, online-hour and start/stop costs. Balance includes utilized renewables; strict supply fixes ENS to zero. Commitment transitions, initial conditions, ramping and minimum up/down times are also enforced. Reserve is online capacity headroom only, not a deliverability or security guarantee.

## 运行、检查与输出 / Run, check and outputs

在已安装 `.[native-sparse]` 的项目根目录运行；每次给全新的结果目录。生成器只允许在项目目录内写新的 JSON，不覆盖既有输入。

```powershell
python -B examples/generate_synthetic_uc.py --seed 20261014 --profile balanced --output examples/my_synthetic_uc.json
python -B -m unittest tests.test_public_synthetic_uc -v
python -I examples/run_uc.py --input examples/synthetic_uc_balanced_20261014.json --output results-balanced --engine native_sparse
python -I examples/run_uc.py --input examples/synthetic_uc_tight_reserve_20261014.json --output results-tight-sparse --engine native_sparse
python -I examples/run_uc.py --input examples/synthetic_uc_tight_reserve_20261014.json --output results-tight-dense --engine native
```

`run_uc.py` 的原生进程阻止外部优化器导入与自动回退。`result.json` 保存实际引擎、版本、参数、输入/模型哈希、真实终止状态、目标/界、时长、模型残差、独立物理复核与成本重算；有可用解时另存逐时 `trace.csv/json`。返回码 0 仅表示该次结果经既有最优与独立检查门验收。其它求解器只能在另一个隔离进程做对照，不参与原生求解链。

The native runner blocks external optimizer imports and fallback. Its archived receipt includes the actual engine, input/model hashes, termination reason, objective/bound, independent original-model and physical residual checks, and hourly trace when available. A separate comparison process must never feed a result into the native chain.

2026-10-14 开发机的单次、单线程实测，两份输入均为 264 变量、48 二元变量、480 约束、1472 约束矩阵非零元：

| 场景 / Case | 引擎 / Engine | 真实状态 / Status | 目标 / Cost | 独立物理检查 / Physical check |
| --- | --- | --- | ---: | --- |
| `balanced` | ZYO `native_sparse` | `OPTIMAL` | 33254.471000569254 | 通过，ENS=0 / Pass |
| `tight_reserve` | ZYO `native_sparse`（原式校正增量） | `OPTIMAL` | 39856.801000227846 | 通过，ENS=0 / Pass |
| `tight_reserve` | ZYO `native` | `OPTIMAL` | 39856.80100000001 | 通过，ENS=0 / Pass |
| `tight_reserve` | 隔离 HiGHS 1.8.0 / isolated comparison | `OPTIMAL` | 39856.801 | 通过，ENS=0 / Pass |

原生稀疏法方程回退现在最多进行三次基于**原牛顿方向**残差的有界校正。若 $D=\operatorname{diag}(x/s)$、$N=ADA^\top$，每次解 $N\delta y=r_p-A\Delta x$，恢复 $\Delta x,\Delta s$ 后仍按原式验收；不会调宽可行性/互补方向门。上述紧备用模型经27个搜索节点、288次累计 LP 迭代得到原生整数解；独立原行残差 $6.83\times10^{-10}$、物理残差最大 $6.61\times10^{-10}$、归一化 Gap $7.98\times10^{-12}$。三轮固定种子逐小时前缀复核均为23/24最优，2h前缀仍返回 `NUMERICAL_ERROR`。这些是自写教学输入的局部实测；更换种子或参数后须重新运行独立物理与原模型检查。

The native fallback now applies at most three bounded corrections against the **reconstructed original Newton equations**, retaining their original acceptance gates. The authored full-day sparse MILP closes 27 nodes and passes independent row, physical and cost checks. Three repeated prefix scans each accept 23 of 24 prefixes; the two-hour prefix retains its numerical stop. These local teaching-case receipts are not a general solver ranking; new data require fresh model and physical audits.
