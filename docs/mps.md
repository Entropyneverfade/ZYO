# 受控 MPS 读取 / Controlled MPS reader

```python
import zyo
model = zyo.Model.read("tests/fixtures/mps/hand_min.mps")
result = model.solve("native")
print(result.status, result.objective)  # 手算目标 / hand-computed objective: 15
```

也支持 `from zyo.mps import loads_mps, read_mps`。原 JSON 读取保持兼容；MPS 文件扩展名大小写兼容。

This release reads an explicit whitespace-delimited free subset using ZYO's own parser. JSON reading remains compatible and the `.mps` extension is case-insensitive.

支持单一 `NAME`、`OBJSENSE`（MIN/MAX）、`ROWS`（N/E/L/G）、`COLUMNS`、`RHS`、`BOUNDS`、`ENDATA`；节标题从首列开始，数据行缩进，整行 `*` 注释。每行系数/RHS 一或两组行名和值，列记录连续；数值支持 E/D 指数。目标常数为目标行 RHS 的负数。默认连续变量 `[0,+inf]`，INTORG 标记内部默认整数 `[0,1]`，可用显式边界修改。边界支持 LO/UP/FX/FR/MI/PL/BV/LI/UI。

Supported sections and bound codes are listed above. There is one objective and one RHS/bound set. Objective offsets use the negative objective RHS. Integer markers default to [0,1], following the documented dialect; explicit bounds override defaults. Infinite bounds retain their original mathematical meaning.

该首个子集明确拒绝：RANGES、SOS、二次/一般非线性、多目标、多 RHS/边界集、重复/覆盖式边界或系数、固定格式续行、包含空格的名称、NAME 附加说明字段。为避免方言差异，负 UP/UI 必须给显式下界。非法数值（包括极端指数导致的溢出/非零下溢）报带行号的 `ValueError`。`native_sparse` 仍要求有限盒；读入自由变量不意味着它能够求解。

Unsupported/ambiguous features fail explicitly with a line-numbered `ValueError`. Negative UP/UI requires an explicit lower bound. Fixed-field continuations, additional NAME metadata, ranges, nonlinear sections, multiple objectives/sets and overlapping data are outside this controlled subset. Parser acceptance and backend capability are separate.

语义来源：[Gurobi 官方 MPS 格式说明](https://docs.gurobi.com/projects/optimizer/en/current/reference/fileformats/modelformats.html)。来源用于核对格式，不调用其解析器或优化引擎。
