"""绘制微小未定价改善与窄变量盒的两个精确手算反例。"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from decimal import Decimal
from pathlib import Path

FIGURE = Path(__file__).resolve().parents[1]
# 绘图库缓存只落在本图目录，发布时由本图 .gitignore 排除。
os.environ["MPLCONFIGDIR"] = str(FIGURE / "qa" / "mplconfig")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import FancyBboxPatch


# 筛选发行树比本地 docs/public 少一层；按包元数据向上定位，避免读取错误环境。
SOURCE = next((parent for parent in FIGURE.parents if (parent / "pyproject.toml").is_file()), None)
if SOURCE is None:
    raise RuntimeError("请在含 pyproject.toml 的 ZYO 源码树内重绘此图")
TEST = SOURCE / "tests" / "test_sparse_simplex_unpriced_gap.py"
STEM = "Fig01_UnpricedNarrow"
CASES = (
    {"case": "wide_box", "coefficient": Decimal("-1e-10"),
     "upper": Decimal("1e12"), "old": "FALSE_OPTIMAL_AT_ZERO",
     "current": "NUMERICAL_ERROR_AT_ZERO"},
    {"case": "narrow_box", "coefficient": Decimal("-1e15"),
     "upper": Decimal("1e-13"), "old": "FALSE_OPTIMAL_AT_ZERO",
     "current": "OPTIMAL_AT_UPPER"},
)


def sha(path: Path) -> str:
    """对源文件原始字节记录哈希。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_data() -> tuple[list[dict[str, str]], Path]:
    """用十进制精确乘法复算两条成本直线，不从求解器输出反造预期。"""
    rows: list[dict[str, str]] = []
    for case in CASES:
        coefficient = case["coefficient"]
        upper = case["upper"]
        if coefficient * upper != Decimal("-100"):
            raise RuntimeError("手算终点与 -100 不一致")
        for step in range(5):
            fraction = Decimal(step) / Decimal(4)
            x = fraction * upper
            objective = coefficient * x
            rows.append({
                "case": case["case"],
                "coefficient": str(coefficient),
                "x_lower": "0",
                "x_upper": str(upper),
                "fraction_of_upper": str(fraction),
                "physical_x": str(x),
                "objective": str(objective),
                "pre_fix_status_at_zero": case["old"],
                "current_native_status": case["current"],
            })
    path = FIGURE / "source_data" / "analytic_cost_paths.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows, path


def draw(rows: list[dict[str, str]]) -> dict[str, str]:
    """目标曲线共用纵轴尺度，界域在各面板分别标清。"""
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
        "savefig.facecolor": "white",
    })
    chinese = FontProperties(family="SimSun")
    latin = FontProperties(family="Times New Roman")
    findfont(chinese, fallback_to_default=False)
    findfont(latin, fallback_to_default=False)
    red, teal, navy, gray = "#A94945", "#137F76", "#274B68", "#697781"
    fig = plt.figure(figsize=(190 / 25.4, 130 / 25.4), facecolor="white")
    axes = (fig.add_axes([0.115, 0.43, 0.345, 0.345]),
            fig.add_axes([0.61, 0.43, 0.345, 0.345]))
    titles = ("(a) Wide box: c = -1e-10, U = 1e12",
              "(b) Narrow box: c = -1e15, U = 1e-13")

    for index, (ax, case) in enumerate(zip(axes, CASES)):
        selected = [row for row in rows if row["case"] == case["case"]]
        u = [float(row["fraction_of_upper"]) for row in selected]
        cost = [float(row["objective"]) for row in selected]
        if u != [0.0, 0.25, 0.5, 0.75, 1.0] or cost != [0, -25, -50, -75, -100]:
            raise RuntimeError("待绘图数据与手算两点连线不一致")
        ax.plot(u, cost, color=gray, linewidth=1.5, zorder=2)
        # 原点对应旧错误声明；大空心方框标识当前保守拒绝时仍处于原点。
        if index == 0:
            ax.scatter([0], [0], marker="s", s=225, facecolors="none",
                       edgecolors=navy, linewidths=1.5, clip_on=False, zorder=4)
        ax.scatter([0], [0], marker="X", s=90, color=red,
                   clip_on=False, zorder=5)
        # 右面板的双环是同一物理点：当前原生结果与解析最优重合。
        ax.scatter([1], [-100], marker="o", s=90, color=teal,
                   clip_on=False, zorder=5)
        if index == 1:
            ax.scatter([1], [-100], marker="o", s=180, facecolors="none",
                       edgecolors=navy, linewidths=1.4, clip_on=False, zorder=4)
        # 留出显示边缘避免端点符号压住轴与数字；额外端点也有无标签实刻度。
        ax.set_xlim(-0.08, 1.08)
        ax.set_ylim(-108, 8)
        ax.set_xticks([-0.08, 0, 0.25, 0.5, 0.75, 1, 1.08])
        ax.set_xticklabels(["", "0", "0.25", "0.5", "0.75", "1", ""])
        ax.set_yticks([-108, -100, -75, -50, -25, 0, 8])
        ax.set_yticklabels(["", "-100", "-75", "-50", "-25", "0", ""])
        ax.set_xlabel("x / U (physical upper bound U)", fontproperties=latin, fontsize=8)
        if index == 0:
            ax.set_ylabel("Objective c x", fontproperties=latin, fontsize=8.5)
        ax.grid(color="#E4E9ED", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_position(("outward", 3))
        ax.spines["left"].set_position(("outward", 3))
        ax.tick_params(axis="both", direction="in", length=4.5, width=0.8, pad=3)
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontproperties(latin)
            tick.set_fontsize(8)
        fig.text(0.285 if index == 0 else 0.785, 0.808, titles[index],
                 ha="center", fontproperties=latin, fontsize=8.7)

    def status_box(left: float, lines: tuple[tuple[str, str, str], ...]) -> None:
        panel = fig.add_axes([left, 0.130, 0.365, 0.180])
        panel.set_axis_off()
        box = FancyBboxPatch((0.02, 0.05), 0.96, 0.9,
                             boxstyle="round,pad=0.004,rounding_size=0.03",
                             transform=panel.transAxes, facecolor="#F6F8F9",
                             edgecolor="#B7C3C9", linewidth=0.9)
        panel.add_patch(box)
        for y, (zh, en, color) in zip((0.77, 0.50, 0.23), lines):
            panel.text(0.055, y, zh, fontproperties=chinese, fontsize=8.1,
                       color=color, transform=panel.transAxes)
            panel.text(0.30, y, en, fontproperties=latin, fontsize=7.8,
                       color=color, transform=panel.transAxes)

    status_box(0.105, (("旧状态", "false OPTIMAL, cost 0", red),
                       ("现原生", "NUMERICAL_ERROR at 0", navy),
                       ("解析最优", "x = 1e12, cost -100", teal)))
    status_box(0.595, (("旧状态", "false OPTIMAL, cost 0", red),
                       ("现原生", "OPTIMAL at upper, -100", navy),
                       ("解析最优", "x = 1e-13, cost -100", teal)))

    fig.text(0.5, 0.95, "定价门限与窄盒界侧：两种一百单位目标差", ha="center",
             fontproperties=chinese, fontsize=11)
    fig.text(0.5, 0.907, "Tiny unpriced improvement and narrow bound side: two analytic gaps",
             ha="center", fontproperties=latin, fontsize=9.5)
    fig.text(0.5, 0.338,
             "左图原生安全状态保留在起点；右图原生到达解析终点。横轴均为各自物理上界 U 的比例。",
             ha="center", fontproperties=chinese, fontsize=7.9)
    fig.text(0.5, 0.080,
             "手算模型，无求解速度比较；旧假最优、新状态与解析最优分别核对，未外推一般 LP 能力。",
             ha="center", fontproperties=chinese, fontsize=7.9)
    fig.text(0.5, 0.042,
             "Analytic one-variable cases; no solver speed comparison or general LP claim.",
             ha="center", fontproperties=latin, fontsize=7.3)
    images = FIGURE / "images"
    paths = {ext: images / f"{STEM}.{ext}" for ext in ("png", "pdf", "svg")}
    fig.savefig(paths["png"], dpi=600)
    fig.savefig(paths["pdf"])
    fig.savefig(paths["svg"])
    plt.close(fig)
    return {ext: sha(path) for ext, path in paths.items()}


def main() -> None:
    rows, data_file = source_data()
    images = draw(rows)
    qa = {
        "actual_backend": "Python / matplotlib Agg fallback",
        "fallback_reason": "Same-day Origin native export carried a demo watermark over plotted data",
        "origin_trial": "Demo-watermarked trial is retained in the private research archive",
        "analytic_input": [{"case": row["case"], "coefficient": str(row["coefficient"]),
                            "upper": str(row["upper"]), "current_status": row["current"]}
                           for row in CASES],
        "source_data_sha256": sha(data_file),
        "source_test_relative": "tests/test_sparse_simplex_unpriced_gap.py",
        "source_test_sha256_at_render": sha(TEST),
        "script_sha256": sha(Path(__file__)),
        "figure_sha256": images,
        "declared_axes": {
            "panel_a_x_display": [-0.08, 1.08], "panel_a_x_end_ticks": [-0.08, 1.08],
            "panel_a_y_display": [-108, 8], "panel_a_y_end_ticks": [-108, 8],
            "panel_b_x_display": [-0.08, 1.08], "panel_b_x_end_ticks": [-0.08, 1.08],
            "panel_b_y_display": [-108, 8], "panel_b_y_end_ticks": [-108, 8],
        },
        "scope": "Two analytic one-variable LP witnesses, not a benchmark",
        "visual_review": "PENDING final image and PDF raster inspection",
        "origin_processes_after": "PENDING process check",
    }
    path = FIGURE / "qa" / "figure_qa.json"
    path.write_text(json.dumps(qa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figures": images, "source_data_sha256": qa["source_data_sha256"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
