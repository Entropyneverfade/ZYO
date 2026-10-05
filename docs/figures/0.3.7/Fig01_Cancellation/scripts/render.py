"""绘制二进制浮点乘积抵消的解析反例，并保留可复算的源数据。"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from fractions import Fraction
from pathlib import Path

FIGURE = Path(__file__).resolve().parents[1]
# 绘图库缓存只写入本图目录；发布时由本图 .gitignore 排除。
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
STEM = "Fig01_Cancellation"
TEST = SOURCE / "tests" / "test_lp_certificate_product_cancellation.py"
TOLERANCE = Fraction(1, 10_000_000)
STEP = 1.23456789012345
ROWS = (
    ("row_1", 1421803362854.3792, 1151660734277.0),
    ("row_2", 2029678531776.0918, 1644039625535.0),
)


def sha(path: Path) -> str:
    """用原始文件字节记录可复核身份。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_data() -> list[dict[str, str]]:
    """分别算先舍入乘积的残差和原存储浮点数的精确残差。"""
    output: list[dict[str, str]] = []
    for name, x, d in ROWS:
        old = x - d * STEP
        exact = Fraction.from_float(x) - Fraction.from_float(d) * Fraction.from_float(STEP)
        if old != 0.0 or abs(exact) <= TOLERANCE:
            raise RuntimeError("解析反例或门限不符，停止绘图")
        output.append({
            "row": name,
            "stored_x_repr": repr(x),
            "stored_x_hex": x.hex(),
            "stored_d_repr": repr(d),
            "stored_d_hex": d.hex(),
            "stored_step_repr": repr(STEP),
            "stored_step_hex": STEP.hex(),
            "rounded_product_first_residual": repr(old),
            "exact_stored_float_residual_fraction": str(exact),
            "exact_stored_float_residual_decimal": format(float(exact), ".17g"),
            "absolute_residual_over_tolerance": format(float(abs(exact) / TOLERANCE), ".17g"),
        })
    path = FIGURE / "source_data" / "analytic_rows.csv"
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    return output


def draw(rows: list[dict[str, str]]) -> dict[str, str]:
    """定量显示两行残差，再并列显示两种证书判定路径。"""
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
    values = [float(row["exact_stored_float_residual_decimal"]) / 1e-5 for row in rows]

    fig = plt.figure(figsize=(190 / 25.4, 118 / 25.4), facecolor="white")
    ax = fig.add_axes([0.105, 0.34, 0.41, 0.45])
    # 接受区间真实为 ±1e-7，即本图显示尺度上的 ±0.01。
    ax.axhspan(-0.01, 0.01, color="#E9F3F1", zorder=0)
    ax.axhline(0, color=gray, linewidth=0.7, zorder=1)
    ax.vlines([1, 2], [0, 0], values, color=teal, linewidth=2.0, zorder=2)
    ax.scatter([1, 2], values, s=68, marker="o", color=teal,
               label="Exact stored floats", zorder=4)
    ax.scatter([1, 2], [0, 0], s=76, marker="D", facecolors="white",
               edgecolors=red, linewidths=1.6, label="Rounded product first", zorder=5)
    for x, y in zip((1, 2), values):
        ax.annotate(f"{y:+.5f}", (x, y), xytext=(7, 4 if y > 0 else -12),
                    textcoords="offset points", color=teal, fontsize=8,
                    fontproperties=latin)
    ax.set_xlim(0.5, 2.5)
    ax.set_ylim(-4.0, 4.0)
    ax.set_xticks([0.5, 1.0, 2.0, 2.5])
    ax.set_xticklabels(["", "Row 1", "Row 2", ""], fontproperties=latin, fontsize=8)
    ax.set_yticks([-4, -2, 0, 2, 4])
    ax.set_ylabel("Signed row residual / 1e-5", fontproperties=latin, fontsize=8.5)
    ax.set_xlabel("Equality row", fontproperties=latin, fontsize=8.5)
    ax.grid(axis="y", color="#E3E8EB", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_position(("outward", 3))
    ax.spines["left"].set_position(("outward", 3))
    ax.tick_params(axis="both", direction="in", length=4.5, width=0.8, pad=4)
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(latin)
        tick.set_fontsize(8)
    ax.legend(loc="upper right", frameon=False, prop=latin, fontsize=7.6)

    panel = fig.add_axes([0.57, 0.325, 0.37, 0.50])
    panel.set_axis_off()
    for bottom, face, edge in ((0.53, "#FFF1EE", red),
                               (0.06, "#E8F4F1", teal)):
        box = FancyBboxPatch((0.03, bottom), 0.94, 0.37,
                             boxstyle="round,pad=0.006,rounding_size=0.035",
                             transform=panel.transAxes, facecolor=face,
                             edgecolor=edge, linewidth=1.0)
        panel.add_patch(box)
    panel.text(0.07, 0.82, "初始乘积先舍入", fontproperties=chinese,
               fontsize=9.4, color=red, transform=panel.transAxes)
    panel.text(0.07, 0.68, "Residuals: 0, 0  ->  FALSE PASS", fontproperties=latin,
               fontsize=8.3, color=red, transform=panel.transAxes)
    panel.text(0.07, 0.59, "for this infeasible candidate", fontproperties=latin,
               fontsize=7.5, color=red, transform=panel.transAxes)
    panel.text(0.07, 0.35, "修正：原存储数精确乘减", fontproperties=chinese,
               fontsize=9.4, color=teal, transform=panel.transAxes)
    panel.text(0.07, 0.21, "max |residual| = 3.37077e-5", fontproperties=latin,
               fontsize=8.3, color=teal, transform=panel.transAxes)
    panel.text(0.07, 0.12, "Tolerance = 1e-7  ->  REJECT", fontproperties=latin,
               fontsize=8.3, color=teal, transform=panel.transAxes)

    fig.text(0.5, 0.95, "原始行证书：乘积舍入造成虚假可行", ha="center",
             fontproperties=chinese, fontsize=11)
    fig.text(0.5, 0.904, "Original-row certificate: product rounding hides infeasibility",
             ha="center", fontproperties=latin, fontsize=9.6)
    fig.text(0.31, 0.825, "(a) Same candidate, two arithmetic paths", ha="center",
             fontproperties=latin, fontsize=8.2)
    fig.text(0.755, 0.825, "(b) Certificate decision", ha="center",
             fontproperties=latin, fontsize=8.2)
    fig.text(0.5, 0.238, "r_i = x_i - d_i s; s = 1.23456789012345. 两行各自的存储值及十六进制表示见源 CSV。",
             ha="center", fontproperties=chinese, fontsize=8.0)
    fig.text(0.5, 0.174, "浮点乘积先舍入时两行残差均为 0；精确计算原存储浮点数后，两行都超过 1e-7 门限。",
             ha="center", fontproperties=chinese, fontsize=8.0)
    fig.text(0.5, 0.107, "One analytic candidate and one certificate gate; no general solver-speed or LP-optimality claim.",
             ha="center", fontproperties=latin, fontsize=7.5)

    image_dir = FIGURE / "images"
    paths = {ext: image_dir / f"{STEM}.{ext}" for ext in ("png", "pdf", "svg")}
    fig.savefig(paths["png"], dpi=600)
    fig.savefig(paths["pdf"])
    fig.savefig(paths["svg"])
    plt.close(fig)
    return {ext: sha(path) for ext, path in paths.items()}


def main() -> None:
    rows = build_data()
    images = draw(rows)
    data = FIGURE / "source_data" / "analytic_rows.csv"
    evidence = {
        "actual_backend": "Python / matplotlib Agg fallback",
        "fallback_reason": "Same-day Origin native export carried a demo watermark over plotted data",
        "origin_trial": "Demo-watermarked trial is retained in the private research archive",
        "source_test_relative": "tests/test_lp_certificate_product_cancellation.py",
        "source_test_sha256_at_render": sha(TEST),
        "source_data_sha256": sha(data),
        "script_sha256": sha(Path(__file__)),
        "figure_sha256": images,
        "analytic_tolerance": "1e-7 row-unit absolute threshold for this didactic gate",
        "rounded_product_first_residuals": [row["rounded_product_first_residual"] for row in rows],
        "exact_residuals": [row["exact_stored_float_residual_decimal"] for row in rows],
        "declared_axes": {
            "panel_a_x_display": [0.5, 2.5],
            "panel_a_x_end_ticks": [0.5, 2.5],
            "panel_a_y_display": [-4.0, 4.0],
            "panel_a_y_end_ticks": [-4.0, 4.0],
        },
        "scope": "Analytic certificate arithmetic counterexample; not a solver-speed benchmark",
        "visual_review": "PENDING independent rendered-file inspection",
        "origin_processes_after": "PENDING process check",
    }
    path = FIGURE / "qa" / "figure_qa.json"
    path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"images": images, "source_data_sha256": evidence["source_data_sha256"]},
                     indent=2))


if __name__ == "__main__":
    main()
