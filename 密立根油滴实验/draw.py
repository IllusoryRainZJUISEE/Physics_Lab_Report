"""复核四种数据处理结果，并以12组原始测量生成 figures/图5.png。

运行：python draw.py
依赖：numpy、matplotlib。
电量的单位均为 10^-19 C。按报告约定，各 e_i、e_n 先修约为3位
有效数字再平均；拟合与 R² 使用未修约的原始电量和斜率。
"""
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

BASE = Path(__file__).resolve().parent
REFERENCE = Decimal("1.602")
RAW_Q_TEXT = ["4.71", "6.46", "6.49", "6.59", "8.08", "9.17", "10.6", "13.0", "14.6", "18.6", "31.1", "34.2"]
RAW_Q_DECIMAL = tuple(Decimal(value) for value in RAW_Q_TEXT)
RAW_Q = np.array(RAW_Q_TEXT, dtype=float)
RAW_N = np.array([3, 4, 4, 4, 5, 6, 7, 8, 9, 12, 19, 21], dtype=int)


def round_three(value):
    """本实验 e 系数在 1 至 10 之间，3位有效数字即两位小数。"""
    return Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def main():
    # 逐差法：保留原始12组，先按报告约定修约各 q/n，再等权平均。
    raw_e = [round_three(q / int(n)) for q, n in zip(RAW_Q_DECIMAL, RAW_N)]
    e_successive = round_three(sum(raw_e) / len(raw_e))
    u_successive = (sum((value - e_successive) ** 2 for value in raw_e) / (len(raw_e) * (len(raw_e) - 1))).sqrt()

    # 平均值法：排序、作差判定 n 后，同 n 电量取平均，再对各类 e_n 等权平均。
    # 分组直接来自原始测量，避免另一份手写数组与原始数据失去同步。
    group_n = np.unique(RAW_N)
    counts = np.array([np.count_nonzero(RAW_N == n) for n in group_n])
    group_q_exact = [sum(q for q, raw_n in zip(RAW_Q_DECIMAL, RAW_N) if raw_n == n) / int(count)
                     for n, count in zip(group_n, counts)]
    group_e = [round_three(q / int(n)) for q, n in zip(group_q_exact, group_n)]
    e_mean = round_three(sum(group_e) / len(group_e))

    # 多重差值法：保留有符号差分，绝对值仅用于二阶项的列示及取平均。
    # 依据电荷数的一、二阶差为 ±1 选项，不依据与参考 e 的接近程度筛选。
    d1 = [b - a for a, b in zip(RAW_Q_DECIMAL, RAW_Q_DECIMAL[1:])]
    d2 = [b - a for a, b in zip(d1, d1[1:])]
    dn1 = np.diff(RAW_N)
    dn2 = np.diff(dn1)
    selected_d1 = [q for q, dn in zip(d1, dn1) if dn == 1]
    selected_d2 = [abs(q) for q, dn in zip(d2, dn2) if abs(dn) == 1]
    selected = selected_d1 + selected_d2
    e_multi = round_three(sum(selected) / len(selected))

    # 全部12组原始测量的等权、过原点最小二乘拟合。
    slope = float(np.dot(RAW_N, RAW_Q) / np.dot(RAW_N, RAW_N))
    prediction = slope * RAW_N
    r_squared = 1 - np.sum((RAW_Q - prediction) ** 2) / np.sum((RAW_Q - np.mean(RAW_Q)) ** 2)
    e_fit = round_three(slope)

    # 同 n 均值若以原始次数加权，应与全部原始测量给出相同斜率。
    group_q = np.array([float(q) for q in group_q_exact])
    grouped_weighted_slope = float(np.dot(counts * group_n, group_q) / np.dot(counts * group_n, group_n))
    if not np.isclose(slope, grouped_weighted_slope, rtol=1e-12, atol=0):
        raise ValueError("原始数据拟合与按测量次数加权的分组拟合不一致")

    print("raw e:", ", ".join(map(str, raw_e)))
    print(f"successive difference A uncertainty: {u_successive:.8f} x 10^-19 C")
    print("group counts:", counts.tolist())
    print("group e:", ", ".join(map(str, group_e)))
    print("first differences:", ", ".join(map(str, d1)))
    print("absolute second differences:", ", ".join(str(abs(q)) for q in d2))
    print("selected multi-difference terms:", ", ".join(map(str, selected)))
    print(f"fit slope before rounding: {slope:.12f}; centered R^2: {r_squared:.6f}")
    print(f"count-weighted grouped fit slope: {grouped_weighted_slope:.12f}")
    for label, value in [("successive difference", e_successive), ("group average", e_mean), ("multiple differences", e_multi), ("origin fit", e_fit)]:
        error = abs(value - REFERENCE) / REFERENCE * 100
        print(f"{label}: e = {value} x 10^-19 C; relative error = {error:.2f}%")

    available = {font.name for font in font_manager.fontManager.ttflist}
    for preferred in ["Microsoft YaHei", "SimHei", "SimSun", "Noto Sans CJK SC"]:
        if preferred in available:
            plt.rcParams["font.sans-serif"] = [preferred, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["font.size"] = 12

    fig, ax = plt.subplots(figsize=(7.2, 4.5), layout="constrained")
    x = np.linspace(0, 23, 240)
    ax.plot(x, slope * x, color="#255f85", linewidth=1.8, label="过原点最小二乘拟合")
    ax.scatter(RAW_N, RAW_Q, s=43, color="#bd5d34", edgecolors="white", linewidths=0.65,
               zorder=3, label="12组原始测量数据")
    ax.set(xlabel="电荷数 $n$", ylabel=r"电荷量 $q/(10^{-19}\,\mathrm{C})$", xlim=(0, 23), ylim=(0, 38))
    ax.set_xticks(np.arange(0, 24, 3))
    ax.grid(True, color="#d9dfe3", linewidth=0.6, alpha=0.85)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.045, 0.95, "$q=en$\n" + rf"$e={e_fit}\times10^{{-19}}\,\mathrm{{C}}$" + "\n" + rf"$R^2={r_squared:.4f}$",
            transform=ax.transAxes, va="top", bbox=dict(facecolor="white", edgecolor="#d9dfe3", boxstyle="round,pad=0.5"))
    ax.legend(loc="lower right", frameon=True, edgecolor="#d9dfe3", fontsize=10)

    # 原图尺度下 n=4 的三点几乎重合，用局部放大显示真实散布，不人为移动坐标。
    inset = ax.inset_axes([0.64, 0.25, 0.29, 0.36])
    inset.plot([3.72, 4.28], slope * np.array([3.72, 4.28]), color="#255f85", linewidth=1.1)
    mask = RAW_N == 4
    inset.scatter(RAW_N[mask], RAW_Q[mask], s=32, color="#bd5d34", edgecolors="white", linewidths=0.6, zorder=3)
    inset.set(xlim=(3.72, 4.28), ylim=(6.42, 6.64), xticks=[4], yticks=[6.45, 6.50, 6.55, 6.60])
    inset.set_title("$n=4$：三次测量", fontsize=10, pad=5)
    inset.tick_params(labelsize=9)
    inset.grid(True, color="#d9dfe3", linewidth=0.5, alpha=0.8)
    inset.set_axisbelow(True)
    for spine in inset.spines.values():
        spine.set_color("#a8b4bd")

    output = BASE / "figures" / "图5.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=320, facecolor="white")
    plt.close(fig)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
