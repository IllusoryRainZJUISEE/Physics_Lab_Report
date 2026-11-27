"""读取实验报告中的原始位置读数，计算两组结果并生成 figures/图7.png。

运行：python draw.py
依赖：numpy、matplotlib。
数据源：当前目录《迈克尔逊干涉仪的使用报告.tex》的两张数据表。
每两个相邻位置之间吞吐50条条纹；采用报告中已核对的读数。
第二组末点因实验时条纹模糊而按实验者说明排除，保留原始记录和排除原因。
B类按课程将Δd视作直接测量量的简化约定，取0.00005/sqrt(3) mm；
不从位置读数逐层传播，也不按重复次数缩小该仪器分量。
拟合保留自由截距，两组独立处理；按要求不计算回归波长的不确定度。
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
import numpy as np

ROOT = Path(__file__).resolve().parent
REPORT = ROOT / "迈克尔逊干涉仪的使用报告.tex"
FRINGES_PER_INTERVAL = 50
ERROR_LIMIT_MM = 0.00005  # 实验者提供的仪器误差限图片
SECOND_EXCLUDED_INDEX = 3  # 从0计数，即第二组 d4
SECOND_EXCLUSION_REASON = "测量末点时条纹过于模糊，实验者确认该点不可靠"
MM_TO_NM = 1_000_000.0
WAVELENGTH_FACTOR = 2 * MM_TO_NM / FRINGES_PER_INTERVAL


def read_positions(text: str, label: str, expected_count: int) -> np.ndarray:
    """从带指定标签的数据表读取位置，按下标排序，保留原始数值。"""
    tables = re.findall(r"\\begin\{table\}.*?\\end\{table\}", text, re.S)
    table = next((s for s in tables if rf"\label{{{label}}}" in s), None)
    if table is None:
        raise ValueError(f"报告中找不到数据表 {label}")
    records = re.findall(r"\$d_\{?(\d+)\}?\s*=\s*(\d+\.\d+)\$", table)
    indexed = {int(index): float(value) for index, value in records}
    if len(records) != expected_count or set(indexed) != set(range(1, expected_count + 1)):
        raise ValueError(f"{label} 应含连续的 {expected_count} 个原始读数，实际为 {records}")
    return np.array([indexed[i] for i in range(1, expected_count + 1)], dtype=float)


def analyze(positions: np.ndarray, method: str) -> dict:
    if method == "difference6":
        if len(positions) != 9 or not np.all(np.diff(positions) > 0):
            raise ValueError("第一组须有9个连续递增的位置读数")
        intervals = (positions[6:9] - positions[:3]) / 6
    elif method == "adjacent_mean":
        if not (np.all(np.diff(positions) < 0) or np.all(np.diff(positions) > 0)):
            raise ValueError("第二组应为同一方向的连续读数；请核查数据与条纹计数")
        intervals = np.abs(np.diff(positions))
    else:
        raise ValueError(f"未知处理方法：{method}")

    count = len(intervals)
    mean_mm = float(intervals.mean())
    sample_sd_mm = float(intervals.std(ddof=1))
    ua_mm = sample_sd_mm / np.sqrt(count)
    # 课程简化约定：将仪器误差限直接赋予每50条条纹的位移Δd。
    ub_mm = ERROR_LIMIT_MM / np.sqrt(3)
    uc_mm = float(np.hypot(ua_mm, ub_mm))

    # 初始读数的累计条纹数设为0；实际记录间隔为50，不能以序号代替条纹数。
    fringes = FRINGES_PER_INTERVAL * np.arange(len(positions), dtype=float)
    centered_n = fringes - fringes.mean()
    slope = float(np.dot(centered_n, positions - positions.mean()) / np.dot(centered_n, centered_n))
    intercept = float(positions.mean() - slope * fringes.mean())
    predicted = intercept + slope * fringes
    residuals = positions - predicted
    r_squared = 1 - float(np.dot(residuals, residuals) / np.sum((positions - positions.mean()) ** 2))
    return {
        "method": method,
        "positions_mm": positions.tolist(),
        "fringes": fringes.tolist(),
        "intervals_mm": intervals.tolist(),
        "wavelengths_nm": (intervals * WAVELENGTH_FACTOR).tolist(),
        "mean_interval_mm": mean_mm,
        "sample_sd_mm": sample_sd_mm,
        "uA_mm": float(ua_mm),
        "uB_mm": float(ub_mm),
        "uB_model": "course_direct_displacement",
        "uB_displacement_error_limit_mm": ERROR_LIMIT_MM,
        "uc_mm": uc_mm,
        "mean_wavelength_nm": mean_mm * WAVELENGTH_FACTOR,
        "uA_wavelength_nm": float(ua_mm * WAVELENGTH_FACTOR),
        "uB_wavelength_nm": float(ub_mm * WAVELENGTH_FACTOR),
        "uc_wavelength_nm": uc_mm * WAVELENGTH_FACTOR,
        "relative_deviation_589_3_percent": (mean_mm * WAVELENGTH_FACTOR / 589.3 - 1) * 100,
        "fit_slope_mm_per_fringe": slope,
        "fit_intercept_mm": intercept,
        "fit_r_squared": r_squared,
        "fit_wavelength_nm": 2 * abs(slope) * MM_TO_NM,
        "residuals_um": (residuals * 1000).tolist(),
    }


def configure_fonts() -> str:
    available = {font.name for font in font_manager.fontManager.ttflist}
    chinese_font = next((f for f in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "SimSun") if f in available), None)
    if chinese_font is None:
        raise RuntimeError("未找到中文字体，请安装或指定 Microsoft YaHei / SimHei / Noto Sans CJK SC")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [chinese_font, "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "mathtext.fontset": "dejavusans",
        "savefig.facecolor": "white",
    })
    return chinese_font


def plot_results(results: list[dict]) -> Path:
    configure_fonts()
    fig = plt.figure(figsize=(11.2, 7.0), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(3.2, 1.0), hspace=0.035, wspace=0.07)
    colors = ("#2468A0", "#B55F27")
    titles = ("（a）第一组：9个位置读数", "（b）第二组：3个有效位置读数")
    for col, (result, color, title) in enumerate(zip(results, colors, titles)):
        ax = fig.add_subplot(grid[0, col])
        residual_ax = fig.add_subplot(grid[1, col], sharex=ax)
        fringes = np.asarray(result["fringes"])
        positions = np.asarray(result["positions_mm"])
        dense_n = np.linspace(fringes[0], fringes[-1], 250)
        slope = result["fit_slope_mm_per_fringe"]
        intercept = result["fit_intercept_mm"]
        ax.plot(dense_n, slope * dense_n + intercept, color=color, lw=1.8, label="线性拟合", zorder=2)
        ax.scatter(fringes, positions, color=color, edgecolor="white", lw=0.8, s=45, label="原始数据", zorder=3)
        ax.set_title(title, loc="left", pad=12, fontweight="bold")
        ax.set_ylabel(r"动镜位置 $d$ / mm")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.5f"))
        ax.tick_params(labelbottom=False)
        ax.grid(True, color="#DFE4E8", lw=0.65, alpha=0.75)
        ax.margins(x=0.07, y=0.12)
        ax.legend(loc="lower right" if col == 0 else "upper right", framealpha=0.97)
        sign = "+" if slope >= 0 else "-"
        details = (
            rf"$d = {sign if slope < 0 else ''}{abs(slope) * 1e4:.6f}\times10^{{-4}}N + {intercept:.5f}$" "\n"
            rf"$R^2 = {result['fit_r_squared']:.8f}$" "\n"
            rf"$\lambda_{{\mathrm{{fit}}}} = {result['fit_wavelength_nm']:.1f}\,\mathrm{{nm}}$"
        )
        ax.text(0.045, 0.95 if col == 0 else 0.055, details,
                transform=ax.transAxes, ha="left", va="top" if col == 0 else "bottom",
                fontsize=10.5, linespacing=1.6,
                bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": "#D6DDE3", "alpha": 0.97})
        residuals = np.asarray(result["residuals_um"])
        residual_ax.axhline(0, color="#59656F", lw=1.0)
        residual_ax.vlines(fringes, 0, residuals, color=color, lw=1.0, alpha=0.65)
        residual_ax.scatter(fringes, residuals, s=25, color=color, zorder=3)
        residual_ax.set_ylabel(r"残差 / $\mu$m")
        residual_ax.set_xlabel(r"累计吞吐条纹数 $N$ / 条")
        residual_ax.xaxis.set_major_locator(MultipleLocator(100 if col == 0 else 50))
        residual_ax.grid(True, axis="x", color="#DFE4E8", lw=0.65, alpha=0.75)
        limit = max(0.001, float(np.max(np.abs(residuals))) * 1.45)
        residual_ax.set_ylim(-limit, limit)
        for axis in (ax, residual_ax):
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
    fig.suptitle("动镜位置与累计吞吐条纹数的线性回归", fontsize=15, fontweight="bold")
    fig.supxlabel("各组首个读数对应 N = 0；第二组末点因条纹模糊排除；拟合保留自由截距。", fontsize=10, color="#4A545D")
    target = ROOT / "figures" / "图7.png"
    target.parent.mkdir(exist_ok=True)
    fig.savefig(target, dpi=320)
    plt.close(fig)
    return target


def main() -> None:
    text = REPORT.read_text(encoding="utf-8-sig")
    first = read_positions(text, "tab:wavelength-first", 9)
    second = read_positions(text, "tab:wavelength-second", 4)
    # 排除实验者确认不可靠的末点，保持其余点的条纹计数间隔。
    if SECOND_EXCLUDED_INDEX != len(second) - 1:
        raise ValueError("当前排除规则仅适用于第二组末点")
    second_result = analyze(second[:SECOND_EXCLUDED_INDEX], "adjacent_mean")
    second_result["raw_positions_mm"] = second.tolist()
    second_result["excluded_readings"] = [{
        "reading_index": SECOND_EXCLUDED_INDEX + 1,
        "position_mm": float(second[SECOND_EXCLUDED_INDEX]),
        "fringes": FRINGES_PER_INTERVAL * SECOND_EXCLUDED_INDEX,
        "reason": SECOND_EXCLUSION_REASON,
    }]
    results = [analyze(first, "difference6"), second_result]
    image_path = plot_results(results)
    audit_path = ROOT / "tmp" / "analysis-results.json"
    audit_path.parent.mkdir(exist_ok=True)
    audit_path.write_text(json.dumps({
        "fringes_per_interval": FRINGES_PER_INTERVAL,
        "error_limit_mm": ERROR_LIMIT_MM,
        "B_model": "direct displacement B component, rectangular distribution; instrument error limit divided by sqrt(3)",
        "error_limit_source": "实验者提供并确认的仪器误差限图片：0.00005 mm",
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    for index, result in enumerate(results, start=1):
        print(f"Group {index}: mean wavelength = {result['mean_wavelength_nm']:.8f} nm, "
              f"uA = {result['uA_wavelength_nm']:.8f} nm, "
              f"uB = {result['uB_wavelength_nm']:.8f} nm, "
              f"uc = {result['uc_wavelength_nm']:.8f} nm")
        print(f"  Fit: k = {result['fit_slope_mm_per_fringe']:.12f} mm/fringe, "
              f"b = {result['fit_intercept_mm']:.9f} mm, "
              f"R2 = {result['fit_r_squared']:.9f}, "
              f"wavelength = {result['fit_wavelength_nm']:.8f} nm")
    print(f"Figure: {image_path}")
    print(f"Full-precision results: {audit_path}")


if __name__ == "__main__":
    main()
