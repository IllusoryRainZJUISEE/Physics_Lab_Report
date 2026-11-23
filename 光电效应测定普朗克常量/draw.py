"""绘制光电效应实验图6至图9。运行：python draw.py。

原始测量值保留在数组中；中间计算使用完整精度，输出统一保留三位小数。
需要 numpy 和 matplotlib。图片保存到本文件同级的 figures 文件夹。
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
# SI 定义常量，计算时不提前舍入。
C = 299792458.0
E = 1.602176634e-19
H_REFERENCE = 6.62607015e-34

WAVELENGTH_NM = np.array([365, 405, 436, 546, 577])
STOPPING_VOLTAGE_V = np.array([-1.841, -1.449, -1.248, -0.731, -0.610])
VOLTAGE_4MM_V = np.array([-1.57, 0.65, 2.81, 4.43, 6.10, 8.73,
                         11.52, 14.56, 18.50, 21.65, 24.85, 28.60])
CURRENT_4MM_1E11_A = np.array([0, 58, 198, 305, 387, 503,
                              578, 646, 713, 756, 791, 831])
VOLTAGE_8MM_V = np.array([-2.67, 0.02, 2.85, 4.03, 7.10, 9.35,
                         12.62, 14.91, 17.57, 20.20, 22.22, 27.25])
CURRENT_8MM_1E10_A = np.array([0, 23, 127, 177, 267, 318,
                              360, 395, 421, 442, 458, 489])
DISTANCE_CM = np.array([40, 38, 36, 34, 32])
DISTANCE_CURRENT_1E11_A = np.array([824, 951, 1118, 1290, 1554])
APERTURE_MM = np.array([2, 4, 8])
APERTURE_CURRENT_1E10_A = np.array([21.7, 84.2, 336.0])


def linear_fit(x, y):
    """带截距的普通最小二乘拟合，不强制过原点。"""
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    r_squared = 1.0 - np.sum(residual**2) / np.sum((y - y.mean())**2)
    return float(slope), float(intercept), float(r_squared)


def calculate():
    frequency = C / (WAVELENGTH_NM * 1e-9) / 1e14
    # 将记录中的反向电压转为正的遏制电压绝对值。
    planck_fit = linear_fit(frequency, np.abs(STOPPING_VOLTAGE_V))
    h = E * planck_fit[0] / 1e14
    relative_error = abs(h - H_REFERENCE) / H_REFERENCE * 100.0
    inverse_distance_squared = 1e4 / DISTANCE_CM.astype(float)**2
    distance_current = DISTANCE_CURRENT_1E11_A / 100.0  # 换算为 10^-9 A
    aperture_squared = APERTURE_MM.astype(float)**2
    return {
        "frequency": frequency,
        "planck_fit": planck_fit,
        "h": h,
        "relative_error": relative_error,
        "inverse_distance_squared": inverse_distance_squared,
        "distance_current": distance_current,
        "distance_fit": linear_fit(inverse_distance_squared, distance_current),
        "aperture_squared": aperture_squared,
        "aperture_fit": linear_fit(aperture_squared, APERTURE_CURRENT_1E10_A),
    }


def new_axes(xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(7.3, 4.1), layout="constrained")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, color="#DCE1E7", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=10)
    return fig, ax


def fit_plot(x, y, fit, xlabel, ylabel, filename):
    fig, ax = new_axes(xlabel, ylabel)
    k, b, r2 = fit
    span = np.ptp(x)
    x_line = np.linspace(x.min() - 0.045 * span, x.max() + 0.045 * span, 300)
    ax.plot(x_line, k * x_line + b, color="#C46A30", lw=1.8,
            label="最小二乘拟合")
    ax.scatter(x, y, s=43, color="#235A85", edgecolors="white", linewidths=0.7,
               zorder=3, label="实验数据")
    sign = "+" if b >= 0 else "-"
    ax.text(0.97, 0.07,
            rf"$y={k:.3f}x {sign} {abs(b):.3f}$" + "\n" + rf"$R^2={r2:.3f}$",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=11,
            bbox={"facecolor": "white", "edgecolor": "#DCE1E7", "alpha": 0.95,
                  "boxstyle": "round,pad=0.45"})
    ax.legend(loc="upper left", frameon=False)
    fig.savefig(FIGURES / filename, dpi=320)
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({
        "font.family": ["SimSun", "Microsoft YaHei", "DejaVu Sans"],
        "font.size": 11,
        "mathtext.fontset": "stix",
        "axes.unicode_minus": False,
        "savefig.facecolor": "white",
    })
    result = calculate()
    fit_plot(result["frequency"], np.abs(STOPPING_VOLTAGE_V), result["planck_fit"],
             r"入射光频率 $\nu$ / ($10^{14}\,\mathrm{Hz}$)",
             r"遏制电压绝对值 $|U_a|$ / V", "图6.png")

    fig, ax = new_axes(r"光电管电压 $U_{\mathrm{AK}}$ / V",
                      r"光电流 $i$ / ($10^{-9}\,\mathrm{A}$)")
    # 两组原始电流的量级不同，先统一单位再比较；仅连接测量点。
    ax.plot(VOLTAGE_4MM_V, CURRENT_4MM_1E11_A / 100.0, "o-", color="#235A85",
            markersize=4.5, linewidth=1.6, label="光阑孔径 4 mm，光源距离 40 cm")
    ax.plot(VOLTAGE_8MM_V, CURRENT_8MM_1E10_A / 10.0, "s-", color="#C46A30",
            markersize=4.5, linewidth=1.6, label="光阑孔径 8 mm，光源距离 34 cm")
    ax.set_ylim(bottom=-1.0)
    ax.text(0.97, 0.48, r"$\lambda=436\,\mathrm{nm}$", transform=ax.transAxes,
            ha="right", fontsize=11)
    ax.legend(loc="upper left", frameon=False)
    fig.savefig(FIGURES / "图7.png", dpi=320)
    plt.close(fig)

    fit_plot(result["inverse_distance_squared"], result["distance_current"],
             result["distance_fit"],
             r"距离平方倒数 $L^{-2}$ / ($10^{-4}\,\mathrm{cm}^{-2}$)",
             r"饱和光电流 $i_s$ / ($10^{-9}\,\mathrm{A}$)", "图8.png")
    fit_plot(result["aperture_squared"], APERTURE_CURRENT_1E10_A,
             result["aperture_fit"], r"光阑孔径平方 $\Phi^2$ / $\mathrm{mm}^2$",
             r"饱和光电流 $i_s$ / ($10^{-10}\,\mathrm{A}$)", "图9.png")

    for name in ("planck_fit", "distance_fit", "aperture_fit"):
        k, b, r2 = result[name]
        print(f"{name}: slope={k:.3f}, intercept={b:.3f}, R2={r2:.3f}")
    print(f"h={result['h']/1e-34:.3f}e-34 J s")
    print(f"relative_error={result['relative_error']:.3f}%")
    print("frequency / 1e14 Hz:", ", ".join(f"{v:.3f}" for v in result["frequency"]))
    print("L^-2 / 1e-4 cm^-2:", ", ".join(f"{v:.3f}" for v in result["inverse_distance_squared"]))


if __name__ == "__main__":
    main()