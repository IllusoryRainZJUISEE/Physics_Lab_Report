"""从当前实验报告的 LaTeX 表格读取数据，绘制三棱镜色散曲线。

运行：python draw.py
可选：python draw.py --report 路径/报告.tex
依赖：numpy、matplotlib。输出：脚本目录下的 figures/图5.png。
"""

from pathlib import Path
import argparse
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter, MultipleLocator
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_REPORT = ROOT / '分光计的调整和使用+棱镜偏向角特性报告.tex'
OUTPUT = ROOT / 'figures' / '图5.png'


def read_report_data(report: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """仅读取折射率表，使用报告已保留三位小数的 n，不另行重算。"""
    source = report.read_text(encoding='utf-8-sig')
    source = re.sub(r'(?<!\\)%[^\n]*', '', source)
    tables = re.findall(r'\\begin\{table\}.*?\\end\{table\}', source, re.S)
    matches = [t for t in tables if '三棱镜对汞灯单色光的折射率' in t]
    if len(matches) != 1:
        raise ValueError('未能唯一定位“三棱镜对汞灯单色光的折射率”表。')

    rows = []
    for row in matches[0].split(r'\\'):
        cells = row.split('&')
        if len(cells) != 4:
            continue
        wavelength = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*\\,\s*\\mathrm\{nm\}', cells[0])
        index = re.fullmatch(r'\s*\$?\s*([0-9]+(?:\.[0-9]+)?)\s*\$?\s*', cells[3])
        if wavelength and index:
            color = re.search(r'（([^）]+)）', cells[0])
            rows.append((float(wavelength[1]), float(index[1]), color[1] if color else '谱线'))

    if len(rows) < 3:
        raise ValueError('三项柯西拟合至少需要三个不同波长的数据点。')
    rows.sort()
    wavelengths = np.array([row[0] for row in rows])
    indices = np.array([row[1] for row in rows])
    if len(np.unique(wavelengths)) != len(rows) or np.any(wavelengths <= 0):
        raise ValueError('波长必须为互不相同的正数。')
    return wavelengths, indices, [row[2] for row in rows]


def fit_cauchy(wavelengths: np.ndarray, indices: np.ndarray):
    """等权最小二乘；缩放自变量避免直接使用 nm^-4 导致病态矩阵。"""
    scale_nm = 500.0
    q = (scale_nm / wavelengths) ** 2
    design = np.column_stack((np.ones_like(q), q, q**2))
    scaled_coefficients, _, rank, _ = np.linalg.lstsq(design, indices, rcond=None)
    if rank != 3:
        raise ValueError('拟合矩阵秩不足，无法确定三个柯西系数。')
    coefficients = scaled_coefficients * np.array([1.0, scale_nm**2, scale_nm**4])
    fitted = design @ scaled_coefficients
    residual_sum = np.sum((indices - fitted) ** 2)
    total_sum = np.sum((indices - indices.mean()) ** 2)
    r_squared = 1.0 - residual_sum / total_sum if total_sum > 0 else np.nan
    return coefficients, r_squared


def draw(wavelengths: np.ndarray, indices: np.ndarray, coefficients: np.ndarray, r_squared: float, colors: list[str]):
    available = {font.name for font in font_manager.fontManager.ttflist}
    candidates = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'SimSun']
    chinese_fonts = [name for name in candidates if name in available]
    if not chinese_fonts:
        raise RuntimeError('未找到中文字体，请安装微软雅黑、黑体或 Noto Sans CJK SC。')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': chinese_fonts + ['DejaVu Sans'],
        'axes.unicode_minus': False,
        'font.size': 11,
        'mathtext.fontset': 'stix',
        'axes.linewidth': 1.0,
        'savefig.facecolor': 'white',
    })

    a, b, c = coefficients
    curve_x = np.linspace(wavelengths.min(), wavelengths.max(), 600)
    curve_y = a + b / curve_x**2 + c / curve_x**4
    fig, ax = plt.subplots(figsize=(7.6, 5.4), layout='constrained')
    ax.plot(curve_x, curve_y, color='#bb493b', linewidth=1.9, label='柯西公式拟合', zorder=2)
    ax.scatter(wavelengths, indices, marker='s', s=48, color='#315c78',
               edgecolors='white', linewidths=0.65, label='实验数据', zorder=3)
    for wavelength, index, color in zip(wavelengths, indices, colors):
        label = color + '光' if color in {'紫', '蓝', '绿', '黄'} else color
        ax.annotate(label, xy=(wavelength, index), xytext=(0, 11),
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=10.5, color='#253b4b', zorder=4)

    ax.set_title('三棱镜色散曲线', fontsize=15, pad=13)
    ax.set_xlabel(r'波长 $\lambda\,/\,\mathrm{nm}$', labelpad=8)
    ax.set_ylabel(r'折射率 $n$', labelpad=9)
    x_pad = 0.09 * np.ptp(wavelengths)
    ax.set_xlim(wavelengths.min() - x_pad, wavelengths.max() + x_pad)
    y_span = np.ptp(indices)
    ax.set_ylim(indices.min() - 0.20 * y_span, indices.max() + 0.26 * y_span)
    ax.xaxis.set_major_locator(MultipleLocator(25))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_major_locator(MultipleLocator(0.005))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.tick_params(which='major', length=5, pad=7)
    ax.tick_params(which='minor', length=2.5)
    ax.grid(which='major', color='#d9dfe3', linewidth=0.65, alpha=0.6)
    ax.set_axisbelow(True)

    parameter_text = '\n'.join([
        r'$n(\lambda)=A+B/\lambda^2+C/\lambda^4$',
        rf'$A={a:.6f}$',
        rf'$B={b / 1e4:.6f}\times10^4\;\mathrm{{nm}}^2$',
        rf'$C={c / 1e8:.6f}\times10^8\;\mathrm{{nm}}^4$',
        rf'$R^2={r_squared:.6f}$',
    ])
    ax.text(0.97, 0.96, parameter_text, transform=ax.transAxes,
            ha='right', va='top', fontsize=10.5, linespacing=1.6,
            bbox={'boxstyle': 'round,pad=0.65', 'facecolor': 'white',
                  'edgecolor': '#aeb9c1', 'linewidth': 0.8, 'alpha': 0.97})
    ax.legend(loc='lower left', frameon=False, fontsize=10.5)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=400, metadata={'Title': '三棱镜色散曲线', 'Description': '实验数据与三项柯西公式的等权最小二乘拟合'})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=DEFAULT_REPORT, help='含折射率表的 LaTeX 报告路径')
    args = parser.parse_args()
    wavelengths, indices, colors = read_report_data(args.report)
    coefficients, r_squared = fit_cauchy(wavelengths, indices)
    draw(wavelengths, indices, coefficients, r_squared, colors)
    print('Source:', args.report)
    for wavelength, index, color in zip(wavelengths, indices, colors):
        print(f'  {wavelength:.1f} nm: n = {index:.3f}')
    print(f'A = {coefficients[0]:.9f}')
    print(f'B = {coefficients[1]:.9e} nm^2')
    print(f'C = {coefficients[2]:.9e} nm^4')
    print(f'R^2 = {r_squared:.9f}')
    print('Saved:', OUTPUT)


if __name__ == '__main__':
    main()

