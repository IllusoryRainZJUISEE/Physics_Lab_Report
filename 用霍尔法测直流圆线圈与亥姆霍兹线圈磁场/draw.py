import os
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimSun", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT_DIR, exist_ok=True)
x_axial = list(range(-10, 11))
circular_axial_exp = [370.0, 429.5, 497.0, 575.5, 654.5, 738.5, 825.0, 902.5, 959.5, 996.5, 1005.0, 983.0, 936.5, 870.0, 788.5, 705.5, 620.5, 540.5, 470.0, 403.5, 348.0]
circular_axial_theory = [355.4, 412.8, 478.7, 552.7, 633.9, 719.3, 804.7, 883.4, 947.9, 990.4, 1005.3, 990.4, 947.9, 883.4, 804.7, 719.3, 633.9, 552.7, 478.7, 412.8, 355.4]
y_radial = list(range(-5, 6))
circular_radial_exp = [1245.0, 1140.5, 1079.5, 1033.0, 1012.5, 1009.0, 1015.0, 1037.0, 1078.0, 1139.5, 1246.0]
helmholtz_R_exp = [882.5, 995.0, 1104.0, 1208.5, 1294.0, 1362.0, 1411.0, 1434.0, 1449.5, 1454.0, 1456.0, 1454.5, 1449.5, 1439.5, 1413.5, 1368.5, 1297.0, 1213.5, 1108.0, 1001.5, 890.0]
helmholtz_R_theory = [890.9, 1002.1, 1111.3, 1211.6, 1296.4, 1360.7, 1403.3, 1426.5, 1436.1, 1438.5, 1438.7, 1438.5, 1436.1, 1426.5, 1403.3, 1360.7, 1296.4, 1211.6, 1111.3, 1002.1, 890.9]
helmholtz_half_R_exp = [756, 872, 1005, 1145, 1284, 1424, 1553, 1670, 1755, 1806, 1824, 1804, 1753, 1659, 1556, 1427, 1284, 1140, 1001, 876, 764]
helmholtz_half_R_theory = [759.8, 876.6, 1006.0, 1145.5, 1290.0, 1432.6, 1564.9, 1677.8, 1763.9, 1817.6, 1835.8, 1817.6, 1763.9, 1677.8, 1564.9, 1432.6, 1290.0, 1145.5, 1006.0, 876.6, 759.8]
helmholtz_2R_exp = [1084, 1081, 1052, 1004, 940, 880, 822, 770, 735, 711, 705, 712, 739, 776, 828, 895, 953, 1015, 1064, 1094, 1096]
helmholtz_2R_theory = [1095.2, 1092.0, 1063.0, 1014.4, 954.3, 890.9, 831.3, 780.6, 742.4, 718.8, 710.9, 718.8, 742.4, 780.6, 831.3, 890.9, 954.3, 1014.4, 1063.0, 1092.0, 1095.2]

def check_lengths():
    assert len(x_axial) == 21
    assert len(circular_axial_exp) == len(circular_axial_theory) == 21
    assert len(y_radial) == len(circular_radial_exp) == 11
    for values in [helmholtz_R_exp, helmholtz_R_theory, helmholtz_half_R_exp, helmholtz_half_R_theory, helmholtz_2R_exp, helmholtz_2R_theory]:
        assert len(values) == 21

def style_axes(ax):
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.55)
    ax.set_axisbelow(True)
    ax.tick_params(direction="in", top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

def save(fig, filename):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close(fig)

check_lengths()
fig, ax = plt.subplots(figsize=(7.0, 4.6))
ax.plot(x_axial, circular_axial_exp, "o-", color="#1f77b4", linewidth=1.5, markersize=4.5, label="实验值")
ax.plot(x_axial, circular_axial_theory, "s--", color="#d62728", linewidth=1.4, markersize=4.0, label="理论值")
ax.set_xlabel(r"轴向距离 $x/\mathrm{cm}$")
ax.set_ylabel(r"磁感应强度 $B/\mathrm{\mu T}$")
ax.set_title("载流圆线圈轴向磁场分布")
ax.set_xticks(range(-10, 11, 2))
ax.legend(frameon=True)
style_axes(ax)
save(fig, "图6.png")

fig, ax = plt.subplots(figsize=(7.0, 4.6))
ax.plot(y_radial, circular_radial_exp, "o-", color="#2ca02c", linewidth=1.5, markersize=4.8, label="实验平均值")
ax.set_xlabel(r"径向距离 $y/\mathrm{cm}$")
ax.set_ylabel(r"磁感应强度 $B/\mathrm{\mu T}$")
ax.set_title("载流圆线圈径向磁场分布")
ax.set_xticks(range(-5, 6, 1))
ax.legend(frameon=True)
style_axes(ax)
save(fig, "图7.png")

fig, ax = plt.subplots(figsize=(8.2, 5.2))
colors = {"R/2": "#d62728", "R": "#1f77b4", "2R": "#2ca02c"}
series = [("R/2", helmholtz_half_R_exp, helmholtz_half_R_theory), ("R", helmholtz_R_exp, helmholtz_R_theory), ("2R", helmholtz_2R_exp, helmholtz_2R_theory)]
for ratio, exp, theory in series:
    color = colors[ratio]
    ax.plot(x_axial, exp, "o-", color=color, linewidth=1.45, markersize=4.0, label=fr"实验值，$d={ratio}$")
    ax.plot(x_axial, theory, "s--", color=color, linewidth=1.25, markersize=3.7, alpha=0.9, label=fr"理论值，$d={ratio}$")
ax.set_xlabel(r"轴向距离 $x/\mathrm{cm}$")
ax.set_ylabel(r"磁感应强度 $B/\mathrm{\mu T}$")
ax.set_title("亥姆霍兹线圈不同间距下的轴向磁场分布")
ax.set_xticks(range(-10, 11, 2))
ax.legend(ncol=2, fontsize=9, frameon=True)
style_axes(ax)
save(fig, "图8.png")

print("绘图完成，文件已保存到：")
for filename in ["图6.png", "图7.png", "图8.png"]:
    print(os.path.join(OUT_DIR, filename))


