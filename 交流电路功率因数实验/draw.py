from pathlib import Path
import matplotlib.pyplot as plt

# ==================== 字体设置 ====================
plt.rcParams.update({
    "font.sans-serif": [
        "Microsoft YaHei", "SimSun", "SimHei", "DejaVu Sans"
    ],
    "axes.unicode_minus": False,
    "font.size": 12,
    "mathtext.fontset": "stix",
})

# ==================== 绘图数据 ====================
# 电容量，单位：μF
C = [0, 1, 2, 3, 4, 5, 6, 7]

# 不同频率下的功率因数 cosφ
cos_phi_50 = [0.353, 0.450, 0.580, 0.761, 0.914, 0.884, 0.708, 0.556]
cos_phi_45 = [0.361, 0.417, 0.498, 0.595, 0.712, 0.820, 0.873, 0.835]
cos_phi_55 = [0.361, 0.490, 0.689, 0.927, 0.863, 0.633, 0.462, 0.364]

# ==================== 绘制三条折线 ====================
fig, ax = plt.subplots(figsize=(8, 5.5), layout="constrained")

ax.plot(
    C, cos_phi_50,
    color="#555555",
    marker="s",
    linewidth=1.6,
    markersize=7,
    label="50 Hz",
)

ax.plot(
    C, cos_phi_45,
    color="#F04444",
    marker="o",
    linewidth=1.6,
    markersize=7,
    label="45 Hz",
)

ax.plot(
    C, cos_phi_55,
    color="#2375E8",
    marker="^",
    linewidth=1.6,
    markersize=7,
    label="55 Hz",
)

# ==================== 坐标轴设置 ====================
ax.set_xlabel(r"电容量 $C$ / $\mu\mathrm{F}$", fontsize=14)
ax.set_ylabel(r"功率因数 $\cos\varphi$", fontsize=14)

# 两端留少量空间，避免数据点被裁切
ax.set_xlim(-0.15, 7.15)
ax.set_ylim(0.3, 1.0)

ax.set_xticks(C)
ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

ax.tick_params(
    axis="both",
    direction="in",
    length=5,
    width=1,
    labelsize=12,
)

for spine in ax.spines.values():
    spine.set_linewidth(1.2)

# 图例
ax.legend(
    loc="upper right",
    frameon=True,
    fancybox=False,
    facecolor="white",
    edgecolor="black",
    framealpha=1,
)

# ==================== 保存 PNG ====================
output_dir = Path(__file__).resolve().parent / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / "图10.png"

fig.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

print(f"图片已保存：{output_file}")

plt.show()