"""
RoboGauge Stress Benchmark 结果可视化

用法:
  python plot_results.py
"""
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.ticker as mticker

RESULT_FILE = Path(__file__).parent / "go2_moe_stress_go2_moe_5500_stress" / \
              "20260504-14-48-59_run" / "stress_benchmark_results.yaml"

with open(RESULT_FILE, 'r') as f:
    data = yaml.safe_load(f)

# ============================================================
# 图 1: Radar + Bar —— 7 地形对比
# ============================================================
terrain_names = ['Flat', 'Obstacle', 'Slope Down', 'Slope Up', 'Stairs Down', 'Stairs Up', 'Wave']
terrain_keys = ['flat', 'obstacle', 'slope_bd', 'slope_fd', 'stairs_bd', 'stairs_fd', 'wave']

scores = [data['scores'][k] for k in terrain_keys]
benchmark = data['benchmark_score']

fig, (ax_radar, ax_bar) = plt.subplots(1, 2, figsize=(14, 5),
    gridspec_kw={'width_ratios': [1, 1.2]})

# ---- Radar ----
angles = np.linspace(0, 2 * np.pi, len(terrain_names), endpoint=False).tolist()
scores_closed = scores + [scores[0]]
angles_closed = angles + [angles[0]]

ax_radar = plt.subplot(1, 2, 1, projection='polar')
ax_radar.fill(angles_closed, scores_closed, alpha=0.25, color='#2196F3')
ax_radar.plot(angles_closed, scores_closed, 'o-', color='#2196F3', linewidth=2, markersize=6)
ax_radar.set_xticks(angles)
ax_radar.set_xticklabels(terrain_names, fontsize=10)
ax_radar.set_ylim(0, 0.8)
ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8])
ax_radar.set_yticklabels(['0.2', '0.4', '0.6', '0.8'], fontsize=7, color='gray')
ax_radar.set_title(f'RoboGauge Benchmark: {benchmark:.3f}', fontsize=13, fontweight='bold', pad=20)
# 在每个点上标注数值
for a, s in zip(angles, scores):
    ax_radar.annotate(f'{s:.3f}', xy=(a, s), fontsize=8, ha='center',
                      xytext=(0, 6), textcoords='offset points')

# ---- Bar ----
colors_bar = ['#4CAF50' if s >= 0.5 else '#FF9800' if s >= 0.4 else '#f44336' for s in scores]
bars = ax_bar.barh(terrain_names[::-1], scores[::-1], color=colors_bar[::-1], edgecolor='white')
ax_bar.axvline(x=benchmark, color='gray', linestyle='--', linewidth=1.5, label=f'Overall: {benchmark:.3f}')
ax_bar.set_xlim(0, 0.8)
ax_bar.set_xlabel('Score', fontsize=11)
ax_bar.set_title('Terrain Scores', fontsize=13, fontweight='bold')
ax_bar.legend(fontsize=9)
for bar, s in zip(bars, scores[::-1]):
    ax_bar.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{s:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(Path(__file__).parent / 'benchmark_terrain_scores.png', dpi=150, bbox_inches='tight')

# ============================================================
# 图 2: 8 项指标汇总（水平柱状图，避免标签重叠）
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))

metric_labels = [
    ('lin_vel_err',           'Lin Vel Track'),
    ('ang_vel_err',           'Ang Vel Track'),
    ('dof_power',             'DOF Power'),
    ('dof_limits',            'DOF Limits'),
    ('orientation_stability', 'Orientation'),
    ('torque_smoothness',     'Torque Smooth'),
    ('friction_margin',       'Friction Margin'),
    ('zmp_margin',            'ZMP Margin'),
]

summary = data['summary']
metric_names = [m[1] for m in metric_labels]
metrics = [m[0] for m in metric_labels]

means = [summary[m]['mean'] for m in metrics]
mean50s = [summary[m]['mean@50'] for m in metrics]

y = np.arange(len(metrics))
height = 0.3

ax.barh(y + height/2, means, height, label='Mean (all commands)', color='#90CAF9', edgecolor='white')
bars = ax.barh(y - height/2, mean50s, height, label='Mean@50 (worst 50%)', color='#1565C0', edgecolor='white')

ax.set_yticks(y)
ax.set_yticklabels(metric_names, fontsize=10)
ax.set_xlim(0, 0.80)
ax.set_xlabel('Score', fontsize=11)
ax.set_title('Per-Metric Summary (across all terrains)', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.xaxis.grid(True, alpha=0.3)
ax.invert_yaxis()

# 在柱子上清晰标注数值
for i, (m_val, m50_val) in enumerate(zip(means, mean50s)):
    ax.text(m_val + 0.008, i + height/2, f'{m_val:.3f}', va='center', fontsize=8, color='#555555')
    ax.text(m50_val + 0.008, i - height/2, f'{m50_val:.3f}', va='center', fontsize=8, color='#FFFFFF', fontweight='bold')

fig.subplots_adjust(left=0.18, right=0.95, top=0.93, bottom=0.08)
plt.savefig(Path(__file__).parent / 'benchmark_metrics_summary.png', dpi=150, bbox_inches='tight')

# ============================================================
# 图 3: 摩擦 vs 最高难度级别（论文 Fig.15 风格）
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

fric_terrains = ['wave', 'slope_fd', 'slope_bd', 'stairs_fd', 'stairs_bd', 'obstacle']
fric_titles = ['Wave', 'Slope ↑', 'Slope ↓', 'Stairs ↑', 'Stairs ↓', 'Obstacle']
frictions = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for ax_i, (tkey, ttitle) in enumerate(zip(fric_terrains, fric_titles)):
    ax = axes[ax_i]
    levels = []
    for fric in frictions:
        key = f'{tkey}_None_baseMass0_friction{fric}'
        if key in data:
            level = data[key].get('terrain_level')
            levels.append(level if level is not None else -1)
        else:
            levels.append(-1)

    valid_x = [f for f, l in zip(frictions, levels) if l >= 0]
    valid_y = [l for l in levels if l >= 0]
    invalid_x = [f for f, l in zip(frictions, levels) if l < 0]

    ax.plot(valid_x, valid_y, 'o-', color='#1565C0', linewidth=2, markersize=8, markerfacecolor='white')
    if invalid_x:
        ax.scatter(invalid_x, [-1]*len(invalid_x), color='#f44336', marker='x', s=100, linewidths=2)
    ax.set_title(ttitle, fontsize=12, fontweight='bold')
    ax.set_xlabel('Friction', fontsize=9)
    ax.set_ylabel('Max Level', fontsize=9)
    ax.set_ylim(-1.5, 11)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.grid(True, alpha=0.3)

fig.suptitle(f'Max Terrain Level vs Friction -- go2_moe_cts @ iter 5500\n(Benchmark Score: {benchmark:.3f})',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(Path(__file__).parent / 'benchmark_friction_levels.png', dpi=150, bbox_inches='tight')

print(f"Benchmark Score: {benchmark:.4f}")
print(f"Terrain Scores: {dict(zip(terrain_keys, scores))}")
print(f"\nCharts saved:")
print(f"  {Path(__file__).parent / 'benchmark_terrain_scores.png'}")
print(f"  {Path(__file__).parent / 'benchmark_metrics_summary.png'}")
print(f"  {Path(__file__).parent / 'benchmark_friction_levels.png'}")
plt.close('all')
