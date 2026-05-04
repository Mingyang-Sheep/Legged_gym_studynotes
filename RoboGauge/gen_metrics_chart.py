"""
Regenerate benchmark_metrics_summary.png with clean horizontal bar chart.
No CJK fonts, no overlapping labels.
"""
import yaml
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


import re

def parse_mean(s):
    """Parse '0.5238 ± 0.2671' or similar -> extract first number"""
    numbers = re.findall(r'[-]?\d+\.?\d*', str(s))
    if numbers:
        return float(numbers[0])
    return 0.0


RESULT_FILE = (Path(__file__).parent /
               'go2_moe_stress_go2_moe_5500_stress' /
               '20260504-14-48-59_run' /
               'stress_benchmark_results.yaml')

with open(RESULT_FILE, 'r') as f:
    data = yaml.safe_load(f)

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
metrics     = [m[0] for m in metric_labels]

means  = [parse_mean(summary[m]['mean'])    for m in metrics]
mean50 = [parse_mean(summary[m]['mean@50']) for m in metrics]

y = np.arange(len(metrics))
height = 0.3

ax.barh(y + height/2, means,  height, label='Mean (all commands)',
        color='#90CAF9', edgecolor='white')
ax.barh(y - height/2, mean50, height, label='Mean@50 (worst 50%)',
        color='#1565C0', edgecolor='white')

ax.set_yticks(y)
ax.set_yticklabels(metric_names, fontsize=10)
ax.set_xlim(0, 0.80)
ax.set_xlabel('Score', fontsize=11)
ax.set_title('Per-Metric Summary (across all terrains)', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.xaxis.grid(True, alpha=0.3)
ax.invert_yaxis()

# Annotate values clearly
for i, (m_val, m50_val) in enumerate(zip(means, mean50)):
    ax.text(m_val + 0.008, i + height/2,
            f'{m_val:.3f}', va='center', fontsize=8, color='#555555')
    ax.text(m50_val + 0.008, i - height/2,
            f'{m50_val:.3f}', va='center', fontsize=8, color='white', fontweight='bold')

fig.subplots_adjust(left=0.18, right=0.95, top=0.93, bottom=0.08)
out = Path(__file__).parent / 'benchmark_metrics_summary.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Done: {out}')
