import csv
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

BASE = Path(__file__).resolve().parents[1]
META_DIR = BASE / 'data' / 'metadata'
ANA_DIR = BASE / 'analysis'
OUT_DIR = BASE / 'figures'
OUT_DIR.mkdir(parents=True, exist_ok=True)

def center_square_or_fit(im, size=360):
    w, h = im.size
    canvas = Image.new('RGB', (size, size), (248, 249, 250))
    scale = min(size / w, size / h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = im.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas

# 1. Domain samples figure (3 columns x 2 rows)
fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.4), dpi=300)
col_titles = [
    '植物病害（Apple leaf）\n源域 PlantVillage → 目标域 PlantDoc',
    '日常物品（Alarm_Clock）\n源域 Product → 目标域 Real World',
    '合成到真实（aeroplane）\n源域 Synthetic → 目标域 Real',
]
plant_tgt_raw = Image.open(META_DIR / 'plant_target.jpg').convert('RGB')
plant_tgt_crop = plant_tgt_raw.crop((32, 1289, 2304, 3655))
imgs = [
    [
        Image.open(META_DIR / 'plant_source.jpg').convert('RGB'),
        Image.open(META_DIR / 'office_source.jpg').convert('RGB'),
        Image.open(META_DIR / 'visda_source.png').convert('RGB'),
    ],
    [
        plant_tgt_crop,
        Image.open(META_DIR / 'office_target.jpg').convert('RGB'),
        Image.open(META_DIR / 'visda_target.jpg').convert('RGB'),
    ],
]
row_labels = ['源域样本', '目标域样本']
for r in range(2):
    for c in range(3):
        ax = axes[r, c]
        ax.imshow(center_square_or_fit(imgs[r][c], 360))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor('#CBD5E1'); sp.set_linewidth(0.8)
        if r == 0:
            ax.set_title(col_titles[c], fontsize=10.5, fontweight='bold', pad=8)
        if c == 0:
            ax.set_ylabel(row_labels[r], fontsize=10.5, fontweight='bold', labelpad=8)
plt.tight_layout(pad=1.2)
plt.savefig(OUT_DIR / 'fig_2_1_domain_samples.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# 2. Three-seed summary figure (Accuracy & ECE mean +/- sd across seeds 42, 7, 2026)
rows = list(csv.DictReader((ANA_DIR / 'three_seed_summary.csv').open(encoding='utf-8-sig')))
lookup = {(r['domain'], r['method']): r for r in rows}
domains = [('plant', '植物病害'), ('office', '日常物品'), ('visda', 'VisDA')]
methods = [
    ('baseline', '原模型', '#64748B'),
    ('augmix', 'AugMix', '#2563EB'),
    ('ls_entropy', '平滑＋熵', '#059669'),
    ('alis', 'ALiS', '#D97706'),
]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.3), dpi=300)
x = np.arange(len(domains))
width = 0.19
for ax, metric, ylabel, title, ylim in [
    (ax1, 'accuracy', '准确率 / %', '(a) 3颗随机种子目标测试准确率（均值±标准差）', (0, 86)),
    (ax2, 'ece', '期望校准误差 ECE / %', '(b) 3颗随机种子目标测试 ECE（均值±标准差）', (0, 86)),
]:
    for idx, (mkey, mlabel, color) in enumerate(methods):
        means = [float(lookup[(d, mkey)][metric + '_mean']) * 100 for d, _ in domains]
        sds = [float(lookup[(d, mkey)][metric + '_sd']) * 100 for d, _ in domains]
        pos = x + (idx - 1.5) * width
        bars = ax.bar(pos, means, width, yerr=sds, capsize=3, label=mlabel,
                      color=color, alpha=0.88, error_kw={'elinewidth': 1.0, 'ecolor': '#1E293B'})
        for b, m, sd in zip(bars, means, sds):
            ax.text(b.get_x() + b.get_width() / 2, m + sd + 1.4, f'{m:.1f}',
                    ha='center', va='bottom', fontsize=7.2, color='#1E293B')
    ax.set_xticks(x); ax.set_xticklabels([dl for _, dl in domains], fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10.5, fontweight='bold', pad=8)
    ax.set_ylim(*ylim); ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=4, frameon=False, fontsize=9.5)
plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(OUT_DIR / 'fig_3_4_three_seeds.png', dpi=300, bbox_inches='tight')
plt.close(fig)

# 3. Reliability diagrams (seed=42)
bin_rows = list(csv.DictReader((ANA_DIR / 'reliability_bins.csv').open(encoding='utf-8-sig')))
run_rows = list(csv.DictReader((ANA_DIR / 'all_run_metrics.csv').open(encoding='utf-8-sig')))
ece_map = {(r['domain'], r['method'], int(r['seed'])): float(r['ece']) * 100 for r in run_rows}
fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.1), dpi=300, sharey=True)
rel_methods = [
    ('baseline', '原模型', '#64748B', 'o', '-'),
    ('smoothing_plus_temperature', '平滑＋温度', '#DC2626', 's', '--'),
    ('ls_entropy', '平滑＋熵', '#059669', '^', '-'),
    ('dac', 'DAC', '#2563EB', 'D', '-'),
]
for ax, (dkey, dtitle) in zip(axes, domains):
    ax.plot([0, 100], [0, 100], linestyle=':', color='#0F172A', linewidth=1.2, label='理想校准线 (y=x)')
    for mkey, mlabel, color, marker, ls in rel_methods:
        pts = [r for r in bin_rows if r['domain'] == dkey and r['method'] == mkey and int(r['seed']) == 42 and int(r['n']) >= 5]
        xs = [float(r['confidence']) * 100 for r in pts]
        ys = [float(r['accuracy']) * 100 for r in pts]
        ece_val = ece_map[(dkey, mkey, 42)]
        ax.plot(xs, ys, marker=marker, linestyle=ls, color=color, linewidth=1.7, markersize=4.5,
                label=f'{mlabel} ({ece_val:.1f}%)')
    ax.set_title(dtitle, fontsize=10.5, fontweight='bold', pad=8)
    ax.set_xlabel('分箱平均置信度 / %', fontsize=9.5)
    ax.set_xlim(5, 102); ax.set_ylim(0, 102)
    ax.grid(True, linestyle='--', alpha=0.32)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', fontsize=7.5, frameon=True, framealpha=0.88)
axes[0].set_ylabel('分箱经验准确率 / %', fontsize=9.5)
plt.tight_layout(pad=1.1)
plt.savefig(OUT_DIR / 'fig_3_3_reliability.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print('FIGURES GENERATED IN:', OUT_DIR)
