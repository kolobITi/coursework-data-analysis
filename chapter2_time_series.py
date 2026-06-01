# -*- coding: utf-8 -*-
"""
Курсовой проект. Глава 2.
Первичный анализ набора данных с временными рядами
Датасет: CWRU Bearing Dataset (Kaggle, astrollama)
Автор: Колобов Егор Евгеньевич, ЕТ-113
"""
import warnings, json
from pathlib import Path
from collections import defaultdict

import numpy as np
import scipy.io
import scipy.signal
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import kurtosis, skew

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE = Path('/Users/admin/Desktop/coursework_ch2_cwru/cwru')
OUT  = Path('/Users/admin/Desktop/coursework_ch2_cwru/figures')
OUT.mkdir(exist_ok=True)

FS   = 12000   # частота дискретизации, Гц
SEG  = 4096    # длина сегмента для анализа

CLASSES = {
    'Normal':          ('Normal',  'Норма',              '#4472C4'),
    'IR':              ('IR',      'Дефект внутр. кольца','#70AD47'),
    'OR':              ('OR',      'Дефект внешн. кольца','#ED7D31'),
    'B':               ('B',       'Дефект тела качения', '#FF4444'),
}

DPI = 180
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.titlesize': 13, 'axes.titleweight': 'bold',
    'axes.labelsize': 11, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
})

# ── Загрузка всех сигналов ────────────────────────────────────────────────────
print('Загрузка данных...')

def load_de_signal(fpath):
    """Load Drive-End accelerometer signal from .mat file."""
    data = scipy.io.loadmat(str(fpath))
    de_keys = [k for k in data.keys() if 'DE_time' in k]
    if not de_keys:
        de_keys = [k for k in data.keys() if not k.startswith('_')]
    return data[de_keys[0]].flatten()

def get_class(fname):
    n = fname.name
    if n.startswith('No'): return 'Normal'
    if n.startswith('IR'): return 'IR'
    if n.startswith('OR'): return 'OR'
    if n.startswith('B0'): return 'B'
    return None

def get_load(fname):
    return int(fname.stem.split('_')[-1])

signals = defaultdict(list)  # class -> list of (signal, load, filename)
for f in sorted(BASE.glob('*.mat')):
    cls = get_class(f)
    if cls:
        sig = load_de_signal(f)
        signals[cls].append({'signal': sig, 'load': get_load(f), 'name': f.stem})

print('Загружено:')
for cls, items in signals.items():
    total_pts = sum(len(it['signal']) for it in items)
    print(f'  {CLASSES[cls][1]}: {len(items)} записей, {total_pts:,} отсчётов')

# ── РИС. 1 — Загрузка и первичное знакомство ─────────────────────────────────
print('Рис. 1...')
counts = {cls: len(items) for cls, items in signals.items()}
total_pts_by_class = {cls: sum(len(it['signal']) for it in items)
                      for cls, items in signals.items()}

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
cls_labels = [CLASSES[c][1] for c in CLASSES]
colors     = [CLASSES[c][2] for c in CLASSES]

axes[0].bar(cls_labels, [counts[c] for c in CLASSES], color=colors, edgecolor='white', alpha=0.9)
axes[0].set_title('Количество записей по классам')
axes[0].set_ylabel('Количество файлов'); axes[0].grid(axis='y', alpha=0.4)
for i, (cls, cnt) in enumerate(zip(CLASSES, [counts[c] for c in CLASSES])):
    axes[0].text(i, cnt + 0.1, str(cnt), ha='center', fontweight='bold', fontsize=12)

durations = {cls: sum(len(it['signal']) for it in items)/FS
             for cls, items in signals.items()}
axes[1].bar(cls_labels, [durations[c] for c in CLASSES], color=colors, edgecolor='white', alpha=0.9)
axes[1].set_title('Суммарная длительность сигналов по классам')
axes[1].set_ylabel('Длительность, секунды'); axes[1].grid(axis='y', alpha=0.4)
for i, c in enumerate(CLASSES):
    axes[1].text(i, durations[c]+0.5, f'{durations[c]:.1f}s', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
fig.savefig(OUT / 'fig1_overview.png', dpi=DPI, bbox_inches='tight')
plt.close()

# ── РИС. 2 — Визуализация исходных сигналов ──────────────────────────────────
print('Рис. 2...')
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
fig.suptitle('Вибросигналы подшипника при нагрузке 0 HP (Drive-End, нагрузка 0)', fontsize=13, fontweight='bold')

t_show = np.arange(SEG) / FS * 1000  # ms

for ax, cls in zip(axes, CLASSES):
    # Load = 0
    item = next(it for it in signals[cls] if it['load'] == 0)
    seg = item['signal'][:SEG]
    color = CLASSES[cls][2]
    ax.plot(t_show, seg, color=color, linewidth=0.7, alpha=0.9)
    ax.set_ylabel(CLASSES[cls][1], fontsize=10, color=color, fontweight='bold')
    ax.grid(alpha=0.3); ax.set_ylim(-3, 3)
    rms = np.sqrt(np.mean(seg**2))
    ax.axhline(rms, color='red', linestyle='--', linewidth=0.8, alpha=0.7, label=f'RMS={rms:.3f}')
    ax.legend(loc='upper right', fontsize=9)

axes[-1].set_xlabel('Время, мс')
plt.tight_layout()
fig.savefig(OUT / 'fig2_raw_signals.png', dpi=DPI, bbox_inches='tight')
plt.close()

# ── РИС. 3 — Статистический анализ ───────────────────────────────────────────
print('Рис. 3...')
stats_records = []
for cls, items in signals.items():
    for it in items:
        seg = it['signal'][:SEG]
        stats_records.append({
            'class': CLASSES[cls][1], 'load': it['load'],
            'mean': np.mean(seg), 'std': np.std(seg),
            'rms': np.sqrt(np.mean(seg**2)),
            'peak': np.max(np.abs(seg)),
            'kurtosis': kurtosis(seg),
            'skewness': skew(seg),
            'crest_factor': np.max(np.abs(seg)) / (np.sqrt(np.mean(seg**2)) + 1e-9),
        })
df_stats = pd.DataFrame(stats_records)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle('Статистические характеристики вибросигналов по классам', fontsize=13, fontweight='bold')
metrics = ['rms', 'kurtosis', 'crest_factor', 'peak']
titles  = ['RMS (среднеквадратичное значение)', 'Куртозис (эксцесс)',
           'Крест-фактор', 'Пиковое значение |x|_max']

for ax, metric, title in zip(axes.flat, metrics, titles):
    data_by_cls = [df_stats[df_stats['class']==CLASSES[c][1]][metric].values for c in CLASSES]
    bp = ax.boxplot(data_by_cls, labels=cls_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_title(title, fontsize=11); ax.grid(axis='y', alpha=0.4)

plt.tight_layout()
fig.savefig(OUT / 'fig3_statistics.png', dpi=DPI, bbox_inches='tight')
plt.close()

# ── РИС. 4 — Пропуски и выбросы ──────────────────────────────────────────────
print('Рис. 4...')
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Пропуски (NaN/Inf)
missing_data = {}
for cls, items in signals.items():
    n_missing = sum(np.sum(~np.isfinite(it['signal'])) for it in items)
    n_total   = sum(len(it['signal']) for it in items)
    missing_data[CLASSES[cls][1]] = (n_missing, n_total)

labels_ = list(missing_data.keys())
miss_pct = [v[0]/v[1]*100 for v in missing_data.values()]
axes[0].bar(labels_, miss_pct, color=colors, edgecolor='white', alpha=0.9)
axes[0].set_title('Доля пропущенных/некорректных значений')
axes[0].set_ylabel('Доля, %'); axes[0].set_ylim(0, 5); axes[0].grid(axis='y', alpha=0.4)
for i, v in enumerate(miss_pct):
    axes[0].text(i, v + 0.05, f'{v:.3f}%', ha='center', fontsize=10, fontweight='bold')

# Выбросы (IQR метод)
outlier_pct = {}
for cls, items in signals.items():
    sig_all = np.concatenate([it['signal'] for it in items])
    q1, q3 = np.percentile(sig_all, [25, 75])
    iqr = q3 - q1
    n_out = np.sum((sig_all < q1 - 3*iqr) | (sig_all > q3 + 3*iqr))
    outlier_pct[CLASSES[cls][1]] = n_out / len(sig_all) * 100

axes[1].bar(list(outlier_pct.keys()), list(outlier_pct.values()),
            color=colors, edgecolor='white', alpha=0.9)
axes[1].set_title('Доля выбросов (метод IQR×3)')
axes[1].set_ylabel('Доля, %'); axes[1].grid(axis='y', alpha=0.4)
for i, v in enumerate(outlier_pct.values()):
    axes[1].text(i, v + 0.02, f'{v:.2f}%', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
fig.savefig(OUT / 'fig4_missing_outliers.png', dpi=DPI, bbox_inches='tight')
plt.close()

# ── РИС. 5 — Частотный анализ (FFT) ──────────────────────────────────────────
print('Рис. 5...')
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
fig.suptitle('Спектр мощности (FFT) вибросигналов — нагрузка 0 HP', fontsize=13, fontweight='bold')

for ax, cls in zip(axes, CLASSES):
    item  = next(it for it in signals[cls] if it['load'] == 0)
    seg   = item['signal'][:SEG*4]  # longer segment for better freq resolution
    freqs = np.fft.rfftfreq(len(seg), 1/FS)
    fft   = np.abs(np.fft.rfft(seg)) ** 2 / len(seg)
    color = CLASSES[cls][2]
    ax.semilogy(freqs, fft, color=color, linewidth=0.8, alpha=0.9)
    ax.set_ylabel(CLASSES[cls][1], fontsize=10, color=color, fontweight='bold')
    ax.set_xlim(0, 3000); ax.grid(alpha=0.3)

axes[-1].set_xlabel('Частота, Гц')
plt.tight_layout()
fig.savefig(OUT / 'fig5_fft_spectrum.png', dpi=DPI, bbox_inches='tight')
plt.close()

# ── РИС. 6 — Анализ диапазонов и распределений ───────────────────────────────
print('Рис. 6...')
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle('Распределения амплитуд вибросигналов по классам и нагрузкам', fontsize=13, fontweight='bold')

# Row 0: распределение амплитуд
for cls, ax_col in zip(CLASSES, [0, 1, 0, 1]):
    pass  # use different layout

for i, cls in enumerate(CLASSES):
    ax = axes[i // 2][i % 2]
    color = CLASSES[cls][2]
    # All loads
    for it in signals[cls]:
        seg = it['signal'][:SEG]
        ax.hist(seg, bins=60, alpha=0.4, color=color, density=True, label=f'Load {it["load"]}')
    ax.set_title(CLASSES[cls][1], color=color, fontweight='bold')
    ax.set_xlabel('Амплитуда'); ax.set_ylabel('Плотность вероятности')
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

plt.tight_layout()
fig.savefig(OUT / 'fig6_distributions.png', dpi=DPI, bbox_inches='tight')
plt.close()

# ── РИС. 7 — Корреляционный анализ + шумы (SNR) ──────────────────────────────
print('Рис. 7...')
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Корреляционная матрица RMS по классу × нагрузке
pivot_data = df_stats.pivot_table(values='rms', index='class', columns='load')
sns.heatmap(pivot_data, ax=axes[0], annot=True, fmt='.3f',
            cmap='YlOrRd', linewidths=0.5, cbar_kws={'label': 'RMS'})
axes[0].set_title('RMS по классу и нагрузке (HP)')
axes[0].set_xlabel('Нагрузка, HP'); axes[0].set_ylabel('Класс дефекта')

# SNR оценка
snr_data = {}
normal_signals = [it['signal'][:SEG] for it in signals['Normal']]
noise_var = np.mean([np.var(sig) for sig in normal_signals])

for cls, items in signals.items():
    snr_vals = []
    for it in items:
        seg = it['signal'][:SEG]
        signal_power = np.var(seg)
        snr_db = 10 * np.log10((signal_power) / (noise_var + 1e-9))
        snr_vals.append(snr_db)
    snr_data[CLASSES[cls][1]] = snr_vals

axes[1].boxplot(list(snr_data.values()), labels=list(snr_data.keys()),
                patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='navy'))
axes[1].axhline(0, color='red', linestyle='--', linewidth=1, label='0 dB')
axes[1].set_title('Отношение сигнал/шум (SNR) по классам')
axes[1].set_ylabel('SNR, dB'); axes[1].legend(); axes[1].grid(axis='y', alpha=0.4)

plt.tight_layout()
fig.savefig(OUT / 'fig7_correlation_snr.png', dpi=DPI, bbox_inches='tight')
plt.close()

# ── Статистика ────────────────────────────────────────────────────────────────
stats_out = {
    'dataset':      'CWRU Bearing Dataset',
    'source':       'https://www.kaggle.com/datasets/astrollama/cwru-case-western-reserve-university-dataset',
    'license':      'MIT',
    'total_files':  sum(len(v) for v in signals.values()),
    'classes':      {cls: {'ru': CLASSES[cls][1], 'files': len(items)}
                     for cls, items in signals.items()},
    'sampling_rate_hz': FS,
    'loads_hp':     [0, 1, 2, 3],
    'rms_by_class': {CLASSES[cls][1]: round(float(df_stats[df_stats['class']==CLASSES[cls][1]]['rms'].mean()), 4)
                     for cls in CLASSES},
    'kurtosis_by_class': {CLASSES[cls][1]: round(float(df_stats[df_stats['class']==CLASSES[cls][1]]['kurtosis'].mean()), 2)
                          for cls in CLASSES},
}
(OUT / 'stats.json').write_text(json.dumps(stats_out, ensure_ascii=False, indent=2), encoding='utf-8')
print('\n✅ Готово!')
print(json.dumps(stats_out, ensure_ascii=False, indent=2))
