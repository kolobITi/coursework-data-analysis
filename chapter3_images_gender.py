# -*- coding: utf-8 -*-
"""
Курсовой проект. Глава 3.
Первичный анализ набора данных с изображениями
Датасет: UTKFace (Kaggle, moritzm00) — распознавание пола
Автор: Колобов Егор Евгеньевич, ЕТ-113
"""
import os, json, random, warnings
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageStat

warnings.filterwarnings('ignore')
random.seed(42); np.random.seed(42)

BASE = Path('/Users/admin/Desktop/coursework_ch3_gender/UTKFace')
OUT  = Path('/Users/admin/Desktop/coursework_ch3_gender/figures')
OUT.mkdir(exist_ok=True)

DPI = 180
plt.rcParams.update({'font.family':'DejaVu Sans','axes.titlesize':13,
                     'axes.titleweight':'bold','axes.labelsize':11,
                     'xtick.labelsize':10,'ytick.labelsize':10})

# ── Parse filenames: [age]_[gender]_[race]_[date].jpg ────────────────────────
# gender: 0=Male, 1=Female  race: 0=White,1=Black,2=Asian,3=Indian,4=Others
print('Загрузка метаданных...')
records = []
for p in BASE.glob('*.jpg'):
    parts = p.stem.replace('.chip','').split('_')
    if len(parts) >= 3:
        try:
            age = int(parts[0]); gender = int(parts[1]); race = int(parts[2])
            records.append({'path':str(p),'age':age,'gender':gender,'race':race,
                            'gender_ru':'Мужской' if gender==0 else 'Женский'})
        except: pass

import pandas as pd
df = pd.DataFrame(records)
print(f'Загружено: {len(df)} изображений')
print(df['gender_ru'].value_counts().to_string())

COLORS = {'Мужской':'#4472C4','Женский':'#ED7D31'}
RACE_NAMES = {0:'Европеоид',1:'Негроид',2:'Азиат',3:'Индиец',4:'Другие'}

# ── РИС. 1 — Баланс классов ──────────────────────────────────────────────────
print('Рис. 1...')
gc = df['gender_ru'].value_counts()
fig, axes = plt.subplots(1,2, figsize=(12,5))
axes[0].bar(gc.index, gc.values,
            color=[COLORS[g] for g in gc.index], edgecolor='white', alpha=0.9)
axes[0].set_title('Количество изображений по полу')
axes[0].set_ylabel('Количество'); axes[0].grid(axis='y', alpha=0.4)
for i,(g,v) in enumerate(zip(gc.index, gc.values)):
    axes[0].text(i, v+50, str(v), ha='center', fontweight='bold', fontsize=12)

axes[1].pie(gc.values, labels=gc.index,
            colors=[COLORS[g] for g in gc.index],
            autopct='%1.1f%%', startangle=140,
            wedgeprops={'edgecolor':'white','linewidth':1.5})
axes[1].set_title('Доли классов (пол)')
plt.tight_layout()
fig.savefig(OUT/'fig1_class_balance.png', dpi=DPI, bbox_inches='tight'); plt.close()

# ── РИС. 2 — Примеры изображений по полу ─────────────────────────────────────
print('Рис. 2...')
fig, axes = plt.subplots(2,6, figsize=(16,7))
fig.suptitle('Примеры изображений из UTKFace (по полу)', fontsize=13, fontweight='bold')
for row, (gender_val, label) in enumerate([(0,'Мужской'),(1,'Женский')]):
    samples = df[df.gender==gender_val].sample(6, random_state=row).itertuples()
    for col, row_data in enumerate(samples):
        ax = axes[row][col]
        with Image.open(row_data.path) as im:
            ax.imshow(im.resize((100,100)))
        ax.set_title(f'{label}\nвозраст {row_data.age}', fontsize=8)
        ax.axis('off')
plt.tight_layout()
fig.savefig(OUT/'fig2_samples.png', dpi=DPI, bbox_inches='tight'); plt.close()

# ── РИС. 3 — Возрастное распределение по полу ────────────────────────────────
print('Рис. 3...')
fig, axes = plt.subplots(1,2, figsize=(13,5))
for label, color in COLORS.items():
    sub = df[df.gender_ru==label]
    axes[0].hist(sub.age, bins=30, alpha=0.6, label=label, color=color, edgecolor='white')
axes[0].set_title('Возрастное распределение по полу')
axes[0].set_xlabel('Возраст'); axes[0].set_ylabel('Количество')
axes[0].legend(); axes[0].grid(axis='y', alpha=0.4)

# По расе
rc = df['race'].map(RACE_NAMES).value_counts()
axes[1].bar(rc.index, rc.values,
            color=['#4472C4','#70AD47','#ED7D31','#FF4444','#9966CC'],
            edgecolor='white', alpha=0.9)
axes[1].set_title('Распределение по расовой принадлежности')
axes[1].set_ylabel('Количество'); axes[1].grid(axis='y', alpha=0.4)
axes[1].tick_params(axis='x', rotation=15)
plt.tight_layout()
fig.savefig(OUT/'fig3_age_race.png', dpi=DPI, bbox_inches='tight'); plt.close()

# ── РИС. 4 — Качество изображений ────────────────────────────────────────────
print('Рис. 4 (анализ качества ~500 изображений)...')
sample_paths = df.sample(500, random_state=42)['path'].tolist()
brightness = []; contrast = []; sizes = []
for p in sample_paths:
    try:
        with Image.open(p) as im:
            w, h = im.size; sizes.append((w,h))
            stat = ImageStat.Stat(im.convert('RGB'))
            brightness.append(sum(stat.mean[:3])/3)
            contrast.append(sum(stat.stddev[:3])/3)
    except: pass

fig, axes = plt.subplots(1,3, figsize=(15,5))
axes[0].hist(brightness, bins=25, color='#4472C4', edgecolor='white', alpha=0.9)
axes[0].axvline(np.mean(brightness), color='red', linestyle='--',
                label=f'Среднее: {np.mean(brightness):.1f}')
axes[0].set_title('Яркость изображений'); axes[0].set_xlabel('Яркость')
axes[0].set_ylabel('Кол-во'); axes[0].legend(); axes[0].grid(axis='y', alpha=0.4)

axes[1].hist(contrast, bins=25, color='#70AD47', edgecolor='white', alpha=0.9)
axes[1].axvline(np.mean(contrast), color='red', linestyle='--',
                label=f'Среднее: {np.mean(contrast):.1f}')
axes[1].set_title('Контрастность изображений'); axes[1].set_xlabel('Контраст (σ)')
axes[1].legend(); axes[1].grid(axis='y', alpha=0.4)

ws = [s[0] for s in sizes]; hs = [s[1] for s in sizes]
axes[2].scatter(ws, hs, alpha=0.3, s=10, color='#ED7D31')
axes[2].set_title('Разрешения изображений (выборка)')
axes[2].set_xlabel('Ширина, пкс'); axes[2].set_ylabel('Высота, пкс')
axes[2].grid(alpha=0.4)
plt.tight_layout()
fig.savefig(OUT/'fig4_quality.png', dpi=DPI, bbox_inches='tight'); plt.close()

# ── РИС. 5 — Фильтрация по качеству ─────────────────────────────────────────
print('Рис. 5...')
ok_bright = sum(1 for b in brightness if 30 <= b <= 230)
ok_contr  = sum(1 for c in contrast if c >= 10)
ok_size   = sum(1 for s in sizes if min(s) >= 50)
ok_all    = sum(1 for b,c,s in zip(brightness,contrast,sizes) if 30<=b<=230 and c>=10 and min(s)>=50)
n = len(brightness)

fig, ax = plt.subplots(figsize=(10,5))
labels_ = ['Яркость\n30–230','Контраст\n≥10','Размер\n≥50px','Все\nкритерии']
vals_   = [ok_bright, ok_contr, ok_size, ok_all]
colors_ = ['#4472C4','#70AD47','#ED7D31','#5B9BD5']
bars = ax.bar(labels_, vals_, color=colors_, edgecolor='white')
ax.axhline(n, color='red', linestyle='--', label=f'Выборка: {n}')
ax.set_title('Оценка качества изображений (выборка 500 шт.)')
ax.set_ylabel('Количество'); ax.legend(); ax.grid(axis='y', alpha=0.4)
ax.set_ylim(0, n*1.1)
for bar, v in zip(bars, vals_):
    ax.text(bar.get_x()+bar.get_width()/2, v+2, str(v), ha='center', fontweight='bold')
plt.tight_layout()
fig.savefig(OUT/'fig5_quality_filter.png', dpi=DPI, bbox_inches='tight'); plt.close()

# ── РИС. 6 — Аннотации: формат и структура ───────────────────────────────────
print('Рис. 6...')
age_bins = pd.cut(df.age, bins=[0,12,18,30,45,60,100],
                  labels=['0–12','13–18','19–30','31–45','46–60','61+'])
age_gender = pd.crosstab(age_bins, df.gender_ru)

fig, axes = plt.subplots(1,2, figsize=(13,5))
age_gender.plot(kind='bar', ax=axes[0],
                color=[COLORS['Женский'], COLORS['Мужской']],
                edgecolor='white', alpha=0.9)
axes[0].set_title('Распределение по возрастным группам и полу')
axes[0].set_xlabel('Возрастная группа'); axes[0].set_ylabel('Количество')
axes[0].tick_params(axis='x', rotation=15); axes[0].legend()
axes[0].grid(axis='y', alpha=0.4)

# Баланс по полу в каждой возрастной группе
ratio = age_gender.div(age_gender.sum(axis=1), axis=0) * 100
ratio.plot(kind='bar', stacked=True, ax=axes[1],
           color=[COLORS['Женский'], COLORS['Мужской']],
           edgecolor='white', alpha=0.9)
axes[1].set_title('Доля мужчин/женщин по возрастным группам (%)')
axes[1].set_xlabel('Возрастная группа'); axes[1].set_ylabel('%')
axes[1].tick_params(axis='x', rotation=15)
axes[1].axhline(50, color='white', linestyle='--', linewidth=1)
axes[1].grid(axis='y', alpha=0.4)
plt.tight_layout()
fig.savefig(OUT/'fig6_annotations.png', dpi=DPI, bbox_inches='tight'); plt.close()

# ── РИС. 7 — Аудит разметки ──────────────────────────────────────────────────
print('Рис. 7...')
fig, axes = plt.subplots(2,6, figsize=(16,7))
fig.suptitle('Аудит разметки: разнообразие условий съёмки (UTKFace)', fontsize=13, fontweight='bold')
# Mix ages and races
for idx in range(12):
    row_val = 0 if idx < 6 else 1
    subset = df[(df.gender==row_val) & (df.age > 10) & (df.age < 70)]
    sample = subset.sample(1, random_state=idx*7).iloc[0]
    ax = axes[idx//6][idx%6]
    with Image.open(sample.path) as im:
        ax.imshow(im.resize((100,100)))
    label = 'Муж.' if row_val==0 else 'Жен.'
    ax.set_title(f'✓ {label}, {sample.age}л\n{RACE_NAMES.get(sample.race,"")}', fontsize=7)
    ax.axis('off')
plt.tight_layout()
fig.savefig(OUT/'fig7_audit.png', dpi=DPI, bbox_inches='tight'); plt.close()

# Stats
stats = {
    'dataset': 'UTKFace (Cropped & Aligned)',
    'source': 'https://www.kaggle.com/datasets/moritzm00/utkface-cropped',
    'license': 'Creative Commons BY-NC-SA',
    'total_images': len(df),
    'male': int((df.gender==0).sum()),
    'female': int((df.gender==1).sum()),
    'age_min': int(df.age.min()), 'age_max': int(df.age.max()),
    'avg_brightness': round(float(np.mean(brightness)),1),
    'ok_quality_pct': round(ok_all/n*100, 1),
}
(OUT/'stats.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2))
print('\n✅ Готово!'); print(json.dumps(stats, ensure_ascii=False, indent=2))
