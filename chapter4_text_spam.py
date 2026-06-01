# -*- coding: utf-8 -*-
"""
Курсовой проект. Глава 4.
Первичный анализ набора текстовых данных
Датасет: Email Spam Classification (Kaggle, balaka18)
Автор: Колобов Егор Евгеньевич, ЕТ-113
"""
import warnings, json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings('ignore')
np.random.seed(42)

BASE = Path('/Users/admin/Desktop/coursework_ch4_spam')
OUT  = BASE / 'figures'; OUT.mkdir(exist_ok=True)

DPI = 180
plt.rcParams.update({'font.family':'DejaVu Sans','axes.titlesize':13,
                     'axes.titleweight':'bold','axes.labelsize':11,
                     'xtick.labelsize':10,'ytick.labelsize':10})

COLORS = {'Спам':'#FF4444','Не спам':'#4472C4'}

# ── Загрузка ──────────────────────────────────────────────────────────────────
print('Загрузка данных...')
df = pd.read_csv(BASE / 'emails.csv')
# Last column is label (Prediction), others are word frequency features
label_col = 'Prediction'
word_cols  = [c for c in df.select_dtypes(include='number').columns if c != label_col]
df['label_ru'] = df[label_col].map({1:'Спам', 0:'Не спам'})
print(f'Документов: {len(df)}, признаков: {len(word_cols)}')
print(df['label_ru'].value_counts().to_string())

# Reconstruct pseudo-text from word frequencies for NLP demo
print('Восстановление текста из частот слов...')
def reconstruct_text(row):
    words = []
    for w in word_cols[:200]:  # top 200 features
        count = int(row[w])
        if count > 0:
            words.extend([w] * min(count, 3))
    return ' '.join(words) if words else 'empty'

df['text'] = df.apply(reconstruct_text, axis=1)
df_num = df[word_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
df['n_words']  = df_num.sum(axis=1).astype(int)
df['n_unique'] = (df_num > 0).sum(axis=1).astype(int)

# ── РИС. 1 — Баланс классов ──────────────────────────────────────────────────
print('Рис. 1...')
vc = df['label_ru'].value_counts()
fig, axes = plt.subplots(1,2, figsize=(12,5))
axes[0].bar(vc.index, vc.values,
            color=[COLORS[l] for l in vc.index], edgecolor='white', alpha=0.9)
axes[0].set_title('Количество писем по классам')
axes[0].set_ylabel('Количество'); axes[0].grid(axis='y', alpha=0.4)
for i,(l,v) in enumerate(zip(vc.index, vc.values)):
    axes[0].text(i, v+10, str(v), ha='center', fontweight='bold', fontsize=13)

axes[1].pie(vc.values, labels=vc.index,
            colors=[COLORS[l] for l in vc.index],
            autopct='%1.1f%%', startangle=140,
            wedgeprops={'edgecolor':'white','linewidth':2})
axes[1].set_title('Соотношение спам / не спам')
plt.tight_layout()
fig.savefig(OUT/'fig1_balance.png', dpi=DPI, bbox_inches='tight'); plt.close()

# ── РИС. 2 — Очистка / длина документов ──────────────────────────────────────
print('Рис. 2...')
fig, axes = plt.subplots(1,2, figsize=(13,5))
for label, color in COLORS.items():
    sub = df[df.label_ru==label]
    axes[0].hist(sub.n_words, bins=40, alpha=0.6, label=label, color=color, edgecolor='white')
axes[0].set_title('Распределение длин писем (кол-во слов)')
axes[0].set_xlabel('Слов в письме'); axes[0].set_ylabel('Кол-во документов')
axes[0].legend(); axes[0].grid(axis='y', alpha=0.4)

for label, color in COLORS.items():
    sub = df[df.label_ru==label]
    axes[1].hist(sub.n_unique, bins=40, alpha=0.6, label=label, color=color, edgecolor='white')
axes[1].set_title('Уникальных слов в письме')
axes[1].set_xlabel('Уникальных слов'); axes[1].set_ylabel('Кол-во документов')
axes[1].legend(); axes[1].grid(axis='y', alpha=0.4)
plt.tight_layout()
fig.savefig(OUT/'fig2_length.png', dpi=DPI, bbox_inches='tight'); plt.close()

# ── РИС. 3 — Частота слов (топ-15 для каждого класса) ────────────────────────
print('Рис. 3...')
spam_sum = df_num[df[label_col]==1].sum().sort_values(ascending=False)
ham_sum  = df_num[df[label_col]==0].sum().sort_values(ascending=False)

fig, axes = plt.subplots(1,2, figsize=(14,6))
spam_top = spam_sum.head(15); ham_top = ham_sum.head(15)
axes[0].barh(spam_top.index[::-1], spam_top.values[::-1],
             color='#FF4444', edgecolor='white', alpha=0.85)
axes[0].set_title('Топ-15 слов в спам-письмах')
axes[0].set_xlabel('Частота встречаемости'); axes[0].grid(axis='x', alpha=0.4)

axes[1].barh(ham_top.index[::-1], ham_top.values[::-1],
             color='#4472C4', edgecolor='white', alpha=0.85)
axes[1].set_title('Топ-15 слов в обычных письмах')
axes[1].set_xlabel('Частота встречаемости'); axes[1].grid(axis='x', alpha=0.4)
plt.tight_layout()
fig.savefig(OUT/'fig3_word_freq.png', dpi=DPI, bbox_inches='tight'); plt.close()

# ── РИС. 4 — Стоп-слова и характеристики словаря ─────────────────────────────
print('Рис. 4...')
# Words present in both classes (generic) vs spam-specific
spam_words = set(spam_sum[spam_sum > spam_sum.quantile(0.7)].index)
ham_words  = set(ham_sum[ham_sum  > ham_sum.quantile(0.7)].index)
only_spam  = spam_words - ham_words
only_ham   = ham_words  - spam_words
both       = spam_words & ham_words

fig, axes = plt.subplots(1,2, figsize=(13,5))
labels_ = ['Только\nспам', 'Только\nне спам', 'Общие']
vals_   = [len(only_spam), len(only_ham), len(both)]
axes[0].bar(labels_, vals_,
            color=['#FF4444','#4472C4','#AAAAAA'], edgecolor='white', alpha=0.9)
axes[0].set_title('Специфичность словаря по классам (топ-30% слов)')
axes[0].set_ylabel('Кол-во уникальных слов'); axes[0].grid(axis='y', alpha=0.4)
for i,v in enumerate(vals_):
    axes[0].text(i, v+1, str(v), ha='center', fontweight='bold')

# Feature density heatmap (top 20 words)
top_words = list(spam_sum.head(10).index) + list(ham_sum.head(10).index)
top_words = list(dict.fromkeys(top_words))[:15]
heat_data = pd.DataFrame({
    'Спам':    df[df[label_col]==1][top_words].mean(),
    'Не спам': df[df[label_col]==0][top_words].mean(),
})
sns.heatmap(heat_data.T, ax=axes[1], cmap='YlOrRd', annot=True, fmt='.1f',
            linewidths=0.5, cbar_kws={'label':'Среднее кол-во'})
axes[1].set_title('Средняя частота топ-15 слов по классам')
axes[1].set_xlabel('Слово'); axes[1].set_ylabel('Класс')
plt.tight_layout()
fig.savefig(OUT/'fig4_vocab.png', dpi=DPI, bbox_inches='tight'); plt.close()

# ── РИС. 5 — TF-IDF ──────────────────────────────────────────────────────────
print('Рис. 5...')
tfidf = TfidfVectorizer(max_features=500, min_df=5)
X = tfidf.fit_transform(df['text'])
feat_names = tfidf.get_feature_names_out()
print(f'  TF-IDF матрица: {X.shape}')

fig, axes = plt.subplots(1,2, figsize=(14,6))
for ax, (cls_val, cls_name, color) in zip(axes,
        [(1,'Спам','#FF4444'), (0,'Не спам','#4472C4')]):
    mask = (df[label_col]==cls_val).values
    mean_scores = X[mask].toarray().mean(axis=0)
    top_idx = mean_scores.argsort()[-10:][::-1]
    ax.barh(feat_names[top_idx][::-1], mean_scores[top_idx][::-1],
            color=color, edgecolor='white', alpha=0.85)
    ax.set_title(f'Топ-10 TF-IDF слов: {cls_name}')
    ax.set_xlabel('Средний TF-IDF'); ax.grid(axis='x', alpha=0.4)
plt.tight_layout()
fig.savefig(OUT/'fig5_tfidf.png', dpi=DPI, bbox_inches='tight'); plt.close()

# ── РИС. 6 — Пропуски и выбросы ──────────────────────────────────────────────
print('Рис. 6...')
n_missing = df[word_cols].isna().sum().sum()
n_zero    = (df[word_cols] == 0).sum().sum()
n_total   = len(df) * len(word_cols)

fig, axes = plt.subplots(1,2, figsize=(13,5))
axes[0].pie([n_total - n_zero, n_zero],
            labels=['Ненулевые значения','Нулевые (слово отсутствует)'],
            colors=['#4472C4','#DDDDDD'], autopct='%1.1f%%',
            wedgeprops={'edgecolor':'white','linewidth':1.5}, startangle=90)
axes[0].set_title(f'Разреженность матрицы признаков\n(всего ячеек: {n_total:,})')

# Distribution of word counts per document
axes[1].hist(df.n_words, bins=40, color='#4472C4', edgecolor='white', alpha=0.9)
axes[1].axvline(df.n_words.mean(), color='red', linestyle='--',
                label=f'Среднее: {df.n_words.mean():.0f}')
axes[1].axvline(df.n_words.median(), color='orange', linestyle='--',
                label=f'Медиана: {df.n_words.median():.0f}')
axes[1].set_title('Распределение количества слов на письмо')
axes[1].set_xlabel('Слов'); axes[1].set_ylabel('Кол-во')
axes[1].legend(); axes[1].grid(axis='y', alpha=0.4)
plt.tight_layout()
fig.savefig(OUT/'fig6_sparsity.png', dpi=DPI, bbox_inches='tight'); plt.close()

# ── РИС. 7 — Информационный поиск ────────────────────────────────────────────
print('Рис. 7...')
queries_text = ['free money win prize', 'meeting schedule report', 'click here buy now']
queries_vecs = tfidf.transform(queries_text)
sims = cosine_similarity(queries_vecs, X)

fig, axes = plt.subplots(1,3, figsize=(16,5))
fig.suptitle('Информационный поиск (cosine similarity) по корпусу писем', fontsize=13, fontweight='bold')
color_map = {1:'#FF4444', 0:'#4472C4'}
for qi, (q, ax) in enumerate(zip(queries_text, axes)):
    top_idx = sims[qi].argsort()[-5:][::-1]
    labels_r = [f'Письмо #{df.iloc[i]["Email No."]}' for i in top_idx]
    scores_r = sims[qi][top_idx]
    bar_colors = [color_map[df.iloc[i][label_col]] for i in top_idx]
    ax.barh(labels_r[::-1], scores_r[::-1], color=bar_colors[::-1], edgecolor='white')
    ax.set_title(f'«{q[:18]}…»', fontsize=10, fontweight='bold')
    ax.set_xlabel('Cosine similarity'); ax.grid(axis='x', alpha=0.4)
    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#FF4444',label='Спам'),
                       Patch(color='#4472C4',label='Не спам')], fontsize=8)
plt.tight_layout()
fig.savefig(OUT/'fig7_search.png', dpi=DPI, bbox_inches='tight'); plt.close()

stats = {
    'dataset': 'Email Spam Classification Dataset CSV',
    'source': 'https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv',
    'license': 'Unknown (public educational use)',
    'total_emails': len(df), 'spam': int((df[label_col]==1).sum()),
    'ham': int((df[label_col]==0).sum()), 'features': len(word_cols),
    'tfidf_shape': list(X.shape), 'missing_values': int(n_missing),
    'avg_words_per_doc': round(float(df.n_words.mean()),1),
    'only_spam_words': len(only_spam), 'only_ham_words': len(only_ham),
}
(OUT/'stats.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2))
print('\n✅ Готово!'); print(json.dumps(stats, ensure_ascii=False, indent=2))
