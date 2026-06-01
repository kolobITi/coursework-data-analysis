# -*- coding: utf-8 -*-
"""
Курсовой проект. Глава 1.
Первичный анализ набора табличных данных
Датасет: World Population by Age Group 2020
Автор работы: Колобов Егор Евгеньевич

Что делает скрипт:
1) загружает CSV-файл;
2) выполняет первичный анализ табличных данных;
3) сохраняет графики, таблицы и краткие текстовые выводы в отдельные папки.

Перед запуском:
- скачай CSV с Kaggle;
- укажи имя файла в переменной DATA_FILE ниже;
- при необходимости установи библиотеки:
  pip install pandas numpy matplotlib seaborn plotly scipy scikit-learn
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

warnings.filterwarnings("ignore")

# ---------------------------
# 1. НАСТРОЙКИ
# ---------------------------
DATA_FILE = "WorldPopulationByAge2020.csv"   # <-- замени, если файл называется иначе
OUTPUT_DIR = Path("coursework_tabular_output")

PLOTS_DIR = OUTPUT_DIR / "plots"
TABLES_DIR = OUTPUT_DIR / "tables"
TEXT_DIR = OUTPUT_DIR / "text"

for folder in [OUTPUT_DIR, PLOTS_DIR, TABLES_DIR, TEXT_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 10
sns.set_theme(style="whitegrid")


# ---------------------------
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ---------------------------
def save_text(filename: str, text: str) -> None:
    path = TEXT_DIR / filename
    path.write_text(text, encoding="utf-8")


def save_df(df: pd.DataFrame, filename: str) -> None:
    path = TABLES_DIR / filename
    if filename.lower().endswith(".xlsx"):
        df.to_excel(path, index=False)
    else:
        df.to_csv(path, index=False, encoding="utf-8-sig")


def detect_country_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "country", "Country", "name", "Name", "location", "Location",
        "country_name", "Country Name"
    ]
    for col in candidates:
        if col in df.columns:
            return col

    # запасной вариант: первый объектный столбец с большим числом уникальных значений
    object_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
    if object_cols:
        scored = sorted(object_cols, key=lambda c: df[c].nunique(dropna=True), reverse=True)
        return scored[0]
    return None


def detect_numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def detect_categorical_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=["object", "string", "category"]).columns.tolist()


def make_additional_categories(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    """
    Если в датасете мало категориальных признаков, создаем дополнительные:
    1) population_band — категория по сумме возрастных групп;
    2) dominant_age_group — возрастная группа с максимальным населением.
    """
    df = df.copy()

    if len(numeric_cols) >= 2:
        total_col = "__total_population__"
        df[total_col] = df[numeric_cols].sum(axis=1)

        # Категория по суммарной численности
        try:
            df["population_band"] = pd.qcut(
                df[total_col],
                q=4,
                labels=["низкая", "ниже средней", "выше средней", "высокая"],
                duplicates="drop"
            )
        except Exception:
            df["population_band"] = "смешанная"

        # Доминирующая возрастная группа
        df["dominant_age_group"] = df[numeric_cols].idxmax(axis=1).astype(str)

    return df


def remove_duplicate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    df_clean = df.drop_duplicates().copy()
    removed = before - len(df_clean)
    return df_clean, removed


def outlier_bounds_iqr(series: pd.Series) -> tuple[float, float]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return lower, upper


def summarize_outliers(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for col in numeric_cols:
        s = df[col].dropna()
        if s.empty:
            continue
        lower, upper = outlier_bounds_iqr(s)
        mask = (df[col] < lower) | (df[col] > upper)
        rows.append({
            "Признак": col,
            "Нижняя граница": round(lower, 3),
            "Верхняя граница": round(upper, 3),
            "Количество выбросов": int(mask.sum()),
            "Доля выбросов, %": round(mask.mean() * 100, 2)
        })
    return pd.DataFrame(rows)


def add_noise(df: pd.DataFrame, columns: list[str], noise_level: float = 0.03, random_state: int = 42) -> pd.DataFrame:
    """
    Добавляет гауссов шум к двум числовым признакам.
    noise_level = 0.03 -> шум около 3% от std признака.
    """
    rng = np.random.default_rng(random_state)
    df_noisy = df.copy()

    for col in columns:
        if col not in df_noisy.columns:
            continue
        std = df_noisy[col].std()
        if pd.isna(std) or std == 0:
            continue
        noise = rng.normal(loc=0, scale=std * noise_level, size=len(df_noisy))
        df_noisy[f"{col}_noisy"] = df_noisy[col] + noise

    return df_noisy


def safe_label_encode(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    df_encoded = df.copy()
    for col in categorical_cols:
        df_encoded[f"{col}_label"] = pd.factorize(df_encoded[col].astype(str))[0]
    return df_encoded


def one_hot_encode_small_categories(df: pd.DataFrame, categorical_cols: list[str], max_unique: int = 15) -> pd.DataFrame:
    small_cols = [col for col in categorical_cols if df[col].nunique(dropna=True) <= max_unique]
    if not small_cols:
        return df.copy()
    return pd.get_dummies(df, columns=small_cols, drop_first=False)


def save_basic_info(df: pd.DataFrame) -> None:
    info_lines = []
    info_lines.append(f"Размер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов")
    info_lines.append("\nСписок столбцов:")
    for col in df.columns:
        info_lines.append(f"- {col}: {df[col].dtype}")

    info_lines.append("\nКоличество пропусков:")
    na = df.isna().sum()
    for col, cnt in na.items():
        info_lines.append(f"- {col}: {cnt}")

    save_text("00_basic_info.txt", "\n".join(info_lines))


# ---------------------------
# 3. ЗАГРУЗКА ДАННЫХ
# ---------------------------
data_path = Path(DATA_FILE)
if not data_path.exists():
    raise FileNotFoundError(
        f"Файл {DATA_FILE} не найден.\n"
        f"Скачай CSV с Kaggle и положи его рядом со скриптом, либо укажи правильный путь в DATA_FILE."
    )

df = pd.read_csv(data_path)
original_df = df.copy()

save_basic_info(df)
save_df(df.head(20), "01_head.csv")

country_col = detect_country_column(df)
numeric_cols = detect_numeric_columns(df)
categorical_cols = detect_categorical_columns(df)

# Если числовых признаков мало, пытаемся конвертировать потенциально числовые столбцы
if len(numeric_cols) < 5:
    for col in df.columns:
        if df[col].dtype == "object":
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() >= len(df) * 0.8:
                df[col] = converted
    numeric_cols = detect_numeric_columns(df)
    categorical_cols = detect_categorical_columns(df)

# Создаем дополнительные категории, если нужно
df = make_additional_categories(df, numeric_cols)
numeric_cols = detect_numeric_columns(df)
categorical_cols = detect_categorical_columns(df)

# Сохраняем описание признаков
feature_rows = []
for col in df.columns:
    sample_values = df[col].dropna().astype(str).head(3).tolist()
    feature_rows.append({
        "Признак": col,
        "Тип данных": str(df[col].dtype),
        "Количество уникальных": int(df[col].nunique(dropna=True)),
        "Пример значений": "; ".join(sample_values)
    })
feature_table = pd.DataFrame(feature_rows)
save_df(feature_table, "02_feature_description.csv")


# ---------------------------
# 4. ГИСТОГРАММЫ РАСПРЕДЕЛЕНИЙ (MATPLOTLIB)
# ---------------------------
for col in numeric_cols:
    plt.figure()
    plt.hist(df[col].dropna(), bins=30, edgecolor="black")
    plt.title(f"Распределение признака {col}")
    plt.xlabel(col)
    plt.ylabel("Частота")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"hist_{col}.png", dpi=200)
    plt.close()

# Итоговая сетка гистограмм
if numeric_cols:
    cols_n = 2
    rows_n = int(np.ceil(len(numeric_cols) / cols_n))
    fig, axes = plt.subplots(rows_n, cols_n, figsize=(14, 4 * rows_n))
    axes = np.array(axes).reshape(-1)

    for ax, col in zip(axes, numeric_cols):
        ax.hist(df[col].dropna(), bins=25, edgecolor="black")
        ax.set_title(col)
        ax.set_xlabel(col)
        ax.set_ylabel("Частота")

    for ax in axes[len(numeric_cols):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "hist_all_numeric.png", dpi=200)
    plt.close()


# ---------------------------
# 5. ВИЗУАЛИЗАЦИЯ ПАР ПРИЗНАКОВ (SEABORN)
# ---------------------------
selected_pairs = []
if len(numeric_cols) >= 2:
    # берем первые 3 разумные пары
    for i in range(min(3, len(numeric_cols) - 1)):
        selected_pairs.append((numeric_cols[i], numeric_cols[i + 1]))

for x_col, y_col in selected_pairs:
    plt.figure()
    if country_col and country_col in df.columns and df[country_col].nunique() <= 12:
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=country_col)
    else:
        sns.scatterplot(data=df, x=x_col, y=y_col)
    plt.title(f"Диаграмма рассеяния: {x_col} и {y_col}")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"seaborn_scatter_{x_col}_{y_col}.png", dpi=200)
    plt.close()

# boxplot для двух числовых признаков
for col in numeric_cols[:2]:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"Ящик с усами для признака {col}")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"boxplot_{col}.png", dpi=200)
    plt.close()


# ---------------------------
# 6. ИНТЕРАКТИВНЫЕ ГРАФИКИ (PLOTLY)
# ---------------------------
if len(numeric_cols) >= 2:
    fig = px.scatter(
        df,
        x=numeric_cols[0],
        y=numeric_cols[1],
        hover_name=country_col if country_col else None,
        title=f"Интерактивный scatter: {numeric_cols[0]} vs {numeric_cols[1]}"
    )
    fig.write_html(PLOTS_DIR / "plotly_scatter_1.html")

if len(numeric_cols) >= 3:
    fig = px.line(
        df.sort_values(by=numeric_cols[0]),
        x=numeric_cols[0],
        y=numeric_cols[2],
        hover_name=country_col if country_col else None,
        title=f"Интерактивный line: {numeric_cols[2]} по {numeric_cols[0]}"
    )
    fig.write_html(PLOTS_DIR / "plotly_line_1.html")


# ---------------------------
# 7. ПРОПУСКИ
# ---------------------------
missing_table = pd.DataFrame({
    "Признак": df.columns,
    "Количество пропусков": df.isna().sum().values,
    "Доля пропусков, %": (df.isna().mean().values * 100).round(2)
})
save_df(missing_table, "03_missing_values.csv")

# тепловая карта пропусков
plt.figure(figsize=(12, 6))
sns.heatmap(df.isna(), cbar=True)
plt.title("Тепловая карта пропущенных значений")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "heatmap_missing.png", dpi=200)
plt.close()

# Обработка пропусков:
# - числовые -> медиана
# - категориальные -> мода
df_filled = df.copy()
for col in numeric_cols:
    if df_filled[col].isna().sum() > 0:
        df_filled[col] = df_filled[col].fillna(df_filled[col].median())

for col in categorical_cols:
    if df_filled[col].isna().sum() > 0:
        mode_values = df_filled[col].mode(dropna=True)
        fill_value = mode_values.iloc[0] if not mode_values.empty else "unknown"
        df_filled[col] = df_filled[col].fillna(fill_value)

save_df(df_filled.head(20), "04_filled_data_preview.csv")


# ---------------------------
# 8. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ
# ---------------------------
corr = df_filled[numeric_cols].corr(numeric_only=True)
save_df(corr.reset_index(), "05_correlation_matrix.csv")

plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Корреляционная матрица числовых признаков")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "heatmap_correlation.png", dpi=200)
plt.close()

# вторая тепловая карта: корреляция только для первых 5-6 признаков, если их много
subset_cols = numeric_cols[: min(6, len(numeric_cols))]
if subset_cols:
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_filled[subset_cols].corr(), annot=True, fmt=".2f", cmap="viridis")
    plt.title("Тепловая карта корреляций (подмножество признаков)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "heatmap_correlation_subset.png", dpi=200)
    plt.close()


# ---------------------------
# 9. ДУБЛИКАТЫ
# ---------------------------
duplicates_count = int(df_filled.duplicated().sum())
df_no_duplicates, removed_duplicates = remove_duplicate_rows(df_filled)
save_text(
    "01_duplicates_info.txt",
    f"Количество дубликатов до удаления: {duplicates_count}\n"
    f"Удалено дубликатов: {removed_duplicates}\n"
    f"Размер после удаления: {df_no_duplicates.shape}"
)


# ---------------------------
# 10. ВЫБРОСЫ
# ---------------------------
outliers_table = summarize_outliers(df_no_duplicates, numeric_cols)
save_df(outliers_table, "06_outliers_summary.csv")

# Мягкая обработка выбросов методом winsorization через clip по IQR-границам
df_outlier_processed = df_no_duplicates.copy()
for col in numeric_cols:
    s = df_outlier_processed[col].dropna()
    if s.empty:
        continue
    lower, upper = outlier_bounds_iqr(s)
    df_outlier_processed[col] = df_outlier_processed[col].clip(lower=lower, upper=upper)

save_df(df_outlier_processed.head(20), "07_outlier_processed_preview.csv")


# ---------------------------
# 11. УСЛОВНАЯ ФИЛЬТРАЦИЯ (не менее 3 примеров)
# ---------------------------
filter_reports = []

# Фильтр 1: страны с высокой суммарной численностью
if len(numeric_cols) >= 2:
    total_population = df_outlier_processed[numeric_cols].sum(axis=1)
    threshold_high = total_population.quantile(0.75)
    filtered_1 = df_outlier_processed[total_population >= threshold_high].copy()
    save_df(filtered_1, "08_filter_high_population.csv")
    filter_reports.append(
        f"Фильтрация 1: отобраны объекты с суммарной численностью населения >= 75-го перцентиля. "
        f"Получено {len(filtered_1)} строк."
    )

# Фильтр 2: по одной возрастной группе
if numeric_cols:
    col = numeric_cols[0]
    threshold = df_outlier_processed[col].median()
    filtered_2 = df_outlier_processed[df_outlier_processed[col] >= threshold].copy()
    save_df(filtered_2, "09_filter_by_first_numeric.csv")
    filter_reports.append(
        f"Фильтрация 2: отобраны объекты, где {col} >= медианы ({round(threshold, 3)}). "
        f"Получено {len(filtered_2)} строк."
    )

# Фильтр 3: по двум признакам
if len(numeric_cols) >= 2:
    col1, col2 = numeric_cols[0], numeric_cols[1]
    thr1 = df_outlier_processed[col1].median()
    thr2 = df_outlier_processed[col2].median()
    filtered_3 = df_outlier_processed[
        (df_outlier_processed[col1] >= thr1) &
        (df_outlier_processed[col2] >= thr2)
    ].copy()
    save_df(filtered_3, "10_filter_by_two_numeric.csv")
    filter_reports.append(
        f"Фильтрация 3: отобраны объекты, где одновременно {col1} >= медианы и {col2} >= медианы. "
        f"Получено {len(filtered_3)} строк."
    )

save_text("02_filters_info.txt", "\n".join(filter_reports))


# ---------------------------
# 12. ДОБАВЛЕНИЕ ШУМА (в 2 признака)
# ---------------------------
noise_columns = numeric_cols[:2]
df_noisy = add_noise(df_outlier_processed, noise_columns, noise_level=0.03, random_state=42)
save_df(df_noisy.head(20), "11_noisy_data_preview.csv")

for col in noise_columns:
    noisy_col = f"{col}_noisy"
    if noisy_col in df_noisy.columns:
        plt.figure()
        plt.hist(df_noisy[col].dropna(), bins=25, alpha=0.6, label=col, edgecolor="black")
        plt.hist(df_noisy[noisy_col].dropna(), bins=25, alpha=0.6, label=noisy_col, edgecolor="black")
        plt.title(f"Сравнение распределений: {col} и {noisy_col}")
        plt.xlabel("Значение")
        plt.ylabel("Частота")
        plt.legend()
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"noise_compare_{col}.png", dpi=200)
        plt.close()


# ---------------------------
# 13. ПРЕОБРАЗОВАНИЕ ЧИСЛОВЫХ ДАННЫХ В КАТЕГОРИАЛЬНЫЕ
# ---------------------------
df_transformed = df_noisy.copy()

if len(numeric_cols) >= 1:
    first_num = numeric_cols[0]
    try:
        df_transformed[f"{first_num}_band"] = pd.qcut(
            df_transformed[first_num],
            q=4,
            labels=["низкий", "средний-низкий", "средний-высокий", "высокий"],
            duplicates="drop"
        )
    except Exception:
        df_transformed[f"{first_num}_band"] = "не определено"

save_df(df_transformed.head(20), "12_transformed_numeric_to_category.csv")


# ---------------------------
# 14. КАТЕГОРИАЛЬНЫЕ ПРИЗНАКИ
# ---------------------------
categorical_cols = detect_categorical_columns(df_transformed)

# список категорий
category_rows = []
for col in categorical_cols:
    values = df_transformed[col].astype(str).value_counts(dropna=False)
    listed = values.index.tolist()[:30]  # чтобы не раздувать
    category_rows.append({
        "Признак": col,
        "Количество категорий": int(df_transformed[col].nunique(dropna=True)),
        "Категории (первые 30)": ", ".join(map(str, listed))
    })
save_df(pd.DataFrame(category_rows), "13_categories_list.csv")

# диаграммы распределения категориальных данных
for col in categorical_cols[:5]:
    plt.figure(figsize=(12, 6))
    order = df_transformed[col].astype(str).value_counts().index[:15]
    sns.countplot(data=df_transformed, y=df_transformed[col].astype(str), order=order)
    plt.title(f"Распределение категориального признака {col}")
    plt.xlabel("Количество")
    plt.ylabel(col)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"countplot_{col}.png", dpi=200)
    plt.close()

# кодирование категориальных данных
df_label_encoded = safe_label_encode(df_transformed, categorical_cols)
save_df(df_label_encoded.head(20), "14_label_encoded_preview.csv")

df_one_hot = one_hot_encode_small_categories(df_transformed, categorical_cols, max_unique=15)
save_df(df_one_hot.head(20), "15_one_hot_preview.csv")

# агрегация редких категорий
aggregation_text = []
for col in categorical_cols:
    freq = df_transformed[col].astype(str).value_counts()
    rare_categories = freq[freq < max(2, int(0.03 * len(df_transformed)))].index.tolist()
    if rare_categories:
        aggregation_text.append(
            f"В признаке {col} редкими считаются категории: {', '.join(map(str, rare_categories[:20]))}"
        )

if not aggregation_text:
    aggregation_text.append("Редкие категории в исследуемых признаках выражены слабо или отсутствуют.")
save_text("03_rare_categories.txt", "\n".join(aggregation_text))


# ---------------------------
# 15. ВВЕДЕНИЕ НОВОЙ КАТЕГОРИИ
# ---------------------------
df_final = df_transformed.copy()

if "population_band" in df_final.columns and "dominant_age_group" in df_final.columns:
    df_final["demographic_profile"] = (
        df_final["population_band"].astype(str) + " / " +
        df_final["dominant_age_group"].astype(str)
    )
else:
    df_final["demographic_profile"] = "базовый профиль"

save_df(df_final.head(20), "16_new_category_preview.csv")


# ---------------------------
# 16. ПОВТОРНАЯ ВИЗУАЛИЗАЦИЯ ПОСЛЕ ТРАНСФОРМАЦИЙ
# ---------------------------
for col in numeric_cols[:4]:
    plt.figure()
    plt.hist(df_final[col].dropna(), bins=25, edgecolor="black")
    plt.title(f"Распределение после преобразований: {col}")
    plt.xlabel(col)
    plt.ylabel("Частота")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"after_transform_hist_{col}.png", dpi=200)
    plt.close()


# ---------------------------
# 17. ИТОГОВЫЕ ТАБЛИЦЫ И ВЫВОДЫ
# ---------------------------
summary_rows = [
    {"Показатель": "Количество строк (исходно)", "Значение": len(original_df)},
    {"Показатель": "Количество столбцов (исходно)", "Значение": original_df.shape[1]},
    {"Показатель": "Количество строк (после обработки)", "Значение": len(df_final)},
    {"Показатель": "Количество числовых признаков", "Значение": len(numeric_cols)},
    {"Показатель": "Количество категориальных признаков", "Значение": len(categorical_cols)},
    {"Показатель": "Удалено дубликатов", "Значение": removed_duplicates},
    {"Показатель": "Всего пропусков (исходно)", "Значение": int(original_df.isna().sum().sum())},
]
summary_df = pd.DataFrame(summary_rows)
save_df(summary_df, "17_summary_table.csv")

text_conclusion = f"""
Краткие выводы по первичному анализу:

1. Исходный набор данных содержит {original_df.shape[0]} строк и {original_df.shape[1]} столбцов.
2. После автоматического определения признаков выявлено:
   - числовых признаков: {len(numeric_cols)};
   - категориальных признаков: {len(categorical_cols)}.
3. Общее количество пропусков в исходных данных: {int(original_df.isna().sum().sum())}.
4. Количество удаленных дубликатов: {removed_duplicates}.
5. Для числовых признаков построены гистограммы распределения, boxplot и тепловые карты.
6. Для категориальных признаков выполнен анализ категорий и кодирование.
7. Для повышения устойчивости данных добавлен небольшой шум в 2 числовых признака.
8. Выполнены три примера фильтрации, а также преобразование части числовых признаков в категориальные.

Общий вывод:
Набор данных пригоден для решения задачи анализа и прогнозирования возрастной структуры населения.
Перед обучением модели рекомендуется:
- использовать данные после заполнения пропусков;
- контролировать влияние выбросов;
- применять масштабирование числовых признаков;
- использовать кодирование категориальных переменных.
"""
save_text("04_final_conclusion.txt", text_conclusion.strip())

# технический отчет
report = {
    "data_file": str(data_path),
    "shape_before": original_df.shape,
    "shape_after": df_final.shape,
    "country_column": country_col,
    "numeric_columns": numeric_cols,
    "categorical_columns": categorical_cols,
    "output_dir": str(OUTPUT_DIR)
}
save_text("05_report.json", json.dumps(report, ensure_ascii=False, indent=2))

print("Готово.")
print(f"Результаты сохранены в папку: {OUTPUT_DIR.resolve()}")
print("Подпапки:")
print(f"- графики: {PLOTS_DIR.resolve()}")
print(f"- таблицы: {TABLES_DIR.resolve()}")
print(f"- текстовые выводы: {TEXT_DIR.resolve()}")
