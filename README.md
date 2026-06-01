# Поиск и первичный анализ наборов данных

**Курсовой проект** по дисциплине «Введение в проектную деятельность»  
ЮУрГУ, направление 02.03.02 «Прикладная математика и информатика»  
Центр «ВиртУм»

**Автор:** Колобов Егор Евгеньевич, группа ЕТ-113  
**Руководитель:** В.А. Сурин, доцент  
**Год:** 2026

---

## Структура проекта

| Файл | Описание | Набор данных |
|------|----------|-------------|
| `chapter1_tabular_data.py` | Глава 1. Первичный анализ табличных данных | World Population by Age Group 2020 ([Kaggle](https://www.kaggle.com/datasets/alizahidraja/world-population-by-age-group-2020)) |
| `chapter2_time_series.py` | Глава 2. Первичный анализ временных рядов | CWRU MAT Full Dataset ([Kaggle](https://www.kaggle.com/datasets/sufian79/cwru-mat-full-dataset)) |
| `chapter3_images_gender.py` | Глава 3. Первичный анализ данных с изображениями | UTKFace Cropped ([Kaggle](https://www.kaggle.com/datasets/moritzm00/utkface-cropped)) |
| `chapter4_text_spam.py` | Глава 4. Первичный анализ текстовых данных | Email Spam Classification Dataset CSV ([Kaggle](https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv)) |

---

## Что делают скрипты

Каждый скрипт выполняет **полный цикл первичного анализа данных** для своего типа:

- Загрузка и первичное знакомство с данными
- Визуализация (гистограммы, диаграммы рассеяния, тепловые карты)
- Статистический анализ (описательная статистика, распределения)
- Анализ пропусков и выбросов
- Корреляционный анализ
- Специфические процедуры для каждого типа данных

---

## Зависимости

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn
pip install plotly                        # Глава 1
pip install Pillow                        # Глава 3
```

---

## Запуск

```bash
python chapter1_tabular_data.py
python chapter2_time_series.py
python chapter3_images_gender.py
python chapter4_text_spam.py
```

> **Примечание:** перед запуском скачайте соответствующий набор данных и укажите путь к нему в настройках нужного скрипта.
