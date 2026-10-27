# Прогнозирование 30-дневной летальности при сепсисе (MIMIC-IV, временная валидация)

Код к статье «Прогнозирование 30-дневной летальности при сепсисе: разработка и временная валидация модели
градиентного бустинга на основе данных MIMIC-IV».

*English summary.* Analysis code for a LightGBM model predicting 30-day mortality in adult ICU patients with
Sepsis-3 (MIMIC-IV v3.1), with temporal validation (train ≤ 2019, test ≥ 2020). The repository holds the BigQuery
SQL for the primary cohort and the Python pipeline: cohort selection, feature extraction up to the index time T0,
model training, a one-time test evaluation guarded by checksums, subgroup analysis, tables and figures.
No data, predictions or trained models are included.

## Что есть и чего нет в репозитории

Есть:
- SQL для Google BigQuery: первичная когорта всех эпизодов Sepsis-3 и выгрузка нужных производных таблиц mimic-code;
- код всех этапов: отбор когорты, формирование признаков, обучение моделей, оценка на тестовой выборке,
  анализ подгрупп, таблицы и рисунки статьи.

Нет:
- данных MIMIC-IV и любых файлов, полученных из них (промежуточные таблицы, прогнозы, результаты);
- обученных моделей. Модели, обученные на MIMIC-IV, не распространяются; их можно воспроизвести этим кодом.

Для работы нужен собственный доступ к MIMIC-IV v3.1 на PhysioNet (обучение CITI и подписанное
соглашение об использовании данных — PhysioNet Credentialed Health Data Use Agreement).

## Подготовка данных

Все исходные файлы загружаются в одну папку (`SEPSIS_RAW_DIR`) в формате `*.csv.gz` или `*.csv`.

1. **Первичная когорта (BigQuery).** Выполните запрос `sql/01_sepsis3_episodes.sql` и сохраните результат как
   `sepsis3_new.csv.gz`. Запрос повторяет логику mimic-code (`concepts/sepsis/sepsis3_new.sql`). Этап 0 воспроизводит
   ту же логику по исходным таблицам MIMIC IV и сверяет результат с этим файлом.
2. **Производные таблицы mimic-code (BigQuery).** Выгрузите таблицы, перечисленные в `sql/02_export_derived_tables.sql` под
   указанными именами: `antibiotic`, `sofa`, `charlson`, `ventilation`, `ventilator_setting`, `rrt`,
   `urine_output_rate`, `vitals_derived` (таблица `vitalsign`), `gcs`, `sapsii`, `oasis`.
3. **Исходные таблицы MIMIC-IV v3.1 (PhysioNet).**
   - модуль `hosp`: `patients`, `admissions`, `diagnoses_icd`, `labevents`, `d_labitems`, `microbiologyevents`, `omr`;
   - модуль `icu`: `icustays`, `inputevents`, `chartevents`.

## Настройка

Пути задаются переменными окружения:

| Переменная | Назначение | По умолчанию |
|---|---|---|
| `SEPSIS_RAW_DIR` | исходные таблицы (п. 1–3 выше) | `/content/drive/MyDrive` |
| `SEPSIS_WORK_DIR` | промежуточные данные, модели, результаты, журналы | `$SEPSIS_RAW_DIR/sepsis_v3` |

Значения по умолчанию рассчитаны на Google Colab с подключённым Google Drive.

```bash
pip install -r requirements.txt
export SEPSIS_RAW_DIR=/path/to/mimic_tables
export SEPSIS_WORK_DIR=/path/to/workdir
python src/stage0_sepsis3.py
```

В Colab:

```python
from google.colab import drive; drive.mount('/content/drive')
# скопируйте репозиторий в /content/sepsis-mimic4-temporal (git clone или Code → Download ZIP)
!pip -q install duckdb lightgbm optuna
!python /content/sepsis-mimic4-temporal/src/stage0_sepsis3.py
```

## Порядок запуска

Этапы запускаются строго по порядку; каждый читает только исходные таблицы и выходы предыдущих этапов.

| Этап | Файл | Что делает | В статье |
|---|---|---|---|
| 0 | `stage0_sepsis3.py` | воспроизводит Sepsis-3 mimic-code и сверяет с `sepsis3_new.csv.gz` (100% совпадение эпизодов и времён, иначе остановка); применяет уточнения определения: скрининговые посевы, несистемные антибиотики, периоперационный цефазолин | Методы: определение сепсиса |
| 1 | `stage1_cohort.py` | T0, первый эпизод пациента, исход (смерть в течение 30 сут), разделение на обучающий (≤ 2019) и тестовый (≥ 2020) периоды, доступность данных об инфузиях | Методы: когорта; схема отбора |
| 2 | `stage2_baseline.py` | масса тела, рост, код-статус (только данные до T0); индекс Чарлсон и COVID-19 по диагнозам госпитализации — только для описания когорты и подгрупп, в модель не входят | Таблица 1 |
| 3 | `stage3_support.py` | дыхательная поддержка, вазопрессоры (норадреналиновый эквивалент), седация, ЗПТ, диурез | Признаки |
| 4 | `stage4_labs.py` | лабораторные показатели и газы крови за 24 ч до T0; результат должен быть готов к T0 | Признаки |
| 5 | `stage5_vitals.py` | витальные показатели, ШКГ, FiO₂ за 24 ч до T0 | Признаки |
| 6 | `stage6_master.py` | итоговая таблица, исключение пациентов с ограничением объёма лечения, ограничение значений клиническими границами, перечень признаков | Методы; схема отбора |
| 6б | `stage6b_benchmarks.py` | SAPS II и OASIS из mimic-code | Сравнение со шкалами |
| 7 | `stage7_models.py` | обучение на обучающем периоде: LightGBM + Optuna, логистическая регрессия со сплайнами, сокращённая модель, перекалибровка SAPS II и OASIS; фиксация моделей и SHA-256 | Методы: модели |
| 8 | `stage8_evaluate.py` | однократная оценка на тестовом периоде: AUROC/AUPRC с бутстреп-ДИ, DeLong с поправкой Холма, калибровка, кривая решений, SHAP | Таблица метрик; сравнение моделей |
| 8б | `stage8b_shap_beeswarm.py` | сводная диаграмма SHAP на тестовой выборке | Рисунок SHAP |
| 8в | `stage8c_figures.py` | калибровочные кривые и кривая принятия решений | `fig2_calibration.png`, `fig3_dca.png` |
| 9 | `stage9_subgroups.py` | AUROC и калибровка в подгруппах | Анализ подгрупп |
| 9б | `stage9b_subgroup_pvalues.py` | сравнение AUROC между подгруппами (DeLong, поправка Холма на 12 сравнений) | Анализ подгрупп |
| 10 | `stage10_tables.py` | Таблица 1 (p: Манна–Уитни / χ²), схема отбора, сводная книга Excel | Таблица 1; схема отбора |

Вспомогательные модули: `common.py` (пути, журналы, контрольные проверки), `modeling.py` (подготовка признаков),
`evaluation.py` (метрики, DeLong, бутстреп, калибровка, кривая решений).

Подробный перечень признаков — `docs/features.md`.

## Защита тестовой выборки

- Этап 7 читает только обучающий период и записывает SHA-256 всех файлов моделей в `models/stage7_checksums.json`.
- Этап 8 перед оценкой сверяет контрольные суммы и при первом запуске создаёт `models/TEST_OPENED.json`
  (время открытия и сводная контрольная сумма). Повторный запуск разрешён, только если модели не менялись,
  и даёт те же результаты.
- Этапы 8б, 9 и 9б работают с уже зафиксированными моделями; этап 9 проверяет, что тест уже открыт и что
  прогнозы совпадают с этапом 8.

## Воспроизводимость

- Единый seed проекта (`SEED` в `common.py`) для фолдов, Optuna и бутстрепа.
- Каждый этап пишет полный вывод в `run_logs/`, метрики и результаты контрольных проверок в `logs/*.json`,
  включая версии библиотек. При невыполнении контрольной проверки этап останавливается с ошибкой.
- Этапы 0–6 дополнительно формируют стандартную когорту Sepsis-3 (без уточнений определения) и рассчитанные
  индексы (`idx_*`). В модели статьи они не используются и остаются только для сверки и описания.

## Цитирование

Статья направлена в печать; библиографическая ссылка будет добавлена после публикации.
До этого при использовании кода, пожалуйста, ссылайтесь на этот репозиторий.

Данные: Johnson A., Bulgarelli L., Pollard T. et al. MIMIC-IV (version 3.1). PhysioNet. https://doi.org/10.13026/kpb9-mt58

Определение Sepsis-3 и производные таблицы: MIT-LCP/mimic-code, https://github.com/MIT-LCP/mimic-code

## Лицензия

MIT (см. `LICENSE`). Лицензия распространяется только на код; использование данных MIMIC-IV регулируется
соглашением PhysioNet.
