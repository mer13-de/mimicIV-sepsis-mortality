-- =====================================================================================
-- Производные таблицы mimic-code (MIMIC-IV v3.1, Google BigQuery), необходимые для работы.
-- Каждую таблицу выгрузите целиком (например, EXPORT DATA в Cloud Storage или «Save results»)
-- и сохраните в SEPSIS_RAW_DIR под указанным именем файла (*.csv.gz или *.csv).
-- =====================================================================================

-- antibiotic.csv.gz          (этап 0)
SELECT * FROM `physionet-data.mimiciv_3_1_derived.antibiotic`;
-- sofa.csv.gz                (этапы 0, 6)
SELECT * FROM `physionet-data.mimiciv_3_1_derived.sofa`;
-- charlson.csv.gz            (этап 2; только для описания когорты)
SELECT * FROM `physionet-data.mimiciv_3_1_derived.charlson`;
-- ventilation.csv.gz         (этап 3)
SELECT * FROM `physionet-data.mimiciv_3_1_derived.ventilation`;
-- ventilator_setting.csv.gz  (этапы 3, 4, 5)
SELECT * FROM `physionet-data.mimiciv_3_1_derived.ventilator_setting`;
-- rrt.csv.gz                 (этап 3)
SELECT * FROM `physionet-data.mimiciv_3_1_derived.rrt`;
-- urine_output_rate.csv.gz   (этап 3)
SELECT * FROM `physionet-data.mimiciv_3_1_derived.urine_output_rate`;
-- vitals_derived.csv.gz      (этап 5) — таблица vitalsign
SELECT * FROM `physionet-data.mimiciv_3_1_derived.vitalsign`;
-- gcs.csv.gz                 (этап 5)
SELECT * FROM `physionet-data.mimiciv_3_1_derived.gcs`;
-- sapsii.csv.gz              (этап 6б)
SELECT * FROM `physionet-data.mimiciv_3_1_derived.sapsii`;
-- oasis.csv.gz               (этап 6б)
SELECT * FROM `physionet-data.mimiciv_3_1_derived.oasis`;
