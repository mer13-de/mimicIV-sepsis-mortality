-- =====================================================================================
-- Первичная когорта: все эпизоды сепсиса по Sepsis-3 в MIMIC-IV v3.1 (Google BigQuery)
--
-- Логика mimic-code (concepts/sepsis/sepsis3.sql): подозрение на инфекцию
-- (suspicion_of_infection) и SOFA >= 2 в окне [-48 ч; +24 ч] от времени подозрения;
-- для каждого пребывания в ОРИТ берётся первый эпизод (rn_sus = 1).
--
-- Результат сохраните как sepsis3_new.csv.gz в папку SEPSIS_RAW_DIR.
-- Этап 0 (src/stage0_sepsis3.py) воспроизводит эту же логику по исходным таблицам и
-- останавливается, если набор эпизодов и все четыре времени не совпадают с этим файлом на 100%.
--
-- Примечание: фильтр WHERE sepsis3 не нужен. Таблица sofa заранее ограничена SOFA >= 2, а строки
-- без подозрения на инфекцию (suspected_infection_time IS NULL) отпадают при соединении по времени,
-- поэтому во всех выходных строках sepsis3 = TRUE.
-- =====================================================================================
WITH sofa AS (
    SELECT stay_id
         , starttime, endtime
         , respiration_24hours    AS respiration
         , coagulation_24hours    AS coagulation
         , liver_24hours          AS liver
         , cardiovascular_24hours AS cardiovascular
         , cns_24hours            AS cns
         , renal_24hours          AS renal
         , sofa_24hours           AS sofa_score
    FROM `physionet-data.mimiciv_3_1_derived.sofa`
    WHERE sofa_24hours >= 2
)
, s1 AS (
    SELECT
          soi.subject_id
        , soi.stay_id
        -- suspicion columns
        , soi.ab_id
        , soi.antibiotic
        , soi.antibiotic_time
        , soi.culture_time
        , soi.suspected_infection
        , soi.suspected_infection_time
        , soi.specimen
        , soi.positive_culture
        -- sofa columns
        , starttime, endtime
        , respiration, coagulation, liver, cardiovascular, cns, renal
        , sofa_score
        , sofa_score >= 2 AND suspected_infection = 1 AS sepsis3
        , ROW_NUMBER() OVER (
              PARTITION BY soi.stay_id
              ORDER BY suspected_infection_time, antibiotic_time, culture_time, endtime
          ) AS rn_sus
    FROM `physionet-data.mimiciv_3_1_derived.suspicion_of_infection` AS soi
    INNER JOIN sofa
        ON soi.stay_id = sofa.stay_id
       AND sofa.endtime >= DATETIME_SUB(soi.suspected_infection_time, INTERVAL '48' HOUR)
       AND sofa.endtime <= DATETIME_ADD(soi.suspected_infection_time, INTERVAL '24' HOUR)
    -- only include in-ICU rows
    WHERE soi.stay_id IS NOT NULL
)
SELECT
      subject_id, stay_id
    -- note: there may be more than one antibiotic given at this time
    , antibiotic_time
    -- culture times may be dates, rather than times
    , culture_time
    , suspected_infection_time
    -- endtime is latest time at which the SOFA score is valid
    , endtime AS sofa_time
    , sofa_score
    , respiration, coagulation, liver, cardiovascular, cns, renal
    , sepsis3
FROM s1
WHERE rn_sus = 1;
