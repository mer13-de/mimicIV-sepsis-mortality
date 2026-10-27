# -*- coding: utf-8 -*-
"""
ЭТАП 1. Когорта, индексное время, исход, расчётный год, доступность данных об инфузиях.

Для основного (с уточнениями) и стандартного определения Sepsis-3:
  1. T0 = max(sofa_time, culture_time, antibiotic_time);
  2. первый эпизод пациента по T0; если он не проходит шаги 3–4, пациент исключается целиком;
  3. T0 внутри ICU-пребывания;
  4. смерть не ранее T0;
  5. исход — смерть в течение 30 суток после T0 (внутрибольничная — по deathtime, после выписки — по dod);
  6. период — интервальная классификация. Реальный год эпизода = реальный опорный год + Δ,
     где Δ = год T0 − anchor_year, а реальный опорный год — любой из трёх лет anchor_year_group.
     Возможные годы: [начало группы + Δ; конец группы + Δ]. Обучение — если верхняя граница ≤ 2019,
     тест — если нижняя ≥ 2020, иначе «ambiguous» (исключается из основного анализа).
     Эквивалентно: расчётный год (середина) ≤ 2018 — обучение, ≥ 2021 — тест, 2019–2020 — неоднозначно;
  7. данные об инфузиях доступны, если хотя бы одна запись inputevents перекрывает окно
     [max(intime, T0 − 24 ч); T0] (записи после T0 не используются);
  8. main_analysis = данные об инфузиях доступны и период однозначен. Исключение по ограничению терапии
     (DNR/DNI/CMO на T0) применяется на этапе 6, где доступен code_status_t0 (этап 2); там же строится
     итоговая схема отбора (stage6_flowchart.json).

Вход: data/sepsis3_modified.parquet, sepsis3_standard.parquet; icustays, patients, admissions, inputevents
Выход: data/stage1_cohort.parquet (основная), stage1_cohort_standard.parquet, stage1_flowchart.json
Контрольные проверки выполняются после сохранения файлов (при ошибке файлы остаются для разбора).
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import duckdb

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:                       # код вставлен прямо в ячейку Colab
    _HERE = os.getcwd()
sys.path.insert(0, _HERE)
from common import Stage, DATA, raw, read_csv_any, is_gzip

TRAIN_MAX_YEAR = 2019
WINDOW_H = 24
INPUT_SCHEMA = {'subject_id': 'BIGINT', 'hadm_id': 'BIGINT', 'stay_id': 'BIGINT', 'caregiver_id': 'BIGINT',
                'starttime': 'TIMESTAMP', 'endtime': 'TIMESTAMP', 'storetime': 'TIMESTAMP', 'itemid': 'BIGINT',
                'amount': 'DOUBLE', 'amountuom': 'VARCHAR', 'rate': 'DOUBLE', 'rateuom': 'VARCHAR',
                'orderid': 'BIGINT', 'linkorderid': 'BIGINT', 'ordercategoryname': 'VARCHAR',
                'secondaryordercategoryname': 'VARCHAR', 'ordercomponenttypedescription': 'VARCHAR',
                'ordercategorydescription': 'VARCHAR', 'patientweight': 'DOUBLE', 'totalamount': 'DOUBLE',
                'totalamountuom': 'VARCHAR', 'isopenbag': 'BIGINT', 'continueinnextdept': 'BIGINT',
                'statusdescription': 'VARCHAR', 'originalamount': 'DOUBLE', 'originalrate': 'DOUBLE'}
hours = lambda td: td.dt.total_seconds() / 3600


def load_reference_tables():
    icu = read_csv_any(raw('icustays'), usecols=['subject_id', 'hadm_id', 'stay_id', 'first_careunit', 'intime', 'outtime'])
    pat = read_csv_any(raw('patients'), usecols=['subject_id', 'gender', 'anchor_age', 'anchor_year', 'anchor_year_group', 'dod'])
    adm = read_csv_any(raw('admissions'), usecols=['hadm_id', 'admittime', 'dischtime', 'deathtime', 'admission_type',
                                                   'admission_location', 'discharge_location', 'insurance', 'race',
                                                   'hospital_expire_flag'])
    for df, cols in [(icu, ['intime', 'outtime']), (pat, ['dod']), (adm, ['admittime', 'dischtime', 'deathtime'])]:
        for c in cols:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    return icu, pat, adm


def build_cohort(sep, icu, pat, adm):
    """Шаги 1–7 для одной таблицы эпизодов. Возвращает (когорта, схема отбора)."""
    flow = []
    step = lambda name, d: flow.append({'step': name, 'patients': int(d['subject_id'].nunique()), 'rows': int(len(d))})
    d = sep.copy()
    step('ICU-эпизоды сепсиса', d)
    d = d.merge(icu.drop(columns='subject_id'), on='stay_id', how='inner', validate='1:1')
    d['t0'] = d[['sofa_time', 'culture_time', 'antibiotic_time']].max(axis=1)
    d = d.sort_values(['subject_id', 't0', 'stay_id']).drop_duplicates('subject_id', keep='first')
    step('Первый эпизод пациента', d)
    d = d[(d['t0'] >= d['intime']) & (d['t0'] <= d['outtime'])]
    step('T0 внутри ICU-пребывания', d)
    d = d.merge(pat, on='subject_id', how='left', validate='1:1').merge(adm, on='hadm_id', how='left', validate='1:1')
    d['age'] = (d['anchor_age'] + (d['t0'].dt.year - d['anchor_year'])).clip(upper=91)
    if not (d['age'] >= 18).all():
        raise RuntimeError(f"В когорте есть пациенты < 18 лет: {int((d['age'] < 18).sum())}")
    died_before = (d['deathtime'].notna() & (d['deathtime'] < d['t0'])) | \
                  (d['dod'].notna() & (d['dod'].dt.normalize() < d['t0'].dt.normalize()))
    d = d[~died_before].copy()
    step('Смерть не ранее T0', d)

    # Исход: внутрибольничная смерть — точное время; после выписки — календарные дни
    exact_days = (d['deathtime'] - d['t0']).dt.total_seconds() / 86400
    date_days = (d['dod'].dt.normalize() - d['t0'].dt.normalize()).dt.days
    d['days_to_death'] = np.where(d['deathtime'].notna(), exact_days, date_days)
    d['death_30d'] = (pd.Series(d['days_to_death'], index=d.index) <= 30).astype(int)

    # Период: интервальная классификация (см. шаг 6 в описании этапа)
    g_start = d['anchor_year_group'].str[:4].astype(int)
    g_end = d['anchor_year_group'].str[-4:].astype(int)
    delta = d['t0'].dt.year - d['anchor_year']
    d['year_low'] = g_start + delta
    d['year_high'] = g_end + delta
    d['est_year'] = g_start + 1 + delta          # середина интервала — только для описания и проверок по годам
    d['split'] = np.select([d['year_high'] <= TRAIN_MAX_YEAR, d['year_low'] >= TRAIN_MAX_YEAR + 1],
                           ['train', 'test'], default='ambiguous')

    # Описательные интервалы (не предикторы)
    d['t0_shift_h'] = hours(d['t0'] - d['sofa_time'])
    d['icu_to_t0_h'] = hours(d['t0'] - d['intime'])
    d['hosp_to_t0_h'] = hours(d['t0'] - d['admittime'])
    d['sex_male'] = (d['gender'] == 'M').astype(int)
    return d.reset_index(drop=True), flow


def infusion_availability(cohorts):
    """Флаг: хотя бы одна запись inputevents перекрывает окно [max(intime, T0 − 24 ч); T0]."""
    keys = pd.concat([c[['stay_id', 't0', 'intime']] for c in cohorts]).drop_duplicates('stay_id')
    keys['w0'] = keys[['intime']].assign(x=keys['t0'] - pd.Timedelta(hours=WINDOW_H)).max(axis=1)
    p = raw('inputevents')
    cols = ', '.join(f"'{k}': '{v}'" for k, v in INPUT_SCHEMA.items())
    src = (f"read_csv('{p}', header=true, compression='{'gzip' if is_gzip(p) else 'none'}', "
           f"columns={{{cols}}}, quote='\"', escape='\"')")
    con = duckdb.connect()
    con.register('k', keys[['stay_id', 't0', 'w0']])
    res = con.execute(f"""SELECT k.stay_id, COUNT(*) AS n_input_window
                          FROM {src} e JOIN k ON e.stay_id = k.stay_id
                          WHERE e.starttime <= k.t0 AND coalesce(e.endtime, e.starttime) >= k.w0
                          GROUP BY 1""").df()
    return res.set_index('stay_id')['n_input_window']


def summary(d, st, prefix):
    by = d.groupby(['split', 'has_infusion_data'])['death_30d'].agg(n='size', deaths='sum', mortality='mean')
    by['mortality'] = (by['mortality'] * 100).round(1)
    print(by.to_string())
    st.log(f'{prefix}_by_split_infusion', {f'{s}|{int(h)}': {'n': int(r.n), 'deaths': int(r.deaths),
                                                               'mortality_pct': float(r.mortality)}
                                           for (s, h), r in by.iterrows()})


with Stage('stage1_cohort') as st:
    st.section('1. Исходные таблицы')
    icu, pat, adm = load_reference_tables()
    sep_mod = pd.read_parquet(f'{DATA}/sepsis3_modified.parquet')
    sep_std = pd.read_parquet(f'{DATA}/sepsis3_standard.parquet')
    st.log('episodes_modified', int(len(sep_mod)))
    st.log('episodes_standard', int(len(sep_std)))

    st.section('2. Когорты (шаги 1–7)')
    cohort, flow = build_cohort(sep_mod, icu, pat, adm)
    cohort_std, flow_std = build_cohort(sep_std, icu, pat, adm)
    for f in flow:
        print(f"  {f['step']:<32} пациентов: {f['patients']:>6,}")
    st.log('final_patients_modified', int(len(cohort)))
    st.log('final_patients_standard', int(len(cohort_std)))
    st.log('mortality_30d_pct_modified', round(float(cohort['death_30d'].mean() * 100), 2))
    st.log('mortality_30d_pct_standard', round(float(cohort_std['death_30d'].mean() * 100), 2))
    st.log('t0_shift_h_median_iqr', [round(float(x), 1) for x in cohort['t0_shift_h'].quantile([.5, .25, .75])])
    st.log('icu_to_t0_h_median_iqr', [round(float(x), 1) for x in cohort['icu_to_t0_h'].quantile([.5, .25, .75])])

    st.section('3. Доступность данных об инфузиях (inputevents в окне до T0)')
    n_in = infusion_availability([cohort, cohort_std])
    for d in (cohort, cohort_std):
        d['n_input_window'] = d['stay_id'].map(n_in).fillna(0).astype(int)
        d['has_infusion_data'] = (d['n_input_window'] > 0).astype(int)
    by_year = cohort.groupby('est_year').agg(n=('stay_id', 'size'), pct_infusion=('has_infusion_data', 'mean'),
                                             mortality=('death_30d', 'mean'))
    by_year[['pct_infusion', 'mortality']] = (by_year[['pct_infusion', 'mortality']] * 100).round(1)
    print(by_year.to_string())
    st.log('infusion_by_est_year', {int(y): {'n': int(r.n), 'pct_infusion': float(r.pct_infusion)}
                                    for y, r in by_year.iterrows()})
    tr = cohort[cohort['split'] == 'train']
    st.log('train_infusion_pct', round(float(tr['has_infusion_data'].mean() * 100), 2))
    # у кого нет данных в обучающем периоде: T0 вскоре после поступления?
    print('Обучающий период без данных об инфузиях — ICU→T0, ч (медиана):',
          round(float(tr.loc[tr['has_infusion_data'] == 0, 'icu_to_t0_h'].median()), 1))

    st.section('4. Периоды и основной анализ')
    summary(cohort, st, 'modified')
    amb = cohort.groupby(['split', 'est_year']).size().unstack(0, fill_value=0)
    print('Период по расчётному году (середина интервала):\n' + amb.to_string())
    st.log('split_counts', cohort['split'].value_counts().to_dict())
    st.log('ambiguous_with_infusion', int(((cohort['split'] == 'ambiguous') & (cohort['has_infusion_data'] == 1)).sum()))
    for d in (cohort, cohort_std):
        d['main_analysis'] = ((d['has_infusion_data'] == 1) & (d['split'] != 'ambiguous')).astype(int)
    n_train = int(((cohort['split'] == 'train') & (cohort['main_analysis'] == 1)).sum())
    n_test = int(((cohort['split'] == 'test') & (cohort['main_analysis'] == 1)).sum())
    d_test = int(cohort.loc[(cohort['split'] == 'test') & (cohort['main_analysis'] == 1), 'death_30d'].sum())
    st.log('main_train_n_before_limited_care', n_train)
    st.log('main_test_n_before_limited_care', n_test)
    st.log('main_test_deaths_before_limited_care', d_test)
    print(f'До исключения по ограничению терапии (этап 6): обучение {n_train:,}, тест {n_test:,} (смертей {d_test:,})')

    st.section('5. Сохранение и контрольные проверки')
    cohort.to_parquet(f'{DATA}/stage1_cohort.parquet', index=False)
    cohort_std.to_parquet(f'{DATA}/stage1_cohort_standard.parquet', index=False)
    with open(f'{DATA}/stage1_flowchart.json', 'w', encoding='utf-8') as f:
        json.dump({'modified': flow, 'standard': flow_std}, f, ensure_ascii=False, indent=2)
    print(f'Сохранено: {DATA}/stage1_cohort.parquet, stage1_cohort_standard.parquet, stage1_flowchart.json')

    st.check(cohort['subject_id'].is_unique and cohort['stay_id'].is_unique, 'одна строка на пациента')
    st.check(cohort.loc[cohort['split'] == 'train', 'year_high'].max() <= TRAIN_MAX_YEAR and
             cohort.loc[cohort['split'] == 'test', 'year_low'].min() >= TRAIN_MAX_YEAR + 1,
             'обучающий и тестовый периоды не пересекаются ни при каком реальном опорном годе')
    st.check(26_000 <= len(cohort) <= 29_000, 'размер основной когорты в ожидаемом диапазоне 26–29 тыс.')
    # Решение 13 (26.09.2026, до моделирования): порог проверяется по обучающему периоду в целом;
    # по годам — не ниже 94% (2019 г. — 94,9%: из-за неточности расчётного года часть пребываний
    # этого года может приходиться на период после обрыва данных и корректно исключается правилом).
    st.check(tr['has_infusion_data'].mean() >= 0.95, 'данные об инфузиях доступны у ≥ 95% пациентов обучающего периода')
    big_years = by_year[(by_year.index <= TRAIN_MAX_YEAR) & (by_year['n'] >= 200)]
    st.check((big_years['pct_infusion'] >= 94).all(),
             'в каждом расчётном году ≤ 2019 (n ≥ 200) данные об инфузиях доступны у ≥ 94% пациентов')
    st.check(n_test >= 1500, 'в тестовом периоде основного анализа ≥ 1 500 пациентов')
