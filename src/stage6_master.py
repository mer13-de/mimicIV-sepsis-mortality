# -*- coding: utf-8 -*-
"""
ЭТАП 6. SOFA-признаки, объединение, второй уровень очистки, рассчитанные индексы, перечень признаков.

  * SOFA — производная таблица sofa (почасовые записи со скользящим 24-часовым окном). На T0 — последняя запись
    с endtime ≤ T0: сумма и 6 компонентов; за (T0 − 24 ч; T0] — минимум и максимум суммы и изменение
    (значение на T0 − первое значение в окне).
  * Второй уровень очистки: экстремальные, но возможные значения ограничиваются клинически заданными границами
    (таблица сохраняется как приложение к протоколу). Границы фиксированы заранее и не подбираются по данным.
    Ограничение монотонно, поэтому его применение после агрегации (последнее, минимум, максимум) даёт тот же
    результат, что и применение к отдельным измерениям.
  * Рассчитанные индексы (ИМТ, ППТ, NLR, PLR, мочевина/креатинин, лактат/альбумин, шоковый индекс,
    индекс ROX = SpO2/FiO2/ЧД по последним значениям; сам SpO2/FiO2 не используется) — отдельные
    столбцы idx_*; в основную модель не входят (анализ чувствительности).
  * Признаки основной модели с долей пропусков > 90% в обучающем периоде (основной анализ) исключаются.
  * Основной анализ: main_analysis этапа 1 (данные об инфузиях есть, период однозначен) И нет ограничения терапии
    на T0 (limited_care = 0: DNR/DNI/CMO по последней записи code status до T0, этап 2).
    Итоговая схема отбора в согласованном порядке — stage6_flowchart.json:
    эпизоды → первый эпизод → T0 в ОРИТ → смерть не ранее T0 → без ограничения терапии →
    данные об инфузиях в окне → однозначный период → обучение / тест.

Вход: stage1_cohort*.parquet, stage1_flowchart.json, stage2_baseline, stage3_support, stage4_labs, stage5_vitals; sofa
Выход: stage6_master.parquet (одна строка на stay_id обеих когорт), stage6_features.json, stage6_flowchart.json,
       stage6_missingness.csv, stage6_clip_bounds.csv
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import duckdb

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
sys.path.insert(0, _HERE)
from common import Stage, DATA, raw, is_gzip

WINDOW_H = 24
MAX_MISSING = 0.90
SOFA_COMP = ['respiration', 'coagulation', 'liver', 'cardiovascular', 'cns', 'renal']
SOFA_SCHEMA = {'stay_id': 'BIGINT', 'hr': 'BIGINT', 'starttime': 'TIMESTAMP', 'endtime': 'TIMESTAMP',
               'pao2fio2ratio_novent': 'DOUBLE', 'pao2fio2ratio_vent': 'DOUBLE', 'rate_epinephrine': 'DOUBLE',
               'rate_norepinephrine': 'DOUBLE', 'rate_dopamine': 'DOUBLE', 'rate_dobutamine': 'DOUBLE',
               'meanbp_min': 'DOUBLE', 'gcs_min': 'DOUBLE', 'uo_24hr': 'DOUBLE', 'bilirubin_max': 'DOUBLE',
               'creatinine_max': 'DOUBLE', 'platelet_min': 'DOUBLE', 'respiration': 'DOUBLE', 'coagulation': 'DOUBLE',
               'liver': 'DOUBLE', 'cardiovascular': 'DOUBLE', 'cns': 'DOUBLE', 'renal': 'DOUBLE',
               'respiration_24hours': 'DOUBLE', 'coagulation_24hours': 'DOUBLE', 'liver_24hours': 'DOUBLE',
               'cardiovascular_24hours': 'DOUBLE', 'cns_24hours': 'DOUBLE', 'renal_24hours': 'DOUBLE',
               'sofa_24hours': 'DOUBLE'}
AGG_SUFFIX = ('_last', '_min', '_max')
# Второй уровень очистки: клинические границы (нижняя, верхняя) для показателя (все агрегаты last/min/max)
CLIP = {
    'wbc': (0.1, 100), 'hemoglobin': (3, 20), 'platelets': (5, 1000), 'rdw': (10, 30), 'neut_abs': (0, 80),
    'lymph_abs': (0, 30), 'neut_pct': (0, 100), 'lymph_pct': (0, 100), 'sodium': (110, 170), 'potassium': (1.5, 9),
    'mono_abs': (0, 20), 'eos_abs': (0, 10), 'mono_pct': (0, 100), 'eos_pct': (0, 100), 'mcv': (60, 130),
    'chloride': (70, 140), 'bicarbonate': (5, 50), 'bun': (2, 200), 'creatinine': (0.2, 15), 'glucose': (30, 1000),
    'calcium': (4, 15), 'aniongap': (2, 45), 'albumin': (1, 6), 'magnesium': (0.5, 6), 'phosphate': (0.5, 15),
    'inr': (0.8, 10), 'ptt': (15, 150), 'fibrinogen': (50, 1000), 'alt': (3, 5000), 'ast': (5, 10000),
    'alp': (10, 2000), 'bilirubin': (0.1, 40), 'ldh': (50, 10000), 'troponin_t': (0.01, 10), 'lactate': (0.3, 25),
    'ph': (6.8, 7.7), 'base_excess': (-30, 30), 'ionized_ca': (0.5, 2), 'pao2': (30, 600), 'paco2': (15, 120),
    'pf_ratio': (30, 700), 'heart_rate': (30, 220), 'resp_rate': (4, 60), 'temperature': (32, 42),
    'sbp': (50, 250), 'mbp': (30, 180), 'dbp': (20, 150), 'spo2': (60, 100), 'fio2': (21, 100),
    'sf_ratio': (60, 480),
}
CLIP_EXACT = {'nee_t0': (0, 3), 'nee_max_24h': (0, 3), 'uo_mlkgh_6h': (0, 20), 'uo_mlkgh_24h': (0, 20),
              'weight_kg': (35, 250), 'height_cm': (130, 210)}
CATEGORICAL = ['admission_group', 'careunit_group', 'resp_support_t0']
DESCRIPTIVE = ['race_group', 'code_status_t0', 'limited_care', 'non_icu_level_unit', 'covid19_dx',
               'weight_source', 'height_source', 'has_infusion_data', 'est_year', 't0_shift_h', 'icu_to_t0_h',
               'hosp_to_t0_h', 'any_sedative_t0', 'vasopressor_t0']
SENSITIVITY_CCI = ['cci_no_age']
# Считаются этапами 3, 5, 6, но в модель не входят (решение по составу признаков 08.10.2026); остаются как описательные
NOT_IN_MODEL = (['sofa_min_24h', 'sofa_max_24h'] + [f'sofa_{c}_t0' for c in SOFA_COMP] +
                ['arterial_line_24h', 'n_vasopressors_t0', 'vasopressin_t0', 'angiotensin2_24h', 'inotrope_24h',
                 'uo_mlkgh_6h'] +
                # лейкоцитарная формула: в модели только абсолютные значения (проценты дублируют их вместе с wbc)
                ['neut_pct_last', 'lymph_pct_last', 'mono_pct_last', 'eos_pct_last'])
DESCRIPTIVE = DESCRIPTIVE + NOT_IN_MODEL
H = lambda x: pd.Timedelta(hours=x)


def base_of(col):
    for s in AGG_SUFFIX:
        if col.endswith(s):
            return col[: -len(s)]
    return None


with Stage('stage6_master') as st:
    st.section('1. Когорты и выходы этапов 2–5')
    c1 = pd.read_parquet(f'{DATA}/stage1_cohort.parquet')
    c2 = pd.read_parquet(f'{DATA}/stage1_cohort_standard.parquet')
    base_cols = ['stay_id', 'subject_id', 'hadm_id', 't0', 'intime', 'age', 'sex_male', 'death_30d', 'split',
                 'est_year', 'has_infusion_data', 'main_analysis', 't0_shift_h', 'icu_to_t0_h', 'hosp_to_t0_h']
    df = pd.concat([c1[base_cols], c2[base_cols]]).drop_duplicates('stay_id').reset_index(drop=True)
    df['in_main_cohort'] = df['stay_id'].isin(set(c1['stay_id'])).astype(int)
    df['in_standard_cohort'] = df['stay_id'].isin(set(c2['stay_id'])).astype(int)
    df['t0'] = df['t0'].astype('datetime64[ns]')
    n0 = len(df)
    for name in ['stage2_baseline', 'stage3_support', 'stage4_labs', 'stage5_vitals']:
        part = pd.read_parquet(f'{DATA}/{name}.parquet')
        dup = (set(part.columns) & set(df.columns)) - {'stay_id'}
        st.check(not dup, f'{name}: нет повторяющихся столбцов ({sorted(dup)})')
        st.check(part['stay_id'].is_unique and set(part['stay_id']) == set(df['stay_id']),
                 f'{name}: одна строка на каждое пребывание обеих когорт')
        df = df.merge(part, on='stay_id', how='left')
    st.log('stays', int(n0))

    # ------------------------------------------------------------------
    st.section('1б. Ограничение терапии на T0 и итоговая схема отбора')
    df['main_analysis'] = ((df['main_analysis'] == 1) & (df['limited_care'] == 0)).astype(int)
    with open(f'{DATA}/stage1_flowchart.json', encoding='utf-8') as f:
        flow1 = json.load(f)
    flows = {}
    for key, flag in [('modified', 'in_main_cohort'), ('standard', 'in_standard_cohort')]:
        s = df[df[flag] == 1]
        fl = [f for f in flow1[key] if not f['step'].startswith('  ')][:4]   # до «Смерть не ранее T0»
        st.check(fl[-1]['patients'] == len(s), f'{key}: число пациентов совпадает со схемой этапа 1')
        add = lambda name, d: fl.append({'step': name, 'patients': int(len(d)), 'rows': int(len(d))})
        a = s[s['limited_care'] == 0];           add('Без ограничения терапии на T0 (DNR/DNI/CMO)', a)
        b = a[a['has_infusion_data'] == 1];      add('Данные об инфузиях в окне до T0', b)
        c = b[b['split'] != 'ambiguous'];        add('Однозначный период (основной анализ)', c)
        add('  из них обучающий период', c[c['split'] == 'train'])
        add('  из них тестовый период', c[c['split'] == 'test'])
        st.check(len(c) == int(s['main_analysis'].sum()), f'{key}: схема отбора согласована с main_analysis')
        flows[key] = fl
        for f in fl:
            print(f"  [{key}] {f['step']:<46} {f['patients']:>7,}")
    with open(f'{DATA}/stage6_flowchart.json', 'w', encoding='utf-8') as f:
        json.dump(flows, f, ensure_ascii=False, indent=2)
    st.log('limited_care_excluded_main', int(((df['in_main_cohort'] == 1) & (df['limited_care'] == 1)).sum()))

    # ------------------------------------------------------------------
    st.section('2. SOFA на T0 и за 24 ч')
    con = duckdb.connect()
    k = df[['stay_id', 't0']].assign(w0=lambda d: d['t0'] - H(WINDOW_H))
    con.register('k', k)
    p = raw('sofa')
    cols = ', '.join(f"'{a}': '{b}'" for a, b in SOFA_SCHEMA.items())
    src = (f"read_csv('{p}', header=true, compression='{'gzip' if is_gzip(p) else 'none'}', "
           f"columns={{{cols}}}, quote='\"', escape='\"')")
    comp24 = ', '.join(f's.{c}_24hours' for c in SOFA_COMP)
    sofa = con.execute(f"""SELECT s.stay_id, s.endtime, s.sofa_24hours, {comp24}
                           FROM {src} s JOIN k ON s.stay_id = k.stay_id
                           WHERE s.endtime > k.w0 AND s.endtime <= k.t0""").df()
    sofa['endtime'] = sofa['endtime'].astype('datetime64[ns]')
    sofa = sofa.sort_values(['stay_id', 'endtime'])
    g = sofa.groupby('stay_id')
    at = g.last()
    df['sofa_t0'] = df['stay_id'].map(at['sofa_24hours'])
    for c in SOFA_COMP:
        df[f'sofa_{c}_t0'] = df['stay_id'].map(at[f'{c}_24hours'])
    df['sofa_min_24h'] = df['stay_id'].map(g['sofa_24hours'].min())
    df['sofa_max_24h'] = df['stay_id'].map(g['sofa_24hours'].max())
    df['sofa_delta_24h'] = df['sofa_t0'] - df['stay_id'].map(g['sofa_24hours'].first())
    main = df[df['in_main_cohort'] == 1]
    st.log('sofa_t0_available_pct', round(float(df['sofa_t0'].notna().mean() * 100), 1))
    st.log('sofa_t0_ge2_pct_main', round(float((main['sofa_t0'] >= 2).mean() * 100), 1))
    st.log('sofa_t0_median_iqr_main', [float(x) for x in main['sofa_t0'].quantile([.5, .25, .75])])

    # ------------------------------------------------------------------
    st.section('3. Второй уровень очистки (клинические границы)')
    clip_rows, n_clipped = [], {}
    for col in df.columns:
        b = base_of(col)
        rng = CLIP.get(b) if b else CLIP_EXACT.get(col)
        if rng is None:
            continue
        lo, hi = rng
        before = df[col].copy()
        df[col] = df[col].clip(lo, hi)
        n = int((before.notna() & (before != df[col])).sum())
        n_clipped[col] = n
        clip_rows.append({'feature': col, 'lower': lo, 'upper': hi, 'n_clipped': n})
    pd.DataFrame(clip_rows).to_csv(f'{DATA}/stage6_clip_bounds.csv', index=False)
    st.log('clipped_values_top', dict(sorted(n_clipped.items(), key=lambda x: -x[1])[:10]))

    # ------------------------------------------------------------------
    st.section('4. Рассчитанные индексы (только анализ чувствительности)')
    df = df.copy()   # дефрагментация после многих merge
    # SpO2/FiO2 не используется; из него считается только индекс ROX (анализ чувствительности, решение 18)
    df['idx_rox'] = df['sf_ratio_last'] / df['resp_rate_last']
    df = df.drop(columns=['sf_ratio_last'])
    h_m = df['height_cm'] / 100
    df['idx_bmi'] = df['weight_kg'] / h_m ** 2
    df['idx_bsa'] = np.sqrt(df['weight_kg'] * df['height_cm'] / 3600)
    df['idx_nlr'] = df['neut_abs_last'] / df['lymph_abs_last'].where(df['lymph_abs_last'] > 0)
    df['idx_plr'] = df['platelets_last'] / df['lymph_abs_last'].where(df['lymph_abs_last'] > 0)
    df['idx_bun_creatinine'] = df['bun_last'] / df['creatinine_last']
    df['idx_lactate_albumin'] = df['lactate_last'] / df['albumin_last']
    df['idx_shock_index'] = df['heart_rate_last'] / df['sbp_last']
    idx_cols = [c for c in df.columns if c.startswith('idx_')]
    st.log('idx_available_pct', {c: round(float(df[c].notna().mean() * 100), 1) for c in idx_cols})

    # ------------------------------------------------------------------
    st.section('5. Перечень признаков основной модели')
    s3 = pd.read_parquet(f'{DATA}/stage3_support.parquet').columns
    s4 = pd.read_parquet(f'{DATA}/stage4_labs.parquet').columns
    s5 = pd.read_parquet(f'{DATA}/stage5_vitals.parquet').columns
    candidates = (['age', 'sex_male', 'admission_group', 'careunit_group', 'weight_kg', 'height_cm',
                   'sofa_t0'] + [f'sofa_{c}_t0' for c in SOFA_COMP] + ['sofa_min_24h', 'sofa_max_24h', 'sofa_delta_24h']
                  + [c for c in s3 if c not in ('stay_id', 'any_sedative_t0', 'vasopressor_t0')]
                  + [c for c in s4 if c != 'stay_id']
                  + [c for c in s5 if c not in ('stay_id', 'sf_ratio_last', 'sf_ratio_min')])
    candidates = [c for c in candidates if c not in NOT_IN_MODEL]
    for c in CATEGORICAL:
        df[c] = df[c].astype('string')
    tr = df[(df['in_main_cohort'] == 1) & (df['main_analysis'] == 1) & (df['split'] == 'train')]
    te = df[(df['in_main_cohort'] == 1) & (df['main_analysis'] == 1) & (df['split'] == 'test')]
    miss = pd.DataFrame({'train': tr[candidates + idx_cols + SENSITIVITY_CCI].isna().mean(),
                         'test': te[candidates + idx_cols + SENSITIVITY_CCI].isna().mean()}).mul(100).round(1)
    miss.index.name = 'feature'
    miss.to_csv(f'{DATA}/stage6_missingness.csv')
    dropped = [c for c in candidates if miss.loc[c, 'train'] > MAX_MISSING * 100]
    features = [c for c in candidates if c not in dropped]
    st.log('n_candidates', len(candidates))
    st.log('dropped_missing_gt90', dropped)
    st.log('n_features_main', len(features))
    manifest = {
        'outcome': 'death_30d', 'id': 'stay_id', 'split': 'split',
        'main_filter': 'in_main_cohort == 1 & main_analysis == 1',
        'features_main': features, 'categorical': CATEGORICAL,
        'features_idx_sensitivity': idx_cols, 'features_cci_sensitivity': SENSITIVITY_CCI,
        'descriptive_only': [c for c in DESCRIPTIVE if c in df.columns] + [c for c in df.columns if c.startswith('cm_')],
        'dropped_missing_gt90': dropped, 'max_missing': MAX_MISSING,
    }
    with open(f'{DATA}/stage6_features.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    df.to_parquet(f'{DATA}/stage6_master.parquet', index=False)
    print(f'Сохранено: {DATA}/stage6_master.parquet ({len(df):,} строк, {df.shape[1]} столбцов)')
    print(f'Признаков основной модели: {len(features)}')

    st.log('main_analysis', {'train_n': int(len(tr)), 'train_deaths': int(tr['death_30d'].sum()),
                             'train_mortality_pct': round(float(tr['death_30d'].mean() * 100), 1),
                             'test_n': int(len(te)), 'test_deaths': int(te['death_30d'].sum()),
                             'test_mortality_pct': round(float(te['death_30d'].mean() * 100), 1),
                             'events_per_feature_train': round(float(tr['death_30d'].sum() / len(features)), 1)})

    forbidden = [c for c in features if any(w in c for w in ('death', 'dod', 'mortal', 'outcome', 'race',
                                                              'insurance', 'code_status', 'cci', 'cm_', 'idx_'))]
    st.check(df['stay_id'].is_unique and len(df) == n0, 'одна строка на пребывание, число строк не изменилось')
    st.check(not forbidden, f'в признаках основной модели нет исхода и исключённых по протоколу переменных ({forbidden})')
    st.check(df.loc[df['in_main_cohort'] == 1, 'sofa_t0'].notna().mean() >= 0.99, 'SOFA на T0 есть у ≥ 99% основной когорты')
    st.check((main['sofa_t0'] >= 2).mean() >= 0.90, 'SOFA на T0 ≥ 2 у ≥ 90% основной когорты')
    # Решение 08.10.2026: сдвиг дат MIMIC не выровнен по 1 января, поэтому единичные эпизоды 2020 г. могут получить
    # верхнюю границу интервала ≤ 2019 (выявлено 4 случая U07.1 + J12.89, т. е. 2020 г.). Пациенты не исключаются
    # (отбор по коду выписки = информация после T0); примесь оценивается и сообщается в ограничениях.
    n_cov_tr = int(tr['covid19_dx'].sum())
    st.log('covid19_in_train', {'n': n_cov_tr, 'pct': round(n_cov_tr / len(tr) * 100, 3)})
    st.check(n_cov_tr <= 0.001 * len(tr), f'пациентов с COVID-19 в обучающем периоде ≤ 0,1% ({n_cov_tr})')
    st.check(int(tr['limited_care'].sum() + te['limited_care'].sum()) == 0, 'в основном анализе нет пациентов с DNR/DNI/CMO на T0')
    st.check(set(tr['split']) == {'train'} and set(te['split']) == {'test'}, 'в основном анализе нет неоднозначного периода')
    st.check(set(features) <= set(df.columns), 'все признаки из перечня есть в таблице')
    st.check(not set(features) & set(NOT_IN_MODEL), 'исключённые из модели признаки не попали в перечень')
    print('Признаки основной модели:\n  ' + '\n  '.join(features))
