# -*- coding: utf-8 -*-
"""
ЭТАП 2. Исходные характеристики: демография, антропометрия, отделение и тип госпитализации,
код-статус, коды диагнозов (только для описания и анализа чувствительности).

Считается для объединения ICU-пребываний основной и стандартной когорт (стандартная нужна для анализа чувствительности).

  * Масса тела — только записи до T0: вес при поступлении из карты ОРИТ (226512 кг / 226531 фунты)
    → последний амбулаторный вес OMR за 365 сут до даты T0 → последнее ежедневное взвешивание (224639) до T0.
  * Рост — только записи до T0 (наличие более поздних записей зависит от выживания): индексное ICU-пребывание
    (226730 см / 226707 дюймы) → предшествующие ICU-пребывания → OMR (рост; рост из ИМТ и массы) с датой < даты T0.
  * Отдельные измерения вне 30–300 кг и 120–230 см отбрасываются до агрегации.
  * Код-статус — последняя запись до T0 по трём itemid (223758, 229784, 228687); не предиктор.
  * Коды диагнозов (индексная + предшествующие госпитализации): индекс Чарлсон без возрастного компонента —
    анализ чувствительности; флаги хронических заболеваний — описание когорты (таблица 1); COVID-19 (U07.1) — подгруппа.
  * Раса — только оценка в подгруппах.

Вход: stage1_cohort.parquet, stage1_cohort_standard.parquet; chartevents, omr, charlson, diagnoses_icd, admissions
Выход: data/stage2_baseline.parquet (одна строка на stay_id), cache_chartevents_stage2.parquet
"""
import os
import sys
import numpy as np
import pandas as pd
import duckdb

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
sys.path.insert(0, _HERE)
from common import Stage, DATA, raw, read_csv_any, is_gzip

ITEM = {'weight_admit_kg': 226512, 'weight_admit_lb': 226531, 'weight_daily_kg': 224639,
        'height_cm': 226730, 'height_in': 226707,
        'code_status': 223758, 'code_status_intubation': 229784, 'code_status_md': 228687}
CODE_STATUS_ITEMS = [ITEM['code_status'], ITEM['code_status_intubation'], ITEM['code_status_md']]
LB_TO_KG, IN_TO_CM = 0.453592, 2.54
W_RANGE, H_RANGE = (30, 300), (120, 230)
CACHE = f'{DATA}/cache_chartevents_stage2.parquet'
CE_SCHEMA = {'subject_id': 'BIGINT', 'hadm_id': 'BIGINT', 'stay_id': 'BIGINT', 'caregiver_id': 'BIGINT',
             'charttime': 'TIMESTAMP', 'storetime': 'TIMESTAMP', 'itemid': 'BIGINT', 'value': 'VARCHAR',
             'valuenum': 'DOUBLE', 'valueuom': 'VARCHAR', 'warning': 'BIGINT'}

UNIT_MAP = {
    'Medical Intensive Care Unit (MICU)': 'medical', 'Medicine': 'medical', 'Medicine/Cardiology Intermediate': 'medical',
    'Surgical Intensive Care Unit (SICU)': 'surgical_trauma', 'Trauma SICU (TSICU)': 'surgical_trauma',
    'Surgery/Vascular/Intermediate': 'surgical_trauma', 'Surgery/Trauma': 'surgical_trauma', 'PACU': 'surgical_trauma',
    'Cardiac Vascular Intensive Care Unit (CVICU)': 'cardiac', 'Coronary Care Unit (CCU)': 'cardiac',
    'Medical/Surgical Intensive Care Unit (MICU/SICU)': 'mixed', 'Med/Surg': 'mixed', 'Intensive Care Unit (ICU)': 'mixed',
    'Neuro Surgical Intensive Care Unit (Neuro SICU)': 'neuro', 'Neuro Intermediate': 'neuro',
    'Neuro Stepdown': 'neuro', 'Neurology': 'neuro',
}
NON_ICU_UNITS = {'Neuro Intermediate', 'Neuro Stepdown', 'Neurology', 'Med/Surg', 'Medicine',
                 'Medicine/Cardiology Intermediate', 'Surgery/Vascular/Intermediate', 'Surgery/Trauma', 'PACU'}
CCI_COMP = ['myocardial_infarct', 'congestive_heart_failure', 'peripheral_vascular_disease', 'cerebrovascular_disease',
            'dementia', 'chronic_pulmonary_disease', 'rheumatic_disease', 'peptic_ulcer_disease', 'mild_liver_disease',
            'diabetes_without_cc', 'diabetes_with_cc', 'paraplegia', 'renal_disease', 'malignant_cancer',
            'severe_liver_disease', 'metastatic_solid_tumor', 'aids']
CHRONIC = {'cm_chf': ['congestive_heart_failure'], 'cm_copd': ['chronic_pulmonary_disease'],
           'cm_diabetes': ['diabetes_without_cc', 'diabetes_with_cc'], 'cm_ckd': ['renal_disease'],
           'cm_liver': ['mild_liver_disease', 'severe_liver_disease'],
           'cm_malignancy': ['malignant_cancer', 'metastatic_solid_tumor'], 'cm_metastatic': ['metastatic_solid_tumor'],
           'cm_dementia': ['dementia']}


def cci_no_age(c):
    """Индекс Чарлсон с весами и иерархиями mimic-code, без возрастного компонента."""
    return (c['myocardial_infarct'] + c['congestive_heart_failure'] + c['peripheral_vascular_disease']
            + c['cerebrovascular_disease'] + c['dementia'] + c['chronic_pulmonary_disease']
            + c['rheumatic_disease'] + c['peptic_ulcer_disease']
            + np.maximum(c['mild_liver_disease'], 3 * c['severe_liver_disease'])
            + np.maximum(2 * c['diabetes_with_cc'], c['diabetes_without_cc'])
            + np.maximum(2 * c['malignant_cancer'], 6 * c['metastatic_solid_tumor'])
            + 2 * c['paraplegia'] + 2 * c['renal_disease'] + 6 * c['aids'])


def map_code_status(v):
    v = str(v).lower()
    if 'comfort' in v or 'cmo' in v:
        return 'cmo'
    if 'dnr' in v or 'dni' in v or 'do not' in v or 'dnar' in v:
        return 'limited'
    if 'full' in v:
        return 'full'
    return None


def race_group(v):
    v = str(v).upper()
    if v.startswith('WHITE') or v == 'PORTUGUESE':
        return 'white'
    if v.startswith('BLACK'):
        return 'black'
    if v.startswith('HISPANIC') or v == 'SOUTH AMERICAN':
        return 'hispanic'
    if v.startswith('ASIAN'):
        return 'asian'
    if v in ('UNKNOWN', 'UNABLE TO OBTAIN', 'PATIENT DECLINED TO ANSWER', 'NAN', 'NONE'):
        return 'unknown'
    return 'other'


def admission_group(v):
    v = str(v).upper()
    if v in ('ELECTIVE', 'SURGICAL SAME DAY ADMISSION'):
        return 'elective'
    if 'OBSERVATION' in v:
        return 'observation'
    return 'emergency_urgent'


with Stage('stage2_baseline') as st:
    st.section('1. Когорты')
    keep = ['subject_id', 'hadm_id', 'stay_id', 't0', 'first_careunit', 'admission_type', 'race', 'insurance',
            'admittime', 'split', 'main_analysis', 'death_30d']
    c_mod = pd.read_parquet(f'{DATA}/stage1_cohort.parquet')[keep]
    c_std = pd.read_parquet(f'{DATA}/stage1_cohort_standard.parquet')[keep]
    df = pd.concat([c_mod, c_std]).drop_duplicates('stay_id').reset_index(drop=True)
    st.log('stays_union', int(len(df)))
    sid = df['stay_id']

    st.section('2. Выборка из chartevents (DuckDB, с кэшем)')
    if os.path.exists(CACHE):
        ce = pd.read_parquet(CACHE)
        print(f'Кэш: {len(ce):,} строк')
    else:
        p = raw('chartevents')
        cols = ', '.join(f"'{k}': '{v}'" for k, v in CE_SCHEMA.items())
        src = (f"read_csv('{p}', header=true, compression='{'gzip' if is_gzip(p) else 'none'}', "
               f"columns={{{cols}}}, quote='\"', escape='\"')")
        ids = ','.join(str(v) for v in ITEM.values())
        ce = duckdb.connect().execute(f"""SELECT stay_id, charttime, itemid, value, valuenum FROM {src}
                                          WHERE itemid IN ({ids}) AND stay_id IS NOT NULL""").df()
        ce.to_parquet(CACHE, index=False)
        print(f'Кэш построен: {len(ce):,} строк (все ICU-пребывания)')
    ce = ce[ce['stay_id'].isin(set(sid))].merge(df[['stay_id', 't0']], on='stay_id')
    print(ce['itemid'].map({v: k for k, v in ITEM.items()}).value_counts().to_string())

    st.section('3. Масса тела (только до T0)')
    aw = ce[ce['itemid'].isin([ITEM['weight_admit_kg'], ITEM['weight_admit_lb']]) & (ce['charttime'] <= ce['t0'])].copy()
    aw['kg'] = np.where(aw['itemid'] == ITEM['weight_admit_lb'], aw['valuenum'] * LB_TO_KG, aw['valuenum'])
    aw = aw[aw['kg'].between(*W_RANGE)].sort_values('charttime')
    w1 = aw.groupby('stay_id')['kg'].first()
    omr = read_csv_any(raw('omr'), usecols=['subject_id', 'chartdate', 'result_name', 'result_value'])
    omr = omr[omr['subject_id'].isin(set(df['subject_id']))].copy()
    omr['chartdate'] = pd.to_datetime(omr['chartdate'])
    omr['val'] = pd.to_numeric(omr['result_value'], errors='coerce')
    ow = omr[omr['result_name'].str.startswith('Weight (Lbs)', na=False)].copy()
    ow['kg'] = ow['val'] * LB_TO_KG
    ow = ow[ow['kg'].between(*W_RANGE)].merge(df[['subject_id', 'stay_id', 't0']], on='subject_id')
    t0d = ow['t0'].dt.normalize()
    ow = ow[(ow['chartdate'] < t0d) & (ow['chartdate'] >= t0d - pd.Timedelta(days=365))].sort_values('chartdate')
    w2 = ow.groupby('stay_id')['kg'].last()
    dw = ce[(ce['itemid'] == ITEM['weight_daily_kg']) & (ce['charttime'] <= ce['t0'])].copy()
    dw = dw[dw['valuenum'].between(*W_RANGE)].sort_values('charttime')
    w3 = dw.groupby('stay_id')['valuenum'].last()
    df['weight_kg'], df['weight_source'] = sid.map(w1), np.where(sid.map(w1).notna(), 'icu_admission', None)
    for name, s in [('omr_365d', w2), ('daily_before_t0', w3)]:
        fill = df['weight_kg'].isna() & sid.map(s).notna()
        df.loc[fill, 'weight_kg'] = sid[fill].map(s)
        df.loc[fill, 'weight_source'] = name
    df['weight_source'] = df['weight_source'].fillna('missing')
    st.log('weight_source', df['weight_source'].value_counts().to_dict())
    st.log('weight_available_pct', round(float(df['weight_kg'].notna().mean() * 100), 1))

    st.section('4. Рост (только записи до T0)')
    # Рост взрослого постоянен, но НАЛИЧИЕ записи после T0 зависит от выживания (будущие госпитализации и визиты),
    # поэтому используются только записи, сделанные до T0: индексное пребывание → предшествующие ICU-пребывания
    # → амбулаторные OMR (дата < даты T0).
    hall = pd.read_parquet(CACHE)
    hall = hall[hall['itemid'].isin([ITEM['height_cm'], ITEM['height_in']])].copy()
    hall['cm'] = np.where(hall['itemid'] == ITEM['height_in'], hall['valuenum'] * IN_TO_CM, hall['valuenum'])
    hall = hall[hall['cm'].between(*H_RANGE)]
    stay_subj = read_csv_any(raw('icustays'), usecols=['subject_id', 'stay_id'])
    hall = hall.merge(stay_subj, on='stay_id', how='left').rename(columns={'stay_id': 'src_stay'})
    hall = hall.merge(df[['subject_id', 'stay_id', 't0']], on='subject_id')
    hall = hall[hall['charttime'] <= hall['t0']]
    h1 = hall[hall['src_stay'] == hall['stay_id']].groupby('stay_id')['cm'].median()
    h2 = hall[hall['src_stay'] != hall['stay_id']].groupby('stay_id')['cm'].median()
    names = omr['result_name'].fillna('')
    t0d_all = df[['subject_id', 'stay_id', 't0']].assign(t0d=lambda d: d['t0'].dt.normalize())
    # (а) все варианты «Height*»: единицы по диапазону (48–90 — дюймы, 120–230 — см)
    oh = omr[names.str.lower().str.startswith('height')].copy()
    oh['cm'] = np.where(oh['val'].between(48, 90), oh['val'] * IN_TO_CM,
                        np.where(oh['val'].between(*H_RANGE), oh['val'], np.nan))
    oh = oh[oh['cm'].between(*H_RANGE)].merge(t0d_all, on='subject_id')
    h_omr = oh[oh['chartdate'] < oh['t0d']].groupby('stay_id')['cm'].median()
    # (б) рост из ИМТ и массы, записанных в один день: рост = √(масса / ИМТ)
    ob = omr[names.str.lower().str.startswith('bmi')][['subject_id', 'chartdate', 'val']].rename(columns={'val': 'bmi'})
    owt = omr[names.str.lower().str.startswith('weight')][['subject_id', 'chartdate', 'val']].rename(columns={'val': 'lb'})
    bw = ob.merge(owt, on=['subject_id', 'chartdate']).dropna()
    bw = bw[bw['bmi'].between(10, 80) & (bw['lb'] * LB_TO_KG).between(*W_RANGE)]
    bw['cm'] = np.sqrt(bw['lb'] * LB_TO_KG / bw['bmi']) * 100
    bw = bw[bw['cm'].between(*H_RANGE)].merge(t0d_all, on='subject_id')
    h_bmi = bw[bw['chartdate'] < bw['t0d']].groupby('stay_id')['cm'].median()
    df['height_cm'] = sid.map(h1)
    df['height_source'] = np.where(df['height_cm'].notna(), 'index_icu_stay_before_t0', None)
    for name, s_ in [('prior_icu_stays', h2), ('omr_height_before_t0', h_omr), ('omr_bmi_weight_before_t0', h_bmi)]:
        fill = df['height_cm'].isna() & sid.map(s_).notna()
        df.loc[fill, 'height_cm'] = sid[fill].map(s_)
        df.loc[fill, 'height_source'] = name
    df['height_source'] = df['height_source'].fillna('missing')
    st.log('height_source', df['height_source'].value_counts().to_dict())
    st.log('height_available_pct', round(float(df['height_cm'].notna().mean() * 100), 1))
    trm = df[(df['split'] == 'train') & (df['main_analysis'] == 1)]
    print('Летальность по источнику роста (обучающий период):\n' +
          trm.groupby('height_source')['death_30d'].agg(['size', 'mean']).round(3).to_string())
    print('Летальность по источнику массы (обучающий период):\n' +
          trm.groupby('weight_source')['death_30d'].agg(['size', 'mean']).round(3).to_string())

    st.section('5. Код-статус на T0 (не предиктор)')
    cs = ce[ce['itemid'].isin(CODE_STATUS_ITEMS) & (ce['charttime'] <= ce['t0'])].copy()
    print('Записей по itemid:', cs['itemid'].value_counts().to_dict())
    cs['cat'] = cs['value'].map(map_code_status)
    unm = cs.loc[cs['cat'].isna(), 'value'].value_counts().head(10)
    if len(unm):
        print('Несопоставленные значения (топ-10):\n' + unm.to_string())
    last = cs.dropna(subset=['cat']).sort_values('charttime').groupby('stay_id')['cat'].last()
    df['code_status_t0'] = sid.map(last).fillna('not_documented')
    df['limited_care'] = df['code_status_t0'].isin(['limited', 'cmo']).astype(int)
    tab = pd.crosstab(df['split'], df['code_status_t0'], normalize='index').mul(100).round(1)
    print(tab.to_string())
    st.log('code_status_pct', df['code_status_t0'].value_counts(normalize=True).mul(100).round(1).to_dict())

    st.section('6. Отделение, тип госпитализации, раса')
    df['careunit_group'] = df['first_careunit'].map(UNIT_MAP)
    unmapped = df.loc[df['careunit_group'].isna(), 'first_careunit'].value_counts()
    df['careunit_group'] = df['careunit_group'].fillna('other')
    df['non_icu_level_unit'] = df['first_careunit'].isin(NON_ICU_UNITS).astype(int)
    df['admission_group'] = df['admission_type'].map(admission_group)
    df['race_group'] = df['race'].map(race_group)
    for c in ['careunit_group', 'admission_group', 'race_group']:
        st.log(f'{c}_counts', df[c].value_counts().to_dict())
    st.log('non_icu_level_unit_pct', round(float(df['non_icu_level_unit'].mean() * 100), 1))

    st.section('7. Коды диагнозов (описание и анализ чувствительности)')
    adm = read_csv_any(raw('admissions'), usecols=['subject_id', 'hadm_id', 'dischtime'])
    adm['dischtime'] = pd.to_datetime(adm['dischtime'])
    # Пул госпитализаций для каждого ICU-пребывания: индексная + завершившиеся до её начала.
    # (У одного пациента основная и стандартная когорты могут иметь разные индексные госпитализации.)
    prior = adm.merge(df[['stay_id', 'subject_id', 'admittime']], on='subject_id')
    prior = prior.loc[prior['dischtime'] <= prior['admittime'], ['stay_id', 'hadm_id']]
    pool = pd.concat([df[['stay_id', 'hadm_id']], prior]).drop_duplicates()
    ch = read_csv_any(raw('charlson'), usecols=['hadm_id'] + CCI_COMP)
    ch[CCI_COMP] = ch[CCI_COMP].fillna(0).astype(int)
    comp = pool.merge(ch, on='hadm_id', how='inner').groupby('stay_id')[CCI_COMP].max()
    df['cci_no_age'] = sid.map(cci_no_age(comp))
    for name, cols in CHRONIC.items():
        df[name] = sid.map(comp[cols].max(axis=1)).fillna(0).astype(int)
    st.log('stays_with_prior_admissions_pct', round(float(sid.isin(set(prior['stay_id'])).mean() * 100), 1))
    dx = read_csv_any(raw('diagnoses_icd'), usecols=['hadm_id', 'icd_code', 'icd_version'])
    dx = dx[dx['hadm_id'].isin(set(df['hadm_id']))]
    covid = set(dx.loc[(dx['icd_version'] == 10) & dx['icd_code'].astype(str).str.upper().str.startswith('U071'), 'hadm_id'])
    df['covid19_dx'] = df['hadm_id'].isin(covid).astype(int)
    st.log('cci_no_age_median', float(df['cci_no_age'].median()))
    st.log('chronic_pct', {c: round(float(df[c].mean() * 100), 1) for c in CHRONIC})
    st.log('covid19_by_split', df.groupby('split')['covid19_dx'].sum().to_dict())

    st.section('8. Сохранение и контрольные проверки')
    out_cols = ['stay_id', 'weight_kg', 'weight_source', 'height_cm', 'height_source', 'code_status_t0', 'limited_care',
                'careunit_group', 'non_icu_level_unit', 'admission_group', 'race_group', 'cci_no_age',
                'covid19_dx'] + list(CHRONIC)
    df[out_cols].to_parquet(f'{DATA}/stage2_baseline.parquet', index=False)
    print(f'Сохранено: {DATA}/stage2_baseline.parquet ({len(df):,} строк)')
    st.check(df['stay_id'].is_unique, 'одна строка на ICU-пребывание')
    st.check(len(unmapped) == 0, f'все отделения сопоставлены с группами (не сопоставлены: {unmapped.to_dict()})')
    st.check(df['weight_kg'].notna().mean() >= 0.90, 'масса тела до T0 доступна у ≥ 90%')
