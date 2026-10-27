# -*- coding: utf-8 -*-
"""
ЭТАП 0. Определение сепсиса (Sepsis-3).

1. Воспроизведение операционализации mimic-code (suspicion_of_infection.sql + sepsis3_new.sql) и
   жёсткая сверка с эталоном sepsis3_new.csv.gz (расчёт mimic-code на MIMIC-IV v3.1):
   совпадение набора эпизодов и всех четырёх времён должно быть 100%, иначе этап останавливается.
2. Та же логика с тремя уточнениями, повышающими специфичность подозрения на инфекцию:
   а) скрининговые и серологические исследования не считаются посевом;
   б) несистемные формы антибиотиков не считаются антибактериальной терапией;
   в) цефазолин в/в в первые 48 ч плановой госпитализации — периоперационная профилактика.
   Уточнения применяются к логике (подозрение переопределяется по следующему подходящему посеву),
   а не к готовым строкам.

Вход (SEPSIS_RAW_DIR): antibiotic, microbiologyevents, sofa, admissions, sepsis3_new (эталон)
Выход (data/):
  sepsis3_standard.parquet  — стандартное определение (для анализа чувствительности)
  sepsis3_modified.parquet  — основное определение
  stage0_lost_episodes.parquet — эпизоды, исключённые уточнениями, с причиной (для описания)
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Stage, DATA, raw, read_csv_any

# Правила уточнений
SCREEN_SPEC_RE = r'SCREEN|STAPH AUREUS SWAB|SEROLOGY|IMMUNOLOGY|\(CMV|\(EBV|TOXO'
SCREEN_TEST_RE = r'SCREEN|VANCOMYCIN RESISTANT|MRSA'
RESP_VIRAL_RE = r'RESPIRATORY VIRAL'
NONSYSTEMIC_ROUTES = {'TP', 'NU', 'IH', 'NEB', 'OU', 'OS', 'OD', 'AU', 'AS', 'AD', 'ED', 'VG',
                      'LOCK', 'IRR', 'IR', 'INTRAVITREAL', 'IVT',
                      'INHALATION', 'NASAL', 'TOPICAL', 'VAGINAL', 'BOTH EYES', 'LEFT EYE',
                      'RIGHT EYE', 'BOTH EARS', 'LEFT EAR', 'RIGHT EAR'}
NONSYSTEMIC_DRUG_RE = (r'ointment|cream|gel\b|ophth|otic|drops|nasal|inhal|neb|topical|'
                       r'mupirocin|bacitracin|rifaximin|neomycin|vaginal|lotion')
PROPH_DRUG_RE = r'cefazolin'
PROPH_ROUTES = {'IV', 'PB', 'IVPB', 'IV DRIP'}
PROPH_ADMISSION_TYPES = {'ELECTIVE', 'SURGICAL SAME DAY ADMISSION'}
PROPH_WINDOW_H = 48
NEGATIVE_ORG_ITEMID = 90856            # как в mimic-code: «организм» = отрицательный результат
TIME_COLS = ['sofa_time', 'suspected_infection_time', 'antibiotic_time', 'culture_time']

H = lambda x: pd.Timedelta(hours=x)


def build_suspicion(ab_df, me_df):
    """Подозрение на инфекцию — логика mimic-code (suspicion_of_infection.sql).
    Посев с временем: до антибиотика в [abx − 72 ч, abx) или после в (abx, abx + 24 ч].
    Посев только с датой: дата антибиотика в [дата посева, +3 дня] (до) или в [дата посева − 1 день, дата посева] (после).
    Из подходящих — самый ранний по дате; при равной дате — посев с временем."""
    a = ab_df[['subject_id', 'stay_id', 'antibiotic', 'antibiotic_time']].copy().reset_index(drop=True)
    a['subject_id'] = a['subject_id'].astype('int64')
    a['row_id'] = np.arange(len(a))
    a['abx_date'] = a['antibiotic_time'].dt.normalize()
    m = me_df[['subject_id', 'charttime', 'chartdate', 'culture_time', 'specimen', 'positive_culture']].copy()
    m['subject_id'] = m['subject_id'].astype('int64')
    m['cdate'] = m['chartdate'].dt.normalize().fillna(m['charttime'].dt.normalize())
    m['culture_dateonly'] = m['charttime'].isna().astype(float)
    mt = m[m['charttime'].notna()].sort_values('charttime')
    md = m[m['charttime'].isna()].sort_values('cdate')
    res_cols = ['culture_time', 'specimen', 'positive_culture', 'cdate', 'culture_dateonly']

    def first_match(key_col, right, right_on, exact, valid):
        if right.empty:
            return pd.DataFrame(np.nan, index=a['row_id'].sort_values().values, columns=res_cols)
        cols = ['subject_id'] + ([right_on] if right_on not in res_cols else []) + res_cols
        x = pd.merge_asof(a.sort_values(key_col), right[cols], left_on=key_col, right_on=right_on,
                          by='subject_id', direction='forward', allow_exact_matches=exact)
        x = x.set_index('row_id').sort_index()
        x.loc[~valid(x), res_cols] = np.nan
        return x[res_cols]

    def earliest(t, d):
        use_d = d['cdate'].notna() & (t['cdate'].isna() | (d['cdate'] < t['cdate']))
        return t.where(~use_d, d)

    a['k_bt'] = a['antibiotic_time'] - H(72)
    bt = first_match('k_bt', mt, 'charttime', True, lambda x: x['culture_time'] < x['antibiotic_time'])
    a['k_bd'] = a['abx_date'] - pd.Timedelta(days=3)
    bd = first_match('k_bd', md, 'cdate', True, lambda x: x['cdate'] <= x['abx_date'])
    before = earliest(bt, bd)

    at = first_match('antibiotic_time', mt, 'charttime', False,
                     lambda x: x['culture_time'] <= x['antibiotic_time'] + H(24))
    ad = first_match('abx_date', md, 'cdate', True, lambda x: x['cdate'] <= x['abx_date'] + pd.Timedelta(days=1))
    after = earliest(at, ad)

    s = a.set_index('row_id').sort_index()[['subject_id', 'stay_id', 'antibiotic', 'antibiotic_time']].copy()
    has_b, has_a = before['culture_time'].notna(), after['culture_time'].notna()
    s['suspected_infection'] = (has_b | has_a).astype(int)
    s['suspected_infection_time'] = before['culture_time'].where(has_b, s['antibiotic_time'].where(has_a))
    for c in ['culture_time', 'specimen', 'positive_culture', 'culture_dateonly']:
        s[c] = before[c].where(has_b, after[c])
    return s.reset_index(drop=True)


def build_sepsis3(soi, sofa2):
    """SOFA ≥ 2 в окне [подозрение − 48 ч, подозрение + 24 ч]; первый эпизод на ICU-пребывание (sepsis3.sql)."""
    s = soi[(soi['suspected_infection'] == 1) & soi['stay_id'].notna()].copy()
    s['stay_id'] = s['stay_id'].astype('int64')
    s['key'] = s['suspected_infection_time'] - H(48)
    j = pd.merge_asof(s.sort_values('key'), sofa2, left_on='key', right_on='sofa_time',
                      by='stay_id', direction='forward', allow_exact_matches=True)
    j = j[j['sofa_time'] <= j['suspected_infection_time'] + H(24)]
    j = (j.sort_values(['stay_id', 'suspected_infection_time', 'antibiotic_time', 'culture_time', 'sofa_time'])
          .drop_duplicates('stay_id', keep='first'))
    cols = ['subject_id', 'stay_id', 'antibiotic', 'antibiotic_time', 'culture_time', 'suspected_infection_time',
            'specimen', 'positive_culture', 'culture_dateonly', 'sofa_time', 'sofa_score']
    out = j[cols].reset_index(drop=True)
    out['positive_culture'] = out['positive_culture'].astype(int)
    out['culture_dateonly'] = out['culture_dateonly'].astype(int)
    return out


with Stage('stage0_sepsis3') as st:
    # ------------------------------------------------------------------
    st.section('1. Микробиология: уровень образца')
    mi = read_csv_any(raw('microbiologyevents'),
                      usecols=['subject_id', 'micro_specimen_id', 'chartdate', 'charttime',
                               'spec_type_desc', 'test_name', 'org_itemid', 'org_name'])
    mi['charttime'] = pd.to_datetime(mi['charttime'], errors='coerce')
    mi['chartdate'] = pd.to_datetime(mi['chartdate'], errors='coerce')
    spec_u = mi['spec_type_desc'].fillna('').astype(str).str.upper()
    test_u = mi['test_name'].fillna('').astype(str).str.upper()
    resp_viral = spec_u.str.contains(RESP_VIRAL_RE) | test_u.str.contains(RESP_VIRAL_RE)
    mi['row_screen'] = (spec_u.str.contains(SCREEN_SPEC_RE) | test_u.str.contains(SCREEN_TEST_RE)) & ~resp_viral
    org = mi['org_name'].fillna('').astype(str).str.strip()
    mi['row_pos'] = (org != '') & (mi['org_itemid'] != NEGATIVE_ORG_ITEMID)
    me = mi.groupby('micro_specimen_id').agg(
        subject_id=('subject_id', 'max'), charttime=('charttime', 'max'), chartdate=('chartdate', 'max'),
        specimen=('spec_type_desc', 'max'), positive_culture=('row_pos', 'max'),
        screening=('row_screen', 'min'))            # образец скрининговый, если все его тесты скрининговые
    me['culture_time'] = me['charttime'].fillna(me['chartdate'])
    me = me.dropna(subset=['culture_time']).reset_index()
    me['positive_culture'] = me['positive_culture'].astype(int)
    st.log('specimens', int(len(me)))
    st.log('specimens_screening', int(me['screening'].sum()))
    st.log('specimens_screening_pct', round(me['screening'].mean() * 100, 1))
    st.log('specimens_date_only', int(me['charttime'].isna().sum()))
    print('Скрининговые образцы по типу (топ-10):\n' + me.loc[me['screening'], 'specimen'].value_counts().head(10).to_string())

    # ------------------------------------------------------------------
    st.section('2. Антибиотики')
    ab = read_csv_any(raw('antibiotic')).rename(columns={'starttime': 'antibiotic_time'})
    ab['antibiotic_time'] = pd.to_datetime(ab['antibiotic_time'], errors='coerce')
    ab = ab.dropna(subset=['antibiotic_time'])
    ab['stay_id'] = pd.to_numeric(ab['stay_id'], errors='coerce')
    adm = read_csv_any(raw('admissions'), usecols=['hadm_id', 'admittime', 'admission_type'])
    adm['admittime'] = pd.to_datetime(adm['admittime'], errors='coerce')
    ab = ab.merge(adm, on='hadm_id', how='left').reset_index(drop=True)
    route_u = ab['route'].fillna('').astype(str).str.upper().str.strip()
    drug_l = ab['antibiotic'].fillna('').astype(str).str.lower()
    ab['nonsystemic'] = route_u.isin(NONSYSTEMIC_ROUTES) | drug_l.str.contains(NONSYSTEMIC_DRUG_RE)
    h_from_adm = (ab['antibiotic_time'] - ab['admittime']).dt.total_seconds() / 3600
    ab['periop_prophylaxis'] = (ab['admission_type'].fillna('').str.upper().isin(PROPH_ADMISSION_TYPES)
                                & drug_l.str.contains(PROPH_DRUG_RE) & route_u.isin(PROPH_ROUTES)
                                & h_from_adm.le(PROPH_WINDOW_H) & ~ab['nonsystemic'])
    st.log('abx_orders', int(len(ab)))
    st.log('abx_nonsystemic', int(ab['nonsystemic'].sum()))
    st.log('abx_nonsystemic_pct', round(ab['nonsystemic'].mean() * 100, 1))
    st.log('abx_periop_prophylaxis', int(ab['periop_prophylaxis'].sum()))
    st.log('abx_periop_prophylaxis_pct', round(ab['periop_prophylaxis'].mean() * 100, 1))
    print('Исключённые несистемные препараты (топ-10):\n' + drug_l[ab['nonsystemic']].value_counts().head(10).to_string())

    # ------------------------------------------------------------------
    st.section('3. Почасовая SOFA (часы с SOFA ≥ 2)')
    icu_stays = set(ab['stay_id'].dropna().astype('int64'))
    parts = []
    for ch in read_csv_any(raw('sofa'), usecols=['stay_id', 'endtime', 'sofa_24hours'], chunksize=3_000_000):
        ch = ch[(ch['sofa_24hours'] >= 2) & ch['stay_id'].isin(icu_stays)]
        if len(ch):
            parts.append(ch)
    sofa2 = pd.concat(parts, ignore_index=True)
    sofa2['endtime'] = pd.to_datetime(sofa2['endtime'])
    sofa2['stay_id'] = sofa2['stay_id'].astype('int64')
    sofa2 = sofa2.rename(columns={'endtime': 'sofa_time', 'sofa_24hours': 'sofa_score'}).sort_values('sofa_time')
    st.log('sofa_hours_ge2', int(len(sofa2)))

    # ------------------------------------------------------------------
    st.section('4. Стандартное определение и сверка с эталоном')
    std = build_sepsis3(build_suspicion(ab, me), sofa2)
    ref = read_csv_any(raw('sepsis3'), usecols=['stay_id'] + TIME_COLS)
    for c in TIME_COLS:
        ref[c] = pd.to_datetime(ref[c])
    r_set, s_set = set(ref['stay_id']), set(std['stay_id'])
    st.log('reference_episodes', len(r_set))
    st.log('reproduced_episodes', len(s_set))
    st.log('only_in_reference', len(r_set - s_set))
    st.log('only_in_reproduced', len(s_set - r_set))
    cmp_ = ref.merge(std[['stay_id'] + TIME_COLS], on='stay_id', suffixes=('_ref', '_rep'))
    match = {c: float(((cmp_[f'{c}_ref'] == cmp_[f'{c}_rep']) |
                       (cmp_[f'{c}_ref'].isna() & cmp_[f'{c}_rep'].isna())).mean() * 100) for c in TIME_COLS}
    st.log('time_match_pct', {k: round(v, 3) for k, v in match.items()})
    st.check(r_set == s_set, 'набор эпизодов совпадает с эталоном mimic-code')
    st.check(all(v == 100.0 for v in match.values()), 'все четыре времени совпадают с эталоном в 100% эпизодов')
    std.to_parquet(f'{DATA}/sepsis3_standard.parquet', index=False)
    st.log('standard_patients', int(std['subject_id'].nunique()))

    # ------------------------------------------------------------------
    st.section('5. Определение с уточнениями')
    mod1 = build_sepsis3(build_suspicion(ab[~ab['nonsystemic']], me[~me['screening']]), sofa2)
    mod = build_sepsis3(build_suspicion(ab[~ab['nonsystemic'] & ~ab['periop_prophylaxis']], me[~me['screening']]), sofa2)
    m1_set, m_set = set(mod1['stay_id']), set(mod['stay_id'])
    st.log('step1_screening_and_nonsystemic_episodes', len(m1_set))
    st.log('step2_plus_prophylaxis_episodes', len(m_set))
    st.log('modified_patients', int(mod['subject_id'].nunique()))
    st.log('lost_vs_standard', len(s_set - m_set))
    st.log('added_vs_standard', len(m_set - s_set))
    same = std.merge(mod[['stay_id'] + TIME_COLS], on='stay_id', suffixes=('_std', '_mod'))
    st.log('retained_episodes', int(len(same)))
    st.log('retained_changed_pct', {c: round(float((same[f'{c}_std'] != same[f'{c}_mod']).mean() * 100), 1)
                                    for c in TIME_COLS})
    t0_std = same[['sofa_time_std', 'culture_time_std', 'antibiotic_time_std']].max(axis=1)
    t0_mod = same[['sofa_time_mod', 'culture_time_mod', 'antibiotic_time_mod']].max(axis=1)
    st.log('retained_t0_changed_pct', round(float((t0_std != t0_mod).mean() * 100), 1))

    lost = std[std['stay_id'].isin(s_set - m_set)].copy()
    lost['reason'] = np.where(lost['stay_id'].isin(s_set - m1_set), 'screening_or_nonsystemic', 'prophylaxis')
    lost.to_parquet(f'{DATA}/stage0_lost_episodes.parquet', index=False)
    st.log('lost_by_reason', lost['reason'].value_counts().to_dict())
    screen_share = lost['specimen'].fillna('').str.upper().str.contains('MRSA|SCREEN').mean() * 100
    st.log('lost_based_on_mrsa_or_screen_pct', round(float(screen_share), 1))
    print('Утраченные эпизоды — посев (топ-8):\n' + lost['specimen'].value_counts().head(8).to_string())
    print('Утраченные эпизоды — антибиотик (топ-8):\n' + lost['antibiotic'].value_counts().head(8).to_string())

    st.log('positive_culture_pct_standard', round(float(std['positive_culture'].mean() * 100), 1))
    st.log('positive_culture_pct_modified', round(float(mod['positive_culture'].mean() * 100), 1))
    st.log('specimen_modified_top_pct', (mod['specimen'].value_counts(normalize=True).head(5) * 100).round(1).to_dict())
    st.log('culture_dateonly_pct_modified', round(float(mod['culture_dateonly'].mean() * 100), 1))
    st.check(30_000 <= len(m_set) <= 40_000, 'размер модифицированной когорты в ожидаемом диапазоне 30–40 тыс.')
    mod.to_parquet(f'{DATA}/sepsis3_modified.parquet', index=False)
    print(f'\nСохранено: {DATA}/sepsis3_standard.parquet, sepsis3_modified.parquet, stage0_lost_episodes.parquet')
