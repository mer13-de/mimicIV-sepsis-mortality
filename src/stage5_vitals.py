# -*- coding: utf-8 -*-
"""
ЭТАП 5. Витальные показатели, ШКГ, SpO2, FiO2, SpO2/FiO2.

  * Витальные показатели — производная таблица vitals_derived (ЧСС, ЧД, температура, САД/СрАД/ДАД, SpO2);
    окно (T0 − 24 ч; T0]; последнее непустое значение, изменение (последнее − первое) и размах (максимум − минимум)
    при ≥ 2 измерениях с интервалом между первым и последним ≥ 1 ч (для всех витальных показателей, включая АД). Физиологически невозможные значения
    удаляются до агрегации. Глюкоза глюкометром не используется (глюкоза — только центральная лаборатория).
  * АД (систолическое, среднее, диастолическое) — из chartevents с приоритетом инвазивного измерения:
    в каждый момент времени берётся значение артериальной линии (220050/220051/220052, 225309/225310/225312),
    а неинвазивное (220179/220180/220181) — только если инвазивного в этот момент нет. Для каждого источника
    отбрасываются артефакты: значения вне физиологических границ и тройки, где САД ≤ ДАД или СрАД вне [ДАД; САД].
  * Инвазивный мониторинг АД за 24 ч: хотя бы одно достоверное измерение артериальной линии в окне.
  * ШКГ — из компонентов таблицы gcs (итоговый балл таблицы не используется: в нём у интубированных
    подставляется 15). Признаки модели: сумма последней оценки до T0 (если оценены все три компонента и речь
    оценима, иначе пропуск); двигательный ответ (категории 6, 5, 4, 2–3, 1) и открывание глаз — последние
    оценённые значения; речь — из последней оценки, пропуск, если речь неоценима (интубация).
  * FiO2 — только зарегистрированный (ventilator_setting); последнее значение за окно.
  * SpO2/FiO2 — только при SpO2 ≤ 97% и зарегистрированном FiO2 за 4 ч до измерения; последнее
    (не входит в модель; используется на этапе 6 только для индекса ROX).

Вход: stage1_cohort*.parquet; vitals_derived, chartevents (АД), gcs, ventilator_setting
Выход: data/stage5_vitals.parquet (одна строка на stay_id), cache_vitals_stage5.parquet, cache_bp_stage5.parquet
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

WINDOW_H = 24
FIO2_LOOKBACK_H = 4
SF_SPO2_MAX = 97
CACHE = f'{DATA}/cache_vitals_stage5.parquet'
VITALS = {  # признак: (граница нижняя, верхняя)
    'heart_rate': (20, 300), 'resp_rate': (2, 80), 'temperature': (25, 45),
    'sbp': (30, 300), 'mbp': (20, 250), 'dbp': (10, 200), 'spo2': (50, 100),
}
DELTA_VITALS = ['heart_rate', 'resp_rate', 'temperature', 'spo2', 'sbp', 'mbp', 'dbp']   # изменение и размах за окно
DELTA_MIN_H = 1                                # минимальный интервал между первым и последним измерением
BP_CACHE = f'{DATA}/cache_bp_stage5.parquet'
BP_ITEMS = {220050: ('inv', 'sbp'), 220051: ('inv', 'dbp'), 220052: ('inv', 'mbp'),
            225309: ('inv', 'sbp'), 225310: ('inv', 'dbp'), 225312: ('inv', 'mbp'),
            220179: ('ni', 'sbp'), 220180: ('ni', 'dbp'), 220181: ('ni', 'mbp')}
CE_SCHEMA = {'subject_id': 'BIGINT', 'hadm_id': 'BIGINT', 'stay_id': 'BIGINT', 'caregiver_id': 'BIGINT',
             'charttime': 'TIMESTAMP', 'storetime': 'TIMESTAMP', 'itemid': 'BIGINT', 'value': 'VARCHAR',
             'valuenum': 'DOUBLE', 'valueuom': 'VARCHAR', 'warning': 'BIGINT'}
VITALS_SCHEMA = {'subject_id': 'BIGINT', 'stay_id': 'BIGINT', 'charttime': 'TIMESTAMP', 'heart_rate': 'DOUBLE',
                 'sbp': 'DOUBLE', 'dbp': 'DOUBLE', 'mbp': 'DOUBLE', 'sbp_ni': 'DOUBLE', 'dbp_ni': 'DOUBLE',
                 'mbp_ni': 'DOUBLE', 'resp_rate': 'DOUBLE', 'temperature': 'DOUBLE', 'temperature_site': 'VARCHAR',
                 'spo2': 'DOUBLE', 'glucose': 'DOUBLE'}
H = lambda x: pd.Timedelta(hours=x)


def fio2_pct(x):
    x = pd.to_numeric(x, errors='coerce')
    x = x.where(x > 1, x * 100)
    return x.where(x.between(21, 100))


def last_min_max(df, col, prefix):
    d = df.dropna(subset=[col]).sort_values(['stay_id', 'charttime'])
    g = d.groupby('stay_id')[col]
    return pd.DataFrame({f'{prefix}_last': g.last(), f'{prefix}_min': g.min(), f'{prefix}_max': g.max()})


def last_delta(df, col, prefix, with_delta, min_h=DELTA_MIN_H):
    """Последнее значение; при with_delta — изменение (последнее − первое) и размах (максимум − минимум),
    если в окне ≥ 2 измерений и между первым и последним ≥ min_h ч."""
    d = df.dropna(subset=[col]).sort_values(['stay_id', 'charttime'])
    g = d.groupby('stay_id')
    out = pd.DataFrame({f'{prefix}_last': g[col].last()})
    if with_delta:
        span_h = (g['charttime'].max() - g['charttime'].min()).dt.total_seconds() / 3600
        ok = (g.size() >= 2) & (span_h >= min_h)
        out[f'{prefix}_delta'] = (g[col].last() - g[col].first()).where(ok)
        out[f'{prefix}_range'] = (g[col].max() - g[col].min()).where(ok)
    return out


with Stage('stage5_vitals') as st:
    st.section('1. Когорты')
    keep = ['stay_id', 't0', 'split', 'main_analysis']
    c1 = pd.read_parquet(f'{DATA}/stage1_cohort.parquet')[keep]
    c2 = pd.read_parquet(f'{DATA}/stage1_cohort_standard.parquet')[keep]
    coh = pd.concat([c1, c2]).drop_duplicates('stay_id').reset_index(drop=True)
    coh['t0'] = coh['t0'].astype('datetime64[ns]')
    coh['w0'] = coh['t0'] - H(WINDOW_H)
    feat = coh[['stay_id']].copy()
    st.log('stays', int(len(coh)))

    # ------------------------------------------------------------------
    st.section('2. Витальные показатели (vitals_derived, DuckDB, кэш)')
    if os.path.isfile(CACHE):
        vit = pd.read_parquet(CACHE)
        print(f'Кэш: {CACHE} ({len(vit):,} строк)')
    else:
        con = duckdb.connect()
        con.register('k', coh[['stay_id', 't0', 'w0']])
        p = raw('vitals_derived')
        cols = ', '.join(f"'{k}': '{v}'" for k, v in VITALS_SCHEMA.items())
        src = (f"read_csv('{p}', header=true, compression='{'gzip' if is_gzip(p) else 'none'}', "
               f"columns={{{cols}}}, quote='\"', escape='\"')")
        vit = con.execute(f"""SELECT v.stay_id, v.charttime, v.heart_rate, v.sbp, v.dbp, v.mbp, v.sbp_ni, v.mbp_ni,
                                     v.resp_rate, v.temperature, v.spo2
                              FROM {src} v JOIN k ON v.stay_id = k.stay_id
                              WHERE v.charttime > k.w0 AND v.charttime <= k.t0""").df()
        vit.to_parquet(CACHE, index=False)
        print(f'Сохранён кэш: {CACHE} ({len(vit):,} строк)')
    vit['charttime'] = vit['charttime'].astype('datetime64[ns]')
    st.log('vital_rows', int(len(vit)))

    removed = {}
    for v, (lo, hi) in VITALS.items():
        bad = vit[v].notna() & ~vit[v].between(lo, hi)
        removed[v] = int(bad.sum())
        vit.loc[bad, v] = np.nan
    st.log('removed_implausible', removed)
    for v in ['heart_rate', 'resp_rate', 'temperature', 'spo2']:
        feat = feat.merge(last_delta(vit, v, v, v in DELTA_VITALS), left_on='stay_id', right_index=True, how='left')
    for v in ['heart_rate', 'resp_rate', 'temperature', 'spo2']:
        st.log(f'{v}_delta_available_pct', round(float(feat[f'{v}_delta'].notna().mean() * 100), 1))

    # ------------------------------------------------------------------
    st.section('2б. АД с приоритетом инвазивного измерения (chartevents, кэш)')
    if os.path.isfile(BP_CACHE):
        bp = pd.read_parquet(BP_CACHE)
        print(f'Кэш: {BP_CACHE} ({len(bp):,} строк)')
    else:
        con = duckdb.connect()
        con.register('k', coh[['stay_id', 't0', 'w0']])
        p = raw('chartevents')
        cols = ', '.join(f"'{a}': '{b}'" for a, b in CE_SCHEMA.items())
        src = (f"read_csv('{p}', header=true, compression='{'gzip' if is_gzip(p) else 'none'}', "
               f"columns={{{cols}}}, quote='\"', escape='\"')")
        ids = ','.join(map(str, BP_ITEMS))
        bp = con.execute(f"""SELECT c.stay_id, c.charttime, c.itemid, c.valuenum
                             FROM {src} c JOIN k ON c.stay_id = k.stay_id
                             WHERE c.itemid IN ({ids}) AND c.valuenum IS NOT NULL
                               AND c.charttime > k.w0 AND c.charttime <= k.t0""").df()
        bp.to_parquet(BP_CACHE, index=False)
        print(f'Сохранён кэш: {BP_CACHE} ({len(bp):,} строк)')
    bp['charttime'] = bp['charttime'].astype('datetime64[ns]')
    bp['src'] = bp['itemid'].map({i: v[0] for i, v in BP_ITEMS.items()})
    bp['var'] = bp['itemid'].map({i: v[1] for i, v in BP_ITEMS.items()})
    w = bp.pivot_table(index=['stay_id', 'charttime'], columns=['src', 'var'], values='valuenum', aggfunc='median')
    w.columns = [f'{a}_{b}' for a, b in w.columns]
    w = w.reset_index()
    art_stats = {}
    for src_ in ['inv', 'ni']:
        cs = [f'{src_}_{v}' for v in ('sbp', 'mbp', 'dbp')]
        for c in cs:
            if c not in w:
                w[c] = np.nan
        n_before = int(w[cs].notna().any(axis=1).sum())
        for v in ('sbp', 'mbp', 'dbp'):
            lo, hi = VITALS[v]
            w.loc[~w[f'{src_}_{v}'].between(lo, hi), f'{src_}_{v}'] = np.nan
        sb, mb, db = w[cs[0]], w[cs[1]], w[cs[2]]
        bad = (sb.notna() & db.notna() & (sb <= db)) | \
              (sb.notna() & db.notna() & mb.notna() & ((mb < db) | (mb > sb)))
        w.loc[bad, cs] = np.nan
        art_stats[src_] = {'timepoints': n_before, 'artifact_triples_removed': int(bad.sum())}
    st.log('bp_artifacts', art_stats)
    inv_any = w[['inv_sbp', 'inv_mbp', 'inv_dbp']].notna().any(axis=1)
    for v in ('sbp', 'mbp', 'dbp'):
        w[v] = w[f'inv_{v}'].where(inv_any, w[f'ni_{v}'])     # приоритет инвазивного в каждый момент времени
    w['bp_source_inv'] = inv_any.astype(int)
    for v in ('sbp', 'mbp', 'dbp'):
        feat = feat.merge(last_delta(w, v, v, v in DELTA_VITALS), left_on='stay_id', right_index=True, how='left')
    feat['arterial_line_24h'] = feat['stay_id'].isin(set(w.loc[inv_any, 'stay_id'])).astype(int)
    last_src = w.dropna(subset=['mbp']).sort_values(['stay_id', 'charttime']).groupby('stay_id')['bp_source_inv'].last()
    st.log('arterial_line_24h_pct', round(float(feat['arterial_line_24h'].mean() * 100), 1))
    st.log('mbp_last_from_invasive_pct', round(float(last_src.mean() * 100), 1))
    old_mbp = last_min_max(vit, 'mbp', 'old')['old_last']
    cmp_ = feat.set_index('stay_id')['mbp_last'].to_frame().join(old_mbp, how='inner').dropna()
    st.log('mbp_last_new_vs_vitals_derived', {'n': int(len(cmp_)),
                                              'median_diff': round(float((cmp_['mbp_last'] - cmp_['old_last']).median()), 1),
                                              'identical_pct': round(float((cmp_['mbp_last'] - cmp_['old_last']).abs().lt(0.5).mean() * 100), 1)})
    vcols = [f'{v}_{a}' for v in VITALS for a in ('last', 'delta', 'range') if f'{v}_{a}' in feat.columns]
    feat = feat[['stay_id'] + vcols + [c for c in feat.columns if c not in ['stay_id'] + vcols]]

    # ------------------------------------------------------------------
    st.section('3. ШКГ из компонентов')
    gcs = read_csv_any(raw('gcs'), usecols=['stay_id', 'charttime', 'gcs_motor', 'gcs_verbal', 'gcs_eyes', 'gcs_unable'])
    gcs['charttime'] = pd.to_datetime(gcs['charttime'], errors='coerce').astype('datetime64[ns]')
    gcs = gcs.merge(coh[['stay_id', 't0', 'w0']], on='stay_id')
    gcs = gcs[(gcs['charttime'] > gcs['w0']) & (gcs['charttime'] <= gcs['t0'])]
    for c, (lo, hi) in {'gcs_motor': (1, 6), 'gcs_verbal': (0, 5), 'gcs_eyes': (1, 4)}.items():
        gcs.loc[~gcs[c].between(lo, hi), c] = np.nan
    gcs['verbal_unable'] = ((gcs['gcs_verbal'] == 0) | (gcs['gcs_unable'] == 1)).astype(int)
    gcs = gcs[gcs[['gcs_motor', 'gcs_verbal', 'gcs_eyes']].notna().any(axis=1) | (gcs['verbal_unable'] == 1)]
    full = gcs[['gcs_motor', 'gcs_verbal', 'gcs_eyes']].notna().all(axis=1) & (gcs['verbal_unable'] == 0)
    gcs['gcs_total'] = (gcs['gcs_motor'] + gcs['gcs_verbal'] + gcs['gcs_eyes']).where(full)
    last = gcs.sort_values(['stay_id', 'charttime']).groupby('stay_id').last()   # последнее непустое по каждому столбцу
    lastrow = gcs.sort_values(['stay_id', 'charttime']).drop_duplicates('stay_id', keep='last').set_index('stay_id')
    feat['gcs_total_last'] = feat['stay_id'].map(lastrow['gcs_total'])   # сумма последней оценки (если оценима)
    feat['gcs_motor_last'] = feat['stay_id'].map(last['gcs_motor'].replace({3: 2}))   # категории 6, 5, 4, 2–3, 1
    feat['gcs_eyes_last'] = feat['stay_id'].map(last['gcs_eyes'])
    # Речь — только из последней оценки и только если оценима (иначе пропуск: не подставляем балл до интубации)
    feat['gcs_verbal_last'] = feat['stay_id'].map(lastrow['gcs_verbal'].where(lastrow['verbal_unable'] == 0))
    # Диагностика (не признаки модели): была ли хоть какая-то оценка и почему суммы нет
    diag = coh[['stay_id']].copy()
    diag['gcs_any_assessment'] = diag['stay_id'].isin(set(gcs['stay_id'])).astype(int)
    diag['gcs_verbal_unable_last'] = diag['stay_id'].map(lastrow['verbal_unable'])
    miss = diag.assign(total=diag['stay_id'].map(feat.set_index('stay_id')['gcs_total_last']))
    miss = miss[miss['total'].isna()]
    st.log('gcs', {'any_assessment_pct': round(float(diag['gcs_any_assessment'].mean() * 100), 1),
                   'total_last_pct': round(float(feat['gcs_total_last'].notna().mean() * 100), 1),
                   'total_missing_reason_pct': {
                       'not_assessed_in_window': round(float((miss['gcs_any_assessment'] == 0).mean() * 100), 1),
                       'verbal_unable_last': round(float((miss['gcs_verbal_unable_last'] == 1).mean() * 100), 1)}})

    # ------------------------------------------------------------------
    st.section('4. FiO2 и SpO2/FiO2')
    vs = read_csv_any(raw('ventilator_setting'), usecols=['stay_id', 'charttime', 'fio2'])
    vs['charttime'] = pd.to_datetime(vs['charttime'], errors='coerce').astype('datetime64[ns]')
    vs['fio2'] = fio2_pct(vs['fio2'])
    vs = vs.dropna(subset=['charttime', 'fio2']).merge(coh[['stay_id', 't0', 'w0']], on='stay_id')
    vs_win = vs[(vs['charttime'] > vs['w0']) & (vs['charttime'] <= vs['t0'])]
    feat['fio2_last'] = feat['stay_id'].map(vs_win.sort_values('charttime').groupby('stay_id')['fio2'].last())
    sp = vit.loc[vit['spo2'].notna() & (vit['spo2'] <= SF_SPO2_MAX), ['stay_id', 'charttime', 'spo2']].sort_values('charttime')
    vsa = vs[['stay_id', 'charttime', 'fio2']].rename(columns={'charttime': 'fio2_time'}).sort_values('fio2_time')
    sp = pd.merge_asof(sp, vsa, left_on='charttime', right_on='fio2_time', by='stay_id', direction='backward',
                       tolerance=H(FIO2_LOOKBACK_H))
    sp['sf'] = sp['spo2'] / (sp['fio2'] / 100)
    sf = sp.dropna(subset=['sf']).sort_values(['stay_id', 'charttime']).groupby('stay_id')['sf']
    feat['sf_ratio_last'] = feat['stay_id'].map(sf.last())
    st.log('sf_pairs', {'spo2_le_97': int(len(sp)), 'with_fio2': int(sp['fio2'].notna().sum())})

    # ------------------------------------------------------------------
    st.section('5. Доступность и сохранение')
    feat.to_parquet(f'{DATA}/stage5_vitals.parquet', index=False)
    print(f'Сохранено: {DATA}/stage5_vitals.parquet ({len(feat):,} строк, {feat.shape[1] - 1} признаков)')
    m = feat.merge(c1[['stay_id', 'split', 'main_analysis']], on='stay_id').merge(diag, on='stay_id')
    m = m[m['main_analysis'] == 1]
    show = [f'{v}_last' for v in VITALS] + [f'{v}_{a}' for v in DELTA_VITALS for a in ('delta', 'range')] + \
           ['gcs_any_assessment', 'gcs_total_last', 'gcs_motor_last', 'gcs_eyes_last', 'gcs_verbal_last',
            'fio2_last', 'sf_ratio_last']
    m['gcs_any_assessment'] = m['gcs_any_assessment'].where(m['gcs_any_assessment'] == 1)   # 0 -> пропуск для доступности
    avail = pd.DataFrame({c: m.groupby('split')[c].apply(lambda s: s.notna().mean() * 100) for c in show}).T.round(1)
    print('Доступность за 24 ч до T0 (основной анализ), %:\n' + avail.to_string())
    st.log('availability_pct', avail.to_dict(orient='index'))
    med = {c: round(float(m[c].median()), 2) for c in show if c != 'gcs_any_assessment'}
    st.log('median_main', med)
    by = m.groupby('split')[['arterial_line_24h', 'gcs_verbal_unable_last']].mean().mul(100).round(1)
    print('Инвазивное АД и неоценимая речь по периодам, %:\n' + by.to_string())

    st.check(feat['stay_id'].is_unique, 'одна строка на ICU-пребывание')
    for c in ['heart_rate_last', 'mbp_last', 'resp_rate_last', 'spo2_last']:
        st.check(avail.loc[c].min() >= 90, f'{c}: доступен у ≥ 90% в обучении и тесте')
    st.check(avail.loc['gcs_any_assessment'].min() >= 80, 'ШКГ (хотя бы один компонент) оценивалась в окне у ≥ 80%')
    st.check(70 <= med['heart_rate_last'] <= 120, 'медиана ЧСС правдоподобна (70–120)')
    st.check(60 <= med['mbp_last'] <= 95, 'медиана СрАД правдоподобна (60–95)')
    st.check(36 <= med['temperature_last'] <= 38.5, 'медиана температуры правдоподобна (36–38,5 °C)')
    st.check(all(abs(avail.loc[c, 'train'] - avail.loc[c, 'test']) < 10 for c in show
                 if not c.endswith(('_delta', '_range')) and avail.loc[c, 'train'] >= 50),
             'доступность основных показателей в обучении и тесте различается < 10 п. п.')
