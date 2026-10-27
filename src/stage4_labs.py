# -*- coding: utf-8 -*-
"""
ЭТАП 4. Лабораторные показатели, газы крови и PaO2/FiO2.

  * Источник — исходная таблица labevents (включая анализы приёмного отделения: отбор по subject_id и времени,
    а не по hadm_id). Окно (T0 − 24 ч; T0] по времени взятия образца (charttime); кроме того, результат должен
    быть готов к T0: coalesce(storetime, charttime) ≤ T0 (результаты, известные только после T0, не используются).
  * Электролиты, гемоглобин и глюкоза — только центральная лаборатория (коды анализаторов газов крови
    для них не используются). Лактат — все три кода.
  * Коды проверяются по d_labitems (ожидаемое название); единицы приводятся к одной на показатель,
    значения вида «<0.01» разбираются как число.
  * Первый уровень границ: физиологически невозможные значения удаляются ДО агрегации
    (второй уровень — винзоризация по обучающему периоду — на этапе 6).
  * Абсолютные нейтрофилы, лимфоциты, моноциты и эозинофилы: измеренное значение, иначе лейкоциты × доля из того же образца
    (вывод абсолютных значений менялся во времени). Палочкоядерные (ручной подсчёт, доступность упала вдвое)
    и тропонин I (в окне практически не определялся; в госпитале используется тропонин T) исключены.
  * Агрегаты: последнее значение за окно; для лактата, креатинина и тромбоцитов — ещё изменение (дельта)
    (последнее − первое) при ≥ 2 измерениях с интервалом между первым и последним ≥ 4 ч.
  * pH, PaO2 и PaCO2 — только артериальные образцы (тип образца 52033 = ART). PaO2/FiO2 — только при
    зарегистрированном FiO2: из того же образца (50816), иначе последнее FiO2 из ventilator_setting за 4 ч до
    взятия; пара отбрасывается, если PaO2 больше альвеолярного PAO2 + 10 мм рт. ст. (уравнение альвеолярного газа).

Вход: stage1_cohort*.parquet; labevents, d_labitems, ventilator_setting
Выход: data/stage4_labs.parquet (одна строка на stay_id), cache_labevents_stage4_<хэш кодов>.parquet
"""
import os
import sys
import re
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
DELTA_MIN_H = 4          # минимальный интервал между первым и последним измерением для расчёта изменения
FIO2_LOOKBACK_H = 4
PAO2_TOL = 10.0
SPECIMEN_TYPE = 52033
FIO2_LAB = 50816

# признак: (itemid, регулярное выражение для названия в d_labitems, {единица (нижний регистр): множитель},
#           (нижняя, верхняя граница), агрегаты)
LABS = {
    'wbc':         ([51300, 51301], r'wbc|white blood', {'k/ul': 1}, (0, 500), ['last']),
    'hemoglobin':  ([51222], r'^hemoglobin$', {'g/dl': 1}, (1, 25), ['last']),
    'platelets':   ([51265], r'platelet', {'k/ul': 1}, (1, 2000), ['last', 'delta']),
    'rdw':         ([51277], r'^rdw$', {'%': 1}, (8, 40), ['last']),
    'neut_abs':    ([52075], r'absolute neutrophil', {'k/ul': 1, '#/ul': 0.001}, (0, 300), ['last']),
    'lymph_abs':   ([51133], r'absolute lymphocyte', {'k/ul': 1, '#/ul': 0.001}, (0, 300), ['last']),
    'neut_pct':    ([51256], r'^neutrophils$', {'%': 1}, (0, 100), ['last']),
    'lymph_pct':   ([51244], r'^lymphocytes$', {'%': 1}, (0, 100), ['last']),
    'mono_abs':    ([52074], r'absolute monocyte', {'k/ul': 1, '#/ul': 0.001}, (0, 100), ['last']),
    'eos_abs':     ([52073], r'absolute eosinophil', {'k/ul': 1, '#/ul': 0.001}, (0, 100), ['last']),
    'mono_pct':    ([51254], r'^monocytes$', {'%': 1}, (0, 100), ['last']),
    'eos_pct':     ([51200], r'^eosinophils$', {'%': 1}, (0, 100), ['last']),
    'mcv':         ([51250], r'^mcv$', {'fl': 1}, (40, 150), ['last']),
    'sodium':      ([50983], r'^sodium$', {'meq/l': 1, 'mmol/l': 1}, (90, 200), ['last']),
    'potassium':   ([50971], r'^potassium$', {'meq/l': 1, 'mmol/l': 1}, (1, 12), ['last']),
    'chloride':    ([50902], r'^chloride$', {'meq/l': 1, 'mmol/l': 1}, (50, 160), ['last']),
    'bicarbonate': ([50882], r'^bicarbonate$', {'meq/l': 1, 'mmol/l': 1}, (2, 60), ['last']),
    'bun':         ([51006], r'urea nitrogen', {'mg/dl': 1}, (1, 300), ['last']),
    'creatinine':  ([50912], r'^creatinine$', {'mg/dl': 1}, (0.1, 30), ['last', 'delta']),
    'glucose':     ([50931], r'^glucose$', {'mg/dl': 1}, (10, 2500), ['last']),
    'calcium':     ([50893], r'calcium, total', {'mg/dl': 1}, (2, 20), ['last']),
    'aniongap':    ([50868], r'anion gap', {'meq/l': 1, 'mmol/l': 1}, (0, 60), ['last']),
    'albumin':     ([50862], r'^albumin$', {'g/dl': 1}, (0.5, 7), ['last']),
    'magnesium':   ([50960], r'^magnesium$', {'mg/dl': 1}, (0.2, 15), ['last']),
    'phosphate':   ([50970], r'^phosphate$', {'mg/dl': 1}, (0.2, 25), ['last']),
    'inr':         ([51237], r'inr', {'': 1}, (0.5, 25), ['last']),
    'ptt':         ([51275], r'^ptt$', {'sec': 1}, (10, 200), ['last']),
    'fibrinogen':  ([51214], r'fibrinogen', {'mg/dl': 1}, (20, 2000), ['last']),
    'alt':         ([50861], r'alanine aminotransferase', {'iu/l': 1, 'u/l': 1}, (1, 20000), ['last']),
    'ast':         ([50878], r'asparate aminotransferase|aspartate aminotransferase', {'iu/l': 1, 'u/l': 1}, (1, 30000), ['last']),
    'alp':         ([50863], r'alkaline phosphatase', {'iu/l': 1, 'u/l': 1}, (5, 5000), ['last']),
    'bilirubin':   ([50885], r'bilirubin, total', {'mg/dl': 1}, (0.05, 80), ['last']),
    'ldh':         ([50954], r'lactate dehydrogenase', {'iu/l': 1, 'u/l': 1}, (20, 30000), ['last']),
    'troponin_t':  ([51003], r'troponin t', {'ng/ml': 1, 'ng/l': 0.001}, (0, 30), ['last']),
    'lactate':     ([50813, 52442, 53154], r'^lactate$', {'mmol/l': 1}, (0.1, 35), ['last', 'delta']),
    'ph':          ([50820], r'^ph$', {'units': 1, '': 1}, (6.5, 8.0), ['last']),
    'base_excess': ([50802], r'base excess', {'meq/l': 1, 'mmol/l': 1}, (-40, 40), ['last']),
    'ionized_ca':  ([50808], r'free calcium', {'mmol/l': 1}, (0.3, 3), ['last']),
    'pao2':        ([50821], r'^po2$', {'mm hg': 1, 'mmhg': 1}, (15, 700), ['last']),
    'paco2':       ([50818], r'^pco2$', {'mm hg': 1, 'mmhg': 1}, (5, 200), ['last']),
}
ARTERIAL_ONLY = {'ph', 'pao2', 'paco2'}
ITEM2FEAT = {i: f for f, v in LABS.items() for i in v[0]}
EXCLUDED_IDS = [51144, 51002, 52642]  # палочкоядерные, тропонин I: извлекаются (общий кэш), но не используются
ALL_IDS = sorted(list(ITEM2FEAT) + [SPECIMEN_TYPE, FIO2_LAB] + EXCLUDED_IDS)
import zlib
# новый список кодов -> новый кэш; суффикс _st — кэш со столбцом storetime (старый кэш его не содержит)
CACHE = f"{DATA}/cache_labevents_stage4_{zlib.crc32(','.join(map(str, ALL_IDS)).encode()):08x}_st.parquet"
LAB_SCHEMA = {'labevent_id': 'BIGINT', 'subject_id': 'BIGINT', 'hadm_id': 'BIGINT', 'specimen_id': 'BIGINT',
              'itemid': 'BIGINT', 'order_provider_id': 'VARCHAR', 'charttime': 'TIMESTAMP', 'storetime': 'TIMESTAMP',
              'value': 'VARCHAR', 'valuenum': 'DOUBLE', 'valueuom': 'VARCHAR', 'ref_range_lower': 'DOUBLE',
              'ref_range_upper': 'DOUBLE', 'flag': 'VARCHAR', 'priority': 'VARCHAR', 'comments': 'VARCHAR'}
NUM_RE = re.compile(r'^\s*(?:[<>]=?|less than|greater than)?\s*(-?\d+(?:\.\d+)?)\s*$', re.I)
H = lambda x: pd.Timedelta(hours=x)


def parse_value(valuenum, value):
    """valuenum, а при его отсутствии — число из текстового значения («<0.01», «>70»)."""
    parsed = pd.to_numeric(value.astype('string').str.extract(NUM_RE, expand=False), errors='coerce')
    return valuenum.astype('float64').fillna(parsed.astype('float64'))


def fio2_pct(x):
    x = pd.to_numeric(x, errors='coerce')
    x = x.where(x > 1, x * 100)
    return x.where(x.between(21, 100))


with Stage('stage4_labs') as st:
    st.section('1. Когорты и справочник кодов')
    keep = ['stay_id', 'subject_id', 't0', 'split', 'main_analysis']
    c1 = pd.read_parquet(f'{DATA}/stage1_cohort.parquet')[keep]
    c2 = pd.read_parquet(f'{DATA}/stage1_cohort_standard.parquet')[keep]
    coh = pd.concat([c1, c2]).drop_duplicates('stay_id').reset_index(drop=True)
    coh['w0'] = coh['t0'] - H(WINDOW_H)
    st.log('stays', int(len(coh)))

    dl = read_csv_any(raw('d_labitems'))
    dl = dl.set_index('itemid')
    bad_label = []
    for f, (ids, pat, *_ ) in LABS.items():
        for i in ids:
            lab = str(dl['label'].get(i, '<нет кода>'))
            fl = str(dl['fluid'].get(i, ''))
            print(f'  {f:12s} {i:6d}  {lab} [{fl}]')
            if not re.search(pat, lab.lower()) or fl.lower() != 'blood':
                bad_label.append((f, i, lab, fl))
    for i in (SPECIMEN_TYPE, FIO2_LAB):
        print(f'  {"служебный":12s} {i:6d}  {dl["label"].get(i, "<нет кода>")}')
    st.check(not bad_label, f'все коды найдены в d_labitems с ожидаемым названием (несовпадения: {bad_label})')

    # ------------------------------------------------------------------
    st.section('2. Выборка из labevents (один проход DuckDB, кэш)')
    if os.path.isfile(CACHE):
        lab = pd.read_parquet(CACHE)
        print(f'Кэш: {CACHE} ({len(lab):,} строк)')
    else:
        con = duckdb.connect()
        con.register('k', coh[['stay_id', 'subject_id', 't0', 'w0']])
        p = raw('labevents')
        cols = ', '.join(f"'{k}': '{v}'" for k, v in LAB_SCHEMA.items())
        src = (f"read_csv('{p}', header=true, compression='{'gzip' if is_gzip(p) else 'none'}', "
               f"columns={{{cols}}}, quote='\"', escape='\"')")
        ids = ','.join(map(str, ALL_IDS))
        lab = con.execute(f"""SELECT k.stay_id, l.specimen_id, l.itemid, l.charttime, l.storetime,
                                     l.value, l.valuenum, l.valueuom
                              FROM {src} l JOIN k ON l.subject_id = k.subject_id
                              WHERE l.itemid IN ({ids}) AND l.charttime > k.w0 AND l.charttime <= k.t0""").df()
        lab.to_parquet(CACHE, index=False)
        print(f'Сохранён кэш: {CACHE} ({len(lab):,} строк)')
    lab['charttime'] = lab['charttime'].astype('datetime64[ns]')
    lab['storetime'] = lab['storetime'].astype('datetime64[ns]')
    st.log('lab_rows_charttime_window', int(len(lab)))
    # Результат должен быть известен к T0
    t0_of = coh.set_index('stay_id')['t0'].astype('datetime64[ns]')
    ready = lab['storetime'].fillna(lab['charttime']) <= lab['stay_id'].map(t0_of)
    st.log('lab_rows_not_ready_by_t0_pct', round(float((~ready).mean() * 100), 2))
    lab = lab[ready].copy()
    st.log('lab_rows', int(len(lab)))

    # ------------------------------------------------------------------
    st.section('3. Тип образца газов крови, единицы, границы')
    spec = lab[lab['itemid'] == SPECIMEN_TYPE].drop_duplicates('specimen_id').set_index('specimen_id')['value']
    spec = spec.astype(str).str.upper().str.strip()
    st.log('specimen_type_pct', spec.value_counts(normalize=True).mul(100).round(1).head(6).to_dict())
    arterial = set(spec[spec.str.startswith('ART')].index)
    fio2_lab = lab[lab['itemid'] == FIO2_LAB].copy()

    x = lab[lab['itemid'].isin(ITEM2FEAT)].copy()
    x['feat'] = x['itemid'].map(ITEM2FEAT)
    x['val'] = parse_value(x['valuenum'], x['value'])
    x['uom'] = x['valueuom'].fillna('').str.lower().str.strip()
    st.log('parsed_from_text', int((x['valuenum'].isna() & x['val'].notna()).sum()))
    print('Единицы по показателям:\n' + x.groupby(['feat', 'uom']).size().to_string())
    unconv = {}
    x['factor'] = np.nan
    for f, (ids, pat, conv, (lo, hi), aggs) in LABS.items():
        m = x['feat'] == f
        fac = x.loc[m, 'uom'].map(conv)
        # пустая единица трактуется как основная (первая в списке, множитель 1)
        if '' not in conv:
            fac = fac.where(x.loc[m, 'uom'] != '', 1.0)
        x.loc[m, 'factor'] = fac
        n = int(m.sum())
        unconv[f] = round(float(fac.isna().sum() / n * 100), 2) if n else 0.0
    bad_units = {f: v for f, v in unconv.items() if v > 1}
    st.log('unconverted_units_pct_gt1', bad_units)
    x['val'] = x['val'] * x['factor']
    x = x.dropna(subset=['val'])
    # артериальные показатели
    art = x['feat'].isin(ARTERIAL_ONLY)
    st.log('bg_pao2_paco2_arterial_pct', round(float(x.loc[art, 'specimen_id'].isin(arterial).mean() * 100), 1))
    x = x[~art | x['specimen_id'].isin(arterial)]
    # первый уровень границ
    lo = x['feat'].map({f: v[3][0] for f, v in LABS.items()})
    hi = x['feat'].map({f: v[3][1] for f, v in LABS.items()})
    out = ~x['val'].between(lo, hi)
    st.log('removed_implausible', x[out].groupby('feat').size().to_dict())
    x = x[~out]
    # Абсолютные нейтрофилы/лимфоциты: автоматический вывод абсолютных значений вводился постепенно
    # (27% в обучении против 68% в тесте при одинаковой доступности лейкоформулы), поэтому при отсутствии
    # измеренного абсолютного значения оно берётся как лейкоциты × доля / 100 из того же образца.
    key = ['stay_id', 'specimen_id']
    wbc_s = x[x['feat'] == 'wbc'].drop_duplicates(key).set_index(key)['val']
    added = {}
    for pct_f, abs_f in [('neut_pct', 'neut_abs'), ('lymph_pct', 'lymph_abs'),
                         ('mono_pct', 'mono_abs'), ('eos_pct', 'eos_abs')]:
        have = set(map(tuple, x.loc[x['feat'] == abs_f, key].drop_duplicates().values))
        pr = x[x['feat'] == pct_f].drop_duplicates(key)
        pr = pr[[tuple(r) not in have for r in pr[key].values]]
        w = pd.Series(list(map(tuple, pr[key].values)), index=pr.index).map(wbc_s)
        new_rows = pr.assign(feat=abs_f, val=pr['val'] * w / 100).dropna(subset=['val'])
        added[abs_f] = int(len(new_rows))
        x = pd.concat([x, new_rows], ignore_index=True)
    st.log('abs_counts_from_wbc_x_pct', added)

    # ------------------------------------------------------------------
    st.section('4. Агрегация за окно')
    x = x.sort_values(['stay_id', 'feat', 'charttime'])
    g = x.groupby(['stay_id', 'feat'])
    agg = pd.DataFrame({'last': g['val'].last(), 'first': g['val'].first(), 'n': g.size(),
                        'span_h': (g['charttime'].max() - g['charttime'].min()).dt.total_seconds() / 3600})
    agg['delta'] = (agg['last'] - agg['first']).where((agg['n'] >= 2) & (agg['span_h'] >= DELTA_MIN_H))
    feat = coh[['stay_id']].copy()
    for f, (*_, aggs) in LABS.items():
        a = agg.xs(f, level='feat') if f in agg.index.get_level_values('feat') else pd.DataFrame(columns=agg.columns)
        for how in aggs:
            feat[f'{f}_{how}'] = feat['stay_id'].map(a[how]).astype(float)

    # ------------------------------------------------------------------
    st.section('5. PaO2/FiO2')
    bgp = x[x['feat'] == 'pao2'][['stay_id', 'specimen_id', 'charttime', 'val']].rename(columns={'val': 'pao2'})
    pc = x[x['feat'] == 'paco2'].drop_duplicates('specimen_id').set_index('specimen_id')['val']
    bgp['paco2'] = bgp['specimen_id'].map(pc)
    fl = fio2_lab.assign(f=fio2_pct(parse_value(fio2_lab['valuenum'], fio2_lab['value'])))
    bgp['fio2'] = bgp['specimen_id'].map(fl.dropna(subset=['f']).drop_duplicates('specimen_id').set_index('specimen_id')['f'])
    bgp['fio2_source'] = np.where(bgp['fio2'].notna(), 'same_specimen', None)
    vs = read_csv_any(raw('ventilator_setting'), usecols=['subject_id', 'charttime', 'fio2'])
    vs['charttime'] = pd.to_datetime(vs['charttime'], errors='coerce').astype('datetime64[ns]')
    vs['fio2'] = fio2_pct(vs['fio2'])
    vs = vs.dropna(subset=['charttime', 'fio2'])
    sid = coh.set_index('stay_id')['subject_id']
    bgp['subject_id'] = bgp['stay_id'].map(sid)
    vs = vs[vs['subject_id'].isin(set(bgp['subject_id']))].sort_values('charttime')
    bgp = bgp.sort_values('charttime')
    bgp = pd.merge_asof(bgp, vs.rename(columns={'fio2': 'fio2_vs', 'charttime': 'vs_time'}),
                        left_on='charttime', right_on='vs_time', by='subject_id', direction='backward',
                        tolerance=H(FIO2_LOOKBACK_H))
    use_vs = bgp['fio2'].isna() & bgp['fio2_vs'].notna()
    bgp.loc[use_vs, 'fio2_source'] = 'ventilator_setting_4h'
    bgp['fio2'] = bgp['fio2'].fillna(bgp['fio2_vs'])
    PAO2_alv = bgp['fio2'] / 100 * (760 - 47) - bgp['paco2'].fillna(40) / 0.8
    implaus = bgp['fio2'].notna() & (bgp['pao2'] > PAO2_alv + PAO2_TOL)
    st.log('pf_pairs', {'arterial_pao2': int(len(bgp)), 'with_fio2': int(bgp['fio2'].notna().sum()),
                        'fio2_source': bgp['fio2_source'].value_counts().to_dict(),
                        'rejected_alveolar_gas': int(implaus.sum())})
    bgp['pf'] = (bgp['pao2'] / (bgp['fio2'] / 100)).where(bgp['fio2'].notna() & ~implaus)
    pf = bgp.dropna(subset=['pf']).sort_values(['stay_id', 'charttime']).groupby('stay_id')['pf']
    feat['pf_ratio_last'] = feat['stay_id'].map(pf.last())

    # ------------------------------------------------------------------
    st.section('6. Доступность и сохранение')
    feat.to_parquet(f'{DATA}/stage4_labs.parquet', index=False)
    print(f'Сохранено: {DATA}/stage4_labs.parquet ({len(feat):,} строк, {feat.shape[1] - 1} признаков)')
    m = feat.merge(c1[['stay_id', 'split', 'main_analysis']], on='stay_id')
    m = m[m['main_analysis'] == 1]
    first_col = {f: f'{f}_{aggs[0]}' for f, (*_, aggs) in LABS.items()}
    first_col['pf_ratio'] = 'pf_ratio_last'
    avail = pd.DataFrame({f: m.groupby('split')[c].apply(lambda s: s.notna().mean() * 100)
                          for f, c in first_col.items()}).T.round(1)
    print('Доступность за 24 ч до T0 (основной анализ), %:\n' + avail.to_string())
    st.log('availability_pct', avail.to_dict(orient='index'))
    med = {f: round(float(m[c].median()), 3) for f, c in first_col.items()}
    st.log('median_main', med)

    st.check(feat['stay_id'].is_unique, 'одна строка на ICU-пребывание')
    st.check(not bad_units, 'единицы однородны: не пересчитано не более 1% значений каждого показателя')
    for f in ['creatinine', 'wbc', 'platelets', 'hemoglobin']:
        st.check(avail.loc[f].min() >= 75, f'{f}: доступен у ≥ 75% в обучении и тесте')
    drift = {f: float(avail.loc[f, 'train'] - avail.loc[f, 'test']) for f in avail.index
             if avail.loc[f, 'train'] >= 30}
    st.log('availability_drop_train_minus_test_pp', {k: round(v, 1) for k, v in drift.items() if abs(v) >= 10})
    st.check(0.6 <= med['creatinine'] <= 2.5, 'медиана креатинина правдоподобна (0,6–2,5 мг/дл)')
    st.check(1.0 <= med['lactate'] <= 4.0, 'медиана лактата правдоподобна (1–4 ммоль/л)')
    st.check(7.25 <= med['ph'] <= 7.45, 'медиана pH правдоподобна (7,25–7,45)')
    st.check(implaus.sum() <= 0.10 * max(bgp['fio2'].notna().sum(), 1),
             'пар PaO2/FiO2, противоречащих уравнению альвеолярного газа, ≤ 10%')
