# -*- coding: utf-8 -*-
"""
ЭТАП 3. Органная поддержка на T0 и за 24 ч до T0.

  * Дыхательная поддержка — производная таблица ventilation (без дрейфа по периодам): статус в T0 по иерархии
    ИВЛ > трахеостома > НИВЛ > высокопоточная оксигенотерапия > кислород > нет; трахеостома уточняется по режиму
    вентилятора за 4 ч до T0; часы ИВЛ за 24 ч.
  * Вазопрессоры, инотропы, седация — из ИСХОДНОЙ таблицы inputevents по всем кодам (производная vasoactive_agent
    не учитывает новые коды фенилэфрина и адреналина). Строки со статусом Rewritten исключаются.
    Дозы приводятся к мкг/кг/мин (вазопрессин — ЕД/ч, ангиотензин II — нг/кг/мин).
    Норадреналиновый эквивалент (NEE) = норадреналин + адреналин + фенилэфрин/10 + дофамин/100 + вазопрессин(ЕД/ч)×2,5/60;
    считается по временным отрезкам с учётом одновременного введения: NEE в T0, максимум и часы за 24 ч.
  * ЗПТ за 24 ч — производная rrt.
  * Темп диуреза за 6 и 24 ч — объём (urine_output_rate.uo) / масса / длительность окна; окно не раньше поступления
    в ОРИТ; не рассчитывается при окне < 2 ч.

Вход: stage1_cohort*.parquet, stage2_baseline.parquet; ventilation, ventilator_setting, inputevents, rrt, urine_output_rate
Выход: data/stage3_support.parquet (одна строка на stay_id)
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
TRACH_LOOKBACK_H = 4
UO_MIN_OBS_H = 2
DRUGS = {  # itemid -> препарат
    221906: 'norepinephrine',
    221289: 'epinephrine', 229617: 'epinephrine',
    221749: 'phenylephrine', 229630: 'phenylephrine', 229632: 'phenylephrine',
    222315: 'vasopressin', 221662: 'dopamine',
    229709: 'angiotensin2', 229764: 'angiotensin2',
    221653: 'dobutamine', 221986: 'milrinone',
    222168: 'propofol', 221668: 'midazolam', 225150: 'dexmedetomidine', 229420: 'dexmedetomidine',
    221744: 'fentanyl', 225942: 'fentanyl',
}
PRESSORS = ['norepinephrine', 'epinephrine', 'phenylephrine', 'vasopressin', 'dopamine', 'angiotensin2']
INOTROPES = ['dobutamine', 'milrinone']
SEDATIVES = ['propofol', 'midazolam', 'dexmedetomidine', 'fentanyl']
NEE_FACTOR = {'norepinephrine': 1.0, 'epinephrine': 1.0, 'phenylephrine': 0.1, 'dopamine': 0.01,
              'vasopressin': 2.5 / 60}           # вазопрессин в ЕД/ч
INPUT_SCHEMA = {'subject_id': 'BIGINT', 'hadm_id': 'BIGINT', 'stay_id': 'BIGINT', 'caregiver_id': 'BIGINT',
                'starttime': 'TIMESTAMP', 'endtime': 'TIMESTAMP', 'storetime': 'TIMESTAMP', 'itemid': 'BIGINT',
                'amount': 'DOUBLE', 'amountuom': 'VARCHAR', 'rate': 'DOUBLE', 'rateuom': 'VARCHAR',
                'orderid': 'BIGINT', 'linkorderid': 'BIGINT', 'ordercategoryname': 'VARCHAR',
                'secondaryordercategoryname': 'VARCHAR', 'ordercomponenttypedescription': 'VARCHAR',
                'ordercategorydescription': 'VARCHAR', 'patientweight': 'DOUBLE', 'totalamount': 'DOUBLE',
                'totalamountuom': 'VARCHAR', 'isopenbag': 'BIGINT', 'continueinnextdept': 'BIGINT',
                'statusdescription': 'VARCHAR', 'originalamount': 'DOUBLE', 'originalrate': 'DOUBLE'}
H = lambda x: pd.Timedelta(hours=x)


def std_rate(drug, rate, uom, weight):
    """Скорость в стандартных единицах: мкг/кг/мин; вазопрессин — ЕД/ч; ангиотензин II — нг/кг/мин."""
    u = uom.fillna('').str.lower().str.strip()
    w = weight.where(weight.between(20, 400))
    out = pd.Series(np.nan, index=rate.index)
    if drug == 'vasopressin':
        out = np.where(u == 'units/hour', rate, np.where(u == 'units/min', rate * 60, np.nan))
    elif drug == 'angiotensin2':
        out = np.select([u == 'ng/kg/min', u == 'mcg/kg/min', u == 'ng/min'], [rate, rate * 1000, rate / w], np.nan)
    else:
        out = np.select([u == 'mcg/kg/min', u == 'mg/kg/min', u == 'mcg/min', u == 'mg/min',
                         u == 'mcg/kg/hour', u == 'mg/kg/hour', u == 'mcg/hour', u == 'mg/hour'],
                        [rate, rate * 1000, rate / w, rate * 1000 / w,
                         rate / 60, rate * 1000 / 60, rate / w / 60, rate * 1000 / w / 60], np.nan)
    return pd.Series(out, index=rate.index, dtype=float)


with Stage('stage3_support') as st:
    st.section('1. Когорты')
    keep = ['stay_id', 't0', 'intime', 'est_year', 'split', 'main_analysis', 'death_30d']
    c1 = pd.read_parquet(f'{DATA}/stage1_cohort.parquet')[keep]
    c2 = pd.read_parquet(f'{DATA}/stage1_cohort_standard.parquet')[keep]
    coh = pd.concat([c1, c2]).drop_duplicates('stay_id').reset_index(drop=True)
    coh['w0'] = coh['t0'] - H(WINDOW_H)
    weight = pd.read_parquet(f'{DATA}/stage2_baseline.parquet').set_index('stay_id')['weight_kg']
    feat = coh[['stay_id']].copy()
    st.log('stays', int(len(coh)))
    con = duckdb.connect()
    con.register('k', coh[['stay_id', 't0', 'w0']])

    # ------------------------------------------------------------------
    st.section('2. Дыхательная поддержка')
    vent = read_csv_any(raw('ventilation'))
    for c in ['starttime', 'endtime']:
        vent[c] = pd.to_datetime(vent[c], errors='coerce')
    vent = vent.merge(coh[['stay_id', 't0', 'w0']], on='stay_id')
    PRIO = {'InvasiveVent': 1, 'Tracheostomy': 2, 'NonInvasiveVent': 3, 'HFNC': 4, 'SupplementalOxygen': 5}
    LABEL = {'InvasiveVent': 'invasive', 'Tracheostomy': 'tracheostomy', 'NonInvasiveVent': 'noninvasive',
             'HFNC': 'hfnc', 'SupplementalOxygen': 'oxygen'}
    at = vent[(vent['starttime'] <= vent['t0']) & (vent['endtime'] > vent['t0']) & vent['ventilation_status'].isin(PRIO)]
    at = at.assign(p=at['ventilation_status'].map(PRIO)).sort_values('p').drop_duplicates('stay_id')
    feat['resp_support_t0'] = feat['stay_id'].map(at.set_index('stay_id')['ventilation_status'].map(LABEL)).fillna('none')
    trach = set(feat.loc[feat['resp_support_t0'] == 'tracheostomy', 'stay_id'])
    if trach:
        vs = read_csv_any(raw('ventilator_setting'), usecols=['stay_id', 'charttime', 'ventilator_mode', 'ventilator_mode_hamilton'])
        vs['charttime'] = pd.to_datetime(vs['charttime'], errors='coerce')
        vs = vs[vs['stay_id'].isin(trach)].merge(coh[['stay_id', 't0']], on='stay_id')
        vs = vs[(vs['charttime'] <= vs['t0']) & (vs['charttime'] >= vs['t0'] - H(TRACH_LOOKBACK_H))]
        mode = vs['ventilator_mode'].fillna(vs['ventilator_mode_hamilton']).astype(str).str.upper()
        on_vent = set(vs.loc[vs['ventilator_mode'].notna() | vs['ventilator_mode_hamilton'].notna(), 'stay_id']) - \
                  set(vs.loc[mode.str.contains('STANDBY'), 'stay_id'])
        is_tr = feat['stay_id'].isin(trach)
        feat.loc[is_tr, 'resp_support_t0'] = np.where(feat.loc[is_tr, 'stay_id'].isin(on_vent), 'invasive', 'oxygen')
        st.log('tracheostomy_at_t0', {'n': len(trach), 'on_ventilator': len(on_vent & trach)})
    s = vent['starttime'].where(vent['starttime'] > vent['w0'], vent['w0'])
    e = vent['endtime'].where(vent['endtime'] < vent['t0'], vent['t0'])
    vent['ov_h'] = ((e - s).dt.total_seconds() / 3600).clip(lower=0)
    feat['imv_hours_24h'] = feat['stay_id'].map(
        vent[vent['ventilation_status'] == 'InvasiveVent'].groupby('stay_id')['ov_h'].sum()).fillna(0).clip(upper=WINDOW_H)
    st.log('resp_support_t0_pct', feat['resp_support_t0'].value_counts(normalize=True).mul(100).round(1).to_dict())

    # ------------------------------------------------------------------
    st.section('3. Вазопрессоры, инотропы, седация (inputevents, все коды)')
    p = raw('inputevents')
    cols = ', '.join(f"'{k}': '{v}'" for k, v in INPUT_SCHEMA.items())
    src = (f"read_csv('{p}', header=true, compression='{'gzip' if is_gzip(p) else 'none'}', "
           f"columns={{{cols}}}, quote='\"', escape='\"')")
    ids = ','.join(map(str, DRUGS))
    ev = con.execute(f"""SELECT e.stay_id, e.itemid, e.starttime, e.endtime, e.rate, e.rateuom, e.patientweight
                         FROM {src} e JOIN k ON e.stay_id = k.stay_id
                         WHERE e.itemid IN ({ids}) AND coalesce(e.statusdescription, '') <> 'Rewritten'
                           AND e.starttime <= k.t0 AND e.endtime >= k.w0 AND e.rate IS NOT NULL AND e.rate > 0""").df()
    ev['drug'] = ev['itemid'].map(DRUGS)
    ev['w'] = ev['patientweight'].fillna(ev['stay_id'].map(weight))
    print('Единицы скорости по препаратам:\n' + ev.groupby(['drug', 'rateuom']).size().to_string())
    ev['rate_std'] = np.nan
    for d in ev['drug'].unique():
        m = ev['drug'] == d
        ev.loc[m, 'rate_std'] = std_rate(d, ev.loc[m, 'rate'], ev.loc[m, 'rateuom'], ev.loc[m, 'w']).values
    n_bad, n_rows = int(ev['rate_std'].isna().sum()), len(ev)
    st.log('rows_unconverted_units', {'n': n_bad, 'of': n_rows})
    if n_bad:
        print('Не пересчитаны:\n' + ev[ev['rate_std'].isna()].groupby(['drug', 'rateuom'], dropna=False).size().to_string())
    # Флаги «препарат вводился» не зависят от единиц: считаются по всем строкам (ev_all).
    # Пересчёт в стандартные единицы нужен только для доз вазопрессоров (NEE).
    ev_all = ev.copy()
    ev = ev.dropna(subset=['rate_std'])
    med = ev.groupby('drug')['rate_std'].median().round(4).to_dict()
    st.log('median_rate_std', med)
    ev['nee_contrib'] = ev['rate_std'] * ev['drug'].map(NEE_FACTOR).fillna(0)
    con.register('ev', ev[['stay_id', 'drug', 'starttime', 'endtime', 'rate_std', 'nee_contrib']])

    # Временные отрезки: все начала и окончания записей в окне + границы окна
    seg = con.execute("""
        WITH pv AS (SELECT * FROM ev WHERE drug IN ('norepinephrine','epinephrine','phenylephrine','dopamine','vasopressin')),
        t AS (
            SELECT pv.stay_id, greatest(least(pv.starttime, k.t0), k.w0) AS t FROM pv JOIN k USING (stay_id)
            UNION SELECT pv.stay_id, greatest(least(pv.endtime, k.t0), k.w0) FROM pv JOIN k USING (stay_id)
            UNION SELECT k.stay_id, k.w0 FROM k WHERE k.stay_id IN (SELECT stay_id FROM pv)
            UNION SELECT k.stay_id, k.t0 FROM k WHERE k.stay_id IN (SELECT stay_id FROM pv)),
        s AS (SELECT stay_id, t, lead(t) OVER (PARTITION BY stay_id ORDER BY t) AS t_next FROM t),
        d AS (SELECT s.stay_id, s.t, s.t_next, pv.drug, max(pv.nee_contrib) AS c
              FROM s JOIN pv ON pv.stay_id = s.stay_id AND pv.starttime <= s.t AND pv.endtime > s.t
              GROUP BY 1, 2, 3, 4)
        SELECT s.stay_id, s.t, s.t_next, coalesce(sum(d.c), 0) AS nee, count(d.drug) AS n_agents
        FROM s LEFT JOIN d ON d.stay_id = s.stay_id AND d.t = s.t
        GROUP BY 1, 2, 3""").df()
    seg = seg.merge(coh[['stay_id', 't0']], on='stay_id')
    at_t0 = seg[seg['t'] == seg['t0']].set_index('stay_id')
    seg['dur_h'] = (seg['t_next'] - seg['t']).dt.total_seconds() / 3600
    feat['nee_t0'] = feat['stay_id'].map(at_t0['nee']).fillna(0)
    feat['nee_max_24h'] = feat['stay_id'].map(seg.groupby('stay_id')['nee'].max()).fillna(0)
    feat['vasopressor_hours_24h'] = feat['stay_id'].map(
        seg[(seg['nee'] > 0) & seg['dur_h'].notna()].groupby('stay_id')['dur_h'].sum()).fillna(0).clip(upper=WINDOW_H)

    ev = ev_all.merge(coh[['stay_id', 't0']], on='stay_id')
    active_t0 = ev[(ev['starttime'] <= ev['t0']) & (ev['endtime'] > ev['t0'])]
    feat['n_vasopressors_t0'] = feat['stay_id'].map(
        active_t0[active_t0['drug'].isin(PRESSORS)].groupby('stay_id')['drug'].nunique()).fillna(0).astype(int)
    feat['vasopressin_t0'] = feat['stay_id'].isin(set(active_t0.loc[active_t0['drug'] == 'vasopressin', 'stay_id'])).astype(int)
    feat['angiotensin2_24h'] = feat['stay_id'].isin(set(ev.loc[ev['drug'] == 'angiotensin2', 'stay_id'])).astype(int)
    feat['inotrope_24h'] = feat['stay_id'].isin(set(ev.loc[ev['drug'].isin(INOTROPES), 'stay_id'])).astype(int)
    for d in SEDATIVES:
        feat[f'{d}_t0'] = feat['stay_id'].isin(set(active_t0.loc[active_t0['drug'] == d, 'stay_id'])).astype(int)
    feat['any_sedative_t0'] = feat[[f'{d}_t0' for d in SEDATIVES]].max(axis=1)
    feat['vasopressor_t0'] = (feat['n_vasopressors_t0'] > 0).astype(int)
    for c in ['vasopressor_t0', 'inotrope_24h', 'any_sedative_t0', 'propofol_t0', 'fentanyl_t0']:
        st.log(f'{c}_pct', round(float(feat[c].mean() * 100), 1))
    st.log('nee_t0_median_iqr_among_treated',
           [round(float(x), 3) for x in feat.loc[feat['nee_t0'] > 0, 'nee_t0'].quantile([.5, .25, .75])])

    # ------------------------------------------------------------------
    st.section('4. ЗПТ')
    rrt = read_csv_any(raw('rrt'), usecols=['stay_id', 'charttime', 'dialysis_active'])
    rrt['charttime'] = pd.to_datetime(rrt['charttime'], errors='coerce')
    rrt = rrt.merge(coh[['stay_id', 't0', 'w0']], on='stay_id')
    rrt = rrt[(rrt['charttime'] > rrt['w0']) & (rrt['charttime'] <= rrt['t0']) & (rrt['dialysis_active'] == 1)]
    feat['rrt_24h'] = feat['stay_id'].isin(set(rrt['stay_id'])).astype(int)
    st.log('rrt_24h_pct', round(float(feat['rrt_24h'].mean() * 100), 1))

    # ------------------------------------------------------------------
    st.section('5. Диурез')
    uo = read_csv_any(raw('urine_output_rate'), usecols=['stay_id', 'charttime', 'uo'])
    uo['charttime'] = pd.to_datetime(uo['charttime'], errors='coerce')
    uo = uo.merge(coh[['stay_id', 't0', 'intime']], on='stay_id')
    base = coh.set_index('stay_id')
    w = feat['stay_id'].map(weight)
    for W in (6, 24):
        start = (base['t0'] - H(W)).where(base['t0'] - H(W) > base['intime'], base['intime'])
        obs_h = (base['t0'] - start).dt.total_seconds() / 3600
        u = uo.assign(start=uo['stay_id'].map(start))
        vol = u[(u['charttime'] > u['start']) & (u['charttime'] <= u['t0'])].groupby('stay_id')['uo'].sum()
        hrs = feat['stay_id'].map(obs_h)
        feat[f'uo_mlkgh_{W}h'] = (feat['stay_id'].map(vol).clip(lower=0) / w / hrs).where(hrs >= UO_MIN_OBS_H)
        st.log(f'uo_mlkgh_{W}h_available_pct', round(float(feat[f'uo_mlkgh_{W}h'].notna().mean() * 100), 1))

    # ------------------------------------------------------------------
    st.section('6. Сохранение и контрольные проверки')
    feat.to_parquet(f'{DATA}/stage3_support.parquet', index=False)
    print(f'Сохранено: {DATA}/stage3_support.parquet ({len(feat):,} строк, {feat.shape[1] - 1} признаков)')
    chk = feat.merge(c1[['stay_id', 'est_year', 'main_analysis']], on='stay_id')
    chk = chk[chk['main_analysis'] == 1]
    by_year = chk.groupby('est_year').agg(n=('stay_id', 'size'), vaso=('vasopressor_t0', 'mean'),
                                          sed=('any_sedative_t0', 'mean'),
                                          nee_treated=('nee_t0', lambda x: x[x > 0].median()))
    by_year[['vaso', 'sed']] = (by_year[['vaso', 'sed']] * 100).round(1)
    print('Основной анализ по расчётному году: вазопрессоры и седация в T0, %:\n' + by_year.to_string())
    st.log('vasopressor_t0_pct_by_year', by_year['vaso'].to_dict())
    early = chk[chk['est_year'] <= 2016]['vasopressor_t0'].mean()
    late = chk[(chk['est_year'] >= 2017) & (chk['est_year'] <= 2019)]['vasopressor_t0'].mean()
    st.check(feat['stay_id'].is_unique, 'одна строка на ICU-пребывание')
    st.check(n_bad <= 0.01 * max(n_rows, 1), 'не пересчитаны единицы не более чем у 1% строк')
    st.check(0.03 <= med.get('norepinephrine', 0) <= 0.5, 'медиана норадреналина правдоподобна (0,03–0,5 мкг/кг/мин)')
    st.check(late >= 0.8 * early, 'доля вазопрессоров в 2017–2019 не провалилась относительно 2009–2016 (≥ 80%)')
