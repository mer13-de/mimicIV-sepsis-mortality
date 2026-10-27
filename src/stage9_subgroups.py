# -*- coding: utf-8 -*-
"""
ЭТАП 9. Анализ подгрупп (тестовый период; модели зафиксированы на этапе 7, тест уже открыт на этапе 8).

Подгруппы (основная модель LightGBM; AUROC с 95% ДИ — бутстреп 1 000, калибровка):
  COVID-19; возраст < 65, 65–79, ≥ 80 лет; тип ОРИТ; пол; ИВЛ в T0; вазопрессоры в T0;
  септический шок (вазопрессоры в T0 и лактат > 2 ммоль/л).
Сравнение AUROC между подгруппами с поправкой Холма — stage9b_subgroup_pvalues.py.

Вход: models/*, stage6_master.parquet, stage6b_benchmarks.parquet, stage8_test_predictions.parquet
Выход: results/stage9_subgroups.csv, results/stage9_subgroups_forest.png
"""
import os
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import lightgbm as lgb

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
sys.path.insert(0, _HERE)
from common import Stage, DATA, ROOT, SEED
from modeling import prep_lgb
from evaluation import discrimination, calibration, bootstrap_ci, delong

MODELS = f'{ROOT}/models'
RESULTS = f'{ROOT}/results'
os.makedirs(RESULTS, exist_ok=True)
N_BOOT_SUB = 1000
MIN_N, MIN_EVENTS = 100, 20


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def summarize(name, y, p, n_boot, p_saps=None, p_ref=None, ref_name=None, group=None):
    y = np.asarray(y).astype(int)
    r = {'analysis': name, 'n': int(len(y)), 'events': int(y.sum()),
         'mortality_pct': round(float(y.mean() * 100), 1) if len(y) else np.nan}
    if group is not None:
        r = {'group': group, **r}
    if len(y) < MIN_N or y.sum() < MIN_EVENTS or y.sum() == len(y):
        r['note'] = f'n < {MIN_N} или событий < {MIN_EVENTS}: метрики не рассчитаны'
        return r
    r.update(discrimination(y, p))
    c = calibration(y, p)
    r.update({k: c[k] for k in ['oe', 'cal_intercept', 'cal_slope', 'cal_slope_lo', 'cal_slope_hi', 'brier', 'ici']})
    preds = {'m': p}
    if p_saps is not None:
        preds['saps'] = p_saps
    ci, dci = bootstrap_ci(y, preds, n_boot=n_boot, seed=SEED, stats=('auroc', 'oe'))
    r['auroc_lo'], r['auroc_hi'] = ci['m']['auroc']
    r['oe_lo'], r['oe_hi'] = ci['m']['oe']
    if p_saps is not None:
        a1, a2, d, pv = delong(y, p, p_saps)
        r.update({'auroc_sapsii': a2, 'diff_vs_sapsii': d, 'diff_vs_sapsii_lo': dci[('m', 'saps')][0],
                  'diff_vs_sapsii_hi': dci[('m', 'saps')][1], 'p_delong_vs_sapsii': pv})
    if p_ref is not None:
        a1, a2, d, pv = delong(y, p, p_ref)
        r.update({f'auroc_{ref_name}': a2, f'diff_vs_{ref_name}': d, f'p_delong_vs_{ref_name}': pv})
    return r


with Stage('stage9_subgroups') as st:
    st.section('0. Сверка контрольных сумм и факта открытия теста')
    ref = json.load(open(f'{MODELS}/stage7_checksums.json'))
    bad = [f for f in ref if sha256(f'{MODELS}/{f}') != ref[f]]
    st.check(not bad, f'зафиксированные модели не менялись (расхождения: {bad})')
    st.check(os.path.isfile(f'{MODELS}/TEST_OPENED.json'), 'основная оценка на тесте (этап 8) уже выполнена')
    fr = json.load(open(f'{MODELS}/stage7_frozen.json', encoding='utf-8'))
    LV = fr['cat_levels']

    M = pd.read_parquet(f'{DATA}/stage6_master.parquet')
    M = M.merge(pd.read_parquet(f'{DATA}/stage6b_benchmarks.parquet'), on='stay_id', how='left')
    boosters = {}

    def predict(df, model):
        if model not in boosters:
            boosters[model] = lgb.Booster(model_file=f'{MODELS}/lgbm_{model}.txt')
        cols = fr['features_main']
        return boosters[model].predict(prep_lgb(df, cols, {c: LV[c] for c in cols if c in LV}))

    te = M[(M['in_main_cohort'] == 1) & (M['main_analysis'] == 1) & (M['split'] == 'test')].reset_index(drop=True)
    y = te['death_30d'].to_numpy()
    p_main = predict(te, 'main')
    p8 = pd.read_parquet(f'{DATA}/stage8_test_predictions.parquet').set_index('stay_id')['p_lgbm']
    st.check(np.allclose(p_main, te['stay_id'].map(p8).to_numpy(), atol=1e-9),
             'прогнозы основной модели совпадают с этапом 8')
    st.log('n_test', int(len(te)))

    overall = summarize('Все пациенты тестовой выборки', y, p_main, N_BOOT_SUB)
    st.log('overall_auroc', round(float(overall['auroc']), 3))

    # ------------------------------------------------------------------
    st.section('1. Подгруппы (основная модель)')
    age_band = pd.cut(te['age'], [17, 64, 79, 200], labels=['< 65', '65–79', '≥ 80'])
    vaso = te['vasopressor_t0'] == 1
    groups = {
        'COVID-19': np.where(te['covid19_dx'] == 1, 'да', 'нет'),
        'Возраст': age_band.astype(str),
        'Отделение': te['careunit_group'].astype(str),
        'Пол': te['sex_male'].map({1: 'мужской', 0: 'женский'}),
        'ИВЛ в T0': np.where(te['resp_support_t0'].astype(str) == 'invasive', 'да', 'нет'),
        'Вазопрессоры в T0': np.where(vaso, 'да', 'нет'),
        'Септический шок': np.where(vaso & (te['lactate_last'] > 2), 'да', 'нет'),
    }
    srows = []
    for g, lab in groups.items():
        lab = pd.Series(np.asarray(lab), index=te.index)
        for level in sorted(lab.dropna().unique()):
            m = (lab == level).to_numpy()
            srows.append(summarize(str(level), y[m], p_main[m], N_BOOT_SUB, group=g))
    sub = pd.DataFrame(srows)
    sub.to_csv(f'{RESULTS}/stage9_subgroups.csv', index=False)
    print(sub[[c for c in ['group', 'analysis', 'n', 'events', 'auroc', 'auroc_lo', 'auroc_hi', 'oe', 'cal_slope', 'note']
               if c in sub]].round(3).to_string(index=False))

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        s = sub.dropna(subset=['auroc']).reset_index(drop=True)
        fig, ax = plt.subplots(figsize=(7, 0.32 * len(s) + 1.2))
        yy = np.arange(len(s))[::-1]
        ax.errorbar(s['auroc'], yy, xerr=[s['auroc'] - s['auroc_lo'], s['auroc_hi'] - s['auroc']],
                    fmt='o', color='#4c72b0', ms=4, capsize=2)
        ax.axvline(float(overall['auroc']), color='grey', ls='--', lw=1)
        ax.set_yticks(yy)
        ax.set_yticklabels([f'{a}: {b} (n={n})' for a, b, n in zip(s['group'], s['analysis'], s['n'])], fontsize=8)
        ax.set_xlabel('AUROC (95% ДИ)')
        fig.tight_layout()
        fig.savefig(f'{RESULTS}/stage9_subgroups_forest.png', dpi=200)
        plt.close(fig)
    except ImportError:
        print('matplotlib недоступен — лесной график пропущен')

    st.log('subgroup_auroc_range', [round(float(sub['auroc'].min()), 3), round(float(sub['auroc'].max()), 3)])
