# -*- coding: utf-8 -*-
"""
ЭТАП 9б. Сравнение AUROC основной модели между независимыми подгруппами тестовой выборки.

  * AUROC и его дисперсия — по DeLong; сравнение двух непересекающихся подгрупп — z-тест для несвязанных выборок.
  * 12 сравнений (возраст, COVID-19, пол, ИВЛ, вазопрессоры, септический шок, каждый тип ОРИТ против остальных);
    поправка Холма на все 12 сравнений.

Вход: data/stage8_test_predictions.parquet, data/stage6_master.parquet
Выход: results/stage9b_subgroup_pvalues.csv
"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import rankdata, norm
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
sys.path.insert(0, _HERE)
from common import DATA, ROOT

RESULTS = f'{ROOT}/results'
os.makedirs(RESULTS, exist_ok=True)


def auc_delong(y, p):
    y, p = np.asarray(y, int), np.asarray(p, float)
    pos, neg = p[y == 1], p[y == 0]
    m, n = len(pos), len(neg)
    r = rankdata(np.concatenate([pos, neg]))
    v10 = (r[:m] - rankdata(pos)) / n
    v01 = 1 - (r[m:] - rankdata(neg)) / m
    return v10.mean(), v10.var(ddof=1) / m + v01.var(ddof=1) / n


def compare(name, a, b, d):
    (a1, v1), (a2, v2) = auc_delong(d.y[a], d.p[a]), auc_delong(d.y[b], d.p[b])
    z = (a1 - a2) / np.sqrt(v1 + v2)
    pv = 2 * norm.sf(abs(z))
    ci = lambda a_, v_: f'{a_:.3f} ({a_ - 1.96 * v_ ** .5:.3f}–{a_ + 1.96 * v_ ** .5:.3f})'
    return {'сравнение': name, 'n1': int(a.sum()), 'AUROC 1': ci(a1, v1), 'n2': int(b.sum()), 'AUROC 2': ci(a2, v2),
            'разность': round(a1 - a2, 3), 'p_raw': pv}


pred = pd.read_parquet(f'{DATA}/stage8_test_predictions.parquet')[['stay_id', 'y', 'p_lgbm']]
M = pd.read_parquet(f'{DATA}/stage6_master.parquet', columns=['stay_id', 'age', 'covid19_dx', 'careunit_group', 'sex_male',
                                                              'resp_support_t0', 'vasopressor_t0', 'lactate_last'])
d = pred.merge(M, on='stay_id', how='left').rename(columns={'p_lgbm': 'p'})
d = d[d['p'].notna()].reset_index(drop=True)
age = d['age']
unit = d['careunit_group'].astype(str)
imv = d['resp_support_t0'].astype(str) == 'invasive'
vaso = d['vasopressor_t0'] == 1
shock = vaso & (d['lactate_last'] > 2)
c = lambda name, m1, m2: compare(name, np.asarray(m1), np.asarray(m2), d)
rows = [
    c('Возраст: ≥ 80 vs < 65 лет', age >= 80, age < 65),
    c('Возраст: 65–79 vs < 65 лет', (age >= 65) & (age < 80), age < 65),
    c('COVID-19: да vs нет', d['covid19_dx'] == 1, d['covid19_dx'] != 1),
    c('Пол: женский vs мужской', d['sex_male'] == 0, d['sex_male'] == 1),
    c('ИВЛ в T0: да vs нет', imv, ~imv),
    c('Вазопрессоры в T0: да vs нет', vaso, ~vaso),
    c('Септический шок: да vs нет', shock, ~shock),
] + [c(f'ОРИТ: {u} vs остальные', unit == u, unit != u)
     for u in ['medical', 'surgical_trauma', 'cardiac', 'mixed', 'neuro']]
res = pd.DataFrame(rows)
order = np.argsort(res['p_raw'].to_numpy()); m = len(res); adj = np.empty(m); run = 0.0
for i, j in enumerate(order):                       # поправка Холма
    run = max(run, min(1.0, (m - i) * res['p_raw'].iat[j])); adj[j] = run
fmt = lambda v: '< 0,001' if v < 0.001 else f'{v:.3f}'.replace('.', ',')
res['p'] = res['p_raw'].map(fmt); res['p (Холм)'] = [fmt(v) for v in adj]
res['p_holm'] = adj
res.to_csv(f'{RESULTS}/stage9b_subgroup_pvalues.csv', index=False)
res = res.drop(columns=['p_raw', 'p_holm'])
with pd.option_context('display.width', 200, 'display.max_columns', 20):
    print(res.to_string(index=False))
print(f'Сохранено: {RESULTS}/stage9b_subgroup_pvalues.csv')
