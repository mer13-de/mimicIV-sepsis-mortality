# -*- coding: utf-8 -*-
"""
ЭТАП 6б. Клинические эталоны: официальные SAPS II и OASIS (mimic-code, первые сутки ОРИТ).

  * Баллы и опубликованные вероятности смерти берутся как есть из производных таблиц sapsii и oasis.
  * Шкалы используют данные первых 24 ч ОРИТ, в том числе после T0 (консервативно в пользу шкал).
  * Логистическая перекалибровка балла выполняется на этапе 7 (только обучающий период).

Вход: stage6_master.parquet; sapsii, oasis
Выход: data/stage6b_benchmarks.parquet (stay_id, sapsii, sapsii_prob, oasis, oasis_prob)
"""
import os
import sys
import pandas as pd
from sklearn.metrics import roc_auc_score

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
sys.path.insert(0, _HERE)
from common import Stage, DATA, raw, read_csv_any

with Stage('stage6b_benchmarks') as st:
    st.section('1. SAPS II и OASIS')
    m = pd.read_parquet(f'{DATA}/stage6_master.parquet',
                        columns=['stay_id', 'split', 'in_main_cohort', 'main_analysis', 'death_30d'])
    saps = read_csv_any(raw('sapsii'), usecols=['stay_id', 'sapsii', 'sapsii_prob']).drop_duplicates('stay_id')
    oas = read_csv_any(raw('oasis'), usecols=['stay_id', 'oasis', 'oasis_prob']).drop_duplicates('stay_id')
    b = m[['stay_id']].merge(saps, on='stay_id', how='left').merge(oas, on='stay_id', how='left')
    b.to_parquet(f'{DATA}/stage6b_benchmarks.parquet', index=False)
    print(f'Сохранено: {DATA}/stage6b_benchmarks.parquet ({len(b):,} строк)')

    x = m.merge(b, on='stay_id')
    tr = x[(x['in_main_cohort'] == 1) & (x['main_analysis'] == 1) & (x['split'] == 'train')]
    st.log('available_pct', {c: round(float(b[c].notna().mean() * 100), 2) for c in ['sapsii', 'oasis']})
    st.log('median_iqr_train', {c: [float(v) for v in tr[c].quantile([.5, .25, .75])] for c in ['sapsii', 'oasis']})
    st.log('auroc_train_only', {c: round(float(roc_auc_score(tr.dropna(subset=[c])['death_30d'],
                                                               tr.dropna(subset=[c])[c])), 3)
                                for c in ['sapsii', 'oasis']})
    st.log('published_prob_mean_vs_observed_train', {
        'observed': round(float(tr['death_30d'].mean()), 3),
        'sapsii_prob': round(float(tr['sapsii_prob'].mean()), 3), 'oasis_prob': round(float(tr['oasis_prob'].mean()), 3)})
    st.check(b['stay_id'].is_unique, 'одна строка на пребывание')
    st.check(b[['sapsii', 'oasis']].notna().all(axis=1).mean() >= 0.99, 'SAPS II и OASIS есть у ≥ 99% пребываний')
