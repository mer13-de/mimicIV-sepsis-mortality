# -*- coding: utf-8 -*-
"""
ЭТАП 8. Оценка на тестовом периоде (открывается один раз, после фиксации моделей на этапе 7).

  0. Сверка SHA-256 всех зафиксированных файлов с stage7_checksums.json; при несовпадении — остановка.
     Факт первого открытия теста записывается в models/TEST_OPENED.json (время, контрольные суммы);
     повторный запуск лишь детерминированно пересчитывает те же результаты и отмечается в журнале.
  1. Прогнозы на тестовом периоде: LightGBM (основная), логистическая регрессия со сплайнами, экономная модель,
     SAPS II и OASIS (перекалиброванные на обучающем периоде; для дискриминации — исходный балл, AUROC тот же).
  2. Дискриминация: AUROC и AUPRC с 95% ДИ (перцентильный бутстреп, 2 000 парных выборок).
     Сравнения LightGBM с SAPS II (первичное), OASIS, логистической регрессией и экономной моделью:
     тест DeLong + бутстреп-ДИ разности; поправка Холма внутри семейства.
  3. Калибровка: O/E, интерсепт при фиксированном наклоне, наклон, Брайер и масштабированный Брайер,
     ICI/E50/E90 (локально-линейное сглаживание), сглаженные калибровочные кривые с распределением прогнозов.
  4. Кривая принятия решений 0,05–0,60.
  5. TreeSHAP основной модели на тесте: рейтинг и графики зависимости для ведущих признаков.
  6. Внутренняя валидация: те же метрики out-of-fold на обучающем периоде.

Вход: models/*, stage6_master.parquet, stage6b_benchmarks.parquet, stage7_oof.parquet
Выход: results/stage8_*.csv, results/*.png, data/stage8_test_predictions.parquet
"""
import os
import sys
import json
import hashlib
from datetime import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
import joblib

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
sys.path.insert(0, _HERE)
from common import Stage, DATA, ROOT, SEED
from modeling import prep_lgb, prep_lr  # нужен и для загрузки lr_spline.joblib
from evaluation import (discrimination, calibration, bootstrap_ci, delong, holm, net_benefit, lowess_fit)

MODELS = f'{ROOT}/models'
RESULTS = f'{ROOT}/results'
os.makedirs(RESULTS, exist_ok=True)
N_BOOT = 2000
THRESHOLDS = np.round(np.arange(0.05, 0.601, 0.01), 2)
MODEL_LABELS = {'lgbm': 'LightGBM', 'lr_spline': 'ЛР со сплайнами', 'parsimonious': 'Экономная LightGBM',
                'sapsii': 'SAPS II', 'oasis': 'OASIS'}
COMPARISONS = ['sapsii', 'oasis', 'lr_spline', 'parsimonious']   # LightGBM против каждой; первичное — SAPS II
N_DEPENDENCE = 6


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def evaluate(y, preds, st, tag):
    """Полный набор метрик для словаря прогнозов (одинаковый набор пациентов)."""
    ci, dci = bootstrap_ci(y, preds, n_boot=N_BOOT, seed=SEED)
    rows = []
    for k, p in preds.items():
        r = {'set': tag, 'model': k, 'n': int(len(y)), 'events': int(np.sum(y))}
        r.update(discrimination(y, p))
        r.update(calibration(y, p))
        for s, (lo, hi) in ci[k].items():
            r[f'{s}_lo'], r[f'{s}_hi'] = lo, hi
        rows.append(r)
    met = pd.DataFrame(rows)
    comp = []
    for other in [c for c in COMPARISONS if c in preds]:
        a1, a2, d, pv = delong(y, preds['lgbm'], preds[other])
        key = ('lgbm', other) if ('lgbm', other) in dci else (other, 'lgbm')
        lo, hi = dci[key] if key[0] == 'lgbm' else (-dci[key][1], -dci[key][0])
        comp.append({'set': tag, 'comparison': f'lgbm_vs_{other}', 'auroc_lgbm': a1, 'auroc_other': a2,
                     'diff': d, 'diff_lo_boot': lo, 'diff_hi_boot': hi, 'p_delong': pv})
    comp = pd.DataFrame(comp)
    if len(comp):
        comp['p_holm'] = holm(comp['p_delong'].to_numpy())
    show = met[['model', 'auroc', 'auroc_lo', 'auroc_hi', 'auprc', 'oe', 'cal_intercept', 'cal_slope', 'brier', 'ici']]
    print(f'\n[{tag}] n = {len(y)}, событий = {int(np.sum(y))}\n' + show.round(3).to_string(index=False))
    if len(comp):
        print(comp[['comparison', 'diff', 'diff_lo_boot', 'diff_hi_boot', 'p_delong', 'p_holm']].round(4).to_string(index=False))
    return met, comp


def plots(y, preds, shap_df, Xte, tag):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib недоступен — графики пропущены')
        return
    # калибровка
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(6.5, 7.5), gridspec_kw={'height_ratios': [4, 1]}, sharex=True)
    ax.plot([0, 1], [0, 1], ls='--', c='grey', lw=1)
    for k, p in preds.items():
        o = np.argsort(p)
        ax.plot(np.asarray(p)[o], lowess_fit(p, y)[o], lw=1.8, label=MODEL_LABELS.get(k, k))
    ax.set_ylabel('Наблюдаемая частота смерти'); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(frameon=False)
    axh.hist(preds['lgbm'], bins=50, color='grey'); axh.set_xlabel('Прогнозируемая вероятность (LightGBM)')
    fig.tight_layout(); fig.savefig(f'{RESULTS}/stage8_calibration_{tag}.png', dpi=200); plt.close(fig)
    # кривая принятия решений
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    prev = np.mean(y)
    ax.plot(THRESHOLDS, prev - (1 - prev) * THRESHOLDS / (1 - THRESHOLDS), c='grey', lw=1, label='Лечить всех')
    ax.axhline(0, c='black', lw=1, label='Не лечить никого')
    for k, p in preds.items():
        ax.plot(THRESHOLDS, net_benefit(y, p, THRESHOLDS), lw=1.8, label=MODEL_LABELS.get(k, k))
    ax.set_ylim(-0.02, max(prev, 0.05) + 0.02); ax.set_xlabel('Пороговая вероятность'); ax.set_ylabel('Чистая выгода')
    ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(f'{RESULTS}/stage8_dca_{tag}.png', dpi=200)
    plt.close(fig)
    # SHAP
    if shap_df is not None:
        rank = shap_df.abs().mean().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6.5, 6))
        rank.head(20)[::-1].plot.barh(ax=ax, color='#4c72b0'); ax.set_xlabel('Среднее |SHAP| (лог-шансы)')
        fig.tight_layout(); fig.savefig(f'{RESULTS}/stage8_shap_bar_{tag}.png', dpi=200); plt.close(fig)
        top = rank.index[:N_DEPENDENCE]
        fig, axes = plt.subplots(2, 3, figsize=(12, 7))
        for a, f in zip(axes.ravel(), top):
            xv = Xte[f]
            xv = xv.cat.codes.where(xv.notna(), np.nan) if hasattr(xv, 'cat') else xv
            a.scatter(xv, shap_df[f], s=4, alpha=0.4); a.axhline(0, c='grey', lw=0.8)
            a.set_title(f, fontsize=9); a.set_ylabel('SHAP')
        fig.tight_layout(); fig.savefig(f'{RESULTS}/stage8_shap_dependence_{tag}.png', dpi=200); plt.close(fig)


with Stage('stage8_evaluate') as st:
    st.section('0. Сверка контрольных сумм зафиксированных моделей')
    ref = json.load(open(f'{MODELS}/stage7_checksums.json'))
    now = {f: sha256(f'{MODELS}/{f}') for f in ref}
    bad = [f for f in ref if ref[f] != now[f]]
    st.check(not bad, f'все зафиксированные файлы совпадают с контрольными суммами этапа 7 (расхождения: {bad})')
    ref_digest = hashlib.sha256(json.dumps(ref, sort_keys=True).encode()).hexdigest()
    st.log('checksums_digest', ref_digest)
    marker = f'{MODELS}/TEST_OPENED.json'
    if os.path.isfile(marker):
        mk = json.load(open(marker))
        st.log('test_already_opened', mk['opened'])
        st.check(mk['checksums_digest'] == ref_digest, 'модели не менялись с момента первого открытия теста')
    else:
        json.dump({'opened': datetime.now().isoformat(timespec='seconds'), 'checksums_digest': ref_digest},
                  open(marker, 'w'), indent=2)
        st.log('test_opened_first_time', True)
    fr = json.load(open(f'{MODELS}/stage7_frozen.json', encoding='utf-8'))

    # ------------------------------------------------------------------
    st.section('1. Прогнозы на тестовом периоде')
    M = pd.read_parquet(f'{DATA}/stage6_master.parquet')
    te = M[(M['in_main_cohort'] == 1) & (M['main_analysis'] == 1) & (M['split'] == 'test')].reset_index(drop=True)
    te = te.merge(pd.read_parquet(f'{DATA}/stage6b_benchmarks.parquet'), on='stay_id', how='left')
    y = te['death_30d'].to_numpy().astype(int)
    FEATS, CATS, LV = fr['features_main'], fr['categorical'], fr['cat_levels']
    Xte = prep_lgb(te, FEATS, LV)
    b_main = lgb.Booster(model_file=f'{MODELS}/lgbm_main.txt')
    b_pars = lgb.Booster(model_file=f'{MODELS}/lgbm_parsimonious.txt')
    lr_model = joblib.load(f'{MODELS}/lr_spline.joblib')
    P = {'lgbm': b_main.predict(Xte),
         'lr_spline': lr_model.predict_proba(prep_lr(te, FEATS, CATS))[:, 1],
         'parsimonious': b_pars.predict(Xte[fr['parsimonious']['features']])}
    for s in ['sapsii', 'oasis']:
        rc = fr['recalibration'][s]
        P[s] = 1 / (1 + np.exp(-(rc['intercept'] + rc['slope'] * te[s].to_numpy(float))))
    pred = pd.DataFrame({'stay_id': te['stay_id'], 'y': y, **{f'p_{k}': v for k, v in P.items()}})
    pred.to_parquet(f'{DATA}/stage8_test_predictions.parquet', index=False)
    ok = pred[[f'p_{k}' for k in P]].notna().all(axis=1).to_numpy()
    st.log('test', {'n': int(len(y)), 'deaths': int(y.sum()), 'mortality_pct': round(float(y.mean() * 100), 1),
                    'complete_all_models': int(ok.sum())})
    st.check(ok.mean() >= 0.99, 'прогнозы всех моделей есть у ≥ 99% тестовых пациентов')

    # ------------------------------------------------------------------
    st.section('2–4. Дискриминация, калибровка, сравнения, кривая принятия решений (тест)')
    Pc = {k: v[ok] for k, v in P.items()}
    met_te, comp_te = evaluate(y[ok], Pc, st, 'test')
    dca = pd.DataFrame({'threshold': THRESHOLDS, 'treat_all': y[ok].mean() - (1 - y[ok].mean()) * THRESHOLDS / (1 - THRESHOLDS),
                        **{k: net_benefit(y[ok], p, THRESHOLDS) for k, p in Pc.items()}})
    dca.to_csv(f'{RESULTS}/stage8_dca_test.csv', index=False)

    # ------------------------------------------------------------------
    st.section('5. SHAP на тесте')
    sh = b_main.predict(Xte, pred_contrib=True)[:, :-1]
    shap_df = pd.DataFrame(sh, columns=FEATS)
    rank_te = shap_df.abs().mean().sort_values(ascending=False)
    rank_tr = pd.read_csv(f'{DATA}/stage7_shap_oof.csv', index_col=0).iloc[:, 0]
    shap_tab = pd.DataFrame({'mean_abs_shap_test': rank_te, 'mean_abs_shap_train_oof': rank_tr})
    shap_tab['rank_test'] = shap_tab['mean_abs_shap_test'].rank(ascending=False)
    shap_tab['rank_train_oof'] = shap_tab['mean_abs_shap_train_oof'].rank(ascending=False)
    shap_tab.sort_values('rank_test').to_csv(f'{RESULTS}/stage8_shap_ranking.csv', index_label='feature')
    print('Топ-15 по SHAP на тесте (ранг на обучении out-of-fold):\n' +
          shap_tab.sort_values('rank_test').head(15)[['mean_abs_shap_test', 'rank_train_oof']].round(4).to_string())
    top20 = set(shap_tab.nsmallest(20, 'rank_test').index) & set(shap_tab.nsmallest(20, 'rank_train_oof').index)
    st.log('shap_top20_overlap_train_test', len(top20))

    # ------------------------------------------------------------------
    st.section('6. Внутренняя валидация (out-of-fold, обучающий период)')
    oof = pd.read_parquet(f'{DATA}/stage7_oof.parquet')
    oof = oof.rename(columns={'p_sapsii_recal': 'p_sapsii', 'p_oasis_recal': 'p_oasis'})
    okt = oof[['p_lgbm', 'p_lr', 'p_parsimonious', 'p_sapsii', 'p_oasis']].notna().all(axis=1)
    o = oof[okt]
    Po = {'lgbm': o['p_lgbm'].to_numpy(), 'lr_spline': o['p_lr'].to_numpy(),
          'parsimonious': o['p_parsimonious'].to_numpy(), 'sapsii': o['p_sapsii'].to_numpy(),
          'oasis': o['p_oasis'].to_numpy()}
    met_tr, comp_tr = evaluate(o['y'].to_numpy(), Po, st, 'train_oof')

    # ------------------------------------------------------------------
    st.section('7. Сохранение')
    met = pd.concat([met_te, met_tr], ignore_index=True)
    comp = pd.concat([comp_te, comp_tr], ignore_index=True)
    met.to_csv(f'{RESULTS}/stage8_metrics.csv', index=False)
    comp.to_csv(f'{RESULTS}/stage8_comparisons.csv', index=False)
    plots(y[ok], Pc, shap_df[ok].reset_index(drop=True), Xte[ok].reset_index(drop=True), 'test')
    print(f'Результаты: {RESULTS}')
    prim = comp_te[comp_te['comparison'] == 'lgbm_vs_sapsii'].iloc[0]
    lg = met_te[met_te['model'] == 'lgbm'].iloc[0]
    st.log('primary', {'auroc_lgbm': round(float(lg['auroc']), 4),
                       'ci': [round(float(lg['auroc_lo']), 4), round(float(lg['auroc_hi']), 4)],
                       'auroc_sapsii': round(float(prim['auroc_other']), 4), 'diff': round(float(prim['diff']), 4),
                       'p_delong': float(prim['p_delong']), 'p_holm': float(prim['p_holm'])})
    st.log('calibration_lgbm_test', {k: round(float(lg[k]), 3) for k in
                                     ['oe', 'cal_intercept', 'cal_slope', 'brier', 'brier_scaled', 'ici', 'e50', 'e90']})
    st.check(len(met_te) == 5 and len(comp_te) == 4, 'рассчитаны метрики 5 моделей и 4 сравнения на тесте')
