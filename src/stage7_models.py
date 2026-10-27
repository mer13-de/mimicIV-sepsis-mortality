# -*- coding: utf-8 -*-
"""
ЭТАП 7. Обучение и фиксация всех моделей (только обучающий период; тестовый период не читается).

  1. Фиксированные 5 фолдов (стратификация по исходу, seed проекта).
  2. LightGBM: Optuna (TPE, 100 попыток, фиксированный seed); цель — средний log-loss в 5-кратной CV;
     скорость обучения 0,03; число деревьев — по усреднённой по фолдам кривой log-loss (ранняя остановка 100,
     максимум 5 000). Исследование сохраняется в models/ после каждой попытки: при обрыве сессии Colab повторный
     запуск продолжает с того же места.
  3. Out-of-fold прогнозы и TreeSHAP основной модели; итоговая модель на всём обучающем периоде.
  4. Логистическая регрессия со сплайнами: L2; C по сетке 10⁻⁴–10 по log-loss в тех же фолдах; для непрерывных
     признаков — медианная импутация + индикатор пропуска, логарифм для асимметричных неотрицательных
     (асимметрия > 2), кубические сплайны с 4 узлами по квантилям; дискретные — линейно; категориальные — one-hot.
     Всё, что подбирается по данным, подбирается внутри фолда.
  5. Экономная модель: кандидаты с заполненностью > 70% в каждом из трёх последних расчётных лет обучающей выборки
     (после интервальной классификации периода это 2016–2018); рейтинг — среднее
     |SHAP| out-of-fold; наборы 10/15/20/25/30/40; наименьший набор с AUROC(CV) ≥ AUROC(все кандидаты) − 0,010.
  6. Логистическая перекалибровка SAPS II и OASIS на обучающем периоде.
  7. Фиксация: параметры, перечни признаков и модели сохраняются в models/; SHA-256 всех файлов
     записывается в stage7_checksums.json и в журнал. Этап 8 сверяет контрольные суммы перед оценкой.

Вход: stage6_master.parquet, stage6_features.json, stage6b_benchmarks.parquet
Выход: models/*, data/stage7_folds.parquet, data/stage7_oof.parquet, data/stage7_shap_oof.csv,
       data/stage7_parsimony.csv, data/stage7_optuna_trials.csv
"""
import os
import sys
import json
import time
import pickle
import hashlib
import subprocess
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
sys.path.insert(0, _HERE)
from common import Stage, DATA, ROOT, SEED
from modeling import prep_lgb, prep_lr, lr_preprocessor

try:
    import optuna
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', '-q', 'install', 'optuna'], check=True)
    import optuna
import lightgbm as lgb
import joblib

MODELS = f'{ROOT}/models'
os.makedirs(MODELS, exist_ok=True)
STUDY_PKL = f'{MODELS}/optuna_study.pkl'
N_TRIALS = 100
LR = 0.03
MAX_ROUNDS = 5000
ES_ROUNDS = 100
N_FOLDS = 5
C_GRID = np.logspace(-4, 1, 11)
PARS_SIZES = [10, 15, 20, 25, 30, 40]
PARS_TOL = 0.010
PARS_FILL = 0.70
PARS_N_YEARS = 3          # последние расчётные годы обучающей выборки для критерия заполненности
BASE = dict(objective='binary', learning_rate=LR, verbosity=-1, seed=SEED, deterministic=True,
            force_row_wise=True, feature_pre_filter=False, bagging_freq=1)


# ---------------------------------------------------------------------- утилиты
def make_folds(y):
    fold = np.empty(len(y), dtype=int)
    for k, (_, va) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y)):
        fold[va] = k
    return fold, [(np.where(fold != k)[0], np.where(fold == k)[0]) for k in range(N_FOLDS)]


def cv_lgb(X, y, folds, params, with_auc=False):
    """Число деревьев по усреднённой по фолдам кривой log-loss; возвращает (n, logloss, auc)."""
    p = {**BASE, **params, 'metric': ['binary_logloss', 'auc'] if with_auc else ['binary_logloss']}
    r = lgb.cv(p, lgb.Dataset(X, y, free_raw_data=False), num_boost_round=MAX_ROUNDS, folds=folds,
               callbacks=[lgb.early_stopping(ES_ROUNDS, first_metric_only=True, verbose=False)])
    kl = [k for k in r if k.endswith('binary_logloss-mean')][0]
    ka = [k for k in r if k.endswith('auc-mean')]
    return len(r[kl]), float(r[kl][-1]), (float(r[ka[0]][-1]) if ka else None)


def oof_lgb(X, y, folds, params, n_trees, shap=False):
    oof = np.zeros(len(y))
    sh = np.zeros((len(y), X.shape[1])) if shap else None
    for tri, vai in folds:
        b = lgb.train({**BASE, **params}, lgb.Dataset(X.iloc[tri], y[tri]), num_boost_round=n_trees)
        oof[vai] = b.predict(X.iloc[vai])
        if shap:
            sh[vai] = b.predict(X.iloc[vai], pred_contrib=True)[:, :-1]
    return oof, sh


def fit_lgb(X, y, params, n_trees, path):
    b = lgb.train({**BASE, **params}, lgb.Dataset(X, y), num_boost_round=n_trees)
    b.save_model(path)
    return b


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------- этап
with Stage('stage7_models') as st:
    st.section('1. Данные обучающего периода и фолды')
    M = pd.read_parquet(f'{DATA}/stage6_master.parquet')
    man = json.load(open(f'{DATA}/stage6_features.json', encoding='utf-8'))
    FEATS, CATS = man['features_main'], man['categorical']
    tr = M[(M['in_main_cohort'] == 1) & (M['main_analysis'] == 1) & (M['split'] == 'train')].reset_index(drop=True)
    st.check((tr['split'] == 'train').all() and tr['est_year'].max() <= 2019, 'используется только обучающий период')
    y = tr['death_30d'].to_numpy().astype(int)
    cat_levels = {c: sorted(tr[c].dropna().astype(str).unique().tolist()) for c in CATS}
    fold, FOLDS = make_folds(y)
    tr[['stay_id']].assign(fold=fold).to_parquet(f'{DATA}/stage7_folds.parquet', index=False)
    st.log('train', {'n': int(len(y)), 'deaths': int(y.sum()), 'features': len(FEATS),
                     'fold_mortality_pct': [round(float(y[fold == k].mean() * 100), 1) for k in range(N_FOLDS)]})
    X = prep_lgb(tr, FEATS, cat_levels)

    # ------------------------------------------------------------------
    st.section('2. LightGBM: Optuna (TPE, 100 попыток)')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    data_sig = hashlib.sha256(pd.util.hash_pandas_object(X, index=False).values.tobytes() + y.tobytes()
                              + json.dumps(FEATS).encode()).hexdigest()
    if os.path.isfile(STUDY_PKL):
        study = pickle.load(open(STUDY_PKL, 'rb'))
        st.check(study.user_attrs.get('data_sig') == data_sig,
                 'сохранённое исследование Optuna построено на тех же данных (иначе удалите models/optuna_study.pkl)')
        print(f'Продолжение исследования: завершено попыток {len(study.trials)}')
    else:
        study = optuna.create_study(direction='minimize', study_name='lgbm_main',
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
        study.set_user_attr('data_sig', data_sig)
    st.log('data_signature', data_sig[:16])

    def suggest(trial):
        depth = trial.suggest_int('max_depth', 3, 12)
        p = dict(num_leaves=trial.suggest_int('num_leaves', 8, 128, log=True), max_depth=depth,
                 min_child_samples=trial.suggest_int('min_child_samples', 20, 400, log=True),
                 feature_fraction=trial.suggest_float('feature_fraction', 0.3, 1.0),
                 bagging_fraction=trial.suggest_float('bagging_fraction', 0.5, 1.0),
                 lambda_l1=trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
                 lambda_l2=trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
                 min_gain_to_split=trial.suggest_float('min_gain_to_split', 0.0, 2.0),
                 cat_smooth=trial.suggest_float('cat_smooth', 1.0, 100.0, log=True))
        p['num_leaves'] = min(p['num_leaves'], 2 ** depth)
        return p

    def objective(trial):
        t = time.time()
        n, ll, _ = cv_lgb(X, y, FOLDS, suggest(trial))
        trial.set_user_attr('n_trees', n)
        trial.set_user_attr('seconds', round(time.time() - t, 1))
        return ll

    def save_cb(study_, trial):
        pickle.dump(study_, open(STUDY_PKL, 'wb'))
        print(f'  попытка {trial.number + 1:3d}/{N_TRIALS}: log-loss {trial.value:.5f}, деревьев '
              f'{trial.user_attrs["n_trees"]}, {trial.user_attrs["seconds"]} с | лучший {study_.best_value:.5f}',
              flush=True)

    done = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if done < N_TRIALS:
        study.optimize(objective, n_trials=N_TRIALS - done, callbacks=[save_cb])
    best = study.best_trial
    P = {k: v for k, v in best.params.items()}
    P['num_leaves'] = min(P['num_leaves'], 2 ** P['max_depth'])
    N_MAIN = int(best.user_attrs['n_trees'])
    pd.DataFrame([{**t.params, 'logloss': t.value, **t.user_attrs, 'number': t.number} for t in study.trials]
                 ).to_csv(f'{DATA}/stage7_optuna_trials.csv', index=False)
    st.log('optuna', {'trials_complete': len(study.trials), 'best_logloss': round(best.value, 5),
                      'best_trial': best.number, 'n_trees': N_MAIN, 'params': P})

    # ------------------------------------------------------------------
    st.section('3. Основная модель: out-of-fold, SHAP, итоговое обучение')
    oof_main, shap_main = oof_lgb(X, y, FOLDS, P, N_MAIN, shap=True)
    auc_main = roc_auc_score(y, oof_main)
    st.log('oof_lgbm', {'auroc': round(auc_main, 4), 'logloss': round(log_loss(y, oof_main), 5)})
    fit_lgb(X, y, P, N_MAIN, f'{MODELS}/lgbm_main.txt')
    shap_rank = pd.Series(np.abs(shap_main).mean(axis=0), index=FEATS).sort_values(ascending=False)
    shap_rank.rename('mean_abs_shap_oof').to_csv(f'{DATA}/stage7_shap_oof.csv', index_label='feature')
    print('Топ-20 признаков по среднему |SHAP| (out-of-fold):\n' + shap_rank.head(20).round(4).to_string())

    # ------------------------------------------------------------------
    st.section('4. Логистическая регрессия со сплайнами')
    num = [c for c in FEATS if c not in CATS]
    cont = [c for c in num if tr[c].nunique() > 10]
    disc = [c for c in num if c not in cont]
    Xl = prep_lr(tr, FEATS, CATS)
    ll_grid = np.zeros((len(C_GRID), N_FOLDS))
    oof_lr_by_c = np.zeros((len(C_GRID), len(y)))
    for k, (tri, vai) in enumerate(FOLDS):
        pre = Pipeline([('pre', lr_preprocessor(cont, disc, CATS)), ('sc', StandardScaler())])
        A = pre.fit_transform(Xl.iloc[tri])
        B = pre.transform(Xl.iloc[vai])
        for i, C in enumerate(C_GRID):
            m = LogisticRegression(C=C, max_iter=5000).fit(A, y[tri])
            p = m.predict_proba(B)[:, 1]
            oof_lr_by_c[i, vai] = p
            ll_grid[i, k] = log_loss(y[vai], p)
        print(f'  фолд {k + 1}/{N_FOLDS}: базисных столбцов {A.shape[1]}', flush=True)
    i_best = int(ll_grid.mean(axis=1).argmin())
    C_BEST = float(C_GRID[i_best])
    oof_lr = oof_lr_by_c[i_best]
    st.log('lr_spline', {'C_grid_mean_logloss': {f'{c:.0e}': round(float(v), 5) for c, v in zip(C_GRID, ll_grid.mean(axis=1))},
                         'C_best': C_BEST, 'oof_auroc': round(float(roc_auc_score(y, oof_lr)), 4),
                         'oof_logloss': round(float(log_loss(y, oof_lr)), 5), 'n_cont': len(cont), 'n_disc': len(disc)})
    lr_model = Pipeline([('pre', lr_preprocessor(cont, disc, CATS)), ('sc', StandardScaler()),
                         ('lr', LogisticRegression(C=C_BEST, max_iter=5000))]).fit(Xl, y)
    joblib.dump(lr_model, f'{MODELS}/lr_spline.joblib')

    # ------------------------------------------------------------------
    st.section('5. Экономная модель')
    PARS_YEARS = sorted(tr['est_year'].unique())[-PARS_N_YEARS:]
    st.log('parsimony_fill_years', [int(v) for v in PARS_YEARS])
    fill = {c: min(tr.loc[tr['est_year'] == yr, c].notna().mean() for yr in PARS_YEARS) for c in FEATS}
    cand = [c for c in shap_rank.index if fill[c] > PARS_FILL]
    st.log('parsimony_candidates', {'n': len(cand), 'excluded_low_fill': [c for c in FEATS if c not in cand]})
    n_pool, ll_pool, auc_pool = cv_lgb(X[cand], y, FOLDS, P, with_auc=True)
    rows = [{'size': len(cand), 'n_trees': n_pool, 'cv_logloss': ll_pool, 'cv_auroc': auc_pool, 'set': 'all_candidates'}]
    for s in PARS_SIZES:
        if s >= len(cand):
            continue
        n_s, ll_s, auc_s = cv_lgb(X[cand[:s]], y, FOLDS, P, with_auc=True)
        rows.append({'size': s, 'n_trees': n_s, 'cv_logloss': ll_s, 'cv_auroc': auc_s, 'set': f'top{s}'})
        print(f'  {s:2d} признаков: AUROC(CV) {auc_s:.4f} (все кандидаты {auc_pool:.4f})', flush=True)
    pars = pd.DataFrame(rows)
    pars['within_tol'] = pars['cv_auroc'] >= auc_pool - PARS_TOL
    pars.to_csv(f'{DATA}/stage7_parsimony.csv', index=False)
    ok = pars[(pars['set'] != 'all_candidates') & pars['within_tol']]
    S_PARS = int(ok['size'].min()) if len(ok) else len(cand)
    F_PARS = cand[:S_PARS]
    N_PARS = int(pars.loc[pars['size'] == S_PARS, 'n_trees'].iloc[0])
    oof_pars, _ = oof_lgb(X[F_PARS], y, FOLDS, P, N_PARS)
    fit_lgb(X[F_PARS], y, P, N_PARS, f'{MODELS}/lgbm_parsimonious.txt')
    st.log('parsimonious', {'size': S_PARS, 'n_trees': N_PARS, 'features': F_PARS,
                            'oof_auroc': round(roc_auc_score(y, oof_pars), 4)})

    # ------------------------------------------------------------------
    st.section('6. Перекалибровка SAPS II и OASIS (обучающий период)')
    bm = pd.read_parquet(f'{DATA}/stage6b_benchmarks.parquet')
    tb = tr[['stay_id']].merge(bm, on='stay_id', how='left')
    recal, oof_bm = {}, {}
    for s_ in ['sapsii', 'oasis']:
        ok_ = tb[s_].notna().to_numpy()
        m = LogisticRegression(penalty=None, max_iter=1000).fit(tb.loc[ok_, [s_]].to_numpy(), y[ok_])
        recal[s_] = {'intercept': float(m.intercept_[0]), 'slope': float(m.coef_[0, 0]),
                     'train_auroc': round(float(roc_auc_score(y[ok_], tb.loc[ok_, s_])), 4)}
        p = np.full(len(y), np.nan)
        p[ok_] = m.predict_proba(tb.loc[ok_, [s_]].to_numpy())[:, 1]
        oof_bm[s_] = p
    st.log('recalibration', recal)

    # ------------------------------------------------------------------
    st.section('7. Сохранение и фиксация')
    pd.DataFrame({'stay_id': tr['stay_id'], 'fold': fold, 'y': y, 'p_lgbm': oof_main, 'p_lr': oof_lr,
                  'p_parsimonious': oof_pars, 'p_sapsii_recal': oof_bm['sapsii'],
                  'p_oasis_recal': oof_bm['oasis']}).to_parquet(f'{DATA}/stage7_oof.parquet', index=False)
    frozen = {
        'seed': SEED, 'base_params': BASE, 'lgbm_params': P, 'n_trees_main': N_MAIN,
        'features_main': FEATS, 'categorical': CATS, 'cat_levels': cat_levels,
        'lr': {'C': C_BEST, 'continuous': cont, 'discrete': disc},
        'parsimonious': {'features': F_PARS, 'n_trees': N_PARS, 'tolerance': PARS_TOL},
        'recalibration': recal,
        'oof': {'lgbm_auroc': round(auc_main, 4), 'lr_auroc': round(roc_auc_score(y, oof_lr), 4)},
    }
    frozen_path = f'{MODELS}/stage7_frozen.json'
    with open(frozen_path, 'w', encoding='utf-8') as f:
        json.dump(frozen, f, ensure_ascii=False, indent=2)
    files = sorted(f for f in os.listdir(MODELS) if f.endswith(('.txt', '.joblib', '.json')) and f != 'stage7_checksums.json')
    checks = {f: sha256(f'{MODELS}/{f}') for f in files}
    with open(f'{MODELS}/stage7_checksums.json', 'w', encoding='utf-8') as f:
        json.dump(checks, f, indent=2)
    st.log('sha256', checks)

    st.check(len(study.trials) >= N_TRIALS, f'выполнено {N_TRIALS} попыток Optuna')
    st.check(N_MAIN < MAX_ROUNDS, 'число деревьев не упёрлось в максимум')
    st.check(auc_main >= 0.75, 'AUROC out-of-fold основной модели ≥ 0,75 (санитарная проверка)')
    st.check(len(checks) >= 4, 'все модели и параметры сохранены и зафиксированы контрольными суммами')
