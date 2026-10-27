# -*- coding: utf-8 -*-
"""
ЭТАП 8б. Сводная диаграмма SHAP (beeswarm) основной модели LightGBM на тестовой выборке — рисунок для статьи.

  * 30 признаков с наибольшим средним |SHAP| на тесте (тот же рейтинг, что в stage8_shap_ranking.csv).
  * Каждая точка — пациент; цвет — величина признака (процентильный ранг внутри признака),
    серый — категориальные признаки и пропущенные значения.
  * Звёздочка (*) — признак не вошёл в сокращённую (экономную) модель.

Вход: models/lgbm_main.txt, models/stage7_frozen.json, stage6_master.parquet
Выход: results/stage8_shap_beeswarm_test.png
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import lightgbm as lgb

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
sys.path.insert(0, _HERE)
from common import DATA, ROOT
from modeling import prep_lgb

MODELS = f'{ROOT}/models'
RESULTS = f'{ROOT}/results'
TOP_N = 30

LABELS = {
    'age': 'Возраст', 'sex_male': 'Мужской пол', 'admission_group': 'Тип госпитализации',
    'careunit_group': 'Тип ОРИТ', 'weight_kg': 'Масса тела', 'height_cm': 'Рост',
    'sofa_t0': 'SOFA', 'sofa_delta_24h': 'Изменение SOFA за 24 ч', 'resp_support_t0': 'Дыхательная поддержка',
    'imv_hours_24h': 'Продолжительность ИВЛ', 'nee_t0': 'Норадреналиновый эквивалент',
    'nee_max_24h': 'Норадреналиновый эквивалент (макс.)', 'vasopressor_hours_24h': 'Продолжительность инфузии вазопрессоров',
    'propofol_t0': 'Пропофол', 'midazolam_t0': 'Мидазолам', 'dexmedetomidine_t0': 'Дексмедетомидин',
    'fentanyl_t0': 'Фентанил', 'rrt_24h': 'ЗПТ', 'uo_mlkgh_24h': 'Диурез за 24 ч',
    'wbc_last': 'Лейкоциты', 'hemoglobin_last': 'Гемоглобин', 'platelets_last': 'Тромбоциты',
    'platelets_delta': 'Изменение тромбоцитов', 'rdw_last': 'Ширина распределения эритроцитов',
    'neut_abs_last': 'Нейтрофилы', 'lymph_abs_last': 'Лимфоциты', 'mono_abs_last': 'Моноциты',
    'eos_abs_last': 'Эозинофилы', 'mcv_last': 'Средний объём эритроцита', 'sodium_last': 'Натрий',
    'potassium_last': 'Калий', 'chloride_last': 'Хлор', 'bicarbonate_last': 'Бикарбонат',
    'bun_last': 'Азот мочевины крови', 'creatinine_last': 'Креатинин', 'creatinine_delta': 'Изменение креатинина',
    'glucose_last': 'Глюкоза', 'calcium_last': 'Кальций общий', 'aniongap_last': 'Анионный интервал',
    'albumin_last': 'Альбумин', 'magnesium_last': 'Магний', 'phosphate_last': 'Фосфат', 'inr_last': 'МНО',
    'ptt_last': 'АЧТВ', 'fibrinogen_last': 'Фибриноген', 'alt_last': 'АЛТ', 'ast_last': 'АСТ', 'alp_last': 'ЩФ',
    'bilirubin_last': 'Билирубин', 'ldh_last': 'ЛДГ', 'troponin_t_last': 'Тропонин Т', 'lactate_last': 'Лактат',
    'lactate_delta': 'Изменение лактата', 'ph_last': 'pH', 'base_excess_last': 'Дефицит оснований',
    'ionized_ca_last': 'Кальций ионизированный', 'pao2_last': 'PaO₂', 'paco2_last': 'PaCO₂',
    'pf_ratio_last': 'PaO₂/FiO₂', 'heart_rate_last': 'ЧСС', 'heart_rate_delta': 'Изменение ЧСС',
    'heart_rate_range': 'Размах ЧСС', 'resp_rate_last': 'ЧД', 'resp_rate_delta': 'Изменение ЧД',
    'resp_rate_range': 'Размах ЧД', 'temperature_last': 'Температура', 'temperature_delta': 'Изменение температуры',
    'temperature_range': 'Размах температуры', 'sbp_last': 'Систолическое АД', 'sbp_delta': 'Изменение САД',
    'sbp_range': 'Размах САД', 'mbp_last': 'Среднее АД', 'mbp_delta': 'Изменение среднего АД',
    'mbp_range': 'Размах среднего АД', 'dbp_last': 'Диастолическое АД', 'dbp_delta': 'Изменение ДАД',
    'dbp_range': 'Размах ДАД', 'spo2_last': 'Сатурация по пульсоксиметру', 'spo2_delta': 'Изменение SpO₂',
    'spo2_range': 'Размах SpO₂', 'gcs_total_last': 'ШКГ (сумма)', 'gcs_motor_last': 'ШКГ: двигательный ответ',
    'gcs_eyes_last': 'ШКГ: открывание глаз', 'gcs_verbal_last': 'ШКГ: речевой ответ', 'fio2_last': 'Фракция кислорода',
}


def swarm_offsets(x, half_width=0.4, n_bins=120):
    """Вертикальные смещения точек для пчелиного роя: точки одного интервала по оси X раскладываются
    симметрично вверх и вниз; масштаб — по самому плотному интервалу."""
    x = np.asarray(x, float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    bins = np.clip(((x - lo) / (hi - lo + 1e-12) * n_bins).astype(int), 0, n_bins - 1)
    off = np.zeros(len(x))
    counts = np.bincount(bins, minlength=n_bins)
    step = half_width / max(counts.max() / 2, 1)
    rng = np.random.default_rng(0)
    for b in np.nonzero(counts)[0]:
        idx = np.nonzero(bins == b)[0]
        idx = idx[rng.permutation(len(idx))]
        k = np.arange(len(idx))
        off[idx] = ((k + 1) // 2) * np.where(k % 2 == 0, 1, -1) * step
    return off


def beeswarm(shap_df, X, cats, pars_feats, path, top_n=TOP_N):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    import matplotlib.ticker

    rank = shap_df.abs().mean().sort_values(ascending=False).index[:top_n]
    cmap = plt.get_cmap('coolwarm')
    fig, ax = plt.subplots(figsize=(8.5, 0.34 * top_n + 1.6))
    for i, f in enumerate(rank):
        y0 = top_n - 1 - i
        s = shap_df[f].to_numpy(float)
        off = swarm_offsets(s)
        v = X[f]
        if f in cats or hasattr(v, 'cat'):
            col = np.tile(np.array([[0.6, 0.6, 0.6, 1.0]]), (len(s), 1))
        else:
            v = pd.to_numeric(v, errors='coerce')
            pr = v.rank(pct=True).to_numpy()
            col = cmap(np.nan_to_num(pr, nan=0.5))
            col[np.isnan(pr)] = [0.75, 0.75, 0.75, 1.0]
        order = np.argsort(np.abs(off))[::-1]           # плотная середина рисуется поверх
        ax.scatter(s[order], y0 + off[order], c=col[order], s=3, linewidths=0, rasterized=True)
    ax.axvline(0, color='grey', lw=0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([LABELS.get(f, f) + (' *' if f not in pars_feats else '') for f in rank][::-1])
    ax.set_ylim(-0.7, top_n - 0.3)
    ax.set_xlabel('Значение SHAP (влияние на прогноз, лог-шансы)')
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda t, _: f'{t:g}'.replace('.', ',').replace('-', '−')))
    for sp in ('top', 'right', 'left'):
        ax.spines[sp].set_visible(False)
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=cmap), ax=ax, fraction=0.03, pad=0.02, aspect=30)
    cb.set_ticks([0, 1]); cb.set_ticklabels(['Низкое', 'Высокое']); cb.set_label('Значение признака')
    cb.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return list(rank)


if __name__ == '__main__':
    fr = json.load(open(f'{MODELS}/stage7_frozen.json', encoding='utf-8'))
    FEATS, CATS, LV = fr['features_main'], fr['categorical'], fr['cat_levels']
    M = pd.read_parquet(f'{DATA}/stage6_master.parquet')
    te = M[(M['in_main_cohort'] == 1) & (M['main_analysis'] == 1) & (M['split'] == 'test')].reset_index(drop=True)
    Xte = prep_lgb(te, FEATS, LV)
    booster = lgb.Booster(model_file=f'{MODELS}/lgbm_main.txt')
    sh = pd.DataFrame(booster.predict(Xte, pred_contrib=True)[:, :-1], columns=FEATS)
    out = f'{RESULTS}/stage8_shap_beeswarm_test.png'
    top = beeswarm(sh, Xte, set(CATS), set(fr['parsimonious']['features']), out)
    print(f'n = {len(te)}; сохранено: {out}')
    print('Топ-30:', ', '.join(f + ('*' if f not in fr['parsimonious']['features'] else '') for f in top))
    miss = [f for f in top if f not in LABELS]
    if miss:
        print('Нет русской подписи для:', miss)
