# -*- coding: utf-8 -*-
"""
ЭТАП 10. Описательные таблицы и рисунки для статьи.

  * Таблица 1 — характеристики когорты основного анализа: все, обучающая и тестовая выборки, выжившие
    и умершие (p: Манна–Уитни / χ²). Для непрерывных — медиана [МКИ] и доля
    пропусков, для категориальных — n (%).
  * Таблица S — тестовые пациенты (без ограничения объёма лечения) с данными и без данных об инфузиях.
  * Таблица S — характеристики по расчётному году (сдвиг популяции во времени).
  * Схема отбора пациентов (PNG) по stage6_flowchart.json (порядок шагов и причины исключения).
  * Сводная книга Excel со всеми таблицами для статьи и приложений.

Вход: stage6_master.parquet, stage6b_benchmarks.parquet, stage6_flowchart.json, результаты этапов 6–9
Выход: results/stage10_table1.csv, results/stage10_tables.xlsx, results/stage10_flowchart.png
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
sys.path.insert(0, _HERE)
from common import Stage, DATA, ROOT

RESULTS = f'{ROOT}/results'
MODELS = f'{ROOT}/models'
os.makedirs(RESULTS, exist_ok=True)

LEVELS = {
    'race_group': {'white': 'белая', 'black': 'чернокожая', 'hispanic': 'латиноамериканская', 'asian': 'азиатская',
                   'other': 'другая', 'unknown': 'не указана'},
    'admission_group': {'emergency_urgent': 'экстренная/срочная', 'elective': 'плановая', 'observation': 'наблюдение'},
    'careunit_group': {'medical': 'терапевтическое', 'surgical_trauma': 'хирургическое/травматологическое',
                       'cardiac': 'кардиологическое', 'mixed': 'смешанное', 'neuro': 'неврологическое'},
    'resp_support_t0': {'invasive': 'ИВЛ', 'noninvasive': 'НИВЛ', 'hfnc': 'высокопоточная оксигенотерапия',
                        'oxygen': 'обычная оксигенотерапия', 'none': 'без поддержки'},
}
# (раздел, подпись, столбец, тип, знаков после запятой)
SPEC = [
    ('Демография', 'Возраст, лет', 'age', 'cont', 0),
    ('Демография', 'Мужской пол', 'sex_male', 'bin', 0),
    ('Демография', 'Тип госпитализации', 'admission_group', 'cat', 0),
    ('Демография', 'Тип отделения', 'careunit_group', 'cat', 0),
    ('Демография', 'Отделение не уровня ОРИТ', 'non_icu_level_unit', 'bin', 0),
    ('Антропометрия', 'Масса тела, кг', 'weight_kg', 'cont', 1),
    ('Антропометрия', 'Рост, см', 'height_cm', 'cont', 0),
    ('Сопутствующие заболевания (коды выписки)', 'Индекс Чарлсон без возраста', 'cci_no_age', 'cont', 0),
    ('Сопутствующие заболевания (коды выписки)', 'Хроническая сердечная недостаточность', 'cm_chf', 'bin', 0),
    ('Сопутствующие заболевания (коды выписки)', 'ХОБЛ / хронические заболевания лёгких', 'cm_copd', 'bin', 0),
    ('Сопутствующие заболевания (коды выписки)', 'Сахарный диабет', 'cm_diabetes', 'bin', 0),
    ('Сопутствующие заболевания (коды выписки)', 'Хроническая болезнь почек', 'cm_ckd', 'bin', 0),
    ('Сопутствующие заболевания (коды выписки)', 'Заболевания печени', 'cm_liver', 'bin', 0),
    ('Сопутствующие заболевания (коды выписки)', 'Злокачественное новообразование', 'cm_malignancy', 'bin', 0),
    ('Сопутствующие заболевания (коды выписки)', 'Метастатическая опухоль', 'cm_metastatic', 'bin', 0),
    ('Сопутствующие заболевания (коды выписки)', 'Деменция', 'cm_dementia', 'bin', 0),
    ('Сопутствующие заболевания (коды выписки)', 'COVID-19 (U07.1)', 'covid19_dx', 'bin', 0),
    ('Тяжесть', 'SOFA на T0', 'sofa_t0', 'cont', 0),
    ('Тяжесть', 'SAPS II (первые сутки)', 'sapsii', 'cont', 0),
    ('Тяжесть', 'OASIS (первые сутки)', 'oasis', 'cont', 0),
    ('Органная поддержка на T0', 'Дыхательная поддержка', 'resp_support_t0', 'cat', 0),
    ('Органная поддержка на T0', 'Вазопрессоры', 'vasopressor_t0', 'bin', 0),
    ('Органная поддержка на T0', 'Норадреналиновый эквивалент у получавших, мкг/кг/мин', 'nee_t0', 'cont_pos', 2),
    ('Органная поддержка на T0', 'Седация', 'any_sedative_t0', 'bin', 0),
    ('Органная поддержка на T0', 'ЗПТ за 24 ч', 'rrt_24h', 'bin', 0),
    ('Лабораторные показатели (последние до T0)', 'Лактат, ммоль/л', 'lactate_last', 'cont', 1),
    ('Лабораторные показатели (последние до T0)', 'Креатинин, мг/дл', 'creatinine_last', 'cont', 1),
    ('Лабораторные показатели (последние до T0)', 'Мочевина (BUN), мг/дл', 'bun_last', 'cont', 0),
    ('Лабораторные показатели (последние до T0)', 'Билирубин, мг/дл', 'bilirubin_last', 'cont', 1),
    ('Лабораторные показатели (последние до T0)', 'Тромбоциты, ×10⁹/л', 'platelets_last', 'cont', 0),
    ('Лабораторные показатели (последние до T0)', 'Лейкоциты, ×10⁹/л', 'wbc_last', 'cont', 1),
    ('Лабораторные показатели (последние до T0)', 'RDW, %', 'rdw_last', 'cont', 1),
    ('Лабораторные показатели (последние до T0)', 'pH', 'ph_last', 'cont', 2),
    ('Лабораторные показатели (последние до T0)', 'PaO₂/FiO₂', 'pf_ratio_last', 'cont', 0),
    ('Витальные показатели и ШКГ (последние до T0)', 'Среднее АД, мм рт. ст.', 'mbp_last', 'cont', 0),
    ('Витальные показатели и ШКГ (последние до T0)', 'ЧСС, уд/мин', 'heart_rate_last', 'cont', 0),
    ('Витальные показатели и ШКГ (последние до T0)', 'ЧД, в мин', 'resp_rate_last', 'cont', 0),
    ('Витальные показатели и ШКГ (последние до T0)', 'Температура, °C', 'temperature_last', 'cont', 1),
    ('Витальные показатели и ШКГ (последние до T0)', 'Сумма ШКГ (если оценима)', 'gcs_total_last', 'cont', 0),
    ('Время', 'От поступления в ОРИТ до T0, ч', 'icu_to_t0_h', 'cont', 1),
    ('Время', 'От госпитализации до T0, ч', 'hosp_to_t0_h', 'cont', 1),
    ('Исход', 'Летальность за 30 суток', 'death_30d', 'bin', 0),
]


def ru(v, d=1):
    """Число с десятичной запятой."""
    return f'{v:.{d}f}'.replace('.', ',')


def nfmt(k, pct):
    """n (%) с пробелом-разделителем тысяч и десятичной запятой."""
    return f'{int(k):,}'.replace(',', ' ') + f' ({ru(pct)})'


def fmt_cont(s, d):
    s = s.dropna()
    if not len(s):
        return '—'
    q = s.quantile([.5, .25, .75]).to_numpy()
    return f'{ru(q[0], d)} [{ru(q[1], d)}–{ru(q[2], d)}]'


def smd_cont(a, b):
    a, b = a.dropna(), b.dropna()
    sd = np.sqrt((a.var() + b.var()) / 2)
    return abs(a.mean() - b.mean()) / sd if sd > 0 else 0.0


def smd_bin(p1, p2):
    sd = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    return abs(p1 - p2) / sd if sd > 0 else 0.0


def table(df, groups, pair_smd=None, pair_p=None):
    """groups: {подпись столбца: маска}. pair_smd / pair_p: пары подписей для SMD и p."""
    rows = []
    for section, label, col, typ, d in SPEC:
        if col not in df:
            continue
        base = {'Раздел': section}
        if typ in ('cont', 'cont_pos'):
            x = pd.to_numeric(df[col], errors='coerce')
            if typ == 'cont_pos':
                x = x.where(x > 0)
            r = {**base, 'Показатель': label}
            for g, m in groups.items():
                r[g] = fmt_cont(x[m], d)
            if typ == 'cont':
                r['Пропуски, %'] = ru(x.isna().mean() * 100) if x.isna().any() else ''
            if pair_smd:
                r['SMD'] = ru(smd_cont(x[groups[pair_smd[0]]], x[groups[pair_smd[1]]]), 2)
            if pair_p:
                a, b = x[groups[pair_p[0]]].dropna(), x[groups[pair_p[1]]].dropna()
                r['p'] = mannwhitneyu(a, b).pvalue if len(a) and len(b) else np.nan
            rows.append(r)
        elif typ == 'bin':
            x = pd.to_numeric(df[col], errors='coerce').fillna(0)
            r = {**base, 'Показатель': label}
            for g, m in groups.items():
                r[g] = nfmt(x[m].sum(), x[m].mean() * 100)
            if pair_smd:
                r['SMD'] = ru(smd_bin(x[groups[pair_smd[0]]].mean(), x[groups[pair_smd[1]]].mean()), 2)
            if pair_p:
                ct = pd.crosstab(np.where(groups[pair_p[0]], 0, np.where(groups[pair_p[1]], 1, -1)), x.to_numpy())
                ct = ct.loc[[i for i in (0, 1) if i in ct.index]]
                r['p'] = chi2_contingency(ct)[1] if ct.shape == (2, 2) else np.nan
            rows.append(r)
        else:
            x = df[col].astype('string').fillna('нет данных')
            lv = LEVELS.get(col, {})
            order = [k for k in lv if (x == k).any()] + sorted(set(x.unique()) - set(lv))
            head = {**base, 'Показатель': label}
            if pair_p:
                m = groups[pair_p[0]] | groups[pair_p[1]]
                ct = pd.crosstab(np.asarray(groups[pair_p[1]])[m], x[m].to_numpy())
                head['p'] = chi2_contingency(ct)[1] if ct.shape[0] == 2 and ct.shape[1] > 1 else np.nan
            rows.append(head)
            for k in order:
                r = {**base, 'Показатель': f'  {lv.get(k, k)}'}
                for g, m in groups.items():
                    r[g] = nfmt((x[m] == k).sum(), (x[m] == k).mean() * 100)
                if pair_smd:
                    r['SMD'] = ru(smd_bin((x[groups[pair_smd[0]]] == k).mean(), (x[groups[pair_smd[1]]] == k).mean()), 2)
                if pair_p:   # каждая категория против остальных (χ² 2×2)
                    m = groups[pair_p[0]] | groups[pair_p[1]]
                    ct = pd.crosstab(np.asarray(groups[pair_p[1]])[m], (x[m] == k).to_numpy())
                    r['p'] = chi2_contingency(ct)[1] if ct.shape == (2, 2) else np.nan
                rows.append(r)
    t = pd.DataFrame(rows)
    if 'p' in t:
        t['p'] = t['p'].map(lambda v: '' if pd.isna(v) else ('< 0,001' if v < 0.001 else f'{v:.3f}'.replace('.', ',')))
    return t


EXCL_LABEL = {   # начало названия шага в stage6_flowchart.json -> подпись в блоке исключений
    'Первый эпизод': 'повторные эпизоды сепсиса',
    'T0 внутри': 'T0 вне ОРИТ',
    'Смерть не ранее': 'смерть до T0',
    'Без ограничения': 'ограничение объёма лечения на T0*',
    'Данные об инфузиях': 'нет данных об инфузиях за 24 ч до T0',
    'Однозначный период': 'период госпитализации не определён однозначно',
}


def plural(k, one, few, many):
    """Согласование существительного с числом: 1 пациент, 2 пациента, 5 пациентов."""
    k = int(k) % 100
    if 11 <= k <= 14:
        return many
    return {1: one, 2: few, 3: few, 4: few}.get(k % 10, many)


def flowchart(flow, path, mort):
    """Компактная схема: исходная когорта -> уточнение критерия -> все исключения одним блоком ->
    включены в анализ -> обучающая / тестовая выборки. mort: {'train': (n, умерло), 'test': (n, умерло)}."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib недоступен — схема пропущена')
        return
    n = lambda x: f'{int(x):,}'.replace(',', '\u00a0')
    pat = lambda x: f'{n(x)} {plural(x, "пациент", "пациента", "пациентов")}'
    epi = lambda x: f'{n(x)} {plural(x, "эпизод", "эпизода", "эпизодов")}'
    std, mod = flow['standard'], flow['modified']
    main_steps = [f for f in mod[1:] if not f['step'].startswith('  ')]
    split = [f for f in mod if f['step'].startswith('  ')]

    excl, prev = [], mod[0]
    for f in main_steps:
        key = next((k for k in EXCL_LABEL if f['step'].startswith(k)), None)
        label = EXCL_LABEL.get(key, f['step'])
        k = prev['rows'] - f['rows'] if key == 'Первый эпизод' else prev['patients'] - f['patients']
        excl.append(f'• {label}: {n(k)}')
        prev = f
    final = main_steps[-1]

    W, H = 10, 7.6
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')
    xm, xe = 3.0, 5.4   # xe — левый край блоков исключений
    box = dict(ha='center', va='center', fontsize=10, bbox=dict(boxstyle='round,pad=0.6', fc='white', ec='black'))
    ebox = dict(ha='left', va='center', multialignment='left', fontsize=9, bbox=dict(boxstyle='round,pad=0.5', fc='#f2f2f2', ec='grey'))
    arrow = dict(arrowstyle='->', lw=1, shrinkA=0, shrinkB=0)
    line = lambda x0, y0, x1, y1: ax.plot([x0, x1], [y0, y1], color='black', lw=1)

    y1, y2, y3, y4 = 7.0, 5.0, 2.65, 0.95
    ax.text(xm, y1, f"Эпизоды сепсиса (Sepsis-3) в ОРИТ,\nMIMIC-IV v3.1\n{epi(std[0]['rows'])}, {pat(std[0]['patients'])}", **box)
    ax.text(xm, y2, f"После уточнения критерия\nподозрения на инфекцию\n{epi(mod[0]['rows'])}, {pat(mod[0]['patients'])}", **box)
    ax.text(xm, y3, f"Включены в анализ\n{pat(final['patients'])}", **box)

    # исключение при уточнении критерия
    ye1 = (y1 + y2) / 2
    line(xm, y1 - 0.55, xm, ye1); ax.annotate('', xy=(xm, y2 + 0.55), xytext=(xm, ye1), arrowprops=arrow)
    ax.annotate('', xy=(xe - 0.1, ye1), xytext=(xm, ye1), arrowprops=arrow)
    ax.text(xe, ye1, f"Скрининговые посевы, несистемные\nантибиотики, периоперационная\nпрофилактика: "
                     f"{epi(std[0]['rows'] - mod[0]['rows'])}\n({pat(std[0]['patients'] - mod[0]['patients'])})", **ebox)

    # все последующие исключения одним блоком
    ye2 = (y2 + y3) / 2
    line(xm, y2 - 0.55, xm, ye2); ax.annotate('', xy=(xm, y3 + 0.4), xytext=(xm, ye2), arrowprops=arrow)
    ax.annotate('', xy=(xe - 0.1, ye2), xytext=(xm, ye2), arrowprops=arrow)
    ax.text(xe, ye2, 'Исключены последовательно:\n' + '\n'.join(excl), **ebox)

    # разделение по периоду
    xs = (1.4, 4.6)
    yb = (y3 + y4) / 2 + 0.2
    line(xm, y3 - 0.4, xm, yb); line(xs[0], yb, xs[1], yb)
    for x, f in zip(xs, split):
        kind = 'train' if 'обучающий' in f['step'] else 'test'
        label = 'Обучающая выборка\n(2008–2019 гг.)' if kind == 'train' else 'Тестовая выборка\n(2020–2022 гг.)'
        nn, dd = mort[kind]
        pct = f'{dd / nn * 100:.1f}'.replace('.', ',')
        ax.annotate('', xy=(x, y4 + 0.62), xytext=(x, yb), arrowprops=arrow)
        ax.text(x, y4, f"{label}\n{pat(f['patients'])}\nлетальность {pct}%", **box)

    ax.text(0.1, 0.05, '* Документированный до T0 отказ от СЛР и/или интубации (DNR/DNI) '
                       'либо переход к паллиативной помощи (comfort measures only).', fontsize=7.5, ha='left')
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)

with Stage('stage10_tables') as st:
    st.section('1. Данные')
    M = pd.read_parquet(f'{DATA}/stage6_master.parquet')
    M = M.merge(pd.read_parquet(f'{DATA}/stage6b_benchmarks.parquet'), on='stay_id', how='left')
    main = M[(M['in_main_cohort'] == 1) & (M['main_analysis'] == 1)].reset_index(drop=True)
    st.log('main_cohort', {'n': int(len(main)), 'train': int((main['split'] == 'train').sum()),
                           'test': int((main['split'] == 'test').sum())})

    st.section('2. Таблица 1')
    g = {'Все': np.ones(len(main), bool), 'Обучающая выборка (2008–2019 гг.)': (main['split'] == 'train').to_numpy(),
         'Тестовая выборка (2020–2022 гг.)': (main['split'] == 'test').to_numpy(),
         'Выжившие': (main['death_30d'] == 0).to_numpy(), 'Умершие': (main['death_30d'] == 1).to_numpy()}
    pair_split = ('Обучающая выборка (2008–2019 гг.)', 'Тестовая выборка (2020–2022 гг.)')
    t_split = table(main, {k: g[k] for k in list(g)[:3]}, pair_smd=pair_split, pair_p=pair_split)
    t_split = t_split.rename(columns={'p': 'p (обуч. vs тест)'})
    t_out = table(main, {k: g[k] for k in ['Выжившие', 'Умершие']}, pair_p=('Выжившие', 'Умершие'))
    t1 = t_split.merge(t_out[['Раздел', 'Показатель', 'Выжившие', 'Умершие', 'p']], on=['Раздел', 'Показатель'], how='left')
    # исход в столбцах «выжившие/умершие» не имеет смысла (0 % и 100 %)
    t1.loc[t1['Показатель'] == 'Летальность за 30 суток', ['Выжившие', 'Умершие', 'p']] = ''
    for k, m in g.items():
        t1 = t1.rename(columns={k: f'{k} (n = {int(m.sum()):,})'.replace(',', ' ')})
    t1.to_csv(f'{RESULTS}/stage10_table1.csv', index=False)
    with pd.option_context('display.width', 250, 'display.max_colwidth', 45, 'display.max_rows', 200):
        print(t1.drop(columns='Раздел').to_string(index=False))
    smd = pd.to_numeric(t_split['SMD'].str.replace(',', '.'), errors='coerce')
    st.log('smd_gt_0.1', t_split.loc[smd > 0.1, 'Показатель'].str.strip().tolist())

    st.section('3. Тест: с данными и без данных об инфузиях')
    # main_analysis теперь включает и исключение по ограничению лечения, поэтому сравниваем по has_infusion_data
    # среди тестовых пациентов без ограничения объёма лечения
    te_all = M[(M['in_main_cohort'] == 1) & (M['split'] == 'test') & (M['limited_care'] == 0)].reset_index(drop=True)
    gi = {'С данными об инфузиях': (te_all['has_infusion_data'] == 1).to_numpy(),
          'Без данных об инфузиях': (te_all['has_infusion_data'] == 0).to_numpy()}
    st.log('test_mortality_by_infusion_data', {k: round(float(te_all.loc[m, 'death_30d'].mean() * 100), 1) for k, m in gi.items()})
    t_inf = table(te_all, gi, pair_smd=tuple(gi))
    for k, m in gi.items():
        t_inf = t_inf.rename(columns={k: f'{k} (n = {int(m.sum()):,})'.replace(',', ' ')})

    st.section('4. По расчётному году')
    coh = M[M['in_main_cohort'] == 1]
    by_year = coh.groupby('est_year').agg(
        n=('stay_id', 'size'), infusion_data_pct=('has_infusion_data', lambda s: s.mean() * 100),
        mortality_pct=('death_30d', lambda s: s.mean() * 100), age_median=('age', 'median'),
        sofa_median=('sofa_t0', 'median'),
        vasopressor_pct_if_infusion_data=('vasopressor_t0', lambda s: s[coh.loc[s.index, 'has_infusion_data'] == 1].mean() * 100),
        imv_pct=('resp_support_t0', lambda s: (s.astype(str) == 'invasive').mean() * 100),
        covid_pct=('covid19_dx', lambda s: s.mean() * 100)).round(1).reset_index()
    print(by_year.to_string(index=False))

    st.section('5. Схема отбора')
    flow = json.load(open(f'{DATA}/stage6_flowchart.json', encoding='utf-8'))
    mort = {k: (int((main['split'] == k).sum()), int(main.loc[main['split'] == k, 'death_30d'].sum()))
            for k in ('train', 'test')}
    st.log('mortality_by_split', {k: {'n': v[0], 'deaths': v[1], 'pct': round(v[1] / v[0] * 100, 1)} for k, v in mort.items()})
    flowchart(flow, f'{RESULTS}/stage10_flowchart.png', mort)
    fl = pd.DataFrame([{**f, 'cohort': k} for k, v in flow.items() for f in v])
    print(fl.to_string(index=False))

    st.section('6. Сводная книга Excel')
    fr = json.load(open(f'{MODELS}/stage7_frozen.json', encoding='utf-8'))
    params = pd.DataFrame(
        [{'параметр': k, 'значение': v} for k, v in fr['lgbm_params'].items()] +
        [{'параметр': 'learning_rate', 'значение': fr['base_params']['learning_rate']},
         {'параметр': 'число деревьев (основная)', 'значение': fr['n_trees_main']},
         {'параметр': 'ЛР со сплайнами: C', 'значение': fr['lr']['C']},
         {'параметр': 'экономная модель: число признаков', 'значение': len(fr['parsimonious']['features'])},
         {'параметр': 'экономная модель: признаки', 'значение': ', '.join(fr['parsimonious']['features'])},
         {'параметр': 'экономная модель: число деревьев', 'значение': fr['parsimonious']['n_trees']}])
    sheets = {
        'Таблица 1': t1,
        'S Инфузии (тест)': t_inf,
        'S По годам': by_year,
        'S Схема отбора': fl,
        'S Пропуски': pd.read_csv(f'{DATA}/stage6_missingness.csv'),
        'S Границы очистки': pd.read_csv(f'{DATA}/stage6_clip_bounds.csv'),
        'S Параметры моделей': params,
        'S Экономная модель': pd.read_csv(f'{DATA}/stage7_parsimony.csv'),
        'Метрики': pd.read_csv(f'{RESULTS}/stage8_metrics.csv'),
        'Сравнения': pd.read_csv(f'{RESULTS}/stage8_comparisons.csv'),
        'Кривая решений': pd.read_csv(f'{RESULTS}/stage8_dca_test.csv'),
        'SHAP': pd.read_csv(f'{RESULTS}/stage8_shap_ranking.csv'),
        'Подгруппы': pd.read_csv(f'{RESULTS}/stage9_subgroups.csv'),
        'Подгруппы: сравнение AUROC': pd.read_csv(f'{RESULTS}/stage9b_subgroup_pvalues.csv'),
    }
    out = f'{RESULTS}/stage10_tables.xlsx'
    with pd.ExcelWriter(out, engine='openpyxl') as xw:
        for name, d in sheets.items():
            d.to_excel(xw, sheet_name=name[:31], index=False)
            ws = xw.sheets[name[:31]]
            for i, c in enumerate(d.columns, 1):
                width = min(max([len(str(c))] + [len(str(v)) for v in d[c].head(200)]) + 2, 60)
                ws.column_dimensions[ws.cell(1, i).column_letter].width = width
    print(f'Сохранено: {out} ({len(sheets)} листов), {RESULTS}/stage10_flowchart.png, {RESULTS}/stage10_table1.csv')
    st.check(len(t1) > 35, 'таблица 1 сформирована')
    st.check(os.path.isfile(out), 'сводная книга Excel сохранена')
