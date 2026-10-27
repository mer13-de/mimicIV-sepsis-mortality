# -*- coding: utf-8 -*-
"""
ЭТАП 8в. Рисунки 2 и 3 для статьи (калибровка и кривая решений) в оформлении рукописи:
русские подписи, десятичная запятая, легенда с «Идеальной калибровкой», гистограмма с подписью оси.
Модели не пересчитываются — используются сохранённые прогнозы этапа 8.

Вход: data/stage8_test_predictions.parquet, results/stage8_dca_test.csv
Выход: results/fig2_calibration.png, results/fig3_dca.png
"""
import os
import sys
import numpy as np
import pandas as pd

try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _HERE = os.getcwd()
sys.path.insert(0, _HERE)
from common import DATA, ROOT

try:
    from evaluation import lowess_fit          # та же сглаживающая функция, что на этапе 8
except Exception:
    from statsmodels.nonparametric.smoothers_lowess import lowess

    def lowess_fit(p, y):
        p = np.asarray(p, float)
        return lowess(np.asarray(y, float), p, frac=2 / 3, return_sorted=False)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

RESULTS = f'{ROOT}/results'
MODELS = [('lgbm', 'LightGBM'), ('lr_spline', 'ЛР со сплайнами'), ('parsimonious', 'Сокращённая модель'),
          ('sapsii', 'SAPS II'), ('oasis', 'OASIS')]
COLORS = {'lgbm': '#1f77b4', 'lr_spline': '#ff7f0e', 'parsimonious': '#2ca02c', 'sapsii': '#d62728', 'oasis': '#9467bd'}


def ru(dec):
    return FuncFormatter(lambda v, _: (f'{v:.{dec}f}'.replace('.', ',').replace('-', '−')) if v != 0 else '0')


def despine(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)


def calibration(pred, path):
    y = pred['y'].to_numpy(int)
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(6.5, 7.2), sharex=True,
                                  gridspec_kw={'height_ratios': [4.2, 1.2], 'hspace': 0.08})
    ax.plot([0, 1], [0, 1], ls='--', c='grey', lw=1.2, label='Идеальная калибровка')
    for k, lab in MODELS:
        p = pred[f'p_{k}'].to_numpy(float)
        ok = ~np.isnan(p)
        o = np.argsort(p[ok])
        ax.plot(p[ok][o], np.clip(lowess_fit(p[ok], y[ok])[o], 0, 1), lw=2, c=COLORS[k], label=lab)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_ylabel('Наблюдаемая летальность')
    ax.legend(frameon=False, loc='upper left')
    ax.yaxis.set_major_formatter(ru(1))
    despine(ax)
    axh.hist(pred['p_lgbm'].dropna(), bins=np.linspace(0, 1, 45), color='#7aadd3')
    axh.set_ylabel('Число пациентов\n(LightGBM)', fontsize=9)
    axh.set_xlabel('Прогнозируемая вероятность смерти')
    axh.xaxis.set_major_formatter(ru(1))
    despine(axh)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def dca(tab, path):
    t = tab['threshold'].to_numpy(float)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(t, tab['treat_all'], c='grey', lw=1.2, label='Лечить всех')
    ax.axhline(0, c='black', lw=1.2, label='Не лечить никого')
    for k, lab in MODELS:
        ax.plot(t, tab[k], lw=2, c=COLORS[k], label=lab)
    ax.set_xlim(t.min(), t.max())
    ax.set_ylim(-0.02, max(0.30, float(tab[[k for k, _ in MODELS]].max().max()) + 0.02))
    ax.set_xlabel('Пороговая вероятность'); ax.set_ylabel('Чистая выгода')
    ax.xaxis.set_major_formatter(ru(1)); ax.yaxis.set_major_formatter(ru(2))
    ax.legend(frameon=False, loc='upper right')
    despine(ax)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    pred = pd.read_parquet(f'{DATA}/stage8_test_predictions.parquet')
    tab = pd.read_csv(f'{RESULTS}/stage8_dca_test.csv')
    calibration(pred, f'{RESULTS}/fig2_calibration.png')
    dca(tab, f'{RESULTS}/fig3_dca.png')
    print(f'n = {len(pred)}, умерло = {int(pred["y"].sum())}')
    print(f'Сохранено: {RESULTS}/fig2_calibration.png, {RESULTS}/fig3_dca.png')
