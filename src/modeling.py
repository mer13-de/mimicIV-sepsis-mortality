# -*- coding: utf-8 -*-
"""Общие для этапов 7–9 преобразования признаков (модуль нужен, чтобы сохранённые модели загружались в этапах 8–9)."""
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import SplineTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


def prep_lgb(df, cols, cat_levels):
    X = df[cols].copy()
    for c in cols:
        if c in cat_levels:
            X[c] = pd.Categorical(X[c].astype(object), categories=cat_levels[c])
        else:
            X[c] = pd.to_numeric(X[c], errors='coerce').astype(float)
    return X


class SkewLogSpline(BaseEstimator, TransformerMixin):
    """Непрерывные признаки: медиана + индикатор пропуска, log1p при асимметрии > порога, кубический сплайн
    с узлами по квантилям (дубликаты узлов удаляются; < 3 уникальных узлов — линейный член). Всё — по fit-данным."""

    def __init__(self, n_knots=4, skew_thr=2.0):
        self.n_knots = n_knots
        self.skew_thr = skew_thr

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.med_, self.log_, self.spl_, self.miss_ = [], [], [], []
        for j in range(X.shape[1]):
            col = X[:, j]
            ok = ~np.isnan(col)
            c = col[ok]
            med = float(np.median(c)) if len(c) else 0.0
            lg = bool(len(c) > 2 and c.min() >= 0 and skew(c) > self.skew_thr)
            v = np.log1p(np.where(ok, col, med)) if lg else np.where(ok, col, med)
            q = np.unique(np.quantile(v[ok] if ok.any() else v, np.linspace(0, 1, self.n_knots)))
            spl = None
            if len(q) >= 3:
                spl = SplineTransformer(knots=q.reshape(-1, 1), degree=3, extrapolation='linear').fit(v.reshape(-1, 1))
            self.med_.append(med)
            self.log_.append(lg)
            self.spl_.append(spl)
            self.miss_.append(bool((~ok).any()))
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        out = []
        for j in range(X.shape[1]):
            col = X[:, j]
            miss = np.isnan(col)
            v = np.where(miss, self.med_[j], col)
            if self.log_[j]:
                v = np.log1p(np.clip(v, 0, None))
            out.append(self.spl_[j].transform(v.reshape(-1, 1)) if self.spl_[j] is not None else v.reshape(-1, 1))
            if self.miss_[j]:
                out.append(miss.astype(float).reshape(-1, 1))
        return np.hstack(out)


def lr_preprocessor(cont, disc, cats):
    return ColumnTransformer([
        ('cont', SkewLogSpline(), cont),
        ('disc', SimpleImputer(strategy='median', add_indicator=True), disc),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cats),
    ], sparse_threshold=0)


def prep_lr(df, cols, cats):
    X = df[cols].copy()
    for c in cols:
        if c in cats:
            X[c] = X[c].astype(object).where(X[c].notna(), 'missing').astype(str)
        else:
            X[c] = pd.to_numeric(X[c], errors='coerce').astype(float)
    return X
