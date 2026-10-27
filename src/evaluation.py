# -*- coding: utf-8 -*-
"""Метрики оценки для этапов 8–9: дискриминация, калибровка, DeLong, бутстреп, кривая принятия решений."""
import numpy as np
from scipy.stats import norm, rankdata
from sklearn.metrics import roc_auc_score, average_precision_score

EPS = 1e-6


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def logistic_fit(y, x=None, offset=None, n_iter=100):
    """Логистическая регрессия Ньютоном: y ~ 1 (+ x) (+ offset). Возвращает (коэффициенты, ковариация)."""
    y = np.asarray(y, float)
    X = np.ones((len(y), 1)) if x is None else np.column_stack([np.ones(len(y)), x])
    off = np.zeros(len(y)) if offset is None else np.asarray(offset, float)
    b = np.zeros(X.shape[1])
    for _ in range(n_iter):
        mu = 1 / (1 + np.exp(-(X @ b + off)))
        W = mu * (1 - mu)
        H = X.T @ (X * W[:, None])
        step = np.linalg.solve(H, X.T @ (y - mu))
        b = b + step
        if np.max(np.abs(step)) < 1e-10:
            break
    mu = 1 / (1 + np.exp(-(X @ b + off)))
    cov = np.linalg.inv(X.T @ (X * (mu * (1 - mu))[:, None]))
    return b, cov


def lowess_fit(x, y, frac=0.75):
    """Локально-линейное сглаживание с трикубическими весами (одна итерация, как для бинарного исхода)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    k = max(int(np.ceil(frac * n)), 3)
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    fit = np.empty(n)
    for s in range(0, n, 256):
        xi = xs[s:s + 256][:, None]
        d = np.abs(xs[None, :] - xi)
        h = np.partition(d, k - 1, axis=1)[:, k - 1][:, None]
        h = np.where(h > 0, h, 1e-12)
        w = np.clip(1 - (d / h) ** 3, 0, None) ** 3
        sw, swx, swy = w.sum(1), (w * xs).sum(1), (w * ys).sum(1)
        swxx, swxy = (w * xs ** 2).sum(1), (w * xs * ys).sum(1)
        den = sw * swxx - swx ** 2
        slope = np.where(np.abs(den) > 1e-12, (sw * swxy - swx * swy) / np.where(den == 0, 1, den), 0)
        icpt = (swy - slope * swx) / sw
        fit[s:s + 256] = icpt + slope * xs[s:s + 256]
    out = np.empty(n)
    out[order] = np.clip(fit, 0, 1)
    return out


def calibration(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    lp = logit(p)
    b_int, c_int = logistic_fit(y, offset=lp)
    b_sl, c_sl = logistic_fit(y, x=lp)
    sm = lowess_fit(p, y)
    err = np.abs(sm - p)
    brier = float(np.mean((p - y) ** 2))
    prev = y.mean()
    return {
        'oe': float(y.sum() / p.sum()),
        'cal_intercept': float(b_int[0]),
        'cal_intercept_lo': float(b_int[0] - 1.96 * np.sqrt(c_int[0, 0])),
        'cal_intercept_hi': float(b_int[0] + 1.96 * np.sqrt(c_int[0, 0])),
        'cal_slope': float(b_sl[1]),
        'cal_slope_lo': float(b_sl[1] - 1.96 * np.sqrt(c_sl[1, 1])),
        'cal_slope_hi': float(b_sl[1] + 1.96 * np.sqrt(c_sl[1, 1])),
        'brier': brier,
        'brier_scaled': float(1 - brier / (prev * (1 - prev))),
        'ici': float(err.mean()), 'e50': float(np.median(err)), 'e90': float(np.quantile(err, 0.9)),
    }


def discrimination(y, p):
    return {'auroc': float(roc_auc_score(y, p)), 'auprc': float(average_precision_score(y, p))}


def bootstrap_ci(y, preds, n_boot=2000, seed=0, stats=('auroc', 'auprc', 'oe', 'brier')):
    """Перцентильные 95% ДИ; одни и те же выборки для всех моделей (парный бутстреп).
    preds: {имя: вероятности}. Возвращает {имя: {метрика: (lo, hi)}} и {(a, b): (lo, hi)} для разностей AUROC."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    n = len(y)
    res = {k: {s: [] for s in stats} for k in preds}
    names = list(preds)
    diffs = {(a, b): [] for i, a in enumerate(names) for b in names[i + 1:]}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():
            continue
        au = {}
        for k, p in preds.items():
            pb = np.asarray(p)[idx]
            au[k] = roc_auc_score(yb, pb)
            if 'auroc' in stats:
                res[k]['auroc'].append(au[k])
            if 'auprc' in stats:
                res[k]['auprc'].append(average_precision_score(yb, pb))
            if 'oe' in stats:
                res[k]['oe'].append(yb.sum() / pb.sum())
            if 'brier' in stats:
                res[k]['brier'].append(np.mean((pb - yb) ** 2))
        for a, b in diffs:
            diffs[(a, b)].append(au[a] - au[b])
    ci = {k: {s: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for s, v in d.items() if v}
          for k, d in res.items()}
    dci = {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in diffs.items()}
    return ci, dci


def delong(y, p1, p2):
    """Парный тест DeLong (быстрый алгоритм Sun & Xu). Возвращает (auc1, auc2, разность, p)."""
    y = np.asarray(y).astype(int)
    P = np.vstack([np.asarray(p1, float), np.asarray(p2, float)])
    pos, neg = P[:, y == 1], P[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    tx = np.vstack([rankdata(r) for r in pos])
    ty = np.vstack([rankdata(r) for r in neg])
    tz = np.vstack([rankdata(r) for r in np.hstack([pos, neg])])
    aucs = (tz[:, :m].sum(1) - m * (m + 1) / 2) / (m * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    S = np.cov(v01) / m + np.cov(v10) / n
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    diff = aucs[0] - aucs[1]
    z = diff / np.sqrt(var) if var > 0 else 0.0
    return float(aucs[0]), float(aucs[1]), float(diff), float(2 * norm.sf(abs(z)))


def holm(pvals):
    """Поправка Холма; возвращает скорректированные p в исходном порядке."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (len(p) - r) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def net_benefit(y, p, thresholds):
    y = np.asarray(y)
    p = np.asarray(p)
    n = len(y)
    out = []
    for t in thresholds:
        tp = np.sum((p >= t) & (y == 1))
        fp = np.sum((p >= t) & (y == 0))
        out.append(tp / n - fp / n * t / (1 - t))
    return np.array(out)
