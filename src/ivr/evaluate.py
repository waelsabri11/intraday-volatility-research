"""Walk-forward driver, loss functions and Diebold-Mariano tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def walk_forward(n, valid, min_train, refit_every, fit_predict):
    """Expanding-window walk-forward.

    Row i holds features known at end of period i and the target for period i+1 (known at end
    of period i+1). A block starting at origin s is therefore trained ONLY on rows j <= s-1
    (whose targets are all realised by the end of period s) and forecasts origins s..s+refit-1.
    fit_predict(train_idx, test_idx) -> forecasts for test_idx.
    Returns (forecasts[n] with NaN outside OOS, list of (train_last_row, first_origin)).
    """
    out = np.full(n, np.nan)
    folds = []
    for s in range(min_train, n, refit_every):
        e = min(s + refit_every, n)
        tr = np.flatnonzero(valid[:s])
        te = np.arange(s, e)[valid[s:e]]
        if len(te) == 0 or len(tr) < 100:
            continue
        out[te] = fit_predict(tr, te)
        folds.append((int(tr.max()), s))
    return out, folds


def qlike(rv, f):
    """QLIKE (Patton 2011): rv/f - log(rv/f) - 1. Zero iff f == rv; robust loss for variance."""
    x = rv / f
    return x - np.log(x) - 1


def mse(rv, f):
    return (rv - f) ** 2


def diebold_mariano(l_bench, l_model, max_lag=None):
    """DM test of equal expected loss. d = L_bench - L_model: positive => model better.

    Newey-West (Bartlett) long-run variance, Harvey-Leybourne-Newbold small-sample correction,
    t(T-1) reference distribution. Returns (stat, two-sided p).
    """
    d = np.asarray(l_bench) - np.asarray(l_model)
    d = d[np.isfinite(d)]
    T = len(d)
    L = max_lag if max_lag is not None else int(np.floor(T ** (1 / 3)))
    dc = d - d.mean()
    lrv = dc @ dc / T
    for k in range(1, L + 1):
        w = 1 - k / (L + 1)
        lrv += 2 * w * (dc[k:] @ dc[:-k]) / T
    stat = d.mean() / np.sqrt(lrv / T)
    stat *= np.sqrt((T - 1) / T)  # HLN correction for 1-step horizon
    p = 2 * (1 - stats.t.cdf(abs(stat), df=T - 1))
    return float(stat), float(p)


def loss_table(rv: pd.Series, forecasts: dict[str, pd.Series], bench="HAR") -> pd.DataFrame:
    """Mean QLIKE/MSE per model on the common OOS sample, ratio to benchmark, DM vs benchmark."""
    df = pd.DataFrame(forecasts)
    common = df.dropna().index
    rv = rv.loc[common]
    df = df.loc[common]
    rows = []
    ql = {m: qlike(rv.values, df[m].values) for m in df}
    ms = {m: mse(rv.values, df[m].values) for m in df}
    for m in df:
        row = {"model": m, "n_oos": len(df), "QLIKE": ql[m].mean(), "MSE": ms[m].mean(),
               "QLIKE_vs_HAR": ql[m].mean() / ql[bench].mean(),
               "MSE_vs_HAR": ms[m].mean() / ms[bench].mean()}
        if m != bench:
            row["DM_QLIKE_t"], row["DM_QLIKE_p"] = diebold_mariano(ql[bench], ql[m])
            row["DM_MSE_t"], row["DM_MSE_p"] = diebold_mariano(ms[bench], ms[m])
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")
