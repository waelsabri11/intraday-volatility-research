"""Walk-forward forecasting experiments (daily and hourly)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from . import features as F
from . import models as M
from .config import MIN_TRAIN_DAYS, PROC, REFIT_EVERY_DAYS, REPORTS
from .data import load_1m
from .evaluate import loss_table, walk_forward

SEQ_LEN = 30
RES = REPORTS / "results"


def _sequences(X: np.ndarray, L: int) -> np.ndarray:
    """seq[i] = X[i-L+1 .. i] (trailing window, includes row i only up to row i)."""
    pad = np.full((L - 1, X.shape[1]), np.nan)
    return sliding_window_view(np.vstack([pad, X]), L, axis=0).transpose(0, 2, 1)


def run_daily(symbol: str) -> dict:
    d = F.daily_features(load_1m(symbol))
    n = len(d)
    y = d["target_lrv"].values
    seq = _sequences(d[F.DAILY_BASE].values, SEQ_LEN)
    cols = sorted(set(F.DAILY_HAR + F.DAILY_BASE + F.DAILY_GBM))
    valid = d[cols].notna().all(axis=1).values & np.isfinite(y) & ~np.isnan(seq).any(axis=(1, 2))
    ret_pct = d["ret"].fillna(0.0).values * 100

    Xh, Xg = d[F.DAILY_HAR].values, d[F.DAILY_GBM].values
    dow = pd.get_dummies(d["dow_next"], dtype=float).iloc[:, 1:].values
    Xd = np.c_[Xh, dow]                                   # HAR + weekday-of-target dummies
    gbm_nodow = [c for c in F.DAILY_GBM if c != "dow_next"]
    Xx = np.c_[d[gbm_nodow].values, dow]                  # linear model on all GBM features
    out, folds = {}, None

    def sk(fn, X):
        return lambda tr, te: fn(X[tr], y[tr], X[te])

    specs = {
        "Naive": sk(M.naive_fit_predict, Xh),
        "HAR": sk(M.har_fit_predict, Xh),
        "HAR-DoW": sk(M.har_fit_predict, Xd),
        "HAR-X (linear, all feats)": sk(M.har_fit_predict, Xx),
        "GARCH": lambda tr, te: M.garch_forecast(ret_pct, tr.max() + 1, False)[te] / 1e4,
        "GJR-GARCH": lambda tr, te: M.garch_forecast(ret_pct, tr.max() + 1, True)[te] / 1e4,
        "GBM": sk(M.gbm_fit_predict, Xg),
        "LSTM": sk(M.lstm_fit_predict, seq),
    }
    for name, fn in specs.items():
        print(f"  [{symbol}] {name}", flush=True)
        out[name], folds = walk_forward(n, valid, MIN_TRAIN_DAYS, REFIT_EVERY_DAYS, fn)
    fc = pd.DataFrame(out, index=d.index)
    fc["rv_target"] = d["target_rv"]
    fc["ret_target"] = d["target_ret"]
    fc["target_date"] = d.index + pd.Timedelta(days=1)
    fc = fc.dropna(subset=list(specs))
    RES.mkdir(parents=True, exist_ok=True)
    fc.to_csv(RES / f"daily_forecasts_{symbol}.csv")
    pd.DataFrame(folds, columns=["train_last_row", "first_origin_row"]).to_csv(
        RES / f"daily_folds_{symbol}.csv", index=False)
    tab = loss_table(fc["rv_target"], {m: fc[m] for m in specs})
    tab.to_csv(RES / f"daily_losses_{symbol}.csv")
    by_year = {}
    for yr, g in fc.groupby(fc.index.year):
        t = loss_table(g["rv_target"], {m: g[m] for m in specs})
        by_year[yr] = t["QLIKE_vs_HAR"]
    pd.DataFrame(by_year).T.to_csv(RES / f"daily_qlike_ratio_by_year_{symbol}.csv")
    d.to_parquet(PROC / f"{symbol}_daily_features.parquet")
    return {"table": tab, "fc": fc}


def run_hourly(symbol: str) -> dict:
    h = F.hourly_features(load_1m(symbol))
    n = len(h)
    y = h["target_lrv"].values
    dummies = pd.get_dummies(h["hour_next"], prefix="h", dtype=float).drop(columns="h_0")
    Xh = h[F.HOURLY_HAR].values
    Xs = np.c_[Xh, dummies.values]
    Xg = h[F.HOURLY_GBM].values
    valid = h[sorted(set(F.HOURLY_HAR + F.HOURLY_GBM))].notna().all(axis=1).values & np.isfinite(y)

    def sk(fn, X):
        return lambda tr, te: fn(X[tr], y[tr], X[te])

    specs = {"Naive": sk(M.naive_fit_predict, Xh), "HAR": sk(M.har_fit_predict, Xh),
             "HAR-S (hour dummies)": sk(M.har_fit_predict, Xs),
             "GBM": sk(M.gbm_fit_predict, Xg)}
    out = {}
    for name, fn in specs.items():
        print(f"  [{symbol} hourly] {name}", flush=True)
        out[name], _ = walk_forward(n, valid, MIN_TRAIN_DAYS * 24, REFIT_EVERY_DAYS * 24, fn)
    fc = pd.DataFrame(out, index=h.index)
    fc["rv_target"] = h["target_rv"]
    fc = fc.dropna(subset=list(specs)).dropna(subset=["rv_target"])
    fc = fc[fc["rv_target"] > 0]
    tab = loss_table(fc["rv_target"], {m: fc[m] for m in specs})
    tab.to_csv(RES / f"hourly_losses_{symbol}.csv")
    return {"table": tab, "fc": fc}
