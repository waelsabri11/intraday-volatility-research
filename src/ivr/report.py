"""Volatility-targeting experiment and figures."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SYMBOLS
from .forecast import RES
from .sizing import DAYS, backtest, perf, target_weights

TARGET_VOL = 0.40
COSTS_BPS = [0, 5, 10, 20, 50]
BASE_COST = 10
SIZING_MODELS = ["HAR", "GJR-GARCH", "GBM", "LSTM"]


def build_strategies(fc: pd.DataFrame) -> dict[str, np.ndarray]:
    strategies = {"Buy & hold (w=1)": np.ones(len(fc))}
    for m in SIZING_MODELS:
        strategies[f"VolTarget {m}"] = target_weights(fc[m].values, TARGET_VOL)
    strategies["VolTarget oracle (true RV)"] = target_weights(fc["rv_target"].values, TARGET_VOL)
    strategies["VolTarget HAR + 10% band"] = target_weights(fc["HAR"].values, TARGET_VOL, band=0.10)
    strategies["Static weight (= mean HAR weight)"] = np.full(len(fc), strategies["VolTarget HAR"].mean())
    return strategies


def run_sizing(symbol: str) -> pd.DataFrame:
    fc = pd.read_csv(RES / f"daily_forecasts_{symbol}.csv", index_col=0, parse_dates=True)
    r_next = np.expm1(fc["ret_target"].values)       # simple return of the target day
    rows, series = [], {}

    def add(name, w, cost, key=None):
        net, turn = backtest(w, r_next, cost)
        p = perf(net, turn, w, TARGET_VOL)
        p.update({"strategy": name, "cost_bps": cost})
        rows.append(p)
        if cost == BASE_COST:
            series[name] = (net, w)

    strategies = build_strategies(fc)
    for name, w in strategies.items():
        for c in COSTS_BPS:
            add(name, w, c)
    out = pd.DataFrame(rows)
    out.to_csv(RES / f"sizing_{symbol}.csv", index=False)
    return out
