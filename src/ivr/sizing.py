"""Volatility-targeting position sizing with explicit transaction costs.

A risk-management illustration: forecast quality is judged by how well realised volatility is
controlled, NOT by profit. The underlying (spot BTC/ETH, long-only, no leverage) has strongly
positive sample drift in 2022-2026 that has nothing to do with the sizing rule.

Timing: weight w_t is chosen at the close of day t from the forecast made at the close of day t
(uses data <= t), and earns the close-to-close return of day t+1. The cost of moving from the
drifted pre-trade weight to w_t is charged to day t+1's return.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAYS = 365


def target_weights(var_fc: np.ndarray, target_vol=0.40, cap=1.0, band=0.0, floor=0.0):
    """w = clip(target / forecast annual vol, floor, cap). With band > 0, trade only when the
    desired weight differs from the current weight by more than `band` (no-trade zone)."""
    desired = np.clip(target_vol / np.sqrt(var_fc * DAYS), floor, cap)
    if band <= 0:
        return desired
    w = np.empty_like(desired)
    w[0] = desired[0]
    for i in range(1, len(w)):
        w[i] = desired[i] if abs(desired[i] - w[i - 1]) > band else w[i - 1]
    return w


def backtest(weights: np.ndarray, simple_ret_next: np.ndarray, cost_bps: float):
    """weights[t] earns simple_ret_next[t] (= return of day t+1). Returns daily net returns,
    turnover series (fraction of NAV traded at each rebalance)."""
    n = len(weights)
    w_prev_end = np.empty(n)  # weight just before the rebalance at close of t
    turn = np.zeros(n)
    w_prev_end[0] = 0.0
    turn[0] = weights[0]  # initial investment
    for t in range(1, n):
        r = simple_ret_next[t - 1]  # return of day t earned by weights[t-1]
        w_prev_end[t] = weights[t - 1] * (1 + r) / (1 + weights[t - 1] * r)
        turn[t] = abs(weights[t] - w_prev_end[t])
    gross = weights * simple_ret_next
    net = gross - turn * cost_bps / 1e4
    return net, turn


def perf(net: np.ndarray, turn: np.ndarray, weights: np.ndarray, target_vol: float):
    r = pd.Series(net)
    eq = (1 + r).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    ann_vol = r.std() * np.sqrt(DAYS)
    roll = r.rolling(30).std() * np.sqrt(DAYS)
    return {
        "ann_return": float(eq.iloc[-1] ** (DAYS / len(r)) - 1),
        "ann_vol": float(ann_vol),
        "sharpe": float(r.mean() / r.std() * np.sqrt(DAYS)),
        "max_drawdown": float(dd),
        "worst_day": float(r.min()),
        "vol_30d_std": float(roll.std()),
        "vol_30d_rmse_vs_target": float(np.sqrt(np.nanmean((roll - target_vol) ** 2))),
        "avg_weight": float(np.mean(weights)),
        "ann_turnover": float(turn.sum() / len(r) * DAYS),
    }
