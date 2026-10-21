"""Volatility-targeting experiment and figures."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import features as F
from .config import PROC, REPORTS, SYMBOLS
from .data import load_1m
from .evaluate import qlike
from .forecast import RES
from .sizing import DAYS, backtest, perf, target_weights

TARGET_VOL = 0.40
COSTS_BPS = [0, 5, 10, 20, 50]
BASE_COST = 10
SIZING_MODELS = ["HAR", "GJR-GARCH", "GBM", "LSTM"]
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 9})


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

    ones = np.ones(len(fc))
    strategies = {"Buy & hold (w=1)": ones}
    for m in SIZING_MODELS:
        strategies[f"VolTarget {m}"] = target_weights(fc[m].values, TARGET_VOL)
    strategies["VolTarget oracle (true RV)"] = target_weights(fc["rv_target"].values, TARGET_VOL)
    har_w = strategies["VolTarget HAR"]
    strategies["VolTarget HAR + 10% band"] = target_weights(fc["HAR"].values, TARGET_VOL, band=0.10)
    strategies["Static weight (= mean HAR weight)"] = np.full(len(fc), har_w.mean())
    for name, w in strategies.items():
        for c in COSTS_BPS:
            add(name, w, c)
    out = pd.DataFrame(rows)
    out.to_csv(RES / f"sizing_{symbol}.csv", index=False)
    figure_sizing(symbol, fc, series)
    return out


def figure_sizing(symbol, fc, series):
    idx = fc["target_date"]
    fig, ax = plt.subplots(3, 1, figsize=(9, 8.5), sharex=True)
    for name, c in [("Buy & hold (w=1)", "0.45"), ("VolTarget HAR", "C0"), ("VolTarget HAR + 10% band", "C1")]:
        net = pd.Series(series[name][0], index=idx)
        ax[0].plot(net.rolling(30).std() * np.sqrt(DAYS), label=name, color=c, lw=1.1)
    ax[0].axhline(TARGET_VOL, color="k", ls="--", lw=0.8, label=f"target {TARGET_VOL:.0%}")
    ax[0].set_title(f"{symbol}: 30-day realised annualised vol (net of 10 bps costs)")
    ax[0].legend(ncol=4, fontsize=7)
    ax[1].plot(idx, series["VolTarget HAR"][1], lw=0.7, label="HAR weight")
    ax[1].plot(idx, series["VolTarget HAR + 10% band"][1], lw=0.9, label="HAR weight, 10% band")
    ax[1].set_title("Position weight (long-only, cap 1.0)")
    ax[1].legend(fontsize=7)
    for name, c in [("Buy & hold (w=1)", "0.45"), ("VolTarget HAR", "C0"), ("VolTarget HAR + 10% band", "C1")]:
        eq = (1 + pd.Series(series[name][0], index=idx)).cumprod()
        ax[2].plot(eq / eq.cummax() - 1, label=name, color=c, lw=1)
    ax[2].set_title("Drawdown")
    fig.tight_layout()
    fig.savefig(REPORTS / f"sizing_{symbol}.png")
    plt.close(fig)


def figure_cost_sensitivity(tables: dict[str, pd.DataFrame]):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, (sym, t) in zip(axes, tables.items()):
        for name in ["Buy & hold (w=1)", "VolTarget HAR", "VolTarget HAR + 10% band"]:
            s = t[t["strategy"] == name].sort_values("cost_bps")
            ax.plot(s["cost_bps"], s["ann_return"], marker="o", label=name)
        ax.set_title(f"{sym}: annualised return vs one-way cost")
        ax.set_xlabel("one-way cost (bps)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(REPORTS / "cost_sensitivity.png")
    plt.close(fig)


def figure_forecasts():
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5))
    for j, sym in enumerate(SYMBOLS):
        fc = pd.read_csv(RES / f"daily_forecasts_{sym}.csv", index_col=0, parse_dates=True)
        ann = lambda v: np.sqrt(v * DAYS)
        ax = axes[0, j]
        ax.plot(fc["target_date"], ann(fc["rv_target"]), color="0.7", lw=0.7, label="realised (5-min RV)")
        ax.plot(fc["target_date"], ann(fc["HAR"]), lw=0.9, label="HAR forecast")
        ax.plot(fc["target_date"], ann(fc["GBM"]), lw=0.9, label="GBM forecast")
        ax.set_title(f"{sym}: next-day volatility, annualised (OOS)")
        ax.legend(fontsize=7)
        ax = axes[1, j]
        for m in ["GARCH", "GJR-GARCH", "GBM", "LSTM"]:
            d = qlike(fc["rv_target"].values, fc["HAR"].values) - qlike(fc["rv_target"].values, fc[m].values)
            ax.plot(fc["target_date"], np.cumsum(d), label=m, lw=1)
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{sym}: cumulative QLIKE gain vs HAR (up = better than HAR)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(REPORTS / "forecast_comparison.png")
    plt.close(fig)


def figure_loss_ratios():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, sym in zip(axes, SYMBOLS):
        t = pd.read_csv(RES / f"daily_losses_{sym}.csv", index_col=0)
        t = t.drop(index="HAR")
        bars = ax.bar(t.index, t["QLIKE_vs_HAR"], color="C0")
        ax.axhline(1, color="k", lw=0.8)
        for b, p in zip(bars, t["DM_QLIKE_p"]):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"p={p:.2f}", ha="center",
                    va="bottom", fontsize=7)
        ax.set_title(f"{sym}: QLIKE / QLIKE(HAR)  (<1 = better)")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(REPORTS / "qlike_ratios.png")
    plt.close(fig)


def figure_seasonality():
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for sym in SYMBOLS:
        h = F.hourly_features(load_1m(sym))
        h = h.dropna(subset=["rv"])
        g = h.groupby(h.index.hour)
        axes[0].plot(np.sqrt(g["rv"].mean() * 24 * 365) , label=sym, marker=".")
        axes[1].plot(g["qv"].mean() / g["qv"].mean().mean(), label=sym, marker=".")
        axes[2].plot(g["cs"].mean() * 1e4, label=sym, marker=".")
    axes[0].set_title("Intraday vol profile (annualised, by UTC hour)")
    axes[1].set_title("Relative traded value by UTC hour")
    axes[2].set_title("Corwin-Schultz spread proxy (bps)")
    for a in axes:
        a.set_xlabel("UTC hour")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(REPORTS / "intraday_seasonality.png")
    plt.close(fig)
