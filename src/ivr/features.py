"""Feature construction.

Timing convention (the single most important rule of this repo):
    A row indexed by period p (UTC day, or UTC hour) contains ONLY information that is
    known at the END of period p. The target of that row is log realised variance of period
    p+1. Every rolling window is trailing (includes p, excludes nothing from the future).
tests/test_no_lookahead.py verifies this by truncating the raw data and recomputing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BARS_5M = "5min"
MAX_MISSING_FRAC = 0.10  # drop days with more than 10% of 1-min bars missing (exchange downtime)


def corwin_schultz(high: pd.Series, low: pd.Series) -> pd.Series:
    """Corwin-Schultz (2012) high-low spread estimator from two consecutive bars.

    Uses bar p-1 and bar p only, so it is known at the end of bar p. Negative values -> 0.
    """
    hl = np.log(high / low) ** 2
    beta = hl + hl.shift(1)
    h2 = np.maximum(high, high.shift(1))
    l2 = np.minimum(low, low.shift(1))
    gamma = np.log(h2 / l2) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return s.clip(lower=0)


def _bipower(abs_r: pd.Series, key: pd.Series) -> pd.Series:
    """Bipower variation per period: (pi/2) * n/(n-1) * sum |r_i||r_{i-1}| within the period."""
    prod = abs_r * abs_r.shift(1)
    same = key == key.shift(1)
    prod = prod.where(same)
    g = prod.groupby(key)
    n = abs_r.groupby(key).size()
    return (np.pi / 2) * g.sum() * n / (n - 1)


def five_min_returns(df1m: pd.DataFrame) -> pd.DataFrame:
    """5-minute log returns (+ volume pieces). Empty bars carry the last close (zero return)."""
    close5 = df1m["close"].resample(BARS_5M).last().ffill()
    out = pd.DataFrame({"r": np.log(close5).diff()})
    out["quote_volume"] = df1m["quote_volume"].resample(BARS_5M).sum()
    return out.dropna(subset=["r"])


def _period_core(r5: pd.DataFrame, df1m: pd.DataFrame, freq: str, ppd: int) -> pd.DataFrame:
    """Realised measures + microstructure features aggregated to `freq` ('D' or 'h')."""
    key = r5.index.floor(freq)
    key = pd.Series(key, index=r5.index)
    r = r5["r"]
    g = r.groupby(key)
    out = pd.DataFrame({
        "rv": (r ** 2).groupby(key).sum(),
        "bv": _bipower(r.abs(), key),
        "rs_neg": ((r.clip(upper=0)) ** 2).groupby(key).sum(),
        "ret": g.sum(),
        "n5": g.size(),
    })
    out["jump"] = (out["rv"] - out["bv"]).clip(lower=0)
    out["jump_share"] = out["jump"] / out["rv"]
    out["rs_neg_share"] = out["rs_neg"] / out["rv"]

    k1 = df1m.index.floor(freq)
    agg = df1m.groupby(k1).agg(
        high=("high", "max"), low=("low", "min"), qv=("quote_volume", "sum"),
        trades=("trades", "sum"), vol=("volume", "sum"), tb=("taker_buy_base", "sum"),
        n1=("close", "size"),
    )
    out = out.join(agg, how="inner")
    out["log_qv"] = np.log(out["qv"])
    out["log_trades"] = np.log(out["trades"].clip(lower=1))
    out["imb"] = (2 * out["tb"] - out["vol"]) / out["vol"]        # taker-buy imbalance in [-1, 1]
    out["abs_imb"] = out["imb"].abs()
    out["cs"] = corwin_schultz(out["high"], out["low"])
    out["log_amihud"] = np.log((out["ret"].abs() + 1e-8) / out["qv"])
    out["missing_frac"] = 1 - out["n1"] / (60 * (24 if freq == "D" else 1))
    return out


def daily_features(df1m: pd.DataFrame) -> pd.DataFrame:
    r5 = five_min_returns(df1m)
    d = _period_core(r5, df1m, "D", 288)

    # Roll (1984) spread from 1-min return autocovariance within the day.
    r1 = np.log(df1m["close"]).diff()
    day = pd.Series(df1m.index.floor("D"), index=df1m.index)
    r1l = r1.where(day == day.shift(1))
    cov = (r1l * r1l.shift(1)).groupby(day).mean()
    d["roll"] = 2 * np.sqrt((-cov).clip(lower=0))

    # share of the day's variance realised in the last 4 hours (recency of the vol)
    last4 = r5["r"].where(r5.index.hour >= 20) ** 2
    d["rv_last4h_share"] = last4.groupby(r5.index.floor("D")).sum() / d["rv"]

    d = d[d["missing_frac"] <= MAX_MISSING_FRAC].copy()
    d.index.name = "date"
    d = d.asfreq("D")  # rows for dropped days become NaN; they are excluded downstream

    lrv = np.log(d["rv"])
    d["lrv"] = lrv
    d["lrv_w"] = np.log(d["rv"].rolling(7, min_periods=7).mean())
    d["lrv_m"] = np.log(d["rv"].rolling(30, min_periods=30).mean())
    d["lbv"] = np.log(d["bv"])
    d["ljump_share"] = d["jump_share"]
    d["lrs_neg"] = np.log(d["rs_neg"].clip(lower=1e-12))
    d["imb_w"] = d["imb"].rolling(7, min_periods=7).mean()
    d["dow_next"] = (d.index.dayofweek + 1) % 7  # weekday of the TARGET day

    d["target_lrv"] = d["lrv"].shift(-1)
    d["target_rv"] = d["rv"].shift(-1)
    d["target_ret"] = d["ret"].shift(-1)
    return d


DAILY_BASE = ["lrv", "lbv", "ljump_share", "lrs_neg", "rs_neg_share", "log_qv", "log_trades",
              "imb", "abs_imb", "cs", "roll", "log_amihud", "rv_last4h_share", "ret"]
DAILY_HAR = ["lrv", "lrv_w", "lrv_m"]
DAILY_GBM = DAILY_BASE + ["lrv_w", "lrv_m", "imb_w", "dow_next"]


def hourly_features(df1m: pd.DataFrame) -> pd.DataFrame:
    r5 = five_min_returns(df1m)
    h = _period_core(r5, df1m, "h", 12)
    h = h[h["missing_frac"] <= 0.5].copy()
    h.index.name = "ts"
    h = h.asfreq("h")
    h["lrv"] = np.log(h["rv"].clip(lower=1e-12))
    h["lrv_d"] = np.log(h["rv"].rolling(24, min_periods=24).mean())
    h["lrv_w"] = np.log(h["rv"].rolling(168, min_periods=168).mean())
    h["lbv"] = np.log(h["bv"].clip(lower=1e-12))
    h["hour_next"] = (h.index.hour + 1) % 24
    h["target_lrv"] = h["lrv"].shift(-1)
    h["target_rv"] = h["rv"].shift(-1)
    return h


HOURLY_HAR = ["lrv", "lrv_d", "lrv_w"]
HOURLY_GBM = ["lrv", "lrv_d", "lrv_w", "lbv", "jump_share", "log_qv", "log_trades", "imb",
              "abs_imb", "cs", "ret", "hour_next"]
