"""All README figures (reports/figures/*.png). Style: Okabe-Ito colour-blind-safe palette."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from . import features as F
from . import models as M
from .config import PROC, REPORTS, SYMBOLS
from .data import load_1m
from .evaluate import qlike
from .forecast import RES
from .report import BASE_COST, TARGET_VOL, build_strategies
from .sizing import DAYS, backtest

FIG = REPORTS / "figures"
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "verm": "#D55E00",
      "sky": "#56B4E9", "purple": "#CC79A7", "yellow": "#F0E442", "grey": "#7F7F7F", "black": "#000000"}
MCOL = {"Naive": OI["grey"], "HAR": OI["blue"], "HAR-DoW": OI["sky"], "HAR-X (linear, all feats)": OI["green"],
        "GARCH": "#999933", "GJR-GARCH": OI["verm"], "GBM": OI["orange"], "LSTM": OI["purple"],
        "HAR-S (hour dummies)": OI["sky"]}
SCOL = {"BTCUSDT": OI["blue"], "ETHUSDT": OI["orange"]}
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 170, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "axes.prop_cycle": plt.cycler(color=list(OI.values())[:7])})


def _save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


def _daily(sym):
    return pd.read_parquet(PROC / f"{sym}_daily_features.parquet")


def _fc(sym, kind="daily"):
    return pd.read_csv(RES / f"{kind}_forecasts_{sym}.csv", index_col=0, parse_dates=True)


def _close_daily(sym):
    c = pd.read_parquet(PROC / f"{sym}_1m.parquet", columns=["ts", "close"]).set_index("ts")["close"]
    return c.resample("D").last()


def _ann(v, per=DAYS):
    return np.sqrt(np.asarray(v) * per)


def _acf(x, nlags):
    x = pd.Series(x).dropna().values
    x = x - x.mean()
    d = x @ x
    return np.array([x[k:] @ x[:-k] / d for k in range(1, nlags + 1)])


# ---------------------------------------------------------------- data & stylised facts
def fig01_price_vol():
    fig, ax = plt.subplots(2, 2, figsize=(11, 5.6), sharex=True)
    ev = [("2020-03-12", "COVID\ncrash"), ("2021-05-19", "May-21\nsell-off"),
          ("2022-05-12", "Terra/Luna"), ("2022-11-09", "FTX")]
    for j, s in enumerate(SYMBOLS):
        d = _daily(s)
        ax[0, j].semilogy(_close_daily(s), color=SCOL[s], lw=0.9)
        ax[0, j].set_title(f"{s[:3]}/USDT close (log scale, USDT)")
        v = pd.Series(_ann(d["rv"]), index=d.index)
        ax[1, j].plot(v, color="0.75", lw=0.5, label="daily")
        ax[1, j].plot(v.rolling(30).mean(), color=SCOL[s], lw=1.2, label="30-day mean")
        ax[1, j].set_title("Daily realised vol from 5-min returns (annualised)")
        ax[1, j].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax[1, j].set_ylim(0, 2.5)
        ax[1, j].legend(fontsize=7, loc="upper right")
        for k, (dt, lab) in enumerate(ev):
            ax[1, j].axvline(pd.Timestamp(dt), color="0.35", lw=0.6, ls=":")
            ax[1, j].text(pd.Timestamp(dt), 2.4 - 0.22 * (k % 2), " " + lab.replace("\n", " "), fontsize=6.5, va="top")
    _save(fig, "01_price_and_realised_vol")


def fig02_seasonality():
    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(2, 2)
    a0, a1 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    hs = {s: F.hourly_features(load_1m(s)).dropna(subset=["rv"]) for s in SYMBOLS}
    for s, h in hs.items():
        g = h.groupby(h.index.hour)
        a0.plot(_ann(g["rv"].mean(), 24 * 365), marker="o", ms=3, color=SCOL[s], label=s[:3])
        a1.plot(g["qv"].mean() / g["qv"].mean().mean(), marker="o", ms=3, color=SCOL[s], label=s[:3])
    a0.set_title("Mean realised vol by UTC hour (annualised, 5-min based)")
    a1.set_title("Traded value by UTC hour (relative to daily mean)")
    for a in (a0, a1):
        a.set_xlabel("UTC hour"); a.legend(fontsize=7)
    for k, s in enumerate(SYMBOLS):
        a = fig.add_subplot(gs[1, k]); h = hs[s]
        pv = _ann(h.groupby([h.index.dayofweek, h.index.hour])["rv"].mean().unstack(), 24 * 365)
        im = a.imshow(pv, aspect="auto", cmap="viridis", origin="upper")
        a.set_yticks(range(7)); a.set_yticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        a.set_xlabel("UTC hour"); a.set_title(f"{s[:3]}: mean annualised vol, weekday x hour"); a.grid(False)
        plt.colorbar(im, ax=a, format=matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    _save(fig, "02_intraday_seasonality")


def fig03_acf():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for a, s in zip(ax, SYMBOLS):
        d = _daily(s)
        n = d["ret"].notna().sum()
        lags = np.arange(1, 61)
        a.plot(lags, _acf(d["ret"], 60), color=OI["grey"], label="return")
        a.plot(lags, _acf(d["ret"].abs(), 60), color=OI["orange"], label="|return|")
        a.plot(lags, _acf(d["lrv"], 60), color=OI["blue"], label="log RV")
        a.axhspan(-1.96 / np.sqrt(n), 1.96 / np.sqrt(n), color="0.85", label="95% band (white noise)")
        a.set_title(f"{s[:3]}: autocorrelation of daily returns vs volatility")
        a.set_xlabel("lag (days)")
    ax[0].legend(fontsize=7); ax[0].set_ylabel("ACF")
    _save(fig, "03_volatility_clustering_acf")


def fig04_distributions():
    from scipy import stats
    fig, ax = plt.subplots(2, 2, figsize=(10, 6))
    for j, s in enumerate(SYMBOLS):
        d = _daily(s)
        for i, (x, ttl) in enumerate([(d["ret"].dropna(), "daily log return"),
                                      (d["lrv"].dropna(), "log realised variance")]):
            a = ax[i, j]
            a.hist(x, bins=80, density=True, color=SCOL[s], alpha=0.75)
            g = np.linspace(x.min(), x.max(), 400)
            a.plot(g, stats.norm.pdf(g, x.mean(), x.std()), color="k", lw=1, label="normal, same mean/sd")
            if i == 0:
                a.set_yscale("log"); a.set_ylim(1e-2, None)
            a.set_title(f"{s[:3]}: {ttl}")
            a.text(0.02, 0.95, f"excess kurtosis {stats.kurtosis(x):.1f}\nskew {stats.skew(x):.2f}",
                   transform=a.transAxes, va="top", fontsize=8)
            a.legend(fontsize=7, loc="upper right")
    _save(fig, "04_return_and_logrv_distributions")


def fig05_jumps():
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for s in SYMBOLS:
        d = _daily(s).dropna(subset=["rv", "bv"])
        ax[0].loglog(d["bv"], d["rv"], ".", ms=2, alpha=0.4, color=SCOL[s], label=s[:3])
        js = d["jump"].rolling(90).sum() / d["rv"].rolling(90).sum()
        ax[1].plot(js, color=SCOL[s], label=s[:3])
    lim = [2e-6, 0.12]
    ax[0].plot(lim, lim, "k--", lw=0.8)
    ax[0].set_xlabel("bipower variation (continuous part)"); ax[0].set_ylabel("realised variance")
    ax[0].set_title("RV vs BV: points above the line = jump variance")
    ax[0].legend(fontsize=7, markerscale=4)
    ax[1].set_title("Share of variance from jumps (RV-BV)+ , 90-day window")
    ax[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax[1].legend(fontsize=7)
    _save(fig, "05_jumps_rv_vs_bv")


def fig06_microstructure():
    fig = plt.figure(figsize=(11, 6.5))
    gs = fig.add_gridspec(2, 3)
    spec = [("cs", "Corwin-Schultz spread proxy (bps, 30-day mean)", 1e4),
            ("roll", "Roll spread proxy from 1-min returns (bps, 30-day mean)", 1e4),
            ("imb", "Taker-buy imbalance (buy-sell)/total, 30-day mean", 1.0)]
    for k, (c, ttl, mult) in enumerate(spec):
        a = fig.add_subplot(gs[0, k])
        for s in SYMBOLS:
            a.plot(_daily(s)[c].rolling(30).mean() * mult, color=SCOL[s], lw=1, label=s[:3])
        a.set_title(ttl, fontsize=8)
        if k == 0:
            a.legend(fontsize=7)
    a = fig.add_subplot(gs[1, :])
    feats = ["cs", "roll", "imb", "abs_imb", "log_qv", "log_trades", "log_amihud", "jump_share"]
    w = 0.2
    for i, (s, kind) in enumerate([(s, k) for k in ("target", "resid") for s in SYMBOLS]):
        d, fc = _daily(s), _fc(s)
        x = d.loc[fc.index]
        y = np.log(x["target_rv"]) if kind == "target" else np.log(fc["rv_target"] / fc["HAR"])
        vals = [np.corrcoef(x[f].values, y.values)[0, 1] if x[f].notna().all() else
                x[f].corr(y) for f in feats]
        a.bar(np.arange(len(feats)) + (i - 1.5) * w, vals, w, color=SCOL[s], alpha=1 if kind == "target" else 0.45,
              hatch="" if kind == "target" else "//", label=f"{s[:3]}: corr with " + ("next-day log RV" if kind == "target" else "HAR forecast error"))
    a.set_xticks(range(len(feats))); a.set_xticklabels(feats)
    a.axhline(0, color="k", lw=0.6)
    a.set_title("Microstructure proxies: correlation with next-day volatility and with HAR's log forecast error (OOS days)")
    a.legend(fontsize=7, ncol=2)
    _save(fig, "06_microstructure_proxies")


# ---------------------------------------------------------------- forecasting
SHOW = ["HAR", "HAR-X (linear, all feats)", "GJR-GARCH", "GBM", "LSTM"]


def fig07_forecast_vs_realised():
    fig, ax = plt.subplots(2, 1, figsize=(10.5, 6.5), sharex=True)
    for a, s in zip(ax, SYMBOLS):
        fc = _fc(s).loc["2022-04-20":"2022-12-31"]
        a.plot(fc["target_date"], _ann(fc["rv_target"]), color="k", lw=2.2, alpha=0.35, label="realised (5-min RV)")
        for m in SHOW:
            a.plot(fc["target_date"], _ann(fc[m]), color=MCOL[m], lw=1.0, label=m)
        a.set_title(f"{s[:3]}: next-day volatility forecasts vs realised, Terra/Luna (May-22) and FTX (Nov-22) stress, OOS")
        a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax[0].legend(fontsize=7, ncol=3)
    _save(fig, "07_forecast_vs_realised_stress")


def _stars(p):
    return "**" if p < 0.01 else "*" if p < 0.05 else ""


def _loss_bars(prefix, name, models_drop=("HAR",), title_suffix=""):
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.8))
    for j, s in enumerate(SYMBOLS):
        t = pd.read_csv(RES / f"{prefix}_losses_{s}.csv", index_col=0).drop(index=list(models_drop))
        for k, (col, pcol, lab) in enumerate([("QLIKE_vs_HAR", "DM_QLIKE_p", "QLIKE"), ("MSE_vs_HAR", "DM_MSE_p", "MSE")]):
            a = ax[2 * j + k]
            b = a.bar(range(len(t)), t[col], color=[MCOL.get(m, "0.5") for m in t.index])
            a.axhline(1, color="k", lw=1, ls="--")
            a.set_xticks(range(len(t))); a.set_xticklabels([m.replace(" (linear, all feats)", "\n(linear)").replace(" (hour dummies)", "") for m in t.index], rotation=40, ha="right", fontsize=7)
            for r, p in zip(b, t[pcol]):
                a.text(r.get_x() + r.get_width() / 2, r.get_height(), _stars(p), ha="center", va="bottom", fontsize=11)
            a.set_title(f"{s[:3]}: {lab} / {lab}(HAR)", fontsize=9)
            a.set_ylim(0, max(1.3, t[col].max() * 1.12))
    fig.suptitle("Loss relative to HAR (dashed = HAR; <1 better). DM test vs HAR: * p<0.05, ** p<0.01" + title_suffix, fontsize=9, y=1.02)
    _save(fig, name)


def fig08_loss_bars():
    _loss_bars("daily", "08_daily_loss_vs_har", title_suffix=" - next-day horizon")


def fig09_rolling_ratio():
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for a, s in zip(ax, SYMBOLS):
        fc = _fc(s)
        ql = {m: pd.Series(qlike(fc["rv_target"].values, fc[m].values), index=fc.index) for m in fc.columns if m in MCOL}
        for m in ["HAR-DoW", "HAR-X (linear, all feats)", "GJR-GARCH", "GBM", "LSTM"]:
            a.plot((ql[m].rolling(180).mean() / ql["HAR"].rolling(180).mean()), color=MCOL[m], lw=1.2, label=m)
        a.axhline(1, color="k", ls="--", lw=0.8)
        a.set_title(f"{s[:3]}: rolling 180-day QLIKE / QLIKE(HAR)  (<1 beats HAR)")
        a.set_ylim(0.4, 2.0)
    ax[0].legend(fontsize=7, ncol=2)
    _save(fig, "09_rolling_qlike_ratio")


def _train_test(sym, split="2024-01-01"):
    d = _daily(sym)
    cols = sorted(set(F.DAILY_GBM))
    ok = d[cols + ["target_lrv"]].notna().all(axis=1)
    d = d[ok]
    return d[d.index < split], d[d.index >= split]


def fig10_importance_and_coefs():
    s = "BTCUSDT"
    tr, te = _train_test(s)
    cols = F.DAILY_GBM
    m = HistGradientBoostingRegressor(**{**M.GBM_PARAMS, "max_iter": 150}).fit(tr[cols], tr["target_lrv"])
    base = np.mean((te["target_lrv"] - m.predict(te[cols])) ** 2)
    rng = np.random.default_rng(0)
    imp = {}
    for c in cols:
        inc = []
        for _ in range(10):
            x = te[cols].copy(); x[c] = rng.permutation(x[c].values)
            inc.append(np.mean((te["target_lrv"] - m.predict(x)) ** 2) - base)
        imp[c] = (np.mean(inc), np.std(inc))
    imp = pd.DataFrame(imp, index=["mean", "sd"]).T.sort_values("mean")
    # HAR-X: standardised OLS with Newey-West SE on the training window
    dow = pd.get_dummies(tr["dow_next"], dtype=float).iloc[:, 1:]
    nodow = [c for c in cols if c != "dow_next"]
    X = tr[nodow]; Xs = (X - X.mean()) / X.std()
    A = np.c_[np.ones(len(Xs)), Xs.values, dow.values]
    y = tr["target_lrv"].values
    b = np.linalg.lstsq(A, y, rcond=None)[0]
    u = y - A @ b
    Z = A * u[:, None]
    L = 10
    S = Z.T @ Z
    for k in range(1, L + 1):
        G = Z[k:].T @ Z[:-k]
        S += (1 - k / (L + 1)) * (G + G.T)
    Ainv = np.linalg.inv(A.T @ A)
    se = np.sqrt(np.diag(Ainv @ S @ Ainv))
    nm = len(nodow)
    co = pd.DataFrame({"b": b[1:1 + nm], "se": se[1:1 + nm]}, index=nodow).sort_values("b")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].barh(imp.index, imp["mean"], xerr=imp["sd"], color=OI["orange"])
    ax[0].set_title("GBM permutation importance (BTC, fit pre-2024, scored 2024+)\nincrease in test MSE of log RV")
    ax[1].errorbar(co["b"], range(nm), xerr=1.96 * co["se"], fmt="o", color=OI["green"], capsize=2)
    ax[1].set_yticks(range(nm)); ax[1].set_yticklabels(co.index)
    ax[1].axvline(0, color="k", lw=0.7)
    ax[1].set_title("HAR-X coefficients (BTC, pre-2024, standardised features)\n95% Newey-West CI, weekday dummies included as controls")
    _save(fig, "10_feature_importance_and_coefficients")


def fig11_calibration():
    fig, ax = plt.subplots(2, 5, figsize=(14, 5.8), sharex=True, sharey=True)
    for i, s in enumerate(SYMBOLS):
        fc = _fc(s)
        for j, m in enumerate(SHOW):
            a = ax[i, j]
            x, y = np.log(fc[m]), np.log(fc["rv_target"])
            a.plot(x, y, ".", ms=2, alpha=0.25, color=MCOL[m])
            q = pd.qcut(x, 10, duplicates="drop")
            g = pd.DataFrame({"x": x, "y": y}).groupby(q, observed=True).agg(x=("x", "mean"), y=("y", "mean"))
            a.plot(g["x"], g["y"], "o-", color="k", ms=3, lw=1, label="decile means")
            lim = [-11, -3.5]
            a.plot(lim, lim, "--", color="0.4", lw=0.8)
            a.set_xlim(lim); a.set_ylim(-12, -2)
            a.set_title(f"{s[:3]} {m.split(' (')[0]}", fontsize=8)
            if i == 1: a.set_xlabel("log forecast variance")
            if j == 0: a.set_ylabel("log realised variance")
    ax[0, 0].legend(fontsize=6)
    fig.suptitle("Calibration (OOS): dashed = perfect calibration; black = mean realised log-variance per forecast decile", fontsize=9)
    _save(fig, "11_calibration")


def fig12_walk_forward():
    d = _daily("BTCUSDT")
    folds = pd.read_csv(RES / "daily_folds_BTCUSDT.csv")
    idx = d.index
    fig, ax = plt.subplots(figsize=(11, 3.8))
    show = folds.iloc[::10].reset_index(drop=True)
    for k, r in show.iterrows():
        t0, tl, s = idx[0], idx[int(r.train_last_row)], idx[int(r.first_origin_row)]
        e = idx[min(int(r.first_origin_row) + 29, len(idx) - 1)]
        ax.barh(k, (tl - t0).days, left=mdates(t0), color=OI["blue"], height=0.6, label="train: rows with realised targets (<= s-1)" if k == 0 else None)
        ax.barh(k, (e - s).days + 1, left=mdates(s), color=OI["orange"], height=0.6, label="test: 30 forecast origins, model frozen" if k == 0 else None)
    ax.set_yticks(range(len(show))); ax.set_yticklabels([f"fit {i * 10 + 1}" for i in range(len(show))], fontsize=7)
    ax.invert_yaxis()
    ax.xaxis_date(); ax.set_xlim(mdates(idx[0]), mdates(idx[-1]))
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title("Expanding-window walk-forward (every 10th of the refits shown). Feature row t = data up to end of day t; target = RV(t+1).\n"
                 "A fit at origin s uses rows <= s-1 only, whose targets are realised by end of day s. Re-fit every 30 days; no hyper-parameter tuning on test data.", fontsize=8)
    _save(fig, "12_walk_forward_scheme")


def mdates(ts):
    return matplotlib.dates.date2num(ts.to_pydatetime())


# ---------------------------------------------------------------- use case
def _strategy_series(sym):
    fc = _fc(sym)
    r = np.expm1(fc["ret_target"].values)
    out = {}
    for name, w in build_strategies(fc).items():
        net, turn = backtest(w, r, BASE_COST)
        out[name] = (pd.Series(net, index=fc["target_date"]), pd.Series(w, index=fc["target_date"]), turn)
    return out


def fig13_vol_targeting():
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    lines = [("Buy & hold (w=1)", OI["grey"]), ("Static weight (= mean HAR weight)", OI["sky"]),
             ("VolTarget HAR", OI["blue"]), ("VolTarget HAR + 10% band", OI["orange"])]
    tabs = {}
    for j, s in enumerate(SYMBOLS):
        ser = _strategy_series(s)
        for n, c in lines:
            ax[0, j].plot(ser[n][0].rolling(30).std() * np.sqrt(DAYS), color=c, lw=1, label=n)
        ax[0, j].axhline(TARGET_VOL, color="k", ls="--", lw=0.8, label="target 40%")
        ax[0, j].set_title(f"{s[:3]}: 30-day realised vol of the position (net, 10 bps one-way)")
        ax[0, j].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        tabs[s] = pd.read_csv(RES / f"sizing_{s}.csv")
    ax[0, 0].legend(fontsize=6.5, ncol=2)
    strat = ["VolTarget HAR", "VolTarget GJR-GARCH", "VolTarget GBM", "VolTarget LSTM", "VolTarget oracle (true RV)", "VolTarget HAR + 10% band"]
    x = np.arange(len(strat)); w = 0.38
    for k, s in enumerate(SYMBOLS):
        t = tabs[s]
        c10 = t[t.cost_bps == 10].set_index("strategy"); c0 = t[t.cost_bps == 0].set_index("strategy")
        ax[1, 0].bar(x + (k - .5) * w, c10.loc[strat, "ann_turnover"], w, color=SCOL[s], label=s[:3])
        ax[1, 1].bar(x + (k - .5) * w, (c0.loc[strat, "ann_return"] - c10.loc[strat, "ann_return"]) * 100, w, color=SCOL[s], label=s[:3])
    lab = [n.replace("VolTarget ", "").replace(" (true RV)", "") for n in strat]
    for a, ttl in zip(ax[1], ["Annual turnover (x NAV traded per year)", "Cost drag at 10 bps one-way (% return per year)"]):
        a.set_xticks(x); a.set_xticklabels(lab, rotation=30, ha="right", fontsize=7); a.set_title(ttl)
    ax[1, 0].legend(fontsize=7)
    fig.suptitle("Volatility targeting as RISK CONTROL - an illustration, not a profitable strategy", fontsize=10, y=1.01)
    _save(fig, "13_vol_targeting_control_and_costs")


def fig14_equity_drawdown():
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.5), sharex="col", gridspec_kw={"height_ratios": [2, 1]})
    lines = [("Buy & hold (w=1)", OI["grey"]), ("Static weight (= mean HAR weight)", OI["sky"]),
             ("VolTarget HAR", OI["blue"]), ("VolTarget HAR + 10% band", OI["orange"])]
    for j, s in enumerate(SYMBOLS):
        ser = _strategy_series(s)
        for n, c in lines:
            eq = (1 + ser[n][0]).cumprod()
            ax[0, j].plot(eq, color=c, lw=1.1, label=n)
            ax[1, j].plot(eq / eq.cummax() - 1, color=c, lw=0.9)
        ax[0, j].set_yscale("log")
        ax[0, j].set_title(f"{s[:3]}: growth of 1 USDT, net of 10 bps one-way costs\nILLUSTRATION of risk control, not a strategy; sample drift dominates returns")
        ax[1, j].set_title("Drawdown"); ax[1, j].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax[0, 0].legend(fontsize=6.5)
    _save(fig, "14_equity_and_drawdown_illustration")


# ---------------------------------------------------------------- hourly
def fig15_hourly_bars():
    _loss_bars("hourly", "15_hourly_loss_vs_har", title_suffix=" - next-hour horizon")


def fig16_hourly_forecast():
    fig, ax = plt.subplots(2, 1, figsize=(10.5, 5.6), sharex=True)
    for a, s in zip(ax, SYMBOLS):
        fc = _fc(s, "hourly").loc["2022-11-07":"2022-11-14"]
        a.plot(fc.index + pd.Timedelta(hours=1), _ann(fc["rv_target"], 24 * 365), color="k", alpha=0.35, lw=2, label="realised hourly vol")
        for m in ["HAR", "HAR-S (hour dummies)", "GBM"]:
            a.plot(fc.index + pd.Timedelta(hours=1), _ann(fc[m], 24 * 365), color=MCOL[m], lw=1, label=m)
        a.set_title(f"{s[:3]}: next-hour volatility forecasts (annualised), FTX week, OOS")
        a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax[0].legend(fontsize=7, ncol=4)
    _save(fig, "16_hourly_forecast_vs_realised")


def make_all():
    for f in [fig01_price_vol, fig02_seasonality, fig03_acf, fig04_distributions, fig05_jumps,
              fig06_microstructure, fig07_forecast_vs_realised, fig08_loss_bars, fig09_rolling_ratio,
              fig10_importance_and_coefs, fig11_calibration, fig12_walk_forward, fig13_vol_targeting,
              fig14_equity_drawdown, fig15_hourly_bars, fig16_hourly_forecast]:
        f()
