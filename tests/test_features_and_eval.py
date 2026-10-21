import numpy as np
import pandas as pd

from ivr import features as F
from ivr.evaluate import diebold_mariano, qlike, mse
from ivr.sizing import backtest, target_weights


def test_bipower_below_rv_with_jump_and_close_without():
    rng = np.random.default_rng(3)
    idx = pd.date_range("2022-01-01", periods=288 * 2, freq="5min")
    r = pd.Series(rng.standard_normal(len(idx)) * 1e-3, index=idx)
    key = pd.Series(idx.floor("D"), index=idx)
    rv = (r ** 2).groupby(key).sum()
    bv = F._bipower(r.abs(), key)
    assert abs(bv.iloc[0] / rv.iloc[0] - 1) < 0.25
    r.iloc[100] = 0.05                               # jump on day 1
    rv2 = (r ** 2).groupby(key).sum()
    bv2 = F._bipower(r.abs(), key)
    assert (rv2.iloc[0] - bv2.iloc[0]) / rv2.iloc[0] > 0.5
    assert abs(bv2.iloc[1] - bv.iloc[1]) < 1e-12     # day 2 untouched (no cross-day pairs)


def test_corwin_schultz_nonnegative_and_zero_for_flat_bars():
    h = pd.Series([101.0, 101.0, 102.0, 101.5]); l = pd.Series([100.0, 100.0, 100.5, 100.2])
    s = F.corwin_schultz(h, l)
    assert (s.dropna() >= 0).all()
    flat = F.corwin_schultz(pd.Series([100.0] * 4) * 1.0 + 1e-9, pd.Series([100.0] * 4))
    assert flat.dropna().max() < 1e-6


def test_qlike_properties():
    rv = np.array([1.0, 2.0, 0.5])
    assert np.allclose(qlike(rv, rv), 0)
    assert (qlike(rv, rv * 2) > 0).all() and (qlike(rv, rv / 2) > 0).all()
    assert qlike(rv, rv / 2)[0] > qlike(rv, rv * 2)[0]    # under-prediction penalised more
    assert np.allclose(mse(rv, rv), 0)


def test_dm_detects_better_model_and_is_antisymmetric():
    rng = np.random.default_rng(0)
    base = rng.gamma(2, 1, 1500)
    better = base - rng.gamma(2, 0.1, 1500) - 0.05
    s, p = diebold_mariano(base, better)
    assert s > 3 and p < 0.01
    s2, p2 = diebold_mariano(better, base)
    assert np.isclose(s, -s2) and np.isclose(p, p2)
    noise_s, noise_p = diebold_mariano(base, base + rng.normal(0, 0.1, 1500))
    assert noise_p > 0.01


def test_sizing_costs_turnover_and_cap():
    var = np.full(300, (0.4 ** 2) / 365)
    w = target_weights(var, 0.40)
    assert np.allclose(w, 1.0)
    w = target_weights(var / 4, 0.40, cap=1.0)         # lower vol -> capped at 1, no leverage
    assert w.max() <= 1.0
    r = np.random.default_rng(0).normal(0, 0.02, 300)
    net0, turn0 = backtest(w, r, 0.0)
    assert np.allclose(net0, w * r)
    net1, _ = backtest(w, r, 10.0)
    assert net1.sum() < net0.sum()
    # constant weight 0.5: after the initial purchase, drift rebalancing only
    wc = np.full(300, 0.5)
    _, tc = backtest(wc, r, 10)
    assert np.isclose(tc[0], 0.5)
    # no-trade band lowers turnover
    v = (0.4 ** 2) / 365 * np.exp(np.random.default_rng(1).normal(0, 0.5, 300))
    _, t_free = backtest(target_weights(v, 0.4, cap=1.5), r, 10)
    _, t_band = backtest(target_weights(v, 0.4, cap=1.5, band=0.1), r, 10)
    assert t_band.sum() < t_free.sum()
