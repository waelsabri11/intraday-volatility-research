"""Leakage tests: features at period p must not depend on any data after the end of p."""
import numpy as np
import pandas as pd

from ivr import features as F
from ivr.evaluate import walk_forward
from ivr.models import garch_forecast

TARGETS = ["target_lrv", "target_rv", "target_ret"]


def _feature_cols(df):
    return [c for c in df.columns if c not in TARGETS]


def test_daily_features_unchanged_by_truncation(bars):
    full = F.daily_features(bars)
    for cut_day in ["2022-02-10", "2022-02-25", "2022-03-05"]:
        end = pd.Timestamp(cut_day) + pd.Timedelta(days=1)       # keep data through end of cut_day
        trunc = F.daily_features(bars[bars.index < end])
        a = full.loc[:cut_day, _feature_cols(full)]
        b = trunc.loc[:cut_day, _feature_cols(trunc)]
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12, atol=1e-14)


def test_daily_features_unchanged_by_future_shock(bars):
    cut = pd.Timestamp("2022-02-20") + pd.Timedelta(days=1)
    shocked = bars.copy()
    fut = shocked.index >= cut
    shocked.loc[fut, ["open", "high", "low", "close"]] *= 3.0          # absurd future prices
    shocked.loc[fut, ["volume", "quote_volume", "trades", "taker_buy_base"]] *= 50
    a = F.daily_features(bars).loc[:"2022-02-20", _feature_cols(F.daily_features(bars))]
    sh = F.daily_features(shocked)
    # the price jump at `cut` creates a return on day cut only; days before are untouched
    b = sh.loc[:"2022-02-20", _feature_cols(sh)]
    pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-12, atol=1e-14)


def test_hourly_features_unchanged_by_truncation(bars):
    full = F.hourly_features(bars)
    cut = pd.Timestamp("2022-02-15 13:00")
    trunc = F.hourly_features(bars[bars.index < cut + pd.Timedelta(hours=1)])
    cols = _feature_cols(full)
    pd.testing.assert_frame_equal(full.loc[:cut, cols], trunc.loc[:cut, cols],
                                  check_exact=False, rtol=1e-12, atol=1e-14)


def test_target_is_next_period_rv(bars):
    d = F.daily_features(bars)
    np.testing.assert_allclose(d["target_lrv"].iloc[:-1].values, d["lrv"].iloc[1:].values)
    np.testing.assert_allclose(d["target_rv"].iloc[:-1].values, d["rv"].iloc[1:].values)
    h = F.hourly_features(bars)
    np.testing.assert_allclose(h["target_lrv"].iloc[:-1].values, h["lrv"].iloc[1:].values)


def test_walk_forward_trains_only_on_realised_targets():
    n, seen = 400, []

    def fp(tr, te):
        seen.append((tr.max(), te.min(), te.max()))
        assert tr.max() < te.min()        # train rows strictly precede every test origin
        assert set(tr).isdisjoint(set(te))
        return np.zeros(len(te))

    out, folds = walk_forward(n, np.ones(n, bool), 100, 30, fp)
    # row j's target is period j+1; it is realised by the end of period s iff j+1 <= s
    assert all(last + 1 <= s for last, s in folds)
    assert np.isnan(out[:100]).all() and np.isfinite(out[100:]).all()


def test_garch_forecast_uses_only_past_returns():
    rng = np.random.default_rng(1)
    r = rng.standard_t(5, 900) * 2
    full = garch_forecast(r, 600, gjr=True)
    r2 = r.copy()
    r2[750:] *= 10                                   # alter the future after origin 749
    alt = garch_forecast(r2, 600, gjr=True)
    np.testing.assert_allclose(full[600:750], alt[600:750])   # forecasts up to 749 unchanged
    r3 = r.copy()
    r3[600:] *= 10                                   # params are estimated on r[:600] only
    np.testing.assert_allclose(garch_forecast(r3, 600, gjr=True)[:600][-1:], full[:600][-1:])


def test_leakage_check_would_catch_a_leaky_feature(bars):
    """Meta-test: the truncation methodology flags a feature that peeks one period ahead."""
    def leaky(df):
        d = F.daily_features(df)
        d["leaky"] = d["lrv"].shift(-1)
        return d

    cut = "2022-02-20"
    full, trunc = leaky(bars), leaky(bars[bars.index < pd.Timestamp(cut) + pd.Timedelta(days=1)])
    a, b = full.loc[cut, "leaky"], trunc.loc[cut, "leaky"]
    assert not (np.isfinite(b) and np.isclose(a, b))
