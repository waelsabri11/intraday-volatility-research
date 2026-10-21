import numpy as np
import pandas as pd
import pytest


def make_1m(days=70, seed=0, start="2022-01-01"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 1440, freq="min")
    hod = idx.hour.values
    season = 1 + 0.5 * np.exp(-((hod - 14) ** 2) / 8)
    vol_day = np.repeat(np.exp(rng.normal(0, 0.3, days)), 1440)
    r = rng.standard_normal(len(idx)) * 6e-4 * season * vol_day
    close = 20000 * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]]
    spread = np.abs(rng.standard_normal(len(idx))) * 2e-4
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    vol = rng.gamma(2.0, 3.0, len(idx)) * season
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": vol,
        "quote_volume": vol * close, "trades": rng.integers(50, 500, len(idx)).astype(float),
        "taker_buy_base": vol * rng.uniform(0.3, 0.7, len(idx)),
    }, index=pd.Index(idx, name="ts"))


@pytest.fixture(scope="session")
def bars():
    return make_1m()
