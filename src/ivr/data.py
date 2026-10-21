"""Download and parse Binance public 1-minute klines (data.binance.vision).

Raw zips are untrusted: they are verified against the published SHA-256 checksum, parsed
with pandas as plain CSV, and never executed or extracted to disk.
"""
from __future__ import annotations

import hashlib
import io
import zipfile

import duckdb
import pandas as pd
import requests

from .config import END_MONTH, PROC, RAW, START_MONTH, SYMBOLS

BASE = "https://data.binance.vision/data/spot/monthly/klines"
COLS = ["open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
KEEP = ["open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base"]


def months(start: str = START_MONTH, end: str = END_MONTH) -> list[str]:
    return [p.strftime("%Y-%m") for p in pd.period_range(start, end, freq="M")]


def download(symbol: str, month: str, session: requests.Session) -> bool:
    out = RAW / symbol / f"{symbol}-1m-{month}.zip"
    if out.exists():
        return True
    url = f"{BASE}/{symbol}/1m/{symbol}-1m-{month}.zip"
    r = session.get(url, timeout=120)
    if r.status_code == 404:
        return False
    r.raise_for_status()
    c = session.get(url + ".CHECKSUM", timeout=60)
    c.raise_for_status()
    expected = c.text.split()[0]
    if hashlib.sha256(r.content).hexdigest() != expected:
        raise ValueError(f"checksum mismatch for {url}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(r.content)
    return True


def download_all(symbols=SYMBOLS) -> None:
    s = requests.Session()
    for sym in symbols:
        missing = [m for m in months() if not download(sym, m, s)]
        print(f"{sym}: downloaded/cached, months not available: {missing}")


def _read_zip(path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as z:
        name = z.namelist()[0]
        raw = z.read(name)
    # Some files carry a header row, newer ones do not.
    first = raw[:20].decode("ascii", "ignore")
    df = pd.read_csv(io.BytesIO(raw), header=0 if first[:1].isalpha() else None, names=COLS)
    t = df["open_time"].astype("int64")
    unit = "us" if t.iloc[0] > 1e14 else "ms"  # Binance switched to microseconds in 2025
    df.index = pd.to_datetime(t, unit=unit, utc=True).dt.tz_localize(None)
    df.index.name = "ts"
    return df[KEEP].astype("float64")


def build_1m(symbol: str) -> pd.DataFrame:
    parts = [_read_zip(p) for p in sorted((RAW / symbol).glob(f"{symbol}-1m-*.zip"))]
    df = pd.concat(parts)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df


def process(symbols=SYMBOLS) -> None:
    PROC.mkdir(parents=True, exist_ok=True)
    for sym in symbols:
        df = build_1m(sym)
        path = PROC / f"{sym}_1m.parquet"
        con = duckdb.connect()
        con.register("d", df.reset_index())
        con.execute(f"COPY d TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        q = con.execute(f"SELECT count(*), min(ts), max(ts) FROM '{path}'").fetchone()
        full = int((df.index[-1] - df.index[0]).total_seconds() // 60) + 1
        print(f"{sym}: {q[0]:,} 1-min bars {q[1]} -> {q[2]}; missing minutes {full - q[0]:,} "
              f"({(full - q[0]) / full:.4%})")


def load_1m(symbol: str) -> pd.DataFrame:
    return pd.read_parquet(PROC / f"{symbol}_1m.parquet").set_index("ts")
