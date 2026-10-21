from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
START_MONTH = "2019-01"
END_MONTH = "2026-09"  # last complete month at time of writing; missing months are skipped

# Walk-forward settings (fixed a priori, never tuned on the test period)
MIN_TRAIN_DAYS = 730
REFIT_EVERY_DAYS = 30
SEED = 7

# Day = UTC calendar day (crypto trades 24/7), 288 five-minute bars per day.
BARS_PER_DAY = 288
