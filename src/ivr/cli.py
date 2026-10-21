"""Command line: `ivr download|process|daily|hourly|sizing|figures|all`."""
from __future__ import annotations

import argparse

from . import data, report
from .config import SYMBOLS
from .config import REPORTS


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ivr")
    ap.add_argument("step", choices=["download", "process", "daily", "hourly", "sizing", "figures", "all"])
    a = ap.parse_args(argv)
    REPORTS.mkdir(exist_ok=True)
    s = a.step
    if s in ("download", "all"):
        data.download_all()
    if s in ("process", "all"):
        data.process()
    if s in ("daily", "all"):
        from . import forecast
        for sym in SYMBOLS:
            print(forecast.run_daily(sym)["table"].round(4).to_string())
    if s in ("hourly", "all"):
        from . import forecast
        for sym in SYMBOLS:
            print(forecast.run_hourly(sym)["table"].round(4).to_string())
    if s in ("sizing", "all"):
        tabs = {sym: report.run_sizing(sym) for sym in SYMBOLS}
        report.figure_cost_sensitivity(tabs)
    if s in ("figures", "all"):
        report.figure_forecasts()
        report.figure_loss_ratios()
        report.figure_seasonality()


if __name__ == "__main__":
    main()
