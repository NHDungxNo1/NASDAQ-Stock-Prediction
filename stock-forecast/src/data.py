"""Getting prices in and out of the project.

Everything downstream expects one tidy ("long") table:

    date | ticker | open | high | low | close | volume

with `close` already adjusted for splits and dividends.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

COLUMNS = ["date", "ticker", "open", "high", "low", "close", "volume"]


def download_prices(tickers=None, start=None, end=None) -> pd.DataFrame:
    """Download daily adjusted prices from Yahoo Finance as a tidy table."""
    import yfinance as yf  # imported here so the rest of the project works offline

    tickers = list(tickers or config.TICKERS)
    raw = yf.download(
        tickers,
        start=start or config.START,
        end=end or config.END,
        auto_adjust=True,   # adjusted prices: splits/dividends handled for us
        group_by="ticker",
        progress=False,
    )
    if raw.empty:
        raise RuntimeError("Yahoo Finance returned no data. Check tickers and connection.")
    if not isinstance(raw.columns, pd.MultiIndex):  # older yfinance, single ticker
        raw = pd.concat({tickers[0]: raw}, axis=1)

    frames = []
    for ticker in tickers:
        one = raw[ticker].dropna(how="all").copy()
        one.columns = [str(c).lower() for c in one.columns]
        one.index.name = "date"
        one = one.reset_index()
        one["ticker"] = ticker
        frames.append(one[COLUMNS])
    return pd.concat(frames, ignore_index=True)


def save_prices(prices: pd.DataFrame, path=None) -> None:
    path = path or config.PRICES_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(path, index=False)


def load_prices(path=None) -> pd.DataFrame:
    path = path or config.PRICES_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )
    prices = pd.read_csv(path, parse_dates=["date"])
    return prices.sort_values(["ticker", "date"]).reset_index(drop=True)


def make_demo_prices(tickers=("AAA", "BBB"), n_days=1500, seed=0) -> pd.DataFrame:
    """Simulated prices with volatility clustering.

    For checking that the pipeline runs without an internet connection.
    NEVER report results from this data: it is fake.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_days)
    frames = []
    for ticker in tickers:
        vol = np.empty(n_days)
        ret = np.empty(n_days)
        vol[0] = 0.012
        for t in range(n_days):
            if t > 0:  # simple GARCH(1,1)-style recursion
                var = 2e-6 + 0.08 * ret[t - 1] ** 2 + 0.90 * vol[t - 1] ** 2
                vol[t] = np.sqrt(var)
            ret[t] = 0.0003 + vol[t] * rng.standard_normal()
        close = 100 * np.exp(np.cumsum(ret))
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "open": close * (1 + 0.002 * rng.standard_normal(n_days)),
                    "high": close * (1 + np.abs(0.005 * rng.standard_normal(n_days))),
                    "low": close * (1 - np.abs(0.005 * rng.standard_normal(n_days))),
                    "close": close,
                    "volume": rng.integers(1_000_000, 5_000_000, n_days),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)[COLUMNS]
