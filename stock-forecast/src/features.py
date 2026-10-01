"""Turning prices into a modelling table.

THE ONE RULE OF THIS FILE
-------------------------
A row dated `t` may only contain information known at the CLOSE of day `t`.
The target on that row is what happens on day `t+1`.

So: `.shift(k)` with k >= 0 and backward-looking `.rolling()` are fine.
`.shift(-1)` is allowed in exactly one place: building the target.
`tests/test_features.py` checks this rule automatically. Keep it passing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Add daily log returns, computed separately for each ticker."""
    df = prices.sort_values(["ticker", "date"]).reset_index(drop=True).copy()
    df["log_ret"] = np.log(df["close"]).groupby(df["ticker"]).diff()
    return df


def build_features(prices: pd.DataFrame, lags=(0, 1, 2, 5), windows=(5, 21)) -> pd.DataFrame:
    """Starter feature set plus the target. Returns one row per ticker-day."""
    df = add_returns(prices)
    by_ticker = df.groupby("ticker")["log_ret"]

    # Lagged returns. lag0 = today's return, lag1 = yesterday's, ...
    for k in lags:
        df[f"ret_lag{k}"] = by_ticker.shift(k)

    # Rolling mean and rolling volatility over the last `w` days (including today).
    for w in windows:
        df[f"roll_mean{w}"] = by_ticker.transform(lambda s, w=w: s.rolling(w).mean())
        df[f"roll_vol{w}"] = by_ticker.transform(lambda s, w=w: s.rolling(w).std())

    # TODO (you): add more features and justify each one in the README.
    #   - RSI, MACD            - volume change vs. its rolling average
    #   - high-low range       - day of week / month effects
    # After adding one, add its name to FEATURE_PREFIXES below and run pytest.

    # Target: tomorrow's log return. The only negative shift in the project.
    df["target_ret"] = by_ticker.shift(-1)

    # TODO (you): add a volatility target, e.g. tomorrow's absolute or squared
    # return, or a forward 5-day realized volatility. Volatility is far more
    # forecastable than returns, so this is where the interesting results are.

    return df


FEATURE_PREFIXES = ("ret_lag", "roll_mean", "roll_vol")


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]


def model_table(df: pd.DataFrame, target="target_ret") -> pd.DataFrame:
    """Drop warm-up rows (rolling windows not full yet) and the last row (no target)."""
    return df.dropna(subset=feature_columns(df) + [target]).reset_index(drop=True)
