"""Forecasting models.

Every model is a function with the same shape, so the evaluation loop in
`scripts/run_baselines.py` can treat them all identically:

    predict(train: DataFrame, test: DataFrame) -> array of len(test)

`train` and `test` are slices of the model table (see features.model_table).
Fit on `train` only. Never look at test[target].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "target_ret"

# --- Baselines (done: these are the bar every model below has to clear) -----


def zero_forecast(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Random walk in prices: the best guess for tomorrow's return is 0."""
    return np.zeros(len(test))


def mean_forecast(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Random walk with drift: tomorrow's return = average past return."""
    return np.full(len(test), train[TARGET].mean())


def last_value_forecast(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """Naive persistence: tomorrow's return = today's return."""
    return test["ret_lag0"].to_numpy()


BASELINES = {
    "zero (random walk)": zero_forecast,
    "mean (drift)": mean_forecast,
    "last value (naive)": last_value_forecast,
}

# --- Your models -------------------------------------------------------------
# Write these yourself; that is the part of the project you will be asked about
# in interviews. Add each finished one to MODELS and it shows up in the results.


def arima_forecast(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """TODO: ARIMA on the return series.

    Hints: statsmodels.tsa.arima.model.ARIMA. Pick (p, d, q) from your ACF/PACF
    plots in the EDA notebook, and say why in the README. Returns are already
    roughly stationary, so what should d be?
    """
    raise NotImplementedError


def garch_forecast(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """TODO: GARCH(1,1) volatility forecast (needs the volatility target first).

    Hints: `arch` package, arch_model(returns * 100, vol="GARCH", p=1, q=1).
    Compare against a rolling-volatility baseline, not against zero_forecast.
    """
    raise NotImplementedError


def xgboost_forecast(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """TODO: gradient-boosted trees on the lag features.

    Hints: features.feature_columns(train) gives the X columns. Tune
    hyperparameters inside the training window only (e.g. hold out its last
    20%), never on the test fold.
    """
    raise NotImplementedError


MODELS = {
    **BASELINES,
    # "ARIMA": arima_forecast,
    # "GARCH(1,1)": garch_forecast,
    # "XGBoost": xgboost_forecast,
}
