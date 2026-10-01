"""Forecast error metrics."""
from __future__ import annotations

import numpy as np


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mase(y_true, y_pred, y_train) -> float:
    """MAE divided by the in-sample MAE of the naive 'same as yesterday' forecast.

    Below 1 = better than naive. Above 1 = worse than naive.
    """
    y_train = np.asarray(y_train, float)
    scale = np.mean(np.abs(np.diff(y_train)))
    return mae(y_true, y_pred) / float(scale)


def directional_accuracy(y_true, y_pred) -> float:
    """Share of days where the forecast got the sign right.

    Days where the forecast is exactly zero make no directional call and are
    skipped; returns NaN if the model never makes a call.
    """
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    called = y_pred != 0
    if not called.any():
        return float("nan")
    return float(np.mean(np.sign(y_true[called]) == np.sign(y_pred[called])))
