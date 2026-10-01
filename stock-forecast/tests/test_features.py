"""The look-ahead test: the most important test in the project.

If changing FUTURE prices changes a feature on a PAST row, that feature is
using information it could not have had. Run this every time you add a feature.
"""
import numpy as np

from src import data, features


def test_features_do_not_see_the_future():
    prices = data.make_demo_prices(tickers=("AAA",), n_days=300)
    cutoff = 200  # rows after this get scrambled

    tampered = prices.copy()
    tampered.loc[cutoff + 1 :, ["open", "high", "low", "close"]] *= 3.0
    tampered.loc[cutoff + 1 :, "volume"] *= 7

    original = features.build_features(prices)
    changed = features.build_features(tampered)
    cols = features.feature_columns(original)

    assert cols, "no feature columns found: update FEATURE_PREFIXES"
    np.testing.assert_allclose(
        original.loc[:cutoff, cols].to_numpy(),
        changed.loc[:cutoff, cols].to_numpy(),
        equal_nan=True,
        err_msg="A feature on a past row changed when future prices changed.",
    )


def test_target_is_next_day_return():
    df = features.build_features(data.make_demo_prices(tickers=("AAA",), n_days=50))
    np.testing.assert_allclose(
        df["target_ret"].to_numpy()[:-1], df["log_ret"].to_numpy()[1:]
    )


def test_tickers_do_not_bleed_into_each_other():
    df = features.build_features(data.make_demo_prices(tickers=("AAA", "BBB"), n_days=50))
    first_rows = df.groupby("ticker").head(1)
    assert first_rows["log_ret"].isna().all()
