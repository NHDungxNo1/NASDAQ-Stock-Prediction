"""Walk-forward evaluation of every model listed in src/models.py::MODELS.

    python scripts/run_baselines.py          # real data from data/raw/prices.csv
    python scripts/run_baselines.py --demo   # simulated data, to test the pipeline
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, data, features, metrics, models  # noqa: E402
from src.validation import walk_forward_splits  # noqa: E402


def evaluate(table: pd.DataFrame, predict, initial_train: int, step: int) -> dict:
    """Run one model through walk-forward validation on one ticker's table."""
    y_true, y_pred = [], []
    for train_idx, test_idx in walk_forward_splits(len(table), initial_train, step):
        train, test = table.iloc[train_idx], table.iloc[test_idx]
        y_pred.append(np.asarray(predict(train, test), float))
        y_true.append(test[models.TARGET].to_numpy())
    y_true, y_pred = np.concatenate(y_true), np.concatenate(y_pred)
    first_train = table[models.TARGET].iloc[:initial_train]
    return {
        "RMSE": metrics.rmse(y_true, y_pred),
        "MAE": metrics.mae(y_true, y_pred),
        "MASE": metrics.mase(y_true, y_pred, first_train),
        "DirAcc": metrics.directional_accuracy(y_true, y_pred),
        "n_forecasts": len(y_true),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="use simulated prices")
    args = parser.parse_args()

    prices = data.make_demo_prices() if args.demo else data.load_prices()
    full = features.model_table(features.build_features(prices))

    rows = []
    for ticker, table in full.groupby("ticker"):
        table = table.reset_index(drop=True)
        for name, predict in models.MODELS.items():
            scores = evaluate(table, predict, config.INITIAL_TRAIN, config.STEP)
            rows.append({"ticker": ticker, "model": name, **scores})

    results = pd.DataFrame(rows).set_index(["ticker", "model"])
    pd.set_option("display.float_format", "{:.5f}".format)
    print(results.to_string())

    if args.demo:
        print("\n(Simulated data: pipeline check only, not a result.)")
    else:
        out = config.ROOT / "reports" / "results.csv"
        results.to_csv(out)
        print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
