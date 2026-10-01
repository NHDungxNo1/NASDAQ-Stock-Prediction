"""Download prices for the tickers in src/config.py and save them to data/raw/.

    python scripts/download_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, data  # noqa: E402


def main() -> None:
    print(f"Downloading {config.TICKERS} from {config.START} ...")
    prices = data.download_prices()
    data.save_prices(prices)
    summary = prices.groupby("ticker")["date"].agg(["min", "max", "count"])
    print(summary.to_string())
    print(f"\nSaved {len(prices):,} rows to {config.PRICES_FILE}")


if __name__ == "__main__":
    main()
