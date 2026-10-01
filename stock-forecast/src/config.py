"""Project-wide settings. Change things here, not scattered through the code."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
FIGURES_DIR = ROOT / "reports" / "figures"

# --- Scope (Phase 1 of the roadmap) -----------------------------------------
# QQQ tracks the NASDAQ-100. Keep the list short: depth beats breadth here.
TICKERS = ["QQQ", "AAPL", "MSFT", "NVDA"]
START = "2014-01-01"
END = None  # None = up to today

PRICES_FILE = RAW_DIR / "prices.csv"

# --- Walk-forward validation (Phase 4) --------------------------------------
INITIAL_TRAIN = 756  # ~3 years of trading days before the first forecast
STEP = 21            # refit roughly once a month
