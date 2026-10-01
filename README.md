# NASDAQ Stock Forecasting

> **Status:** workspace scaffold. Replace this README with your own write-up as
> you go (template at the bottom).

## Setup (once)

```bash
cd stock-forecast
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Check that everything works

```bash
pytest                                   # 7 tests should pass
python scripts/run_baselines.py --demo   # pipeline on simulated data
python scripts/download_data.py          # real prices -> data/raw/prices.csv
python scripts/run_baselines.py          # baselines on real data -> reports/results.csv
jupyter lab notebooks/01_eda.ipynb
```

## What's where

```
src/config.py       tickers, dates, validation settings: change scope here
src/data.py         download / save / load prices (+ simulated demo data)
src/features.py     returns, lag + rolling features, target       <- TODOs
src/models.py       baselines (done) and your models (stubs)      <- TODOs
src/validation.py   walk-forward splits
src/metrics.py      RMSE, MAE, MASE, directional accuracy
scripts/            download_data.py, run_baselines.py
notebooks/          01_eda.ipynb                                  <- TODOs
tests/              look-ahead leak test + split tests
reports/            results.csv and figures for the README
```

## What's done vs. what's yours

Working: data download, return calculation, a starter feature set, three
baselines, walk-forward evaluation, metrics, and a test that fails if any
feature uses future information.

Yours to build (in this order):

- [ ] **Frame the question.** Write one sentence: what are you forecasting, for
      which tickers, at what horizon, judged by which metric? Edit `config.py`.
- [ ] **EDA.** Work through `notebooks/01_eda.ipynb` and answer each question.
- [ ] **Volatility target.** Add it in `features.py` (see TODO).
- [ ] **Features.** Add 3-5 more, each with a one-line reason. Keep `pytest` green.
- [ ] **ARIMA**, then **GARCH**, then **XGBoost** in `models.py`. Register each in
      `MODELS` and rerun `run_baselines.py`.
- [ ] **Interpret.** Which models beat which baseline, by how much, and why?
- [ ] **Write up.** Fill in the template below, add 3-4 figures.

## Two rules that keep the project honest

1. **No look-ahead.** A row dated *t* holds only what was known at the close of
   day *t*. Run `pytest` after every new feature.
2. **No shuffled splits.** All evaluation goes through `walk_forward_splits`.
   Tune hyperparameters inside the training window only.

---

## README template (fill in, then delete everything above)

### Question
### Data
### Exploration (2-3 key plots and what they show)
### Method (features, models, validation scheme)
### Results (table vs. baselines)
### Limitations
### How to reproduce
