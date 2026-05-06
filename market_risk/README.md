# Project Collection

This repository is intended to hold a collection of my personal projects. At the moment,
only one project is actively implemented: the market risk engine future works on data engineering pipelines

## Current Project

The current project is a market risk analysis workspace built around daily market data for the S&P 500 and
five large-cap technology stocks:

- `AAPL`
- `AMZN`
- `GOOG`
- `MSFT`
- `NVDA`

The current market risk engine workflow has two main stages:

1. Download and store adjusted market data.
2. Run exploratory analysis on adjusted close prices and derived log returns.

## Current Project Workflow

`01_get_historical_data.py`

- Downloads daily Yahoo Finance data from `2016-05-05` to `2026-05-04`.
- Writes local parquet files under `data/`.
- Uses `auto_adjust=True`, so the stored price series are already adjusted for splits and dividends.

`02_historical_study.py`

- Preprocesses the raw market data once to obtain adjusted close price frames.
- Reuses shared helpers for date slicing, log-return construction, and distribution diagnostics.
- Produces exploratory plots and summary diagnostics such as excess kurtosis and Jarque-Bera statistics.

## Current Repository Tree

The tree below intentionally excludes anything matched by `.gitignore`.

```text
market_risk/
|-- 01_get_historical_data.py
|-- 02_historical_study.py
|-- README.md
`-- figure/
    |-- individual_assets_close_price_over_time.png
    |-- individual_assets_log_returns_over_time.png
    |-- jarque_bera_table.csv
    |-- kurtosis_table.csv
    |-- sp500_and_individual_assets_close_prices_side_by_side.png
    `-- sp500_log_returns_over_time.png
```

## Ignored Local Artifacts

The following paths are intentionally omitted from the tree because they are ignored by the parent `.gitignore`:

- `data/`
- `additional_materials/`
- `*.pdf`
- `__pycache__/`
- virtual environment directories such as `.venv/`

## Current Outputs

The exploratory study currently generates:

- A side-by-side close-price comparison for the S&P 500 and the five single-name assets.
- A panel of per-asset log return plots.
- An S&P 500 log return plot.
- CSV diagnostics for excess kurtosis and Jarque-Bera normality testing.

## Future Expansion

The repository structure is meant to grow beyond the market risk engine. Additional projects can be added
later, but they are not part of the current workspace yet.
