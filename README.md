# Projects

This repository is intended to hold a collection of different projects.

At the moment, only one project is actively implemented:

- `market_risk`: a market risk engine workspace for downloading market data, running exploratory analysis, and generating supporting report material.

## Repository Tree

The tree below excludes anything matched by `.gitignore`.

```text
.
|-- README.md
`-- market_risk
    |-- 01_get_historical_data.py
    |-- 02_historical_study.py
    |-- README.md
    `-- figure
        |-- individual_assets_close_price_over_time.png
        |-- individual_assets_log_returns_over_time.png
        |-- jarque_bera_table.csv
        |-- kurtosis_table.csv
        |-- sp500_and_individual_assets_close_prices_side_by_side.png
        `-- sp500_log_returns_over_time.png
```