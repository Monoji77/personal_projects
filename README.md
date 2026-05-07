# Projects

This repository is intended to hold multiple projects. At the moment, the only active project is `market_risk`.

## Active Project
![Portfolio Risk Lab](market_risk/additional_materials/image-1.png)
`market_risk` is a market risk engine built around a fixed five-asset technology universe:

- `AAPL`
- `AMZN`
- `GOOG`
- `MSFT`
- `NVDA`

The project currently includes:

- historical data loading and preprocessing
- return-based exploratory analysis
- fixed portfolio construction
- historical VaR/ES backtesting
- EWMA-t and GARCH-t model comparison
- deterministic stress testing
- covariance-based risk attribution
- an employer-facing Streamlit dashboard with an interactive portfolio lab

### Quick Start

From the repository root:

```bash
pip install -r market_risk/requirements.txt
streamlit run market_risk/app.py
```

From inside `market_risk`:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Repository Tree

The tree below is intentionally concise and excludes anything ignored by `.gitignore`.

```text
.
|-- README.md
`-- market_risk
    |-- app.py
    |-- requirements.txt
    |-- data
    |   |-- historical
    |   `-- new_daily
    |-- figure
    |   |-- historical_study
    |   |-- portfolio_study
    |   `-- risk_engine
    `-- scripts
        |-- 01_get_historical_data.py
        |-- 02_historical_study.py
        |-- 03_portfolio_study.py
        |-- 04A_portfolio_construction.py
        |-- 04B_market_risk_engine.py
        |-- 04C_risk_engine_visualizations.py
        |-- 04D_var_backtesting_tests.py
        |-- 05A_ewma_t_var_es.py
        |-- 05B_garch_t_var_es.py
        |-- 05C_volatility_model_comparison.py
        |-- 06_stress_testing.py
        |-- 07_risk_attribution.py
        |-- dashboard_utils.py
        |-- path_utils.py
        `-- risk_engine_utils.py
```

## Notes

- `market_risk/additional_materials/` is intentionally omitted above because it is ignored by `.gitignore`.
- Generated figures and CSV outputs are organized under `market_risk/figure/`.
- The main dashboard entrypoint is `market_risk/app.py`.
