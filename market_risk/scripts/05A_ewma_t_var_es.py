#######################
#
# (0) LIBRARIES
#
#######################
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from path_utils import project_path
from risk_engine_utils import (
    build_latest_model_summary,
    build_model_backtest_summary,
    build_model_risk_metrics_table,
    build_portfolio_returns_from_frozen_weights,
    combine_price_data,
    compute_simple_returns,
    estimate_standardized_t_nu,
    extract_close_prices,
    filter_backtest_period,
    load_frozen_weights,
    load_optional_price_data,
    load_price_data,
    save_dataframe,
    standardized_t_expected_shortfall,
    standardized_t_quantile,
    validate_backtest_output_dates,
    validate_var_es_outputs,
)

logger = logging.getLogger(__name__)


#######################
#
# (1) GLOBAL VARIABLES
#
#######################
TICKERS = ["AAPL", "GOOG", "NVDA", "MSFT", "AMZN"]
BACKTEST_START = "2025-01-01"
BACKTEST_END = None
WINDOW_SIZE = 252
CONFIDENCE_LEVEL = 0.95
TAIL_PROBABILITY = 1.0 - CONFIDENCE_LEVEL
EWMA_LAMBDA = 0.94
CLOSE_COL = "Close"
MODEL_NAME = "EWMA-t"
FALLBACK_NU = 8.0

HISTORICAL_ASSET_FILEPATH = project_path("data", "historical", "top_tech_assets_prices.parquet")
NEW_DAILY_ASSET_FILEPATH = project_path("data", "new_daily", "top_tech_assets_prices.parquet")
FROZEN_WEIGHTS_FILEPATH = project_path("figure", "risk_engine", "frozen_portfolio_weights.csv")

VOLATILITY_MODEL_OUTPUT_DIR = project_path("figure", "risk_engine", "volatility_models")
EWMA_T_PORTFOLIO_VAR_ES_FILEPATH = VOLATILITY_MODEL_OUTPUT_DIR / "ewma_t_portfolio_var_es.csv"
EWMA_T_BACKTEST_SUMMARY_FILEPATH = VOLATILITY_MODEL_OUTPUT_DIR / "ewma_t_backtest_summary.csv"
EWMA_T_PARAMETER_SUMMARY_FILEPATH = VOLATILITY_MODEL_OUTPUT_DIR / "ewma_t_parameter_summary.csv"


#######################
#
# (2) HELPER FUNCTIONS
#
#######################
def load_full_portfolio_returns() -> pd.DataFrame:
    historical_asset_prices = load_price_data(HISTORICAL_ASSET_FILEPATH)
    new_daily_asset_prices = load_optional_price_data(NEW_DAILY_ASSET_FILEPATH)
    all_asset_prices = combine_price_data(historical_asset_prices, new_daily_asset_prices)
    close_prices = extract_close_prices(all_asset_prices, TICKERS, CLOSE_COL)
    asset_returns = compute_simple_returns(close_prices).dropna(how="any")
    frozen_weights = load_frozen_weights(FROZEN_WEIGHTS_FILEPATH, TICKERS)
    return build_portfolio_returns_from_frozen_weights(asset_returns, frozen_weights, TICKERS)


def compute_ewma_sigma_forecast(
    returns: pd.Series,
    window_size: int = WINDOW_SIZE,
    ewma_lambda: float = EWMA_LAMBDA
) -> pd.Series:
    sigma2_forecast = pd.Series(np.nan, index=returns.index, dtype=float)
    if len(returns) <= window_size:
        return sigma2_forecast

    sigma2_previous = float(returns.iloc[:window_size].var(ddof=1))
    sigma2_previous = max(sigma2_previous, 1e-12)
    sigma2_forecast.iloc[window_size] = sigma2_previous

    for return_position in range(window_size + 1, len(returns)):
        sigma2_previous = ewma_lambda * sigma2_previous + (1.0 - ewma_lambda) * returns.iloc[return_position - 1] ** 2
        sigma2_forecast.iloc[return_position] = max(sigma2_previous, 1e-12)

    return np.sqrt(sigma2_forecast)


def build_ewma_t_parameter_summary(portfolio_returns: pd.DataFrame) -> pd.DataFrame:
    parameter_rows = []
    backtest_start_timestamp = pd.Timestamp(BACKTEST_START)

    for portfolio_name in portfolio_returns.columns:
        portfolio_return_series = portfolio_returns[portfolio_name]
        ewma_sigma = compute_ewma_sigma_forecast(portfolio_return_series)
        pre_backtest_mask = portfolio_return_series.index < backtest_start_timestamp
        standardized_residuals = (portfolio_return_series.loc[pre_backtest_mask] / ewma_sigma.loc[pre_backtest_mask]).replace([np.inf, -np.inf], np.nan).dropna()
        estimated_nu, num_residuals_used, estimation_status = estimate_standardized_t_nu(
            standardized_residuals,
            fallback_nu=FALLBACK_NU,
        )

        if estimation_status.lower().startswith("fallback"):
            logger.warning("%s for %s", estimation_status, portfolio_name)

        parameter_rows.append({
            "Portfolio": portfolio_name,
            "Model": MODEL_NAME,
            "EWMA Lambda": EWMA_LAMBDA,
            "Estimated nu": estimated_nu,
            "Number of Residuals Used": num_residuals_used,
            "Estimation Status": estimation_status,
        })

    return pd.DataFrame(parameter_rows)


def build_ewma_t_var_es(
    portfolio_returns: pd.DataFrame,
    parameter_summary: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rolling_var = pd.DataFrame(index=portfolio_returns.index, columns=portfolio_returns.columns, dtype=float)
    rolling_es = pd.DataFrame(index=portfolio_returns.index, columns=portfolio_returns.columns, dtype=float)

    for portfolio_name in portfolio_returns.columns:
        portfolio_return_series = portfolio_returns[portfolio_name]
        ewma_sigma = compute_ewma_sigma_forecast(portfolio_return_series)
        estimated_nu = float(parameter_summary.loc[parameter_summary["Portfolio"] == portfolio_name, "Estimated nu"].iloc[0])
        q_alpha = standardized_t_quantile(estimated_nu, TAIL_PROBABILITY)
        es_alpha_standardized = standardized_t_expected_shortfall(estimated_nu, TAIL_PROBABILITY)

        rolling_var[portfolio_name] = (-ewma_sigma * q_alpha).clip(lower=0.0)
        rolling_es[portfolio_name] = (-ewma_sigma * es_alpha_standardized).clip(lower=0.0)

    validate_var_es_outputs(rolling_var, rolling_es, MODEL_NAME, logger)
    return rolling_var, rolling_es


#######################
#
# (3) MAIN
#
#######################
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger.info("INITIALIZING EWMA-t VaR/ES ENGINE...")

    portfolio_returns = load_full_portfolio_returns()
    parameter_summary = build_ewma_t_parameter_summary(portfolio_returns)
    rolling_var, rolling_es = build_ewma_t_var_es(portfolio_returns, parameter_summary)

    ewma_t_risk_metrics = build_model_risk_metrics_table(
        portfolio_returns,
        rolling_var,
        rolling_es,
        model_name=MODEL_NAME,
        confidence_level=CONFIDENCE_LEVEL,
        window_size=WINDOW_SIZE,
    )
    ewma_t_risk_metrics = filter_backtest_period(ewma_t_risk_metrics, BACKTEST_START, BACKTEST_END)
    validate_backtest_output_dates(ewma_t_risk_metrics, BACKTEST_START, MODEL_NAME)

    ewma_t_backtest_summary = build_model_backtest_summary(
        ewma_t_risk_metrics,
        model_name=MODEL_NAME,
        confidence_level=CONFIDENCE_LEVEL,
        window_size=WINDOW_SIZE,
    )
    latest_ewma_t_summary = build_latest_model_summary(
        ewma_t_risk_metrics,
        confidence_level=CONFIDENCE_LEVEL,
        window_size=WINDOW_SIZE,
    )

    save_dataframe(parameter_summary, str(EWMA_T_PARAMETER_SUMMARY_FILEPATH))
    save_dataframe(ewma_t_risk_metrics, str(EWMA_T_PORTFOLIO_VAR_ES_FILEPATH))
    save_dataframe(ewma_t_backtest_summary, str(EWMA_T_BACKTEST_SUMMARY_FILEPATH))

    logger.info("EWMA-t parameter summary\n%s", parameter_summary.to_string(index=False))
    logger.info("Latest EWMA-t VaR/ES summary\n%s", latest_ewma_t_summary.to_string(index=False))
    logger.info("EWMA-t VaR backtesting summary\n%s", ewma_t_backtest_summary.to_string(index=False))
    logger.info("Saved outputs to %s", VOLATILITY_MODEL_OUTPUT_DIR)


#######################
#
# (4) RUN MAIN
#
#######################
if __name__ == "__main__":
    main()
