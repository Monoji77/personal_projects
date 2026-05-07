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

try:
    from arch import arch_model
except ModuleNotFoundError:
    arch_model = None

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
CLOSE_COL = "Close"
MODEL_NAME = "GARCH-t"
GARCH_REFIT_FREQUENCY = 20

HISTORICAL_ASSET_FILEPATH = project_path("data", "historical", "top_tech_assets_prices.parquet")
NEW_DAILY_ASSET_FILEPATH = project_path("data", "new_daily", "top_tech_assets_prices.parquet")
FROZEN_WEIGHTS_FILEPATH = project_path("figure", "risk_engine", "frozen_portfolio_weights.csv")

VOLATILITY_MODEL_OUTPUT_DIR = project_path("figure", "risk_engine", "volatility_models")
GARCH_T_PORTFOLIO_VAR_ES_FILEPATH = VOLATILITY_MODEL_OUTPUT_DIR / "garch_t_portfolio_var_es.csv"
GARCH_T_BACKTEST_SUMMARY_FILEPATH = VOLATILITY_MODEL_OUTPUT_DIR / "garch_t_backtest_summary.csv"
GARCH_T_PARAMETER_SUMMARY_FILEPATH = VOLATILITY_MODEL_OUTPUT_DIR / "garch_t_parameter_summary.csv"


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


def fit_garch_t_model(returns_scaled: pd.Series):
    model = arch_model(
        returns_scaled,
        mean="Constant",
        vol="GARCH",
        p=1,
        q=1,
        dist="t",
        rescale=False,
    )
    return model.fit(disp="off", show_warning=False)


def compute_garch_t_var_es_for_portfolio(portfolio_returns: pd.Series) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    returns_scaled = portfolio_returns * 100.0
    backtest_mask = portfolio_returns.index >= pd.Timestamp(BACKTEST_START)
    if BACKTEST_END is not None:
        backtest_mask &= portfolio_returns.index < pd.Timestamp(BACKTEST_END)
    backtest_positions = np.flatnonzero(backtest_mask)

    rolling_var = pd.Series(np.nan, index=portfolio_returns.index, dtype=float)
    rolling_es = pd.Series(np.nan, index=portfolio_returns.index, dtype=float)
    parameter_history_rows = []
    failed_refits = 0
    previous_fit_parameters = None
    previous_sigma2 = None
    previous_epsilon = None

    for block_start_index in range(0, len(backtest_positions), GARCH_REFIT_FREQUENCY):
        block_positions = backtest_positions[block_start_index:block_start_index + GARCH_REFIT_FREQUENCY]
        first_position = int(block_positions[0])
        fit_sample = returns_scaled.iloc[:first_position].dropna()

        if len(fit_sample) < WINDOW_SIZE:
            continue

        try:
            fitted_model = fit_garch_t_model(fit_sample)
            fitted_parameters = fitted_model.params
            mu = float(fitted_parameters["mu"])
            omega = float(fitted_parameters["omega"])
            alpha = float(fitted_parameters["alpha[1]"])
            beta = float(fitted_parameters["beta[1]"])
            nu = float(fitted_parameters["nu"])

            one_step_ahead_forecast = fitted_model.forecast(horizon=1, reindex=False)
            sigma2_block_start = float(one_step_ahead_forecast.variance.iloc[-1, 0])

            previous_fit_parameters = {
                "mu": mu,
                "omega": omega,
                "alpha[1]": alpha,
                "beta[1]": beta,
                "nu": nu,
            }
            parameter_history_rows.append({
                "Portfolio": portfolio_returns.name,
                "Refit Start Date": portfolio_returns.index[first_position],
                "Fit Sample End Date": portfolio_returns.index[first_position - 1],
                "Number of Fit Observations": len(fit_sample),
                "mu": mu,
                "omega": omega,
                "alpha[1]": alpha,
                "beta[1]": beta,
                "alpha[1] + beta[1]": alpha + beta,
                "nu": nu,
            })

            if alpha + beta >= 1.0:
                logger.warning(
                    "%s alpha[1] + beta[1] is %.4f for %s at refit date %s.",
                    MODEL_NAME,
                    alpha + beta,
                    portfolio_returns.name,
                    portfolio_returns.index[first_position].date(),
                )
        except Exception as error:
            failed_refits += 1
            if previous_fit_parameters is None or previous_sigma2 is None or previous_epsilon is None:
                raise RuntimeError(f"GARCH-t fitting failed for {portfolio_returns.name} before any successful fit: {error}") from error

            logger.warning(
                "GARCH-t refit failed for %s at %s. Reusing previous fitted parameters. Error: %s",
                portfolio_returns.name,
                portfolio_returns.index[first_position].date(),
                error,
            )
            mu = previous_fit_parameters["mu"]
            omega = previous_fit_parameters["omega"]
            alpha = previous_fit_parameters["alpha[1]"]
            beta = previous_fit_parameters["beta[1]"]
            nu = previous_fit_parameters["nu"]
            sigma2_block_start = omega + alpha * previous_epsilon ** 2 + beta * previous_sigma2

        q_alpha = standardized_t_quantile(nu, TAIL_PROBABILITY)
        es_alpha_standardized = standardized_t_expected_shortfall(nu, TAIL_PROBABILITY)

        for block_position_offset, position in enumerate(block_positions):
            position = int(position)
            if block_position_offset == 0:
                sigma2_t = max(sigma2_block_start, 1e-12)
            else:
                sigma2_t = max(omega + alpha * previous_epsilon ** 2 + beta * previous_sigma2, 1e-12)

            sigma_t = float(np.sqrt(sigma2_t))
            rolling_var.iloc[position] = max(-(mu + sigma_t * q_alpha) / 100.0, 0.0)
            rolling_es.iloc[position] = max(-(mu + sigma_t * es_alpha_standardized) / 100.0, 0.0)

            previous_sigma2 = sigma2_t
            previous_epsilon = float(returns_scaled.iloc[position] - mu)

    parameter_history = pd.DataFrame(parameter_history_rows)
    parameter_history["Failed Refits"] = failed_refits
    return rolling_var, rolling_es, parameter_history


def build_garch_t_parameter_summary(parameter_history: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []

    for portfolio_name, portfolio_history in parameter_history.groupby("Portfolio"):
        latest_row = portfolio_history.iloc[-1]
        failed_refits = int(portfolio_history["Failed Refits"].iloc[-1]) if "Failed Refits" in portfolio_history.columns else 0
        summary_rows.append({
            "Portfolio": portfolio_name,
            "Model": MODEL_NAME,
            "Refit Frequency": GARCH_REFIT_FREQUENCY,
            "Number of Successful Fits": len(portfolio_history),
            "Number of Failed Refits": failed_refits,
            "Latest Fit End Date": latest_row["Fit Sample End Date"],
            "Latest mu": latest_row["mu"],
            "Latest omega": latest_row["omega"],
            "Latest alpha[1]": latest_row["alpha[1]"],
            "Latest beta[1]": latest_row["beta[1]"],
            "Latest alpha[1] + beta[1]": latest_row["alpha[1] + beta[1]"],
            "Latest nu": latest_row["nu"],
            "Average mu": portfolio_history["mu"].mean(),
            "Average omega": portfolio_history["omega"].mean(),
            "Average alpha[1]": portfolio_history["alpha[1]"].mean(),
            "Average beta[1]": portfolio_history["beta[1]"].mean(),
            "Average alpha[1] + beta[1]": portfolio_history["alpha[1] + beta[1]"].mean(),
            "Average nu": portfolio_history["nu"].mean(),
            "Stationarity Warning": "alpha[1] + beta[1] >= 1 observed"
            if (portfolio_history["alpha[1] + beta[1]"] >= 1.0).any()
            else "None",
        })

    return pd.DataFrame(summary_rows)


#######################
#
# (3) MAIN
#
#######################
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger.info("INITIALIZING GARCH-t VaR/ES ENGINE...")
    logger.info("Using refit frequency of %s trading days.", GARCH_REFIT_FREQUENCY)

    if arch_model is None:
        raise ModuleNotFoundError("The 'arch' package is required for 05B_garch_t_var_es.py. Install it using: pip install arch")

    portfolio_returns = load_full_portfolio_returns()
    rolling_var = pd.DataFrame(index=portfolio_returns.index, columns=portfolio_returns.columns, dtype=float)
    rolling_es = pd.DataFrame(index=portfolio_returns.index, columns=portfolio_returns.columns, dtype=float)
    parameter_history_frames = []

    for portfolio_name in portfolio_returns.columns:
        portfolio_var, portfolio_es, parameter_history = compute_garch_t_var_es_for_portfolio(portfolio_returns[portfolio_name])
        rolling_var[portfolio_name] = portfolio_var
        rolling_es[portfolio_name] = portfolio_es
        if not parameter_history.empty:
            parameter_history_frames.append(parameter_history)

    validate_var_es_outputs(rolling_var, rolling_es, MODEL_NAME, logger)
    garch_t_risk_metrics = build_model_risk_metrics_table(
        portfolio_returns,
        rolling_var,
        rolling_es,
        model_name=MODEL_NAME,
        confidence_level=CONFIDENCE_LEVEL,
        window_size=WINDOW_SIZE,
    )
    garch_t_risk_metrics = filter_backtest_period(garch_t_risk_metrics, BACKTEST_START, BACKTEST_END)
    validate_backtest_output_dates(garch_t_risk_metrics, BACKTEST_START, MODEL_NAME)

    if not parameter_history_frames:
        raise ValueError("No successful GARCH-t fits were produced.")

    parameter_history = pd.concat(parameter_history_frames, ignore_index=True)
    parameter_summary = build_garch_t_parameter_summary(parameter_history)
    garch_t_backtest_summary = build_model_backtest_summary(
        garch_t_risk_metrics,
        model_name=MODEL_NAME,
        confidence_level=CONFIDENCE_LEVEL,
        window_size=WINDOW_SIZE,
    )
    latest_garch_t_summary = build_latest_model_summary(
        garch_t_risk_metrics,
        confidence_level=CONFIDENCE_LEVEL,
        window_size=WINDOW_SIZE,
    )

    save_dataframe(garch_t_risk_metrics, str(GARCH_T_PORTFOLIO_VAR_ES_FILEPATH))
    save_dataframe(garch_t_backtest_summary, str(GARCH_T_BACKTEST_SUMMARY_FILEPATH))
    save_dataframe(parameter_summary, str(GARCH_T_PARAMETER_SUMMARY_FILEPATH))

    logger.info("GARCH-t parameter summary\n%s", parameter_summary.to_string(index=False))
    logger.info("Latest GARCH-t VaR/ES summary\n%s", latest_garch_t_summary.to_string(index=False))
    logger.info("GARCH-t VaR backtesting summary\n%s", garch_t_backtest_summary.to_string(index=False))
    logger.info("Saved outputs to %s", VOLATILITY_MODEL_OUTPUT_DIR)


#######################
#
# (4) RUN MAIN
#
#######################
if __name__ == "__main__":
    main()
