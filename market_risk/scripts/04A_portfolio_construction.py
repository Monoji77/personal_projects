#######################
#
# (0) LIBRARIES
#
#######################
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from path_utils import project_path

logger = logging.getLogger(__name__)


#######################
#
# (1) GLOBAL VARIABLES
#
#######################
TICKERS = ["AAPL", "GOOG", "NVDA", "MSFT", "AMZN"]
TRAIN_START = "2016-01-01"
TRAIN_END = "2025-01-01"
CLOSE_COL = "Close"
RF_RATE_DAILY_COL = "rf_rate_daily"
TRADING_DAYS_PER_YEAR = 252

ASSET_PRICE_FILEPATH = project_path("data", "historical", "top_tech_assets_prices.parquet")
RF_RATE_FILEPATH = project_path("data", "historical", "rf_rate_daily.parquet")

RISK_ENGINE_OUTPUT_DIR = project_path("figure", "risk_engine")
FROZEN_WEIGHTS_LONG_FILEPATH = RISK_ENGINE_OUTPUT_DIR / "frozen_portfolio_weights.csv"
FROZEN_WEIGHTS_WIDE_FILEPATH = RISK_ENGINE_OUTPUT_DIR / "frozen_portfolio_weights_wide.csv"
PORTFOLIO_CONSTRUCTION_SUMMARY_FILEPATH = RISK_ENGINE_OUTPUT_DIR / "portfolio_construction_summary.csv"

EQUAL_WEIGHT_PORTFOLIO = "Equal Weighted Portfolio"
GLOBAL_MINIMUM_VARIANCE_PORTFOLIO = "Long-only Global Minimum Variance Portfolio"
TANGENCY_PORTFOLIO = "Long-only Tangency Portfolio"


#######################
#
# (2) HELPER FUNCTIONS
#
#######################
def standardize_daily_index(df: pd.DataFrame) -> pd.DataFrame:
    standardized_df = df.copy()
    standardized_index = pd.to_datetime(standardized_df.index)
    if standardized_index.tz is not None:
        standardized_index = standardized_index.tz_convert("UTC").tz_localize(None)
    standardized_df.index = standardized_index.normalize()
    return standardized_df.sort_index()


def load_dataframe(filepath: str) -> pd.DataFrame:
    return standardize_daily_index(pd.read_parquet(filepath))


def extract_close_prices(price_data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(price_data.columns, pd.MultiIndex):
        raise ValueError("Expected a MultiIndex price dataframe with a 'Close' level.")
    close_prices = price_data[CLOSE_COL].reindex(columns=TICKERS)
    return close_prices.dropna(how="all")


def compute_simple_returns(price_data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return price_data.pct_change(fill_method=None).dropna(how="all")


def filter_training_period(asset_returns: pd.DataFrame) -> pd.DataFrame:
    train_start = pd.Timestamp(TRAIN_START)
    train_end = pd.Timestamp(TRAIN_END)
    training_returns = asset_returns.loc[(asset_returns.index >= train_start) & (asset_returns.index < train_end)].dropna(how="any")

    if training_returns.empty:
        raise ValueError("Training-period returns are empty after filtering.")
    if (training_returns.index >= train_end).any():
        raise ValueError("Training returns contain dates on or after TRAIN_END.")

    return training_returns


def align_risk_free_rate(
    effective_daily_rf_rate: pd.DataFrame,
    portfolio_index: pd.Index
) -> pd.Series:
    if RF_RATE_DAILY_COL not in effective_daily_rf_rate.columns:
        raise KeyError(f"Expected daily risk-free rate column '{RF_RATE_DAILY_COL}'.")
    rf_rate_daily = effective_daily_rf_rate[RF_RATE_DAILY_COL]
    return rf_rate_daily.reindex(portfolio_index).ffill().bfill()


def compute_equal_weights(num_assets: int) -> np.ndarray:
    return np.ones(num_assets) / num_assets


def clean_weights(weights: np.ndarray, tolerance: float = 1e-10) -> np.ndarray:
    cleaned_weights = np.where(np.abs(weights) < tolerance, 0.0, weights)
    cleaned_weights = np.clip(cleaned_weights, 0.0, 1.0)
    weight_sum = cleaned_weights.sum()

    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        raise ValueError("Portfolio weights cannot be normalized to a valid sum.")

    cleaned_weights = cleaned_weights / weight_sum
    if not np.isclose(cleaned_weights.sum(), 1.0, atol=1e-8):
        raise ValueError("Portfolio weights do not sum to 1 after cleaning.")
    if np.isnan(cleaned_weights).any():
        raise ValueError("Portfolio weights contain NaN values after cleaning.")

    return cleaned_weights


def build_long_only_constraints(num_assets: int) -> tuple[list[tuple[float, float]], dict]:
    bounds = [(0.0, 1.0)] * num_assets
    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}
    return bounds, constraints


def optimize_long_only_weights(objective_function, num_assets: int) -> np.ndarray:
    initial_weights = compute_equal_weights(num_assets)
    bounds, constraints = build_long_only_constraints(num_assets)
    optimization_result = minimize(
        objective_function,
        x0=initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 1000,
            "ftol": 1e-12,
        },
    )

    if not optimization_result.success:
        raise ValueError(f"Portfolio optimization failed: {optimization_result.message}")

    return clean_weights(optimization_result.x)


def compute_global_minimum_variance_weights(asset_covariances: pd.DataFrame) -> np.ndarray:
    covariance_matrix = asset_covariances.to_numpy()

    def portfolio_variance(weights: np.ndarray) -> float:
        return float(weights @ covariance_matrix @ weights)

    return optimize_long_only_weights(portfolio_variance, asset_covariances.shape[0])


def compute_tangency_weights(
    asset_returns: pd.DataFrame,
    asset_covariances: pd.DataFrame,
    effective_daily_rf_rate: pd.DataFrame
) -> np.ndarray:
    rf_rate_daily = align_risk_free_rate(effective_daily_rf_rate, asset_returns.index)
    mean_excess_returns = asset_returns.sub(rf_rate_daily, axis=0).mean().to_numpy()
    covariance_matrix = asset_covariances.to_numpy()

    def negative_daily_sharpe_ratio(weights: np.ndarray) -> float:
        portfolio_excess_return = float(weights @ mean_excess_returns)
        portfolio_volatility = float(np.sqrt(weights @ covariance_matrix @ weights))
        if np.isclose(portfolio_volatility, 0.0):
            return np.inf
        return -portfolio_excess_return / portfolio_volatility

    return optimize_long_only_weights(negative_daily_sharpe_ratio, asset_covariances.shape[0])


def validate_weights(weights: np.ndarray, portfolio_name: str) -> None:
    if np.isnan(weights).any():
        raise ValueError(f"{portfolio_name} contains NaN weights.")
    if (weights < 0.0).any() or (weights > 1.0).any():
        raise ValueError(f"{portfolio_name} contains weights outside [0, 1].")
    if not np.isclose(weights.sum(), 1.0, atol=1e-8):
        raise ValueError(f"{portfolio_name} weights do not sum to 1.")


def build_frozen_portfolio_weights(
    training_returns: pd.DataFrame,
    effective_daily_rf_rate: pd.DataFrame
) -> dict[str, np.ndarray]:
    asset_covariances = training_returns.cov()
    num_assets = training_returns.shape[1]

    portfolio_weights = {
        EQUAL_WEIGHT_PORTFOLIO: clean_weights(compute_equal_weights(num_assets)),
        GLOBAL_MINIMUM_VARIANCE_PORTFOLIO: compute_global_minimum_variance_weights(asset_covariances),
        TANGENCY_PORTFOLIO: compute_tangency_weights(training_returns, asset_covariances, effective_daily_rf_rate),
    }

    for portfolio_name, weights in portfolio_weights.items():
        validate_weights(weights, portfolio_name)

    return portfolio_weights


def compute_portfolio_returns(asset_returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    return asset_returns @ weights


def compute_annualized_return(portfolio_returns: pd.Series) -> float:
    compounded_growth = (1.0 + portfolio_returns).prod()
    num_periods = len(portfolio_returns)
    return compounded_growth ** (TRADING_DAYS_PER_YEAR / num_periods) - 1.0


def compute_annualized_volatility(portfolio_returns: pd.Series) -> float:
    return portfolio_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)


def compute_annualized_sharpe_ratio(
    portfolio_returns: pd.Series,
    effective_daily_rf_rate: pd.Series
) -> float:
    excess_returns = portfolio_returns - effective_daily_rf_rate
    daily_sharpe_ratio = excess_returns.mean() / excess_returns.std(ddof=1)
    return daily_sharpe_ratio * np.sqrt(TRADING_DAYS_PER_YEAR)


def build_weights_long_table(portfolio_weights: dict[str, np.ndarray]) -> pd.DataFrame:
    weight_rows = []
    for portfolio_name, weights in portfolio_weights.items():
        for ticker, weight in zip(TICKERS, weights):
            weight_rows.append({
                "Portfolio": portfolio_name,
                "Ticker": ticker,
                "Weight": weight,
            })
    return pd.DataFrame(weight_rows, columns=["Portfolio", "Ticker", "Weight"])


def build_weights_wide_table(portfolio_weights: dict[str, np.ndarray]) -> pd.DataFrame:
    return pd.DataFrame.from_dict(portfolio_weights, orient="index", columns=TICKERS)


def build_portfolio_construction_summary(
    portfolio_weights: dict[str, np.ndarray],
    training_returns: pd.DataFrame,
    aligned_training_rf_rate: pd.Series
) -> pd.DataFrame:
    summary_rows = []
    num_training_observations = len(training_returns)

    for portfolio_name, weights in portfolio_weights.items():
        portfolio_returns = compute_portfolio_returns(training_returns, weights)
        summary_rows.append({
            "Portfolio": portfolio_name,
            "Training Start": TRAIN_START,
            "Training End": TRAIN_END,
            "Number of Training Observations": num_training_observations,
            "In-Sample Annualized Return": compute_annualized_return(portfolio_returns),
            "In-Sample Annualized Volatility": compute_annualized_volatility(portfolio_returns),
            "In-Sample Sharpe Ratio": compute_annualized_sharpe_ratio(portfolio_returns, aligned_training_rf_rate),
            "Largest Weight": weights.max(),
            "Smallest Weight": weights.min(),
            "Number of Nonzero Weights": int(np.count_nonzero(weights > 0.0)),
        })

    return pd.DataFrame(summary_rows)


def save_dataframe(df: pd.DataFrame, filepath: str, include_index: bool = False, index_label: str | None = None) -> None:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=include_index, index_label=index_label)


def log_final_weights(portfolio_weights: dict[str, np.ndarray]) -> None:
    logger.info("Final frozen portfolio weights:")
    for portfolio_name, weights in portfolio_weights.items():
        portfolio_weight_table = pd.DataFrame({
            "Ticker": TICKERS,
            "Weight": weights,
        })
        logger.info("%s\n%s", portfolio_name, portfolio_weight_table.to_string(index=False))


#######################
#
# (3) MAIN
#
#######################
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger.info("INITIALIZING PORTFOLIO CONSTRUCTION...")
    logger.info("Training period: %s <= date < %s", TRAIN_START, TRAIN_END)
    logger.info("Using daily risk-free rate column: %s", RF_RATE_DAILY_COL)

    asset_prices = load_dataframe(ASSET_PRICE_FILEPATH)
    close_prices = extract_close_prices(asset_prices)
    asset_returns = compute_simple_returns(close_prices).dropna(how="any")
    training_returns = filter_training_period(asset_returns)

    effective_daily_rf_rate = load_dataframe(RF_RATE_FILEPATH)
    aligned_training_rf_rate = align_risk_free_rate(effective_daily_rf_rate, training_returns.index)

    frozen_portfolio_weights = build_frozen_portfolio_weights(training_returns, effective_daily_rf_rate)
    log_final_weights(frozen_portfolio_weights)

    frozen_weights_long = build_weights_long_table(frozen_portfolio_weights)
    frozen_weights_wide = build_weights_wide_table(frozen_portfolio_weights)
    portfolio_construction_summary = build_portfolio_construction_summary(
        frozen_portfolio_weights,
        training_returns,
        aligned_training_rf_rate,
    )

    save_dataframe(frozen_weights_long, FROZEN_WEIGHTS_LONG_FILEPATH)
    save_dataframe(frozen_weights_wide, FROZEN_WEIGHTS_WIDE_FILEPATH, include_index=True, index_label="Portfolio")
    save_dataframe(portfolio_construction_summary, PORTFOLIO_CONSTRUCTION_SUMMARY_FILEPATH)

    logger.info("Saved frozen weights to %s and %s", FROZEN_WEIGHTS_LONG_FILEPATH, FROZEN_WEIGHTS_WIDE_FILEPATH)
    logger.info("Saved portfolio construction summary to %s", PORTFOLIO_CONSTRUCTION_SUMMARY_FILEPATH)


#######################
#
# (4) RUN MAIN
#
#######################
if __name__ == "__main__":
    main()
