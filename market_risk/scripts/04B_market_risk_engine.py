#######################
#
# (0) LIBRARIES
#
#######################
import datetime as dt
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from path_utils import project_path

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
CLOSE_COL = "Close"
DATE_COL = "Date"

HISTORICAL_ASSET_FILEPATH = project_path("data", "historical", "top_tech_assets_prices.parquet")
NEW_DAILY_ASSET_FILEPATH = project_path("data", "new_daily", "top_tech_assets_prices.parquet")
FROZEN_WEIGHTS_FILEPATH = project_path("figure", "risk_engine", "frozen_portfolio_weights.csv")

RISK_ENGINE_OUTPUT_DIR = project_path("figure", "risk_engine")
ASSET_RISK_METRICS_FILEPATH = RISK_ENGINE_OUTPUT_DIR / "asset_rolling_var_es.csv"
PORTFOLIO_RISK_METRICS_FILEPATH = RISK_ENGINE_OUTPUT_DIR / "portfolio_rolling_var_es.csv"
LATEST_RISK_SUMMARY_FILEPATH = RISK_ENGINE_OUTPUT_DIR / "latest_rolling_var_es_summary.csv"
LATEST_PORTFOLIO_RISK_SUMMARY_FILEPATH = RISK_ENGINE_OUTPUT_DIR / "latest_portfolio_rolling_var_es_summary.csv"
ASSET_VAR_BACKTEST_SUMMARY_FILEPATH = RISK_ENGINE_OUTPUT_DIR / "asset_var_backtest_summary.csv"
PORTFOLIO_VAR_BACKTEST_SUMMARY_FILEPATH = RISK_ENGINE_OUTPUT_DIR / "portfolio_var_backtest_summary.csv"

ROLLING_VAR_COL = f"Rolling VaR ({CONFIDENCE_LEVEL:.0%}, {WINDOW_SIZE}D)"
ROLLING_ES_COL = f"Rolling ES ({CONFIDENCE_LEVEL:.0%}, {WINDOW_SIZE}D)"


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


def load_optional_dataframe(filepath: str) -> pd.DataFrame:
    if not Path(filepath).exists():
        return pd.DataFrame()
    return load_dataframe(filepath)


def extract_close_prices(price_data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(price_data.columns, pd.MultiIndex):
        raise ValueError("Expected a MultiIndex price dataframe with a 'Close' level.")
    close_prices = price_data[CLOSE_COL].reindex(columns=TICKERS)
    return close_prices.dropna(how="all")


def compute_simple_returns(price_data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    return price_data.pct_change(fill_method=None).dropna(how="all")


def combine_price_data(base_prices: pd.DataFrame, additional_prices: pd.DataFrame) -> pd.DataFrame:
    if additional_prices.empty:
        return base_prices.sort_index()
    aligned_additional_prices = additional_prices.reindex(columns=base_prices.columns)
    combined_prices = pd.concat([base_prices, aligned_additional_prices], axis=0)
    combined_prices = combined_prices[~combined_prices.index.duplicated(keep="last")]
    return combined_prices.sort_index()


def get_download_window(current_prices: pd.DataFrame) -> tuple[dt.date, dt.date]:
    start_date = current_prices.index.max().date() + dt.timedelta(days=1)
    end_date = dt.datetime.now().date() + dt.timedelta(days=1)
    return start_date, end_date


def download_latest_asset_prices(
    tickers: list[str],
    start_date: dt.date,
    end_date: dt.date
) -> pd.DataFrame:
    if start_date >= end_date:
        logger.info("Historical and cached new-daily data are already current through %s.", start_date - dt.timedelta(days=1))
        return pd.DataFrame()

    try:
        latest_prices = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
    except Exception as error:
        logger.warning("Latest data download failed. Continuing with cached data only. Error: %s", error)
        return pd.DataFrame()

    if latest_prices.empty:
        logger.info("No new asset prices found between %s and %s.", start_date, end_date)
        return pd.DataFrame()

    return standardize_daily_index(latest_prices)


def save_new_daily_data(new_daily_prices: pd.DataFrame, filepath: str) -> None:
    if new_daily_prices.empty:
        return
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    new_daily_prices.to_parquet(filepath)


def load_frozen_weights(filepath: str) -> pd.DataFrame:
    if not Path(filepath).exists():
        raise FileNotFoundError("Frozen portfolio weights not found. Run 04A_portfolio_construction.py first.")

    frozen_weights_long = pd.read_csv(filepath)
    required_columns = {"Portfolio", "Ticker", "Weight"}
    if not required_columns.issubset(frozen_weights_long.columns):
        raise ValueError("Frozen portfolio weights file does not contain the required columns.")

    frozen_weights_wide = (
        frozen_weights_long
        .pivot(index="Portfolio", columns="Ticker", values="Weight")
        .reindex(columns=TICKERS)
        .sort_index()
    )
    validate_frozen_weights(frozen_weights_wide)
    return frozen_weights_wide


def validate_frozen_weights(frozen_weights: pd.DataFrame) -> None:
    if frozen_weights.isna().any().any():
        raise ValueError("Frozen portfolio weights contain NaN values.")

    weight_sums = frozen_weights.sum(axis=1)
    if not np.allclose(weight_sums.to_numpy(), 1.0, atol=1e-8):
        raise ValueError("Frozen portfolio weights do not sum to 1.")

    if ((frozen_weights < 0.0) | (frozen_weights > 1.0)).any().any():
        raise ValueError("Frozen portfolio weights must lie between 0 and 1.")


def apply_frozen_weights(asset_returns: pd.DataFrame, frozen_weights: pd.DataFrame) -> pd.DataFrame:
    portfolio_returns = {}
    aligned_asset_returns = asset_returns.reindex(columns=TICKERS)
    for portfolio_name, weight_row in frozen_weights.iterrows():
        portfolio_returns[portfolio_name] = aligned_asset_returns @ weight_row.to_numpy()
    return pd.DataFrame(portfolio_returns, index=aligned_asset_returns.index)


def compute_rolling_historical_var(
    returns: pd.Series | pd.DataFrame,
    confidence_level: float = CONFIDENCE_LEVEL,
    window: int = WINDOW_SIZE
) -> pd.Series | pd.DataFrame:
    shifted_returns = returns.shift(1)
    tail_probability = 1.0 - confidence_level
    return -shifted_returns.rolling(window=window, min_periods=window).quantile(tail_probability)


def compute_historical_es_from_window(
    window_returns: np.ndarray,
    confidence_level: float = CONFIDENCE_LEVEL
) -> float:
    clean_window_returns = window_returns[~np.isnan(window_returns)]
    if len(clean_window_returns) == 0:
        return np.nan

    tail_probability = 1.0 - confidence_level
    var_threshold = np.quantile(clean_window_returns, tail_probability)
    tail_returns = clean_window_returns[clean_window_returns <= var_threshold]
    return -tail_returns.mean()


def compute_rolling_historical_es(
    returns: pd.Series | pd.DataFrame,
    confidence_level: float = CONFIDENCE_LEVEL,
    window: int = WINDOW_SIZE
) -> pd.Series | pd.DataFrame:
    shifted_returns = returns.shift(1)
    return shifted_returns.rolling(window=window, min_periods=window).apply(
        lambda window_returns: compute_historical_es_from_window(window_returns, confidence_level),
        raw=True,
    )


def validate_risk_estimates(
    rolling_var: pd.Series | pd.DataFrame,
    rolling_es: pd.Series | pd.DataFrame,
    entity_label: str
) -> None:
    valid_var = rolling_var.stack().dropna()
    valid_es = rolling_es.stack().dropna()

    if (valid_var < 0.0).any():
        raise ValueError(f"{entity_label} rolling VaR contains negative values.")
    if (valid_es < 0.0).any():
        raise ValueError(f"{entity_label} rolling ES contains negative values.")

    es_less_than_var_count = (rolling_es < rolling_var).stack().dropna().sum()
    if es_less_than_var_count > 0:
        logger.warning(
            "%s rolling ES is below rolling VaR on %s observations. Review the output if this persists.",
            entity_label,
            int(es_less_than_var_count),
        )


def build_risk_metrics_table(
    returns: pd.DataFrame,
    rolling_var: pd.DataFrame,
    rolling_es: pd.DataFrame,
    entity_column: str
) -> pd.DataFrame:
    risk_metric_rows = []

    for entity_name in returns.columns:
        entity_returns = returns[entity_name]
        entity_var = rolling_var[entity_name]
        entity_es = rolling_es[entity_name]
        var_breach = (entity_returns < -entity_var).where(entity_var.notna())
        loss_exceeds_es = (entity_returns < -entity_es).where(entity_es.notna())

        risk_metric_rows.append(pd.DataFrame({
            DATE_COL: entity_returns.index,
            entity_column: entity_name,
            "Return": entity_returns.to_numpy(),
            ROLLING_VAR_COL: entity_var.to_numpy(),
            ROLLING_ES_COL: entity_es.to_numpy(),
            "VaR Breach": var_breach.to_numpy(),
            "Loss Exceeds ES": loss_exceeds_es.to_numpy(),
        }))

    return pd.concat(risk_metric_rows, ignore_index=True)


def filter_backtest_period(risk_metrics: pd.DataFrame) -> pd.DataFrame:
    backtest_start = pd.Timestamp(BACKTEST_START)
    filtered_risk_metrics = risk_metrics.loc[risk_metrics[DATE_COL] >= backtest_start].copy()
    if BACKTEST_END is not None:
        backtest_end = pd.Timestamp(BACKTEST_END)
        filtered_risk_metrics = filtered_risk_metrics.loc[filtered_risk_metrics[DATE_COL] < backtest_end].copy()
    return filtered_risk_metrics


def validate_backtest_output_dates(risk_metrics: pd.DataFrame, entity_label: str) -> None:
    if risk_metrics.empty:
        raise ValueError(f"{entity_label} backtest output is empty after filtering.")

    if risk_metrics[DATE_COL].min() < pd.Timestamp(BACKTEST_START):
        raise ValueError(f"{entity_label} backtest output contains dates earlier than BACKTEST_START.")


def build_latest_risk_summary(risk_metrics: pd.DataFrame, entity_column: str) -> pd.DataFrame:
    latest_date = risk_metrics[DATE_COL].max()
    summary = risk_metrics.loc[risk_metrics[DATE_COL] == latest_date, [DATE_COL, entity_column, ROLLING_VAR_COL, ROLLING_ES_COL]].copy()
    summary.insert(1, "Entity Type", entity_column)
    summary = summary.rename(columns={entity_column: "Name"})
    return summary.reset_index(drop=True)


def build_var_backtest_summary(
    risk_metrics: pd.DataFrame,
    entity_column: str,
    confidence_level: float = CONFIDENCE_LEVEL
) -> pd.DataFrame:
    tail_probability = 1.0 - confidence_level
    summary_rows = []

    for entity_name, entity_metrics in risk_metrics.groupby(entity_column):
        valid_metrics = entity_metrics.loc[entity_metrics[ROLLING_VAR_COL].notna()].copy()
        num_observations = len(valid_metrics)
        actual_breaches = int(valid_metrics["VaR Breach"].fillna(False).sum())
        expected_breaches = tail_probability * num_observations
        breach_rate = actual_breaches / num_observations if num_observations > 0 else np.nan
        expected_breach_rate = tail_probability

        if num_observations == 0:
            interpretation = "No observations"
        elif breach_rate < expected_breach_rate * 0.5:
            interpretation = "Too conservative"
        elif breach_rate > expected_breach_rate * 1.5:
            interpretation = "Too optimistic"
        else:
            interpretation = "Reasonable"

        breach_day_returns = valid_metrics.loc[valid_metrics["VaR Breach"].fillna(False), "Return"]
        summary_rows.append({
            entity_column: entity_name,
            "Number of Backtest Observations": num_observations,
            "Actual VaR Breaches": actual_breaches,
            "Expected VaR Breaches": expected_breaches,
            "Breach Rate": breach_rate,
            "Expected Breach Rate": expected_breach_rate,
            "Average Return on Breach Days": breach_day_returns.mean(),
            "Worst Return": valid_metrics["Return"].min() if num_observations > 0 else np.nan,
            "Average VaR": valid_metrics[ROLLING_VAR_COL].mean(),
            "Average ES": valid_metrics[ROLLING_ES_COL].mean(),
            "Interpretation": interpretation,
        })

    return pd.DataFrame(summary_rows)


def save_dataframe(df: pd.DataFrame, filepath: str) -> None:
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if output_path.exists():
            output_path.unlink()
        df.to_csv(output_path, index=False)
    except PermissionError:
        if output_path.exists():
            logger.warning("Could not overwrite %s because it is currently in use. Keeping the existing file.", output_path)
            return
        raise


def print_table(title: str, df: pd.DataFrame) -> None:
    logger.info("%s\n%s", title, df.to_string(index=False))


#######################
#
# (3) MAIN
#
#######################
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger.info("INITIALIZING MARKET RISK ENGINE...")
    logger.info("Backtest period used: %s to %s", BACKTEST_START, BACKTEST_END or "latest available date")

    historical_prices = load_dataframe(HISTORICAL_ASSET_FILEPATH)
    cached_new_daily_prices = load_optional_dataframe(NEW_DAILY_ASSET_FILEPATH)
    current_prices = combine_price_data(historical_prices, cached_new_daily_prices)

    start_date, end_date = get_download_window(current_prices)
    latest_downloaded_prices = download_latest_asset_prices(TICKERS, start_date, end_date)

    updated_new_daily_prices = combine_price_data(cached_new_daily_prices, latest_downloaded_prices) if not cached_new_daily_prices.empty else latest_downloaded_prices
    save_new_daily_data(updated_new_daily_prices, NEW_DAILY_ASSET_FILEPATH)

    all_asset_prices = combine_price_data(current_prices, latest_downloaded_prices)
    close_prices = extract_close_prices(all_asset_prices)
    asset_returns = compute_simple_returns(close_prices).dropna(how="any")

    frozen_weights = load_frozen_weights(FROZEN_WEIGHTS_FILEPATH)
    logger.info("Loaded frozen portfolio weights from %s", FROZEN_WEIGHTS_FILEPATH)
    print_table("Loaded frozen portfolio weights", frozen_weights.reset_index().rename(columns={"index": "Portfolio"}))
    logger.info("Portfolio weights were loaded from disk only. No portfolio optimization is performed in this script.")

    portfolio_returns = apply_frozen_weights(asset_returns, frozen_weights)

    rolling_asset_var = compute_rolling_historical_var(asset_returns)
    rolling_asset_es = compute_rolling_historical_es(asset_returns)
    validate_risk_estimates(rolling_asset_var, rolling_asset_es, "Asset")
    asset_risk_metrics = build_risk_metrics_table(asset_returns, rolling_asset_var, rolling_asset_es, entity_column="Asset")
    asset_risk_metrics = filter_backtest_period(asset_risk_metrics)
    validate_backtest_output_dates(asset_risk_metrics, "Asset")
    save_dataframe(asset_risk_metrics, ASSET_RISK_METRICS_FILEPATH)

    rolling_portfolio_var = compute_rolling_historical_var(portfolio_returns)
    rolling_portfolio_es = compute_rolling_historical_es(portfolio_returns)
    validate_risk_estimates(rolling_portfolio_var, rolling_portfolio_es, "Portfolio")
    portfolio_risk_metrics = build_risk_metrics_table(portfolio_returns, rolling_portfolio_var, rolling_portfolio_es, entity_column="Portfolio")
    portfolio_risk_metrics = filter_backtest_period(portfolio_risk_metrics)
    validate_backtest_output_dates(portfolio_risk_metrics, "Portfolio")
    save_dataframe(portfolio_risk_metrics, PORTFOLIO_RISK_METRICS_FILEPATH)

    latest_asset_summary = build_latest_risk_summary(asset_risk_metrics, entity_column="Asset")
    latest_portfolio_summary = build_latest_risk_summary(portfolio_risk_metrics, entity_column="Portfolio")
    save_dataframe(latest_asset_summary, LATEST_RISK_SUMMARY_FILEPATH)
    save_dataframe(latest_portfolio_summary, LATEST_PORTFOLIO_RISK_SUMMARY_FILEPATH)

    asset_var_backtest_summary = build_var_backtest_summary(asset_risk_metrics, entity_column="Asset")
    portfolio_var_backtest_summary = build_var_backtest_summary(portfolio_risk_metrics, entity_column="Portfolio")
    save_dataframe(asset_var_backtest_summary, ASSET_VAR_BACKTEST_SUMMARY_FILEPATH)
    save_dataframe(portfolio_var_backtest_summary, PORTFOLIO_VAR_BACKTEST_SUMMARY_FILEPATH)

    print_table("Latest asset VaR/ES summary", latest_asset_summary)
    print_table("Latest portfolio VaR/ES summary", latest_portfolio_summary)
    print_table("Portfolio VaR backtesting summary", portfolio_var_backtest_summary)


#######################
#
# (4) RUN MAIN
#
#######################
if __name__ == "__main__":
    main()
