#######################
#
# (0) LIBRARIES
#
#######################
import logging
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from path_utils import project_path

logger = logging.getLogger(__name__)


#######################
#
# (1) GLOBAL VARIABLES
#
#######################
TICKERS = ["AAPL", "GOOG", "NVDA", "MSFT", "AMZN"]
BACKTEST_START = "2025-01-01"
CLOSE_COL = "Close"
DATE_COL = "Date"

FROZEN_WEIGHTS_FILEPATH = project_path("figure", "risk_engine", "frozen_portfolio_weights.csv")
PORTFOLIO_RISK_METRICS_FILEPATH = project_path("figure", "risk_engine", "portfolio_rolling_var_es.csv")
ASSET_RISK_METRICS_FILEPATH = project_path("figure", "risk_engine", "asset_rolling_var_es.csv")
PORTFOLIO_BACKTEST_SUMMARY_FILEPATH = project_path("figure", "risk_engine", "portfolio_var_backtest_summary.csv")
HISTORICAL_ASSET_FILEPATH = project_path("data", "historical", "top_tech_assets_prices.parquet")
NEW_DAILY_ASSET_FILEPATH = project_path("data", "new_daily", "top_tech_assets_prices.parquet")

PLOTS_OUTPUT_DIR = project_path("figure", "risk_engine", "plots")
FROZEN_WEIGHTS_PLOT_FILEPATH = PLOTS_OUTPUT_DIR / "frozen_portfolio_weights.png"
PORTFOLIO_CUMULATIVE_RETURNS_PLOT_FILEPATH = PLOTS_OUTPUT_DIR / "portfolio_cumulative_returns_backtest.png"
PORTFOLIO_VAR_BREACH_COMPARISON_PLOT_FILEPATH = PLOTS_OUTPUT_DIR / "portfolio_var_breach_comparison.png"

PORTFOLIO_FILENAME_MAP = {
    "Equal Weighted Portfolio": "equal_weighted_portfolio",
    "Long-only Global Minimum Variance Portfolio": "global_minimum_variance_portfolio",
    "Long-only Tangency Portfolio": "tangency_portfolio",
}


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


def validate_required_file(filepath: str) -> None:
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Required input file not found: {filepath}")


def validate_required_columns(df: pd.DataFrame, required_columns: list[str], dataframe_name: str) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{dataframe_name} is missing required columns: {missing_columns}")


def load_required_csv(filepath: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    validate_required_file(filepath)
    logger.info("Loaded input file: %s", filepath)
    return pd.read_csv(filepath, parse_dates=parse_dates)


def load_required_parquet(filepath: str) -> pd.DataFrame:
    validate_required_file(filepath)
    logger.info("Loaded input file: %s", filepath)
    return standardize_daily_index(pd.read_parquet(filepath))


def load_optional_parquet(filepath: str) -> pd.DataFrame:
    if not Path(filepath).exists():
        logger.warning("Optional input file not found: %s", filepath)
        return pd.DataFrame()
    logger.info("Loaded input file: %s", filepath)
    return standardize_daily_index(pd.read_parquet(filepath))


def extract_close_prices(price_data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(price_data.columns, pd.MultiIndex):
        raise ValueError("Expected a MultiIndex asset price dataframe with a 'Close' level.")
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


def load_frozen_weights(filepath: str) -> pd.DataFrame:
    frozen_weights_long = load_required_csv(filepath)
    validate_required_columns(frozen_weights_long, ["Portfolio", "Ticker", "Weight"], "Frozen weights")
    frozen_weights_wide = (
        frozen_weights_long
        .pivot(index="Portfolio", columns="Ticker", values="Weight")
        .reindex(columns=TICKERS)
    )
    if frozen_weights_wide.isna().any().any():
        raise ValueError("Frozen weights contain missing values after pivoting to wide format.")
    return frozen_weights_wide


def build_portfolio_returns(asset_returns: pd.DataFrame, frozen_weights: pd.DataFrame) -> pd.DataFrame:
    aligned_asset_returns = asset_returns.reindex(columns=TICKERS)
    portfolio_returns = {
        portfolio_name: aligned_asset_returns @ weight_row.to_numpy()
        for portfolio_name, weight_row in frozen_weights.iterrows()
    }
    return pd.DataFrame(portfolio_returns, index=aligned_asset_returns.index)


def filter_backtest_period(df: pd.DataFrame, date_column: str | None = None) -> pd.DataFrame:
    backtest_start = pd.Timestamp(BACKTEST_START)
    if date_column is None:
        return df.loc[df.index >= backtest_start].copy()
    return df.loc[df[date_column] >= backtest_start].copy()


def identify_metric_column(df: pd.DataFrame, metric_prefix: str) -> str:
    matching_columns = [column for column in df.columns if column.startswith(metric_prefix)]
    if len(matching_columns) != 1:
        raise ValueError(f"Expected exactly one column starting with '{metric_prefix}', found {matching_columns}")
    return matching_columns[0]


def safe_portfolio_filename(portfolio_name: str) -> str:
    if portfolio_name in PORTFOLIO_FILENAME_MAP:
        return PORTFOLIO_FILENAME_MAP[portfolio_name]
    sanitized_name = re.sub(r"[^a-z0-9]+", "_", portfolio_name.lower()).strip("_")
    return sanitized_name


def save_figure(fig: plt.Figure, filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", filepath)


def plot_frozen_portfolio_weights(frozen_weights: pd.DataFrame) -> None:
    portfolio_names = list(frozen_weights.index)
    x_positions = np.arange(len(TICKERS))
    bar_width = 0.24
    color_map = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for portfolio_index, portfolio_name in enumerate(portfolio_names):
        ax.bar(
            x_positions + (portfolio_index - 1) * bar_width,
            frozen_weights.loc[portfolio_name, TICKERS].to_numpy(),
            width=bar_width,
            label=portfolio_name,
            color=color_map[portfolio_index % len(color_map)],
        )

    ax.set_title("Frozen Portfolio Weights")
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Weight")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(TICKERS)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    save_figure(fig, FROZEN_WEIGHTS_PLOT_FILEPATH)


def plot_portfolio_cumulative_returns(portfolio_returns: pd.DataFrame) -> None:
    cumulative_returns = (1.0 + portfolio_returns).cumprod() - 1.0
    fig, ax = plt.subplots(figsize=(12, 6))

    for portfolio_name in cumulative_returns.columns:
        ax.plot(cumulative_returns.index, cumulative_returns[portfolio_name], label=portfolio_name, linewidth=1.5)

    ax.set_title("Portfolio Cumulative Returns During Backtest")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=45)
    save_figure(fig, PORTFOLIO_CUMULATIVE_RETURNS_PLOT_FILEPATH)


def plot_var_breaches_by_portfolio(portfolio_risk_metrics: pd.DataFrame, rolling_var_col: str) -> None:
    for portfolio_name, portfolio_df in portfolio_risk_metrics.groupby("Portfolio"):
        portfolio_df = portfolio_df.sort_values(DATE_COL)
        breach_mask = portfolio_df["VaR Breach"].fillna(False)
        breach_points = portfolio_df.loc[breach_mask]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(portfolio_df[DATE_COL], portfolio_df["Return"], label="Actual Daily Return", linewidth=1.0, color="#1f77b4")
        ax.plot(portfolio_df[DATE_COL], -portfolio_df[rolling_var_col], label="Negative Rolling VaR Threshold", linewidth=1.5, color="#d62728")
        if not breach_points.empty:
            ax.scatter(
                breach_points[DATE_COL],
                breach_points["Return"],
                label="VaR Breach",
                color="#9467bd",
                s=28,
                zorder=3,
            )

        ax.set_title(f"Actual Returns vs Rolling VaR: {portfolio_name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Return")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45)
        filepath = PLOTS_OUTPUT_DIR / f"var_breaches_{safe_portfolio_filename(portfolio_name)}.png"
        save_figure(fig, filepath)


def plot_rolling_var_es_by_portfolio(
    portfolio_risk_metrics: pd.DataFrame,
    rolling_var_col: str,
    rolling_es_col: str
) -> None:
    for portfolio_name, portfolio_df in portfolio_risk_metrics.groupby("Portfolio"):
        portfolio_df = portfolio_df.sort_values(DATE_COL)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(portfolio_df[DATE_COL], portfolio_df[rolling_var_col], label="Rolling VaR", linewidth=1.5, color="#ff7f0e")
        ax.plot(portfolio_df[DATE_COL], portfolio_df[rolling_es_col], label="Rolling ES", linewidth=1.5, color="#2ca02c")

        ax.set_title(f"Rolling VaR and ES: {portfolio_name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Positive Loss Estimate")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45)
        filepath = PLOTS_OUTPUT_DIR / f"rolling_var_es_{safe_portfolio_filename(portfolio_name)}.png"
        save_figure(fig, filepath)


def plot_var_breach_comparison(portfolio_backtest_summary: pd.DataFrame) -> None:
    validate_required_columns(
        portfolio_backtest_summary,
        ["Portfolio", "Actual VaR Breaches", "Expected VaR Breaches"],
        "Portfolio backtest summary",
    )

    x_positions = np.arange(len(portfolio_backtest_summary))
    bar_width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(
        x_positions - bar_width / 2,
        portfolio_backtest_summary["Actual VaR Breaches"].to_numpy(),
        width=bar_width,
        label="Actual VaR Breaches",
        color="#1f77b4",
    )
    ax.bar(
        x_positions + bar_width / 2,
        portfolio_backtest_summary["Expected VaR Breaches"].to_numpy(),
        width=bar_width,
        label="Expected VaR Breaches",
        color="#ff7f0e",
    )

    ax.set_title("Actual vs Expected VaR Breaches by Portfolio")
    ax.set_xlabel("Portfolio")
    ax.set_ylabel("Number of Breaches")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(portfolio_backtest_summary["Portfolio"], rotation=20, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    save_figure(fig, PORTFOLIO_VAR_BREACH_COMPARISON_PLOT_FILEPATH)


#######################
#
# (3) MAIN
#
#######################
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger.info("INITIALIZING RISK ENGINE VISUALIZATIONS...")
    PLOTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frozen_weights = load_frozen_weights(FROZEN_WEIGHTS_FILEPATH)
    portfolio_risk_metrics = load_required_csv(PORTFOLIO_RISK_METRICS_FILEPATH, parse_dates=[DATE_COL])
    asset_risk_metrics = load_required_csv(ASSET_RISK_METRICS_FILEPATH, parse_dates=[DATE_COL])
    portfolio_backtest_summary = load_required_csv(PORTFOLIO_BACKTEST_SUMMARY_FILEPATH)
    historical_asset_prices = load_required_parquet(HISTORICAL_ASSET_FILEPATH)
    new_daily_asset_prices = load_optional_parquet(NEW_DAILY_ASSET_FILEPATH)

    validate_required_columns(
        portfolio_risk_metrics,
        [DATE_COL, "Portfolio", "Return", "VaR Breach"],
        "Portfolio risk metrics",
    )
    validate_required_columns(
        asset_risk_metrics,
        [DATE_COL, "Asset", "Return", "VaR Breach"],
        "Asset risk metrics",
    )

    rolling_var_col = identify_metric_column(portfolio_risk_metrics, "Rolling VaR")
    rolling_es_col = identify_metric_column(portfolio_risk_metrics, "Rolling ES")

    all_asset_prices = combine_price_data(historical_asset_prices, new_daily_asset_prices)
    close_prices = extract_close_prices(all_asset_prices)
    asset_returns = compute_simple_returns(close_prices).dropna(how="any")
    portfolio_returns = build_portfolio_returns(asset_returns, frozen_weights)
    portfolio_returns = filter_backtest_period(portfolio_returns)

    if portfolio_returns.empty:
        raise ValueError("No portfolio returns are available in the backtest period for plotting.")

    plot_frozen_portfolio_weights(frozen_weights)
    plot_portfolio_cumulative_returns(portfolio_returns)
    plot_var_breaches_by_portfolio(portfolio_risk_metrics, rolling_var_col)
    plot_rolling_var_es_by_portfolio(portfolio_risk_metrics, rolling_var_col, rolling_es_col)
    plot_var_breach_comparison(portfolio_backtest_summary)


#######################
#
# (4) RUN MAIN
#
#######################
if __name__ == "__main__":
    main()
