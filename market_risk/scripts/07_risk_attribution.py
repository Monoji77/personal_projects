#######################
#
# (0) LIBRARIES
#
#######################
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from path_utils import project_path
from risk_engine_utils import (
    build_portfolio_returns_from_frozen_weights,
    combine_price_data,
    compute_simple_returns,
    extract_close_prices,
    load_frozen_weights,
    load_optional_price_data,
    load_price_data,
    safe_filename,
    save_dataframe,
    validate_required_columns,
)

logger = logging.getLogger(__name__)


#######################
#
# (1) GLOBAL VARIABLES
#
#######################
TICKERS = ["AAPL", "GOOG", "NVDA", "MSFT", "AMZN"]
BACKTEST_START = "2025-01-01"
WINDOW_SIZE = 252
ANNUALIZATION_FACTOR = 252
CLOSE_COL = "Close"

FROZEN_WEIGHTS_FILEPATH = project_path("figure", "risk_engine", "frozen_portfolio_weights.csv")
HISTORICAL_ASSET_FILEPATH = project_path("data", "historical", "top_tech_assets_prices.parquet")
NEW_DAILY_ASSET_FILEPATH = project_path("data", "new_daily", "top_tech_assets_prices.parquet")
PORTFOLIO_RISK_METRICS_FILEPATH = project_path("figure", "risk_engine", "portfolio_rolling_var_es.csv")
STRESS_TEST_ASSET_CONTRIBUTIONS_FILEPATH = project_path("figure", "risk_engine", "stress_testing", "stress_test_asset_contributions.csv")
STRESS_TEST_SUMMARY_FILEPATH = project_path("figure", "risk_engine", "stress_testing", "stress_test_summary.csv")

RISK_ATTRIBUTION_OUTPUT_DIR = project_path("figure", "risk_engine", "risk_attribution")
RISK_ATTRIBUTION_PLOTS_DIR = RISK_ATTRIBUTION_OUTPUT_DIR / "plots"

ASSET_VOLATILITY_SUMMARY_FILEPATH = RISK_ATTRIBUTION_OUTPUT_DIR / "asset_volatility_summary.csv"
CORRELATION_MATRIX_FILEPATH = RISK_ATTRIBUTION_OUTPUT_DIR / "correlation_matrix.csv"
PORTFOLIO_RISK_CONTRIBUTIONS_FILEPATH = RISK_ATTRIBUTION_OUTPUT_DIR / "portfolio_risk_contributions.csv"
PORTFOLIO_WEIGHT_VS_RISK_CONTRIBUTION_FILEPATH = RISK_ATTRIBUTION_OUTPUT_DIR / "portfolio_weight_vs_risk_contribution.csv"
DRAWDOWN_ATTRIBUTION_SUMMARY_FILEPATH = RISK_ATTRIBUTION_OUTPUT_DIR / "drawdown_attribution_summary.csv"
RISK_DRIVER_SUMMARY_FILEPATH = RISK_ATTRIBUTION_OUTPUT_DIR / "risk_driver_summary.csv"


#######################
#
# (2) HELPER FUNCTIONS
#
#######################
def load_optional_csv(filepath: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    path = Path(filepath)
    if not path.exists():
        logger.warning("Optional input file not found: %s", filepath)
        return pd.DataFrame()
    logger.info("Loaded input file: %s", filepath)
    return pd.read_csv(filepath, parse_dates=parse_dates)


def save_dataframe_with_index(df: pd.DataFrame, filepath: Path, index_label: str) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    try:
        if filepath.exists():
            filepath.unlink()
        df.to_csv(filepath, index=True, index_label=index_label)
    except PermissionError:
        if filepath.exists():
            logger.warning("Could not overwrite %s because it is currently in use. Keeping the existing file.", filepath)
            return
        raise


def save_figure(fig: plt.Figure, filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", filepath)


def load_asset_returns_and_weights() -> tuple[pd.DataFrame, pd.DataFrame]:
    historical_asset_prices = load_price_data(HISTORICAL_ASSET_FILEPATH)
    new_daily_asset_prices = load_optional_price_data(NEW_DAILY_ASSET_FILEPATH)
    all_asset_prices = combine_price_data(historical_asset_prices, new_daily_asset_prices)
    close_prices = extract_close_prices(all_asset_prices, TICKERS, CLOSE_COL)
    asset_returns = compute_simple_returns(close_prices).dropna(how="any")

    if set(TICKERS) - set(asset_returns.columns):
        raise ValueError("Asset return dataframe does not contain all required tickers.")

    frozen_weights = load_frozen_weights(FROZEN_WEIGHTS_FILEPATH, TICKERS)
    logger.info("Loaded frozen portfolio weights.\n%s", frozen_weights.reset_index().rename(columns={'index': 'Portfolio'}).to_string(index=False))
    return asset_returns.reindex(columns=TICKERS), frozen_weights.reindex(columns=TICKERS)


def get_latest_window_returns(asset_returns: pd.DataFrame) -> pd.DataFrame:
    if len(asset_returns) < WINDOW_SIZE:
        raise ValueError(f"Asset returns contain fewer than {WINDOW_SIZE} observations.")
    latest_window_returns = asset_returns.tail(WINDOW_SIZE).copy()
    if latest_window_returns.isna().any().any():
        raise ValueError("Latest 252-day return window contains NaN values.")
    return latest_window_returns


def get_backtest_asset_returns(asset_returns: pd.DataFrame) -> pd.DataFrame:
    backtest_asset_returns = asset_returns.loc[asset_returns.index >= pd.Timestamp(BACKTEST_START)].copy()
    if backtest_asset_returns.empty:
        raise ValueError("Backtest-period asset returns are unavailable for drawdown attribution.")
    return backtest_asset_returns


def build_asset_volatility_summary(
    latest_window_returns: pd.DataFrame,
    backtest_asset_returns: pd.DataFrame
) -> pd.DataFrame:
    summary_rows = []
    for ticker in TICKERS:
        ticker_returns = latest_window_returns[ticker]
        summary_rows.append({
            "Ticker": ticker,
            "Daily Mean Return": ticker_returns.mean(),
            "Daily Volatility": ticker_returns.std(ddof=1),
            "Annualized Mean Return": ticker_returns.mean() * ANNUALIZATION_FACTOR,
            "Annualized Volatility": ticker_returns.std(ddof=1) * np.sqrt(ANNUALIZATION_FACTOR),
            "Minimum Daily Return": ticker_returns.min(),
            "Maximum Daily Return": ticker_returns.max(),
            "Latest 252D Return": (1.0 + ticker_returns).prod() - 1.0,
            "Backtest Period Volatility": backtest_asset_returns[ticker].std(ddof=1),
        })
    return pd.DataFrame(summary_rows)


def build_correlation_matrix(latest_window_returns: pd.DataFrame) -> pd.DataFrame:
    correlation_matrix = latest_window_returns.corr()
    if correlation_matrix.isna().any().any():
        raise ValueError("Correlation matrix contains NaN values.")
    return correlation_matrix


def build_portfolio_risk_contributions(
    latest_window_returns: pd.DataFrame,
    frozen_weights: pd.DataFrame,
    asset_volatility_summary: pd.DataFrame
) -> pd.DataFrame:
    covariance_matrix = latest_window_returns.cov()
    if covariance_matrix.isna().any().any():
        raise ValueError("Covariance matrix contains NaN values.")

    asset_volatility_map = asset_volatility_summary.set_index("Ticker")
    risk_contribution_rows = []

    for portfolio_name, weight_row in frozen_weights.iterrows():
        weight_vector = weight_row.reindex(TICKERS).to_numpy()
        covariance_array = covariance_matrix.reindex(index=TICKERS, columns=TICKERS).to_numpy()
        portfolio_daily_volatility = float(np.sqrt(weight_vector @ covariance_array @ weight_vector))
        if portfolio_daily_volatility <= 0.0:
            raise ValueError(f"Portfolio daily volatility is not positive for {portfolio_name}.")

        marginal_contribution_to_risk = (covariance_array @ weight_vector) / portfolio_daily_volatility
        component_contribution_to_risk = weight_vector * marginal_contribution_to_risk
        percentage_contribution_to_risk = component_contribution_to_risk / portfolio_daily_volatility

        if not np.isclose(component_contribution_to_risk.sum(), portfolio_daily_volatility, atol=1e-8):
            logger.warning("Component contributions do not sum exactly to portfolio daily volatility for %s.", portfolio_name)
        if not np.isclose(percentage_contribution_to_risk.sum(), 1.0, atol=1e-8):
            logger.warning("Percentage risk contributions do not sum exactly to 1 for %s.", portfolio_name)
        if (percentage_contribution_to_risk < 0.0).any():
            logger.warning("Negative percentage risk contribution detected for %s due to covariance effects.", portfolio_name)

        for ticker_index, ticker in enumerate(TICKERS):
            weight = float(weight_vector[ticker_index])
            percentage_contribution = float(percentage_contribution_to_risk[ticker_index])
            risk_contribution_rows.append({
                "Portfolio": portfolio_name,
                "Ticker": ticker,
                "Weight": weight,
                "Asset Daily Volatility": asset_volatility_map.loc[ticker, "Daily Volatility"],
                "Asset Annualized Volatility": asset_volatility_map.loc[ticker, "Annualized Volatility"],
                "Marginal Contribution to Risk": float(marginal_contribution_to_risk[ticker_index]),
                "Component Contribution to Risk": float(component_contribution_to_risk[ticker_index]),
                "Percentage Contribution to Risk": percentage_contribution,
                "Weight Minus Risk Contribution": weight - percentage_contribution,
                "Risk Contribution Minus Weight": percentage_contribution - weight,
                "Portfolio Daily Volatility": portfolio_daily_volatility,
                "Portfolio Annualized Volatility": portfolio_daily_volatility * np.sqrt(ANNUALIZATION_FACTOR),
            })

    return pd.DataFrame(risk_contribution_rows)


def build_weight_vs_risk_contribution_table(portfolio_risk_contributions: pd.DataFrame) -> pd.DataFrame:
    comparison_df = portfolio_risk_contributions[
        [
            "Portfolio",
            "Ticker",
            "Weight",
            "Percentage Contribution to Risk",
            "Risk Contribution Minus Weight",
        ]
    ].copy()

    def classify_risk_concentration(row: pd.Series) -> str:
        if row["Percentage Contribution to Risk"] > row["Weight"] + 0.10:
            return "High risk contribution relative to weight"
        if row["Percentage Contribution to Risk"] < row["Weight"] - 0.10:
            return "Low risk contribution relative to weight"
        return "Broadly aligned with weight"

    comparison_df["Risk Concentration Flag"] = comparison_df.apply(classify_risk_concentration, axis=1)
    return comparison_df


def compute_portfolio_drawdown_statistics(portfolio_returns: pd.Series) -> tuple[pd.Series, pd.Timestamp, pd.Timestamp, float]:
    wealth = (1.0 + portfolio_returns).cumprod()
    running_peak = wealth.cummax()
    drawdown = wealth / running_peak - 1.0
    trough_date = drawdown.idxmin()
    peak_value = running_peak.loc[trough_date]
    peak_candidates = wealth.loc[:trough_date]
    peak_date = peak_candidates.loc[peak_candidates == peak_value].index[-1]
    maximum_drawdown = float(drawdown.loc[trough_date])
    return drawdown, peak_date, trough_date, maximum_drawdown


def build_drawdown_attribution_summary(
    backtest_asset_returns: pd.DataFrame,
    frozen_weights: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Approximate drawdown attribution using fixed weights and asset cumulative returns
    # over the portfolio drawdown window. This is intended for interpretability, not
    # exact performance attribution under continuous rebalancing.
    backtest_portfolio_returns = build_portfolio_returns_from_frozen_weights(backtest_asset_returns, frozen_weights, TICKERS)
    drawdown_rows = []
    drawdown_time_series = pd.DataFrame(index=backtest_portfolio_returns.index)

    for portfolio_name in backtest_portfolio_returns.columns:
        portfolio_return_series = backtest_portfolio_returns[portfolio_name]
        drawdown_series, peak_date, trough_date, maximum_drawdown = compute_portfolio_drawdown_statistics(portfolio_return_series)
        drawdown_time_series[portfolio_name] = drawdown_series

        asset_drawdown_window_returns = backtest_asset_returns.loc[peak_date:trough_date, TICKERS]
        asset_cumulative_returns = (1.0 + asset_drawdown_window_returns).prod() - 1.0
        weight_row = frozen_weights.loc[portfolio_name, TICKERS]
        weighted_asset_return_contributions = weight_row * asset_cumulative_returns
        asset_drawdown_loss_contributions = -weighted_asset_return_contributions
        total_drawdown_loss_contribution = asset_drawdown_loss_contributions.sum()

        if total_drawdown_loss_contribution > 0.0:
            contribution_percentages = asset_drawdown_loss_contributions / total_drawdown_loss_contribution
        else:
            contribution_percentages = pd.Series(np.nan, index=TICKERS)

        for ticker in TICKERS:
            drawdown_rows.append({
                "Portfolio": portfolio_name,
                "Peak Date": peak_date,
                "Trough Date": trough_date,
                "Maximum Drawdown": maximum_drawdown,
                "Ticker": ticker,
                "Weight": weight_row[ticker],
                "Asset Cumulative Return During Drawdown": asset_cumulative_returns[ticker],
                "Weighted Asset Return Contribution": weighted_asset_return_contributions[ticker],
                "Asset Drawdown Loss Contribution": asset_drawdown_loss_contributions[ticker],
                "Drawdown Contribution Percentage": contribution_percentages[ticker],
            })

    drawdown_attribution_summary = pd.DataFrame(drawdown_rows)
    return drawdown_attribution_summary, drawdown_time_series


def determine_main_stress_contributors(
    stress_test_asset_contributions: pd.DataFrame,
    stress_test_summary: pd.DataFrame
) -> dict[str, str]:
    if not stress_test_summary.empty and "Worst Contributing Asset" in stress_test_summary.columns:
        return dict(zip(stress_test_summary["Portfolio"], stress_test_summary["Worst Contributing Asset"]))

    if stress_test_asset_contributions.empty:
        return {}

    worst_rows = (
        stress_test_asset_contributions
        .sort_values(["Portfolio", "Asset Loss Contribution", "Ticker"], ascending=[True, False, True])
        .drop_duplicates(subset=["Portfolio"])
    )
    return dict(zip(worst_rows["Portfolio"], worst_rows["Ticker"]))


def build_risk_driver_summary(
    frozen_weights: pd.DataFrame,
    asset_volatility_summary: pd.DataFrame,
    portfolio_risk_contributions: pd.DataFrame,
    drawdown_attribution_summary: pd.DataFrame,
    stress_test_asset_contributions: pd.DataFrame,
    stress_test_summary: pd.DataFrame
) -> pd.DataFrame:
    asset_volatility_map = asset_volatility_summary.set_index("Ticker")
    main_stress_contributors = determine_main_stress_contributors(stress_test_asset_contributions, stress_test_summary)
    summary_rows = []

    for portfolio_name in frozen_weights.index:
        weight_row = frozen_weights.loc[portfolio_name, TICKERS]
        held_assets = weight_row.loc[weight_row > 0.0].index.tolist()
        portfolio_risk_subset = portfolio_risk_contributions.loc[portfolio_risk_contributions["Portfolio"] == portfolio_name].copy()
        portfolio_drawdown_subset = drawdown_attribution_summary.loc[drawdown_attribution_summary["Portfolio"] == portfolio_name].copy()

        highest_weight_asset = weight_row.idxmax()
        highest_volatility_asset_held = (
            asset_volatility_map.loc[held_assets, "Annualized Volatility"].idxmax()
            if held_assets else np.nan
        )
        highest_risk_contribution_row = portfolio_risk_subset.sort_values(
            ["Percentage Contribution to Risk", "Ticker"],
            ascending=[False, True],
        ).iloc[0]
        largest_risk_minus_weight_row = portfolio_risk_subset.sort_values(
            ["Risk Contribution Minus Weight", "Ticker"],
            ascending=[False, True],
        ).iloc[0]
        drawdown_contributor_row = portfolio_drawdown_subset.sort_values(
            ["Asset Drawdown Loss Contribution", "Ticker"],
            ascending=[False, True],
        ).iloc[0]
        maximum_drawdown = float(portfolio_drawdown_subset["Maximum Drawdown"].iloc[0])

        highest_risk_asset = highest_risk_contribution_row["Ticker"]
        highest_risk_pct = float(highest_risk_contribution_row["Percentage Contribution to Risk"])
        main_drawdown_contributor = drawdown_contributor_row["Ticker"]
        main_stress_contributor = main_stress_contributors.get(portfolio_name, np.nan)

        if highest_risk_asset == "NVDA" and highest_risk_pct > 0.35:
            overall_risk_comment = "Risk is concentrated in NVDA due to high volatility and large risk contribution."
        elif "Global Minimum Variance" in portfolio_name:
            overall_risk_comment = "Portfolio has defensive allocation but remains exposed to correlated technology drawdowns."
        elif highest_risk_pct > 0.45:
            overall_risk_comment = "Tangency Portfolio has high concentration risk because one asset dominates volatility contribution."
        else:
            overall_risk_comment = "Risk is relatively diversified across assets."

        summary_rows.append({
            "Portfolio": portfolio_name,
            "Highest Weight Asset": highest_weight_asset,
            "Highest Volatility Asset Held": highest_volatility_asset_held,
            "Highest Risk Contribution Asset": highest_risk_asset,
            "Highest Risk Contribution Percentage": highest_risk_pct,
            "Asset With Largest Risk Contribution Minus Weight": largest_risk_minus_weight_row["Ticker"],
            "Maximum Drawdown": maximum_drawdown,
            "Main Drawdown Contributor": main_drawdown_contributor,
            "Main Stress Loss Contributor": main_stress_contributor,
            "Overall Risk Comment": overall_risk_comment,
        })

    return pd.DataFrame(summary_rows)


def plot_asset_annualized_volatility(asset_volatility_summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(asset_volatility_summary["Ticker"], asset_volatility_summary["Annualized Volatility"], color="#1f77b4")
    ax.set_title("Asset Annualized Volatility")
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Annualized Volatility")
    ax.grid(True, axis="y", alpha=0.3)
    save_figure(fig, RISK_ATTRIBUTION_PLOTS_DIR / "asset_annualized_volatility.png")


def plot_asset_correlation_heatmap(correlation_matrix: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(correlation_matrix.to_numpy(), cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_title("Asset Correlation Heatmap")
    ax.set_xticks(np.arange(len(correlation_matrix.columns)))
    ax.set_xticklabels(correlation_matrix.columns)
    ax.set_yticks(np.arange(len(correlation_matrix.index)))
    ax.set_yticklabels(correlation_matrix.index)

    for row_index in range(correlation_matrix.shape[0]):
        for col_index in range(correlation_matrix.shape[1]):
            ax.text(col_index, row_index, f"{correlation_matrix.iloc[row_index, col_index]:.2f}", ha="center", va="center", color="black")

    fig.colorbar(image, ax=ax, label="Correlation")
    save_figure(fig, RISK_ATTRIBUTION_PLOTS_DIR / "asset_correlation_heatmap.png")


def plot_risk_contribution_by_portfolio(portfolio_risk_contributions: pd.DataFrame) -> None:
    portfolio_order = portfolio_risk_contributions["Portfolio"].drop_duplicates().tolist()
    x_positions = np.arange(len(TICKERS))
    bar_width = 0.25
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for portfolio_index, portfolio_name in enumerate(portfolio_order):
        contribution_values = (
            portfolio_risk_contributions
            .loc[portfolio_risk_contributions["Portfolio"] == portfolio_name]
            .set_index("Ticker")
            .reindex(TICKERS)["Percentage Contribution to Risk"]
            .to_numpy()
        )
        ax.bar(
            x_positions + (portfolio_index - 1) * bar_width,
            contribution_values,
            width=bar_width,
            label=portfolio_name,
            color=colors[portfolio_index % len(colors)],
        )

    ax.set_title("Percentage Contribution to Risk by Portfolio")
    ax.set_xlabel("Ticker")
    ax.set_ylabel("Percentage Contribution to Risk")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(TICKERS)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    save_figure(fig, RISK_ATTRIBUTION_PLOTS_DIR / "risk_contribution_by_portfolio.png")


def plot_weight_vs_risk_contribution(portfolio_weight_vs_risk: pd.DataFrame) -> None:
    for portfolio_name, portfolio_subset in portfolio_weight_vs_risk.groupby("Portfolio"):
        portfolio_subset = portfolio_subset.set_index("Ticker").reindex(TICKERS).reset_index()
        x_positions = np.arange(len(TICKERS))
        bar_width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x_positions - bar_width / 2, portfolio_subset["Weight"], width=bar_width, label="Weight", color="#1f77b4")
        ax.bar(
            x_positions + bar_width / 2,
            portfolio_subset["Percentage Contribution to Risk"],
            width=bar_width,
            label="Percentage Contribution to Risk",
            color="#ff7f0e",
        )
        ax.set_title(f"Weight vs Risk Contribution: {portfolio_name}")
        ax.set_xlabel("Ticker")
        ax.set_ylabel("Proportion")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(TICKERS)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        save_figure(fig, RISK_ATTRIBUTION_PLOTS_DIR / f"weight_vs_risk_contribution_{safe_filename(portfolio_name)}.png")


def plot_portfolio_drawdowns(drawdown_time_series: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    for portfolio_name in drawdown_time_series.columns:
        ax.plot(drawdown_time_series.index, drawdown_time_series[portfolio_name], label=portfolio_name, linewidth=1.4)
    ax.set_title("Portfolio Drawdowns from Backtest Start")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="x", rotation=45)
    save_figure(fig, RISK_ATTRIBUTION_PLOTS_DIR / "portfolio_drawdowns.png")


def plot_drawdown_contributions(drawdown_attribution_summary: pd.DataFrame) -> None:
    for portfolio_name, portfolio_subset in drawdown_attribution_summary.groupby("Portfolio"):
        portfolio_subset = portfolio_subset.set_index("Ticker").reindex(TICKERS).reset_index()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(portfolio_subset["Ticker"], portfolio_subset["Asset Drawdown Loss Contribution"], color="#d62728")
        ax.set_title(f"Drawdown Loss Contributions: {portfolio_name}")
        ax.set_xlabel("Ticker")
        ax.set_ylabel("Asset Drawdown Loss Contribution")
        ax.grid(True, axis="y", alpha=0.3)
        save_figure(fig, RISK_ATTRIBUTION_PLOTS_DIR / f"drawdown_contributions_{safe_filename(portfolio_name)}.png")


def plot_risk_driver_summary(risk_driver_summary: pd.DataFrame) -> None:
    display_columns = [
        "Portfolio",
        "Highest Weight Asset",
        "Highest Risk Contribution Asset",
        "Main Drawdown Contributor",
        "Main Stress Loss Contributor",
    ]
    display_df = risk_driver_summary[display_columns].fillna("N/A")

    fig, ax = plt.subplots(figsize=(14, 3 + len(display_df) * 0.5))
    ax.axis("off")
    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    ax.set_title("Risk Driver Summary")
    save_figure(fig, RISK_ATTRIBUTION_PLOTS_DIR / "risk_driver_summary.png")


#######################
#
# (3) MAIN
#
#######################
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    RISK_ATTRIBUTION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RISK_ATTRIBUTION_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("INITIALIZING RISK ATTRIBUTION...")

    asset_returns, frozen_weights = load_asset_returns_and_weights()
    latest_window_returns = get_latest_window_returns(asset_returns)
    backtest_asset_returns = get_backtest_asset_returns(asset_returns)
    stress_test_asset_contributions = load_optional_csv(STRESS_TEST_ASSET_CONTRIBUTIONS_FILEPATH)
    stress_test_summary = load_optional_csv(STRESS_TEST_SUMMARY_FILEPATH)
    stress_files_included = not stress_test_asset_contributions.empty

    latest_window_start = latest_window_returns.index.min().date()
    latest_window_end = latest_window_returns.index.max().date()
    logger.info("Latest 252-day window used for covariance and volatility attribution: %s to %s", latest_window_start, latest_window_end)

    asset_volatility_summary = build_asset_volatility_summary(latest_window_returns, backtest_asset_returns)
    correlation_matrix = build_correlation_matrix(latest_window_returns)
    portfolio_risk_contributions = build_portfolio_risk_contributions(latest_window_returns, frozen_weights, asset_volatility_summary)
    portfolio_weight_vs_risk_contribution = build_weight_vs_risk_contribution_table(portfolio_risk_contributions)
    drawdown_attribution_summary, drawdown_time_series = build_drawdown_attribution_summary(backtest_asset_returns, frozen_weights)
    risk_driver_summary = build_risk_driver_summary(
        frozen_weights,
        asset_volatility_summary,
        portfolio_risk_contributions,
        drawdown_attribution_summary,
        stress_test_asset_contributions,
        stress_test_summary,
    )

    save_dataframe(asset_volatility_summary, str(ASSET_VOLATILITY_SUMMARY_FILEPATH))
    save_dataframe_with_index(correlation_matrix, CORRELATION_MATRIX_FILEPATH, index_label="Ticker")
    save_dataframe(portfolio_risk_contributions, str(PORTFOLIO_RISK_CONTRIBUTIONS_FILEPATH))
    save_dataframe(portfolio_weight_vs_risk_contribution, str(PORTFOLIO_WEIGHT_VS_RISK_CONTRIBUTION_FILEPATH))
    save_dataframe(drawdown_attribution_summary, str(DRAWDOWN_ATTRIBUTION_SUMMARY_FILEPATH))
    save_dataframe(risk_driver_summary, str(RISK_DRIVER_SUMMARY_FILEPATH))

    plot_asset_annualized_volatility(asset_volatility_summary)
    plot_asset_correlation_heatmap(correlation_matrix)
    plot_risk_contribution_by_portfolio(portfolio_risk_contributions)
    plot_weight_vs_risk_contribution(portfolio_weight_vs_risk_contribution)
    plot_portfolio_drawdowns(drawdown_time_series)
    plot_drawdown_contributions(drawdown_attribution_summary)
    plot_risk_driver_summary(risk_driver_summary)

    for _, row in risk_driver_summary.iterrows():
        logger.info(
            "%s: highest risk contribution asset = %s; maximum drawdown = %.4f; main drawdown contributor = %s.",
            row["Portfolio"],
            row["Highest Risk Contribution Asset"],
            row["Maximum Drawdown"],
            row["Main Drawdown Contributor"],
        )

    logger.info("Stress contribution files included: %s", "Yes" if stress_files_included else "No")
    logger.info("Saved CSV outputs to %s", RISK_ATTRIBUTION_OUTPUT_DIR)
    logger.info("Saved plot outputs to %s", RISK_ATTRIBUTION_PLOTS_DIR)


#######################
#
# (4) RUN MAIN
#
#######################
if __name__ == "__main__":
    main()
