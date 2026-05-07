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
    identify_metric_column,
    safe_filename,
    save_dataframe,
    validate_required_columns,
    validate_required_file,
)

logger = logging.getLogger(__name__)


#######################
#
# (1) GLOBAL VARIABLES
#
#######################
CONFIDENCE_LEVEL = 0.95
MODEL_ORDER = ["Historical VaR", "EWMA-t", "GARCH-t"]
HISTORICAL_MODEL_NAME = "Historical VaR"
EWMA_MODEL_NAME = "EWMA-t"
GARCH_MODEL_NAME = "GARCH-t"
DATE_COL = "Date"

HISTORICAL_RISK_METRICS_FILEPATH = project_path("figure", "risk_engine", "portfolio_rolling_var_es.csv")
HISTORICAL_KUPIEC_RESULTS_FILEPATH = project_path("figure", "risk_engine", "portfolio_var_backtesting_tests.csv")
EWMA_RISK_METRICS_FILEPATH = project_path("figure", "risk_engine", "volatility_models", "ewma_t_portfolio_var_es.csv")
EWMA_BACKTEST_SUMMARY_FILEPATH = project_path("figure", "risk_engine", "volatility_models", "ewma_t_backtest_summary.csv")
GARCH_RISK_METRICS_FILEPATH = project_path("figure", "risk_engine", "volatility_models", "garch_t_portfolio_var_es.csv")
GARCH_BACKTEST_SUMMARY_FILEPATH = project_path("figure", "risk_engine", "volatility_models", "garch_t_backtest_summary.csv")
EWMA_PARAMETER_SUMMARY_FILEPATH = project_path("figure", "risk_engine", "volatility_models", "ewma_t_parameter_summary.csv")
GARCH_PARAMETER_SUMMARY_FILEPATH = project_path("figure", "risk_engine", "volatility_models", "garch_t_parameter_summary.csv")

VOLATILITY_MODEL_OUTPUT_DIR = project_path("figure", "risk_engine", "volatility_models")
COMBINED_PORTFOLIO_VAR_ES_FILEPATH = VOLATILITY_MODEL_OUTPUT_DIR / "combined_portfolio_var_es.csv"
COMBINED_BACKTEST_SUMMARY_FILEPATH = VOLATILITY_MODEL_OUTPUT_DIR / "combined_volatility_model_backtest_summary.csv"
MODEL_RANKINGS_FILEPATH = VOLATILITY_MODEL_OUTPUT_DIR / "volatility_model_rankings.csv"
PLOTS_OUTPUT_DIR = VOLATILITY_MODEL_OUTPUT_DIR / "plots"


#######################
#
# (2) HELPER FUNCTIONS
#
#######################
def load_required_csv(filepath: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    validate_required_file(filepath)
    logger.info("Loaded input file: %s", filepath)
    return pd.read_csv(filepath, parse_dates=parse_dates)


def build_historical_backtest_summary(
    historical_risk_metrics: pd.DataFrame,
    historical_kupiec_results: pd.DataFrame
) -> pd.DataFrame:
    rolling_var_col = identify_metric_column(historical_risk_metrics, "Rolling VaR")
    rolling_es_col = identify_metric_column(historical_risk_metrics, "Rolling ES")
    validate_required_columns(
        historical_kupiec_results,
        [
            "Portfolio",
            "Confidence Level",
            "Tail Probability",
            "Actual VaR Breaches",
            "Expected VaR Breaches",
            "Breach Rate",
            "Expected Breach Rate",
            "Kupiec LR Statistic",
            "Kupiec p-value",
            "Test Decision",
            "Interpretation",
        ],
        "Historical Kupiec results",
    )

    summary_rows = []
    historical_kupiec_map = historical_kupiec_results.set_index("Portfolio")

    for portfolio_name, portfolio_metrics in historical_risk_metrics.groupby("Portfolio"):
        valid_metrics = portfolio_metrics.loc[portfolio_metrics[rolling_var_col].notna()].copy()
        kupiec_row = historical_kupiec_map.loc[portfolio_name]
        breach_day_returns = valid_metrics.loc[valid_metrics["VaR Breach"].fillna(False), "Return"]

        summary_rows.append({
            "Model": HISTORICAL_MODEL_NAME,
            "Portfolio": portfolio_name,
            "Confidence Level": kupiec_row["Confidence Level"],
            "Tail Probability": kupiec_row["Tail Probability"],
            "Number of Observations": len(valid_metrics),
            "Actual VaR Breaches": kupiec_row["Actual VaR Breaches"],
            "Expected VaR Breaches": kupiec_row["Expected VaR Breaches"],
            "Breach Rate": kupiec_row["Breach Rate"],
            "Expected Breach Rate": kupiec_row["Expected Breach Rate"],
            "Average Return on Breach Days": breach_day_returns.mean(),
            "Worst Return": valid_metrics["Return"].min() if not valid_metrics.empty else np.nan,
            "Average VaR": valid_metrics[rolling_var_col].mean(),
            "Average ES": valid_metrics[rolling_es_col].mean(),
            "Kupiec LR Statistic": kupiec_row["Kupiec LR Statistic"],
            "Kupiec p-value": kupiec_row["Kupiec p-value"],
            "Test Decision": kupiec_row["Test Decision"],
            "Interpretation": kupiec_row["Interpretation"],
        })

    return pd.DataFrame(summary_rows)


def prepare_combined_portfolio_var_es(
    historical_risk_metrics: pd.DataFrame,
    ewma_risk_metrics: pd.DataFrame,
    garch_risk_metrics: pd.DataFrame
) -> pd.DataFrame:
    historical_metrics = historical_risk_metrics.copy()
    historical_metrics["Model"] = HISTORICAL_MODEL_NAME
    combined_metrics = pd.concat(
        [historical_metrics, ewma_risk_metrics.copy(), garch_risk_metrics.copy()],
        ignore_index=True,
    )
    combined_metrics["Model"] = pd.Categorical(combined_metrics["Model"], categories=MODEL_ORDER, ordered=True)
    return combined_metrics.sort_values(["Portfolio", DATE_COL, "Model"]).reset_index(drop=True)


def prepare_combined_backtest_summary(
    historical_backtest_summary: pd.DataFrame,
    ewma_backtest_summary: pd.DataFrame,
    garch_backtest_summary: pd.DataFrame
) -> pd.DataFrame:
    combined_summary = pd.concat(
        [historical_backtest_summary.copy(), ewma_backtest_summary.copy(), garch_backtest_summary.copy()],
        ignore_index=True,
    )
    combined_summary["Model"] = pd.Categorical(combined_summary["Model"], categories=MODEL_ORDER, ordered=True)
    return combined_summary.sort_values(["Portfolio", "Model"]).reset_index(drop=True)


def save_figure(fig: plt.Figure, filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", filepath)


def plot_breach_rate_comparison(combined_summary: pd.DataFrame) -> None:
    for portfolio_name, portfolio_summary in combined_summary.groupby("Portfolio"):
        portfolio_summary = portfolio_summary.sort_values("Model")
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(portfolio_summary["Model"].astype(str), portfolio_summary["Breach Rate"], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        ax.axhline(
            portfolio_summary["Expected Breach Rate"].iloc[0],
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="Expected Breach Rate",
        )
        ax.set_title(f"Breach Rate Comparison: {portfolio_name}")
        ax.set_xlabel("Model")
        ax.set_ylabel("Breach Rate")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        filepath = PLOTS_OUTPUT_DIR / f"model_breach_rate_comparison_{safe_filename(portfolio_name)}.png"
        save_figure(fig, filepath)


def plot_kupiec_pvalue_comparison(combined_summary: pd.DataFrame) -> None:
    for portfolio_name, portfolio_summary in combined_summary.groupby("Portfolio"):
        portfolio_summary = portfolio_summary.sort_values("Model")
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(portfolio_summary["Model"].astype(str), portfolio_summary["Kupiec p-value"], color=["#1f77b4", "#ff7f0e", "#2ca02c"])
        ax.axhline(0.05, color="black", linestyle="--", linewidth=1.2, label="0.05 Threshold")
        ax.set_title(f"Kupiec p-value Comparison: {portfolio_name}")
        ax.set_xlabel("Model")
        ax.set_ylabel("Kupiec p-value")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        filepath = PLOTS_OUTPUT_DIR / f"kupiec_pvalue_comparison_{safe_filename(portfolio_name)}.png"
        save_figure(fig, filepath)


def plot_average_var_es_comparison(combined_summary: pd.DataFrame) -> None:
    for portfolio_name, portfolio_summary in combined_summary.groupby("Portfolio"):
        portfolio_summary = portfolio_summary.sort_values("Model")
        x_positions = np.arange(len(portfolio_summary))
        bar_width = 0.35

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x_positions - bar_width / 2, portfolio_summary["Average VaR"], width=bar_width, label="Average VaR", color="#ff7f0e")
        ax.bar(x_positions + bar_width / 2, portfolio_summary["Average ES"], width=bar_width, label="Average ES", color="#2ca02c")
        ax.set_title(f"Average VaR and ES Comparison: {portfolio_name}")
        ax.set_xlabel("Model")
        ax.set_ylabel("Positive Loss Estimate")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(portfolio_summary["Model"].astype(str))
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        filepath = PLOTS_OUTPUT_DIR / f"average_var_es_comparison_{safe_filename(portfolio_name)}.png"
        save_figure(fig, filepath)


def plot_var_thresholds_by_model(combined_risk_metrics: pd.DataFrame) -> None:
    rolling_var_col = identify_metric_column(combined_risk_metrics, "Rolling VaR")

    for portfolio_name, portfolio_metrics in combined_risk_metrics.groupby("Portfolio"):
        portfolio_metrics = portfolio_metrics.sort_values([DATE_COL, "Model"])
        actual_returns = (
            portfolio_metrics
            .drop_duplicates(subset=[DATE_COL])
            .sort_values(DATE_COL)
        )
        var_thresholds = portfolio_metrics.pivot(index=DATE_COL, columns="Model", values=rolling_var_col)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(actual_returns[DATE_COL], actual_returns["Return"], label="Actual Return", color="black", linewidth=1.0)

        for model_name, color in zip(MODEL_ORDER, ["#1f77b4", "#ff7f0e", "#2ca02c"]):
            if model_name in var_thresholds.columns:
                ax.plot(var_thresholds.index, -var_thresholds[model_name], label=f"-{model_name} VaR", linewidth=1.2, color=color)

        ax.set_title(f"Actual Returns vs VaR Thresholds by Model: {portfolio_name}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Return / Negative VaR Threshold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45)
        filepath = PLOTS_OUTPUT_DIR / f"var_thresholds_by_model_{safe_filename(portfolio_name)}.png"
        save_figure(fig, filepath)


def plot_volatility_model_parameter_comparison(
    ewma_parameter_summary: pd.DataFrame,
    garch_parameter_summary: pd.DataFrame
) -> None:
    portfolio_order = ewma_parameter_summary["Portfolio"].tolist()
    x_positions = np.arange(len(portfolio_order))
    bar_width = 0.35

    merged_parameters = ewma_parameter_summary.merge(
        garch_parameter_summary,
        on="Portfolio",
        how="inner",
        suffixes=("_EWMA", "_GARCH"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].bar(x_positions - bar_width / 2, merged_parameters["Estimated nu"], width=bar_width, label="EWMA-t nu", color="#ff7f0e")
    axes[0].bar(x_positions + bar_width / 2, merged_parameters["Average nu"], width=bar_width, label="GARCH-t Average nu", color="#2ca02c")
    axes[0].set_title("Student-t Degrees of Freedom by Portfolio")
    axes[0].set_xlabel("Portfolio")
    axes[0].set_ylabel("Estimated nu")
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels(portfolio_order, rotation=20, ha="right")
    axes[0].legend()
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(x_positions, merged_parameters["Average alpha[1] + beta[1]"], color="#1f77b4")
    axes[1].set_title("GARCH-t Average alpha[1] + beta[1]")
    axes[1].set_xlabel("Portfolio")
    axes[1].set_ylabel("Average alpha[1] + beta[1]")
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(portfolio_order, rotation=20, ha="right")
    axes[1].grid(True, axis="y", alpha=0.3)

    save_figure(fig, PLOTS_OUTPUT_DIR / "volatility_model_parameter_comparison.png")


def build_model_comment(row: pd.Series) -> str:
    if row["Test Decision"] != "Do not reject VaR model":
        return "Rejected by Kupiec test."
    if row["Absolute Breach Rate Error"] < 0.005:
        return "Well calibrated and statistically acceptable."
    if row["Average ES"] == row["Average ES Group Minimum"]:
        return "Statistically acceptable with relatively low capital intensity."
    return "Statistically acceptable but not the closest to target breach frequency."


def build_model_rankings(combined_summary: pd.DataFrame) -> pd.DataFrame:
    ranking_rows = []

    for portfolio_name, portfolio_summary in combined_summary.groupby("Portfolio"):
        portfolio_summary = portfolio_summary.copy()
        portfolio_summary["Absolute Breach Rate Error"] = (portfolio_summary["Breach Rate"] - portfolio_summary["Expected Breach Rate"]).abs()
        portfolio_summary["Average ES Group Minimum"] = portfolio_summary["Average ES"].min()
        non_rejected_mask = portfolio_summary["Test Decision"] == "Do not reject VaR model"

        if non_rejected_mask.any():
            portfolio_summary["Rejected Sort Key"] = (~non_rejected_mask).astype(int)
        else:
            portfolio_summary["Rejected Sort Key"] = 0

        ranked_summary = portfolio_summary.sort_values(
            ["Rejected Sort Key", "Absolute Breach Rate Error", "Average ES"],
            ascending=[True, True, True],
        ).reset_index(drop=True)

        for rank, (_, row) in enumerate(ranked_summary.iterrows(), start=1):
            ranking_rows.append({
                "Portfolio": portfolio_name,
                "Rank": rank,
                "Model": row["Model"],
                "Breach Rate": row["Breach Rate"],
                "Expected Breach Rate": row["Expected Breach Rate"],
                "Absolute Breach Rate Error": row["Absolute Breach Rate Error"],
                "Kupiec p-value": row["Kupiec p-value"],
                "Test Decision": row["Test Decision"],
                "Average VaR": row["Average VaR"],
                "Average ES": row["Average ES"],
                "Model Comment": build_model_comment(row),
            })

    return pd.DataFrame(ranking_rows)


def improved_over(benchmark_row: pd.Series, candidate_row: pd.Series) -> str:
    benchmark_non_rejected = benchmark_row["Test Decision"] == "Do not reject VaR model"
    candidate_non_rejected = candidate_row["Test Decision"] == "Do not reject VaR model"
    benchmark_error = abs(benchmark_row["Breach Rate"] - benchmark_row["Expected Breach Rate"])
    candidate_error = abs(candidate_row["Breach Rate"] - candidate_row["Expected Breach Rate"])

    if candidate_non_rejected and not benchmark_non_rejected:
        return "Yes"
    if candidate_non_rejected == benchmark_non_rejected and candidate_error < benchmark_error:
        return "Yes"
    return "No"


def print_portfolio_conclusions(combined_summary: pd.DataFrame, rankings: pd.DataFrame) -> None:
    for portfolio_name, portfolio_summary in combined_summary.groupby("Portfolio"):
        ranking_subset = rankings.loc[rankings["Portfolio"] == portfolio_name].sort_values("Rank")
        best_row = ranking_subset.iloc[0]
        most_conservative_row = portfolio_summary.sort_values(["Breach Rate", "Average ES"], ascending=[True, False]).iloc[0]
        most_optimistic_row = portfolio_summary.sort_values(["Breach Rate", "Average ES"], ascending=[False, True]).iloc[0]
        historical_row = portfolio_summary.loc[portfolio_summary["Model"] == HISTORICAL_MODEL_NAME].iloc[0]
        ewma_row = portfolio_summary.loc[portfolio_summary["Model"] == EWMA_MODEL_NAME].iloc[0]
        garch_row = portfolio_summary.loc[portfolio_summary["Model"] == GARCH_MODEL_NAME].iloc[0]

        ewma_improved = improved_over(historical_row, ewma_row)
        garch_improved = improved_over(ewma_row, garch_row)
        heavy_tail_useful = "Yes" if ewma_improved == "Yes" or garch_improved == "Yes" else "No clear evidence"

        logger.info(
            "%s conclusion: best breach-calibrated model = %s; most conservative model = %s; most optimistic model = %s; EWMA-t improved over historical = %s; GARCH-t improved over EWMA-t = %s; Student-t heavy-tail modelling useful = %s.",
            portfolio_name,
            best_row["Model"],
            most_conservative_row["Model"],
            most_optimistic_row["Model"],
            ewma_improved,
            garch_improved,
            heavy_tail_useful,
        )


#######################
#
# (3) MAIN
#
#######################
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger.info("INITIALIZING VOLATILITY MODEL COMPARISON...")
    PLOTS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    historical_risk_metrics = load_required_csv(HISTORICAL_RISK_METRICS_FILEPATH, parse_dates=[DATE_COL])
    historical_kupiec_results = load_required_csv(HISTORICAL_KUPIEC_RESULTS_FILEPATH)
    ewma_risk_metrics = load_required_csv(EWMA_RISK_METRICS_FILEPATH, parse_dates=[DATE_COL])
    ewma_backtest_summary = load_required_csv(EWMA_BACKTEST_SUMMARY_FILEPATH)
    garch_risk_metrics = load_required_csv(GARCH_RISK_METRICS_FILEPATH, parse_dates=[DATE_COL])
    garch_backtest_summary = load_required_csv(GARCH_BACKTEST_SUMMARY_FILEPATH)
    ewma_parameter_summary = load_required_csv(EWMA_PARAMETER_SUMMARY_FILEPATH)
    garch_parameter_summary = load_required_csv(GARCH_PARAMETER_SUMMARY_FILEPATH)

    validate_required_columns(historical_risk_metrics, [DATE_COL, "Portfolio", "Return", "VaR Breach"], "Historical risk metrics")
    validate_required_columns(ewma_risk_metrics, [DATE_COL, "Portfolio", "Return", "Model", "VaR Breach"], "EWMA risk metrics")
    validate_required_columns(garch_risk_metrics, [DATE_COL, "Portfolio", "Return", "Model", "VaR Breach"], "GARCH risk metrics")
    validate_required_columns(ewma_parameter_summary, ["Portfolio", "Estimated nu"], "EWMA parameter summary")
    validate_required_columns(garch_parameter_summary, ["Portfolio", "Average nu", "Average alpha[1] + beta[1]"], "GARCH parameter summary")

    historical_backtest_summary = build_historical_backtest_summary(historical_risk_metrics, historical_kupiec_results)
    combined_risk_metrics = prepare_combined_portfolio_var_es(historical_risk_metrics, ewma_risk_metrics, garch_risk_metrics)
    combined_backtest_summary = prepare_combined_backtest_summary(
        historical_backtest_summary,
        ewma_backtest_summary,
        garch_backtest_summary,
    )
    rankings = build_model_rankings(combined_backtest_summary)

    save_dataframe(combined_risk_metrics, str(COMBINED_PORTFOLIO_VAR_ES_FILEPATH))
    save_dataframe(combined_backtest_summary, str(COMBINED_BACKTEST_SUMMARY_FILEPATH))
    save_dataframe(rankings, str(MODEL_RANKINGS_FILEPATH))

    plot_breach_rate_comparison(combined_backtest_summary)
    plot_kupiec_pvalue_comparison(combined_backtest_summary)
    plot_average_var_es_comparison(combined_backtest_summary)
    plot_var_thresholds_by_model(combined_risk_metrics)
    plot_volatility_model_parameter_comparison(ewma_parameter_summary, garch_parameter_summary)

    print_portfolio_conclusions(combined_backtest_summary, rankings)
    logger.info("Saved outputs to %s", VOLATILITY_MODEL_OUTPUT_DIR)


#######################
#
# (4) RUN MAIN
#
#######################
if __name__ == "__main__":
    main()
