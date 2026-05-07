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
    load_frozen_weights,
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
TICKERS = ["AAPL", "GOOG", "NVDA", "MSFT", "AMZN"]
CLOSE_COL = "Close"
HISTORICAL_MODEL_NAME = "Historical VaR"

FROZEN_WEIGHTS_FILEPATH = project_path("figure", "risk_engine", "frozen_portfolio_weights.csv")
LATEST_PORTFOLIO_RISK_SUMMARY_FILEPATH = project_path("figure", "risk_engine", "latest_portfolio_rolling_var_es_summary.csv")
COMBINED_PORTFOLIO_VAR_ES_FILEPATH = project_path("figure", "risk_engine", "volatility_models", "combined_portfolio_var_es.csv")
VOLATILITY_MODEL_RANKINGS_FILEPATH = project_path("figure", "risk_engine", "volatility_models", "volatility_model_rankings.csv")

STRESS_TESTING_OUTPUT_DIR = project_path("figure", "risk_engine", "stress_testing")
STRESS_TESTING_PLOTS_DIR = STRESS_TESTING_OUTPUT_DIR / "plots"
STRESS_SCENARIO_DEFINITIONS_FILEPATH = STRESS_TESTING_OUTPUT_DIR / "stress_scenario_definitions.csv"
STRESS_TEST_RESULTS_FILEPATH = STRESS_TESTING_OUTPUT_DIR / "stress_test_results.csv"
STRESS_TEST_ASSET_CONTRIBUTIONS_FILEPATH = STRESS_TESTING_OUTPUT_DIR / "stress_test_asset_contributions.csv"
STRESS_TEST_SUMMARY_FILEPATH = STRESS_TESTING_OUTPUT_DIR / "stress_test_summary.csv"


#######################
#
# (2) HELPER FUNCTIONS
#
#######################
def load_required_csv(filepath: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    validate_required_file(filepath)
    logger.info("Loaded input file: %s", filepath)
    return pd.read_csv(filepath, parse_dates=parse_dates)


def load_optional_csv(filepath: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    if not Path(filepath).exists():
        logger.warning("Optional input file not found: %s", filepath)
        return pd.DataFrame()
    logger.info("Loaded input file: %s", filepath)
    return pd.read_csv(filepath, parse_dates=parse_dates)


def build_stress_scenario_definitions() -> pd.DataFrame:
    scenario_map = {
        "Market-wide selloff": {"AAPL": -0.05, "AMZN": -0.05, "GOOG": -0.05, "MSFT": -0.05, "NVDA": -0.05},
        "NVDA-specific crash": {"AAPL": -0.03, "AMZN": -0.03, "GOOG": -0.03, "MSFT": -0.03, "NVDA": -0.15},
        "Extreme single-day tech crash": {"AAPL": -0.10, "AMZN": -0.10, "GOOG": -0.10, "MSFT": -0.10, "NVDA": -0.15},
    }

    scenario_rows = []
    for scenario_name, shock_map in scenario_map.items():
        for ticker in TICKERS:
            scenario_rows.append({
                "Scenario": scenario_name,
                "Scenario Type": "Deterministic",
                "Scenario Horizon": "1D",
                "Ticker": ticker,
                "Shock Return": shock_map[ticker],
            })

    scenario_definitions = pd.DataFrame(scenario_rows)
    validate_scenario_definitions(scenario_definitions)
    return scenario_definitions


def validate_scenario_definitions(scenario_definitions: pd.DataFrame) -> None:
    validate_required_columns(
        scenario_definitions,
        ["Scenario", "Scenario Type", "Scenario Horizon", "Ticker", "Shock Return"],
        "Stress scenario definitions",
    )
    for scenario_name, scenario_df in scenario_definitions.groupby("Scenario"):
        scenario_tickers = scenario_df["Ticker"].tolist()
        if set(scenario_tickers) != set(TICKERS):
            raise ValueError(f"Scenario '{scenario_name}' does not contain shocks for every ticker.")


def build_latest_var_es_reference() -> pd.DataFrame:
    combined_var_es = load_optional_csv(COMBINED_PORTFOLIO_VAR_ES_FILEPATH, parse_dates=["Date"])
    if not combined_var_es.empty:
        rolling_var_col = identify_metric_column(combined_var_es, "Rolling VaR")
        rolling_es_col = identify_metric_column(combined_var_es, "Rolling ES")
        validate_required_columns(combined_var_es, ["Date", "Portfolio", "Model", rolling_var_col, rolling_es_col], "Combined portfolio VaR/ES")

        latest_reference = (
            combined_var_es
            .sort_values(["Portfolio", "Model", "Date"])
            .groupby(["Portfolio", "Model"], as_index=False)
            .tail(1)
            .rename(columns={rolling_var_col: "Latest VaR", rolling_es_col: "Latest ES"})
        )
        return latest_reference[["Portfolio", "Model", "Latest VaR", "Latest ES"]].reset_index(drop=True)

    latest_historical_summary = load_optional_csv(LATEST_PORTFOLIO_RISK_SUMMARY_FILEPATH, parse_dates=["Date"])
    if latest_historical_summary.empty:
        logger.warning("Latest VaR/ES reference files are unavailable. Stress losses will be produced without VaR/ES comparison.")
        return pd.DataFrame(columns=["Portfolio", "Model", "Latest VaR", "Latest ES"])

    rolling_var_col = identify_metric_column(latest_historical_summary, "Rolling VaR")
    rolling_es_col = identify_metric_column(latest_historical_summary, "Rolling ES")
    validate_required_columns(latest_historical_summary, ["Name", rolling_var_col, rolling_es_col], "Latest historical VaR/ES summary")

    historical_reference = latest_historical_summary.rename(
        columns={"Name": "Portfolio", rolling_var_col: "Latest VaR", rolling_es_col: "Latest ES"}
    ).copy()
    historical_reference["Model"] = HISTORICAL_MODEL_NAME
    return historical_reference[["Portfolio", "Model", "Latest VaR", "Latest ES"]].reset_index(drop=True)


def determine_best_model_map(latest_reference: pd.DataFrame) -> dict[str, str]:
    model_rankings = load_optional_csv(VOLATILITY_MODEL_RANKINGS_FILEPATH)
    if not model_rankings.empty:
        validate_required_columns(model_rankings, ["Portfolio", "Rank", "Model"], "Volatility model rankings")
        best_ranked_models = model_rankings.loc[model_rankings["Rank"] == 1, ["Portfolio", "Model"]]
        return dict(zip(best_ranked_models["Portfolio"], best_ranked_models["Model"]))

    if HISTORICAL_MODEL_NAME in latest_reference["Model"].values:
        return {portfolio_name: HISTORICAL_MODEL_NAME for portfolio_name in latest_reference["Portfolio"].unique()}

    return (
        latest_reference
        .sort_values(["Portfolio", "Model"])
        .drop_duplicates(subset=["Portfolio"])
        .set_index("Portfolio")["Model"]
        .to_dict()
    )


def compute_stress_outputs(
    frozen_weights: pd.DataFrame,
    scenario_definitions: pd.DataFrame,
    latest_reference: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    contribution_rows = []
    stress_result_rows = []

    for scenario_name, scenario_df in scenario_definitions.groupby("Scenario"):
        scenario_shocks = scenario_df.set_index("Ticker")["Shock Return"].reindex(TICKERS)

        for portfolio_name, portfolio_weights in frozen_weights.iterrows():
            contribution_df = pd.DataFrame({
                "Scenario": scenario_name,
                "Scenario Type": scenario_df["Scenario Type"].iloc[0],
                "Scenario Horizon": scenario_df["Scenario Horizon"].iloc[0],
                "Portfolio": portfolio_name,
                "Ticker": TICKERS,
                "Weight": portfolio_weights.reindex(TICKERS).to_numpy(),
                "Asset Shock Return": scenario_shocks.to_numpy(),
            })
            contribution_df["Asset Return Contribution"] = contribution_df["Weight"] * contribution_df["Asset Shock Return"]
            contribution_df["Asset Loss Contribution"] = -contribution_df["Asset Return Contribution"]

            portfolio_stress_return = float(contribution_df["Asset Return Contribution"].sum())
            portfolio_stress_loss = float(-portfolio_stress_return)
            if portfolio_stress_loss > 0.0:
                contribution_df["Contribution Percentage"] = contribution_df["Asset Loss Contribution"] / portfolio_stress_loss
            else:
                contribution_df["Contribution Percentage"] = np.nan

            validate_stress_contributions(contribution_df, portfolio_stress_return, portfolio_stress_loss)
            contribution_df = contribution_df.sort_values(
                ["Scenario", "Portfolio", "Asset Loss Contribution", "Ticker"],
                ascending=[True, True, False, True],
            ).reset_index(drop=True)
            contribution_rows.append(contribution_df)

            worst_row = contribution_df.iloc[0]
            portfolio_reference = latest_reference.loc[latest_reference["Portfolio"] == portfolio_name]

            if portfolio_reference.empty:
                stress_result_rows.append({
                    "Scenario": scenario_name,
                    "Scenario Type": scenario_df["Scenario Type"].iloc[0],
                    "Scenario Horizon": scenario_df["Scenario Horizon"].iloc[0],
                    "Portfolio": portfolio_name,
                    "Model": np.nan,
                    "Portfolio Stress Return": portfolio_stress_return,
                    "Portfolio Stress Loss": portfolio_stress_loss,
                    "Latest VaR": np.nan,
                    "Latest ES": np.nan,
                    "VaR Multiple": np.nan,
                    "ES Multiple": np.nan,
                    "Worst Contributing Asset": worst_row["Ticker"],
                    "Worst Asset Loss Contribution": worst_row["Asset Loss Contribution"],
                    "Interpretation": "VaR/ES comparison unavailable",
                })
                continue

            for _, reference_row in portfolio_reference.iterrows():
                latest_var = reference_row["Latest VaR"]
                latest_es = reference_row["Latest ES"]
                var_multiple = portfolio_stress_loss / latest_var if pd.notna(latest_var) and latest_var > 0.0 else np.nan
                es_multiple = portfolio_stress_loss / latest_es if pd.notna(latest_es) and latest_es > 0.0 else np.nan

                if pd.isna(latest_var) or pd.isna(latest_es):
                    interpretation = "VaR/ES comparison unavailable"
                elif portfolio_stress_loss <= latest_var:
                    interpretation = "Within latest VaR"
                elif portfolio_stress_loss > latest_var and portfolio_stress_loss <= latest_es:
                    interpretation = "Exceeds VaR but within ES"
                elif pd.notna(var_multiple) and var_multiple >= 2.0:
                    interpretation = "Severe stress loss relative to VaR"
                elif portfolio_stress_loss > latest_es:
                    interpretation = "Exceeds ES"
                else:
                    interpretation = "VaR/ES comparison unavailable"

                stress_result_rows.append({
                    "Scenario": scenario_name,
                    "Scenario Type": scenario_df["Scenario Type"].iloc[0],
                    "Scenario Horizon": scenario_df["Scenario Horizon"].iloc[0],
                    "Portfolio": portfolio_name,
                    "Model": reference_row["Model"],
                    "Portfolio Stress Return": portfolio_stress_return,
                    "Portfolio Stress Loss": portfolio_stress_loss,
                    "Latest VaR": latest_var,
                    "Latest ES": latest_es,
                    "VaR Multiple": var_multiple,
                    "ES Multiple": es_multiple,
                    "Worst Contributing Asset": worst_row["Ticker"],
                    "Worst Asset Loss Contribution": worst_row["Asset Loss Contribution"],
                    "Interpretation": interpretation,
                })

    stress_results = pd.DataFrame(stress_result_rows)
    stress_contributions = pd.concat(contribution_rows, ignore_index=True)
    return stress_results, stress_contributions


def validate_stress_contributions(
    contribution_df: pd.DataFrame,
    portfolio_stress_return: float,
    portfolio_stress_loss: float
) -> None:
    if not np.isclose(portfolio_stress_return, contribution_df["Asset Return Contribution"].sum(), atol=1e-12):
        raise ValueError("Portfolio stress return does not equal the sum of weighted asset shocks.")
    if not np.isclose(portfolio_stress_loss, -portfolio_stress_return, atol=1e-12):
        raise ValueError("Portfolio stress loss does not equal the negative of portfolio stress return.")
    if portfolio_stress_loss > 0.0:
        contribution_pct_sum = contribution_df["Contribution Percentage"].sum()
        if not np.isclose(contribution_pct_sum, 1.0, atol=1e-8):
            raise ValueError("Contribution percentages do not sum to approximately 1.")


def build_stress_test_summary(
    stress_results: pd.DataFrame,
    stress_contributions: pd.DataFrame,
    best_model_map: dict[str, str]
) -> pd.DataFrame:
    summary_rows = []
    scenario_portfolio_results = (
        stress_results
        .sort_values(["Scenario", "Portfolio", "Model"])
        .drop_duplicates(subset=["Scenario", "Portfolio"])
        .reset_index(drop=True)
    )

    selected_comparison_rows = []
    for portfolio_name, best_model_name in best_model_map.items():
        comparison_rows = stress_results.loc[
            (stress_results["Portfolio"] == portfolio_name) &
            (stress_results["Model"] == best_model_name)
        ]
        if comparison_rows.empty:
            comparison_rows = scenario_portfolio_results.loc[scenario_portfolio_results["Portfolio"] == portfolio_name]
        selected_comparison_rows.append(comparison_rows)
    comparison_results = pd.concat(selected_comparison_rows, ignore_index=True) if selected_comparison_rows else scenario_portfolio_results.copy()

    average_stress_loss_by_portfolio = scenario_portfolio_results.groupby("Portfolio")["Portfolio Stress Loss"].mean()

    for portfolio_name, portfolio_results in scenario_portfolio_results.groupby("Portfolio"):
        worst_result_row = portfolio_results.sort_values(["Portfolio Stress Loss", "Scenario"], ascending=[False, True]).iloc[0]
        contribution_subset = stress_contributions.loc[stress_contributions["Portfolio"] == portfolio_name].copy()
        scenario_worst_assets = (
            contribution_subset
            .sort_values(["Scenario", "Asset Loss Contribution", "Ticker"], ascending=[True, False, True])
            .drop_duplicates(subset=["Scenario"])
        )
        overall_worst_asset_row = scenario_worst_assets.sort_values(["Asset Loss Contribution", "Ticker"], ascending=[False, True]).iloc[0]

        comparison_subset = comparison_results.loc[comparison_results["Portfolio"] == portfolio_name]
        num_scenarios_exceeding_var = int((comparison_subset["Portfolio Stress Loss"] > comparison_subset["Latest VaR"]).fillna(False).sum())
        num_scenarios_exceeding_es = int((comparison_subset["Portfolio Stress Loss"] > comparison_subset["Latest ES"]).fillna(False).sum())

        average_nvda_contribution = contribution_subset.loc[contribution_subset["Ticker"] == "NVDA", "Contribution Percentage"].mean()
        if average_stress_loss_by_portfolio[portfolio_name] == average_stress_loss_by_portfolio.min():
            main_risk_comment = "Most defensive portfolio under the selected stress scenarios."
        elif average_nvda_contribution > 0.4:
            main_risk_comment = "Stress losses are mainly driven by NVDA exposure."
        elif (contribution_subset["Contribution Percentage"].max() > 0.5) or (contribution_subset["Weight"].max() > 0.5):
            main_risk_comment = "Highest stress sensitivity due to concentrated exposure."
        else:
            main_risk_comment = "Stress losses are broadly distributed across assets."

        summary_rows.append({
            "Portfolio": portfolio_name,
            "Worst Scenario": worst_result_row["Scenario"],
            "Worst Stress Loss": worst_result_row["Portfolio Stress Loss"],
            "Worst Stress Return": worst_result_row["Portfolio Stress Return"],
            "Worst Contributing Asset": overall_worst_asset_row["Ticker"],
            "Average Stress Loss": portfolio_results["Portfolio Stress Loss"].mean(),
            "Median Stress Loss": portfolio_results["Portfolio Stress Loss"].median(),
            "Number of Scenarios": portfolio_results["Scenario"].nunique(),
            "Number of Scenarios Exceeding Latest VaR": num_scenarios_exceeding_var,
            "Number of Scenarios Exceeding Latest ES": num_scenarios_exceeding_es,
            "Main Risk Comment": main_risk_comment,
        })

    return pd.DataFrame(summary_rows), comparison_results


def save_figure(fig: plt.Figure, filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", filepath)


def plot_stress_loss_by_scenario(scenario_portfolio_results: pd.DataFrame) -> None:
    scenario_order = scenario_portfolio_results["Scenario"].drop_duplicates().tolist()
    portfolio_order = scenario_portfolio_results["Portfolio"].drop_duplicates().tolist()
    x_positions = np.arange(len(scenario_order))
    bar_width = 0.25
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for portfolio_index, portfolio_name in enumerate(portfolio_order):
        portfolio_losses = (
            scenario_portfolio_results
            .loc[scenario_portfolio_results["Portfolio"] == portfolio_name]
            .set_index("Scenario")
            .reindex(scenario_order)["Portfolio Stress Loss"]
            .to_numpy()
        )
        ax.bar(
            x_positions + (portfolio_index - 1) * bar_width,
            portfolio_losses,
            width=bar_width,
            label=portfolio_name,
            color=colors[portfolio_index % len(colors)],
        )

    ax.set_title("Stress Loss by Scenario and Portfolio")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Portfolio Stress Loss")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(scenario_order, rotation=20, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    save_figure(fig, STRESS_TESTING_PLOTS_DIR / "stress_loss_by_scenario.png")


def plot_stress_loss_heatmap(scenario_portfolio_results: pd.DataFrame) -> None:
    heatmap_df = scenario_portfolio_results.pivot(index="Portfolio", columns="Scenario", values="Portfolio Stress Loss")

    fig, ax = plt.subplots(figsize=(10, 5))
    image = ax.imshow(heatmap_df.to_numpy(), cmap="Reds", aspect="auto")
    ax.set_title("Stress Loss Heatmap")
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Portfolio")
    ax.set_xticks(np.arange(len(heatmap_df.columns)))
    ax.set_xticklabels(heatmap_df.columns, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(heatmap_df.index)))
    ax.set_yticklabels(heatmap_df.index)

    for row_index in range(heatmap_df.shape[0]):
        for col_index in range(heatmap_df.shape[1]):
            ax.text(col_index, row_index, f"{heatmap_df.iloc[row_index, col_index]:.3f}", ha="center", va="center", color="black")

    fig.colorbar(image, ax=ax, label="Stress Loss")
    save_figure(fig, STRESS_TESTING_PLOTS_DIR / "stress_loss_heatmap.png")


def plot_asset_contributions(stress_contributions: pd.DataFrame) -> None:
    for (scenario_name, portfolio_name), contribution_subset in stress_contributions.groupby(["Scenario", "Portfolio"]):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(contribution_subset["Ticker"], contribution_subset["Asset Loss Contribution"], color="#1f77b4")
        ax.set_title(f"Asset Loss Contributions: {scenario_name} | {portfolio_name}")
        ax.set_xlabel("Ticker")
        ax.set_ylabel("Asset Loss Contribution")
        ax.grid(True, axis="y", alpha=0.3)
        filename = f"asset_contributions_{safe_filename(scenario_name)}_{safe_filename(portfolio_name)}.png"
        save_figure(fig, STRESS_TESTING_PLOTS_DIR / filename)


def plot_var_or_es_multiple(
    comparison_results: pd.DataFrame,
    value_column: str,
    title: str,
    output_filename: str
) -> None:
    scenario_order = comparison_results["Scenario"].drop_duplicates().tolist()
    portfolio_order = comparison_results["Portfolio"].drop_duplicates().tolist()
    x_positions = np.arange(len(scenario_order))
    bar_width = 0.25
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(12, 6))
    for portfolio_index, portfolio_name in enumerate(portfolio_order):
        portfolio_values = (
            comparison_results
            .loc[comparison_results["Portfolio"] == portfolio_name]
            .set_index("Scenario")
            .reindex(scenario_order)[value_column]
            .to_numpy()
        )
        ax.bar(
            x_positions + (portfolio_index - 1) * bar_width,
            portfolio_values,
            width=bar_width,
            label=portfolio_name,
            color=colors[portfolio_index % len(colors)],
        )

    ax.set_title(title)
    ax.set_xlabel("Scenario")
    ax.set_ylabel(value_column)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(scenario_order, rotation=20, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    save_figure(fig, STRESS_TESTING_PLOTS_DIR / output_filename)


#######################
#
# (3) MAIN
#
#######################
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger.info("INITIALIZING STRESS TESTING...")
    STRESS_TESTING_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    STRESS_TESTING_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    frozen_weights = load_frozen_weights(FROZEN_WEIGHTS_FILEPATH, TICKERS)
    logger.info("Loaded frozen portfolio weights.\n%s", frozen_weights.reset_index().rename(columns={"index": "Portfolio"}).to_string(index=False))

    scenario_definitions = build_stress_scenario_definitions()
    latest_var_es_reference = build_latest_var_es_reference()
    best_model_map = determine_best_model_map(latest_var_es_reference)

    stress_results, stress_contributions = compute_stress_outputs(
        frozen_weights,
        scenario_definitions,
        latest_var_es_reference,
    )
    stress_summary, best_model_comparison_results = build_stress_test_summary(
        stress_results,
        stress_contributions,
        best_model_map,
    )

    save_dataframe(scenario_definitions, str(STRESS_SCENARIO_DEFINITIONS_FILEPATH))
    save_dataframe(stress_results, str(STRESS_TEST_RESULTS_FILEPATH))
    save_dataframe(stress_contributions, str(STRESS_TEST_ASSET_CONTRIBUTIONS_FILEPATH))
    save_dataframe(stress_summary, str(STRESS_TEST_SUMMARY_FILEPATH))

    scenario_portfolio_results = (
        stress_results
        .sort_values(["Scenario", "Portfolio", "Model"])
        .drop_duplicates(subset=["Scenario", "Portfolio"])
        .reset_index(drop=True)
    )

    plot_stress_loss_by_scenario(scenario_portfolio_results)
    plot_stress_loss_heatmap(scenario_portfolio_results)
    plot_asset_contributions(stress_contributions)
    plot_var_or_es_multiple(
        best_model_comparison_results,
        value_column="VaR Multiple",
        title="Stress Loss Relative to Latest VaR",
        output_filename="var_multiple_by_scenario.png",
    )
    plot_var_or_es_multiple(
        best_model_comparison_results,
        value_column="ES Multiple",
        title="Stress Loss Relative to Latest ES",
        output_filename="es_multiple_by_scenario.png",
    )

    logger.info("Number of deterministic scenarios: %s", scenario_definitions["Scenario"].nunique())
    for _, summary_row in stress_summary.iterrows():
        logger.info(
            "%s: worst scenario = %s; worst contributing asset = %s; scenarios exceeding latest VaR = %s; scenarios exceeding latest ES = %s.",
            summary_row["Portfolio"],
            summary_row["Worst Scenario"],
            summary_row["Worst Contributing Asset"],
            summary_row["Number of Scenarios Exceeding Latest VaR"],
            summary_row["Number of Scenarios Exceeding Latest ES"],
        )

    logger.info("Saved CSV outputs to %s", STRESS_TESTING_OUTPUT_DIR)
    logger.info("Saved plot outputs to %s", STRESS_TESTING_PLOTS_DIR)


#######################
#
# (4) RUN MAIN
#
#######################
if __name__ == "__main__":
    main()
