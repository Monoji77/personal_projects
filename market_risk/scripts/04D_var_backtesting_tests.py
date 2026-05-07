#######################
#
# (0) LIBRARIES
#
#######################
import logging
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import chi2

from path_utils import project_path

logger = logging.getLogger(__name__)


#######################
#
# (1) GLOBAL VARIABLES
#
#######################
PORTFOLIO_RISK_METRICS_FILEPATH = project_path("figure", "risk_engine", "portfolio_rolling_var_es.csv")
PORTFOLIO_VAR_BACKTESTING_TESTS_FILEPATH = project_path("figure", "risk_engine", "portfolio_var_backtesting_tests.csv")
VAR_BREACH_COL = "VaR Breach"
PORTFOLIO_COL = "Portfolio"
DEFAULT_CONFIDENCE_LEVEL = 0.95
SIGNIFICANCE_LEVEL = 0.05


#######################
#
# (2) HELPER FUNCTIONS
#
#######################
def validate_required_file(filepath: str) -> None:
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Required input file not found: {filepath}")


def validate_required_columns(df: pd.DataFrame, required_columns: list[str], dataframe_name: str) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{dataframe_name} is missing required columns: {missing_columns}")


def identify_metric_column(df: pd.DataFrame, metric_prefix: str) -> str:
    matching_columns = [column for column in df.columns if column.startswith(metric_prefix)]
    if len(matching_columns) != 1:
        raise ValueError(f"Expected exactly one column starting with '{metric_prefix}', found {matching_columns}")
    return matching_columns[0]


def infer_confidence_level(rolling_var_col: str) -> float:
    confidence_match = re.search(r"\((\d+)%", rolling_var_col)
    if confidence_match is None:
        logger.warning("Could not infer confidence level from %s. Falling back to %.2f.", rolling_var_col, DEFAULT_CONFIDENCE_LEVEL)
        return DEFAULT_CONFIDENCE_LEVEL
    return float(confidence_match.group(1)) / 100.0


def coerce_var_breach_to_bool(var_breach_series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(var_breach_series):
        return var_breach_series
    normalized_series = var_breach_series.astype(str).str.strip().str.lower()
    return normalized_series.map({"true": True, "false": False})


def compute_kupiec_lr_statistic(
    num_observations: int,
    actual_breaches: int,
    tail_probability: float,
    epsilon: float = 1e-12
) -> float:
    if num_observations <= 0:
        return np.nan

    breach_rate = actual_breaches / num_observations
    adjusted_tail_probability = np.clip(tail_probability, epsilon, 1.0 - epsilon)
    adjusted_breach_rate = np.clip(breach_rate, epsilon, 1.0 - epsilon)

    log_likelihood_null = (
        (num_observations - actual_breaches) * np.log(1.0 - adjusted_tail_probability) +
        actual_breaches * np.log(adjusted_tail_probability)
    )
    log_likelihood_alternative = (
        (num_observations - actual_breaches) * np.log(1.0 - adjusted_breach_rate) +
        actual_breaches * np.log(adjusted_breach_rate)
    )
    return float(-2.0 * (log_likelihood_null - log_likelihood_alternative))


def build_interpretation(
    p_value: float,
    breach_rate: float,
    expected_breach_rate: float
) -> tuple[str, str]:
    if np.isnan(p_value):
        return "Insufficient data", "Insufficient data to evaluate the Kupiec test."
    if p_value < SIGNIFICANCE_LEVEL and breach_rate > expected_breach_rate:
        return "Reject VaR model", "Reject: VaR model is too optimistic and underestimates risk."
    if p_value < SIGNIFICANCE_LEVEL and breach_rate < expected_breach_rate:
        return "Reject VaR model", "Reject: VaR model is too conservative."
    return "Do not reject VaR model", "Do not reject: breach frequency is statistically consistent with the VaR confidence level."


def build_kupiec_backtesting_table(
    portfolio_risk_metrics: pd.DataFrame,
    rolling_var_col: str,
    confidence_level: float
) -> pd.DataFrame:
    tail_probability = 1.0 - confidence_level
    summary_rows = []

    for portfolio_name, portfolio_metrics in portfolio_risk_metrics.groupby(PORTFOLIO_COL):
        valid_metrics = portfolio_metrics.loc[portfolio_metrics[rolling_var_col].notna()].copy()
        num_observations = len(valid_metrics)
        actual_breaches = int(valid_metrics[VAR_BREACH_COL].fillna(False).sum())
        expected_breaches = tail_probability * num_observations
        breach_rate = actual_breaches / num_observations if num_observations > 0 else np.nan
        kupiec_lr_statistic = compute_kupiec_lr_statistic(num_observations, actual_breaches, tail_probability)
        kupiec_p_value = float(chi2.sf(kupiec_lr_statistic, df=1)) if np.isfinite(kupiec_lr_statistic) else np.nan
        test_decision, interpretation = build_interpretation(kupiec_p_value, breach_rate, tail_probability)

        summary_rows.append({
            "Portfolio": portfolio_name,
            "Confidence Level": confidence_level,
            "Tail Probability": tail_probability,
            "Number of Observations": num_observations,
            "Actual VaR Breaches": actual_breaches,
            "Expected VaR Breaches": expected_breaches,
            "Breach Rate": breach_rate,
            "Expected Breach Rate": tail_probability,
            "Kupiec LR Statistic": kupiec_lr_statistic,
            "Kupiec p-value": kupiec_p_value,
            "Test Decision": test_decision,
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


#######################
#
# (3) MAIN
#
#######################
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    logger.info("INITIALIZING FORMAL VaR BACKTESTING TESTS...")

    validate_required_file(PORTFOLIO_RISK_METRICS_FILEPATH)
    portfolio_risk_metrics = pd.read_csv(PORTFOLIO_RISK_METRICS_FILEPATH)
    logger.info("Loaded input file: %s", PORTFOLIO_RISK_METRICS_FILEPATH)

    rolling_var_col = identify_metric_column(portfolio_risk_metrics, "Rolling VaR")
    validate_required_columns(
        portfolio_risk_metrics,
        [PORTFOLIO_COL, VAR_BREACH_COL, rolling_var_col],
        "Portfolio risk metrics",
    )

    confidence_level = infer_confidence_level(rolling_var_col)
    portfolio_risk_metrics[VAR_BREACH_COL] = coerce_var_breach_to_bool(portfolio_risk_metrics[VAR_BREACH_COL])
    if portfolio_risk_metrics[VAR_BREACH_COL].isna().any():
        raise ValueError("VaR Breach column could not be fully converted to boolean values.")

    kupiec_results = build_kupiec_backtesting_table(portfolio_risk_metrics, rolling_var_col, confidence_level)
    save_dataframe(kupiec_results, PORTFOLIO_VAR_BACKTESTING_TESTS_FILEPATH)

    logger.info("Backtesting test results table\n%s", kupiec_results.to_string(index=False))
    logger.info("Saved CSV file to %s", PORTFOLIO_VAR_BACKTESTING_TESTS_FILEPATH)
    for _, result_row in kupiec_results.iterrows():
        logger.info("%s: %s", result_row["Portfolio"], result_row["Interpretation"])


#######################
#
# (4) RUN MAIN
#
#######################
if __name__ == "__main__":
    main()
