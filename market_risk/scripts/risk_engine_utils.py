import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import chi2, t


DATE_COL = "Date"
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_WINDOW_SIZE = 252
DEFAULT_FALLBACK_NU = 8.0
EPSILON = 1e-12


def validate_required_file(filepath: str | Path) -> None:
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Required input file not found: {filepath}")


def validate_required_columns(df: pd.DataFrame, required_columns: list[str], dataframe_name: str) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{dataframe_name} is missing required columns: {missing_columns}")


def standardize_daily_index(df: pd.DataFrame) -> pd.DataFrame:
    standardized_df = df.copy()
    standardized_index = pd.to_datetime(standardized_df.index)
    if standardized_index.tz is not None:
        standardized_index = standardized_index.tz_convert("UTC").tz_localize(None)
    standardized_df.index = standardized_index.normalize()
    return standardized_df.sort_index()


def load_price_data(filepath: str | Path) -> pd.DataFrame:
    validate_required_file(filepath)
    return standardize_daily_index(pd.read_parquet(filepath))


def load_optional_price_data(filepath: str | Path) -> pd.DataFrame:
    if not Path(filepath).exists():
        return pd.DataFrame()
    return standardize_daily_index(pd.read_parquet(filepath))


def extract_close_prices(price_data: pd.DataFrame, tickers: list[str], close_col: str) -> pd.DataFrame:
    if not isinstance(price_data.columns, pd.MultiIndex):
        raise ValueError("Expected a MultiIndex price dataframe with a close-price level.")
    close_prices = price_data[close_col].reindex(columns=tickers)
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


def load_frozen_weights(filepath: str | Path, tickers: list[str]) -> pd.DataFrame:
    validate_required_file(filepath)
    frozen_weights_long = pd.read_csv(filepath)
    validate_required_columns(frozen_weights_long, ["Portfolio", "Ticker", "Weight"], "Frozen portfolio weights")

    frozen_weights_wide = (
        frozen_weights_long
        .pivot(index="Portfolio", columns="Ticker", values="Weight")
        .reindex(columns=tickers)
    )

    if frozen_weights_wide.isna().any().any():
        raise ValueError("Frozen portfolio weights contain missing values after pivoting to wide format.")

    weight_sums = frozen_weights_wide.sum(axis=1)
    if not np.allclose(weight_sums.to_numpy(), 1.0, atol=1e-8):
        raise ValueError("Frozen portfolio weights do not sum to 1.")
    if ((frozen_weights_wide < 0.0) | (frozen_weights_wide > 1.0)).any().any():
        raise ValueError("Frozen portfolio weights must lie between 0 and 1.")

    return frozen_weights_wide


def build_portfolio_returns_from_frozen_weights(
    asset_returns: pd.DataFrame,
    frozen_weights: pd.DataFrame,
    tickers: list[str]
) -> pd.DataFrame:
    aligned_asset_returns = asset_returns.reindex(columns=tickers)
    portfolio_returns = {
        portfolio_name: aligned_asset_returns @ weight_row.to_numpy()
        for portfolio_name, weight_row in frozen_weights.iterrows()
    }
    return pd.DataFrame(portfolio_returns, index=aligned_asset_returns.index)


def filter_backtest_period(
    df: pd.DataFrame,
    backtest_start: str,
    backtest_end: str | None = None,
    date_column: str = DATE_COL
) -> pd.DataFrame:
    backtest_start_timestamp = pd.Timestamp(backtest_start)
    filtered_df = df.loc[df[date_column] >= backtest_start_timestamp].copy()
    if backtest_end is not None:
        backtest_end_timestamp = pd.Timestamp(backtest_end)
        filtered_df = filtered_df.loc[filtered_df[date_column] < backtest_end_timestamp].copy()
    return filtered_df


def validate_backtest_output_dates(
    df: pd.DataFrame,
    backtest_start: str,
    entity_label: str,
    date_column: str = DATE_COL
) -> None:
    if df.empty:
        raise ValueError(f"{entity_label} backtest output is empty after filtering.")
    if df[date_column].min() < pd.Timestamp(backtest_start):
        raise ValueError(f"{entity_label} backtest output contains dates earlier than BACKTEST_START.")


def identify_metric_column(df: pd.DataFrame, metric_prefix: str) -> str:
    matching_columns = [column for column in df.columns if column.startswith(metric_prefix)]
    if len(matching_columns) != 1:
        raise ValueError(f"Expected exactly one column starting with '{metric_prefix}', found {matching_columns}")
    return matching_columns[0]


def standardized_t_scale(nu: float) -> float:
    if nu <= 2.0:
        raise ValueError("Student-t degrees of freedom must be greater than 2 for finite variance.")
    return float(np.sqrt((nu - 2.0) / nu))


def standardized_t_quantile(nu: float, tail_probability: float) -> float:
    raw_quantile = t.ppf(tail_probability, df=nu)
    standardized_quantile = raw_quantile * standardized_t_scale(nu)
    if standardized_quantile >= 0.0:
        raise ValueError("Standardized Student-t lower-tail quantile must be negative.")
    return float(standardized_quantile)


def standardized_t_expected_shortfall(nu: float, tail_probability: float) -> float:
    raw_quantile = t.ppf(tail_probability, df=nu)
    raw_expected_shortfall = -t.pdf(raw_quantile, df=nu) * (nu + raw_quantile ** 2) / ((nu - 1.0) * tail_probability)
    standardized_expected_shortfall = raw_expected_shortfall * standardized_t_scale(nu)
    if standardized_expected_shortfall >= 0.0:
        raise ValueError("Standardized Student-t lower-tail expected shortfall must be negative.")
    return float(standardized_expected_shortfall)


def estimate_standardized_t_nu(
    residuals: pd.Series | np.ndarray,
    lower_bound: float = 2.1,
    upper_bound: float = 50.0,
    fallback_nu: float = DEFAULT_FALLBACK_NU,
    epsilon: float = EPSILON
) -> tuple[float, int, str]:
    residual_array = np.asarray(residuals, dtype=float)
    residual_array = residual_array[np.isfinite(residual_array)]
    num_residuals = int(len(residual_array))

    if num_residuals == 0:
        return fallback_nu, num_residuals, "Fallback to nu=8.0: no valid standardized residuals."

    def negative_log_likelihood(candidate_nu: float) -> float:
        scale = standardized_t_scale(candidate_nu)
        pdf_values = t.pdf(residual_array / scale, df=candidate_nu) / scale
        return float(-np.sum(np.log(np.maximum(pdf_values, epsilon))))

    try:
        optimization_result = minimize_scalar(
            negative_log_likelihood,
            bounds=(lower_bound, upper_bound),
            method="bounded",
            options={"xatol": 1e-3},
        )
        if optimization_result.success and np.isfinite(optimization_result.x):
            return float(optimization_result.x), num_residuals, "Estimated successfully."
        return fallback_nu, num_residuals, f"Fallback to nu=8.0: optimization failed ({optimization_result.message})."
    except Exception as error:
        return fallback_nu, num_residuals, f"Fallback to nu=8.0: estimation error ({error})."


def compute_kupiec_test(
    num_observations: int,
    actual_breaches: int,
    confidence_level: float,
    epsilon: float = EPSILON
) -> dict[str, float | str]:
    if num_observations <= 0:
        return {
            "Tail Probability": 1.0 - confidence_level,
            "Expected VaR Breaches": np.nan,
            "Breach Rate": np.nan,
            "Expected Breach Rate": 1.0 - confidence_level,
            "Kupiec LR Statistic": np.nan,
            "Kupiec p-value": np.nan,
            "Test Decision": "Insufficient data",
            "Interpretation": "Insufficient data to evaluate the Kupiec test.",
        }

    tail_probability = 1.0 - confidence_level
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
    kupiec_lr_statistic = float(-2.0 * (log_likelihood_null - log_likelihood_alternative))
    kupiec_p_value = float(chi2.sf(kupiec_lr_statistic, df=1))

    if kupiec_p_value < 0.05 and breach_rate > tail_probability:
        test_decision = "Reject VaR model"
        interpretation = "Reject: VaR model is too optimistic and underestimates risk."
    elif kupiec_p_value < 0.05 and breach_rate < tail_probability:
        test_decision = "Reject VaR model"
        interpretation = "Reject: VaR model is too conservative."
    else:
        test_decision = "Do not reject VaR model"
        interpretation = "Do not reject: breach frequency is statistically consistent with the VaR confidence level."

    return {
        "Tail Probability": tail_probability,
        "Expected VaR Breaches": tail_probability * num_observations,
        "Breach Rate": breach_rate,
        "Expected Breach Rate": tail_probability,
        "Kupiec LR Statistic": kupiec_lr_statistic,
        "Kupiec p-value": kupiec_p_value,
        "Test Decision": test_decision,
        "Interpretation": interpretation,
    }


def validate_var_es_outputs(
    rolling_var: pd.DataFrame,
    rolling_es: pd.DataFrame,
    model_name: str,
    logger: logging.Logger
) -> None:
    valid_var = rolling_var.stack().dropna()
    valid_es = rolling_es.stack().dropna()

    if (valid_var < -1e-10).any():
        raise ValueError(f"{model_name} rolling VaR contains negative values.")
    if (valid_es < -1e-10).any():
        raise ValueError(f"{model_name} rolling ES contains negative values.")

    es_less_than_var_count = int((rolling_es < rolling_var).stack().dropna().sum())
    if es_less_than_var_count > 0:
        logger.warning(
            "%s rolling ES is below rolling VaR on %s observations.",
            model_name,
            es_less_than_var_count,
        )


def build_model_risk_metrics_table(
    returns: pd.DataFrame,
    rolling_var: pd.DataFrame,
    rolling_es: pd.DataFrame,
    model_name: str,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    window_size: int = DEFAULT_WINDOW_SIZE
) -> pd.DataFrame:
    var_col = f"Rolling VaR ({confidence_level:.0%}, {window_size}D)"
    es_col = f"Rolling ES ({confidence_level:.0%}, {window_size}D)"
    risk_metric_rows = []

    for portfolio_name in returns.columns:
        portfolio_returns = returns[portfolio_name]
        portfolio_var = rolling_var[portfolio_name]
        portfolio_es = rolling_es[portfolio_name]
        var_breach = (portfolio_returns < -portfolio_var).where(portfolio_var.notna())
        loss_exceeds_es = (portfolio_returns < -portfolio_es).where(portfolio_es.notna())

        risk_metric_rows.append(pd.DataFrame({
            DATE_COL: portfolio_returns.index,
            "Portfolio": portfolio_name,
            "Return": portfolio_returns.to_numpy(),
            "Model": model_name,
            var_col: portfolio_var.to_numpy(),
            es_col: portfolio_es.to_numpy(),
            "VaR Breach": var_breach.to_numpy(),
            "Loss Exceeds ES": loss_exceeds_es.to_numpy(),
        }))

    return pd.concat(risk_metric_rows, ignore_index=True)


def build_model_backtest_summary(
    risk_metrics: pd.DataFrame,
    model_name: str,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    window_size: int = DEFAULT_WINDOW_SIZE
) -> pd.DataFrame:
    var_col = f"Rolling VaR ({confidence_level:.0%}, {window_size}D)"
    es_col = f"Rolling ES ({confidence_level:.0%}, {window_size}D)"
    summary_rows = []

    for portfolio_name, portfolio_metrics in risk_metrics.groupby("Portfolio"):
        valid_metrics = portfolio_metrics.loc[portfolio_metrics[var_col].notna()].copy()
        num_observations = len(valid_metrics)
        actual_breaches = int(valid_metrics["VaR Breach"].fillna(False).sum())
        kupiec_results = compute_kupiec_test(num_observations, actual_breaches, confidence_level)
        breach_day_returns = valid_metrics.loc[valid_metrics["VaR Breach"].fillna(False), "Return"]

        summary_rows.append({
            "Model": model_name,
            "Portfolio": portfolio_name,
            "Confidence Level": confidence_level,
            "Tail Probability": kupiec_results["Tail Probability"],
            "Number of Observations": num_observations,
            "Actual VaR Breaches": actual_breaches,
            "Expected VaR Breaches": kupiec_results["Expected VaR Breaches"],
            "Breach Rate": kupiec_results["Breach Rate"],
            "Expected Breach Rate": kupiec_results["Expected Breach Rate"],
            "Average Return on Breach Days": breach_day_returns.mean(),
            "Worst Return": valid_metrics["Return"].min() if num_observations > 0 else np.nan,
            "Average VaR": valid_metrics[var_col].mean(),
            "Average ES": valid_metrics[es_col].mean(),
            "Kupiec LR Statistic": kupiec_results["Kupiec LR Statistic"],
            "Kupiec p-value": kupiec_results["Kupiec p-value"],
            "Test Decision": kupiec_results["Test Decision"],
            "Interpretation": kupiec_results["Interpretation"],
        })

    return pd.DataFrame(summary_rows)


def build_latest_model_summary(
    risk_metrics: pd.DataFrame,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    window_size: int = DEFAULT_WINDOW_SIZE
) -> pd.DataFrame:
    var_col = f"Rolling VaR ({confidence_level:.0%}, {window_size}D)"
    es_col = f"Rolling ES ({confidence_level:.0%}, {window_size}D)"
    latest_date = risk_metrics[DATE_COL].max()
    return risk_metrics.loc[
        risk_metrics[DATE_COL] == latest_date,
        [DATE_COL, "Portfolio", "Model", var_col, es_col],
    ].reset_index(drop=True)


def save_dataframe(df: pd.DataFrame, filepath: str | Path) -> None:
    output_path = Path(filepath)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        if output_path.exists():
            output_path.unlink()
        df.to_csv(output_path, index=False)
    except PermissionError:
        if output_path.exists():
            logging.getLogger(__name__).warning(
                "Could not overwrite %s because it is currently in use. Keeping the existing file.",
                output_path,
            )
            return
        raise


def safe_filename(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
