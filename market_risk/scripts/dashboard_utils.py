from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from path_utils import project_path
from risk_engine_utils import (
    combine_price_data,
    compute_simple_returns,
    extract_close_prices,
    load_frozen_weights,
    load_optional_price_data,
    load_price_data,
    standardize_daily_index,
    standardized_t_expected_shortfall,
    standardized_t_quantile,
)


TICKERS = ["AAPL", "AMZN", "GOOG", "MSFT", "NVDA"]
FIXED_PORTFOLIOS = [
    "Equal Weighted Portfolio",
    "Long-only Global Minimum Variance Portfolio",
    "Long-only Tangency Portfolio",
]
BACKTEST_START = "2025-01-01"
WINDOW_SIZE = 252
CONFIDENCE_LEVEL = 0.95
TAIL_PROBABILITY = 1.0 - CONFIDENCE_LEVEL
ANNUALIZATION_FACTOR = 252
EWMA_LAMBDA = 0.94
EWMA_DEFAULT_NU = 8.0
CLOSE_COL = "Close"
DATE_COL = "Date"
RF_RATE_DAILY_COL = "rf_rate_daily"
PERCENT_TOLERANCE = 1e-6

HISTORICAL_PRICES_FILE = project_path("data", "historical", "top_tech_assets_prices.parquet")
NEW_DAILY_PRICES_FILE = project_path("data", "new_daily", "top_tech_assets_prices.parquet")
ASSET_RETURNS_FILE = project_path("figure", "risk_engine", "asset_returns.csv")
RISK_FREE_RATE_FILE = project_path("data", "historical", "rf_rate_daily.parquet")
FROZEN_WEIGHTS_FILE = project_path("figure", "risk_engine", "frozen_portfolio_weights.csv")
PORTFOLIO_STUDY_SUMMARY_FILE = project_path("figure", "portfolio_study", "portfolio_summary.csv")
PORTFOLIO_CONSTRUCTION_SUMMARY_FILE = project_path("figure", "risk_engine", "portfolio_construction_summary.csv")
PORTFOLIO_ROLLING_VAR_ES_FILE = project_path("figure", "risk_engine", "portfolio_rolling_var_es.csv")
PORTFOLIO_VAR_BACKTEST_SUMMARY_FILE = project_path("figure", "risk_engine", "portfolio_var_backtest_summary.csv")
PORTFOLIO_VAR_BACKTEST_TESTS_FILE = project_path("figure", "risk_engine", "portfolio_var_backtesting_tests.csv")
LATEST_PORTFOLIO_VAR_ES_FILE = project_path("figure", "risk_engine", "latest_portfolio_rolling_var_es_summary.csv")
COMBINED_VOL_MODEL_FILE = project_path("figure", "risk_engine", "volatility_models", "combined_portfolio_var_es.csv")
COMBINED_VOL_MODEL_SUMMARY_FILE = project_path("figure", "risk_engine", "volatility_models", "combined_volatility_model_backtest_summary.csv")
VOL_MODEL_RANKINGS_FILE = project_path("figure", "risk_engine", "volatility_models", "volatility_model_rankings.csv")
STRESS_TEST_RESULTS_FILE = project_path("figure", "risk_engine", "stress_testing", "stress_test_results.csv")
STRESS_TEST_ASSET_CONTRIBUTIONS_FILE = project_path("figure", "risk_engine", "stress_testing", "stress_test_asset_contributions.csv")
STRESS_TEST_SUMMARY_FILE = project_path("figure", "risk_engine", "stress_testing", "stress_test_summary.csv")
RISK_DRIVER_SUMMARY_FILE = project_path("figure", "risk_engine", "risk_attribution", "risk_driver_summary.csv")
PORTFOLIO_RISK_CONTRIBUTIONS_FILE = project_path("figure", "risk_engine", "risk_attribution", "portfolio_risk_contributions.csv")
DRAWDOWN_ATTRIBUTION_FILE = project_path("figure", "risk_engine", "risk_attribution", "drawdown_attribution_summary.csv")


PRESET_WEIGHTS_PCT = {
    "Equal Weight": {"AAPL": 20.0, "AMZN": 20.0, "GOOG": 20.0, "MSFT": 20.0, "NVDA": 20.0},
    "100% AAPL": {"AAPL": 100.0, "AMZN": 0.0, "GOOG": 0.0, "MSFT": 0.0, "NVDA": 0.0},
    "100% NVDA": {"AAPL": 0.0, "AMZN": 0.0, "GOOG": 0.0, "MSFT": 0.0, "NVDA": 100.0},
    "Tech Balanced": {"AAPL": 25.0, "AMZN": 15.0, "GOOG": 20.0, "MSFT": 25.0, "NVDA": 15.0},
    "High NVDA Exposure": {"AAPL": 15.0, "AMZN": 10.0, "GOOG": 10.0, "MSFT": 15.0, "NVDA": 50.0},
}

SCENARIOS = {
    "Market-wide selloff": {"AAPL": -0.05, "AMZN": -0.05, "GOOG": -0.05, "MSFT": -0.05, "NVDA": -0.05},
    "NVDA-specific crash": {"AAPL": -0.03, "AMZN": -0.03, "GOOG": -0.03, "MSFT": -0.03, "NVDA": -0.15},
    "Extreme single-day tech crash": {"AAPL": -0.10, "AMZN": -0.10, "GOOG": -0.10, "MSFT": -0.10, "NVDA": -0.15},
}


def _read_csv(filepath: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    dataframe = pd.read_csv(filepath)
    if parse_dates:
        for column in parse_dates:
            if column in dataframe.columns:
                dataframe[column] = pd.to_datetime(dataframe[column])
    return dataframe


@st.cache_data(show_spinner=False)
def load_asset_returns() -> pd.DataFrame:
    if ASSET_RETURNS_FILE.exists():
        asset_returns = _read_csv(ASSET_RETURNS_FILE, parse_dates=[DATE_COL])
        if DATE_COL not in asset_returns.columns:
            raise ValueError("asset_returns.csv must contain a Date column.")
        missing_tickers = [ticker for ticker in TICKERS if ticker not in asset_returns.columns]
        if missing_tickers:
            raise ValueError(f"asset_returns.csv is missing required tickers: {missing_tickers}")
        asset_returns = asset_returns.set_index(DATE_COL).sort_index()
        return asset_returns.reindex(columns=TICKERS).dropna(how="all")

    if not HISTORICAL_PRICES_FILE.exists():
        raise FileNotFoundError(
            "Asset return data is required for the Custom Portfolio Lab. "
            "Please include historical price data or asset_returns.csv."
        )

    historical_prices = load_price_data(str(HISTORICAL_PRICES_FILE))
    new_daily_prices = load_optional_price_data(str(NEW_DAILY_PRICES_FILE))
    combined_prices = combine_price_data(historical_prices, new_daily_prices)
    close_prices = extract_close_prices(combined_prices, tickers=TICKERS, close_col=CLOSE_COL)
    asset_returns = compute_simple_returns(close_prices).sort_index()
    missing_tickers = [ticker for ticker in TICKERS if ticker not in asset_returns.columns]
    if missing_tickers:
        raise ValueError(f"Asset returns are missing required tickers: {missing_tickers}")
    return asset_returns.reindex(columns=TICKERS).dropna(how="all")


@st.cache_data(show_spinner=False)
def load_risk_free_rate() -> pd.Series | None:
    if not RISK_FREE_RATE_FILE.exists():
        return None

    risk_free_df = standardize_daily_index(pd.read_parquet(RISK_FREE_RATE_FILE))
    if RF_RATE_DAILY_COL in risk_free_df.columns:
        risk_free_series = risk_free_df[RF_RATE_DAILY_COL]
    elif risk_free_df.shape[1] == 1:
        risk_free_series = risk_free_df.iloc[:, 0]
    else:
        return None

    risk_free_series = risk_free_series.astype(float).sort_index()
    risk_free_series.name = RF_RATE_DAILY_COL
    return risk_free_series


@st.cache_data(show_spinner=False)
def load_frozen_weights_wide() -> pd.DataFrame:
    return load_frozen_weights(str(FROZEN_WEIGHTS_FILE), tickers=TICKERS)


@st.cache_data(show_spinner=False)
def load_precomputed_outputs() -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}

    file_specs = {
        "portfolio_study_summary": (PORTFOLIO_STUDY_SUMMARY_FILE, None),
        "portfolio_construction_summary": (PORTFOLIO_CONSTRUCTION_SUMMARY_FILE, None),
        "portfolio_rolling_var_es": (PORTFOLIO_ROLLING_VAR_ES_FILE, [DATE_COL]),
        "portfolio_var_backtest_summary": (PORTFOLIO_VAR_BACKTEST_SUMMARY_FILE, None),
        "portfolio_var_backtesting_tests": (PORTFOLIO_VAR_BACKTEST_TESTS_FILE, None),
        "latest_portfolio_var_es": (LATEST_PORTFOLIO_VAR_ES_FILE, [DATE_COL]),
        "combined_portfolio_var_es": (COMBINED_VOL_MODEL_FILE, [DATE_COL]),
        "combined_vol_model_summary": (COMBINED_VOL_MODEL_SUMMARY_FILE, None),
        "vol_model_rankings": (VOL_MODEL_RANKINGS_FILE, None),
        "stress_test_results": (STRESS_TEST_RESULTS_FILE, None),
        "stress_test_asset_contributions": (STRESS_TEST_ASSET_CONTRIBUTIONS_FILE, None),
        "stress_test_summary": (STRESS_TEST_SUMMARY_FILE, None),
        "risk_driver_summary": (RISK_DRIVER_SUMMARY_FILE, None),
        "portfolio_risk_contributions": (PORTFOLIO_RISK_CONTRIBUTIONS_FILE, None),
        "drawdown_attribution_summary": (DRAWDOWN_ATTRIBUTION_FILE, [ "Peak Date", "Trough Date"]),
    }

    for key, (filepath, parse_dates) in file_specs.items():
        if filepath.exists():
            outputs[key] = _read_csv(filepath, parse_dates=parse_dates)
        else:
            outputs[key] = pd.DataFrame()

    return outputs


def format_percentage(value: float | int | np.floating | None, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def format_decimal(value: float | int | np.floating | None, decimals: int = 4) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}"


def get_weight_input_state() -> dict[str, float]:
    return {ticker: float(st.session_state.get(weight_input_key(ticker), 0.0)) for ticker in TICKERS}


def weight_input_key(ticker: str) -> str:
    return f"custom_weight_pct_{ticker}"


def initialize_weight_state() -> None:
    for ticker, value in PRESET_WEIGHTS_PCT["Equal Weight"].items():
        st.session_state.setdefault(weight_input_key(ticker), value)


def apply_weight_preset(preset_name: str) -> None:
    for ticker, value in PRESET_WEIGHTS_PCT[preset_name].items():
        st.session_state[weight_input_key(ticker)] = value


def normalize_weight_state() -> None:
    weights_pct = get_weight_input_state()
    total_weight_pct = sum(weights_pct.values())
    if total_weight_pct <= 0.0:
        return
    for ticker, value in weights_pct.items():
        st.session_state[weight_input_key(ticker)] = (value / total_weight_pct) * 100.0


def build_weight_table(weights: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "Ticker": weights.index,
        "Weight": weights.to_numpy(),
        "Weight %": (weights * 100.0).to_numpy(),
    })


def build_portfolio_returns(asset_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    aligned_weights = weights.reindex(TICKERS).astype(float)
    return asset_returns.reindex(columns=TICKERS) @ aligned_weights.to_numpy()


def compute_drawdown_series(returns: pd.Series) -> pd.Series:
    wealth_index = (1.0 + returns).cumprod()
    running_peak = wealth_index.cummax()
    return wealth_index / running_peak - 1.0


def compute_annualized_return(returns: pd.Series) -> float:
    if returns.empty:
        return np.nan
    final_wealth = float((1.0 + returns).prod())
    if final_wealth <= 0.0:
        return np.nan
    return float(final_wealth ** (ANNUALIZATION_FACTOR / len(returns)) - 1.0)


def compute_sharpe_ratio(returns: pd.Series, risk_free_rate: pd.Series | None) -> float:
    if returns.empty or returns.std(ddof=0) == 0.0:
        return np.nan
    if risk_free_rate is None:
        return np.nan
    aligned_rf = risk_free_rate.reindex(returns.index).ffill().fillna(0.0)
    excess_returns = returns - aligned_rf
    excess_volatility = float(excess_returns.std(ddof=0))
    if excess_volatility == 0.0:
        return np.nan
    return float((excess_returns.mean() / excess_volatility) * np.sqrt(ANNUALIZATION_FACTOR))


def historical_expected_shortfall(window_values: pd.Series, tail_probability: float = TAIL_PROBABILITY) -> float:
    threshold = window_values.quantile(tail_probability)
    tail_values = window_values.loc[window_values <= threshold]
    if tail_values.empty:
        return np.nan
    return float(-tail_values.mean())


def compute_historical_var_es(returns: pd.Series) -> pd.DataFrame:
    returns = returns.sort_index()
    shifted_returns = returns.shift(1)
    rolling_var = -shifted_returns.rolling(window=WINDOW_SIZE, min_periods=WINDOW_SIZE).quantile(TAIL_PROBABILITY)
    rolling_es = shifted_returns.rolling(window=WINDOW_SIZE, min_periods=WINDOW_SIZE).apply(
        historical_expected_shortfall,
        raw=False,
    )
    risk_df = pd.DataFrame({
        DATE_COL: returns.index,
        "Return": returns.to_numpy(),
        "Rolling VaR (95%, 252D)": rolling_var.to_numpy(),
        "Rolling ES (95%, 252D)": rolling_es.to_numpy(),
    })
    risk_df["VaR Breach"] = (risk_df["Return"] < -risk_df["Rolling VaR (95%, 252D)"]).where(
        risk_df["Rolling VaR (95%, 252D)"].notna()
    )
    risk_df["Loss Exceeds ES"] = (risk_df["Return"] < -risk_df["Rolling ES (95%, 252D)"]).where(
        risk_df["Rolling ES (95%, 252D)"].notna()
    )
    return risk_df


def compute_ewma_t_var_es(returns: pd.Series, nu: float = EWMA_DEFAULT_NU) -> pd.DataFrame:
    returns = returns.sort_index()
    sigma2 = pd.Series(np.nan, index=returns.index, dtype=float)
    if len(returns) <= WINDOW_SIZE:
        return pd.DataFrame({
            DATE_COL: returns.index,
            "Return": returns.to_numpy(),
            "Rolling VaR (95%, 252D)": np.nan,
            "Rolling ES (95%, 252D)": np.nan,
        })

    sigma2.iloc[WINDOW_SIZE] = float(returns.iloc[:WINDOW_SIZE].var(ddof=0))
    for index_position in range(WINDOW_SIZE + 1, len(returns)):
        sigma2.iloc[index_position] = (
            EWMA_LAMBDA * sigma2.iloc[index_position - 1] +
            (1.0 - EWMA_LAMBDA) * float(returns.iloc[index_position - 1] ** 2)
        )

    sigma = np.sqrt(sigma2)
    q_alpha = standardized_t_quantile(nu, TAIL_PROBABILITY)
    es_alpha = standardized_t_expected_shortfall(nu, TAIL_PROBABILITY)
    rolling_var = -(sigma * q_alpha)
    rolling_es = -(sigma * es_alpha)

    risk_df = pd.DataFrame({
        DATE_COL: returns.index,
        "Return": returns.to_numpy(),
        "Rolling VaR (95%, 252D)": rolling_var.to_numpy(),
        "Rolling ES (95%, 252D)": rolling_es.to_numpy(),
    })
    risk_df["VaR Breach"] = (risk_df["Return"] < -risk_df["Rolling VaR (95%, 252D)"]).where(
        risk_df["Rolling VaR (95%, 252D)"].notna()
    )
    risk_df["Loss Exceeds ES"] = (risk_df["Return"] < -risk_df["Rolling ES (95%, 252D)"]).where(
        risk_df["Rolling ES (95%, 252D)"].notna()
    )
    return risk_df


def filter_backtest_risk_df(risk_df: pd.DataFrame) -> pd.DataFrame:
    return risk_df.loc[risk_df[DATE_COL] >= pd.Timestamp(BACKTEST_START)].copy()


def summarize_portfolio_performance(
    full_returns: pd.Series,
    backtest_returns: pd.Series,
    risk_free_rate: pd.Series | None,
    risk_df_backtest: pd.DataFrame,
) -> dict[str, float]:
    valid_var_mask = risk_df_backtest["Rolling VaR (95%, 252D)"].notna()
    valid_var_df = risk_df_backtest.loc[valid_var_mask].copy()
    cumulative_return = float((1.0 + backtest_returns).prod() - 1.0) if not backtest_returns.empty else np.nan
    annualized_return = compute_annualized_return(backtest_returns)
    annualized_volatility = float(backtest_returns.std(ddof=0) * np.sqrt(ANNUALIZATION_FACTOR)) if not backtest_returns.empty else np.nan
    drawdown_series = compute_drawdown_series(backtest_returns) if not backtest_returns.empty else pd.Series(dtype=float)

    return {
        "Backtest Cumulative Return": cumulative_return,
        "Annualized Return": annualized_return,
        "Annualized Volatility": annualized_volatility,
        "Sharpe Ratio": compute_sharpe_ratio(backtest_returns, risk_free_rate),
        "Maximum Drawdown": float(drawdown_series.min()) if not drawdown_series.empty else np.nan,
        "Latest Historical VaR": float(valid_var_df["Rolling VaR (95%, 252D)"].iloc[-1]) if not valid_var_df.empty else np.nan,
        "Latest Historical ES": float(valid_var_df["Rolling ES (95%, 252D)"].iloc[-1]) if not valid_var_df.empty else np.nan,
        "Number of VaR Breaches": int(valid_var_df["VaR Breach"].fillna(False).sum()) if not valid_var_df.empty else 0,
        "Expected Number of VaR Breaches": float(TAIL_PROBABILITY * len(valid_var_df)) if not valid_var_df.empty else np.nan,
        "Breach Rate": float(valid_var_df["VaR Breach"].fillna(False).mean()) if not valid_var_df.empty else np.nan,
    }


def compute_custom_stress_results(
    weights: pd.Series,
    latest_var: float | None,
    latest_es: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stress_rows: list[dict[str, float | str]] = []
    contribution_rows: list[dict[str, float | str]] = []

    for scenario_name, scenario_shocks in SCENARIOS.items():
        shock_series = pd.Series(scenario_shocks).reindex(TICKERS).astype(float)
        asset_return_contribution = weights * shock_series
        asset_loss_contribution = -asset_return_contribution
        portfolio_stress_return = float(asset_return_contribution.sum())
        portfolio_stress_loss = float(-portfolio_stress_return)
        total_loss_for_pct = portfolio_stress_loss if portfolio_stress_loss > 0 else np.nan
        contribution_pct = asset_loss_contribution / total_loss_for_pct if pd.notna(total_loss_for_pct) else np.nan
        worst_asset = str(asset_loss_contribution.idxmax())
        worst_asset_loss = float(asset_loss_contribution.max())

        var_multiple = np.nan if latest_var is None or pd.isna(latest_var) or latest_var <= 0 else portfolio_stress_loss / latest_var
        es_multiple = np.nan if latest_es is None or pd.isna(latest_es) or latest_es <= 0 else portfolio_stress_loss / latest_es

        if pd.isna(latest_var) or pd.isna(latest_es):
            interpretation = "VaR/ES comparison unavailable"
        elif portfolio_stress_loss <= latest_var:
            interpretation = "Within latest VaR"
        elif portfolio_stress_loss <= latest_es:
            interpretation = "Exceeds VaR but within ES"
        elif pd.notna(var_multiple) and var_multiple >= 2.0:
            interpretation = "Severe stress loss relative to VaR"
        else:
            interpretation = "Exceeds ES"

        stress_rows.append({
            "Scenario": scenario_name,
            "Scenario Type": "Deterministic",
            "Scenario Horizon": "1D",
            "Portfolio Stress Return": portfolio_stress_return,
            "Portfolio Stress Loss": portfolio_stress_loss,
            "Latest VaR": latest_var,
            "Latest ES": latest_es,
            "VaR Multiple": var_multiple,
            "ES Multiple": es_multiple,
            "Worst Contributing Asset": worst_asset,
            "Worst Asset Loss Contribution": worst_asset_loss,
            "Interpretation": interpretation,
        })

        for ticker in TICKERS:
            contribution_rows.append({
                "Scenario": scenario_name,
                "Scenario Type": "Deterministic",
                "Scenario Horizon": "1D",
                "Ticker": ticker,
                "Weight": float(weights[ticker]),
                "Asset Shock Return": float(shock_series[ticker]),
                "Asset Return Contribution": float(asset_return_contribution[ticker]),
                "Asset Loss Contribution": float(asset_loss_contribution[ticker]),
                "Contribution Percentage": float(contribution_pct[ticker]) if pd.notna(total_loss_for_pct) else np.nan,
            })

    stress_results = pd.DataFrame(stress_rows)
    asset_contributions = pd.DataFrame(contribution_rows).sort_values(
        by=["Scenario", "Asset Loss Contribution"],
        ascending=[True, False],
    )
    return stress_results, asset_contributions


def compute_custom_risk_attribution(asset_returns: pd.DataFrame, weights: pd.Series) -> pd.DataFrame:
    if len(asset_returns) < WINDOW_SIZE:
        raise ValueError("At least 252 observations are required for risk attribution.")

    latest_window = asset_returns.reindex(columns=TICKERS).tail(WINDOW_SIZE)
    covariance_matrix = latest_window.cov()
    if covariance_matrix.isna().any().any():
        raise ValueError("Covariance matrix contains NaN values.")

    weight_vector = weights.reindex(TICKERS).to_numpy(dtype=float)
    portfolio_variance = float(weight_vector @ covariance_matrix.to_numpy() @ weight_vector)
    portfolio_volatility = float(np.sqrt(portfolio_variance))
    if portfolio_volatility <= 0.0:
        raise ValueError("Portfolio volatility must be positive.")

    marginal_contribution = covariance_matrix.to_numpy() @ weight_vector / portfolio_volatility
    component_contribution = weight_vector * marginal_contribution
    percentage_contribution = component_contribution / portfolio_volatility
    asset_daily_volatility = latest_window.std(ddof=0)
    asset_annualized_volatility = asset_daily_volatility * np.sqrt(ANNUALIZATION_FACTOR)

    risk_contributions = pd.DataFrame({
        "Ticker": TICKERS,
        "Weight": weight_vector,
        "Asset Daily Volatility": asset_daily_volatility.reindex(TICKERS).to_numpy(),
        "Asset Annualized Volatility": asset_annualized_volatility.reindex(TICKERS).to_numpy(),
        "Marginal Contribution to Risk": marginal_contribution,
        "Component Contribution to Risk": component_contribution,
        "Percentage Contribution to Risk": percentage_contribution,
        "Risk Contribution Minus Weight": percentage_contribution - weight_vector,
        "Weight Minus Risk Contribution": weight_vector - percentage_contribution,
        "Portfolio Daily Volatility": portfolio_volatility,
        "Portfolio Annualized Volatility": portfolio_volatility * np.sqrt(ANNUALIZATION_FACTOR),
    })
    return risk_contributions


def build_interpretation_messages(
    risk_contributions: pd.DataFrame,
    stress_results: pd.DataFrame,
) -> list[str]:
    messages: list[str] = []

    over_contributors = risk_contributions.loc[
        risk_contributions["Percentage Contribution to Risk"] > risk_contributions["Weight"] + 0.10
    ]
    if not over_contributors.empty:
        leading_row = over_contributors.sort_values("Risk Contribution Minus Weight", ascending=False).iloc[0]
        messages.append(
            f"{leading_row['Ticker']} contributes materially more risk than its portfolio weight, "
            f"with {leading_row['Percentage Contribution to Risk'] * 100:.1f}% of risk against "
            f"{leading_row['Weight'] * 100:.1f}% weight."
        )

    nvda_stress = stress_results.loc[stress_results["Worst Contributing Asset"] == "NVDA"]
    if not nvda_stress.empty:
        messages.append("NVDA dominates the worst asset contribution in at least one stress scenario.")

    severe_scenarios = stress_results.loc[stress_results["Portfolio Stress Loss"] > stress_results["Latest ES"]]
    if not severe_scenarios.empty:
        severe_names = ", ".join(severe_scenarios["Scenario"].tolist())
        messages.append(f"{severe_names} exceed the latest expected shortfall, indicating severe scenario risk.")

    if not messages:
        messages.append("Risk is broadly aligned with weights and no single stress scenario dominates the profile.")

    return messages


def build_fixed_portfolio_reference(
    asset_returns: pd.DataFrame,
    risk_free_rate: pd.Series | None,
) -> pd.DataFrame:
    frozen_weights = load_frozen_weights_wide()
    rows: list[dict[str, float | str]] = []

    for portfolio_name, weight_row in frozen_weights.iterrows():
        portfolio_returns = build_portfolio_returns(asset_returns, weight_row)
        backtest_returns = portfolio_returns.loc[portfolio_returns.index >= pd.Timestamp(BACKTEST_START)]
        risk_df_full = compute_historical_var_es(portfolio_returns)
        risk_df_backtest = filter_backtest_risk_df(risk_df_full)
        metrics = summarize_portfolio_performance(
            full_returns=portfolio_returns,
            backtest_returns=backtest_returns,
            risk_free_rate=risk_free_rate,
            risk_df_backtest=risk_df_backtest,
        )
        stress_results, _ = compute_custom_stress_results(
            weight_row.astype(float),
            latest_var=metrics["Latest Historical VaR"],
            latest_es=metrics["Latest Historical ES"],
        )
        stress_loss_map = dict(zip(stress_results["Scenario"], stress_results["Portfolio Stress Loss"]))

        rows.append({
            "Portfolio": portfolio_name,
            "Latest VaR": metrics["Latest Historical VaR"],
            "Latest ES": metrics["Latest Historical ES"],
            "Annualized Volatility": metrics["Annualized Volatility"],
            "Max Drawdown": metrics["Maximum Drawdown"],
            **stress_loss_map,
        })

    return pd.DataFrame(rows)


def make_cumulative_return_chart(returns: pd.Series, title: str) -> go.Figure:
    cumulative_returns = (1.0 + returns).cumprod() - 1.0
    chart_df = pd.DataFrame({DATE_COL: cumulative_returns.index, "Cumulative Return": cumulative_returns.to_numpy()})
    figure = px.line(chart_df, x=DATE_COL, y="Cumulative Return", title=title)
    figure.update_layout(yaxis_tickformat=".1%")
    return figure


def make_returns_vs_var_chart(
    risk_df: pd.DataFrame,
    title: str,
    ewma_df: pd.DataFrame | None = None,
) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(go.Scatter(
        x=risk_df[DATE_COL],
        y=risk_df["Return"],
        mode="lines",
        name="Actual Return",
        line=dict(color="#1f77b4"),
    ))
    figure.add_trace(go.Scatter(
        x=risk_df[DATE_COL],
        y=-risk_df["Rolling VaR (95%, 252D)"],
        mode="lines",
        name="Historical VaR Threshold",
        line=dict(color="#d62728"),
    ))

    breach_df = risk_df.loc[risk_df["VaR Breach"].fillna(False)].copy()
    if not breach_df.empty:
        figure.add_trace(go.Scatter(
            x=breach_df[DATE_COL],
            y=breach_df["Return"],
            mode="markers",
            name="VaR Breach",
            marker=dict(color="#ff7f0e", size=8),
        ))

    if ewma_df is not None and not ewma_df.empty:
        figure.add_trace(go.Scatter(
            x=ewma_df[DATE_COL],
            y=-ewma_df["Rolling VaR (95%, 252D)"],
            mode="lines",
            name="EWMA-t VaR Threshold",
            line=dict(color="#2ca02c", dash="dash"),
        ))

    figure.update_layout(title=title, yaxis_tickformat=".1%")
    return figure


def make_weight_vs_risk_chart(risk_contributions: pd.DataFrame, title: str) -> go.Figure:
    chart_df = risk_contributions.loc[:, ["Ticker", "Weight", "Percentage Contribution to Risk"]].melt(
        id_vars="Ticker",
        var_name="Metric",
        value_name="Value",
    )
    figure = px.bar(chart_df, x="Ticker", y="Value", color="Metric", barmode="group", title=title)
    figure.update_layout(yaxis_tickformat=".1%")
    return figure


def make_stress_loss_chart(stress_results: pd.DataFrame, title: str) -> go.Figure:
    figure = px.bar(
        stress_results,
        x="Scenario",
        y="Portfolio Stress Loss",
        color="Scenario",
        title=title,
    )
    figure.update_layout(showlegend=False, yaxis_tickformat=".1%")
    return figure


def make_asset_contribution_chart(contribution_df: pd.DataFrame, title: str) -> go.Figure:
    figure = px.bar(
        contribution_df,
        x="Ticker",
        y="Asset Loss Contribution",
        color="Ticker",
        title=title,
    )
    figure.update_layout(showlegend=False, yaxis_tickformat=".1%")
    return figure


def make_fixed_weights_chart(weights_wide: pd.DataFrame, title: str) -> go.Figure:
    chart_df = weights_wide.reset_index().melt(id_vars="Portfolio", var_name="Ticker", value_name="Weight")
    figure = px.bar(chart_df, x="Ticker", y="Weight", color="Portfolio", barmode="group", title=title)
    figure.update_layout(yaxis_tickformat=".1%")
    return figure


def make_model_comparison_chart(model_df: pd.DataFrame, portfolio_name: str) -> go.Figure:
    filtered_df = model_df.loc[model_df["Portfolio"] == portfolio_name].copy()
    figure = go.Figure()
    for model_name, model_data in filtered_df.groupby("Model"):
        figure.add_trace(go.Scatter(
            x=model_data[DATE_COL],
            y=-model_data["Rolling VaR (95%, 252D)"],
            mode="lines",
            name=f"{model_name} Threshold",
        ))
    if not filtered_df.empty:
        actual_returns = filtered_df.loc[filtered_df["Model"] == filtered_df["Model"].iloc[0], [DATE_COL, "Return"]]
        figure.add_trace(go.Scatter(
            x=actual_returns[DATE_COL],
            y=actual_returns["Return"],
            mode="lines",
            name="Actual Return",
            line=dict(color="black"),
        ))
    figure.update_layout(
        title=f"VaR Thresholds by Model: {portfolio_name}",
        yaxis_tickformat=".1%",
    )
    return figure
