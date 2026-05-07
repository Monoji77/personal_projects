from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_extras.card_selector import card_selector
from streamlit_extras.mention import mention
from streamlit_extras.specialized_inputs import specialized_text_input

try:
    from st_on_hover_tabs import on_hover_tabs

    HOVER_TABS_IMPORT_ERROR: Exception | None = None
except Exception as error:
    on_hover_tabs = None
    HOVER_TABS_IMPORT_ERROR = error


SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from dashboard_utils import (
    ANNUALIZATION_FACTOR,
    BACKTEST_START,
    COMBINED_VOL_MODEL_FILE,
    CONFIDENCE_LEVEL,
    DATE_COL,
    FIXED_PORTFOLIOS,
    SCENARIOS,
    TICKERS,
    build_interpretation_messages,
    build_portfolio_returns,
    compute_custom_risk_attribution,
    compute_custom_stress_results,
    compute_drawdown_series,
    compute_ewma_t_var_es,
    compute_historical_var_es,
    compute_sharpe_ratio,
    filter_backtest_risk_df,
    format_decimal,
    format_percentage,
    load_asset_returns,
    load_frozen_weights_wide,
    load_precomputed_outputs,
    load_risk_free_rate,
    make_asset_contribution_chart,
    make_cumulative_return_chart,
    make_fixed_weights_chart,
    make_model_comparison_chart,
    make_returns_vs_var_chart,
    make_stress_loss_chart,
    make_weight_vs_risk_chart,
    summarize_portfolio_performance,
)


st.set_page_config(
    page_title="Market Risk Engine",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded",
)


PAGE_CONFIG = {
    "portfolio_risk_lab": {"label": "Portfolio Risk Lab", "icon": "science"},
    "overview": {"label": "Overview", "icon": "dashboard"},
    "risk_models": {"label": "Risk Models & Backtesting", "icon": "query_stats"},
    "stress_attr": {"label": "Stress & Attribution", "icon": "crisis_alert"},
    "methodology": {"label": "Methodology", "icon": "menu_book"},
}

PRESET_ORDER = [
    "Equal Weight",
    "100% AAPL",
    "100% NVDA",
    "Tech Balanced",
    "High NVDA Exposure",
    "Long-only Global Minimum Variance Portfolio",
    "Long-only Tangency Portfolio",
]

PRESET_META = {
    "Equal Weight": {
        "icon": ":material/view_comfy_alt:",
        "description": "Balanced allocation across the five-asset technology basket.",
    },
    "100% AAPL": {
        "icon": ":material/phone_iphone:",
        "description": "Single-name Apple exposure for concentrated risk inspection.",
    },
    "100% NVDA": {
        "icon": ":material/memory:",
        "description": "Single-name NVIDIA exposure with the strongest tail concentration.",
    },
    "Tech Balanced": {
        "icon": ":material/balance:",
        "description": "A hand-tuned mix intended to stay diversified without matching equal weight exactly.",
    },
    "High NVDA Exposure": {
        "icon": ":material/rocket_launch:",
        "description": "Growth-oriented tilt with heavy NVIDIA concentration.",
    },
    "Long-only Global Minimum Variance Portfolio": {
        "icon": ":material/shield:",
        "description": "Frozen low-variance allocation estimated offline from the training sample.",
    },
    "Long-only Tangency Portfolio": {
        "icon": ":material/trending_up:",
        "description": "Frozen maximum-Sharpe long-only allocation estimated offline from the training sample.",
    },
}

REPO_URL = "https://github.com/Monoji77/personal_projects/tree/project1/market-risk-engine"
ACTIVE_PAGE_KEY = "active_page_key"
PRESET_SELECTION_KEY = "preset_selection_key"
PRESET_CARD_KEY = "preset_card_selector"
PLOTLY_HEIGHT = 420


def inject_app_styles() -> None:
    st.markdown(
        """
        <style>
            section.main > div.block-container {
                padding-top: 2rem;
                padding-bottom: 3rem;
                max-width: 1400px;
            }
            .section-caption {
                color: var(--st-secondary-text-color);
                font-size: 0.95rem;
                line-height: 1.5;
            }
            .section-spacer {
                margin-top: 0.5rem;
            }
            [data-testid="stSidebar"] {
                min-width: 270px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_required_context() -> tuple[pd.DataFrame | None, pd.Series | None, pd.DataFrame | None, dict[str, pd.DataFrame]]:
    outputs = load_precomputed_outputs()

    try:
        asset_returns = load_asset_returns()
    except Exception:
        asset_returns = None

    try:
        risk_free_rate = load_risk_free_rate()
    except Exception:
        risk_free_rate = None

    try:
        frozen_weights = load_frozen_weights_wide()
    except Exception:
        frozen_weights = None

    return asset_returns, risk_free_rate, frozen_weights, outputs


def get_navigation_styles() -> dict[str, dict[str, str | dict[str, str]]]:
    return {
        "navtab": {
            "background-color": "transparent",
            "color": "#94A3B8",
            "font-size": "15px",
            "transition": ".2s",
            "white-space": "nowrap",
        },
        "tabStyle": {
            "list-style-type": "none",
            "margin-bottom": "10px",
            "padding-left": "14px",
            "padding-top": "10px",
            "padding-bottom": "10px",
            "border-radius": "12px",
        },
        "iconStyle": {
            "position": "fixed",
            "left": "12px",
            "text-align": "left",
        },
        "labelName": {
            "font-weight": "500",
        },
    }


def render_sidebar_navigation() -> str:
    st.session_state.setdefault(ACTIVE_PAGE_KEY, "portfolio_risk_lab")
    page_keys = list(PAGE_CONFIG.keys())
    page_labels = [PAGE_CONFIG[page_key]["label"] for page_key in page_keys]
    page_icons = [PAGE_CONFIG[page_key]["icon"] for page_key in page_keys]
    current_page_key = st.session_state[ACTIVE_PAGE_KEY]
    default_index = page_keys.index(current_page_key)

    with st.sidebar:
        st.markdown("## Market Risk Engine")
        st.caption("Navigation")

        if on_hover_tabs is not None:
            selected_label = on_hover_tabs(
                tabName=page_labels,
                iconName=page_icons,
                default_choice=default_index,
                styles=get_navigation_styles(),
                key="hover_tabs_navigation",
            )
        else:
            st.warning(
                "The `streamlit-on-hover-tabs` dependency is not available in this runtime. "
                "Falling back to standard navigation."
            )
            selected_label = st.radio(
                "Navigation",
                options=page_labels,
                index=default_index,
                label_visibility="collapsed",
            )

        if HOVER_TABS_IMPORT_ERROR is not None and on_hover_tabs is None:
            st.caption(f"Import detail: {HOVER_TABS_IMPORT_ERROR}")

    label_to_key = {page_meta["label"]: page_key for page_key, page_meta in PAGE_CONFIG.items()}
    selected_page_key = label_to_key.get(selected_label, current_page_key)
    st.session_state[ACTIVE_PAGE_KEY] = selected_page_key
    return selected_page_key


def get_plotly_template() -> str:
    theme_base = st.get_option("theme.base")
    return "plotly_dark" if theme_base == "dark" else "plotly_white"


def style_figure(fig: go.Figure, *, height: int = PLOTLY_HEIGHT, percent_y: bool = False) -> go.Figure:
    fig.update_layout(
        template=get_plotly_template(),
        height=height,
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=14, r=14, t=64, b=18),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(128, 128, 128, 0.15)")
    if percent_y:
        fig.update_yaxes(tickformat=".1%")
    return fig


def render_metric_cards(metric_items: list[tuple[str, float | str | None, str]]) -> None:
    columns = st.columns(len(metric_items))
    for column, (label, value, metric_type) in zip(columns, metric_items):
        if metric_type == "pct":
            display_value = format_percentage(value)
        elif metric_type == "decimal":
            display_value = format_decimal(value)
        elif metric_type == "count":
            display_value = "N/A" if value is None or pd.isna(value) else f"{int(value)}"
        else:
            display_value = "N/A" if value is None or pd.isna(value) else str(value)
        with column.container(border=True):
            st.metric(label, display_value)


def find_metric_column(df: pd.DataFrame, prefix: str) -> str | None:
    matching_columns = [column for column in df.columns if column.startswith(prefix)]
    if len(matching_columns) == 1:
        return matching_columns[0]
    return None


def build_asset_universe_summary(
    asset_returns: pd.DataFrame,
    risk_free_rate: pd.Series | None,
) -> pd.DataFrame:
    latest_window_returns = asset_returns.reindex(columns=TICKERS).dropna(how="all").tail(ANNUALIZATION_FACTOR)
    latest_available_date = asset_returns.index.max().date().isoformat()
    summary_rows: list[dict[str, float | str]] = []

    for ticker in TICKERS:
        ticker_full_returns = asset_returns[ticker].dropna()
        ticker_window_returns = latest_window_returns[ticker].dropna()
        risk_df = compute_historical_var_es(ticker_full_returns)
        valid_risk_df = risk_df.loc[risk_df["Rolling VaR (95%, 252D)"].notna()].copy()

        annualized_volatility = np.nan
        worst_1d_return = np.nan
        sharpe_ratio = np.nan
        if not ticker_window_returns.empty:
            annualized_volatility = float(ticker_window_returns.std(ddof=1) * np.sqrt(ANNUALIZATION_FACTOR))
            worst_1d_return = float(ticker_window_returns.min())
            sharpe_ratio = compute_sharpe_ratio(ticker_window_returns, risk_free_rate)

        summary_rows.append({
            "Ticker": ticker,
            "Latest Available Date": latest_available_date,
            "Trailing 252D Annualized Volatility": annualized_volatility,
            "Latest Historical VaR (95%)": (
                float(valid_risk_df["Rolling VaR (95%, 252D)"].iloc[-1]) if not valid_risk_df.empty else np.nan
            ),
            "Latest Historical ES (95%)": (
                float(valid_risk_df["Rolling ES (95%, 252D)"].iloc[-1]) if not valid_risk_df.empty else np.nan
            ),
            "Worst 1D Return (252D)": worst_1d_return,
            "Sharpe Ratio": sharpe_ratio,
        })

    return pd.DataFrame(summary_rows)


def build_preset_weight_map(frozen_weights: pd.DataFrame | None) -> dict[str, dict[str, float]]:
    preset_weight_map: dict[str, dict[str, float]] = {
        "Equal Weight": {"AAPL": 20.0, "AMZN": 20.0, "GOOG": 20.0, "MSFT": 20.0, "NVDA": 20.0},
        "100% AAPL": {"AAPL": 100.0, "AMZN": 0.0, "GOOG": 0.0, "MSFT": 0.0, "NVDA": 0.0},
        "100% NVDA": {"AAPL": 0.0, "AMZN": 0.0, "GOOG": 0.0, "MSFT": 0.0, "NVDA": 100.0},
        "Tech Balanced": {"AAPL": 25.0, "AMZN": 15.0, "GOOG": 20.0, "MSFT": 25.0, "NVDA": 15.0},
        "High NVDA Exposure": {"AAPL": 15.0, "AMZN": 10.0, "GOOG": 10.0, "MSFT": 15.0, "NVDA": 50.0},
    }

    if frozen_weights is not None and not frozen_weights.empty:
        if "Long-only Global Minimum Variance Portfolio" in frozen_weights.index:
            preset_weight_map["Long-only Global Minimum Variance Portfolio"] = (
                frozen_weights.loc["Long-only Global Minimum Variance Portfolio"]
                .reindex(TICKERS)
                .astype(float)
                .mul(100.0)
                .to_dict()
            )
        if "Long-only Tangency Portfolio" in frozen_weights.index:
            preset_weight_map["Long-only Tangency Portfolio"] = (
                frozen_weights.loc["Long-only Tangency Portfolio"]
                .reindex(TICKERS)
                .astype(float)
                .mul(100.0)
                .to_dict()
            )

    return {
        preset_name: preset_weight_map[preset_name]
        for preset_name in PRESET_ORDER
        if preset_name in preset_weight_map
    }


def weight_input_key(ticker: str) -> str:
    return f"portfolio_weight_input_{ticker}"


def format_weight_value(weight_pct: float) -> str:
    return f"{weight_pct:.1f}".rstrip("0").rstrip(".")


def set_weight_state_from_preset(preset_name: str, preset_weight_map: dict[str, dict[str, float]]) -> None:
    st.session_state[PRESET_SELECTION_KEY] = preset_name
    for ticker, weight_pct in preset_weight_map[preset_name].items():
        st.session_state[weight_input_key(ticker)] = format_weight_value(float(weight_pct))


def initialize_weight_state(preset_weight_map: dict[str, dict[str, float]]) -> None:
    if PRESET_SELECTION_KEY not in st.session_state:
        default_preset = "Equal Weight" if "Equal Weight" in preset_weight_map else next(iter(preset_weight_map))
        set_weight_state_from_preset(default_preset, preset_weight_map)
    else:
        for ticker in TICKERS:
            st.session_state.setdefault(weight_input_key(ticker), "0")


def parse_weight_input_value(raw_value: str) -> tuple[float | None, str | None]:
    cleaned_value = raw_value.strip().replace("%", "").replace(",", "")
    if cleaned_value == "":
        return 0.0, None
    try:
        parsed_value = float(cleaned_value)
    except ValueError:
        return None, "Enter a numeric percentage."
    if parsed_value < 0.0:
        return None, "Short selling is not allowed."
    if parsed_value > 100.0:
        return None, "Each weight must stay between 0% and 100%."
    return parsed_value, None


def get_current_weight_state() -> dict[str, str]:
    return {ticker: str(st.session_state.get(weight_input_key(ticker), "0")) for ticker in TICKERS}


def validate_weight_inputs() -> dict[str, object]:
    raw_values = get_current_weight_state()
    parsed_values: dict[str, float] = {}
    input_errors: dict[str, str] = {}

    for ticker, raw_value in raw_values.items():
        parsed_value, error_message = parse_weight_input_value(raw_value)
        if error_message is not None or parsed_value is None:
            input_errors[ticker] = error_message or "Invalid weight."
        else:
            parsed_values[ticker] = parsed_value

    total_weight_pct = float(sum(parsed_values.values()))
    weights_series = pd.Series(parsed_values, dtype=float).reindex(TICKERS) / 100.0
    weights_valid = (not input_errors) and np.isclose(total_weight_pct, 100.0, atol=1e-8) and total_weight_pct > 0.0

    return {
        "raw_values": raw_values,
        "parsed_values": parsed_values,
        "input_errors": input_errors,
        "total_weight_pct": total_weight_pct,
        "weights_series": weights_series,
        "weights_valid": weights_valid,
    }


def normalize_weight_state(parsed_values: dict[str, float]) -> None:
    total_weight_pct = float(sum(parsed_values.values()))
    if total_weight_pct <= 0.0:
        return
    for ticker in TICKERS:
        normalized_weight_pct = parsed_values.get(ticker, 0.0) / total_weight_pct * 100.0
        st.session_state[weight_input_key(ticker)] = format_weight_value(normalized_weight_pct)


def compute_latest_ewma_metrics(ewma_risk_backtest: pd.DataFrame) -> tuple[float | None, float | None]:
    if ewma_risk_backtest.empty:
        return None, None
    valid_ewma = ewma_risk_backtest.loc[ewma_risk_backtest["Rolling VaR (95%, 252D)"].notna()].copy()
    if valid_ewma.empty:
        return None, None
    return (
        float(valid_ewma["Rolling VaR (95%, 252D)"].iloc[-1]),
        float(valid_ewma["Rolling ES (95%, 252D)"].iloc[-1]),
    )


def build_custom_portfolio_snapshot(
    asset_returns: pd.DataFrame,
    risk_free_rate: pd.Series | None,
    weights: pd.Series,
) -> dict[str, object]:
    custom_returns_full = build_portfolio_returns(asset_returns, weights)
    custom_returns_backtest = custom_returns_full.loc[custom_returns_full.index >= pd.Timestamp(BACKTEST_START)]
    historical_risk_full = compute_historical_var_es(custom_returns_full)
    historical_risk_backtest = filter_backtest_risk_df(historical_risk_full)
    ewma_risk_full = compute_ewma_t_var_es(custom_returns_full)
    ewma_risk_backtest = filter_backtest_risk_df(ewma_risk_full)
    risk_contributions = compute_custom_risk_attribution(asset_returns, weights)
    performance_metrics = summarize_portfolio_performance(
        full_returns=custom_returns_full,
        backtest_returns=custom_returns_backtest,
        risk_free_rate=risk_free_rate,
        risk_df_backtest=historical_risk_backtest,
    )
    stress_results, stress_contributions = compute_custom_stress_results(
        weights=weights,
        latest_var=performance_metrics["Latest Historical VaR"],
        latest_es=performance_metrics["Latest Historical ES"],
    )
    ewma_var, ewma_es = compute_latest_ewma_metrics(ewma_risk_backtest)

    return {
        "custom_returns_full": custom_returns_full,
        "custom_returns_backtest": custom_returns_backtest,
        "historical_risk_backtest": historical_risk_backtest,
        "ewma_risk_backtest": ewma_risk_backtest,
        "risk_contributions": risk_contributions,
        "stress_results": stress_results,
        "stress_contributions": stress_contributions,
        "performance_metrics": performance_metrics,
        "latest_ewma_var": ewma_var,
        "latest_ewma_es": ewma_es,
    }


def render_asset_universe_cards(asset_summary: pd.DataFrame) -> None:
    asset_records = asset_summary.to_dict("records")
    columns = st.columns(len(asset_records))
    for column, asset_row in zip(columns, asset_records):
        with column.container(border=True):
            st.markdown(f"**{asset_row['Ticker']}**")
            st.caption(f"As of {asset_row['Latest Available Date']}")
            st.write(f"Volatility: {format_percentage(asset_row['Trailing 252D Annualized Volatility'])}")
            st.write(f"VaR (95%): {format_percentage(asset_row['Latest Historical VaR (95%)'])}")
            st.write(f"ES (95%): {format_percentage(asset_row['Latest Historical ES (95%)'])}")
            st.write(f"Worst 1D: {format_percentage(asset_row['Worst 1D Return (252D)'])}")


def render_portfolio_risk_lab(
    asset_returns: pd.DataFrame | None,
    risk_free_rate: pd.Series | None,
    frozen_weights: pd.DataFrame | None,
) -> None:
    st.title("Portfolio Risk Lab")
    st.caption(
        "Choose a starting portfolio or manually edit a long-only technology allocation and watch return, volatility, "
        "VaR, expected shortfall, stress loss, and risk concentration update live."
    )
    st.info(
        "Assets are selected from the fixed technology universe used across the project. "
        "Weights must be long-only and sum to 100%. Historical VaR/ES and EWMA-t VaR/ES are recomputed dynamically "
        "from the selected weights. GARCH-t remains precomputed for the predefined fixed portfolios only."
    )

    if asset_returns is None or asset_returns.empty:
        st.error(
            "Asset return data is required for the Portfolio Risk Lab. Please include historical price data or asset_returns.csv."
        )
        return

    preset_weight_map = build_preset_weight_map(frozen_weights)
    initialize_weight_state(preset_weight_map)
    weight_state = validate_weight_inputs()
    current_weights = weight_state["weights_series"]

    snapshot_results: dict[str, object] | None = None
    snapshot_error: str | None = None
    if weight_state["weights_valid"]:
        try:
            snapshot_results = build_custom_portfolio_snapshot(asset_returns, risk_free_rate, current_weights)
        except Exception as error:
            snapshot_error = str(error)

    st.subheader("Live Portfolio Risk Snapshot")
    with st.container(border=True):
        if snapshot_error is not None:
            st.error(snapshot_error)
        elif snapshot_results is None:
            st.warning("Weights must be valid and sum to 100% before live portfolio metrics can update.")
        else:
            performance_metrics = snapshot_results["performance_metrics"]
            risk_contributions = snapshot_results["risk_contributions"]
            stress_results = snapshot_results["stress_results"]
            latest_ewma_var = snapshot_results["latest_ewma_var"]
            latest_ewma_es = snapshot_results["latest_ewma_es"]

            top_risk_driver = risk_contributions.sort_values("Percentage Contribution to Risk", ascending=False).iloc[0]
            worst_stress_scenario = stress_results.sort_values("Portfolio Stress Loss", ascending=False).iloc[0]

            render_metric_cards([
                ("Backtest Cumulative Return", performance_metrics["Backtest Cumulative Return"], "pct"),
                ("Annualized Return", performance_metrics["Annualized Return"], "pct"),
                ("Annualized Volatility", performance_metrics["Annualized Volatility"], "pct"),
                ("Sharpe Ratio", performance_metrics["Sharpe Ratio"], "decimal"),
            ])
            render_metric_cards([
                ("Historical VaR", performance_metrics["Latest Historical VaR"], "pct"),
                ("Historical ES", performance_metrics["Latest Historical ES"], "pct"),
                ("EWMA-t VaR", latest_ewma_var, "pct"),
                ("EWMA-t ES", latest_ewma_es, "pct"),
            ])
            render_metric_cards([
                ("Maximum Drawdown", performance_metrics["Maximum Drawdown"], "pct"),
                ("Worst Stress Loss", float(worst_stress_scenario["Portfolio Stress Loss"]), "pct"),
                ("Worst Stress Scenario", str(worst_stress_scenario["Scenario"]), "text"),
                (
                    "Highest Risk Contributor",
                    f"{top_risk_driver['Ticker']} ({format_percentage(top_risk_driver['Percentage Contribution to Risk'])})",
                    "text",
                ),
            ])

            if pd.isna(latest_ewma_var) or pd.isna(latest_ewma_es):
                st.caption("EWMA-t is being computed every run, but the full 252-day warm-up window is not yet available.")

    st.subheader("Weight Builder")
    with st.container(border=True):
        preset_items = [
            {
                "icon": PRESET_META[preset_name]["icon"],
                "title": preset_name,
                "description": PRESET_META[preset_name]["description"],
            }
            for preset_name in preset_weight_map
        ]
        default_preset_name = st.session_state.get(PRESET_SELECTION_KEY, next(iter(preset_weight_map)))
        default_index = list(preset_weight_map.keys()).index(default_preset_name)
        selected_preset_index = card_selector(
            preset_items,
            selection_mode="single",
            default=default_index,
            key=PRESET_CARD_KEY,
        )
        if selected_preset_index is not None:
            selected_preset_name = list(preset_weight_map.keys())[selected_preset_index]
            if selected_preset_name != st.session_state.get(PRESET_SELECTION_KEY):
                set_weight_state_from_preset(selected_preset_name, preset_weight_map)
                st.rerun()

        st.caption(
            "Preset cards populate the current portfolio. You can still adjust any ticker weight manually afterwards."
        )

        input_columns = st.columns(len(TICKERS))
        for column, ticker in zip(input_columns, TICKERS):
            with column:
                specialized_text_input(
                    label=f"{ticker} weight",
                    value=st.session_state.get(weight_input_key(ticker), "0"),
                    key=weight_input_key(ticker),
                    prefix="%",
                    placeholder="0.0",
                    help="Enter a long-only portfolio weight between 0 and 100.",
                    error=weight_state["input_errors"].get(ticker),
                )

        status_col, normalize_col = st.columns([1.8, 1.0])
        with status_col:
            st.markdown(f"**Current total weight:** {weight_state['total_weight_pct']:.2f}%")
        with normalize_col:
            if st.button("Normalize Weights to 100%", use_container_width=True):
                if weight_state["input_errors"]:
                    st.warning("Fix invalid weight entries before normalization.")
                elif weight_state["total_weight_pct"] <= 0.0:
                    st.warning("Enter at least one positive weight before normalization.")
                else:
                    normalize_weight_state(weight_state["parsed_values"])
                    st.rerun()

        if weight_state["total_weight_pct"] <= 0.0:
            st.error("All weights are zero. Enter at least one positive allocation to continue.")
        elif not weight_state["input_errors"] and np.isclose(weight_state["total_weight_pct"], 100.0, atol=1e-8):
            st.success("Weights are valid. The portfolio is long-only and fully invested.")
        elif not weight_state["input_errors"]:
            st.warning("Weights are valid individually, but the portfolio must sum to exactly 100% before live metrics update.")

    if snapshot_results is not None:
        custom_returns_backtest = snapshot_results["custom_returns_backtest"]
        historical_risk_backtest = snapshot_results["historical_risk_backtest"]
        ewma_risk_backtest = snapshot_results["ewma_risk_backtest"]
        risk_contributions = snapshot_results["risk_contributions"]
        stress_results = snapshot_results["stress_results"]
        stress_contributions = snapshot_results["stress_contributions"]

        charts_left, charts_right = st.columns(2)
        with charts_left:
            cumulative_chart = make_cumulative_return_chart(
                custom_returns_backtest,
                f"Custom Portfolio Cumulative Return Since {BACKTEST_START}",
            )
            st.plotly_chart(style_figure(cumulative_chart, percent_y=True), use_container_width=True)
        with charts_right:
            risk_contribution_chart = make_weight_vs_risk_chart(
                risk_contributions,
                "Weight vs Percentage Risk Contribution",
            )
            st.plotly_chart(style_figure(risk_contribution_chart, percent_y=True), use_container_width=True)

        charts_left, charts_right = st.columns(2)
        with charts_left:
            stress_chart = make_stress_loss_chart(stress_results, "Stress Loss by Scenario")
            st.plotly_chart(style_figure(stress_chart, percent_y=True), use_container_width=True)
        with charts_right:
            var_chart = make_returns_vs_var_chart(
                historical_risk_backtest,
                "Historical VaR Backtest with EWMA-t Overlay",
                ewma_df=ewma_risk_backtest,
            )
            st.plotly_chart(style_figure(var_chart, percent_y=True), use_container_width=True)

        detail_left, detail_right = st.columns([1.1, 1.4])
        with detail_left:
            selected_scenario = st.selectbox(
                "Inspect stress contributions",
                options=list(SCENARIOS.keys()),
                key="portfolio_lab_scenario_select",
            )
            selected_contributions = stress_contributions.loc[
                stress_contributions["Scenario"] == selected_scenario
            ].copy()
            contribution_chart = make_asset_contribution_chart(
                selected_contributions,
                f"Asset Loss Contributions | {selected_scenario}",
            )
            st.plotly_chart(style_figure(contribution_chart, height=380, percent_y=True), use_container_width=True)
        with detail_right:
            st.subheader("Interpretation")
            for message in build_interpretation_messages(risk_contributions, stress_results):
                st.write(f"- {message}")
            st.caption(
                "GARCH-t is intentionally not refit live for the custom portfolio because it would make the employer-facing "
                "app slower and less stable. GARCH-t results remain available only for the predefined fixed portfolios."
            )

    st.divider()
    st.subheader("Asset Universe Snapshot")
    st.markdown(
        "<p class='section-caption'>Risk-first view of the five-asset technology universe using trailing annualized "
        "volatility and the latest historical VaR/ES estimates derived from daily return history.</p>",
        unsafe_allow_html=True,
    )
    asset_summary = build_asset_universe_summary(asset_returns, risk_free_rate)
    render_asset_universe_cards(asset_summary)

    st.divider()
    mention(
        label="Market Risk Engine repository",
        icon="github",
        url=REPO_URL,
    )


def render_overview(
    asset_returns: pd.DataFrame | None,
    frozen_weights: pd.DataFrame | None,
    outputs: dict[str, pd.DataFrame],
) -> None:
    st.title("Overview")
    st.caption(
        "High-level summary of the fixed portfolio construction, current historical tail-risk levels, and where the "
        "predefined portfolios currently sit in the broader market risk stack."
    )

    latest_date = asset_returns.index.max().date().isoformat() if asset_returns is not None and not asset_returns.empty else "N/A"
    render_metric_cards([
        ("Assets", len(TICKERS), "count"),
        ("Fixed Portfolios", len(FIXED_PORTFOLIOS), "count"),
        ("Backtest Start", BACKTEST_START, "text"),
        ("Latest Data Date", latest_date, "text"),
    ])

    overview_left, overview_right = st.columns(2)
    with overview_left:
        with st.container(border=True):
            st.subheader("Frozen Portfolio Weights")
            if frozen_weights is None or frozen_weights.empty:
                st.warning("Frozen portfolio weights are not available.")
            else:
                frozen_weight_chart = make_fixed_weights_chart(
                    frozen_weights,
                    "Frozen Weights Across Fixed Portfolios",
                )
                st.plotly_chart(style_figure(frozen_weight_chart, percent_y=True), use_container_width=True)
    with overview_right:
        with st.container(border=True):
            st.subheader("Latest Historical Tail Risk")
            latest_portfolio_var_es = outputs.get("latest_portfolio_var_es", pd.DataFrame())
            if latest_portfolio_var_es.empty:
                st.warning("Latest portfolio VaR/ES summary is not available.")
            else:
                var_col = find_metric_column(latest_portfolio_var_es, "Rolling VaR")
                es_col = find_metric_column(latest_portfolio_var_es, "Rolling ES")
                portfolio_col = "Name" if "Name" in latest_portfolio_var_es.columns else "Portfolio"
                chart_df = latest_portfolio_var_es.loc[:, [portfolio_col, var_col, es_col]].rename(columns={portfolio_col: "Portfolio"})
                chart_df = chart_df.melt(id_vars="Portfolio", var_name="Metric", value_name="Value")
                chart_df["Metric"] = chart_df["Metric"].replace({
                    var_col: "Latest Historical VaR",
                    es_col: "Latest Historical ES",
                })
                latest_risk_chart = px.bar(
                    chart_df,
                    x="Portfolio",
                    y="Value",
                    color="Metric",
                    barmode="group",
                    title="Historical VaR and ES by Fixed Portfolio",
                )
                st.plotly_chart(style_figure(latest_risk_chart, percent_y=True), use_container_width=True)

    detail_left, detail_right = st.columns(2)
    with detail_left:
        with st.container(border=True):
            st.subheader("Best Model by Fixed Portfolio")
            model_rankings = outputs.get("vol_model_rankings", pd.DataFrame())
            if model_rankings.empty:
                st.info("Volatility model rankings are not available.")
            else:
                st.dataframe(
                    model_rankings.loc[model_rankings["Rank"] == 1].reset_index(drop=True),
                    hide_index=True,
                    use_container_width=True,
                )
    with detail_right:
        with st.container(border=True):
            st.subheader("Risk Driver Summary")
            risk_driver_summary = outputs.get("risk_driver_summary", pd.DataFrame())
            if risk_driver_summary.empty:
                st.info("Risk-driver summary is not available.")
            else:
                st.dataframe(risk_driver_summary, hide_index=True, use_container_width=True)


def render_risk_models_and_backtesting(outputs: dict[str, pd.DataFrame]) -> None:
    st.title("Risk Models & Backtesting")
    st.caption(
        "Historical VaR remains the baseline. EWMA-t and GARCH-t are compared offline for the fixed portfolios "
        "using breach calibration rather than raw fit alone."
    )

    risk_metrics = outputs.get("portfolio_rolling_var_es", pd.DataFrame())
    kupiec_tests = outputs.get("portfolio_var_backtesting_tests", pd.DataFrame())
    latest_summary = outputs.get("latest_portfolio_var_es", pd.DataFrame())
    combined_risk = outputs.get("combined_portfolio_var_es", pd.DataFrame())
    combined_summary = outputs.get("combined_vol_model_summary", pd.DataFrame())
    model_rankings = outputs.get("vol_model_rankings", pd.DataFrame())

    if risk_metrics.empty:
        st.error("Portfolio rolling VaR/ES data is required for this page.")
        return

    selected_portfolio = st.selectbox(
        "Choose a fixed portfolio",
        options=sorted(risk_metrics["Portfolio"].unique().tolist()),
        key="risk_models_portfolio_select",
    )

    selected_metrics = risk_metrics.loc[risk_metrics["Portfolio"] == selected_portfolio].copy()
    summary_portfolio_col = "Name" if "Name" in latest_summary.columns else "Portfolio"
    selected_latest = (
        latest_summary.loc[latest_summary[summary_portfolio_col] == selected_portfolio].copy()
        if not latest_summary.empty and summary_portfolio_col in latest_summary.columns
        else pd.DataFrame()
    )
    selected_kupiec = kupiec_tests.loc[kupiec_tests["Portfolio"] == selected_portfolio].copy() if not kupiec_tests.empty else pd.DataFrame()
    selected_rankings = model_rankings.loc[model_rankings["Portfolio"] == selected_portfolio].copy() if not model_rankings.empty else pd.DataFrame()

    if not selected_latest.empty:
        var_col = find_metric_column(selected_latest, "Rolling VaR")
        es_col = find_metric_column(selected_latest, "Rolling ES")
        render_metric_cards([
            ("Latest Historical VaR", selected_latest[var_col].iloc[0] if var_col else np.nan, "pct"),
            ("Latest Historical ES", selected_latest[es_col].iloc[0] if es_col else np.nan, "pct"),
            ("Kupiec Decision", selected_kupiec["Test Decision"].iloc[0] if not selected_kupiec.empty else "N/A", "text"),
            (
                "Best Model",
                selected_rankings.loc[selected_rankings["Rank"] == 1, "Model"].iloc[0]
                if not selected_rankings.loc[selected_rankings["Rank"] == 1].empty
                else "N/A",
                "text",
            ),
        ])

    charts_left, charts_right = st.columns(2)
    with charts_left:
        with st.container(border=True):
            historical_chart = make_returns_vs_var_chart(
                selected_metrics,
                f"Historical VaR Backtest | {selected_portfolio}",
            )
            st.plotly_chart(style_figure(historical_chart, percent_y=True), use_container_width=True)
    with charts_right:
        with st.container(border=True):
            if combined_risk.empty:
                st.warning(f"Combined model risk data is unavailable. Expected file: {COMBINED_VOL_MODEL_FILE}")
            else:
                comparison_chart = make_model_comparison_chart(combined_risk, selected_portfolio)
                st.plotly_chart(style_figure(comparison_chart, percent_y=True), use_container_width=True)

    with st.expander("Detailed backtesting tables"):
        if not selected_kupiec.empty:
            st.dataframe(selected_kupiec, hide_index=True, use_container_width=True)
        if not combined_summary.empty:
            st.dataframe(
                combined_summary.loc[combined_summary["Portfolio"] == selected_portfolio].reset_index(drop=True),
                hide_index=True,
                use_container_width=True,
            )


def render_stress_and_attribution(outputs: dict[str, pd.DataFrame]) -> None:
    st.title("Stress & Attribution")
    st.caption(
        "Scenario losses and covariance-based attribution explain why the fixed portfolios show different tail-risk "
        "profiles and which holdings drive that difference."
    )

    stress_results = outputs.get("stress_test_results", pd.DataFrame())
    stress_contributions = outputs.get("stress_test_asset_contributions", pd.DataFrame())
    risk_contributions = outputs.get("portfolio_risk_contributions", pd.DataFrame())
    drawdown_attribution = outputs.get("drawdown_attribution_summary", pd.DataFrame())
    risk_driver_summary = outputs.get("risk_driver_summary", pd.DataFrame())

    if stress_results.empty or risk_contributions.empty:
        st.error("Stress-testing and risk-attribution outputs are required for this page.")
        return

    selected_portfolio = st.selectbox(
        "Choose a fixed portfolio",
        options=sorted(risk_contributions["Portfolio"].unique().tolist()),
        key="stress_attr_portfolio_select",
    )
    selected_scenario = st.selectbox(
        "Choose a stress scenario",
        options=list(SCENARIOS.keys()),
        key="stress_attr_scenario_select",
    )

    chart_stress_results = stress_results.copy()
    if "Model" in chart_stress_results.columns and (chart_stress_results["Model"] == "Historical VaR").any():
        chart_stress_results = chart_stress_results.loc[chart_stress_results["Model"] == "Historical VaR"].copy()

    selected_stress_contributions = (
        stress_contributions.loc[
            (stress_contributions["Portfolio"] == selected_portfolio) &
            (stress_contributions["Scenario"] == selected_scenario)
        ].copy()
        if not stress_contributions.empty
        else pd.DataFrame()
    )
    selected_risk_contributions = risk_contributions.loc[risk_contributions["Portfolio"] == selected_portfolio].copy()
    selected_drawdown = (
        drawdown_attribution.loc[drawdown_attribution["Portfolio"] == selected_portfolio].copy()
        if not drawdown_attribution.empty
        else pd.DataFrame()
    )
    selected_driver_summary = (
        risk_driver_summary.loc[risk_driver_summary["Portfolio"] == selected_portfolio].copy()
        if not risk_driver_summary.empty
        else pd.DataFrame()
    )

    if not selected_driver_summary.empty:
        render_metric_cards([
            ("Highest Weight Asset", selected_driver_summary["Highest Weight Asset"].iloc[0], "text"),
            ("Highest Risk Contribution Asset", selected_driver_summary["Highest Risk Contribution Asset"].iloc[0], "text"),
            ("Main Drawdown Contributor", selected_driver_summary["Main Drawdown Contributor"].iloc[0], "text"),
            ("Maximum Drawdown", selected_driver_summary["Maximum Drawdown"].iloc[0], "pct"),
        ])

    charts_left, charts_right = st.columns(2)
    with charts_left:
        with st.container(border=True):
            stress_chart = px.bar(
                chart_stress_results,
                x="Scenario",
                y="Portfolio Stress Loss",
                color="Portfolio",
                barmode="group",
                title="Stress Loss by Scenario and Fixed Portfolio",
            )
            st.plotly_chart(style_figure(stress_chart, percent_y=True), use_container_width=True)
    with charts_right:
        with st.container(border=True):
            if selected_stress_contributions.empty:
                st.info("Stress contribution detail is not available for the selected portfolio and scenario.")
            else:
                contribution_chart = make_asset_contribution_chart(
                    selected_stress_contributions,
                    f"Asset Stress Contributions | {selected_portfolio} | {selected_scenario}",
                )
                st.plotly_chart(style_figure(contribution_chart, percent_y=True), use_container_width=True)

    charts_left, charts_right = st.columns(2)
    with charts_left:
        with st.container(border=True):
            risk_chart = make_weight_vs_risk_chart(
                selected_risk_contributions,
                f"Weight vs Risk Contribution | {selected_portfolio}",
            )
            st.plotly_chart(style_figure(risk_chart, percent_y=True), use_container_width=True)
    with charts_right:
        with st.container(border=True):
            if selected_drawdown.empty:
                st.info("Drawdown attribution detail is not available.")
            else:
                drawdown_chart = px.bar(
                    selected_drawdown,
                    x="Ticker",
                    y="Asset Drawdown Loss Contribution",
                    color="Ticker",
                    title=f"Maximum Drawdown Contributors | {selected_portfolio}",
                )
                st.plotly_chart(style_figure(drawdown_chart, percent_y=True), use_container_width=True)

    with st.expander("Detailed stress and attribution tables"):
        if not selected_stress_contributions.empty:
            st.dataframe(selected_stress_contributions, hide_index=True, use_container_width=True)
        if not selected_driver_summary.empty:
            st.dataframe(selected_driver_summary, hide_index=True, use_container_width=True)


def render_methodology() -> None:
    st.title("Methodology")
    st.markdown(
        f"""
        **Portfolio construction**

        - Three predefined long-only portfolios are estimated offline: Equal Weighted, Global Minimum Variance, and Tangency.
        - Those weights remain fixed across backtesting, volatility-model comparison, stress testing, and attribution.

        **Interactive portfolio lab**

        - The landing page computes custom portfolio returns from the same local historical asset return data used by the engine.
        - Historical VaR/ES and EWMA-t VaR/ES are recomputed dynamically from the selected weights.
        - GARCH-t is not refit live because it would make the dashboard slower and less stable in a recruiter-facing setting.

        **Risk models**

        - Historical VaR/ES uses a `{int(CONFIDENCE_LEVEL * 100)}%` confidence level and a 252-day rolling window.
        - EWMA-t and GARCH-t are compared offline on the predefined fixed portfolios.

        **Stress and attribution**

        - Stress testing uses three deterministic technology-focused scenarios only.
        - Covariance-based attribution explains current volatility concentration using the latest 252 trading days.
        - Drawdown attribution is approximate and intended for interpretability rather than exact rebalancing attribution.
        """
    )


def main() -> None:
    inject_app_styles()
    asset_returns, risk_free_rate, frozen_weights, outputs = load_required_context()
    current_page_key = render_sidebar_navigation()
    if current_page_key == "portfolio_risk_lab":
        render_portfolio_risk_lab(asset_returns, risk_free_rate, frozen_weights)
    elif current_page_key == "overview":
        render_overview(asset_returns, frozen_weights, outputs)
    elif current_page_key == "risk_models":
        render_risk_models_and_backtesting(outputs)
    elif current_page_key == "stress_attr":
        render_stress_and_attribution(outputs)
    elif current_page_key == "methodology":
        render_methodology()


if __name__ == "__main__":
    main()
