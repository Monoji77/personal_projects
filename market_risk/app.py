from __future__ import annotations

import base64
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_extras.card_selector import card_selector
from streamlit_extras.specialized_inputs import specialized_text_input

try:
    from st_on_hover_tabs import on_hover_tabs

    HOVER_TABS_IMPORT_ERROR: Exception | None = None
except Exception as error:
    on_hover_tabs = None
    HOVER_TABS_IMPORT_ERROR = error


APP_DIR = Path(__file__).resolve().parent
STYLE_PATH = APP_DIR / "style.css"
DISPLAY_PHOTO_PATH = APP_DIR / "figure" / "extra" / "display.jpg"
SCRIPTS_DIR = APP_DIR / "scripts"
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
    build_portfolio_returns,
    compute_custom_risk_attribution,
    compute_custom_stress_results,
    compute_drawdown_series,
    compute_ewma_t_var_es,
    compute_garch_t_var_es,
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
    "Long-only Global Minimum Variance Portfolio",
    "Long-only Tangency Portfolio",
    "Tech Balanced",
    "High NVDA Exposure",
]

PRESET_META = {
    "Equal Weight": {
        "icon": ":material/view_comfy_alt:",
        "title": "Equal Weight Portfolio",
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
        "title": "Tech Balanced",
        "description": "A hand-tuned mix intended to stay diversified without matching equal weight exactly.",
    },
    "High NVDA Exposure": {
        "icon": ":material/rocket_launch:",
        "title": "High NVDA Exposure",
        "description": "Growth-oriented tilt with heavy NVIDIA concentration.",
    },
    "Long-only Global Minimum Variance Portfolio": {
        "icon": ":material/shield:",
        "title": "Long-only Global Minimum Variance Portfolio",
        "description": "Frozen low-variance allocation estimated offline from the training sample.",
    },
    "Long-only Tangency Portfolio": {
        "icon": ":material/trending_up:",
        "title": "Long-only Tangency Portfolio",
        "description": "Frozen maximum-Sharpe long-only allocation estimated offline from the training sample.",
    },
}

REPO_URL = "https://github.com/Monoji77/personal_projects/tree/main/market_risk"
ACTIVE_PAGE_KEY = "active_page_key"
LAST_RENDERED_PAGE_KEY = "last_rendered_page_key"
PAGE_QUERY_PARAM_KEY = "page"
PRESET_SELECTION_KEY = "preset_selection_key"
PRESET_CARD_KEY = "preset_card_selector"
PRESET_CARD_INSTANCE_KEY = "preset_card_instance_key"
PLOTLY_HEIGHT = 420
CUSTOM_PORTFOLIO_NAME = "Custom Portfolio"

METRIC_HELP_TEXT = {
    "Historical VaR": "Negative 5th percentile of the prior 252 daily returns. It estimates the minimum one-day loss expected 95% of the time.",
    "Historical ES": "Average of the worst 5% of daily returns over the prior 252-day window. It measures tail loss beyond VaR.",
    "EWMA-t VaR": "One-day 95% VaR from an EWMA volatility estimate with Student-t tails. Recent returns receive more weight through lambda = 0.94.",
    "EWMA-t ES": "One-day expected shortfall from the EWMA-t model. It averages losses in the Student-t tail beyond the EWMA-t VaR cutoff.",
    "Worst Stress Scenario": "Scenario with the largest deterministic one-day portfolio loss across the predefined stress shocks.",
    "Highest Risk Contributor": "Asset with the largest component contribution to portfolio volatility using the latest 252-day covariance matrix.",
    "Maximum Drawdown": "Most negative peak-to-trough decline in cumulative backtest wealth. Formula: current wealth / running peak - 1.",
    "Worst Stress Loss": "Largest deterministic one-day portfolio loss across the predefined stress scenarios.",
    "Sharpe Ratio": "Annualized excess return divided by annualized excess volatility over the backtest window.",
    "Backtest Cumulative Return": "Total compounded return since the backtest start date. Formula: product(1 + daily return) - 1.",
    "Backtest CAGR": "Compound annual growth rate over the backtest period. Formula: terminal wealth^(252 / observations) - 1.",
    "Annualized Volatility": "Standard deviation of backtest daily returns multiplied by sqrt(252).",
}


def inject_hover_tab_styles() -> None:
    if not STYLE_PATH.exists():
        st.warning("The on-hover tab stylesheet `style.css` was not found next to `app.py`.")
        return
    st.markdown(f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def inject_app_styles() -> None:
    st.markdown(
        """
        <style>
            /* Main app background */
            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(56, 189, 248, 0.12), transparent 28%),
                    radial-gradient(circle at top right, rgba(34, 197, 94, 0.08), transparent 26%),
                    linear-gradient(135deg, #07111F 0%, #0A1020 45%, #050814 100%);
                color: #E5E7EB;
            }

            /* Main content width and spacing */
            section.main > div.block-container {
                padding-top: 1.5rem;
                padding-bottom: 3rem;
                max-width: 1500px;
            }

            /* Sidebar */
            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #050814 0%, #0B1220 100%);
                border-right: 1px solid rgba(148, 163, 184, 0.18);
            }

            .sidebar-brand-card {
                display: flex;
                align-items: center;
                gap: 0.85rem;
                margin-bottom: 1rem;
                overflow: hidden;
            }

            .sidebar-brand-photo {
                width: 58px;
                height: 58px;
                border-radius: 999px;
                object-fit: cover;
                object-position: center 18%;
                border: 2px solid rgba(56, 189, 248, 0.55);
                box-shadow: 0 0 0 4px rgba(14, 165, 233, 0.08);
                flex-shrink: 0;
            }

            .sidebar-brand-copy {
                opacity: 0;
                max-width: 0;
                transition: opacity 0.25s ease, max-width 0.25s ease;
                overflow: hidden;
            }

            section[data-testid='stSidebar']:hover .sidebar-brand-copy {
                opacity: 1;
                max-width: 220px;
            }

            .sidebar-brand-kicker {
                color: #7DD3FC;
                font-family: "Segoe UI", "Helvetica Neue", sans-serif;
                font-size: 0.7rem;
                font-weight: 700;
                letter-spacing: 0.18em;
                text-transform: uppercase;
                margin-bottom: 0.15rem;
            }

            .sidebar-brand-name {
                color: #F8FAFC;
                font-family: Cambria, Georgia, "Times New Roman", serif;
                font-size: 1.02rem;
                font-weight: 700;
                line-height: 1.2;
            }

            .page-link {
                color: #7DD3FC;
                text-decoration: none;
                font-weight: 600;
            }

            .page-link:hover {
                text-decoration: underline;
            }

            /* Headings */
            h1, h2, h3 {
                letter-spacing: -0.035em;
                color: #F8FAFC;
            }

            h1 {
                font-size: 2.25rem;
                font-weight: 750;
            }

            h2 {
                font-size: 1.35rem;
                font-weight: 700;
            }

            /* Captions / helper text */
            .section-caption,
            [data-testid="stCaptionContainer"] {
                color: #94A3B8 !important;
                font-size: 0.92rem;
                line-height: 1.5;
            }

            .section-spacer {
                margin-top: 0.5rem;
            }

            /* Bordered Streamlit containers */
            [data-testid="stVerticalBlockBorderWrapper"] {
                background: rgba(15, 23, 42, 0.72);
                border: 1px solid rgba(148, 163, 184, 0.18);
                border-radius: 18px;
                box-shadow:
                    0 18px 45px rgba(0, 0, 0, 0.28),
                    inset 0 1px 0 rgba(255, 255, 255, 0.04);
                backdrop-filter: blur(12px);
            }

            /* Metric cards */
            [data-testid="stMetric"] {
                background:
                    linear-gradient(180deg, rgba(15, 23, 42, 0.92), rgba(15, 23, 42, 0.66));
                border-radius: 16px;
                padding: 0.85rem 0.9rem;
            }

            [data-testid="stMetricLabel"] {
                color: #94A3B8;
                font-size: 0.78rem;
                font-weight: 650;
                text-transform: uppercase;
                letter-spacing: 0.055em;
            }

            [data-testid="stMetricValue"] {
                color: #F8FAFC;
                font-size: 1.75rem;
                font-weight: 720;
                letter-spacing: -0.04em;
            }

            /* Info boxes */
            [data-testid="stAlert"] {
                background: rgba(14, 165, 233, 0.10);
                border: 1px solid rgba(56, 189, 248, 0.25);
                border-radius: 14px;
                color: #BAE6FD;
            }

            /* Dataframes */
            [data-testid="stDataFrame"] {
                border-radius: 14px;
                overflow: hidden;
                border: 1px solid rgba(148, 163, 184, 0.16);
            }

            /* Inputs */
            [data-testid="stSelectbox"],
            [data-testid="stNumberInput"],
            [data-testid="stTextInput"] {
                color: #E5E7EB;
            }

            /* Horizontal rule style */
            hr {
                border-color: rgba(148, 163, 184, 0.16);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def encode_image_as_data_uri(image_path: Path) -> str | None:
    if not image_path.exists():
        return None
    image_base64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:image/{image_path.suffix.lstrip('.').lower()};base64,{image_base64}"


def render_sidebar_branding() -> None:
    image_data_uri = encode_image_as_data_uri(DISPLAY_PHOTO_PATH)
    image_html = (
        f'<img src="{image_data_uri}" alt="Chris Yong" class="sidebar-brand-photo" />'
        if image_data_uri is not None
        else '<div class="sidebar-brand-photo"></div>'
    )
    st.markdown(
        f"""
        <div class="sidebar-brand-card">
            {image_html}
            <div class="sidebar-brand-copy">
                <div class="sidebar-brand-kicker">Made by Chris Yong</div>
                <div class="sidebar-brand-name">Market Risk Engine</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sync_page_from_query_params() -> None:
    requested_page = st.query_params.get(PAGE_QUERY_PARAM_KEY, None)
    if isinstance(requested_page, list):
        requested_page = requested_page[0] if requested_page else None
    if requested_page in PAGE_CONFIG:
        st.session_state[ACTIVE_PAGE_KEY] = requested_page
    else:
        st.session_state.setdefault(ACTIVE_PAGE_KEY, "portfolio_risk_lab")


def set_current_page(page_key: str) -> None:
    st.session_state[ACTIVE_PAGE_KEY] = page_key
    st.query_params[PAGE_QUERY_PARAM_KEY] = page_key


def build_internal_page_link(page_key: str, label: str) -> str:
    return f'<a href="?{PAGE_QUERY_PARAM_KEY}={page_key}" target="_self" class="page-link">{label}</a>'


def get_preset_display_name(preset_name: str) -> str:
    return PRESET_META.get(preset_name, {}).get("title", preset_name)


def reset_portfolio_risk_lab_state(frozen_weights: pd.DataFrame | None) -> None:
    preset_weight_map = build_preset_weight_map(frozen_weights)
    default_preset = "Equal Weight" if "Equal Weight" in preset_weight_map else next(iter(preset_weight_map))
    set_weight_state_from_preset(default_preset, preset_weight_map)
    advance_preset_card_instance()


def advance_preset_card_instance() -> None:
    st.session_state[PRESET_CARD_INSTANCE_KEY] = int(st.session_state.get(PRESET_CARD_INSTANCE_KEY, 0)) + 1


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
        render_sidebar_branding()
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
    set_current_page(selected_page_key)
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


def render_metric_cards(metric_items: list[tuple[str, float | str | None, str] | dict[str, object]]) -> None:
    columns = st.columns(len(metric_items))
    for column, metric_item in zip(columns, metric_items):
        if isinstance(metric_item, dict):
            label = str(metric_item["label"])
            value = metric_item["value"]
            metric_type = str(metric_item["metric_type"])
            help_text = str(metric_item["help"]) if metric_item.get("help") is not None else None
        else:
            label, value, metric_type = metric_item[:3]
            help_text = str(metric_item[3]) if len(metric_item) > 3 and metric_item[3] is not None else None

        if metric_type == "pct":
            display_value = format_percentage(value)
        elif metric_type == "decimal":
            display_value = format_decimal(value)
        elif metric_type == "count":
            display_value = "N/A" if value is None or pd.isna(value) else f"{int(value)}"
        else:
            display_value = "N/A" if value is None or pd.isna(value) else str(value)
        with column.container(border=True):
            st.metric(label, display_value, help=help_text)


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


def weight_input_widget_key(ticker: str) -> str:
    return f"{weight_input_key(ticker)}_widget"


def format_weight_value(weight_pct: float) -> str:
    return f"{weight_pct:.1f}".rstrip("0").rstrip(".")


def set_weight_input_value(ticker: str, raw_value: str) -> None:
    normalized_value = str(raw_value)
    st.session_state[weight_input_key(ticker)] = normalized_value
    st.session_state[weight_input_widget_key(ticker)] = {"value": normalized_value}


def set_weight_state_from_preset(preset_name: str, preset_weight_map: dict[str, dict[str, float]]) -> None:
    st.session_state[PRESET_SELECTION_KEY] = preset_name
    if preset_name == CUSTOM_PORTFOLIO_NAME:
        return
    for ticker, weight_pct in preset_weight_map[preset_name].items():
        set_weight_input_value(ticker, format_weight_value(float(weight_pct)))


def initialize_weight_state(preset_weight_map: dict[str, dict[str, float]]) -> None:
    if PRESET_SELECTION_KEY not in st.session_state:
        default_preset = "Equal Weight" if "Equal Weight" in preset_weight_map else next(iter(preset_weight_map))
        set_weight_state_from_preset(default_preset, preset_weight_map)
    else:
        selected_preset = st.session_state.get(PRESET_SELECTION_KEY)
        for ticker in TICKERS:
            default_value = 0.0
            if selected_preset in preset_weight_map:
                default_value = float(preset_weight_map[selected_preset].get(ticker, 0.0))
            st.session_state.setdefault(weight_input_key(ticker), format_weight_value(default_value))
            st.session_state.setdefault(
                weight_input_widget_key(ticker),
                {"value": str(st.session_state.get(weight_input_key(ticker), format_weight_value(default_value)))},
            )


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


def validate_weight_inputs(raw_values: dict[str, str] | None = None) -> dict[str, object]:
    if raw_values is None:
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


def get_weight_builder_card_names(preset_weight_map: dict[str, dict[str, float]]) -> list[str]:
    return list(preset_weight_map.keys()) + [CUSTOM_PORTFOLIO_NAME]


def get_current_preset_card_key() -> str:
    return f"{PRESET_CARD_KEY}_{st.session_state.get(PRESET_CARD_INSTANCE_KEY, 0)}"


def should_mark_custom_portfolio(
    selected_preset_name: str,
    raw_values: dict[str, str],
    preset_weight_map: dict[str, dict[str, float]],
) -> bool:
    if selected_preset_name == CUSTOM_PORTFOLIO_NAME or selected_preset_name not in preset_weight_map:
        return False

    reference_weights = preset_weight_map[selected_preset_name]
    for ticker in TICKERS:
        parsed_value, error_message = parse_weight_input_value(raw_values.get(ticker, "0"))
        if error_message is not None or parsed_value is None:
            return True
        current_display_value = format_weight_value(parsed_value)
        reference_display_value = format_weight_value(float(reference_weights.get(ticker, 0.0)))
        if current_display_value != reference_display_value:
            return True
    return False


def compute_latest_model_metrics(risk_backtest: pd.DataFrame) -> tuple[float | None, float | None]:
    if risk_backtest.empty:
        return None, None
    valid_model = risk_backtest.loc[risk_backtest["Rolling VaR (95%, 252D)"].notna()].copy()
    if valid_model.empty:
        return None, None
    return (
        float(valid_model["Rolling VaR (95%, 252D)"].iloc[-1]),
        float(valid_model["Rolling ES (95%, 252D)"].iloc[-1]),
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
    garch_risk_full = compute_garch_t_var_es(custom_returns_full)
    garch_risk_backtest = filter_backtest_risk_df(garch_risk_full)
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
    ewma_var, ewma_es = compute_latest_model_metrics(ewma_risk_backtest)
    garch_var, garch_es = compute_latest_model_metrics(garch_risk_backtest)

    return {
        "custom_returns_full": custom_returns_full,
        "custom_returns_backtest": custom_returns_backtest,
        "historical_risk_backtest": historical_risk_backtest,
        "ewma_risk_backtest": ewma_risk_backtest,
        "garch_risk_backtest": garch_risk_backtest,
        "risk_contributions": risk_contributions,
        "stress_results": stress_results,
        "stress_contributions": stress_contributions,
        "performance_metrics": performance_metrics,
        "latest_ewma_var": ewma_var,
        "latest_ewma_es": ewma_es,
        "latest_garch_var": garch_var,
        "latest_garch_es": garch_es,
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
    latest_data_date = asset_returns.index.max().date().isoformat() if asset_returns is not None and not asset_returns.empty else "N/A"
    st.caption(
        "Choose a starting portfolio or manually edit a long-only technology allocation and watch return, volatility, "
        "VaR, expected shortfall, stress loss, and risk concentration update live."
    )
    st.caption(f"As of {latest_data_date}")
    st.info(
        "Assets used are arbitrarily chosen by me and used across the project. "
        "Weights must be long-only and sum to 100%. Historical VaR/ES, EWMA-t VaR/ES, and GARCH-t VaR/ES are recomputed "
        "live from the selected weights."
    )

    if asset_returns is None or asset_returns.empty:
        st.error(
            "Asset return data is required for the Portfolio Risk Lab. Please include historical price data or asset_returns.csv."
        )
        return

    preset_weight_map = build_preset_weight_map(frozen_weights)
    initialize_weight_state(preset_weight_map)

    st.subheader("Weight Builder")
    with st.container(border=True):
        card_names = get_weight_builder_card_names(preset_weight_map)
        preset_items = [
            (
                {
                    "icon": PRESET_META[preset_name]["icon"],
                    "title": get_preset_display_name(preset_name),
                    "description": PRESET_META[preset_name]["description"],
                }
                if preset_name != CUSTOM_PORTFOLIO_NAME
                else {
                    "icon": ":material/edit_note:",
                    "title": CUSTOM_PORTFOLIO_NAME,
                    "description": "Manual weights entered below. This becomes active whenever inputs diverge from a preset.",
                }
            )
            for preset_name in card_names
        ]
        selected_preset_name = str(st.session_state.get(PRESET_SELECTION_KEY, "Equal Weight"))
        if selected_preset_name not in card_names:
            selected_preset_name = CUSTOM_PORTFOLIO_NAME
            st.session_state[PRESET_SELECTION_KEY] = CUSTOM_PORTFOLIO_NAME
        default_index = card_names.index(selected_preset_name)
        selected_preset_index = card_selector(
            preset_items,
            selection_mode="single",
            default=default_index,
            key=get_current_preset_card_key(),
        )
        if selected_preset_index is not None:
            selected_card_name = card_names[selected_preset_index]
            if selected_card_name != st.session_state.get(PRESET_SELECTION_KEY):
                st.session_state[PRESET_SELECTION_KEY] = selected_card_name
                if selected_card_name != CUSTOM_PORTFOLIO_NAME:
                    set_weight_state_from_preset(selected_card_name, preset_weight_map)
                st.rerun()

        st.caption(
            "Preset cards populate the current portfolio. You can still adjust any ticker weight manually afterwards."
        )

        raw_weight_values: dict[str, str] = {}
        input_columns = st.columns(len(TICKERS))
        for column, ticker in zip(input_columns, TICKERS):
            with column:
                raw_value = specialized_text_input(
                    label=f"{ticker} weight",
                    value=st.session_state.get(weight_input_key(ticker), "0"),
                    key=weight_input_widget_key(ticker),
                    prefix="%",
                    placeholder="0.0",
                    help="Enter a long-only portfolio weight between 0 and 100.",
                    error=None,
                )
                raw_weight_values[ticker] = raw_value
                st.session_state[weight_input_key(ticker)] = raw_value

        if should_mark_custom_portfolio(selected_preset_name, raw_weight_values, preset_weight_map):
            st.session_state[PRESET_SELECTION_KEY] = CUSTOM_PORTFOLIO_NAME
            advance_preset_card_instance()
            st.rerun()

        weight_state = validate_weight_inputs(raw_weight_values)

        st.markdown(f"**Current total weight:** {weight_state['total_weight_pct']:.2f}%")

        if weight_state["total_weight_pct"] <= 0.0:
            st.error("All weights are zero. Enter at least one positive allocation to continue.")
        elif not weight_state["input_errors"] and np.isclose(weight_state["total_weight_pct"], 100.0, atol=1e-8):
            st.success("Weights are valid. The portfolio is long-only and fully invested.")
        elif not weight_state["input_errors"]:
            st.warning("Weights are valid individually, but the portfolio must sum to exactly 100% before live metrics update.")

    selected_preset_name = str(st.session_state.get(PRESET_SELECTION_KEY, "Equal Weight"))
    current_weights = weight_state["weights_series"]
    snapshot_results: dict[str, object] | None = None
    snapshot_error: str | None = None
    if weight_state["weights_valid"]:
        try:
            snapshot_results = build_custom_portfolio_snapshot(asset_returns, risk_free_rate, current_weights)
        except Exception as error:
            snapshot_error = str(error)

    if snapshot_results is not None:
        custom_returns_backtest = snapshot_results["custom_returns_backtest"]
        historical_risk_backtest = snapshot_results["historical_risk_backtest"]
        ewma_risk_backtest = snapshot_results["ewma_risk_backtest"]
        garch_risk_backtest = snapshot_results["garch_risk_backtest"]
        risk_contributions = snapshot_results["risk_contributions"]

        var_chart = make_returns_vs_var_chart(
            historical_risk_backtest,
            "Historical VaR Backtest with EWMA-t and GARCH-t Overlays",
            ewma_df=ewma_risk_backtest,
            garch_df=garch_risk_backtest,
        )
        st.plotly_chart(style_figure(var_chart, percent_y=True), use_container_width=True)

    st.subheader("Live Portfolio Risk Snapshot")
    st.caption(f"Portfolio based on {get_preset_display_name(selected_preset_name)}.")
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
            latest_garch_var = snapshot_results["latest_garch_var"]
            latest_garch_es = snapshot_results["latest_garch_es"]

            top_risk_driver = risk_contributions.sort_values("Percentage Contribution to Risk", ascending=False).iloc[0]
            worst_stress_scenario = stress_results.sort_values("Portfolio Stress Loss", ascending=False).iloc[0]

            render_metric_cards([
                {
                    "label": "Historical VaR",
                    "value": performance_metrics["Latest Historical VaR"],
                    "metric_type": "pct",
                    "help": METRIC_HELP_TEXT["Historical VaR"],
                },
                {
                    "label": "Historical ES",
                    "value": performance_metrics["Latest Historical ES"],
                    "metric_type": "pct",
                    "help": METRIC_HELP_TEXT["Historical ES"],
                },
                {
                    "label": "EWMA-t VaR",
                    "value": latest_ewma_var,
                    "metric_type": "pct",
                    "help": METRIC_HELP_TEXT["EWMA-t VaR"],
                },
                {
                    "label": "EWMA-t ES",
                    "value": latest_ewma_es,
                    "metric_type": "pct",
                    "help": METRIC_HELP_TEXT["EWMA-t ES"],
                },
            ])
            render_metric_cards([
                {
                    "label": "Worst Stress Scenario",
                    "value": str(worst_stress_scenario["Scenario"]),
                    "metric_type": "text",
                    "help": METRIC_HELP_TEXT["Worst Stress Scenario"],
                },
                {
                    "label": "Highest Risk Contributor",
                    "value": f"{top_risk_driver['Ticker']} ({format_percentage(top_risk_driver['Percentage Contribution to Risk'])})",
                    "metric_type": "text",
                    "help": METRIC_HELP_TEXT["Highest Risk Contributor"],
                },
                {
                    "label": "Maximum Drawdown",
                    "value": performance_metrics["Maximum Drawdown"],
                    "metric_type": "pct",
                    "help": METRIC_HELP_TEXT["Maximum Drawdown"],
                },
                {
                    "label": "Worst Stress Loss",
                    "value": float(worst_stress_scenario["Portfolio Stress Loss"]),
                    "metric_type": "pct",
                    "help": METRIC_HELP_TEXT["Worst Stress Loss"],
                },
            ])
            render_metric_cards([
                {
                    "label": "Sharpe Ratio",
                    "value": performance_metrics["Sharpe Ratio"],
                    "metric_type": "decimal",
                    "help": METRIC_HELP_TEXT["Sharpe Ratio"],
                },
                {
                    "label": "Backtest Cumulative Return",
                    "value": performance_metrics["Backtest Cumulative Return"],
                    "metric_type": "pct",
                    "help": METRIC_HELP_TEXT["Backtest Cumulative Return"],
                },
                {
                    "label": "Backtest CAGR",
                    "value": performance_metrics["Backtest CAGR"],
                    "metric_type": "pct",
                    "help": METRIC_HELP_TEXT["Backtest CAGR"],
                },
                {
                    "label": "Annualized Volatility",
                    "value": performance_metrics["Annualized Volatility"],
                    "metric_type": "pct",
                    "help": METRIC_HELP_TEXT["Annualized Volatility"],
                },
            ])

            if any(pd.isna(value) for value in (latest_ewma_var, latest_ewma_es, latest_garch_var, latest_garch_es)):
                st.caption("EWMA-t and GARCH-t begin once the rolling estimation window is fully populated.")

    if snapshot_results is not None:
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
    st.markdown(f"Link to market risk engine codebase can be found here: [GitHub repository]({REPO_URL})")


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


def render_methodology(outputs: dict[str, pd.DataFrame]) -> None:
    st.title("Methodology")
    construction_summary = outputs.get("portfolio_construction_summary", pd.DataFrame())
    training_start = "N/A"
    training_end = "N/A"
    if not construction_summary.empty and {"Training Start", "Training End"}.issubset(construction_summary.columns):
        training_start = pd.to_datetime(construction_summary["Training Start"]).min().date().isoformat()
        training_end = pd.to_datetime(construction_summary["Training End"]).max().date().isoformat()

    st.markdown(
        f"""
        <p><strong>Portfolio construction</strong></p>
        <ul>
            <li>Three predefined long-only portfolios are estimated offline and summarized in {build_internal_page_link("overview", "Overview")}: Equal Weighted, Global Minimum Variance, and Tangency.</li>
            <li>The estimation window used to derive those three portfolios runs from <code>{training_start}</code> to <code>{training_end}</code>.</li>
            <li>Those weights remain fixed across {build_internal_page_link("risk_models", "Risk Models & Backtesting")} and {build_internal_page_link("stress_attr", "Stress & Attribution")}.</li>
        </ul>

        <p><strong>Interactive portfolio lab</strong></p>
        <ul>
            <li>{build_internal_page_link("portfolio_risk_lab", "Portfolio Risk Lab")} opens with the Equal Weight Portfolio selected whenever the page is entered, and users can then adjust the weights or switch presets.</li>
            <li>The page computes custom portfolio returns from the same local historical asset return data used by the engine.</li>
            <li>Historical VaR/ES, EWMA-t VaR/ES, and GARCH-t VaR/ES are recomputed live from the currently selected custom weights.</li>
        </ul>

        <p><strong>Risk models</strong></p>
        <ul>
            <li>{build_internal_page_link("risk_models", "Risk Models & Backtesting")} uses a <code>{int(CONFIDENCE_LEVEL * 100)}%</code> confidence level and a 252-day rolling window for the historical VaR/ES baseline.</li>
            <li>EWMA-t and GARCH-t are compared offline on the predefined fixed portfolios using backtest behavior rather than fit alone.</li>
        </ul>

        <p><strong>Stress and attribution</strong></p>
        <ul>
            <li>{build_internal_page_link("stress_attr", "Stress & Attribution")} uses three deterministic technology-focused scenarios.</li>
            <li>Covariance-based attribution explains current volatility concentration using the latest 252 trading days.</li>
            <li>Drawdown attribution is approximate and intended for interpretability rather than exact rebalancing attribution.</li>
        </ul>
        """
        ,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_app_styles()
    inject_hover_tab_styles()
    sync_page_from_query_params()
    asset_returns, risk_free_rate, frozen_weights, outputs = load_required_context()
    current_page_key = render_sidebar_navigation()
    previous_page_key = st.session_state.get(LAST_RENDERED_PAGE_KEY)

    if current_page_key == "portfolio_risk_lab" and previous_page_key != "portfolio_risk_lab":
        reset_portfolio_risk_lab_state(frozen_weights)

    if current_page_key == "portfolio_risk_lab":
        render_portfolio_risk_lab(asset_returns, risk_free_rate, frozen_weights)
    elif current_page_key == "overview":
        render_overview(asset_returns, frozen_weights, outputs)
    elif current_page_key == "risk_models":
        render_risk_models_and_backtesting(outputs)
    elif current_page_key == "stress_attr":
        render_stress_and_attribution(outputs)
    elif current_page_key == "methodology":
        render_methodology(outputs)

    st.session_state[LAST_RENDERED_PAGE_KEY] = current_page_key


if __name__ == "__main__":
    main()
