#######################
#
# (0) LIBRARIES
#
#######################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import jarque_bera

from path_utils import project_path


#######################
#
# (1) GLOBAL VARIABLES
#
#######################
ADJUSTED_CLOSE_COL = 'Adj Close'
CLOSE_COL = 'Close'
RF_RATE_DAILY_COL = 'rf_rate_daily'
SP500_LABEL = 'S&P 500' # rename from ^GSPC to commonly accepted S&P500 for better identification
TRADING_DAYS_PER_YEAR = 252

SP500_DATA_FILEPATH = project_path("data", "historical", "SP500_prices.parquet")
TOP_5_INDIV_ASSETS_DATA_FILEPATH = project_path("data", "historical", "top_tech_assets_prices.parquet")
RF_RATE_DATA_FILEPATH = project_path("data", "historical", "rf_rate_daily.parquet")
HISTORICAL_STUDY_OUTPUT_DIR = project_path("figure", "historical_study")
SP500_AND_INDIV_ASSETS_SIDE_BY_SIDE_FIG_FILEPATH = HISTORICAL_STUDY_OUTPUT_DIR / "sp500_and_individual_assets_close_prices_side_by_side.png"
SP500_LOG_RETURN_FIG_FILEPATH = HISTORICAL_STUDY_OUTPUT_DIR / "sp500_log_returns_over_time.png"
INDIV_ASSETS_LOG_RETURN_FIG_FILEPATH = HISTORICAL_STUDY_OUTPUT_DIR / "individual_assets_log_returns_over_time.png"
KURTOSIS_TABLE_FILEPATH = HISTORICAL_STUDY_OUTPUT_DIR / "kurtosis_table.csv"
JARQUE_BERA_TABLE_FILEPATH = HISTORICAL_STUDY_OUTPUT_DIR / "jarque_bera_table.csv"
SHARPE_RATIO_TABLE_FILEPATH = HISTORICAL_STUDY_OUTPUT_DIR / "sharpe_ratio_table.csv"

#######################
#
# (2) HELPER FUNCTIONS 
#
#######################
def save_current_figure(filepath: str | Path) -> None:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')

def ensure_utc_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')
    return df

# normalize start and end dates to allow for safe pandas indexing
def normalize_date_range(index: pd.DatetimeIndex, start_date: str=None, end_date: str=None) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_date = index.min() if start_date is None else pd.to_datetime(start_date)
    end_date = index.max() if end_date is None else pd.to_datetime(end_date)

    if index.tz is not None:
        if start_date.tzinfo is None:
            start_date = start_date.tz_localize(index.tz)
        else:
            start_date = start_date.tz_convert(index.tz)

        if end_date.tzinfo is None:
            end_date = end_date.tz_localize(index.tz)
        else:
            end_date = end_date.tz_convert(index.tz)

    return start_date, end_date


# extract adjusted close prices, handling both single-level (S&P500) and multi-level (5 individual assets)column formats 
def extract_adjusted_close_prices(df: pd.DataFrame) -> pd.DataFrame:
    price_column = CLOSE_COL
    if isinstance(df.columns, pd.MultiIndex):
        return df.xs(CLOSE_COL, axis=1, level=0).copy()
    return df[[price_column]].copy()

def preprocess_adjusted_close_price_data(
    SP500_df: pd.DataFrame,
    indiv_asset_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    SP500_df = ensure_utc_index(SP500_df.copy())
    indiv_asset_df = ensure_utc_index(indiv_asset_df.copy())

    SP500_adj_close_df = extract_adjusted_close_prices(SP500_df)
    SP500_adj_close_df = SP500_adj_close_df.rename(columns={SP500_adj_close_df.columns[0]: SP500_LABEL})
    indiv_asset_adj_close_df = extract_adjusted_close_prices(indiv_asset_df)
    return SP500_adj_close_df, indiv_asset_adj_close_df

def slice_date_window(df: pd.DataFrame, start_date: str=None, end_date: str=None) -> pd.DataFrame:
    start_date, end_date = normalize_date_range(df.index, start_date, end_date)
    return df.loc[start_date:end_date]

def compute_log_returns(price_df: pd.DataFrame, start_date: str=None, end_date: str=None) -> pd.DataFrame:
    price_window_df = slice_date_window(price_df, start_date, end_date)
    return np.log(price_window_df / price_window_df.shift(1)).dropna()

def compute_simple_returns(price_df: pd.DataFrame, start_date: str=None, end_date: str=None) -> pd.DataFrame:
    price_window_df = slice_date_window(price_df, start_date, end_date)
    return price_window_df.pct_change(fill_method=None).dropna()

def align_risk_free_rate(
    effective_daily_rf_rate: pd.DataFrame,
    portfolio_index: pd.Index
) -> pd.Series:
    if RF_RATE_DAILY_COL not in effective_daily_rf_rate.columns:
        raise KeyError(f"Expected risk-free rate column '{RF_RATE_DAILY_COL}'.")
    rf_rate_daily = effective_daily_rf_rate[RF_RATE_DAILY_COL]
    return rf_rate_daily.reindex(portfolio_index).ffill().bfill()

def build_individual_asset_sharpe_ratio_table(
    indiv_asset_adj_close_df: pd.DataFrame,
    effective_daily_rf_rate: pd.DataFrame,
    start_date: str=None,
    end_date: str=None
) -> pd.DataFrame:
    indiv_asset_return_df = compute_simple_returns(indiv_asset_adj_close_df, start_date, end_date)
    aligned_rf_rate = align_risk_free_rate(effective_daily_rf_rate, indiv_asset_return_df.index)
    excess_returns = indiv_asset_return_df.sub(aligned_rf_rate, axis=0)
    annualized_sharpe_ratios = excess_returns.mean() / excess_returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)

    return pd.DataFrame({
        'Asset': annualized_sharpe_ratios.index,
        'Annualized Sharpe Ratio': annualized_sharpe_ratios.to_numpy(),
    })

def get_asset_colors(asset_columns: pd.Index) -> dict[str, tuple[float, float, float]]:
    return dict(
        zip(
            asset_columns,
            sns.color_palette('tab10', n_colors=len(asset_columns))
        )
    )

def build_distribution_diagnostics(
    SP500_adj_close_df: pd.DataFrame,
    indiv_asset_adj_close_df: pd.DataFrame,
    start_date: str=None,
    end_date: str=None
) -> pd.DataFrame:
    SP500_log_return_df = compute_log_returns(SP500_adj_close_df, start_date, end_date)
    indiv_asset_log_return_df = compute_log_returns(indiv_asset_adj_close_df, start_date, end_date)

    diagnostics_rows = []
    for asset_name, data in [(SP500_LABEL, SP500_log_return_df[SP500_LABEL])]:
        stat, p_value = jarque_bera(data)
        diagnostics_rows.append((asset_name, data.kurt(), stat, p_value))

    for asset_name in indiv_asset_log_return_df.columns:
        data = indiv_asset_log_return_df[asset_name]
        stat, p_value = jarque_bera(data)
        diagnostics_rows.append((asset_name, data.kurt(), stat, p_value))

    return pd.DataFrame(
        diagnostics_rows,
        columns=['Asset', 'Excess Kurtosis', 'Jarque-Bera Statistic', 'P-Value']
    )

def save_and_print_table(df: pd.DataFrame, filepath: str, title: str) -> None:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    print(f"\n{title}")
    print(df)

def plot_sp500_and_individual_assets_close_prices_side_by_side(
    SP500_adj_close_df: pd.DataFrame,
    indiv_asset_adj_close_df: pd.DataFrame,
    start_date: str=None,
    end_date: str=None
) -> None:
    SP500_close_price_df = slice_date_window(SP500_adj_close_df, start_date, end_date)
    indiv_asset_close_price_df = slice_date_window(indiv_asset_adj_close_df, start_date, end_date)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharex=True)
    asset_colors = get_asset_colors(indiv_asset_close_price_df.columns)
    sp500_color = 'gold'

    axes[0].plot(
        SP500_close_price_df.index,
        SP500_close_price_df[SP500_LABEL],
        label=SP500_LABEL,
        linewidth=2.0,
        color=sp500_color
    )
    for asset in indiv_asset_close_price_df.columns:
        asset_color = asset_colors[asset]
        axes[0].plot(
            indiv_asset_close_price_df.index,
            indiv_asset_close_price_df[asset],
            label=asset,
            alpha=0.9,
            color=asset_color
        )
        axes[1].plot(
            indiv_asset_close_price_df.index,
            indiv_asset_close_price_df[asset],
            label=asset,
            alpha=0.9,
            color=asset_color
        )

    axes[0].set_title('S&P 500 and Top 5 Individual Assets Close Prices', pad=12)
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Close Price')
    axes[0].legend(loc='upper left')

    axes[1].set_title('Top 5 Individual Assets Close Prices', pad=12)
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('Close Price')
    axes[1].legend(loc='upper left')

    for axis in axes:
        axis.grid(True, alpha=0.3)

    fig.suptitle('Close Price Comparison Over Time', fontsize=18, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965], w_pad=2.5)
    save_current_figure(SP500_AND_INDIV_ASSETS_SIDE_BY_SIDE_FIG_FILEPATH)
    plt.show()

def plot_individual_assets_log_returns_over_time(
    indiv_asset_adj_close_df: pd.DataFrame,
    start_date: str=None,
    end_date: str=None
) -> None:
    indiv_asset_log_return_df = compute_log_returns(indiv_asset_adj_close_df, start_date, end_date)
    asset_colors = get_asset_colors(indiv_asset_log_return_df.columns)

    fig, axes = plt.subplots(len(indiv_asset_log_return_df.columns), 1, figsize=(14, 18), sharex=True)

    for axis, asset in zip(axes, indiv_asset_log_return_df.columns):
        axis.plot(
            indiv_asset_log_return_df.index,
            indiv_asset_log_return_df[asset],
            label=asset,
            color=asset_colors[asset]
        )
        axis.axhline(y=0.0, color='black', linestyle='--', linewidth=0.8)
        axis.set_title(f'{asset} Log Returns Over Time', pad=12)
        axis.set_ylabel('Log Return')
        axis.grid(True, alpha=0.3)
        axis.legend(loc='upper right')

    axes[-1].set_xlabel('Date')
    fig.suptitle('Top 5 Individual Assets Log Returns Over Time', fontsize=18, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.975], h_pad=1.2)
    save_current_figure(INDIV_ASSETS_LOG_RETURN_FIG_FILEPATH)
    plt.show()

def plot_sp500_log_returns_over_time(
    SP500_adj_close_df: pd.DataFrame,
    start_date: str=None,
    end_date: str=None
) -> None:
    SP500_log_return_df = compute_log_returns(SP500_adj_close_df, start_date, end_date)

    plt.plot(SP500_log_return_df.index, SP500_log_return_df[SP500_LABEL], color='gold')
    plt.axhline(y=0.0, color='black', linestyle='--', linewidth=0.8)
    plt.title('S&P 500 Log Returns Over Time')
    plt.xlabel('Date')
    plt.ylabel('Log Return')
    save_current_figure(SP500_LOG_RETURN_FIG_FILEPATH)
    plt.show()

def generate_kurtosis_table(
    SP500_adj_close_df: pd.DataFrame,
    indiv_asset_adj_close_df: pd.DataFrame,
    start_date: str=None,
    end_date: str=None
) -> None:
    diagnostics_df = build_distribution_diagnostics(
        SP500_adj_close_df,
        indiv_asset_adj_close_df,
        start_date,
        end_date
    )
    kurtosis_values = diagnostics_df[['Asset', 'Excess Kurtosis']]
    save_and_print_table(
        kurtosis_values,
        KURTOSIS_TABLE_FILEPATH,
        'Excess Kurtosis Values for S&P 500 and Top 5 Individual Assets:'
    )

def generate_jarque_bera_table(
    SP500_adj_close_df: pd.DataFrame,
    indiv_asset_adj_close_df: pd.DataFrame,
    start_date: str=None,
    end_date: str=None
) -> None:
    diagnostics_df = build_distribution_diagnostics(
        SP500_adj_close_df,
        indiv_asset_adj_close_df,
        start_date,
        end_date
    )
    jarque_bera_df = diagnostics_df[['Asset', 'Jarque-Bera Statistic', 'P-Value']]
    save_and_print_table(
        jarque_bera_df,
        JARQUE_BERA_TABLE_FILEPATH,
        'Jarque-Bera Test Results for S&P 500 and Top 5 Individual Assets:'
    )

def generate_sharpe_ratio_table(
    indiv_asset_adj_close_df: pd.DataFrame,
    effective_daily_rf_rate: pd.DataFrame,
    start_date: str=None,
    end_date: str=None
) -> None:
    sharpe_ratio_df = build_individual_asset_sharpe_ratio_table(
        indiv_asset_adj_close_df,
        effective_daily_rf_rate,
        start_date,
        end_date
    )
    save_and_print_table(
        sharpe_ratio_df,
        SHARPE_RATIO_TABLE_FILEPATH,
        'Annualized Sharpe Ratios for the Top 5 Individual Assets:'
    )

def main() -> None:
    SP500_raw_df = pd.read_parquet(SP500_DATA_FILEPATH)
    indiv_asset_raw_df = pd.read_parquet(TOP_5_INDIV_ASSETS_DATA_FILEPATH)
    effective_daily_rf_rate = ensure_utc_index(pd.read_parquet(RF_RATE_DATA_FILEPATH))
    SP500_adj_close_df, indiv_asset_adj_close_df = preprocess_adjusted_close_price_data(
        SP500_raw_df,
        indiv_asset_raw_df
    )

    plot_sp500_and_individual_assets_close_prices_side_by_side(
        SP500_adj_close_df=SP500_adj_close_df,
        indiv_asset_adj_close_df=indiv_asset_adj_close_df
    )
    plot_individual_assets_log_returns_over_time(indiv_asset_adj_close_df=indiv_asset_adj_close_df)
    plot_sp500_log_returns_over_time(SP500_adj_close_df=SP500_adj_close_df)
    generate_kurtosis_table(
        SP500_adj_close_df=SP500_adj_close_df,
        indiv_asset_adj_close_df=indiv_asset_adj_close_df
    )
    generate_jarque_bera_table(
        SP500_adj_close_df=SP500_adj_close_df,
        indiv_asset_adj_close_df=indiv_asset_adj_close_df
    )
    generate_sharpe_ratio_table(
        indiv_asset_adj_close_df=indiv_asset_adj_close_df,
        effective_daily_rf_rate=effective_daily_rf_rate
    )

######################
#
# (3) RUN MAIN
#
######################
if __name__ == "__main__":
    main()
