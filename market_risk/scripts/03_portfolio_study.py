#######################
#
# (0) LIBRARIES
#
#######################
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.optimize import minimize

from path_utils import project_path

#######################
#
# (1) GLOBAL VARIABLES
#
#######################
CLOSE_COL = 'Close'
SP500_LABEL = 'S&P 500'
RF_RATE_DAILY_COL = 'rf_rate_daily'
ASSETS_DATA_PATH = project_path("data", "historical", "top_tech_assets_prices.parquet")
SP500_DATA_PATH = project_path("data", "historical", "SP500_prices.parquet")
RF_RATE_DATA_PATH = project_path("data", "historical", "rf_rate_daily.parquet")
ASSET_STATS_PATH = project_path("data", "historical", "asset_stats.parquet")
PORTFOLIO_STUDY_OUTPUT_DIR = project_path("figure", "portfolio_study")
PORTFOLIO_SUMMARY_TABLE_PATH = PORTFOLIO_STUDY_OUTPUT_DIR / "portfolio_summary.csv"
TRADING_DAYS_PER_YEAR = 252
VAR_CONFIDENCE_LEVEL = 0.95

#######################
#
# (2) HELPER FUNCTIONS 
#
#######################
def standardize_daily_index(df: pd.DataFrame) -> pd.DataFrame:
    standardized_df = df.copy()
    standardized_index = pd.to_datetime(standardized_df.index)
    if standardized_index.tz is not None:
        standardized_index = standardized_index.tz_convert('UTC').tz_localize(None)
    standardized_df.index = standardized_index.normalize()
    return standardized_df

def compute_simple_returns(price_data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    # https://www.sciencedirect.com/science/article/abs/pii/S1057521914001380#s0055
    # based on the paper above, we will use returns and not log returns for calculation of Sharpe Ratio
    # NOTE: can revisit this decision as more literature is reviewed.
    return price_data.pct_change().dropna()

def clean_assets_data(df: pd.DataFrame) -> pd.DataFrame:
    close_prices = df[CLOSE_COL]
    return compute_simple_returns(close_prices)

def clean_sp500_data(df: pd.DataFrame) -> pd.Series:
    SP500_close_prices = df[CLOSE_COL]
    SP500_returns = compute_simple_returns(SP500_close_prices)
    return SP500_returns.rename(SP500_LABEL)

def compute_and_save_asset_statistics(asset_returns: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    # compute mean and covariance matrix of the dataset
    asset_means = asset_returns.mean()
    asset_covariances = asset_returns.cov()

    # combine asset means and covariances and save this as a parquet file for later use
    asset_stats = pd.DataFrame({
        'mean': asset_means,
        'covariance': asset_covariances.values.tolist()
    })
    asset_stats.to_parquet(ASSET_STATS_PATH)
    return asset_means, asset_covariances

def align_risk_free_rate(
    effective_daily_rf_rate: pd.DataFrame,
    portfolio_index: pd.Index
) -> pd.Series:
    rf_rate_daily = effective_daily_rf_rate[RF_RATE_DAILY_COL]
    rf_rate_daily = rf_rate_daily.reindex(portfolio_index).ffill().bfill()
    return rf_rate_daily

def compute_excess_returns(
    portfolio_returns: pd.Series,
    effective_daily_rf_rate: pd.DataFrame
) -> pd.Series:
    rf_rate_daily = align_risk_free_rate(effective_daily_rf_rate, portfolio_returns.index)
    return portfolio_returns - rf_rate_daily

def compute_sharpe_ratio(
    portfolio_returns: pd.Series,
    effective_daily_rf_rate: pd.DataFrame
) -> float:
    excess_returns = compute_excess_returns(portfolio_returns, effective_daily_rf_rate)

    # calculate annualized sharpe ratio for the portfolio
    sharpe_ratio_daily = excess_returns.mean() / (excess_returns.std(ddof=1))
    annualized_sharpe_ratio = sharpe_ratio_daily * np.sqrt(TRADING_DAYS_PER_YEAR)

    return annualized_sharpe_ratio

# compute annualized CAGR given a portfolio
# using geometric mean formula
def compute_annualized_cagr(portfolio_returns: pd.Series) -> float:
    cumulative_return = (1 + portfolio_returns).prod() - 1
    num_years = len(portfolio_returns) / TRADING_DAYS_PER_YEAR
    cagr = (1 + cumulative_return) ** (1 / num_years) - 1

    return cagr

# use compounded volatility formula to compute annualized volatility given a portfolio
# cagr = (1 + portfolio_returns).prod() ** (TRADING_DAYS_PER_YEAR / len(portfolio_returns)) - 1
def compute_annualized_volatility(portfolio_returns: pd.Series) -> float:
    daily_volatility = portfolio_returns.std(ddof=1)
    annualized_volatility = daily_volatility * np.sqrt(TRADING_DAYS_PER_YEAR)
    
    return annualized_volatility

def compute_historical_var(portfolio_returns: pd.Series, confidence_level: float=VAR_CONFIDENCE_LEVEL) -> float:
    tail_probability = 1.0 - confidence_level
    var_threshold = portfolio_returns.quantile(tail_probability)
    return -var_threshold

def compute_historical_es(portfolio_returns: pd.Series, confidence_level: float=VAR_CONFIDENCE_LEVEL) -> float:
    tail_probability = 1.0 - confidence_level
    var_threshold = portfolio_returns.quantile(tail_probability)
    tail_returns = portfolio_returns[portfolio_returns <= var_threshold]
    return -tail_returns.mean()

def compute_beta_against_sp500(
    portfolio_returns: pd.Series,
    SP500_returns: pd.Series
) -> float:
    aligned_returns = pd.concat(
        [portfolio_returns.rename('portfolio'), SP500_returns.rename(SP500_LABEL)],
        axis=1,
        join='inner'
    ).dropna()
    portfolio_covariance = aligned_returns['portfolio'].cov(aligned_returns[SP500_LABEL])
    benchmark_variance = aligned_returns[SP500_LABEL].var()
    return portfolio_covariance / benchmark_variance

def compute_equal_weights(num_assets: int) -> np.ndarray:
    return np.ones(num_assets) / num_assets

def build_long_only_constraints(num_assets: int) -> tuple[list[tuple[float, float]], dict]:
    bounds = [(0.0, 1.0)] * num_assets
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1.0}
    return bounds, constraints

def optimize_long_only_weights(
    objective_function,
    num_assets: int
) -> np.ndarray:
    initial_weights = compute_equal_weights(num_assets)
    bounds, constraints = build_long_only_constraints(num_assets)
    optimization_result = minimize(
        objective_function,
        x0=initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={
            'maxiter': 1000,
            'ftol': 1e-12,
        }
    )

    if not optimization_result.success:
        raise ValueError(f"Portfolio optimization failed: {optimization_result.message}")

    return optimization_result.x

def compute_global_minimum_weights(asset_covariances: pd.DataFrame) -> np.ndarray:
    covariance_matrix = asset_covariances.to_numpy()

    def portfolio_variance(weights: np.ndarray) -> float:
        return float(weights @ covariance_matrix @ weights)

    return optimize_long_only_weights(portfolio_variance, asset_covariances.shape[0])

def compute_tangency_weights(
    asset_returns: pd.DataFrame,
    asset_covariances: pd.DataFrame,
    effective_daily_rf_rate: pd.DataFrame
) -> np.ndarray:
    rf_rate_daily = align_risk_free_rate(effective_daily_rf_rate, asset_returns.index)
    mean_excess_returns = asset_returns.sub(rf_rate_daily, axis=0).mean().to_numpy()
    covariance_matrix = asset_covariances.to_numpy()

    def negative_daily_sharpe_ratio(weights: np.ndarray) -> float:
        portfolio_excess_return = weights @ mean_excess_returns
        portfolio_volatility = np.sqrt(weights @ covariance_matrix @ weights)
        if np.isclose(portfolio_volatility, 0.0):
            return np.inf
        return -portfolio_excess_return / portfolio_volatility

    return optimize_long_only_weights(negative_daily_sharpe_ratio, asset_covariances.shape[0])

def compute_portfolio_returns(asset_returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    return asset_returns @ weights

def compute_portfolio_summary(
    portfolio_returns: pd.Series,
    effective_daily_rf_rate: pd.DataFrame,
    SP500_returns: pd.Series
) -> tuple[float, float, float, float, float, float]:
    cagr = compute_annualized_cagr(portfolio_returns)
    annualized_volatility = compute_annualized_volatility(portfolio_returns)
    annualized_sharpe_ratio = compute_sharpe_ratio(portfolio_returns, effective_daily_rf_rate)
    historical_var = compute_historical_var(portfolio_returns)
    historical_es = compute_historical_es(portfolio_returns)
    beta = compute_beta_against_sp500(portfolio_returns, SP500_returns)
    return cagr, annualized_volatility, annualized_sharpe_ratio, historical_var, historical_es, beta

def report_portfolio_summaries(
    portfolio_name: str,
    weights: np.ndarray,
    asset_names: pd.Index,
    cagr: float,
    annualized_volatility: float,
    annualized_sharpe_ratio: float,
    historical_var: float,
    historical_es: float,
    beta: float
) -> None:
    print(f"\n{portfolio_name}")
    print("Weights:")
    for asset_name, weight in zip(asset_names, weights):
        print(f"  {asset_name}: {weight:.4f}")
    print(f"CAGR: {cagr:.4f}")
    print(f"Annualized Volatility: {annualized_volatility:.4f}")
    print(f"Annualized Sharpe Ratio: {annualized_sharpe_ratio:.4f}")
    print(f"Historical VaR ({VAR_CONFIDENCE_LEVEL:.0%}, Daily): {historical_var:.4f}")
    print(f"Historical ES ({VAR_CONFIDENCE_LEVEL:.0%}, Daily): {historical_es:.4f}")
    print(f"Beta vs {SP500_LABEL}: {beta:.4f}")

def save_portfolio_summary_table(portfolio_summary_df: pd.DataFrame, filepath: str) -> None:
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    portfolio_summary_df.to_csv(filepath, index=False)

def main():
    assets_df = standardize_daily_index(pd.read_parquet(ASSETS_DATA_PATH))
    SP500_df = standardize_daily_index(pd.read_parquet(SP500_DATA_PATH))
    effective_daily_rf_rate = standardize_daily_index(pd.read_parquet(RF_RATE_DATA_PATH))
    asset_returns = clean_assets_data(assets_df)
    SP500_returns = clean_sp500_data(SP500_df)
    asset_means, asset_covariances = compute_and_save_asset_statistics(asset_returns)
    asset_names = asset_means.index

    portfolio_weights = {
        'Equal Weighted Portfolio': compute_equal_weights(len(asset_means)),
        'Global Minimum Variance Portfolio': compute_global_minimum_weights(asset_covariances),
        'Tangency Portfolio': compute_tangency_weights(asset_returns, asset_covariances, effective_daily_rf_rate),
    }
    portfolio_summary_rows = []

    for portfolio_name, weights in portfolio_weights.items():
        portfolio_returns = compute_portfolio_returns(asset_returns, weights)
        cagr, annualized_volatility, annualized_sharpe_ratio, historical_var, historical_es, beta = compute_portfolio_summary(
            portfolio_returns,
            effective_daily_rf_rate,
            SP500_returns
        )
        portfolio_summary_rows.append({
            'Portfolio': portfolio_name,
            'CAGR': cagr,
            'Annualized Volatility': annualized_volatility,
            'Annualized Sharpe Ratio': annualized_sharpe_ratio,
            f'Historical VaR ({VAR_CONFIDENCE_LEVEL:.0%}, Daily)': historical_var,
            f'Historical ES ({VAR_CONFIDENCE_LEVEL:.0%}, Daily)': historical_es,
            f'Beta vs {SP500_LABEL}': beta,
        })
        report_portfolio_summaries(
            portfolio_name,
            weights,
            asset_names,
            cagr,
            annualized_volatility,
            annualized_sharpe_ratio,
            historical_var,
            historical_es,
            beta
        )

    portfolio_summary_df = pd.DataFrame(
        portfolio_summary_rows,
        columns=[
            'Portfolio',
            'CAGR',
            'Annualized Volatility',
            'Annualized Sharpe Ratio',
            f'Historical VaR ({VAR_CONFIDENCE_LEVEL:.0%}, Daily)',
            f'Historical ES ({VAR_CONFIDENCE_LEVEL:.0%}, Daily)',
            f'Beta vs {SP500_LABEL}',
        ]
    )
    save_portfolio_summary_table(portfolio_summary_df, PORTFOLIO_SUMMARY_TABLE_PATH)

######################
#
# (3) RUN MAIN
#
######################
if __name__ == "__main__":
    main()
