#######################
#
# (0) LIBRARIES
#
#######################
import pandas as pd
import numpy as np
from scipy.optimize import minimize

#######################
#
# (1) GLOBAL VARIABLES
#
#######################
CLOSE_COL = 'Close'
RF_RATE_DAILY_COL = 'rf_rate_daily'
ASSETS_DATA_PATH = "data/top_tech_assets_prices.parquet"
RF_RATE_DATA_PATH = "data/rf_rate_daily.parquet"
ASSET_STATS_PATH = "data/asset_stats.parquet"
TRADING_DAYS_PER_YEAR = 252

#######################
#
# (2) HELPER FUNCTIONS 
#
#######################
def clean_assets_data(df: pd.DataFrame) -> pd.DataFrame:
    # obtain close price of each asset
    close_prices = df[CLOSE_COL]

    # https://www.sciencedirect.com/science/article/abs/pii/S1057521914001380#s0055
    # based on the paper above, we will use returns and not log returns for calculation of Sharpe Ratio
    # NOTE: can revisit this decision as more literature is reviewed.
    asset_returns = close_prices.pct_change().dropna()
    return asset_returns

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
    effective_daily_rf_rate: pd.DataFrame
) -> tuple[float, float, float]:
    cagr = compute_annualized_cagr(portfolio_returns)
    annualized_volatility = compute_annualized_volatility(portfolio_returns)
    annualized_sharpe_ratio = compute_sharpe_ratio(portfolio_returns, effective_daily_rf_rate)
    return cagr, annualized_volatility, annualized_sharpe_ratio

def report_portfolio_summaries(
    portfolio_name: str,
    weights: np.ndarray,
    asset_names: pd.Index,
    cagr: float,
    annualized_volatility: float,
    annualized_sharpe_ratio: float
) -> None:
    print(f"\n{portfolio_name}")
    print("Weights:")
    for asset_name, weight in zip(asset_names, weights):
        print(f"  {asset_name}: {weight:.4f}")
    print(f"CAGR: {cagr:.4f}")
    print(f"Annualized Volatility: {annualized_volatility:.4f}")
    print(f"Annualized Sharpe Ratio: {annualized_sharpe_ratio:.4f}")

def main():
    assets_df = pd.read_parquet(ASSETS_DATA_PATH)
    effective_daily_rf_rate = pd.read_parquet(RF_RATE_DATA_PATH)
    asset_returns = clean_assets_data(assets_df)
    asset_means, asset_covariances = compute_and_save_asset_statistics(asset_returns)
    asset_names = asset_means.index

    portfolio_weights = {
        'Equal Weighted Portfolio': compute_equal_weights(len(asset_means)),
        'Global Minimum Variance Portfolio': compute_global_minimum_weights(asset_covariances),
        'Tangency Portfolio': compute_tangency_weights(asset_returns, asset_covariances, effective_daily_rf_rate),
    }

    for portfolio_name, weights in portfolio_weights.items():
        portfolio_returns = compute_portfolio_returns(asset_returns, weights)
        cagr, annualized_volatility, annualized_sharpe_ratio = compute_portfolio_summary(
            portfolio_returns,
            effective_daily_rf_rate
        )
        report_portfolio_summaries(
            portfolio_name,
            weights,
            asset_names,
            cagr,
            annualized_volatility,
            annualized_sharpe_ratio
        )

######################
#
# (3) RUN MAIN
#
######################
if __name__ == "__main__":
    main()
