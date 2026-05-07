#######################
#
# (0) LIBRARIES
#
#######################
import pandas as pd
import yfinance as yf

from path_utils import project_path

#######################
#
# (1) GLOBAL VARIABLES
#
#######################
SP500_STOCK_CODE = '^GSPC'
INDIV_ASSET_STOCK_CODE_LST = [
    'AAPL', # apple
    'GOOG', # google
    'NVDA', # nvidia
    'MSFT', # microsoft
    'AMZN', # amazon
]

SP500_FILEPATH = project_path('data', 'historical', 'SP500_prices.parquet')
INDIV_ASSET_FILEPATH = project_path('data', 'historical', 'top_tech_assets_prices.parquet')
RF_RATE_DAILY_FILEPATH = project_path('data', 'historical', 'rf_rate_daily.parquet')
start_date = "2016-05-05"
end_date = "2026-05-04"
CLOSE_COL = 'Close'

# select S&P500

#######################
#
# (2) HELPER FUNCTIONS 
#
#######################
def get_stock_market_index_history(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    S_and_P_500 = yf.Ticker(stock_code)

    # obtain S&P500 history
    SP500_prices = S_and_P_500.history(
        start=start_date,
        end=end_date,
        interval='1d',
        auto_adjust=True,
    )

    return SP500_prices

def get_top_5_individual_assets_history(stock_code_lst: list, start_date: str, end_date: str) -> pd.DataFrame:

    # obtain individual asset history
    individual_assets_prices = yf.download(
        tickers=stock_code_lst,
        start=start_date,
        end=end_date,
        interval='1d',
        auto_adjust=True,
    )

    return individual_assets_prices

def obtain_effective_daily_rf_rate(asset_returns: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    us_treasury_bond_ticker = "^IRX"
    us_treasury_bond_df = yf.download(us_treasury_bond_ticker, 
                                    start=start_date, 
                                    end=pd.to_datetime(end_date) + pd.Timedelta(days=1))[CLOSE_COL]

    # obtain effective daily risk-free rate by solving for rf,t_daily the following relation
    # (rf,t_daily+1)​**252 = 1+rf,t_annual​
    rf_rate_daily = (1 + us_treasury_bond_df.squeeze() / 100) ** (1/252) - 1

    # save the effective daily risk-free rate as a parquet file for later use
    # AttributeError: 'Series' object has no attribute 'to_parquet'
    # change the code below to fix the error
    rf_rate_daily_df = rf_rate_daily.to_frame(name='rf_rate_daily')
    return rf_rate_daily_df

def main():
    SP500_prices = get_stock_market_index_history(
        stock_code=SP500_STOCK_CODE,
        start_date=start_date,
        end_date=end_date
    )

    individual_assets_prices = get_top_5_individual_assets_history(
        stock_code_lst=INDIV_ASSET_STOCK_CODE_LST,
        start_date=start_date,
        end_date=end_date
    )

    effective_daily_rf_rate = obtain_effective_daily_rf_rate(
        asset_returns=individual_assets_prices,
        start_date=start_date,
        end_date=end_date
    )

    SP500_prices.to_parquet(path=SP500_FILEPATH)
    individual_assets_prices.to_parquet(path=INDIV_ASSET_FILEPATH)
    effective_daily_rf_rate.to_parquet(path=RF_RATE_DAILY_FILEPATH)


#######################
#
# (3) RUN MAIN
#
#######################
if __name__ == "__main__":
    main()
