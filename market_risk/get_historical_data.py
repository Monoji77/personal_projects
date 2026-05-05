import pandas as pd
import yfinance as yf

#######################
#
# GLOBAL VARIABLE
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

SP500_FILEPATH = f'data/SP500_prices.parquet'
INDIV_ASSET_FILEPATH = f'data/top_tech_assets_prices.parquet'
start_date = "2016-05-05"
end_date = "2026-05-04"

# select S&P500

#######################
#
# HELPER FUNCTIONS 
#
#######################
def get_stock_market_index_history(stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    S_and_P_500 = yf.Ticker(stock_code)

    # obtain S&P500 history
    SP500_prices = S_and_P_500.history(
        start=start_date,
        end=end_date,
        interval='1d',
        auto_adjust=False,
    )

    return SP500_prices

def get_top_5_individual_assets_history(stock_code_lst: list, start_date: str, end_date: str) -> pd.DataFrame:

    # obtain individual asset history
    individual_assets_prices = yf.download(
        tickers=stock_code_lst,
        start=start_date,
        end=end_date,
        interval='1d',
        auto_adjust=False,
    )

    return individual_assets_prices

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

    SP500_prices.to_parquet(path=SP500_FILEPATH)
    individual_assets_prices.to_parquet(path=INDIV_ASSET_FILEPATH)


#######################
#
# Run main function
#
#######################
if __name__ == "__main__":
    main()