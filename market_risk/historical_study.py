import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#######################
#
# GLOBAL VARIABLE
#
#######################
CLOSE_COL = 'Close'
DATE_COL = 'Date'

#######################
#
# HELPER FUNCTIONS 
#
#######################
def plot_sp500_close_price_over_time(SP500_df: pd.DataFrame, start_date:str=None, end_date:str=None) -> None:
    if not start_date or not end_date:
        start_date = SP500_df.index.min()
        end_date = SP500_df.index.max()

    SP500_close_price_df =SP500_df.loc[start_date:end_date, CLOSE_COL]

    # make a line plot of the S&P 500 close price over time
    sns.lineplot(data=SP500_close_price_df, x=DATE_COL, y=CLOSE_COL)
    plt.title('S&P 500 Close Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.show()

# overlay the close price of the S&P 500 and the close price of the top 5 individual assets over time
# correct this function based on error message: TypeError: Cannot compare tz-naive and tz-aware datetime-like objects.

def plot_sp500_and_individual_assets_close_price_over_time(SP500_df: pd.DataFrame, indiv_asset_df: pd.DataFrame, start_date:str=None, end_date:str=None) -> None:
    if not start_date or not end_date:
        start_date = SP500_df.index.min()
        end_date = SP500_df.index.max()

    # Ensure the datetime index is timezone-aware
    if SP500_df.index.tz is None:
        SP500_df.index = SP500_df.index.tz_localize('UTC')
    if indiv_asset_df.index.tz is None:
        indiv_asset_df.index = indiv_asset_df.index.tz_localize('UTC')

    SP500_close_price_df =SP500_df.loc[start_date:end_date, CLOSE_COL].to_frame()
    indiv_asset_close_price_df = indiv_asset_df.loc[start_date:end_date, (CLOSE_COL, slice(None))]

    # make a line plot of the S&P 500 close price over time
    # sns.lineplot(data=SP500_close_price_df, x=DATE_COL, y=CLOSE_COL, label='S&P 500')
    plt.axvline(x=pd.to_datetime('2020-03-11'), color='red', linestyle='--', label='COVID-19 Outbreak')

    # make a line plot of the top 5 individual assets close price over time
    for asset in indiv_asset_close_price_df.columns.levels[1]:
        sns.lineplot(data=indiv_asset_close_price_df, x=DATE_COL, y=(CLOSE_COL, asset), label=asset)

    plt.title('S&P 500 and Top 5 Individual Assets Close Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.legend()
    plt.show()

# now overlay the returns instead of the close price of the S&P 500 and the close price of the top 5 individual assets over time
# do not do log returns, just returns, and correct this function based on error message: TypeError: Cannot compare tz-naive and tz-aware datetime-like objects.
# draw a red line at covid date (2020-03-11) to show the impact of covid on the stock market
def plot_sp500_and_individual_assets_return_over_time(SP500_df: pd.DataFrame, indiv_asset_df: pd.DataFrame, start_date:str=None, end_date:str=None) -> None:
    if not start_date or not end_date:
        start_date = SP500_df.index.min()
        end_date = SP500_df.index.max()

    # Ensure the datetime index is timezone-aware
    if SP500_df.index.tz is None:
        SP500_df.index = SP500_df.index.tz_localize('UTC')
    if indiv_asset_df.index.tz is None:
        indiv_asset_df.index = indiv_asset_df.index.tz_localize('UTC')

    SP500_return_df = SP500_df.loc[start_date:end_date, CLOSE_COL].pct_change().to_frame()
    indiv_asset_return_df = indiv_asset_df.loc[start_date:end_date, (CLOSE_COL, slice(None))].pct_change()

    # make a line plot of the S&P 500 return over time
    # sns.lineplot(data=SP500_return_df, x=DATE_COL, y=CLOSE_COL, label='S&P 500')
    
    # make a line plot of the top 5 individual assets return over time
    for asset in indiv_asset_return_df.columns.levels[1]:
        sns.lineplot(data=indiv_asset_return_df, x=DATE_COL, y=(CLOSE_COL, asset), label=asset)

    # draw a red line at the COVID date
    plt.axvline(x=pd.to_datetime('2020-03-11'), color='red', linestyle='--', label='COVID-19 Outbreak')

    plt.title('S&P 500 and Top 5 Individual Assets Returns Over Time')
    plt.xlabel('Date')
    plt.ylabel('Return')
    plt.legend()
    plt.show()



SP500_df = pd.read_parquet('data/SP500_prices.parquet')
indiv_asset_df = pd.read_parquet('data/individual_assets.parquet')


# """
# (.venv) PS C:\Users\PC\Documents\Di xd\projects\market_risk> py historical_study.py
# Traceback (most recent call last):
#   File "C:\Users\PC\Documents\Di xd\projects\market_risk\historical_study.py", line 111, in <module>
#     plot_sp500_and_individual_assets_close_price_over_time(SP500_df=SP500_df, indiv_asset_df=indiv_asset_df)
#   File "C:\Users\PC\Documents\Di xd\projects\market_risk\historical_study.py", line 50, in plot_sp500_and_individual_assets_close_price_over_time
#     sns.lineplot(data=SP500_close_price_df, x=DATE_COL, y=CLOSE_COL, label='S&P 500')
#   File "C:\Users\PC\Documents\Di xd\projects\market_risk\.venv\Lib\site-packages\seaborn\relational.py", line 485, in lineplot
#     p = _LinePlotter(
#         ^^^^^^^^^^^^^
#   File "C:\Users\PC\Documents\Di xd\projects\market_risk\.venv\Lib\site-packages\seaborn\relational.py", line 216, in __init__
#     super().__init__(data=data, variables=variables)
#   File "C:\Users\PC\Documents\Di xd\projects\market_risk\.venv\Lib\site-packages\seaborn\_base.py", line 634, in __init__
#     self.assign_variables(data, variables)
#   File "C:\Users\PC\Documents\Di xd\projects\market_risk\.venv\Lib\site-packages\seaborn\_base.py", line 679, in assign_variables
#     plot_data = PlotData(data, variables)
#                 ^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\PC\Documents\Di xd\projects\market_risk\.venv\Lib\site-packages\seaborn\_core\data.py", line 57, in __init__
#     data = handle_data_source(data)
#            ^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\PC\Documents\Di xd\projects\market_risk\.venv\Lib\site-packages\seaborn\_core\data.py", line 278, in handle_data_source
#     raise TypeError(err)
# TypeError: Data source must be a DataFrame or Mapping, not <class 'pandas.Series'>.
# """

# """
# (.venv) PS C:\Users\PC\Documents\Di xd\projects\market_risk> py historical_study.py
# Traceback (most recent call last):
#   File "C:\Users\PC\Documents\Di xd\projects\market_risk\historical_study.py", line 57, in <module>
#     plot_sp500_and_individual_assets_close_price_over_time(SP500_df=SP500_df, indiv_asset_df=indiv_asset_df)
#   File "C:\Users\PC\Documents\Di xd\projects\market_risk\historical_study.py", line 39, in plot_sp500_and_individual_assets_close_price_over_time
#     indiv_asset_close_price_df = indiv_asset_df.loc[start_date:end_date, (CLOSE_COL, slice(None))]
#                                  ~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\PC\Documents\Di xd\projects\market_risk\.venv\Lib\site-packages\pandas\core\indexing.py", line 1200, in __getitem__
#     return self._getitem_tuple(key)
#            ^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\PC\Documents\Di xd\projects\market_risk\.venv\Lib\site-packages\pandas\core\indexing.py", line 1386, in _getitem_tuple
#     return self._getitem_lowerdim(tup)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\PC\Documents\Di xd\projects\market_risk\.venv\Lib\site-packages\pandas\core\indexing.py", line 1067, in _getitem_lowerdim
#     return self._getitem_nested_tuple(tup)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\PC\Documents\Di xd\projects\market_risk\.venv\Lib\site-packages\pandas\core\indexing.py", line 1172, in _getitem_nested_tuple
#     obj = getattr(obj, self.name)._getitem_axis(key, axis=axis)
#           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\PC\Documents\Di xd\projects\market_risk\.venv\Lib\site-packages\pandas\core\indexing.py", line 1429, in _getitem_axis
#     return self._get_slice_axis(key, axis=axis)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\PC\Documents\Di xd\projects\market_risk\.venv\Lib\site-packages\pandas\core\indexing.py", line 1461, in _get_slice_axis
#     indexer = labels.slice_indexer(slice_obj.start, slice_obj.stop, slice_obj.step)
#               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\PC\Documents\Di xd\projects\market_risk\.venv\Lib\site-packages\pandas\core\indexes\datetimes.py", line 1072, in slice_indexer
#     return Index.slice_indexer(self, start, end, step)
#            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\PC\Documents\Di xd\projects\market_risk\.venv\Lib\site-packages\pandas\core\indexes\base.py", line 6804, in slice_indexer
#     start_slice, end_slice = self.slice_locs(start, end, step=step)
#                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\PC\Documents\Di xd\projects\market_risk\.venv\Lib\site-packages\pandas\core\indexes\base.py", line 7062, in slice_locs
#     start_slice = self.get_slice_bound(start, "left")
#                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\PC\Documents\Di xd\projects\market_risk\.venv\Lib\site-packages\pandas\core\indexes\base.py", line 6964, in get_slice_bound
#     label = self._maybe_cast_slice_bound(label, side)
#             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "C:\Users\PC\Documents\Di xd\projects\market_risk\.venv\Lib\site-packages\pandas\core\indexes\datetimes.py", line 1033, in _maybe_cast_slice_bound
#     self._data._assert_tzawareness_compat(label)
#   File "C:\Users\PC\Documents\Di xd\projects\market_risk\.venv\Lib\site-packages\pandas\core\arrays\datetimes.py", line 796, in _assert_tzawareness_compat
#     raise TypeError(
# TypeError: Cannot compare tz-naive and tz-aware datetime-like objects.
# """
# plot_sp500_and_individual_assets_close_price_over_time(SP500_df=SP500_df, indiv_asset_df=indiv_asset_df)
plot_sp500_and_individual_assets_close_price_over_time(SP500_df=SP500_df, indiv_asset_df=indiv_asset_df)