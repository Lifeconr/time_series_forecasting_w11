# data_loader.py

import yfinance as yf
import pandas as pd
from typing import List, Dict

def fetch_data(tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """
    Fetches historical stock data for a list of tickers from Yahoo Finance.

    Args:
        tickers (List[str]): A list of stock ticker symbols.
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary where keys are ticker symbols and
                                 values are pandas DataFrames with the historical data.
    """
    try:
        data = yf.download(tickers, start=start_date, end=end_date)
        if data.empty:
            print("No data fetched. Check tickers or date range.")
            return {}

        if len(tickers) == 1:
            return {tickers[0]: data}
        
        asset_data = {ticker: data.xs(ticker, level=1, axis=1) for ticker in tickers}
        
        print(f"Successfully fetched data for: {list(asset_data.keys())}")
        return asset_data

    except Exception as e:
        print(f"An error occurred while fetching data: {e}")
        return {}