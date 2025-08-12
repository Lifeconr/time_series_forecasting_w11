# preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the financial DataFrame by handling missing values and ensuring correct types.

    Args:
        df (pd.DataFrame): Input DataFrame with financial data.

    Returns:
        pd.DataFrame: A cleaned DataFrame.
    """
    df.index = pd.to_datetime(df.index)
    if df.isnull().sum().sum() > 0:
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        
    print("Data cleaned successfully. Missing values handled.")
    return df

def calculate_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the daily percentage change in the adjusted closing price (using 'Close').

    Args:
        df (pd.DataFrame): DataFrame with a 'Close' column (adjusted due to yfinance auto_adjust).

    Returns:
        pd.DataFrame: The DataFrame with a new 'Daily_Return' column.
    """
    df['Daily_Return'] = df['Close'].pct_change()
    return df

def calculate_rolling_metrics(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """
    Calculates rolling mean of the price and rolling volatility of returns.
    Uses 'Close' for price calculations.

    Args:
        df (pd.DataFrame): DataFrame with 'Close' and 'Daily_Return'.
        window (int): The rolling window size.

    Returns:
        pd.DataFrame: DataFrame with new rolling metric columns.
    """
    df[f'Rolling_Mean_Price_{window}d'] = df['Close'].rolling(window=window).mean()
    df[f'Rolling_Volatility_{window}d'] = df['Daily_Return'].rolling(window=window).std()
    return df

def scale_data(df: pd.DataFrame, column: str = 'Close') -> pd.DataFrame:
    """
    Applies MinMax scaling to a specified column (default 'Close').

    Args:
        df (pd.DataFrame): DataFrame with the column to scale.
        column (str): Column name to scale (default 'Close').

    Returns:
        pd.DataFrame: DataFrame with a new scaled column.
    """
    scaler = MinMaxScaler()
    df[f'{column}_Scaled'] = scaler.fit_transform(df[[column]])
    print(f"{column} scaled successfully using MinMaxScaler.")
    return df