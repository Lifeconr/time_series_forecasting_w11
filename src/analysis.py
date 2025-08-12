# analysis.py

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
import matplotlib.pyplot as plt

def perform_adf_test(series: pd.Series):
    """
    Performs the Augmented Dickey-Fuller test to check for stationarity.

    Args:
        series (pd.Series): The time series data to test.
    """
    series = series.dropna()
    print(f'--- ADF Test Results for "{series.name}" ---')
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value:.4f}')

    if result[1] <= 0.05:
        print("\nConclusion: The series is likely stationary (p-value <= 0.05).")
    else:
        print("\nConclusion: The series is likely non-stationary (p-value > 0.05).")
    print("-" * 40)

def detect_outliers(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """
    Detects outliers using Z-score method, handling NaNs consistently.

    Args:
        series (pd.Series): Series to check (e.g., daily returns).
        threshold (float): Z-score threshold (default 3 for extreme outliers).

    Returns:
        pd.Series: Outliers with dates and values.
    """
    # Create a copy of the series with NaNs dropped
    clean_series = series.dropna()
    if clean_series.empty:
        print(f"No valid data for {series.name} after dropping NaNs.")
        return pd.Series(dtype=float)

    # Compute Z-scores on the cleaned series
    z_scores = np.abs(stats.zscore(clean_series))
    
    # Create a boolean mask aligned with clean_series index
    outliers_mask = z_scores > threshold
    outliers = clean_series[outliers_mask]
    
    print(f"Detected {len(outliers)} outliers beyond {threshold} std in {series.name}.")
    return outliers

def decompose_series(series: pd.Series, period: int = 252) -> None:
    """
    Decomposes series into trend, seasonal, and residual (annual period).

    Args:
        series (pd.Series): Series to decompose.
        period (int): Period for seasonality (252 trading days ~1 year).
    """
    decomp = seasonal_decompose(series.dropna(), model='multiplicative', period=period)
    decomp.plot()
    plt.title(f'Decomposition of {series.name}')
    plt.show()

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    daily_risk_free = (1 + risk_free_rate)**(1/252) - 1
    excess_returns = returns - daily_risk_free
    annualized_sharpe = np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
    return annualized_sharpe

def calculate_var(returns: pd.Series, confidence_level: float = 0.95) -> float:
    var = returns.quantile(1 - confidence_level)
    return -var  # Positive loss value