# src/optimization.py

import pandas as pd
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.plotting import plot_efficient_frontier, plot_weights

def get_portfolio_inputs(processed_data, tsla_forecast_return):
    """
    Prepares the inputs (expected returns, covariance matrix) for optimization.
    """
    # Combine daily returns into a single DataFrame
    returns_df = pd.DataFrame({
        ticker: df['Daily_Return'] for ticker, df in processed_data.items()
    }).dropna()

    # Covariance Matrix (annualized)
    cov_matrix = returns_df.cov() * 252

    # Expected Returns
    # Use historical mean for SPY and BND
    mu = expected_returns.mean_historical_return(returns_df, frequency=252)

    # Replace TSLA's historical return with our model's forecast
    mu['TSLA'] = tsla_forecast_return
    
    print("--- Expected Annual Returns ---")
    print((mu * 100).round(2).astype(str) + '%')
    print("\n--- Annual Covariance Matrix ---")
    print(cov_matrix.round(4))
    
    return mu, cov_matrix

def find_optimal_portfolios(exp_returns, cov_matrix):
    """
    Finds the Maximum Sharpe Ratio and Minimum Volatility portfolios.
    """
    ef = EfficientFrontier(exp_returns, cov_matrix)
    
    # Max Sharpe Ratio Portfolio
    weights_max_sharpe = ef.max_sharpe()
    cleaned_weights_max_sharpe = ef.clean_weights()
    perf_max_sharpe = ef.portfolio_performance(verbose=True, risk_free_rate=0.02)
    
    print("\n--- Max Sharpe Ratio Portfolio ---")
    print(cleaned_weights_max_sharpe)

    # Min Volatility Portfolio
    ef_min_vol = EfficientFrontier(exp_returns, cov_matrix) # Re-instantiate
    weights_min_vol = ef_min_vol.min_volatility()
    cleaned_weights_min_vol = ef_min_vol.clean_weights()
    perf_min_vol = ef_min_vol.portfolio_performance(verbose=True, risk_free_rate=0.02)
    
    print("\n--- Minimum Volatility Portfolio ---")
    print(cleaned_weights_min_vol)

    return ef, cleaned_weights_max_sharpe, perf_max_sharpe, perf_min_vol