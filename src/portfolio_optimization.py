import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
import matplotlib.pyplot as plt

# Load data
data = {ticker: pd.read_csv(f'data/{ticker}_cleaned.csv', index_col='Date', parse_dates=True) for ticker in ['TSLA', 'BND', 'SPY']}
returns = pd.concat([df['Daily_Return'] for df in data.values()], axis=1, keys=['TSLA', 'BND', 'SPY']).dropna()

# Expected returns
tsla_forecast_return = (future_predictions[-1] / df['Close'][-1])**(252/6) - 1  # Annualized 6-month forecast return
hist_returns = returns.mean() * 252  # Annualized historical returns
expected_returns = pd.Series([tsla_forecast_return, hist_returns['BND'], hist_returns['SPY']], index=['TSLA', 'BND', 'SPY'])

# Covariance matrix
cov_matrix = risk_models.sample_cov(returns)

# Optimization
ef = EfficientFrontier(expected_returns, cov_matrix)
ef.max_sharpe(risk_free_rate=0.02)
weights_max_sharpe = ef.clean_weights()
ef.min_volatility()
weights_min_vol = ef.clean_weights()

# Efficient Frontier
frontier = ef.efficient_frontier()
plt.figure(figsize=(10, 6))
plt.scatter(frontier['volatility'], frontier['returns'], c=frontier['sharpe_ratio'], cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.scatter([w['volatility'] for w in [weights_max_sharpe, weights_min_vol]], 
            [w['expected_annual_return'] for w in [weights_max_sharpe, weights_min_vol]], 
            c='red', s=100, label=['Max Sharpe', 'Min Volatility'])
plt.xlabel('Annualized Volatility')
plt.ylabel('Annualized Return')
plt.title('Efficient Frontier')
plt.legend()
plt.show()

# Recommended Portfolio
ef = EfficientFrontier(expected_returns, cov_matrix)
ef.max_sharpe(risk_free_rate=0.02)
weights = ef.clean_weights()
returns, volatility, sharpe = ef.portfolio_performance(verbose=True)

print(f"Recommended Portfolio Weights: {weights}")
print(f"Expected Annual Return: {returns:.2%}")
print(f"Annualized Volatility: {volatility:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")