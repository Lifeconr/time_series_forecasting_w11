import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
data = {ticker: pd.read_csv(f'data/{ticker}_cleaned.csv', index_col='Date', parse_dates=True) for ticker in ['TSLA', 'BND', 'SPY']}
returns = pd.concat([df['Daily_Return'] for df in data.values()], axis=1, keys=['TSLA', 'BND', 'SPY']).dropna()
backtest_period = returns['2024-08-01':'2025-07-31']

# Strategy and Benchmark
weights_strategy = {'TSLA': 0.25, 'BND': 0.20, 'SPY': 0.55}
weights_benchmark = {'SPY': 0.60, 'BND': 0.40}

portfolio_returns_strategy = (backtest_period * pd.DataFrame([weights_strategy] * len(backtest_period))).sum(axis=1)
portfolio_returns_benchmark = (backtest_period * pd.DataFrame([weights_benchmark] * len(backtest_period))).sum(axis=1)

# Cumulative Returns
cumulative_strategy = (1 + portfolio_returns_strategy).cumprod()
cumulative_benchmark = (1 + portfolio_returns_benchmark).cumprod()

# Plot
plt.figure(figsize=(14, 7))
plt.plot(cumulative_strategy, label='Strategy (25% TSLA, 20% BND, 55% SPY)')
plt.plot(cumulative_benchmark, label='Benchmark (60% SPY, 40% BND)')
plt.title('Backtest Cumulative Returns (Aug 2024 - Jul 2025)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.show()

# Performance Metrics
total_return_strategy = cumulative_strategy.iloc[-1] - 1
total_return_benchmark = cumulative_benchmark.iloc[-1] - 1
sharpe_strategy = np.sqrt(252) * (portfolio_returns_strategy.mean() / portfolio_returns_strategy.std())
sharpe_benchmark = np.sqrt(252) * (portfolio_returns_benchmark.mean() / portfolio_returns_benchmark.std())

print(f"Strategy Total Return: {total_return_strategy:.2%}")
print(f"Benchmark Total Return: {total_return_benchmark:.2%}")
print(f"Strategy Sharpe Ratio: {sharpe_strategy:.2f}")
print(f"Benchmark Sharpe Ratio: {sharpe_benchmark:.2f}")