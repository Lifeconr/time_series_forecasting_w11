import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from forecasting_models import LSTMModel, create_sequences, scaler  # Assuming model is saved

# Load data and model
df = pd.read_csv('data/TSLA_cleaned.csv', index_col='Date', parse_dates=True)
last_sequence = df['Close'].values[-10:].reshape(1, 10, 1)
last_sequence = scaler.transform(last_sequence)

model.eval()
with torch.no_grad():
    future_steps = 252  # ~6 months of trading days
    future_predictions = []
    current_seq = torch.FloatTensor(last_sequence)
    for _ in range(future_steps):
        next_pred = model(current_seq)
        future_predictions.append(next_pred.numpy()[0, 0])
        current_seq = torch.cat((current_seq[:, 1:, :], torch.FloatTensor(next_pred.reshape(1, 1, 1))), dim=1)
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Plot
plt.figure(figsize=(14, 7))
plt.plot(df.index, df['Close'], label='Historical')
future_dates = pd.date_range(start=df.index[-1], periods=future_steps + 1, freq='B')[1:]
plt.plot(future_dates, future_predictions, label='Forecast', color='red')
plt.fill_between(future_dates, future_predictions * 0.9, future_predictions * 1.1, color='red', alpha=0.1, label='Confidence Interval (±10%)')
plt.title('TSLA Price Forecast (Aug 2025 - Jan 2026)')
plt.legend()
plt.show()