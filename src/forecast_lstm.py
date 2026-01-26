import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from src.tsla_sequence_builder_forecast import iterative_forecast

# -----------------------------
# 1. Load TSLA data and LSTM model
# -----------------------------
tsla = pd.read_csv("data/processed/TSLA_cleaned.csv", parse_dates=["Date"])
tsla.sort_values("Date", inplace=True)

# The LSTM was trained on returns
tsla_returns = tsla["Return"].values

model = load_model("models/lstm_tsla_model.keras")

# -----------------------------
# 2. Forecast next 6 months (~126 trading days)
# -----------------------------
forecast_steps = 126  # ~6 months
forecasted_returns = iterative_forecast(model, tsla_returns, steps=forecast_steps, sequence_length=20)

# -----------------------------
# 3. Reconstruct TSLA prices from forecasted returns
# -----------------------------
last_price = tsla["Close"].iloc[-1]
forecasted_prices = [last_price * np.exp(forecasted_returns[0])]  # first day

for r in forecasted_returns[1:]:
    forecasted_prices.append(forecasted_prices[-1] * np.exp(r))

forecasted_prices = np.array(forecasted_prices)

# Prepare forecasted dataframe
forecast_dates = pd.bdate_range(start=tsla["Date"].iloc[-1] + pd.Timedelta(days=1), periods=forecast_steps)
tsla_forecast_df = pd.DataFrame({
    "Date": forecast_dates,
    "Forecasted_Return": forecasted_returns,
    "Forecasted_Close": forecasted_prices
})

tsla_forecast_df.to_csv("data/processed/TSLA_forecasted_6mo.csv", index=False)
print("TSLA 6-month forecast saved to data/processed/TSLA_forecasted_6mo.csv")

# -----------------------------
# 4. Compute expected returns for Task 4
# -----------------------------
# TSLA expected return: mean of forecasted returns (annualized)
mu_tsla = np.mean(forecasted_returns) * 252  # annualized

# Load SPY and BND, compute historical average returns (annualized)
spy = pd.read_csv("data/processed/SPY_cleaned.csv", parse_dates=["Date"])
bnd = pd.read_csv("data/processed/BND_cleaned.csv", parse_dates=["Date"])

mu_spy = spy["Return"].mean() * 252
mu_bnd = bnd["Return"].mean() * 252

# Expected returns vector
expected_returns = pd.Series({
    "TSLA": mu_tsla,
    "SPY": mu_spy,
    "BND": mu_bnd
})

print("\nExpected Returns (annualized):")
print(expected_returns)

# Optionally save for Task 4
expected_returns.to_csv("data/processed/expected_returns.csv")
print("Expected returns saved to data/processed/expected_returns.csv")
