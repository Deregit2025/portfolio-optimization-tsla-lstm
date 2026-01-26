# src/forecast_returns.py
import pandas as pd
from pathlib import Path
from statsmodels.tsa.arima.model import ARIMA

def forecast_tsla_returns(tsla_csv_path: str, output_csv_path: str,
                          arima_order=(2,1,2), forecast_horizon=252):
    """
    Forecast TSLA daily returns using ARIMA and save forecast to CSV.
    
    Parameters
    ----------
    tsla_csv_path : str
        Path to TSLA cleaned CSV (must contain 'Close' column)
    output_csv_path : str
        Path to save forecasted returns CSV
    arima_order : tuple
        ARIMA(p,d,q) order
    forecast_horizon : int
        Number of trading days to forecast (default 252 ~ 1 year)
    """
    # Load cleaned TSLA data
    tsla_data = pd.read_csv(tsla_csv_path, index_col="Date", parse_dates=True)
    tsla_data = tsla_data.sort_index()

    # Compute daily returns
    tsla_returns = tsla_data["Close"].pct_change().dropna()

    # Fit ARIMA model
    arima_model = ARIMA(tsla_returns, order=arima_order).fit()

    # Forecast
    forecast_obj = arima_model.get_forecast(steps=forecast_horizon)
    forecast_mean = forecast_obj.predicted_mean

    # Save forecast to CSV
    forecast_df = pd.DataFrame({"forecast": forecast_mean})
    forecast_df.index.name = "Date"
    forecast_df.to_csv(output_csv_path)

    print(f"Forecasted {forecast_horizon} TSLA returns saved to {output_csv_path}")

# Example usage
if __name__ == "__main__":
    processed_path = Path("data/processed")
    tsla_csv = processed_path / "TSLA_cleaned.csv"
    forecast_csv = processed_path / "TSLA_forecast_returns.csv"

    forecast_tsla_returns(tsla_csv_path=tsla_csv, output_csv_path=forecast_csv)
