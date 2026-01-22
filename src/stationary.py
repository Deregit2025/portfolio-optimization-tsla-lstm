from statsmodels.tsa.stattools import adfuller

def check_stationarity(series, ticker=""):
    """Perform Augmented Dickey-Fuller test"""
    result = adfuller(series.dropna())
    print(f"--- {ticker} ADF Test ---")
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    if result[1] < 0.05:
        print("Series is stationary ✅\n")
    else:
        print("Series is non-stationary ❌\n")
