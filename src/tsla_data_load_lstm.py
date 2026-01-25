import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_tsla_for_lstm(
    filepath: str = "data/processed/TSLA_cleaned.csv",
    target_col: str = "Return"
):
    """
    Load processed TSLA data and prepare target column for LSTM.
    
    Returns:
        df: original dataframe with Date index
        series_scaled: scaled numpy array of target column
        scaler: fitted MinMaxScaler object
    """
    # Load CSV
    df = pd.read_csv(filepath, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    df.set_index("Date", inplace=True)

    # Drop NaNs in target column
    series = df[target_col].dropna().values.reshape(-1, 1)

    # Scale target column to [-1,1] for LSTM
    scaler = MinMaxScaler(feature_range=(-1, 1))
    series_scaled = scaler.fit_transform(series)

    return df, series_scaled, scaler


def split_train_test_lstm(series_scaled, df, split_date="2025-01-01"):
    """
    Chronological train-test split for LSTM.
    Handles mismatch caused by dropping NaNs.
    """
    # Align index with series_scaled
    dates_scaled = df.index[~df["Return"].isna()]

    train_scaled = series_scaled[dates_scaled < split_date]
    test_scaled = series_scaled[dates_scaled >= split_date]

    return train_scaled, test_scaled

