import pandas as pd


def load_tsla_processed_data(filepath: str) -> pd.DataFrame:
    """
    Load processed TSLA data from CSV.
    Task 1 output is assumed.
    """
    df = pd.read_csv(filepath, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    df.set_index("Date", inplace=True)
    return df


def split_train_test_by_date(
    df: pd.DataFrame,
    target_col: str,
    split_date: str
):
    """
    Chronological train-test split for time series data.
    Drops NaNs in target column (required for ARIMA).
    """
    series = df[target_col].dropna()

    train = series.loc[series.index < split_date]
    test = series.loc[series.index >= split_date]

    return train, test
