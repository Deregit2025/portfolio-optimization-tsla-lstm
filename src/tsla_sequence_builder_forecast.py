import numpy as np

def create_forecast_sequence(series, sequence_length=20):
    """
    Prepare the most recent sequence for forecasting with LSTM.

    Parameters
    ----------
    series : array-like
        1D array of historical values (e.g., returns or prices)
    sequence_length : int
        Number of past timesteps to use for prediction

    Returns
    -------
    X : np.ndarray
        3D array shaped (1, sequence_length, 1) ready for LSTM prediction
    """
    # Take the last `sequence_length` values
    seq = series[-sequence_length:]
    
    # Reshape for LSTM: (samples, timesteps, features)
    X = np.array(seq).reshape((1, sequence_length, 1))
    
    return X

def iterative_forecast(model, series, steps=126, sequence_length=20):
    """
    Generate multi-step forecast iteratively.

    Parameters
    ----------
    model : keras.Model
        Trained LSTM model
    series : array-like
        1D array of historical values
    steps : int
        Number of future steps to forecast
    sequence_length : int
        Sequence length for the model input

    Returns
    -------
    forecast : np.ndarray
        Forecasted values array of length `steps`
    """
    series_copy = series.copy()
    forecast = []

    for _ in range(steps):
        X_input = create_forecast_sequence(series_copy, sequence_length)
        next_pred = model.predict(X_input, verbose=0)[0][0]
        forecast.append(next_pred)
        series_copy = np.append(series_copy, next_pred)  # append for next step

    return np.array(forecast)
