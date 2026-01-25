import numpy as np

def create_sequences(series, sequence_length=20):
    """
    Converts a 1D time series into LSTM sequences

    X shape: (samples, timesteps, features)
    y shape: (samples,)
    """
    X, y = [], []

    for i in range(sequence_length, len(series)):
        X.append(series[i-sequence_length:i])
        y.append(series[i])

    X = np.array(X)
    y = np.array(y)

    # LSTM expects 3D input
    X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, y
