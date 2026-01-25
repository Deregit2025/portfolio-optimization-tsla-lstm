# scripts/train_arima_tsla.py

from pathlib import Path
import pickle
import pandas as pd

from src.tsla_data_load import load_tsla_processed_data, split_train_test_by_date
from src.arima_model import train_arima_model, arima_forecast
from src.forecast import evaluate_forecast

DATA_PATH = Path("data/processed/TSLA_cleaned.csv")
SPLIT_DATE = "2025-01-01"


def main():
    # =======================
    # 1. Prepare models directory
    # =======================
    MODELS_DIR = Path("models")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARIMA_MODEL_PATH = MODELS_DIR / "arima_tsla_model.pkl"

    # =======================
    # 2. Load processed TSLA data
    # =======================
    df = load_tsla_processed_data(DATA_PATH)

    # Use stationary return series
    train, test = split_train_test_by_date(
        df=df,
        target_col="Return",
        split_date=SPLIT_DATE
    )

    # =======================
    # 3. Train ARIMA
    # =======================
    arima_model = train_arima_model(train)

    print("\nSelected ARIMA order:")
    print(arima_model.order)

    # =======================
    # 4. Forecast
    # =======================
    predictions = arima_forecast(arima_model, n_periods=len(test))

    # =======================
    # 5. Evaluate Forecast
    # =======================
    metrics = evaluate_forecast(test.values, predictions)

    print("\nARIMA Performance Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.6f}")

    # =======================
    # 6. Save ARIMA model
    # =======================
    with open(ARIMA_MODEL_PATH, "wb") as f:
        pickle.dump(arima_model, f)
    print(f"\nTrained ARIMA model saved at: {ARIMA_MODEL_PATH}")

    # =======================
    # 7. Optional: LSTM comparison
    # =======================
    try:
        from src.tsla_data_load_lstm import load_tsla_for_lstm, split_train_test_lstm
        from src.tsla_sequence_builder import create_sequences
        from tensorflow.keras.models import load_model
        import numpy as np
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        # Updated to .keras format
        LSTM_MODEL_PATH = MODELS_DIR / "lstm_tsla_model.keras"
        lstm_model = load_model(LSTM_MODEL_PATH)

        # Prepare LSTM test set
        _, series_scaled, scaler = load_tsla_for_lstm(DATA_PATH)
        train_scaled, test_scaled = split_train_test_lstm(series_scaled, df, split_date=SPLIT_DATE)

        SEQ_LEN = 60
        X_test, y_test = create_sequences(test_scaled, sequence_length=SEQ_LEN)
        y_pred_scaled = lstm_model.predict(X_test, verbose=0)

        # Inverse scale
        y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        # Align LSTM index
        lstm_index = df.index[~df["Return"].isna()][len(train_scaled) + SEQ_LEN:]
        y_pred = pd.Series(y_pred, index=lstm_index)
        y_true = pd.Series(y_true, index=lstm_index)

        # Metrics function
        def evaluate(y_true, y_pred):
            mae = mean_absolute_error(y_true, y_pred)
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
            return mae, rmse, mape

        # ARIMA metrics
        arima_mae = metrics["MAE"]
        arima_rmse = metrics["RMSE"]
        arima_mape = metrics["MAPE"]

        # LSTM metrics
        lstm_mae, lstm_rmse, lstm_mape = evaluate(y_true.values, y_pred.values)

        # Comparison Table
        comparison_df = pd.DataFrame({
            "ARIMA": [arima_mae, arima_rmse, arima_mape],
            "LSTM": [lstm_mae, lstm_rmse, lstm_mape]
        }, index=["MAE", "RMSE", "MAPE"])

        print("\nModel Comparison Table:")
        print(comparison_df)

    except Exception as e:
        print("\nLSTM comparison skipped (model or modules not found).")
        print(e)


if __name__ == "__main__":
    main()
