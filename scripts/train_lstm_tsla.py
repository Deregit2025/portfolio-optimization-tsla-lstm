
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from src.tsla_data_load_lstm import load_tsla_for_lstm, split_train_test_lstm
from src.tsla_sequence_builder import create_sequences

# -----------------------------
# Config
# -----------------------------
BASE_DIR = Path.cwd()
CSV_FILE = BASE_DIR / "data" / "processed" / "TSLA_cleaned.csv"
SPLIT_DATE = "2025-01-01"
SEQ_LEN = 60
EPOCHS = 50
BATCH_SIZE = 32

MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
MODEL_SAVE_PATH = MODELS_DIR / "lstm_tsla_model.keras"  # ✅ use Keras native format

# -----------------------------
# Main
# -----------------------------
def main():
    # 1️⃣ Load data for LSTM
    df, series_scaled, scaler = load_tsla_for_lstm(CSV_FILE)

    # 2️⃣ Split train/test
    train_scaled, test_scaled = split_train_test_lstm(series_scaled, df, split_date=SPLIT_DATE)

    # 3️⃣ Create sequences
    X_train, y_train = create_sequences(train_scaled, sequence_length=SEQ_LEN)
    X_test, y_test = create_sequences(test_scaled, sequence_length=SEQ_LEN)

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # 4️⃣ Build LSTM model
    model = Sequential([
        Input(shape=(SEQ_LEN, 1)),
        LSTM(50, return_sequences=False),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")  # mse is fine in Keras native format

    # 5️⃣ Train
    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[es],
        verbose=1
    )

    # 6️⃣ Predict
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1,1))
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1))

    # 7️⃣ Compute metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print("\nLSTM Performance Metrics:")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAPE: {mape:.3f}%")

    # 8️⃣ Save model in Keras native format
    model.save(MODEL_SAVE_PATH)
    print(f"\nTrained LSTM model saved at: {MODEL_SAVE_PATH}")

    # 9️⃣ Comparison table (fill ARIMA metrics manually from previous run)
    arima_metrics = {
        "MAE": 0.028875,
        "RMSE": 0.039440,
        "MAPE": 125.79
    }

    lstm_metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape
    }

    comparison_df = pd.DataFrame({
        "ARIMA": arima_metrics,
        "LSTM": lstm_metrics
    })

    print("\nModel Comparison Table:")
    print(comparison_df)

if __name__ == "__main__":
    main()
