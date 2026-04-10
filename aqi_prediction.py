import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
np.random.seed(42)
tf.random.set_seed(42)

DATA_PATH   = "air_quality_data.csv"   # path to your CSV
OUTPUT_PATH = "./"                     # folder to save plots

FEATURES   = ['PM2.5', 'PM10', 'Temperature', 'Humidity']
TARGET     = 'AQI'
TIME_STEPS = 5    # how many past readings to use per prediction
EPOCHS     = 50
BATCH_SIZE = 32
TEST_SPLIT = 0.2
# ─────────────────────────────────────────────────────────────────────────────

# 1. Load data
print("Loading data...")
data = pd.read_csv(DATA_PATH)
print(f"Shape: {data.shape}")
print(data.head())

# 2. Check for missing values and drop if any
print("\nMissing values:")
print(data.isnull().sum())
data = data.dropna()
print(f"Rows after dropping nulls: {len(data)}")

# 3. Normalise all columns together using one scaler
#    (kept as one scaler so inverse transform works correctly later)
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data[FEATURES + [TARGET]])

X_all = scaled[:, :-1]   # features
y_all = scaled[:, -1]    # target

# 4. Build sliding time windows
#    Each sample = past TIME_STEPS readings → predict next AQI
X_seq, y_seq = [], []
for i in range(TIME_STEPS, len(X_all)):
    X_seq.append(X_all[i - TIME_STEPS:i])
    y_seq.append(y_all[i])

X_seq = np.array(X_seq)   # shape: (samples, TIME_STEPS, features)
y_seq = np.array(y_seq)   # shape: (samples,)

print(f"\nSequence shape: X={X_seq.shape}, y={y_seq.shape}")

# 5. Train / test split (no shuffle — time series order must be preserved)
split    = int((1 - TEST_SPLIT) * len(X_seq))
X_train  = X_seq[:split]
X_test   = X_seq[split:]
y_train  = y_seq[:split]
y_test   = y_seq[split:]

print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

# 6. Build LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# 7. Early stopping — stops training if val_loss stops improving
early_stop = EarlyStopping(monitor='val_loss', patience=10,
                           restore_best_weights=True, verbose=1)

# 8. Train
print("\nTraining LSTM...")
history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test),
    callbacks=[early_stop],
    verbose=1
)

# 9. Evaluate on test set
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest MSE (normalised): {loss:.5f}")
print(f"Test MAE (normalised): {mae:.5f}")

# 10. Predict
y_pred = model.predict(X_test)

# 11. Inverse transform back to original AQI scale
#     Pair the last timestep's features with predictions/actuals
#     so inverse_transform can undo the full scaler
last_features = X_test[:, -1, :]   # shape: (n_test, n_features)

y_test_full   = np.hstack((last_features, y_test.reshape(-1, 1)))
y_pred_full   = np.hstack((last_features, y_pred))

y_test_actual = scaler.inverse_transform(y_test_full)[:, -1]
y_pred_actual = scaler.inverse_transform(y_pred_full)[:, -1]

# 12. Metrics on original AQI scale
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
mae  = mean_absolute_error(y_test_actual, y_pred_actual)
print(f"\nActual scale  →  RMSE: {rmse:.4f} | MAE: {mae:.4f}")

# 13. Plot 1 — Training vs Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'],     label='Training Loss (MSE)',   color='#378ADD', linewidth=1.5)
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)', color='#D85A30', linewidth=1.5)
plt.title('LSTM Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_PATH + "lstm_loss.png", dpi=150)
plt.close()
print("Saved: lstm_loss.png")

# 14. Plot 2 — Actual vs Predicted AQI
plt.figure(figsize=(12, 5))
plt.plot(y_test_actual, label='Actual AQI',    color='#378ADD', linewidth=1.5)
plt.plot(y_pred_actual, label='Predicted AQI', color='#D85A30', linewidth=1.5, linestyle='--')
plt.title('LSTM Actual vs Predicted AQI')
plt.xlabel('Sample')
plt.ylabel('AQI')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_PATH + "lstm_predictions.png", dpi=150)
plt.close()
print("Saved: lstm_predictions.png")
