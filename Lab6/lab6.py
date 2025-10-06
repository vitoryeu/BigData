
import os
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import time, numpy as np, pandas as pd, matplotlib.pyplot as plt
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pandas_datareader import data as pdr

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

def load_tsla_stooq_5y():
    for sym in ["TSLA", "tsla", "TSLA.US", "tsla.us"]:
        try:
            df = pdr.DataReader(sym, "stooq")
            if df is not None and not df.empty:
                df = df.sort_index()
                end = df.index.max()
                start = end - pd.Timedelta(days=365*5 + 30)
                df = df.loc[df.index >= start].rename(columns=str.title)
                if "Close" in df.columns:
                    return df
        except Exception:
            pass
    raise RuntimeError("TSLA not available from Stooq")

def create_sequences_raw(v, L=60):
    X, y = [], []
    for i in range(len(v) - L):
        X.append(v[i:i+L])
        y.append(v[i+L])
    return np.array(X), np.array(y)

def metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {"MSE": mse, "RMSE": sqrt(mse), "MAE": mean_absolute_error(y_true, y_pred)}

def main():
    outdir = "outputs_lab6"
    os.makedirs(outdir, exist_ok=True)

    print("Loading data...")
    df = load_tsla_stooq_5y()
    close = df["Close"].dropna().astype(float).asfreq("B").ffill()
    pd.DataFrame({"Close": close}).to_csv(os.path.join(outdir, "tsla_stooq_raw.csv"))
    print(f"Rows: {len(close)}")

    values = close.values.reshape(-1, 1)
    seq_len = 60
    X_raw, y_raw = create_sequences_raw(values, seq_len)
    split = int(0.8 * len(X_raw))

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_raw[:split].reshape(-1, 1)).reshape(-1, seq_len, 1)
    y_train = scaler.transform(y_raw[:split].reshape(-1, 1))
    X_test  = scaler.transform(X_raw[split:].reshape(-1, 1)).reshape(-1, seq_len, 1)
    y_test  = scaler.transform(y_raw[split:].reshape(-1, 1))

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    tf.keras.utils.set_random_seed(42)
    model = Sequential([
        tf.keras.Input(shape=(seq_len, 1)),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    print("Training LSTM (50 epochs)...")
    t0 = time.time()
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        verbose=0
    )
    t_lstm = time.time() - t0
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_test_inv = scaler.inverse_transform(y_test).ravel()
    y_pred_inv = scaler.inverse_transform(y_pred_scaled).ravel()
    lstm_m = metrics(y_test_inv, y_pred_inv)
    print(f"LSTM RMSE: {lstm_m['RMSE']:.3f}, MAE: {lstm_m['MAE']:.3f}")

    plt.figure(figsize=(12,6))
    plt.plot(y_test_inv, label="Actual")
    plt.plot(y_pred_inv, label="LSTM")
    plt.title("TSLA Close: Actual vs LSTM (Test)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plot_lstm_vs_actual.png"), dpi=150)
    plt.close()

    print("Baselines...")
    split_raw = seq_len + split
    train_close = close.iloc[:split_raw]
    test_close  = close.iloc[split_raw:]
    horizon = len(y_test_inv)

    t0 = time.time()
    try:
        arima = sm.tsa.ARIMA(train_close, order=(5,1,0), enforce_stationarity=False, enforce_invertibility=False).fit()
        arima_fc = arima.forecast(steps=horizon)
        t_arima = time.time() - t0
    except Exception:
        arima_fc = pd.Series([np.nan]*horizon, index=test_close.index[:horizon])
        t_arima = float("nan")

    t0 = time.time()
    try:
        es = ExponentialSmoothing(train_close, trend="add", seasonal=None).fit(optimized=True)
        es_fc = es.forecast(horizon)
        t_es = time.time() - t0
    except Exception:
        es_fc = pd.Series([np.nan]*horizon, index=test_close.index[:horizon])
        t_es = float("nan")

    actual = test_close.iloc[:horizon].values
    arima_m = metrics(actual, arima_fc.values)
    es_m    = metrics(actual, es_fc.values)

    print(f"ARIMA RMSE: {arima_m['RMSE']:.3f}, MAE: {arima_m['MAE']:.3f}")
    print(f"ES RMSE: {es_m['RMSE']:.3f}, MAE: {es_m['MAE']:.3f}")

    plt.figure(figsize=(12,6))
    idx = test_close.index[:horizon]
    plt.plot(idx, actual, label="Actual")
    plt.plot(idx, y_pred_inv, label="LSTM")
    plt.plot(idx, arima_fc.values, label="ARIMA")
    plt.plot(idx, es_fc.values, label="Exp. Smoothing")
    plt.title("TSLA Close: Actual vs Forecasts (Test Horizon)")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plot_all_models.png"), dpi=150)
    plt.close()

    comp = pd.DataFrame([
        {"Model":"LSTM",          "MSE":lstm_m["MSE"],  "RMSE":lstm_m["RMSE"],  "MAE":lstm_m["MAE"],  "TrainTimeSec":t_lstm, "Complexity":"High"},
        {"Model":"ARIMA(5,1,0)",  "MSE":arima_m["MSE"], "RMSE":arima_m["RMSE"], "MAE":arima_m["MAE"], "TrainTimeSec":t_arima, "Complexity":"Medium"},
        {"Model":"Exp. Smoothing","MSE":es_m["MSE"],    "RMSE":es_m["RMSE"],    "MAE":es_m["MAE"],    "TrainTimeSec":t_es,   "Complexity":"Low"},
    ])
    comp.to_csv(os.path.join(outdir, "comparison_metrics.csv"), index=False)

    preds = pd.DataFrame({
        "Date": idx,
        "Actual": actual,
        "LSTM": y_pred_inv,
        "ARIMA": arima_fc.values,
        "ExpSmoothing": es_fc.values
    }).set_index("Date")
    preds.to_csv(os.path.join(outdir, "predictions_test_horizon.csv"))

if __name__ == "__main__":
    main()
