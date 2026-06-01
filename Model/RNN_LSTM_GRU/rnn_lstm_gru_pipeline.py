from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GRU, LSTM, Dense, Input, SimpleRNN


DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[2] / "Dataset" / "data_VNINDEX.csv"
)

SPLIT_RATIOS = {
    "652510": (0.65, 0.25, 0.10),
    "702010": (0.70, 0.20, 0.10),
    "751510": (0.75, 0.15, 0.10),
}

COLUMN_MAP = {
    "time": "Date",
    "date": "Date",
    "Ngay": "Date",
    "Ngày": "Date",
    "close": "Close",
    "Lan cuoi": "Close",
    "Lần cuối": "Close",
}


def parse_number(value: object) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)

    text = str(value).strip().replace('"', "")
    if not text:
        return np.nan

    multiplier = 1.0
    suffix = text[-1].upper()
    if text.endswith("%"):
        text = text[:-1]
    elif suffix == "K":
        multiplier = 1_000.0
        text = text[:-1]
    elif suffix == "M":
        multiplier = 1_000_000.0
        text = text[:-1]
    elif suffix == "B":
        multiplier = 1_000_000_000.0
        text = text[:-1]

    return float(text.replace(",", "")) * multiplier


def parse_dates(values: pd.Series) -> pd.Series:
    sample = values.dropna().astype(str).head(1)
    if not sample.empty and "-" in sample.iloc[0] and len(sample.iloc[0].split("-")[0]) == 4:
        return pd.to_datetime(values, errors="coerce")
    return pd.to_datetime(values, dayfirst=True, errors="coerce")


def load_close_prices(csv_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df.rename(columns=COLUMN_MAP)

    if "Date" in df.columns:
        df["Date"] = parse_dates(df["Date"])
        df = df.dropna(subset=["Date"]).sort_values("Date")

    if "Close" not in df.columns:
        raise ValueError("Input CSV must contain a close price column.")

    df["Close"] = df["Close"].map(parse_number)
    df = df.dropna(subset=["Close"]).reset_index(drop=True)
    return df[["Close"]]


def split_series(
    values: np.ndarray,
    split_code: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    train_ratio, val_ratio, _ = SPLIT_RATIOS[split_code]
    n = len(values)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    train = values[:train_size]
    val = values[train_size : train_size + val_size]
    test = values[train_size + val_size :]
    sizes = {"total": n, "train": len(train), "val": len(val), "test": len(test)}
    return train, val, test, sizes


def scale_splits(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    val_scaled = scaler.transform(val)
    test_scaled = scaler.transform(test)
    return train_scaled, val_scaled, test_scaled, scaler


def create_windows(data: np.ndarray, time_step: int) -> tuple[np.ndarray, np.ndarray]:
    x_values, y_values = [], []
    for i in range(time_step, len(data)):
        x_values.append(data[i - time_step : i])
        y_values.append(data[i])
    return np.array(x_values), np.array(y_values)


def split_internal_validation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    internal_val_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    val_size = max(1, int(len(x_train) * internal_val_ratio))
    return (
        x_train[:-val_size],
        y_train[:-val_size],
        x_train[-val_size:],
        y_train[-val_size:],
    )


def build_model(model_type: str, input_shape: tuple[int, int], units: int) -> tf.keras.Model:
    recurrent_layer = {
        "rnn": SimpleRNN,
        "lstm": LSTM,
        "gru": GRU,
    }[model_type]

    model = Sequential(
        [
            Input(shape=input_shape),
            recurrent_layer(units, return_sequences=False),
            Dense(1),
        ]
    )
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model


def fit_model(
    model: tf.keras.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_internal_val: np.ndarray,
    y_internal_val: np.ndarray,
    epochs: int,
    batch_size: int,
    patience: int,
) -> tf.keras.callbacks.History:
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        )
    ]
    return model.fit(
        x_train,
        y_train,
        validation_data=(x_internal_val, y_internal_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=callbacks,
    )


def recursive_predict(
    model: tf.keras.Model,
    seed_window: np.ndarray,
    steps: int,
) -> np.ndarray:
    history = seed_window.astype(float).copy()
    predictions = []
    for _ in range(steps):
        x_input = history[-len(seed_window) :].reshape(1, len(seed_window), history.shape[1])
        pred = model.predict(x_input, verbose=0)
        predictions.append(pred[0])
        history = np.vstack([history, pred])
    return np.array(predictions)


def inverse_close(values: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.inverse_transform(np.asarray(values).reshape(-1, 1)).ravel()


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    non_zero = y_true != 0
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100),
    }


def predict_block(
    model: tf.keras.Model,
    data_scaled: np.ndarray,
    scaler: StandardScaler,
    time_step: int,
    forecast_mode: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if forecast_mode == "one_step":
        x_values, y_values = create_windows(data_scaled, time_step)
        pred_scaled = model.predict(x_values, verbose=0)
        y_true = inverse_close(y_values, scaler)
        y_pred = inverse_close(pred_scaled, scaler)
    else:
        seed_window = data_scaled[:time_step]
        true_scaled = data_scaled[time_step:]
        pred_scaled = recursive_predict(model, seed_window, len(true_scaled))
        y_true = inverse_close(true_scaled, scaler)
        y_pred = inverse_close(pred_scaled, scaler)

    return y_true, y_pred, evaluate_predictions(y_true, y_pred)


def plot_predictions(
    train_raw: np.ndarray,
    val_raw: np.ndarray,
    test_raw: np.ndarray,
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    time_step: int,
    split_code: str,
    output_path: str | Path,
    forecast_mode: str,
) -> None:
    x_train = np.arange(len(train_raw))
    x_val = np.arange(len(train_raw), len(train_raw) + len(val_raw))
    x_test = np.arange(len(train_raw) + len(val_raw), len(train_raw) + len(val_raw) + len(test_raw))

    plt.figure(figsize=(15, 6))
    plt.plot(x_train, train_raw.ravel(), label="Train", color="blue")
    plt.plot(x_val, val_raw.ravel(), label="Validate", color="red")
    plt.plot(x_test, test_raw.ravel(), label="Test", color="orange")

    colors = {"rnn": "cyan", "lstm": "purple", "gru": "green"}
    if forecast_mode == "one_step":
        pred_x = x_test[time_step:]
    else:
        pred_x = x_test[time_step:]

    for name, (_true, pred) in predictions.items():
        plt.plot(pred_x, pred, label=f"{name.upper()} Predict", color=colors[name])

    plt.title(f"RNN/LSTM/GRU - Predict VNINDEX price ({split_code})")
    plt.xlabel("Time Step")
    plt.ylabel("Close price")
    plt.legend()
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_pipeline(args: argparse.Namespace) -> dict[str, dict[str, dict[str, float]]]:
    tf.keras.utils.set_random_seed(args.seed)
    df = load_close_prices(args.data)
    values = df[["Close"]].to_numpy(dtype=float)
    train_raw, val_raw, test_raw, sizes = split_series(values, args.split)
    train_scaled, val_scaled, test_scaled, scaler = scale_splits(train_raw, val_raw, test_raw)

    x_train_full, y_train_full = create_windows(train_scaled, args.time_step)
    x_fit, y_fit, x_internal_val, y_internal_val = split_internal_validation(
        x_train_full,
        y_train_full,
        args.internal_val_ratio,
    )

    model_names = ["rnn", "lstm", "gru"] if args.model == "all" else [args.model]
    results: dict[str, dict[str, dict[str, float]]] = {}
    test_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for model_name in model_names:
        model = build_model(model_name, (args.time_step, train_scaled.shape[1]), args.units)
        history = fit_model(
            model,
            x_fit,
            y_fit,
            x_internal_val,
            y_internal_val,
            args.epochs,
            args.batch_size,
            args.patience,
        )
        _val_true, _val_pred, val_metrics = predict_block(
            model,
            val_scaled,
            scaler,
            args.time_step,
            args.forecast_mode,
        )
        test_true, test_pred, test_metrics = predict_block(
            model,
            test_scaled,
            scaler,
            args.time_step,
            args.forecast_mode,
        )
        results[model_name] = {
            "validation": val_metrics,
            "test": test_metrics,
            "best_train_val_loss": float(min(history.history["val_loss"])),
        }
        test_predictions[model_name] = (test_true, test_pred)

    if args.plot_path:
        plot_predictions(
            train_raw,
            val_raw,
            test_raw,
            test_predictions,
            args.time_step,
            args.split,
            args.plot_path,
            args.forecast_mode,
        )

    print(f"Data: {Path(args.data)}")
    print(f"Split sizes: {sizes}")
    print(f"Forecast mode: {args.forecast_mode}")
    for model_name, metrics in results.items():
        print(f"{model_name.upper()}: {metrics}")

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RNN/LSTM/GRU price prediction on VNINDEX data.")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, type=Path)
    parser.add_argument("--split", default="702010", choices=sorted(SPLIT_RATIOS))
    parser.add_argument("--model", default="all", choices=["rnn", "lstm", "gru", "all"])
    parser.add_argument("--time-step", default=30, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--units", default=50, type=int)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--internal-val-ratio", default=0.15, type=float)
    parser.add_argument("--forecast-mode", default="recursive", choices=["recursive", "one_step"])
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--plot-path",
        default=Path(__file__).resolve().parent / "rnn_lstm_gru_prediction.png",
        type=Path,
        help="Set to an empty string to skip plot output.",
    )
    args = parser.parse_args()
    if args.plot_path == Path("."):
        args.plot_path = None
    return args


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
