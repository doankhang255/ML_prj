from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping


DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[2] / "Dataset" / "data_VNINDEX.csv"
)

SPLIT_RATIOS = {
    "652510": (0.65, 0.25, 0.10),
    "702010": (0.70, 0.20, 0.10),
    "751510": (0.75, 0.15, 0.10),
}

COLUMN_MAP = {
    "Ngay": "Date",
    "time": "Date",
    "date": "Date",
    "Lan cuoi": "Close",
    "close": "Close",
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
        raise ValueError("CSV must contain a close price column.")

    df["Close"] = df["Close"].map(parse_number)
    return df.dropna(subset=["Close"]).reset_index(drop=True)[["Close"]]


def split_raw_data(
    data: np.ndarray,
    split_code: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    train_ratio, val_ratio, _ = SPLIT_RATIOS[split_code]
    n = len(data)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    train_data = data[:train_size]
    val_data = data[train_size : train_size + val_size]
    test_data = data[train_size + val_size :]
    sizes = {
        "total": n,
        "train": len(train_data),
        "val": len(val_data),
        "test": len(test_data),
    }
    return train_data, val_data, test_data, sizes


def create_dataset(data: np.ndarray, look_back: int) -> tuple[np.ndarray, np.ndarray]:
    x_values, y_values = [], []
    for i in range(look_back, len(data)):
        x_values.append(data[i - look_back : i])
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


def transformer_encoder(
    inputs: tf.Tensor,
    head_size: int,
    num_heads: int,
    ff_dim: int,
    dropout: float,
) -> tf.Tensor:
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size,
        num_heads=num_heads,
        dropout=dropout,
    )(x, x)
    x = layers.Dropout(dropout)(x)
    x = layers.Add()([x, inputs])

    x_ff = layers.LayerNormalization(epsilon=1e-6)(x)
    x_ff = layers.Dense(ff_dim, activation="relu")(x_ff)
    x_ff = layers.Dropout(dropout)(x_ff)
    x_ff = layers.Dense(inputs.shape[-1])(x_ff)
    return layers.Add()([x, x_ff])


def build_cnn_transformer_model(
    input_shape: tuple[int, int],
    conv_filters: int,
    head_size: int,
    num_heads: int,
    ff_dim: int,
    lstm_units: int,
    dropout: float,
) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(filters=conv_filters, kernel_size=3, activation="relu", padding="same")(inputs)
    x = layers.Conv1D(filters=conv_filters, kernel_size=3, activation="relu", padding="same")(x)
    x = transformer_encoder(
        inputs=x,
        head_size=head_size,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=dropout,
    )
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=False))(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1)(x)

    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


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


def inverse_close(values: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    return scaler.inverse_transform(np.asarray(values).reshape(-1, 1)).ravel()


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    non_zero = y_true != 0
    return float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
    }


def evaluate_naive_previous_close(
    x_values: np.ndarray,
    y_values: np.ndarray,
    scaler: MinMaxScaler,
) -> dict[str, float]:
    y_true = inverse_close(y_values, scaler)
    y_pred = inverse_close(x_values[:, -1, :], scaler)
    return evaluate_predictions(y_true, y_pred)


def evaluate_naive_recursive(
    seed_window: np.ndarray,
    true_block: np.ndarray,
    scaler: MinMaxScaler,
) -> dict[str, float]:
    y_true = inverse_close(true_block, scaler)
    y_pred = inverse_close(np.repeat(seed_window[-1:], repeats=len(true_block), axis=0), scaler)
    return evaluate_predictions(y_true, y_pred)


def predict_block(
    model: tf.keras.Model,
    data_scaled: np.ndarray,
    scaler: MinMaxScaler,
    look_back: int,
    forecast_mode: str,
    seed_window: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    if forecast_mode == "one_step":
        x_values, y_values = create_dataset(data_scaled, look_back)
        pred_scaled = model.predict(x_values, verbose=0)
        y_true = inverse_close(y_values, scaler)
        y_pred = inverse_close(pred_scaled, scaler)
    elif forecast_mode == "recursive":
        if seed_window is None:
            raise ValueError("seed_window is required for recursive forecast mode.")
        true_scaled = data_scaled
        pred_scaled = recursive_predict(model, seed_window, len(true_scaled))
        y_true = inverse_close(true_scaled, scaler)
        y_pred = inverse_close(pred_scaled, scaler)
    else:
        raise ValueError("forecast_mode must be either 'recursive' or 'one_step'.")

    return y_true, y_pred, evaluate_predictions(y_true, y_pred)


def plot_predictions(
    train_raw: np.ndarray,
    val_raw: np.ndarray,
    test_raw: np.ndarray,
    train_pred: np.ndarray,
    val_pred: np.ndarray,
    test_pred: np.ndarray,
    look_back: int,
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

    train_x_pred = x_train[look_back:]
    if forecast_mode == "one_step":
        val_x_pred = x_val[look_back:]
        test_x_pred = x_test[look_back:]
    else:
        val_x_pred = x_val[look_back:]
        test_x_pred = x_test[look_back:]

    plt.plot(train_x_pred, train_pred, label="TrainPred", color="cyan")
    if forecast_mode == "one_step":
        plt.plot(val_x_pred, val_pred, label="ValidatePred", color="purple")
        plt.plot(test_x_pred, test_pred, label="Predict", color="green")
    else:
        plt.plot(x_val, val_pred, label="ValidatePred", color="purple")
        plt.plot(x_test, test_pred, label="Predict", color="green")
    plt.title(f"CNN+Transformer - Predict VNINDEX close ({split_code}, {forecast_mode})")
    plt.xlabel("Time step")
    plt.ylabel("Close price")
    plt.legend()
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_pipeline(
    csv_path: str | Path,
    split_code: str,
    look_back: int = 60,
    epochs: int = 50,
    batch_size: int = 32,
    patience: int = 10,
    internal_val_ratio: float = 0.15,
    forecast_mode: str = "recursive",
    plot_dir: str | Path | None = None,
    seed: int = 42,
    conv_filters: int = 32,
    head_size: int = 64,
    num_heads: int = 4,
    ff_dim: int = 128,
    lstm_units: int = 64,
    dropout: float = 0.1,
) -> dict[str, object]:
    tf.keras.utils.set_random_seed(seed)

    df = load_close_prices(csv_path)
    raw_data = df[["Close"]].to_numpy(dtype=float)
    train_raw, val_raw, test_raw, split_sizes = split_raw_data(raw_data, split_code)

    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_raw)
    val_data = scaler.transform(val_raw)
    test_data = scaler.transform(test_raw)

    x_train_full, y_train_full = create_dataset(train_data, look_back)
    x_val, y_val = create_dataset(val_data, look_back)
    x_test, y_test = create_dataset(test_data, look_back)

    if len(x_train_full) == 0 or len(x_val) == 0 or len(x_test) == 0:
        raise ValueError(
            "One of the splits is too small for the selected "
            f"look_back={look_back}. Try a smaller look_back."
        )

    x_fit, y_fit, x_internal_val, y_internal_val = split_internal_validation(
        x_train_full,
        y_train_full,
        internal_val_ratio,
    )
    if len(x_fit) == 0:
        raise ValueError(
            "Internal validation split consumed all training windows. "
            "Reduce --internal-val-ratio or --look-back."
        )

    model = build_cnn_transformer_model(
        input_shape=x_train_full.shape[1:],
        conv_filters=conv_filters,
        head_size=head_size,
        num_heads=num_heads,
        ff_dim=ff_dim,
        lstm_units=lstm_units,
        dropout=dropout,
    )
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )
    history = model.fit(
        x_fit,
        y_fit,
        validation_data=(x_internal_val, y_internal_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1,
    )

    train_pred_scaled = model.predict(x_train_full, verbose=0)
    train_true = inverse_close(y_train_full, scaler)
    train_pred = inverse_close(train_pred_scaled, scaler)
    train_metrics = evaluate_predictions(train_true, train_pred)

    val_true, val_pred, val_metrics = predict_block(
        model,
        val_data,
        scaler,
        look_back,
        forecast_mode,
        seed_window=train_data[-look_back:],
    )
    test_true, test_pred, test_metrics = predict_block(
        model,
        test_data,
        scaler,
        look_back,
        forecast_mode,
        seed_window=np.vstack([train_data, val_data])[-look_back:],
    )

    if forecast_mode == "one_step":
        baseline_metrics = {
            "train": evaluate_naive_previous_close(x_train_full, y_train_full, scaler),
            "val": evaluate_naive_previous_close(x_val, y_val, scaler),
            "test": evaluate_naive_previous_close(x_test, y_test, scaler),
        }
    else:
        baseline_metrics = {
            "train": evaluate_naive_previous_close(x_train_full, y_train_full, scaler),
            "val": evaluate_naive_recursive(train_data[-look_back:], val_data, scaler),
            "test": evaluate_naive_recursive(np.vstack([train_data, val_data])[-look_back:], test_data, scaler),
        }

    plot_path = None
    if plot_dir is not None:
        plot_path = Path(plot_dir) / f"cnn_transformer_{split_code}_ver2.png"
        plot_predictions(
            train_raw=train_raw,
            val_raw=val_raw,
            test_raw=test_raw,
            train_pred=train_pred,
            val_pred=val_pred,
            test_pred=test_pred,
            look_back=look_back,
            split_code=split_code,
            output_path=plot_path,
            forecast_mode=forecast_mode,
        )

    return {
        "model": model,
        "history": history.history,
        "split": split_code,
        "epochs_ran": len(history.history.get("loss", [])),
        "forecast_mode": forecast_mode,
        "split_sizes": split_sizes,
        "internal_train_windows": len(x_fit),
        "internal_val_windows": len(x_internal_val),
        "sequence_shapes": {
            "x_train": x_train_full.shape,
            "x_val": x_val.shape,
            "x_test": x_test.shape,
        },
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "baseline_metrics": baseline_metrics,
        "plot_path": plot_path,
    }


def print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(title)
    for metric, value in metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run leakage-safer CNN + Transformer stock-price prediction."
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Path to CSV data.")
    parser.add_argument(
        "--split",
        choices=["652510", "702010", "751510", "all"],
        default="702010",
    )
    parser.add_argument("--look-back", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--internal-val-ratio", type=float, default=0.15)
    parser.add_argument(
        "--forecast-mode",
        choices=["recursive", "one_step"],
        default="recursive",
        help=(
            "recursive predicts a whole val/test block without using true values "
            "inside that block; one_step predicts each next close from the latest "
            "available true look-back window."
        ),
    )
    parser.add_argument(
        "--plot-dir",
        default=str(Path(__file__).with_name("cnn_transformer_outputs")),
        help="Directory for prediction plots. Use empty string to skip plots.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conv-filters", type=int, default=32)
    parser.add_argument("--head-size", type=int, default=64)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--lstm-units", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_codes = list(SPLIT_RATIOS) if args.split == "all" else [args.split]
    plot_dir = args.plot_dir or None

    print(f"TensorFlow version: {tf.__version__}")
    print(f"Data: {args.data}")
    print(f"Look back: {args.look_back}")
    print(f"Max epochs: {args.epochs}")
    print(f"Early-stopping patience: {args.patience}")
    print(f"Forecast mode: {args.forecast_mode}")
    print(f"Internal validation ratio: {args.internal_val_ratio}")

    for split_code in split_codes:
        print(f"\n=== CNN+Transformer ver2 {split_code} ===")
        result = run_pipeline(
            csv_path=args.data,
            split_code=split_code,
            look_back=args.look_back,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            internal_val_ratio=args.internal_val_ratio,
            forecast_mode=args.forecast_mode,
            plot_dir=plot_dir,
            seed=args.seed,
            conv_filters=args.conv_filters,
            head_size=args.head_size,
            num_heads=args.num_heads,
            ff_dim=args.ff_dim,
            lstm_units=args.lstm_units,
            dropout=args.dropout,
        )

        print(f"Epochs ran: {result['epochs_ran']}")
        print(f"Split sizes: {result['split_sizes']}")
        print(
            "Internal early-stopping windows: "
            f"train={result['internal_train_windows']}, "
            f"val={result['internal_val_windows']}"
        )
        print(f"Sequence shapes: {result['sequence_shapes']}")
        print_metrics("Train metrics:", result["train_metrics"])
        print_metrics("Validation metrics:", result["val_metrics"])
        print_metrics("Test metrics:", result["test_metrics"])
        print_metrics("Naive baseline - Train:", result["baseline_metrics"]["train"])
        print_metrics("Naive baseline - Validation:", result["baseline_metrics"]["val"])
        print_metrics("Naive baseline - Test:", result["baseline_metrics"]["test"])
        if result["plot_path"]:
            print(f"Plot saved to: {result['plot_path']}")


if __name__ == "__main__":
    main()
