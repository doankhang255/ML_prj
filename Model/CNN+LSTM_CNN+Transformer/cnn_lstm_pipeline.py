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

VIETNAMESE_COLUMN_MAP = {
    "Ngay": "Date",
    "Ngày": "Date",
    "time": "Date",
    "date": "Date",
    "Lan cuoi": "Close",
    "Lần cuối": "Close",
    "close": "Close",
    "Mo": "Open",
    "Mở": "Open",
    "open": "Open",
    "Cao": "High",
    "high": "High",
    "Thap": "Low",
    "Thấp": "Low",
    "low": "Low",
    "KL": "Volume",
    "volume": "Volume",
    "% Thay doi": "ChangePct",
    "% Thay đổi": "ChangePct",
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
    if text.endswith("%"):
        text = text[:-1]
    elif text[-1].upper() == "K":
        multiplier = 1_000.0
        text = text[:-1]
    elif text[-1].upper() == "M":
        multiplier = 1_000_000.0
        text = text[:-1]
    elif text[-1].upper() == "B":
        multiplier = 1_000_000_000.0
        text = text[:-1]

    return float(text.replace(",", "")) * multiplier


def parse_dates(values: pd.Series) -> pd.Series:
    sample = values.dropna().astype(str).head(1)
    if not sample.empty and "-" in sample.iloc[0] and len(sample.iloc[0].split("-")[0]) == 4:
        return pd.to_datetime(values, errors="coerce")
    return pd.to_datetime(values, dayfirst=True, errors="coerce")


def load_close_prices(csv_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load VCB data from either normalized English CSV or raw Vietnamese CSV."""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df.rename(columns=VIETNAMESE_COLUMN_MAP)

    if "Date" in df.columns:
        df["Date"] = parse_dates(df["Date"])
        df = df.dropna(subset=["Date"]).sort_values("Date")

    if "Close" not in df.columns:
        raise ValueError("CSV must contain a Close/Lần cuối column.")

    df["Close"] = df["Close"].map(parse_number)
    df = df.dropna(subset=["Close"]).reset_index(drop=True)
    return df[["Close"]]


def create_dataset(data: np.ndarray, look_back: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding-window sequences for next-step forecasting."""
    x_values, y_values = [], []
    for i in range(look_back, len(data)):
        x_values.append(data[i - look_back : i])
        y_values.append(data[i])
    return np.array(x_values), np.array(y_values)


def split_raw_data(
    data: np.ndarray,
    split_code: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Split sequential data into train, validation, and test sets."""
    train_ratio, val_ratio, _test_ratio = SPLIT_RATIOS[split_code]
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


def build_cnn_lstm_model(input_shape: tuple[int, int]) -> tf.keras.Model:
    """Build the CNN + LSTM architecture used by the source notebook."""
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(filters=64, kernel_size=3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(100, return_sequences=False)(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1)(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    mae = float(mean_absolute_error(y_true_flat, y_pred_flat))
    rmse = float(np.sqrt(mean_squared_error(y_true_flat, y_pred_flat)))

    non_zero_mask = y_true_flat != 0
    if np.any(non_zero_mask):
        mape = float(
            np.mean(
                np.abs(
                    (y_true_flat[non_zero_mask] - y_pred_flat[non_zero_mask])
                    / y_true_flat[non_zero_mask]
                )
            )
            * 100
        )
    else:
        mape = float("nan")

    return {"mae": mae, "rmse": rmse, "mape": mape}


def evaluate_naive_previous_close(
    x_values: np.ndarray,
    y_values: np.ndarray,
    scaler: MinMaxScaler,
) -> dict[str, float]:
    """Baseline: predict the next close as the last close in the input window."""
    naive_pred = x_values[:, -1, :]
    y_true_inv = scaler.inverse_transform(y_values)
    naive_pred_inv = scaler.inverse_transform(naive_pred)
    return evaluate_predictions(y_true_inv, naive_pred_inv)


def evaluate_naive_recursive(
    seed_window: np.ndarray,
    true_block: np.ndarray,
    scaler: MinMaxScaler,
) -> dict[str, float]:
    """Strict baseline: keep predicting the last observed close for the full block."""
    naive_pred = np.repeat(seed_window[-1:].copy(), repeats=len(true_block), axis=0)
    y_true_inv = scaler.inverse_transform(true_block)
    naive_pred_inv = scaler.inverse_transform(naive_pred)
    return evaluate_predictions(y_true_inv, naive_pred_inv)


def recursive_predict(
    model: tf.keras.Model,
    seed_window: np.ndarray,
    steps: int,
) -> np.ndarray:
    """Predict a full future block without using true values inside that block."""
    history = seed_window.astype(float).copy()
    predictions = []

    for _ in range(steps):
        x_input = history[-len(seed_window) :].reshape(1, len(seed_window), history.shape[1])
        pred = model.predict(x_input, verbose=0)
        predictions.append(pred[0])
        history = np.vstack([history, pred])

    return np.array(predictions)


def plot_predictions(
    scaler: MinMaxScaler,
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
    y_pred_train_inv: np.ndarray,
    y_pred_val_inv: np.ndarray,
    y_pred_test_inv: np.ndarray,
    look_back: int,
    split_code: str,
    output_path: str | Path,
    forecast_mode: str,
) -> None:
    train_inv = scaler.inverse_transform(train_data)
    val_inv = scaler.inverse_transform(val_data)
    test_inv = scaler.inverse_transform(test_data)

    x_train = np.arange(0, len(train_inv))
    x_val = np.arange(len(train_inv), len(train_inv) + len(val_inv))
    x_test = np.arange(
        len(train_inv) + len(val_inv),
        len(train_inv) + len(val_inv) + len(test_inv),
    )

    plt.figure(figsize=(15, 6))
    plt.plot(x_train, train_inv, label="Train", color="blue")
    plt.plot(x_val, val_inv, label="Validate", color="red")
    plt.plot(x_test, test_inv, label="Test", color="orange")
    if forecast_mode == "one_step":
        x_pred_train = x_train[look_back:]
        x_pred_val = x_val[look_back:]
        x_pred_test = x_test[look_back:]
    else:
        x_pred_train = x_train[look_back:]
        x_pred_val = x_val
        x_pred_test = x_test

    plt.plot(x_pred_train, y_pred_train_inv, label="TrainPred", color="cyan")
    plt.plot(x_pred_val, y_pred_val_inv, label="ValidatePred", color="purple")
    plt.plot(x_pred_test, y_pred_test_inv, label="Predict", color="green")
    plt.title(f"CNN+LSTM - Predict VCB price stock ({split_code})")
    plt.xlabel("Time Step")
    plt.ylabel("Price stock")
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
    epochs: int = 100,
    batch_size: int = 32,
    patience: int = 10,
    plot_dir: str | Path | None = None,
    forecast_mode: str = "recursive",
    internal_val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, object]:
    np.random.seed(seed)
    tf.random.set_seed(seed)

    df = load_close_prices(csv_path)
    raw_data = df[["Close"]].to_numpy()
    train_raw, val_raw, test_raw, split_sizes = split_raw_data(raw_data, split_code)

    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_raw)
    val_data = scaler.transform(val_raw)
    test_data = scaler.transform(test_raw)

    x_train, y_train = create_dataset(train_data, look_back)
    x_val, y_val = create_dataset(val_data, look_back)
    x_test, y_test = create_dataset(test_data, look_back)

    if len(x_train) == 0 or len(x_val) == 0 or len(x_test) == 0:
        raise ValueError(
            "One of the splits is too small for the selected "
            f"look_back={look_back}. Try a smaller look_back."
        )

    internal_val_size = max(1, int(len(x_train) * internal_val_ratio))
    fit_x_train = x_train[:-internal_val_size]
    fit_y_train = y_train[:-internal_val_size]
    internal_x_val = x_train[-internal_val_size:]
    internal_y_val = y_train[-internal_val_size:]

    if len(fit_x_train) == 0:
        raise ValueError(
            "Internal validation split consumed all training windows. "
            "Reduce --internal-val-ratio or --look-back."
        )

    model = build_cnn_lstm_model(x_train.shape[1:])
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )
    history = model.fit(
        fit_x_train,
        fit_y_train,
        validation_data=(internal_x_val, internal_y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1,
    )

    y_pred_train = model.predict(x_train, verbose=0)
    y_train_inv = scaler.inverse_transform(y_train)
    y_pred_train_inv = scaler.inverse_transform(y_pred_train)

    if forecast_mode == "one_step":
        y_pred_val = model.predict(x_val, verbose=0)
        y_pred_test = model.predict(x_test, verbose=0)
        y_val_inv = scaler.inverse_transform(y_val)
        y_test_inv = scaler.inverse_transform(y_test)
        y_pred_val_inv = scaler.inverse_transform(y_pred_val)
        y_pred_test_inv = scaler.inverse_transform(y_pred_test)
    elif forecast_mode == "recursive":
        y_pred_val = recursive_predict(model, train_data[-look_back:], len(val_data))
        y_pred_test = recursive_predict(
            model,
            np.vstack([train_data, val_data])[-look_back:],
            len(test_data),
        )
        y_val_inv = scaler.inverse_transform(val_data)
        y_test_inv = scaler.inverse_transform(test_data)
        y_pred_val_inv = scaler.inverse_transform(y_pred_val)
        y_pred_test_inv = scaler.inverse_transform(y_pred_test)
    else:
        raise ValueError("forecast_mode must be either 'recursive' or 'one_step'.")

    train_metrics = evaluate_predictions(y_train_inv, y_pred_train_inv)
    val_metrics = evaluate_predictions(y_val_inv, y_pred_val_inv)
    test_metrics = evaluate_predictions(y_test_inv, y_pred_test_inv)
    if forecast_mode == "one_step":
        baseline_metrics = {
            "train": evaluate_naive_previous_close(x_train, y_train, scaler),
            "val": evaluate_naive_previous_close(x_val, y_val, scaler),
            "test": evaluate_naive_previous_close(x_test, y_test, scaler),
        }
    else:
        baseline_metrics = {
            "train": evaluate_naive_previous_close(x_train, y_train, scaler),
            "val": evaluate_naive_recursive(train_data[-look_back:], val_data, scaler),
            "test": evaluate_naive_recursive(
                np.vstack([train_data, val_data])[-look_back:],
                test_data,
                scaler,
            ),
        }

    plot_path = None
    if plot_dir is not None:
        plot_path = Path(plot_dir) / f"cnn_lstm_{split_code}.png"
        plot_predictions(
            scaler=scaler,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            y_pred_train_inv=y_pred_train_inv,
            y_pred_val_inv=y_pred_val_inv,
            y_pred_test_inv=y_pred_test_inv,
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
        "internal_train_windows": len(fit_x_train),
        "internal_val_windows": len(internal_x_val),
        "sequence_shapes": {
            "x_train": x_train.shape,
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
        description="Run CNN + LSTM VCB stock-price notebooks as Python."
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Path to CSV data.")
    parser.add_argument(
        "--split",
        choices=["652510", "702010", "751510", "all"],
        default="702010",
        help="Notebook split to run. Use all to run the three converted notebooks.",
    )
    parser.add_argument("--look-back", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--internal-val-ratio",
        type=float,
        default=0.15,
        help=(
            "Fraction of train windows reserved for early stopping. "
            "External validation/test blocks remain untouched for final evaluation."
        ),
    )
    parser.add_argument(
        "--forecast-mode",
        choices=["recursive", "one_step"],
        default="recursive",
        help=(
            "recursive predicts an entire val/test block without using true values "
            "inside that block; one_step matches the original notebook evaluation."
        ),
    )
    parser.add_argument(
        "--plot-dir",
        default=str(Path(__file__).with_name("cnn_lstm_outputs")),
        help="Directory for prediction plots. Use empty string to skip plots.",
    )
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
        print(f"\n=== CNN+LSTM {split_code} ===")
        result = run_pipeline(
            csv_path=args.data,
            split_code=split_code,
            look_back=args.look_back,
            epochs=args.epochs,
            batch_size=args.batch_size,
            patience=args.patience,
            plot_dir=plot_dir,
            forecast_mode=args.forecast_mode,
            internal_val_ratio=args.internal_val_ratio,
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
