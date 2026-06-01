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
    train_ratio, val_ratio, test_ratio = SPLIT_RATIOS[split_code]
    n = len(data)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    test_size = n - train_size - val_size

    train_data = data[:train_size]
    val_data = data[train_size : train_size + val_size]
    test_data = data[train_size + val_size :]

    sizes = {
        "total": n,
        "train": len(train_data),
        "val": len(val_data),
        "test": len(test_data),
        "test_ratio_requested": int(test_ratio * 100),
    }
    return train_data, val_data, test_data, sizes


def transformer_encoder(
    inputs: tf.Tensor,
    head_size: int,
    num_heads: int,
    ff_dim: int,
    dropout: float = 0.0,
) -> tf.Tensor:
    """Transformer encoder block from the source notebooks."""
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


def build_cnn_transformer_model(input_shape: tuple[int, int]) -> tf.keras.Model:
    """Build the CNN + Transformer architecture used by the HTML notebooks."""
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv1D(filters=32, kernel_size=3, activation="relu", padding="same")(inputs)
    x = layers.Conv1D(filters=32, kernel_size=3, activation="relu", padding="same")(x)
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)

    outputs = layers.Dense(1)(x)
    model = models.Model(inputs, outputs)
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

    x_pred_train = x_train[look_back:]
    x_pred_val = x_val[look_back:]
    x_pred_test = x_test[look_back:]

    plt.figure(figsize=(15, 6))
    plt.plot(x_train, train_inv, label="Train", color="blue")
    plt.plot(x_val, val_inv, label="Validate", color="red")
    plt.plot(x_test, test_inv, label="Test", color="orange")
    plt.plot(x_pred_train, y_pred_train_inv, label="TrainPred", color="cyan")
    plt.plot(x_pred_val, y_pred_val_inv, label="ValidatePred", color="purple")
    plt.plot(x_pred_test, y_pred_test_inv, label="Predict", color="green")
    plt.title(f"CNN+Transformer - Predict VCB price stock ({split_code})")
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
    epochs: int = 50,
    batch_size: int = 32,
    plot_dir: str | Path | None = None,
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

    model = build_cnn_transformer_model(x_train.shape[1:])
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )

    y_pred_train = model.predict(x_train, verbose=0)
    y_pred_val = model.predict(x_val, verbose=0)
    y_pred_test = model.predict(x_test, verbose=0)

    y_train_inv = scaler.inverse_transform(y_train)
    y_val_inv = scaler.inverse_transform(y_val)
    y_test_inv = scaler.inverse_transform(y_test)

    y_pred_train_inv = scaler.inverse_transform(y_pred_train)
    y_pred_val_inv = scaler.inverse_transform(y_pred_val)
    y_pred_test_inv = scaler.inverse_transform(y_pred_test)

    train_metrics = evaluate_predictions(y_train_inv, y_pred_train_inv)
    val_metrics = evaluate_predictions(y_val_inv, y_pred_val_inv)
    test_metrics = evaluate_predictions(y_test_inv, y_pred_test_inv)

    plot_path = None
    if plot_dir is not None:
        plot_path = Path(plot_dir) / f"cnn_transformer_{split_code}.png"
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
        )

    return {
        "model": model,
        "history": history.history,
        "split": split_code,
        "split_sizes": split_sizes,
        "sequence_shapes": {
            "x_train": x_train.shape,
            "x_val": x_val.shape,
            "x_test": x_test.shape,
        },
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "plot_path": plot_path,
    }


def print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(title)
    for metric, value in metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CNN + Transformer VCB stock-price notebooks as Python."
    )
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="Path to CSV data.")
    parser.add_argument(
        "--split",
        choices=["652510", "702010", "751510", "all"],
        default="702010",
        help="Notebook split to run. Use all to run the three converted notebooks.",
    )
    parser.add_argument("--look-back", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--plot-dir",
        default=str(Path(__file__).with_name("cnn_transformer_outputs")),
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
    print(f"Epochs: {args.epochs}")

    for split_code in split_codes:
        print(f"\n=== CNN+Transformer {split_code} ===")
        result = run_pipeline(
            csv_path=args.data,
            split_code=split_code,
            look_back=args.look_back,
            epochs=args.epochs,
            batch_size=args.batch_size,
            plot_dir=plot_dir,
        )

        print(f"Split sizes: {result['split_sizes']}")
        print(f"Sequence shapes: {result['sequence_shapes']}")
        print_metrics("Train metrics:", result["train_metrics"])
        print_metrics("Validation metrics:", result["val_metrics"])
        print_metrics("Test metrics:", result["test_metrics"])
        if result["plot_path"]:
            print(f"Plot saved to: {result['plot_path']}")


if __name__ == "__main__":
    main()
