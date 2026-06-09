from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

sys.path.append(str(Path(__file__).resolve().parents[1]))
from vnindex_feature_utils import (  # noqa: E402
    DEFAULT_PROCESSED_DATA_PATH,
    FEATURE_COLUMNS,
    SPLIT_RATIOS,
    TARGET_COLUMN,
    create_sequence_windows,
    inverse_target,
    load_feature_data,
    scale_feature_splits,
    split_dataframe,
    window_current_close,
)


def build_cnn_lstm_model(input_shape: tuple[int, int]) -> tf.keras.Model:
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


def split_internal_validation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    internal_val_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    val_size = max(1, int(len(x_train) * internal_val_ratio))
    return x_train[:-val_size], y_train[:-val_size], x_train[-val_size:], y_train[-val_size:]


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    non_zero = y_true != 0
    return float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100)


def direction_accuracy(current_close: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_direction = np.asarray(y_true) > np.asarray(current_close)
    pred_direction = np.asarray(y_pred) > np.asarray(current_close)
    return float(np.mean(true_direction == pred_direction))


def evaluate_predictions(
    current_close: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "direction_accuracy": direction_accuracy(current_close, y_true, y_pred),
    }


def predict_block(
    model: tf.keras.Model,
    x_values: np.ndarray,
    y_values: np.ndarray,
    current_close: np.ndarray,
    y_scaler,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], dict[str, float]]:
    pred_scaled = model.predict(x_values, verbose=0)
    y_true = inverse_target(y_values, y_scaler)
    y_pred = inverse_target(pred_scaled, y_scaler)
    metrics = evaluate_predictions(current_close, y_true, y_pred)
    naive_metrics = evaluate_predictions(current_close, y_true, current_close)
    return y_true, y_pred, metrics, naive_metrics


def plot_predictions(
    test,
    y_pred_test: np.ndarray,
    look_back: int,
    split_code: str,
    output_path: str | Path,
) -> None:
    pred_x = np.arange(look_back - 1, len(test))
    plt.figure(figsize=(15, 6))
    plt.plot(np.arange(len(test)), test[TARGET_COLUMN], label="Test target", color="orange")
    plt.plot(pred_x, y_pred_test, label="Predict", color="green")
    plt.title(f"CNN+LSTM - Predict next VNINDEX close ({split_code})")
    plt.xlabel("Test time step")
    plt.ylabel("Next close price")
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
    internal_val_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, object]:
    tf.keras.utils.set_random_seed(seed)
    df = load_feature_data(csv_path)
    train, val, test, split_sizes = split_dataframe(df, split_code)
    x_train_scaled, y_train_scaled, x_val_scaled, y_val_scaled, x_test_scaled, y_test_scaled, _, y_scaler = scale_feature_splits(train, val, test)

    x_train_full, y_train_full = create_sequence_windows(x_train_scaled, y_train_scaled, look_back)
    x_val, y_val = create_sequence_windows(x_val_scaled, y_val_scaled, look_back)
    x_test, y_test = create_sequence_windows(x_test_scaled, y_test_scaled, look_back)
    fit_x_train, fit_y_train, internal_x_val, internal_y_val = split_internal_validation(
        x_train_full,
        y_train_full,
        internal_val_ratio,
    )

    model = build_cnn_lstm_model(x_train_full.shape[1:])
    early_stop = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    history = model.fit(
        fit_x_train,
        fit_y_train,
        validation_data=(internal_x_val, internal_y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1,
    )

    val_current_close = window_current_close(val, look_back)
    test_current_close = window_current_close(test, look_back)
    _val_true, _val_pred, val_metrics, naive_val = predict_block(
        model,
        x_val,
        y_val,
        val_current_close,
        y_scaler,
    )
    test_true, test_pred, test_metrics, naive_test = predict_block(
        model,
        x_test,
        y_test,
        test_current_close,
        y_scaler,
    )

    plot_path = None
    if plot_dir is not None:
        plot_path = Path(plot_dir) / f"cnn_lstm_{split_code}.png"
        plot_predictions(test, test_pred, look_back, split_code, plot_path)

    return {
        "model": model,
        "history": history.history,
        "split": split_code,
        "epochs_ran": len(history.history.get("loss", [])),
        "target": TARGET_COLUMN,
        "features": FEATURE_COLUMNS,
        "split_sizes": split_sizes,
        "sequence_shapes": {
            "x_train": x_train_full.shape,
            "x_val": x_val.shape,
            "x_test": x_test.shape,
        },
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "naive_val_metrics": naive_val,
        "naive_test_metrics": naive_test,
        "plot_path": plot_path,
    }


def print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(title)
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run leakage-safe CNN + LSTM next-close prediction on VNINDEX processed data."
    )
    parser.add_argument("--data", default=str(DEFAULT_PROCESSED_DATA_PATH), help="Path to processed CSV data.")
    parser.add_argument("--split", choices=["652510", "702010", "751510", "all"], default="702010")
    parser.add_argument("--look-back", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--internal-val-ratio", type=float, default=0.15)
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
    print(f"Target: {TARGET_COLUMN}")
    print(f"Features: {FEATURE_COLUMNS}")

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
            internal_val_ratio=args.internal_val_ratio,
        )
        print(f"Split sizes: {result['split_sizes']}")
        print(f"Sequence shapes: {result['sequence_shapes']}")
        print_metrics("Validation metrics:", result["val_metrics"])
        print_metrics("Test metrics:", result["test_metrics"])
        print_metrics("Naive validation:", result["naive_val_metrics"])
        print_metrics("Naive test:", result["naive_test_metrics"])
        if result["plot_path"]:
            print(f"Plot saved to: {result['plot_path']}")


if __name__ == "__main__":
    main()
