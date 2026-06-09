from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import GRU, LSTM, Dense, Input, SimpleRNN

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


def build_model(model_type: str, input_shape: tuple[int, int], units: int) -> tf.keras.Model:
    recurrent_layer = {"rnn": SimpleRNN, "lstm": LSTM, "gru": GRU}[model_type]
    model = Sequential(
        [
            Input(shape=input_shape),
            recurrent_layer(units, return_sequences=False),
            Dense(1),
        ]
    )
    model.compile(loss="mean_squared_error", optimizer="adam")
    return model


def split_internal_validation(
    x_train: np.ndarray,
    y_train: np.ndarray,
    internal_val_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    val_size = max(1, int(len(x_train) * internal_val_ratio))
    return x_train[:-val_size], y_train[:-val_size], x_train[-val_size:], y_train[-val_size:]


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
    predictions: dict[str, tuple[np.ndarray, np.ndarray]],
    time_step: int,
    split_code: str,
    output_path: str | Path,
) -> None:
    pred_x = np.arange(time_step - 1, len(test))
    plt.figure(figsize=(15, 6))
    plt.plot(np.arange(len(test)), test[TARGET_COLUMN], label="Test target", color="orange")
    colors = {"rnn": "cyan", "lstm": "purple", "gru": "green"}
    for name, (_true, pred) in predictions.items():
        plt.plot(pred_x, pred, label=f"{name.upper()} Predict", color=colors[name])
    plt.title(f"RNN/LSTM/GRU - Predict next VNINDEX close ({split_code})")
    plt.xlabel("Test time step")
    plt.ylabel("Next close price")
    plt.legend()
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_pipeline(args: argparse.Namespace) -> dict[str, dict[str, dict[str, float]]]:
    tf.keras.utils.set_random_seed(args.seed)
    df = load_feature_data(args.data)
    train, val, test, sizes = split_dataframe(df, args.split)
    x_train_scaled, y_train_scaled, x_val_scaled, y_val_scaled, x_test_scaled, y_test_scaled, _, y_scaler = scale_feature_splits(train, val, test)

    x_train_full, y_train_full = create_sequence_windows(x_train_scaled, y_train_scaled, args.time_step)
    x_val, y_val = create_sequence_windows(x_val_scaled, y_val_scaled, args.time_step)
    x_test, y_test = create_sequence_windows(x_test_scaled, y_test_scaled, args.time_step)
    x_fit, y_fit, x_internal_val, y_internal_val = split_internal_validation(
        x_train_full,
        y_train_full,
        args.internal_val_ratio,
    )

    val_current_close = window_current_close(val, args.time_step)
    test_current_close = window_current_close(test, args.time_step)
    model_names = ["rnn", "lstm", "gru"] if args.model == "all" else [args.model]
    results: dict[str, dict[str, dict[str, float]]] = {}
    test_predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for model_name in model_names:
        model = build_model(model_name, (args.time_step, x_train_scaled.shape[1]), args.units)
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
        results[model_name] = {
            "validation": val_metrics,
            "test": test_metrics,
            "naive_validation": naive_val,
            "naive_test": naive_test,
            "best_train_val_loss": float(min(history.history["val_loss"])),
        }
        test_predictions[model_name] = (test_true, test_pred)

    if args.plot_path:
        plot_predictions(test, test_predictions, args.time_step, args.split, args.plot_path)

    print(f"Data: {Path(args.data)}")
    print(f"Target: {TARGET_COLUMN}")
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"Split sizes: {sizes}")
    for model_name, metrics in results.items():
        print(f"{model_name.upper()}: {metrics}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run leakage-safe RNN/LSTM/GRU next-close prediction on VNINDEX processed data."
    )
    parser.add_argument("--data", default=DEFAULT_PROCESSED_DATA_PATH, type=Path)
    parser.add_argument("--split", default="702010", choices=sorted(SPLIT_RATIOS))
    parser.add_argument("--model", default="all", choices=["rnn", "lstm", "gru", "all"])
    parser.add_argument("--time-step", default=30, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--units", default=50, type=int)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--internal-val-ratio", default=0.15, type=float)
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
