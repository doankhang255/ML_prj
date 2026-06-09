from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.svm import SVR

sys.path.append(str(Path(__file__).resolve().parents[1]))
from vnindex_feature_utils import (  # noqa: E402
    DEFAULT_PROCESSED_DATA_PATH,
    FEATURE_COLUMNS,
    RETURN_10D_TARGET_COLUMN,
    SPLIT_RATIOS,
    create_sequence_windows,
    load_feature_data,
    scale_feature_splits,
    split_dataframe,
)


def train_svr(
    x_train: np.ndarray,
    y_train: np.ndarray,
    c: float,
    gamma: str | float,
    epsilon: float,
) -> SVR:
    model = SVR(kernel="rbf", C=c, gamma=gamma, epsilon=epsilon)
    model.fit(x_train, np.asarray(y_train).ravel())
    return model


def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true_direction = np.asarray(y_true) > 0
    pred_direction = np.asarray(y_pred) > 0
    return float(np.mean(true_direction == pred_direction))


def evaluate_return_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "direction_accuracy": direction_accuracy(y_true, y_pred),
        "mean_true_return": float(np.mean(y_true)),
        "mean_pred_return": float(np.mean(y_pred)),
    }


def inverse_scaled_return(values: np.ndarray, y_scaler) -> np.ndarray:
    return y_scaler.inverse_transform(np.asarray(values).reshape(-1, 1)).ravel()


def predict_and_evaluate(
    model: SVR,
    x_values: np.ndarray,
    y_values: np.ndarray,
    y_scaler,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    pred_scaled = model.predict(x_values)
    y_true = inverse_scaled_return(y_values, y_scaler)
    y_pred = inverse_scaled_return(pred_scaled, y_scaler)
    return y_true, y_pred, evaluate_return_predictions(y_true, y_pred)


def plot_predictions(
    train,
    val,
    test,
    val_pred: np.ndarray,
    test_pred: np.ndarray,
    look_back: int,
    split_code: str,
    output_path: str | Path,
) -> None:
    x_train = np.arange(len(train))
    x_val = np.arange(len(train), len(train) + len(val))
    x_test = np.arange(len(train) + len(val), len(train) + len(val) + len(test))

    plt.figure(figsize=(15, 6))
    plt.axhline(0, color="black", linewidth=1, alpha=0.5)
    plt.plot(x_train, train[RETURN_10D_TARGET_COLUMN], label="Train target", color="blue")
    plt.plot(x_val, val[RETURN_10D_TARGET_COLUMN], label="Validate target", color="red")
    plt.plot(x_test, test[RETURN_10D_TARGET_COLUMN], label="Test target", color="orange")
    plt.plot(x_val[look_back - 1 :], val_pred, label="ValidatePred", color="purple")
    plt.plot(x_test[look_back - 1 :], test_pred, label="Predict", color="green")
    plt.title(f"SVR - Predict VNINDEX future return 10D ({split_code})")
    plt.xlabel("Time Step")
    plt.ylabel("Future return 10D")
    plt.legend()
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_pipeline(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    df = load_feature_data(args.data, target_column=RETURN_10D_TARGET_COLUMN)
    train, val, test, sizes = split_dataframe(df, args.split)
    x_train_scaled, y_train_scaled, x_val_scaled, y_val_scaled, x_test_scaled, y_test_scaled, _, y_scaler = scale_feature_splits(
        train,
        val,
        test,
        target_column=RETURN_10D_TARGET_COLUMN,
    )

    x_train, y_train = create_sequence_windows(x_train_scaled, y_train_scaled, args.look_back)
    x_val, y_val = create_sequence_windows(x_val_scaled, y_val_scaled, args.look_back)
    x_test, y_test = create_sequence_windows(x_test_scaled, y_test_scaled, args.look_back)
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_val = x_val.reshape(x_val.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)

    model = train_svr(x_train, y_train, args.c, args.gamma, args.epsilon)
    val_true, val_pred, val_metrics = predict_and_evaluate(model, x_val, y_val, y_scaler)
    test_true, test_pred, test_metrics = predict_and_evaluate(model, x_test, y_test, y_scaler)

    zero_val = np.zeros_like(val_true)
    zero_test = np.zeros_like(test_true)
    train_return_mean = float(train[RETURN_10D_TARGET_COLUMN].mean())
    mean_val = np.full_like(val_true, train_return_mean)
    mean_test = np.full_like(test_true, train_return_mean)

    zero_baseline_val = evaluate_return_predictions(val_true, zero_val)
    zero_baseline_test = evaluate_return_predictions(test_true, zero_test)
    mean_baseline_val = evaluate_return_predictions(val_true, mean_val)
    mean_baseline_test = evaluate_return_predictions(test_true, mean_test)

    if args.plot_path:
        plot_predictions(
            train,
            val,
            test,
            val_pred,
            test_pred,
            args.look_back,
            args.split,
            args.plot_path,
        )

    print(f"Data: {Path(args.data)}")
    print(f"Target: {RETURN_10D_TARGET_COLUMN}")
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"Split sizes: {sizes}")
    print(f"Validation: {val_metrics}")
    print(f"Test: {test_metrics}")
    print(f"Zero baseline validation: {zero_baseline_val}")
    print(f"Zero baseline test: {zero_baseline_test}")
    print(f"Train-mean baseline validation: {mean_baseline_val}")
    print(f"Train-mean baseline test: {mean_baseline_test}")

    return {
        "validation": val_metrics,
        "test": test_metrics,
        "zero_baseline_validation": zero_baseline_val,
        "zero_baseline_test": zero_baseline_test,
        "mean_baseline_validation": mean_baseline_val,
        "mean_baseline_test": mean_baseline_test,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SVR prediction for VNINDEX 10-day future return."
    )
    parser.add_argument("--data", default=DEFAULT_PROCESSED_DATA_PATH, type=Path)
    parser.add_argument("--split", default="702010", choices=sorted(SPLIT_RATIOS))
    parser.add_argument("--look-back", default=30, type=int)
    parser.add_argument("--c", default=10.0, type=float)
    parser.add_argument("--gamma", default="scale")
    parser.add_argument("--epsilon", default=0.05, type=float)
    parser.add_argument(
        "--plot-path",
        default=Path(__file__).resolve().parent / "svr_future_return_10d_prediction.png",
        type=Path,
        help="Set to an empty string to skip plot output.",
    )
    args = parser.parse_args()
    if args.plot_path == Path("."):
        args.plot_path = None
    if isinstance(args.gamma, str) and args.gamma not in {"scale", "auto"}:
        try:
            args.gamma = float(args.gamma)
        except ValueError:
            pass
    return args


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
