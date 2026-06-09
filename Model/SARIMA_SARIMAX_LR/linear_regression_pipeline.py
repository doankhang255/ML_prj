from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))
from vnindex_feature_utils import (  # noqa: E402
    DEFAULT_PROCESSED_DATA_PATH,
    FEATURE_COLUMNS,
    SPLIT_RATIOS,
    TARGET_COLUMN,
    load_feature_data,
    split_dataframe,
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


def train_model(train):
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(train[FEATURE_COLUMNS], train[TARGET_COLUMN])
    return model


def predict_split(model: LinearRegression, data) -> tuple[np.ndarray, dict[str, float], dict[str, float]]:
    pred = model.predict(data[FEATURE_COLUMNS])
    y_true = data[TARGET_COLUMN].to_numpy()
    current_close = data["Close"].to_numpy()
    metrics = evaluate_predictions(current_close, y_true, pred)
    naive_metrics = evaluate_predictions(current_close, y_true, current_close)
    return pred, metrics, naive_metrics


def plot_predictions(
    train,
    val,
    test,
    val_pred: np.ndarray,
    test_pred: np.ndarray,
    split_code: str,
    output_path: str | Path,
) -> None:
    plt.figure(figsize=(16, 6))
    x_train = train["Date"] if "Date" in train.columns else np.arange(len(train))
    x_val = val["Date"] if "Date" in val.columns else np.arange(len(train), len(train) + len(val))
    x_test = test["Date"] if "Date" in test.columns else np.arange(len(train) + len(val), len(train) + len(val) + len(test))
    plt.plot(x_train, train[TARGET_COLUMN], label="Train target", color="blue")
    plt.plot(x_val, val[TARGET_COLUMN], label="Validate target", color="red")
    plt.plot(x_test, test[TARGET_COLUMN], label="Test target", color="orange")
    plt.plot(x_val, val_pred, label="ValidatePred", color="purple", linestyle="--")
    plt.plot(x_test, test_pred, label="Predict", color="green", linestyle="--")
    plt.title(f"Linear Regression - Predict next VNINDEX close ({split_code})")
    plt.xlabel("Date")
    plt.ylabel("Next close price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_pipeline(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    df = load_feature_data(args.data)
    train, val, test, sizes = split_dataframe(df, args.split)
    model = train_model(train)
    val_pred, val_metrics, naive_val_metrics = predict_split(model, val)
    test_pred, test_metrics, naive_test_metrics = predict_split(model, test)

    if args.plot_path:
        plot_predictions(train, val, test, val_pred, test_pred, args.split, args.plot_path)

    print(f"Data: {Path(args.data)}")
    print(f"Target: {TARGET_COLUMN} (row t predicts close at t+1)")
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"Split sizes: {sizes}")
    print(f"Validation: {val_metrics}")
    print(f"Test: {test_metrics}")
    print(f"Naive validation: {naive_val_metrics}")
    print(f"Naive test: {naive_test_metrics}")
    return {
        "validation": val_metrics,
        "test": test_metrics,
        "naive_validation": naive_val_metrics,
        "naive_test": naive_test_metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run leakage-safe Linear Regression next-close prediction on VNINDEX processed data."
    )
    parser.add_argument("--data", default=DEFAULT_PROCESSED_DATA_PATH, type=Path)
    parser.add_argument("--split", default="702010", choices=sorted(SPLIT_RATIOS))
    parser.add_argument(
        "--plot-path",
        default=Path(__file__).resolve().parent / "linear_regression_prediction.png",
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
