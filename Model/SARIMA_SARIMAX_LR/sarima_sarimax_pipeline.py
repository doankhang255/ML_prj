from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.append(str(Path(__file__).resolve().parents[1]))
from vnindex_feature_utils import (  # noqa: E402
    DEFAULT_PROCESSED_DATA_PATH,
    FEATURE_COLUMNS,
    SPLIT_RATIOS,
    TARGET_COLUMN,
    load_feature_data,
    split_dataframe,
)


def fit_sarimax_model(
    train: pd.DataFrame,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    use_exog: bool,
):
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except ImportError as exc:
        raise ImportError(
            "statsmodels is required for SARIMA/SARIMAX. Install it with: pip install statsmodels"
        ) from exc

    exog = train[FEATURE_COLUMNS] if use_exog else None
    model = SARIMAX(
        train[TARGET_COLUMN],
        exog=exog,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False)


def forecast_future(
    fitted_model,
    val: pd.DataFrame,
    test: pd.DataFrame,
    use_exog: bool,
) -> tuple[np.ndarray, np.ndarray]:
    future = pd.concat([val, test], axis=0)
    exog_future = future[FEATURE_COLUMNS] if use_exog else None
    forecast = fitted_model.get_forecast(steps=len(future), exog=exog_future).predicted_mean
    forecast = np.asarray(forecast)
    return forecast[: len(val)], forecast[len(val) :]


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


def plot_predictions(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    val_pred: np.ndarray,
    test_pred: np.ndarray,
    model_name: str,
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
    plt.title(f"{model_name.upper()} - Predict next VNINDEX close ({split_code})")
    plt.xlabel("Date")
    plt.ylabel("Next close price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_one_model(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    args: argparse.Namespace,
    model_name: str,
) -> dict[str, dict[str, float]]:
    use_exog = model_name == "sarimax"
    fitted = fit_sarimax_model(train, args.order, args.seasonal_order, use_exog)
    val_pred, test_pred = forecast_future(fitted, val, test, use_exog)
    val_metrics = evaluate_predictions(val["Close"], val[TARGET_COLUMN], val_pred)
    test_metrics = evaluate_predictions(test["Close"], test[TARGET_COLUMN], test_pred)
    naive_val_metrics = evaluate_predictions(val["Close"], val[TARGET_COLUMN], val["Close"])
    naive_test_metrics = evaluate_predictions(test["Close"], test[TARGET_COLUMN], test["Close"])

    if args.plot_dir:
        output_path = Path(args.plot_dir) / f"{model_name}_{args.split}_prediction.png"
        plot_predictions(train, val, test, val_pred, test_pred, model_name, args.split, output_path)

    print(f"{model_name.upper()} validation: {val_metrics}")
    print(f"{model_name.upper()} test: {test_metrics}")
    print(f"{model_name.upper()} naive validation: {naive_val_metrics}")
    print(f"{model_name.upper()} naive test: {naive_test_metrics}")
    return {
        "validation": val_metrics,
        "test": test_metrics,
        "naive_validation": naive_val_metrics,
        "naive_test": naive_test_metrics,
    }


def run_pipeline(args: argparse.Namespace) -> dict[str, dict[str, dict[str, float]]]:
    df = load_feature_data(args.data)
    train, val, test, sizes = split_dataframe(df, args.split)
    model_names = ["sarima", "sarimax"] if args.model == "both" else [args.model]

    print(f"Data: {Path(args.data)}")
    print(f"Target: {TARGET_COLUMN} (row t predicts close at t+1)")
    print(f"Exogenous features: {FEATURE_COLUMNS if args.model != 'sarima' else 'SARIMA uses no exog'}")
    print(f"Split sizes: {sizes}")
    print(f"Order: {args.order}, seasonal_order: {args.seasonal_order}")

    results = {}
    for model_name in model_names:
        results[model_name] = run_one_model(train, val, test, args, model_name)
    return results


def parse_order(value: str | tuple[int, int, int]) -> tuple[int, int, int]:
    if isinstance(value, tuple):
        return value
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("Order must have 3 comma-separated integers, e.g. 1,1,1")
    return tuple(parts)


def parse_seasonal_order(value: str | tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    if isinstance(value, tuple):
        return value
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "Seasonal order must have 4 comma-separated integers, e.g. 1,0,1,7"
        )
    return tuple(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run leakage-safe SARIMA/SARIMAX next-close prediction on VNINDEX processed data."
    )
    parser.add_argument("--data", default=DEFAULT_PROCESSED_DATA_PATH, type=Path)
    parser.add_argument("--split", default="702010", choices=sorted(SPLIT_RATIOS))
    parser.add_argument("--model", default="both", choices=["sarima", "sarimax", "both"])
    parser.add_argument("--order", default=(1, 1, 1), type=parse_order)
    parser.add_argument("--seasonal-order", default=(1, 0, 1, 7), type=parse_seasonal_order)
    parser.add_argument(
        "--plot-dir",
        default=Path(__file__).resolve().parent,
        type=Path,
        help="Set to an empty string to skip plot output.",
    )
    args = parser.parse_args()
    if args.plot_dir == Path("."):
        args.plot_dir = None
    return args


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
