from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.svm import SVR

sys.path.append(str(Path(__file__).resolve().parents[1]))
from vnindex_feature_utils import (  # noqa: E402
    DEFAULT_PROCESSED_DATA_PATH,
    FEATURE_COLUMNS,
    RETURN_1D_TARGET_COLUMN,
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


def cumulative_return(daily_returns: np.ndarray) -> np.ndarray:
    return np.cumprod(1 + np.asarray(daily_returns).ravel()) - 1


def cumulative_return_from_base(base_return: float, daily_returns: np.ndarray) -> np.ndarray:
    return (1 + base_return) * np.cumprod(1 + np.asarray(daily_returns).ravel()) - 1


def window_target_dates(split_df, look_back: int):
    date_column = "Target_Date_1D" if "Target_Date_1D" in split_df.columns else "Date"
    return pd.to_datetime(split_df[date_column].iloc[look_back - 1 :], errors="coerce")


def window_current_returns(split_df, look_back: int) -> np.ndarray:
    return split_df["Return"].iloc[look_back - 1 :].to_numpy(dtype=float)


def plot_cumulative_predictions(
    train,
    val,
    test,
    val_true: np.ndarray,
    val_pred: np.ndarray,
    test_true: np.ndarray,
    test_pred: np.ndarray,
    look_back: int,
    split_code: str,
    output_path: str | Path,
) -> None:
    date_column = "Target_Date_1D" if "Target_Date_1D" in train.columns else "Date"
    all_dates = pd.to_datetime(
        pd.concat(
            [train[date_column], val[date_column], test[date_column]],
            ignore_index=True,
        ),
        errors="coerce",
    )
    all_true_returns = pd.concat(
        [
            train[RETURN_1D_TARGET_COLUMN],
            val[RETURN_1D_TARGET_COLUMN],
            test[RETURN_1D_TARGET_COLUMN],
        ],
        ignore_index=True,
    ).to_numpy(dtype=float)
    all_true_cum = cumulative_return(all_true_returns)

    val_dates = window_target_dates(val, look_back)
    test_dates = window_target_dates(test, look_back)

    val_start_pos = len(train) + look_back - 1
    test_start_pos = len(train) + len(val) + look_back - 1
    val_base = all_true_cum[val_start_pos - 1] if val_start_pos > 0 else 0.0
    test_base = all_true_cum[test_start_pos - 1] if test_start_pos > 0 else 0.0
    val_pred_cum = cumulative_return_from_base(val_base, val_pred)
    test_pred_cum = cumulative_return_from_base(test_base, test_pred)

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.axhline(0, color="black", linewidth=1, alpha=0.5)
    ax.plot(all_dates, all_true_cum * 100, label="VNINDEX actual cumulative return", color="blue")
    ax.plot(val_dates, val_pred_cum * 100, label="Validation predicted cumulative", color="purple")
    ax.plot(test_dates, test_pred_cum * 100, label="Test predicted cumulative", color="green")
    ax.set_title(f"SVR - Predicted vs Actual VNINDEX Cumulative Return ({split_code})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (%)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend()
    ax.grid(True)
    fig.autofmt_xdate()
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def run_pipeline(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    df = load_feature_data(args.data, target_column=RETURN_1D_TARGET_COLUMN)
    train, val, test, sizes = split_dataframe(df, args.split)
    x_train_scaled, y_train_scaled, x_val_scaled, y_val_scaled, x_test_scaled, y_test_scaled, _, y_scaler = scale_feature_splits(
        train,
        val,
        test,
        target_column=RETURN_1D_TARGET_COLUMN,
    )

    x_train, y_train = create_sequence_windows(x_train_scaled, y_train_scaled, args.look_back)
    x_val, y_val = create_sequence_windows(x_val_scaled, y_val_scaled, args.look_back)
    x_test, y_test = create_sequence_windows(x_test_scaled, y_test_scaled, args.look_back)
    sequence_shapes = {
        "x_train_sequence": x_train.shape,
        "x_val_sequence": x_val.shape,
        "x_test_sequence": x_test.shape,
    }
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_val = x_val.reshape(x_val.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    flattened_shapes = {
        "x_train_flat": x_train.shape,
        "x_val_flat": x_val.shape,
        "x_test_flat": x_test.shape,
    }

    model = train_svr(x_train, y_train, args.c, args.gamma, args.epsilon)
    val_true, val_pred, val_metrics = predict_and_evaluate(model, x_val, y_val, y_scaler)
    test_true, test_pred, test_metrics = predict_and_evaluate(model, x_test, y_test, y_scaler)

    current_return_val = window_current_returns(val, args.look_back)
    current_return_test = window_current_returns(test, args.look_back)
    train_return_mean = float(train[RETURN_1D_TARGET_COLUMN].mean())
    mean_val = np.full_like(val_true, train_return_mean)
    mean_test = np.full_like(test_true, train_return_mean)
    zero_val = np.zeros_like(val_true)
    zero_test = np.zeros_like(test_true)

    current_baseline_val = evaluate_return_predictions(val_true, current_return_val)
    current_baseline_test = evaluate_return_predictions(test_true, current_return_test)
    mean_baseline_val = evaluate_return_predictions(val_true, mean_val)
    mean_baseline_test = evaluate_return_predictions(test_true, mean_test)
    zero_baseline_val = evaluate_return_predictions(val_true, zero_val)
    zero_baseline_test = evaluate_return_predictions(test_true, zero_test)

    if args.plot_path:
        plot_cumulative_predictions(
            train,
            val,
            test,
            val_true,
            val_pred,
            test_true,
            test_pred,
            args.look_back,
            args.split,
            args.plot_path,
        )

    print(f"Data: {Path(args.data)}")
    print(f"Target: {RETURN_1D_TARGET_COLUMN}")
    print(f"Look-back sequence length: {args.look_back} trading days")
    print(f"Features: {FEATURE_COLUMNS}")
    print(f"Sequence shapes before SVR flattening: {sequence_shapes}")
    print(f"Flattened shapes used by SVR: {flattened_shapes}")
    print(f"Split sizes: {sizes}")
    print(f"Validation: {val_metrics}")
    print(f"Test: {test_metrics}")
    print(f"Current-return baseline validation: {current_baseline_val}")
    print(f"Current-return baseline test: {current_baseline_test}")
    print(f"Train-mean baseline validation: {mean_baseline_val}")
    print(f"Train-mean baseline test: {mean_baseline_test}")
    print(f"Zero baseline validation: {zero_baseline_val}")
    print(f"Zero baseline test: {zero_baseline_test}")

    return {
        "validation": val_metrics,
        "test": test_metrics,
        "current_baseline_validation": current_baseline_val,
        "current_baseline_test": current_baseline_test,
        "mean_baseline_validation": mean_baseline_val,
        "mean_baseline_test": mean_baseline_test,
        "zero_baseline_validation": zero_baseline_val,
        "zero_baseline_test": zero_baseline_test,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SVR next-day return prediction and plot cumulative return."
    )
    parser.add_argument("--data", default=DEFAULT_PROCESSED_DATA_PATH, type=Path)
    parser.add_argument("--split", default="751510", choices=sorted(SPLIT_RATIOS))
    parser.add_argument("--look-back", default=30, type=int)
    parser.add_argument("--c", default=10.0, type=float)
    parser.add_argument("--gamma", default="scale")
    parser.add_argument("--epsilon", default=0.001, type=float)
    parser.add_argument(
        "--plot-path",
        default=Path(__file__).resolve().parent / "svr_daily_return_cumulative_prediction.png",
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
