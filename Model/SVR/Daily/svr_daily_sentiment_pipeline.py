from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVR


DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[3] / "Dataset" / "vnindex_daily_sentiment_merged.csv"
)

SPLIT_RATIOS = {
    "701515": (0.70, 0.15, 0.15),
    "751510": (0.75, 0.15, 0.10),
    "801010": (0.80, 0.10, 0.10),
}

sys.path.append(str(Path(__file__).resolve().parent))
from daily_feature_utils import (
    TARGET_COLUMNS,
    build_feature_columns,
    load_feature_data,
    read_daily_data,
)


@dataclass(frozen=True)
class PreparedSplit:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    y_scaler: StandardScaler
    shapes: dict[str, tuple[int, ...]]


@dataclass(frozen=True)
class FittedRun:
    model: SVR
    look_back: int
    params: dict[str, object]
    prepared: PreparedSplit
    train_true: np.ndarray
    train_pred: np.ndarray
    val_true: np.ndarray
    val_pred: np.ndarray
    test_true: np.ndarray
    test_pred: np.ndarray
    train_metrics: dict[str, float]
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]


def split_dataframe(
    df: pd.DataFrame,
    split_code: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int | str]]:
    if split_code not in SPLIT_RATIOS:
        raise ValueError(f"Unknown split={split_code}. Choose from {sorted(SPLIT_RATIOS)}")

    train_ratio, val_ratio, _ = SPLIT_RATIOS[split_code]
    n = len(df)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    train = df.iloc[:train_size].copy()
    val = df.iloc[train_size : train_size + val_size].copy()
    test = df.iloc[train_size + val_size :].copy()

    sizes: dict[str, int | str] = {
        "total": n,
        "train": len(train),
        "val": len(val),
        "test": len(test),
    }
    for name, split in [("train", train), ("val", val), ("test", test)]:
        sizes[f"{name}_start"] = str(split["date"].min().date()) if len(split) else "NA"
        sizes[f"{name}_end"] = str(split["date"].max().date()) if len(split) else "NA"
    return train, val, test, sizes


def make_scaler(kind: str):
    if kind == "standard":
        return StandardScaler()
    if kind == "robust":
        return RobustScaler()
    raise ValueError("scaler must be standard or robust")


def create_sequence_windows(
    x_values: np.ndarray,
    y_values: np.ndarray,
    look_back: int,
) -> tuple[np.ndarray, np.ndarray]:
    if look_back < 1:
        raise ValueError("look_back must be at least 1.")
    if len(x_values) < look_back:
        raise ValueError(f"Need at least look_back={look_back} rows, got {len(x_values)}.")

    x_windows, y_targets = [], []
    for i in range(look_back - 1, len(x_values)):
        x_windows.append(x_values[i - look_back + 1 : i + 1])
        y_targets.append(y_values[i])
    return np.asarray(x_windows), np.asarray(y_targets)


def scale_feature_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    scaler: str,
):
    x_scaler = make_scaler(scaler)
    y_scaler = StandardScaler()

    x_train = x_scaler.fit_transform(train[list(feature_columns)])
    x_val = x_scaler.transform(val[list(feature_columns)])
    x_test = x_scaler.transform(test[list(feature_columns)])

    y_train = y_scaler.fit_transform(train[[target_column]])
    y_val = y_scaler.transform(val[[target_column]])
    y_test = y_scaler.transform(test[[target_column]])
    return x_train, y_train, x_val, y_val, x_test, y_test, y_scaler


def prepare_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    look_back: int,
    scaler: str,
) -> PreparedSplit:
    x_train_scaled, y_train_scaled, x_val_scaled, y_val_scaled, x_test_scaled, y_test_scaled, y_scaler = (
        scale_feature_splits(train, val, test, feature_columns, target_column, scaler)
    )

    x_train, y_train = create_sequence_windows(x_train_scaled, y_train_scaled, look_back)
    x_val, y_val = create_sequence_windows(x_val_scaled, y_val_scaled, look_back)
    x_test, y_test = create_sequence_windows(x_test_scaled, y_test_scaled, look_back)

    shapes = {
        "x_train_sequence": x_train.shape,
        "x_val_sequence": x_val.shape,
        "x_test_sequence": x_test.shape,
    }
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_val = x_val.reshape(x_val.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    shapes.update(
        {
            "x_train_flat": x_train.shape,
            "x_val_flat": x_val.shape,
            "x_test_flat": x_test.shape,
        }
    )
    return PreparedSplit(x_train, y_train, x_val, y_val, x_test, y_test, y_scaler, shapes)


def train_svr(x_train: np.ndarray, y_train: np.ndarray, c: float, gamma: str | float, epsilon: float) -> SVR:
    model = SVR(kernel="rbf", C=c, gamma=gamma, epsilon=epsilon)
    model.fit(x_train, np.asarray(y_train).ravel())
    return model


def inverse_scaled_return(values: np.ndarray, y_scaler: StandardScaler) -> np.ndarray:
    return y_scaler.inverse_transform(np.asarray(values).reshape(-1, 1)).ravel()


def safe_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if len(y_true) <= 1 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return np.nan
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean((y_true > 0) == (y_pred > 0)))


def evaluate_return_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "direction_accuracy": direction_accuracy(y_true, y_pred),
        "correlation": safe_corr(y_true, y_pred),
        "mean_true_return": float(np.mean(y_true)),
        "mean_pred_return": float(np.mean(y_pred)),
        "true_std": float(np.std(y_true)),
        "pred_std": float(np.std(y_pred)),
    }


def predict_and_evaluate(
    model: SVR,
    x_values: np.ndarray,
    y_values: np.ndarray,
    y_scaler: StandardScaler,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    pred_scaled = model.predict(x_values)
    y_true = inverse_scaled_return(y_values, y_scaler)
    y_pred = inverse_scaled_return(pred_scaled, y_scaler)
    return y_true, y_pred, evaluate_return_predictions(y_true, y_pred)


def fit_once(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    look_back: int,
    c: float,
    gamma: str | float,
    epsilon: float,
    scaler: str,
) -> FittedRun:
    prepared = prepare_splits(train, val, test, feature_columns, target_column, look_back, scaler)
    model = train_svr(prepared.x_train, prepared.y_train, c, gamma, epsilon)

    train_true, train_pred, train_metrics = predict_and_evaluate(
        model, prepared.x_train, prepared.y_train, prepared.y_scaler
    )
    val_true, val_pred, val_metrics = predict_and_evaluate(
        model, prepared.x_val, prepared.y_val, prepared.y_scaler
    )
    test_true, test_pred, test_metrics = predict_and_evaluate(
        model, prepared.x_test, prepared.y_test, prepared.y_scaler
    )

    return FittedRun(
        model=model,
        look_back=look_back,
        params={"C": c, "gamma": gamma, "epsilon": epsilon},
        prepared=prepared,
        train_true=train_true,
        train_pred=train_pred,
        val_true=val_true,
        val_pred=val_pred,
        test_true=test_true,
        test_pred=test_pred,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
    )


def objective_value(metrics: dict[str, float], objective: str) -> float:
    if objective in {"mae", "rmse"}:
        return metrics[objective]
    if objective in {"diracc", "direction_accuracy"}:
        return -metrics["direction_accuracy"]
    if objective in {"corr", "correlation"}:
        corr = metrics["correlation"]
        return -corr if not np.isnan(corr) else np.inf
    raise ValueError("objective must be mae, rmse, diracc, or corr")


def tune_svr(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    look_back_grid: Sequence[int],
    c_grid: Sequence[float],
    gamma_grid: Sequence[str | float],
    epsilon_grid: Sequence[float],
    scaler: str,
    objective: str,
) -> tuple[FittedRun, pd.DataFrame]:
    results: list[dict[str, object]] = []
    best_run: FittedRun | None = None
    best_score = np.inf

    for look_back in look_back_grid:
        for params in ParameterGrid({"C": c_grid, "gamma": gamma_grid, "epsilon": epsilon_grid}):
            run = fit_once(
                train=train,
                val=val,
                test=test,
                feature_columns=feature_columns,
                target_column=target_column,
                look_back=int(look_back),
                c=float(params["C"]),
                gamma=params["gamma"],
                epsilon=float(params["epsilon"]),
                scaler=scaler,
            )
            score = objective_value(run.val_metrics, objective)
            results.append(
                {
                    "look_back": run.look_back,
                    **run.params,
                    "objective": objective,
                    "objective_score": score,
                    "val_mae": run.val_metrics["mae"],
                    "val_rmse": run.val_metrics["rmse"],
                    "val_diracc": run.val_metrics["direction_accuracy"],
                    "val_corr": run.val_metrics["correlation"],
                    "test_mae": run.test_metrics["mae"],
                    "test_rmse": run.test_metrics["rmse"],
                    "test_diracc": run.test_metrics["direction_accuracy"],
                    "test_corr": run.test_metrics["correlation"],
                }
            )
            if score < best_score:
                best_score = score
                best_run = run

    if best_run is None:
        raise RuntimeError("No SVR runs completed during tuning.")
    result_df = pd.DataFrame(results).sort_values("objective_score").reset_index(drop=True)
    return best_run, result_df


def window_values(split_df: pd.DataFrame, column: str, look_back: int) -> np.ndarray:
    return split_df[column].iloc[look_back - 1 :].to_numpy(dtype=float)


def baseline_predictions(
    split_df: pd.DataFrame,
    y_true: np.ndarray,
    look_back: int,
    train_target_mean: float,
) -> dict[str, np.ndarray]:
    baselines = {
        "Zero return": np.zeros_like(y_true),
        "Train mean": np.full_like(y_true, train_target_mean),
    }
    for column in ["daily_return", "return_lag_1d", "return_ma_5d", "return_ma_20d"]:
        if column in split_df.columns:
            values = window_values(split_df, column, look_back)
            if len(values) == len(y_true) and np.isfinite(values).all():
                baselines[column] = values
    return baselines


def best_baseline_summary(
    model_metrics: dict[str, float],
    baseline_rows: dict[str, dict[str, float]],
) -> dict[str, float | str]:
    best_mae_name, best_mae = min(
        ((name, metrics["mae"]) for name, metrics in baseline_rows.items()),
        key=lambda x: x[1],
    )
    best_rmse_name, best_rmse = min(
        ((name, metrics["rmse"]) for name, metrics in baseline_rows.items()),
        key=lambda x: x[1],
    )
    return {
        "best_mae_baseline": best_mae_name,
        "model_mae_minus_best": model_metrics["mae"] - best_mae,
        "model_mae_improvement_pct": 100 * (best_mae - model_metrics["mae"]) / best_mae,
        "best_rmse_baseline": best_rmse_name,
        "model_rmse_minus_best": model_metrics["rmse"] - best_rmse,
        "model_rmse_improvement_pct": 100 * (best_rmse - model_metrics["rmse"]) / best_rmse,
    }


def format_metric(value: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "nan"
    return f"{value:.6f}"


def print_metric_table(title: str, rows: dict[str, dict[str, float]]) -> None:
    print(f"\n{title}")
    print(
        f"{'Set':<24}"
        f"{'MAE':>11}"
        f"{'RMSE':>11}"
        f"{'DirAcc':>11}"
        f"{'Corr':>11}"
        f"{'TrueMean':>12}"
        f"{'PredMean':>12}"
        f"{'TrueStd':>11}"
        f"{'PredStd':>11}"
    )
    print("-" * 115)
    for label, metrics in rows.items():
        print(
            f"{label:<24}"
            f"{format_metric(metrics['mae']):>11}"
            f"{format_metric(metrics['rmse']):>11}"
            f"{format_metric(metrics['direction_accuracy']):>11}"
            f"{format_metric(metrics['correlation']):>11}"
            f"{format_metric(metrics['mean_true_return']):>12}"
            f"{format_metric(metrics['mean_pred_return']):>12}"
            f"{format_metric(metrics['true_std']):>11}"
            f"{format_metric(metrics['pred_std']):>11}"
        )


def window_target_dates(split_df: pd.DataFrame, look_back: int) -> pd.Series:
    return pd.to_datetime(split_df["date"].iloc[look_back - 1 :], errors="coerce")


def plot_return_predictions(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target_column: str,
    run: FittedRun,
    output_path: str | Path,
) -> None:
    all_dates = pd.to_datetime(
        pd.concat([train["date"], val["date"], test["date"]], ignore_index=True),
        errors="coerce",
    )
    all_true_returns = pd.concat(
        [train[target_column], val[target_column], test[target_column]], ignore_index=True
    ).to_numpy(dtype=float)

    train_dates = window_target_dates(train, run.look_back)
    val_dates = window_target_dates(val, run.look_back)
    test_dates = window_target_dates(test, run.look_back)

    fig, ax = plt.subplots(figsize=(15, 6))
    ax.axhline(0, linewidth=1, alpha=0.5)
    ax.plot(all_dates, all_true_returns * 100, label=f"Actual {target_column}", color="blue", alpha=0.55)
    ax.plot(train_dates, run.train_pred * 100, label="Train predicted", color="orange", alpha=0.85)
    ax.plot(val_dates, run.val_pred * 100, label="Validation predicted", color="purple", alpha=0.85)
    ax.plot(test_dates, run.test_pred * 100, label="Test predicted", color="green", alpha=0.85)
    ax.set_title(f"Daily SVR VNINDEX {target_column} | look_back={run.look_back} | {run.params}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Return (%)")
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


def save_predictions(
    path: str | Path,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    run: FittedRun,
) -> None:
    rows = []
    for split_name, split_df, y_true, y_pred in [
        ("train", train, run.train_true, run.train_pred),
        ("validation", val, run.val_true, run.val_pred),
        ("test", test, run.test_true, run.test_pred),
    ]:
        dates = window_target_dates(split_df, run.look_back)
        rows.append(
            pd.DataFrame(
                {
                    "split": split_name,
                    "date": dates.to_numpy(),
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "true_direction": y_true > 0,
                    "pred_direction": y_pred > 0,
                }
            )
        )
    output = pd.concat(rows, ignore_index=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False, encoding="utf-8-sig")


def comma_list(text: str | None) -> list[str]:
    if text is None or str(text).strip() == "":
        return []
    return [item.strip() for item in str(text).split(",") if item.strip()]


def float_grid(text: str) -> list[float]:
    return [float(item) for item in comma_list(text)]


def int_grid(text: str) -> list[int]:
    return [int(item) for item in comma_list(text)]


def gamma_grid(text: str) -> list[str | float]:
    values: list[str | float] = []
    for item in comma_list(text):
        if item in {"scale", "auto"}:
            values.append(item)
        else:
            values.append(float(item))
    return values


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    raw_df = read_daily_data(args.data)
    feature_columns = build_feature_columns(
        raw_df,
        feature_set=args.feature_set,
        extra_features=comma_list(args.extra_features),
        drop_features=comma_list(args.drop_features),
    )
    df = load_feature_data(args.data, args.target, feature_columns)
    train, val, test, sizes = split_dataframe(df, args.split)

    if args.tune:
        run, tuning_results = tune_svr(
            train=train,
            val=val,
            test=test,
            feature_columns=feature_columns,
            target_column=args.target,
            look_back_grid=int_grid(args.look_back_grid),
            c_grid=float_grid(args.c_grid),
            gamma_grid=gamma_grid(args.gamma_grid),
            epsilon_grid=float_grid(args.epsilon_grid),
            scaler=args.scaler,
            objective=args.objective,
        )
    else:
        run = fit_once(
            train=train,
            val=val,
            test=test,
            feature_columns=feature_columns,
            target_column=args.target,
            look_back=args.look_back,
            c=args.c,
            gamma=args.gamma,
            epsilon=args.epsilon,
            scaler=args.scaler,
        )
        tuning_results = pd.DataFrame()

    train_target_mean = float(train[args.target].mean())
    baseline_rows: dict[str, dict[str, float]] = {}
    for split_name, split_df, y_true in [
        ("Train", train, run.train_true),
        ("Validation", val, run.val_true),
        ("Test", test, run.test_true),
    ]:
        for baseline_name, pred in baseline_predictions(split_df, y_true, run.look_back, train_target_mean).items():
            baseline_rows[f"{split_name} - {baseline_name}"] = evaluate_return_predictions(y_true, pred)

    validation_baselines = {
        name.replace("Validation - ", ""): metrics
        for name, metrics in baseline_rows.items()
        if name.startswith("Validation - ")
    }
    test_baselines = {
        name.replace("Test - ", ""): metrics
        for name, metrics in baseline_rows.items()
        if name.startswith("Test - ")
    }
    val_baseline_summary = best_baseline_summary(run.val_metrics, validation_baselines)
    test_baseline_summary = best_baseline_summary(run.test_metrics, test_baselines)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_path = None
    if args.plot_path:
        plot_path = Path(args.plot_path)
        if not plot_path.is_absolute():
            plot_path = output_dir / plot_path
        plot_return_predictions(train, val, test, args.target, run, plot_path)

    prediction_path = output_dir / "daily_svr_vnindex_predictions.csv"
    save_predictions(prediction_path, train, val, test, run)

    tuning_path = None
    if args.tune:
        tuning_path = output_dir / "daily_svr_vnindex_tuning_results.csv"
        tuning_results.to_csv(tuning_path, index=False, encoding="utf-8-sig")

    print("\nDaily sentiment SVR VNINDEX return prediction")
    print("=" * 90)
    print(f"Data: {Path(args.data)}")
    print(f"Target: {args.target}")
    print(f"Feature set: {args.feature_set}")
    print(f"Split: {args.split}")
    print(f"Scaler: {args.scaler}")
    print(f"Best look-back: {run.look_back} days")
    print(f"Best SVR params: {run.params}")
    print(f"Sequence/flat shapes: {run.prepared.shapes}")
    if plot_path:
        print(f"Plot: {plot_path}")
    print(f"Predictions: {prediction_path}")
    if tuning_path:
        print(f"Tuning results: {tuning_path}")

    print("\nSplit summary")
    print(f"  Total: {sizes['total']} | Train: {sizes['train']} | Validation: {sizes['val']} | Test: {sizes['test']}")
    print(f"  Train:      {sizes['train_start']} -> {sizes['train_end']}")
    print(f"  Validation: {sizes['val_start']} -> {sizes['val_end']}")
    print(f"  Test:       {sizes['test_start']} -> {sizes['test_end']}")

    print(f"\nFeatures ({len(feature_columns)})")
    for idx, feature in enumerate(feature_columns, start=1):
        print(f"  {idx:02d}. {feature}")

    print_metric_table(
        "Model metrics",
        {
            "Train": run.train_metrics,
            "Validation": run.val_metrics,
            "Test": run.test_metrics,
        },
    )
    print_metric_table("Baseline metrics", baseline_rows)

    print("\nBaseline comparison")
    print("  Validation:")
    for key, value in val_baseline_summary.items():
        print(f"    {key}: {value if isinstance(value, str) else format_metric(value)}")
    print("  Test:")
    for key, value in test_baseline_summary.items():
        print(f"    {key}: {value if isinstance(value, str) else format_metric(value)}")

    return {
        "train": run.train_metrics,
        "validation": run.val_metrics,
        "test": run.test_metrics,
        "baselines": baseline_rows,
        "validation_baseline_summary": val_baseline_summary,
        "test_baseline_summary": test_baseline_summary,
        "best_params": run.params,
        "look_back": run.look_back,
        "features": feature_columns,
        "prediction_path": str(prediction_path),
        "plot_path": str(plot_path) if plot_path else None,
        "tuning_path": str(tuning_path) if tuning_path else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SVR on daily VNINDEX sentiment and market features."
    )
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, type=Path)
    parser.add_argument("--target", default="future_ret_5d", choices=TARGET_COLUMNS)
    parser.add_argument("--split", default="701515", choices=sorted(SPLIT_RATIOS))
    parser.add_argument("--feature-set", default="combined", choices=["sentiment", "market", "combined"])
    parser.add_argument("--extra-features", default="")
    parser.add_argument("--drop-features", default="")
    parser.add_argument("--scaler", default="standard", choices=["standard", "robust"])
    parser.add_argument("--look-back", default=5, type=int)
    parser.add_argument("--c", default=1.0, type=float)
    parser.add_argument("--gamma", default="scale")
    parser.add_argument("--epsilon", default=0.01, type=float)

    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--objective", default="mae", choices=["mae", "rmse", "diracc", "corr"])
    parser.add_argument("--look-back-grid", default="1,5,10,20")
    parser.add_argument("--c-grid", default="0.1,0.5,1,2,5")
    parser.add_argument("--gamma-grid", default="scale,auto,0.001,0.005")
    parser.add_argument("--epsilon-grid", default="0.003,0.005,0.01,0.02")

    parser.add_argument(
        "--output-dir",
        default=Path(__file__).resolve().parent / "daily_sentiment_outputs",
        type=Path,
    )
    parser.add_argument(
        "--plot-path",
        default="daily_svr_return_prediction.png",
        help="File name/path for the plot. Omit to skip.",
    )

    args = parser.parse_args()
    if isinstance(args.gamma, str) and args.gamma not in {"scale", "auto"}:
        args.gamma = float(args.gamma)
    return args


def main() -> None:
    args = parse_args()
    result = run_pipeline(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "daily_svr_vnindex_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)


if __name__ == "__main__":
    main()
