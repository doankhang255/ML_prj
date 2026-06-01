from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[2] / "Dataset" / "data_VNINDEX.csv"
)

SPLIT_RATIOS = {
    "652510": (0.65, 0.25, 0.10),
    "702010": (0.70, 0.20, 0.10),
    "751510": (0.75, 0.15, 0.10),
}

COLUMN_MAP = {
    "time": "Date",
    "date": "Date",
    "Ngay": "Date",
    "Ngày": "Date",
    "open": "Open",
    "Mo": "Open",
    "Mở": "Open",
    "high": "High",
    "Cao": "High",
    "low": "Low",
    "Thap": "Low",
    "Thấp": "Low",
    "close": "Close",
    "Lan cuoi": "Close",
    "Lần cuối": "Close",
    "volume": "Volume",
    "KL": "Volume",
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
        raise ValueError("Input CSV must contain a close price column.")

    df["Close"] = df["Close"].map(parse_number)
    df = df.dropna(subset=["Close"]).reset_index(drop=True)
    return df[["Close"]]


def split_series(
    values: np.ndarray,
    split_code: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    train_ratio, val_ratio, _ = SPLIT_RATIOS[split_code]
    n = len(values)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    train = values[:train_size]
    val = values[train_size : train_size + val_size]
    test = values[train_size + val_size :]
    sizes = {"total": n, "train": len(train), "val": len(val), "test": len(test)}
    return train, val, test, sizes


def scale_splits(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    val_scaled = scaler.transform(val)
    test_scaled = scaler.transform(test)
    return train_scaled, val_scaled, test_scaled, scaler


def create_windows(data: np.ndarray, look_back: int) -> tuple[np.ndarray, np.ndarray]:
    x_values, y_values = [], []
    for i in range(look_back, len(data)):
        x_values.append(data[i - look_back : i])
        y_values.append(data[i])
    x = np.array(x_values).reshape(len(x_values), -1)
    y = np.array(y_values).ravel()
    return x, y


def train_svr(
    x_train: np.ndarray,
    y_train: np.ndarray,
    c: float,
    gamma: str | float,
    epsilon: float,
) -> SVR:
    model = SVR(kernel="rbf", C=c, gamma=gamma, epsilon=epsilon)
    model.fit(x_train, y_train)
    return model


def inverse_close(values: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return scaler.inverse_transform(np.asarray(values).reshape(-1, 1)).ravel()


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    non_zero = y_true != 0
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100),
    }


def predict_and_evaluate(
    model: SVR,
    x_values: np.ndarray,
    y_values: np.ndarray,
    scaler: StandardScaler,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    pred_scaled = model.predict(x_values)
    y_true = inverse_close(y_values, scaler)
    y_pred = inverse_close(pred_scaled, scaler)
    return y_true, y_pred, evaluate_predictions(y_true, y_pred)


def naive_previous_close(
    x_values: np.ndarray,
    y_values: np.ndarray,
    scaler: StandardScaler,
) -> dict[str, float]:
    last_scaled = x_values[:, -1]
    y_true = inverse_close(y_values, scaler)
    y_pred = inverse_close(last_scaled, scaler)
    return evaluate_predictions(y_true, y_pred)


def plot_predictions(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
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
    plt.plot(x_train, train.ravel(), label="Train", color="blue")
    plt.plot(x_val, val.ravel(), label="Validate", color="red")
    plt.plot(x_test, test.ravel(), label="Test", color="orange")
    plt.plot(x_val[look_back:], val_pred, label="ValidatePred", color="purple")
    plt.plot(x_test[look_back:], test_pred, label="Predict", color="green")
    plt.title(f"SVR - Predict VNINDEX price ({split_code})")
    plt.xlabel("Time Step")
    plt.ylabel("Close price")
    plt.legend()
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_pipeline(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    df = load_close_prices(args.data)
    close_values = df[["Close"]].to_numpy(dtype=float)
    train_raw, val_raw, test_raw, sizes = split_series(close_values, args.split)
    train_scaled, val_scaled, test_scaled, scaler = scale_splits(train_raw, val_raw, test_raw)

    x_train, y_train = create_windows(train_scaled, args.look_back)
    x_val, y_val = create_windows(val_scaled, args.look_back)
    x_test, y_test = create_windows(test_scaled, args.look_back)

    model = train_svr(x_train, y_train, args.c, args.gamma, args.epsilon)
    _, val_pred, val_metrics = predict_and_evaluate(model, x_val, y_val, scaler)
    _, test_pred, test_metrics = predict_and_evaluate(model, x_test, y_test, scaler)

    baseline_val = naive_previous_close(x_val, y_val, scaler)
    baseline_test = naive_previous_close(x_test, y_test, scaler)

    if args.plot_path:
        plot_predictions(
            train_raw,
            val_raw,
            test_raw,
            val_pred,
            test_pred,
            args.look_back,
            args.split,
            args.plot_path,
        )

    print(f"Data: {Path(args.data)}")
    print(f"Split sizes: {sizes}")
    print(f"Validation: {val_metrics}")
    print(f"Test: {test_metrics}")
    print(f"Naive validation: {baseline_val}")
    print(f"Naive test: {baseline_test}")

    return {
        "validation": val_metrics,
        "test": test_metrics,
        "naive_validation": baseline_val,
        "naive_test": baseline_test,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SVR price prediction on VNINDEX data.")
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, type=Path)
    parser.add_argument("--split", default="702010", choices=sorted(SPLIT_RATIOS))
    parser.add_argument("--look-back", default=30, type=int)
    parser.add_argument("--c", default=100.0, type=float)
    parser.add_argument("--gamma", default=0.1)
    parser.add_argument("--epsilon", default=0.01, type=float)
    parser.add_argument(
        "--plot-path",
        default=Path(__file__).resolve().parent / "svr_prediction.png",
        type=Path,
        help="Set to an empty string to skip plot output.",
    )
    args = parser.parse_args()
    if args.plot_path == Path("."):
        args.plot_path = None
    if isinstance(args.gamma, str):
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
