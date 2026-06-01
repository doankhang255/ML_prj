from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


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

FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Year",
    "MA7",
    "MA14",
    "MA30",
    "Close_Lag_1",
    "Close_Lag_2",
    "Close_Lag_3",
    "Close_Lag_5",
]


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


def load_price_data(csv_path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df.rename(columns=COLUMN_MAP)

    required = ["Date", "Open", "High", "Low", "Close"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Date"] = parse_dates(df["Date"])
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = df[col].map(parse_number)

    df = df.dropna(subset=required).sort_values("Date").reset_index(drop=True)
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Year"] = out["Date"].dt.year
    out["MA7"] = out["Close"].rolling(window=7, min_periods=7).mean()
    out["MA14"] = out["Close"].rolling(window=14, min_periods=14).mean()
    out["MA30"] = out["Close"].rolling(window=30, min_periods=30).mean()
    for lag in [1, 2, 3, 5]:
        out[f"Close_Lag_{lag}"] = out["Close"].shift(lag)
    return out.dropna(subset=FEATURE_COLUMNS + ["Close"]).reset_index(drop=True)


def split_dataframe(
    df: pd.DataFrame,
    split_code: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    train_ratio, val_ratio, _ = SPLIT_RATIOS[split_code]
    n = len(df)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    train = df.iloc[:train_size].copy()
    val = df.iloc[train_size : train_size + val_size].copy()
    test = df.iloc[train_size + val_size :].copy()
    sizes = {"total": n, "train": len(train), "val": len(val), "test": len(test)}
    return train, val, test, sizes


def train_model(train: pd.DataFrame) -> LinearRegression:
    model = LinearRegression()
    model.fit(train[FEATURE_COLUMNS], train["Close"])
    return model


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    non_zero = y_true != 0
    return float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
    }


def predict_split(
    model: LinearRegression,
    data: pd.DataFrame,
) -> tuple[np.ndarray, dict[str, float]]:
    pred = model.predict(data[FEATURE_COLUMNS])
    metrics = evaluate_predictions(data["Close"].to_numpy(), pred)
    return pred, metrics


def plot_predictions(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    val_pred: np.ndarray,
    test_pred: np.ndarray,
    split_code: str,
    output_path: str | Path,
) -> None:
    plt.figure(figsize=(16, 6))
    plt.plot(train["Date"], train["Close"], label="Train", color="blue")
    plt.plot(val["Date"], val["Close"], label="Validate", color="red")
    plt.plot(test["Date"], test["Close"], label="Test", color="orange")
    plt.plot(val["Date"], val_pred, label="ValidatePred", color="purple", linestyle="--")
    plt.plot(test["Date"], test_pred, label="Predict", color="green", linestyle="--")
    plt.title(f"Linear Regression - Predict VNINDEX price ({split_code})")
    plt.xlabel("Date")
    plt.ylabel("Close price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_pipeline(args: argparse.Namespace) -> dict[str, dict[str, float]]:
    df = add_price_features(load_price_data(args.data))
    train, val, test, sizes = split_dataframe(df, args.split)
    model = train_model(train)
    val_pred, val_metrics = predict_split(model, val)
    test_pred, test_metrics = predict_split(model, test)

    if args.plot_path:
        plot_predictions(train, val, test, val_pred, test_pred, args.split, args.plot_path)

    print(f"Data: {Path(args.data)}")
    print(f"Split sizes: {sizes}")
    print(f"Validation: {val_metrics}")
    print(f"Test: {test_metrics}")
    return {"validation": val_metrics, "test": test_metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Linear Regression price prediction on VNINDEX data."
    )
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, type=Path)
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
