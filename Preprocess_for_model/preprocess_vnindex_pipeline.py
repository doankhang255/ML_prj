from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_INPUT_PATH = Path(__file__).resolve().parents[1] / "Dataset" / "data_VNINDEX.csv"
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / "Dataset" / "data_VNINDEX_processed.csv"
)

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
    "% Thay doi": "Pct_Change",
    "% Thay đổi": "Pct_Change",
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


def load_raw_data(input_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    df = df.rename(columns=COLUMN_MAP)
    required = ["Date", "Open", "High", "Low", "Close"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["Date"] = parse_dates(df["Date"])
    for col in ["Open", "High", "Low", "Close", "Volume", "Pct_Change"]:
        if col in df.columns:
            df[col] = df[col].map(parse_number)

    return df.dropna(subset=required).sort_values("Date").reset_index(drop=True)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Return"] = out["Close"].pct_change()
    out["Return_Pct"] = out["Return"] * 100
    out["Target_Date_1D"] = out["Date"].shift(-1)
    out["Future_Return_1D"] = out["Return"].shift(-1)
    out["Future_Return_1D_Pct"] = out["Future_Return_1D"] * 100
    future_close_10d = out["Close"].shift(-10)
    out["Future_Return_10D"] = future_close_10d / out["Close"] - 1
    out["High_Low_Spread"] = out["High"] - out["Low"]
    out["Close_Open_Change"] = out["Close"] - out["Open"]
    out["High_Low_Range_Pct"] = (out["High"] - out["Low"]) / out["Close"]
    out["Close_Open_Return"] = out["Close"] / out["Open"] - 1
    out["DayOfWeek"] = out["Date"].dt.dayofweek
    out["Month"] = out["Date"].dt.month
    out["Year"] = out["Date"].dt.year
    out["MA7"] = out["Close"].rolling(window=7, min_periods=7).mean()
    out["MA14"] = out["Close"].rolling(window=14, min_periods=14).mean()
    out["MA30"] = out["Close"].rolling(window=30, min_periods=30).mean()
    out["Close_MA7_Ratio"] = out["Close"] / out["MA7"] - 1
    out["Close_MA14_Ratio"] = out["Close"] / out["MA14"] - 1
    out["Close_MA30_Ratio"] = out["Close"] / out["MA30"] - 1
    out["MA7_MA14_Ratio"] = out["MA7"] / out["MA14"] - 1
    out["MA7_MA30_Ratio"] = out["MA7"] / out["MA30"] - 1
    out["Return_MA3"] = out["Return"].rolling(window=3, min_periods=3).mean()
    out["Return_MA5"] = out["Return"].rolling(window=5, min_periods=5).mean()
    out["Return_MA10"] = out["Return"].rolling(window=10, min_periods=10).mean()
    out["Volatility_7"] = out["Close"].rolling(window=7, min_periods=7).std()
    out["Volatility_14"] = out["Close"].rolling(window=14, min_periods=14).std()
    out["Return_Volatility_7"] = out["Return"].rolling(window=7, min_periods=7).std()
    out["Return_Volatility_14"] = out["Return"].rolling(window=14, min_periods=14).std()
    if "Volume" in out.columns:
        out["Volume_Change_Pct"] = out["Volume"].pct_change()
        out["Volume_Ratio_7"] = out["Volume"] / out["Volume"].rolling(window=7, min_periods=7).mean()
        out["Volume_Ratio_14"] = out["Volume"] / out["Volume"].rolling(window=14, min_periods=14).mean()
    for lag in [1, 2, 3, 5]:
        out[f"Close_Lag_{lag}"] = out["Close"].shift(lag)
        out[f"Return_Lag_{lag}"] = out["Return"].shift(lag)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.dropna().reset_index(drop=True)


def save_processed_data(df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def run_pipeline(args: argparse.Namespace) -> pd.DataFrame:
    raw = load_raw_data(args.input)
    processed = add_features(raw)
    output_path = save_processed_data(processed, args.output)
    print(f"Input: {Path(args.input)}")
    print(f"Output: {output_path}")
    print(f"Rows: raw={len(raw)}, processed={len(processed)}")
    print(f"Columns: {list(processed.columns)}")
    return processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create processed VNINDEX features, returns, and lag columns."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, type=Path)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
