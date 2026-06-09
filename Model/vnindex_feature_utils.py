from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DEFAULT_PROCESSED_DATA_PATH = (
    Path(__file__).resolve().parents[1] / "Dataset" / "data_VNINDEX_processed.csv"
)

SPLIT_RATIOS = {
    "652510": (0.65, 0.25, 0.10),
    "702010": (0.70, 0.20, 0.10),
    "751510": (0.75, 0.15, 0.10),
}

TARGET_COLUMN = "Next_Close"
RETURN_1D_TARGET_COLUMN = "Future_Return_1D"
RETURN_10D_TARGET_COLUMN = "Future_Return_10D"
FORBIDDEN_FEATURE_COLUMNS = {
    "Next_Close",
    "Next_Return",
    "Next_Return_Pct",
    "Future_Return_1D",
    "Future_Return_1D_Pct",
    "Future_Close_10D",
    "Future_Return_10D",
}

FEATURE_COLUMNS = [
    "Return",
    "Return_Pct",
    "Return_Lag_1",
    "Return_Lag_2",
    "Return_Lag_3",
    "Return_Lag_5",
    "Return_MA3",
    "Return_MA5",
    "Return_MA10",
    "High_Low_Range_Pct",
    "Close_Open_Return",
    "Close_MA7_Ratio",
    "Close_MA14_Ratio",
    "Close_MA30_Ratio",
    "MA7_MA14_Ratio",
    "MA7_MA30_Ratio",
    "Return_Volatility_7",
    "Return_Volatility_14",
    "Volume_Change_Pct",
    "Volume_Ratio_7",
    "Volume_Ratio_14",
]


def load_feature_data(
    csv_path: str | Path = DEFAULT_PROCESSED_DATA_PATH,
    target_column: str = TARGET_COLUMN,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date")
    if "Target_Date_1D" in df.columns:
        df["Target_Date_1D"] = pd.to_datetime(df["Target_Date_1D"], errors="coerce")

    required = FEATURE_COLUMNS + [target_column]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in processed data: {missing}")

    leaked = sorted(FORBIDDEN_FEATURE_COLUMNS.intersection(FEATURE_COLUMNS))
    if leaked:
        raise ValueError(f"Forbidden future columns in FEATURE_COLUMNS: {leaked}")

    required_dates = [col for col in ["Date", "Target_Date_1D"] if col in df.columns]
    df = df.dropna(subset=required + required_dates).reset_index(drop=True)
    if "Date" in df.columns and not df["Date"].is_monotonic_increasing:
        raise ValueError("Date column must be sorted from oldest to newest.")
    return df


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
    if "Date" in df.columns:
        sizes.update(
            {
                "train_start": str(train["Date"].min().date()),
                "train_end": str(train["Date"].max().date()),
                "val_start": str(val["Date"].min().date()),
                "val_end": str(val["Date"].max().date()),
                "test_start": str(test["Date"].min().date()),
                "test_end": str(test["Date"].max().date()),
            }
        )
    return train, val, test, sizes


def scale_feature_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, StandardScaler]:
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_train = x_scaler.fit_transform(train[FEATURE_COLUMNS])
    x_val = x_scaler.transform(val[FEATURE_COLUMNS])
    x_test = x_scaler.transform(test[FEATURE_COLUMNS])

    y_train = y_scaler.fit_transform(train[[target_column]])
    y_val = y_scaler.transform(val[[target_column]])
    y_test = y_scaler.transform(test[[target_column]])
    return x_train, y_train, x_val, y_val, x_test, y_test, x_scaler, y_scaler


def create_sequence_windows(
    x_values: np.ndarray,
    y_values: np.ndarray,
    look_back: int,
) -> tuple[np.ndarray, np.ndarray]:
    if look_back < 1:
        raise ValueError("look_back must be at least 1.")
    if len(x_values) < look_back:
        raise ValueError(
            f"Need at least look_back={look_back} rows, got {len(x_values)}."
        )

    x_windows, y_targets = [], []
    for i in range(look_back - 1, len(x_values)):
        x_windows.append(x_values[i - look_back + 1 : i + 1])
        y_targets.append(y_values[i])
    return np.array(x_windows), np.array(y_targets)


def inverse_target(values: np.ndarray, y_scaler: StandardScaler) -> np.ndarray:
    return y_scaler.inverse_transform(np.asarray(values).reshape(-1, 1)).ravel()


def window_current_close(split_df: pd.DataFrame, look_back: int) -> np.ndarray:
    return split_df["Close"].iloc[look_back - 1 :].to_numpy(dtype=float)
