from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler


DEFAULT_PROCESSED_DATA_PATH = (
    Path(__file__).resolve().parents[1] / "Dataset" / "data_VNINDEX_processed.csv"
)

SPLIT_RATIOS = {
    "652510": (0.65, 0.25, 0.10),
    "702010": (0.70, 0.20, 0.10),
    "751510": (0.75, 0.15, 0.10),
    "801010": (0.80, 0.10, 0.10),
}

TARGET_COLUMN = "Next_Close"
RETURN_1D_TARGET_COLUMN = "Future_Return_1D"
RETURN_10D_TARGET_COLUMN = "Future_Return_10D"

ALLOWED_TARGET_COLUMNS = {
    TARGET_COLUMN,
    RETURN_1D_TARGET_COLUMN,
    RETURN_10D_TARGET_COLUMN,
    "Next_Return",
    "Next_Return_Pct",
    "Future_Return_1D_Pct",
}

FORBIDDEN_FEATURE_COLUMNS = {
    "Next_Close",
    "Next_Return",
    "Next_Return_Pct",
    "Future_Return_1D",
    "Future_Return_1D_Pct",
    "Future_Close_10D",
    "Future_Return_10D",
}

# Technical features only. Keep this list deterministic for fair ablation tests.
TECHNICAL_FEATURE_COLUMNS = [
    "Return",
    "Return_Lag_1",
    "Return_Lag_3",
    "Return_Lag_5",
    "Return_MA10",
    "Return_MA20",
    "Return_MA50",
    "High_Low_Range_Pct",
    "Close_Open_Return",
    "Close_MA10_Ratio",
    "Close_MA50_Ratio",
    "MA10_MA50_Ratio",
    "Drawdown_60",
    "Negative_Return_Count_10",
    "Return_Volatility_14",
    "Return_Volatility_30",
    "Volume_Change_Pct",
    "Volume_Ratio_20",
]

# Backward-compatible alias used by older scripts.
FEATURE_COLUMNS = TECHNICAL_FEATURE_COLUMNS

NON_FEATURE_COLUMNS = {
    "Date",
    "Target_Date_1D",
    "Target_Date_10D",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Ticker",
    "Symbol",
}

SENTIMENT_NAME_HINTS = (
    "sentiment",
    "news",
    "article",
    "headline",
    "positive",
    "negative",
    "neutral",
    "pos_ratio",
    "neg_ratio",
    "neu_ratio",
    "polarity",
    "tone",
    "esi",
)


def _deduplicate_keep_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output


def is_numeric_feature(df: pd.DataFrame, column: str) -> bool:
    return column in df.columns and pd.api.types.is_numeric_dtype(df[column])


def infer_sentiment_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric columns that look like news/sentiment features.

    The function is intentionally conservative: it does not include raw OHLCV,
    date columns, or known future target columns. It only selects numeric columns
    whose names contain a sentiment/news related hint.
    """
    candidates: list[str] = []
    for column in df.columns:
        lower = column.lower()
        if column in NON_FEATURE_COLUMNS or column in FORBIDDEN_FEATURE_COLUMNS:
            continue
        if not is_numeric_feature(df, column):
            continue
        if any(hint in lower for hint in SENTIMENT_NAME_HINTS):
            candidates.append(column)
    return candidates


def build_feature_columns(
    df: pd.DataFrame,
    feature_set: str = "technical",
    extra_features: Sequence[str] | None = None,
    drop_features: Sequence[str] | None = None,
) -> list[str]:
    """Build feature list for technical-only, sentiment-only, or combined tests.

    Parameters
    ----------
    feature_set:
        - technical: only TECHNICAL_FEATURE_COLUMNS
        - sentiment: only inferred/provided sentiment features
        - combined/auto: technical + inferred/provided sentiment features
        - all_numeric: all numeric non-forbidden columns except raw OHLCV/date columns
    extra_features:
        Explicit feature names to add, useful when your sentiment column has a
        custom name not caught by the inference rules.
    drop_features:
        Feature names to remove after construction.
    """
    feature_set = feature_set.lower().strip()
    extras = list(extra_features or [])
    drops = set(drop_features or [])

    available_technical = [
        column for column in TECHNICAL_FEATURE_COLUMNS if is_numeric_feature(df, column)
    ]
    sentiment_features = infer_sentiment_feature_columns(df)

    if feature_set == "technical":
        features = available_technical + extras
    elif feature_set == "sentiment":
        features = sentiment_features + extras
    elif feature_set in {"combined", "auto"}:
        features = available_technical + sentiment_features + extras
    elif feature_set == "all_numeric":
        features = [
            column
            for column in df.columns
            if is_numeric_feature(df, column)
            and column not in NON_FEATURE_COLUMNS
            and column not in FORBIDDEN_FEATURE_COLUMNS
        ] + extras
    else:
        raise ValueError(
            "feature_set must be one of: technical, sentiment, combined, auto, all_numeric"
        )

    features = _deduplicate_keep_order(features)
    leaked = sorted(FORBIDDEN_FEATURE_COLUMNS.intersection(features))
    if leaked:
        raise ValueError(f"Forbidden future columns in feature set: {leaked}")

    missing = [column for column in features if column not in df.columns]
    if missing:
        raise ValueError(f"Requested feature columns do not exist in data: {missing}")

    non_numeric = [column for column in features if not is_numeric_feature(df, column)]
    if non_numeric:
        raise ValueError(f"Feature columns must be numeric: {non_numeric}")

    features = [column for column in features if column not in drops]
    if not features:
        raise ValueError("No feature columns selected.")
    return features


def load_feature_data(
    csv_path: str | Path = DEFAULT_PROCESSED_DATA_PATH,
    target_column: str = TARGET_COLUMN,
    feature_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date")
    if "Target_Date_1D" in df.columns:
        df["Target_Date_1D"] = pd.to_datetime(df["Target_Date_1D"], errors="coerce")
    if "Target_Date_10D" in df.columns:
        df["Target_Date_10D"] = pd.to_datetime(df["Target_Date_10D"], errors="coerce")

    if target_column not in df.columns:
        raise ValueError(f"Target column does not exist in data: {target_column}")

    selected_features = list(feature_columns or FEATURE_COLUMNS)
    leaked = sorted(FORBIDDEN_FEATURE_COLUMNS.intersection(selected_features))
    if leaked:
        raise ValueError(f"Forbidden future columns in selected features: {leaked}")

    required = selected_features + [target_column]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in processed data: {missing}")

    required_dates = [column for column in ["Date", "Target_Date_1D"] if column in df.columns]
    df = df.dropna(subset=required + required_dates).reset_index(drop=True)

    if "Date" in df.columns and not df["Date"].is_monotonic_increasing:
        raise ValueError("Date column must be sorted from oldest to newest.")
    return df


def split_dataframe(
    df: pd.DataFrame,
    split_code: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int | str]]:
    if split_code not in SPLIT_RATIOS:
        raise ValueError(f"Unknown split_code={split_code}. Choose from {sorted(SPLIT_RATIOS)}")

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
    if "Date" in df.columns:
        for name, split in [("train", train), ("val", val), ("test", test)]:
            if len(split) == 0:
                sizes[f"{name}_start"] = "NA"
                sizes[f"{name}_end"] = "NA"
            else:
                sizes[f"{name}_start"] = str(split["Date"].min().date())
                sizes[f"{name}_end"] = str(split["Date"].max().date())
    return train, val, test, sizes


def make_scaler(kind: str):
    kind = kind.lower().strip()
    if kind == "standard":
        return StandardScaler()
    if kind == "robust":
        return RobustScaler()
    raise ValueError("scaler must be either 'standard' or 'robust'")


def scale_feature_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target_column: str = TARGET_COLUMN,
    feature_columns: Sequence[str] | None = None,
    scaler: str = "standard",
):
    selected_features = list(feature_columns or FEATURE_COLUMNS)
    x_scaler = make_scaler(scaler)
    y_scaler = StandardScaler()

    x_train = x_scaler.fit_transform(train[selected_features])
    x_val = x_scaler.transform(val[selected_features])
    x_test = x_scaler.transform(test[selected_features])

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
        raise ValueError(f"Need at least look_back={look_back} rows, got {len(x_values)}.")

    x_windows, y_targets = [], []
    for i in range(look_back - 1, len(x_values)):
        x_windows.append(x_values[i - look_back + 1 : i + 1])
        y_targets.append(y_values[i])
    return np.asarray(x_windows), np.asarray(y_targets)


def inverse_target(values: np.ndarray, y_scaler: StandardScaler) -> np.ndarray:
    return y_scaler.inverse_transform(np.asarray(values).reshape(-1, 1)).ravel()


def window_values(split_df: pd.DataFrame, column: str, look_back: int) -> np.ndarray:
    return split_df[column].iloc[look_back - 1 :].to_numpy(dtype=float)


def window_current_close(split_df: pd.DataFrame, look_back: int) -> np.ndarray:
    return window_values(split_df, "Close", look_back)


def target_date_column(target_column: str, df: pd.DataFrame) -> str:
    if target_column == RETURN_10D_TARGET_COLUMN and "Target_Date_10D" in df.columns:
        return "Target_Date_10D"
    if "Target_Date_1D" in df.columns:
        return "Target_Date_1D"
    return "Date"
