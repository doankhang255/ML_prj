from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_INPUT_PATH = Path(__file__).resolve().parents[1] / "Dataset" / "data_VNINDEX.csv"
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / "Dataset" / "data_VNINDEX_processed.csv"
)

# Raw vnstock-style column names. The loader also accepts common uppercase aliases.
COLUMN_ALIASES = {
    "Date": ["time", "date", "Date", "datetime", "Datetime"],
    "Open": ["open", "Open"],
    "High": ["high", "High"],
    "Low": ["low", "Low"],
    "Close": ["close", "Close"],
    "Volume": ["volume", "Volume"],
}
PROCESSED_BASE_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]

# Must match the technical feature list used by vnindex_feature_utils_fixed.py.
REQUIRED_TECHNICAL_FEATURES = [
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

REQUIRED_TARGET_COLUMNS = [
    "Target_Date_1D",
    "Next_Close",
    "Next_Return",
    "Next_Return_Pct",
    "Future_Return_1D",
    "Future_Return_1D_Pct",
    "Target_Date_10D",
    "Future_Close_10D",
    "Future_Return_10D",
]


LABEL_COLUMN_CANDIDATES = [
    "sentiment_label",
    "Sentiment_Label",
    "label",
    "Label",
    "sentiment",
    "Sentiment",
    "pred_label",
    "Pred_Label",
]


def parse_dates(values: pd.Series) -> pd.Series:
    """Parse both yyyy-mm-dd and Vietnamese dd/mm/yyyy style dates."""
    sample = values.dropna().astype(str).head(1)
    if not sample.empty and "-" in sample.iloc[0] and len(sample.iloc[0].split("-")[0]) == 4:
        return pd.to_datetime(values, errors="coerce")
    return pd.to_datetime(values, dayfirst=True, errors="coerce")


def find_first_column(df: pd.DataFrame, aliases: Iterable[str], logical_name: str) -> str:
    for column in aliases:
        if column in df.columns:
            return column
    raise ValueError(
        f"Missing required column for {logical_name}. "
        f"Accepted names: {list(aliases)}. Available columns: {list(df.columns)}"
    )


def to_numeric_clean(series: pd.Series) -> pd.Series:
    """Convert numeric-looking strings to float while tolerating comma separators."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    text = (
        series.astype(str)
        .str.strip()
        .str.replace("\u00a0", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    return pd.to_numeric(text, errors="coerce")


def safe_divide(numerator, denominator):
    denominator = pd.Series(denominator).replace(0, np.nan)
    return numerator / denominator


def load_raw_data(input_path: str | Path) -> pd.DataFrame:
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw = pd.read_csv(input_path, encoding="utf-8-sig")
    rename_map = {
        find_first_column(raw, aliases, logical_name): logical_name
        for logical_name, aliases in COLUMN_ALIASES.items()
    }

    df = raw[list(rename_map.keys())].rename(columns=rename_map).copy()
    df = df[PROCESSED_BASE_COLUMNS]
    df["Date"] = parse_dates(df["Date"])
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        df[column] = to_numeric_clean(df[column])

    before = len(df)
    df = df.dropna(subset=PROCESSED_BASE_COLUMNS)
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")
    df = df.reset_index(drop=True)

    if len(df) < 80:
        raise ValueError(
            f"Too few valid rows after cleaning: {len(df)}. "
            "Need at least about 80 rows because 60-day rolling features and 10-day future target are created."
        )
    if before != len(df):
        print(f"Dropped/merged raw rows: {before - len(df)}")
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create current/past-only technical features plus future target columns.

    Technical features use information available at the current row or earlier.
    Target columns are shifted future values and must not be used as model features.
    """
    out = df.copy()

    # Core 1-day target and backward-looking return.
    out["Return"] = out["Close"].pct_change()
    out["Return_Pct"] = out["Return"] * 100
    out["Target_Date_1D"] = out["Date"].shift(-1)
    out["Next_Close"] = out["Close"].shift(-1)
    out["Next_Return"] = safe_divide(out["Next_Close"], out["Close"]) - 1
    out["Next_Return_Pct"] = out["Next_Return"] * 100
    out["Future_Return_1D"] = out["Next_Return"]
    out["Future_Return_1D_Pct"] = out["Future_Return_1D"] * 100

    # Less noisy medium-horizon target used by svr_pipeline_vnindex_fixed.py.
    out["Target_Date_10D"] = out["Date"].shift(-10)
    out["Future_Close_10D"] = out["Close"].shift(-10)
    out["Future_Return_10D"] = safe_divide(out["Future_Close_10D"], out["Close"]) - 1

    # Intraday shape features.
    out["High_Low_Spread"] = out["High"] - out["Low"]
    out["Close_Open_Change"] = out["Close"] - out["Open"]
    out["High_Low_Range_Pct"] = safe_divide(out["High"] - out["Low"], out["Close"])
    out["Close_Open_Return"] = safe_divide(out["Close"], out["Open"]) - 1

    # Calendar features, useful when running --feature-set all_numeric.
    out["DayOfWeek"] = out["Date"].dt.dayofweek
    out["Month"] = out["Date"].dt.month
    out["Year"] = out["Date"].dt.year
    out["DayOfWeek_Sin"] = np.sin(2 * np.pi * out["DayOfWeek"] / 5)
    out["DayOfWeek_Cos"] = np.cos(2 * np.pi * out["DayOfWeek"] / 5)
    out["Month_Sin"] = np.sin(2 * np.pi * out["Month"] / 12)
    out["Month_Cos"] = np.cos(2 * np.pi * out["Month"] / 12)

    # Moving averages and price location.
    for window in [5, 10, 20, 50, 100, 200]:
        out[f"MA{window}"] = out["Close"].rolling(window=window, min_periods=window).mean()
        out[f"Close_MA{window}_Ratio"] = safe_divide(out["Close"], out[f"MA{window}"]) - 1

    out["MA10_MA50_Ratio"] = safe_divide(out["MA10"], out["MA50"]) - 1
    out["MA20_MA50_Ratio"] = safe_divide(out["MA20"], out["MA50"]) - 1
    out["MA50_MA200_Ratio"] = safe_divide(out["MA50"], out["MA200"]) - 1

    # Return momentum / reversal features.
    for window in [3, 5, 10, 20, 50]:
        out[f"Return_MA{window}"] = out["Return"].rolling(window=window, min_periods=window).mean()
        out[f"Return_Sum_{window}"] = out["Return"].rolling(window=window, min_periods=window).sum()

    # Risk and drawdown features.
    out["Drawdown_60"] = safe_divide(out["Close"], out["Close"].rolling(window=60, min_periods=60).max()) - 1
    out["Drawdown_120"] = safe_divide(out["Close"], out["Close"].rolling(window=120, min_periods=120).max()) - 1
    out["Negative_Return_Count_10"] = (out["Return"] < 0).rolling(window=10, min_periods=10).sum()
    out["Positive_Return_Count_10"] = (out["Return"] > 0).rolling(window=10, min_periods=10).sum()

    for window in [7, 14, 30, 60]:
        out[f"Volatility_{window}"] = out["Close"].rolling(window=window, min_periods=window).std()
        out[f"Return_Volatility_{window}"] = out["Return"].rolling(window=window, min_periods=window).std()

    # Volume features.
    out["Volume_Change_Pct"] = out["Volume"].pct_change()
    out["Log_Volume"] = np.log1p(out["Volume"].clip(lower=0))
    for window in [7, 14, 20, 50]:
        volume_mean = out["Volume"].rolling(window=window, min_periods=window).mean()
        volume_std = out["Volume"].rolling(window=window, min_periods=window).std()
        out[f"Volume_Ratio_{window}"] = safe_divide(out["Volume"], volume_mean)
        out[f"Volume_Z_{window}"] = safe_divide(out["Volume"] - volume_mean, volume_std)

    # Common technical indicators for optional --feature-set all_numeric testing.
    delta = out["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = safe_divide(avg_gain, avg_loss)
    out["RSI_14"] = 100 - safe_divide(100, 1 + rs)

    ema12 = out["Close"].ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = out["Close"].ewm(span=26, adjust=False, min_periods=26).mean()
    out["MACD_12_26"] = ema12 - ema26
    out["MACD_Signal_9"] = out["MACD_12_26"].ewm(span=9, adjust=False, min_periods=9).mean()
    out["MACD_Hist"] = out["MACD_12_26"] - out["MACD_Signal_9"]

    middle = out["Close"].rolling(window=20, min_periods=20).mean()
    std20 = out["Close"].rolling(window=20, min_periods=20).std()
    upper = middle + 2 * std20
    lower = middle - 2 * std20
    out["BB_Width_20"] = safe_divide(upper - lower, middle)
    out["BB_Position_20"] = safe_divide(out["Close"] - lower, upper - lower)

    prev_close = out["Close"].shift(1)
    true_range = pd.concat(
        [
            out["High"] - out["Low"],
            (out["High"] - prev_close).abs(),
            (out["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["ATR_14"] = true_range.rolling(window=14, min_periods=14).mean()
    out["ATR_14_Ratio"] = safe_divide(out["ATR_14"], out["Close"])

    for lag in [1, 2, 3, 5, 10, 20]:
        out[f"Close_Lag_{lag}"] = out["Close"].shift(lag)
        out[f"Return_Lag_{lag}"] = out["Return"].shift(lag)
        out[f"Volume_Lag_{lag}"] = out["Volume"].shift(lag)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def standardize_label_value(value: object) -> str:
    text = str(value).strip().lower()
    if any(token in text for token in ["positive", "pos", "tich_cuc", "tích cực", "bull", "1"]):
        return "positive"
    if any(token in text for token in ["negative", "neg", "tieu_cuc", "tiêu cực", "bear", "-1"]):
        return "negative"
    if any(token in text for token in ["neutral", "neu", "trung_lap", "trung lập", "0"]):
        return "neutral"
    return "unknown"


def prepare_daily_sentiment(
    sentiment_path: str | Path,
    date_column: str = "Date",
    numeric_prefix: str = "Sentiment_",
) -> tuple[pd.DataFrame, list[str]]:
    """Aggregate article-level or daily sentiment data to one row per date.

    Numeric sentiment columns are averaged by day. If a text label column exists,
    daily positive/negative/neutral counts and ratios are created.
    """
    sentiment_path = Path(sentiment_path)
    if not sentiment_path.exists():
        raise FileNotFoundError(f"Sentiment file not found: {sentiment_path}")

    if sentiment_path.suffix.lower() == ".parquet":
        raw = pd.read_parquet(sentiment_path)
    else:
        raw = pd.read_csv(sentiment_path, encoding="utf-8-sig")

    if date_column not in raw.columns:
        # Try common fallbacks before failing.
        fallback = None
        for candidate in ["Date", "date", "publication_date", "published_date", "time"]:
            if candidate in raw.columns:
                fallback = candidate
                break
        if fallback is None:
            raise ValueError(
                f"Sentiment date column not found: {date_column}. Available columns: {list(raw.columns)}"
            )
        date_column = fallback

    raw = raw.copy()
    raw["Date"] = parse_dates(raw[date_column])
    raw = raw.dropna(subset=["Date"])
    raw["Date"] = raw["Date"].dt.normalize()

    feature_parts: list[pd.DataFrame] = []

    numeric_columns = [
        column
        for column in raw.columns
        if column != "Date" and column != date_column and pd.api.types.is_numeric_dtype(raw[column])
    ]
    if numeric_columns:
        numeric_daily = raw.groupby("Date")[numeric_columns].mean()
        rename = {
            column: column if column.lower().startswith(("sentiment", "news", "positive", "negative", "neutral")) else f"{numeric_prefix}{column}"
            for column in numeric_columns
        }
        numeric_daily = numeric_daily.rename(columns=rename)
        feature_parts.append(numeric_daily)

    label_column = next((column for column in LABEL_COLUMN_CANDIDATES if column in raw.columns), None)
    if label_column is not None and not pd.api.types.is_numeric_dtype(raw[label_column]):
        labels = raw[["Date", label_column]].copy()
        labels["_label_norm"] = labels[label_column].map(standardize_label_value)
        counts = pd.crosstab(labels["Date"], labels["_label_norm"])
        for column in ["positive", "negative", "neutral", "unknown"]:
            if column not in counts.columns:
                counts[column] = 0
        counts = counts[["positive", "negative", "neutral", "unknown"]]
        counts = counts.rename(
            columns={
                "positive": "News_Positive_Count",
                "negative": "News_Negative_Count",
                "neutral": "News_Neutral_Count",
                "unknown": "News_Unknown_Count",
            }
        )
        counts["News_Count"] = counts.sum(axis=1)
        counts["Sentiment_Pos_Ratio"] = safe_divide(counts["News_Positive_Count"], counts["News_Count"])
        counts["Sentiment_Neg_Ratio"] = safe_divide(counts["News_Negative_Count"], counts["News_Count"])
        counts["Sentiment_Neu_Ratio"] = safe_divide(counts["News_Neutral_Count"], counts["News_Count"])
        counts["Sentiment_Index"] = counts["Sentiment_Pos_Ratio"] - counts["Sentiment_Neg_Ratio"]
        feature_parts.append(counts)
    else:
        counts = raw.groupby("Date").size().to_frame("News_Count")
        feature_parts.append(counts)

    if not feature_parts:
        raise ValueError("No usable numeric or label sentiment features were found.")

    daily = pd.concat(feature_parts, axis=1)
    daily = daily.loc[:, ~daily.columns.duplicated()].reset_index()
    sentiment_features = [column for column in daily.columns if column != "Date"]
    return daily, sentiment_features


def merge_sentiment_features(
    price_features: pd.DataFrame,
    sentiment_path: str | Path | None,
    sentiment_date_column: str,
    sentiment_lag: int,
    fill_missing_sentiment: bool,
) -> tuple[pd.DataFrame, list[str]]:
    if not sentiment_path:
        return price_features, []

    daily_sentiment, sentiment_features = prepare_daily_sentiment(
        sentiment_path=sentiment_path,
        date_column=sentiment_date_column,
    )
    out = price_features.copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.normalize()
    out = out.merge(daily_sentiment, on="Date", how="left")

    if sentiment_lag > 0:
        out[sentiment_features] = out[sentiment_features].shift(sentiment_lag)

    if fill_missing_sentiment:
        out[sentiment_features] = out[sentiment_features].fillna(0.0)

    return out, sentiment_features


def finalize_processed_data(df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    out = df.replace([np.inf, -np.inf], np.nan).copy()
    if dropna:
        required = PROCESSED_BASE_COLUMNS + REQUIRED_TECHNICAL_FEATURES + REQUIRED_TARGET_COLUMNS
        missing = [column for column in required if column not in out.columns]
        if missing:
            raise ValueError(f"Missing generated required columns before dropna: {missing}")
        out = out.dropna(subset=required)
    out = out.sort_values("Date").reset_index(drop=True)
    return out


def validate_processed_data(df: pd.DataFrame) -> None:
    required = PROCESSED_BASE_COLUMNS + REQUIRED_TECHNICAL_FEATURES + REQUIRED_TARGET_COLUMNS
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Processed data is missing required columns: {missing}")
    non_numeric = [
        column
        for column in REQUIRED_TECHNICAL_FEATURES + ["Future_Return_1D", "Future_Return_10D"]
        if not pd.api.types.is_numeric_dtype(df[column])
    ]
    if non_numeric:
        raise ValueError(f"Required model columns must be numeric: {non_numeric}")
    if not pd.to_datetime(df["Date"], errors="coerce").is_monotonic_increasing:
        raise ValueError("Processed Date column must be sorted from oldest to newest.")


def save_processed_data(df: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def run_pipeline(
    input_path: str | Path = DEFAULT_INPUT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    sentiment_path: str | Path | None = None,
    sentiment_date_column: str = "Date",
    sentiment_lag: int = 0,
    fill_missing_sentiment: bool = True,
) -> pd.DataFrame:
    raw = load_raw_data(input_path)
    featured = add_price_features(raw)
    featured, sentiment_features = merge_sentiment_features(
        price_features=featured,
        sentiment_path=sentiment_path,
        sentiment_date_column=sentiment_date_column,
        sentiment_lag=sentiment_lag,
        fill_missing_sentiment=fill_missing_sentiment,
    )
    processed = finalize_processed_data(featured, dropna=True)
    validate_processed_data(processed)
    output_path = save_processed_data(processed, output_path)

    print("VNINDEX preprocessing - fixed pipeline")
    print("=" * 80)
    print(f"Input: {Path(input_path)}")
    print(f"Output: {output_path}")
    print(f"Rows: raw={len(raw)}, processed={len(processed)}")
    print(f"Date range: {processed['Date'].min()} -> {processed['Date'].max()}")
    print(f"Technical features ready: {len(REQUIRED_TECHNICAL_FEATURES)}")
    print(f"Sentiment features added: {len(sentiment_features)}")
    if sentiment_features:
        for idx, column in enumerate(sentiment_features, start=1):
            print(f"  {idx:02d}. {column}")
    print(f"Columns: {list(processed.columns)}")
    return processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess VNINDEX OHLCV data for the fixed SVR pipeline."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH, type=Path)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, type=Path)
    parser.add_argument(
        "--sentiment-data",
        default=None,
        type=Path,
        help="Optional daily/article-level sentiment CSV or Parquet to merge by Date.",
    )
    parser.add_argument(
        "--sentiment-date-col",
        default="Date",
        help="Date column name in the optional sentiment file.",
    )
    parser.add_argument(
        "--sentiment-lag",
        default=0,
        type=int,
        help="Lag sentiment features by this many trading rows after merging. Use 1 to avoid same-day timing ambiguity.",
    )
    parser.add_argument(
        "--no-fill-missing-sentiment",
        action="store_true",
        help="Do not fill missing sentiment days with 0. By default they are filled with 0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        input_path=args.input,
        output_path=args.output,
        sentiment_path=args.sentiment_data,
        sentiment_date_column=args.sentiment_date_col,
        sentiment_lag=args.sentiment_lag,
        fill_missing_sentiment=not args.no_fill_missing_sentiment,
    )


if __name__ == "__main__":
    main()
