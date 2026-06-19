from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "Dataset"
VNINDEX_DAILY_PATH = DATASET_DIR / "vnindex_eda_output.csv"
SENTIMENT_DAILY_PATH = DATASET_DIR / "market_sentiment_index_daily.parquet"
OUTPUT_PARQUET_PATH = DATASET_DIR / "vnindex_daily_sentiment_merged.parquet"
OUTPUT_CSV_PATH = DATASET_DIR / "vnindex_daily_sentiment_merged.csv"

SYMBOL_COLUMN = "symbol"
DATE_COLUMN = "date"
CLOSE_COLUMN = "close_price"
OPEN_COLUMN = "open_price"
HIGH_COLUMN = "high_price"
LOW_COLUMN = "low_price"
VOLUME_COLUMN = "vol_total"
VALUE_COLUMN = "val_total"

SENTIMENT_SCORE_COLUMN = "sentiment_index"
ARTICLE_COUNT_COLUMN = "article_count"


def standardize_series(values: pd.Series) -> pd.Series:
    standard_deviation = values.std()
    if pd.isna(standard_deviation) or standard_deviation == 0:
        return pd.Series(0.0, index=values.index)
    return (values - values.mean()) / standard_deviation


def prepare_vnindex_daily(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if SYMBOL_COLUMN in out.columns:
        out[SYMBOL_COLUMN] = out[SYMBOL_COLUMN].astype("string").str.strip().str.upper()
        out = out.loc[out[SYMBOL_COLUMN].eq("VNINDEX")].copy()

    out[DATE_COLUMN] = pd.to_datetime(out[DATE_COLUMN], errors="coerce").dt.normalize()
    for column in [
        OPEN_COLUMN,
        HIGH_COLUMN,
        LOW_COLUMN,
        CLOSE_COLUMN,
        VOLUME_COLUMN,
        VALUE_COLUMN,
    ]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    out = out.loc[
        out[DATE_COLUMN].notna()
        & out[CLOSE_COLUMN].notna()
        & out[CLOSE_COLUMN].gt(0)
    ].copy()
    out = out.sort_values(DATE_COLUMN, kind="mergesort")
    out = out.drop_duplicates(subset=[DATE_COLUMN], keep="last").reset_index(drop=True)

    out["daily_return"] = np.log(out[CLOSE_COLUMN] / out[CLOSE_COLUMN].shift(1))
    out["future_ret_1d"] = np.log(out[CLOSE_COLUMN].shift(-1) / out[CLOSE_COLUMN])
    out["future_ret_5d"] = np.log(out[CLOSE_COLUMN].shift(-5) / out[CLOSE_COLUMN])
    out["future_ret_20d"] = np.log(out[CLOSE_COLUMN].shift(-20) / out[CLOSE_COLUMN])
    out["return_lag_1d"] = out["daily_return"].shift(1)
    out["volatility_20d"] = out["daily_return"].rolling(20).std()
    out["log_vol_total"] = np.log1p(out[VOLUME_COLUMN])
    out["log_val_total"] = np.log1p(out[VALUE_COLUMN])

    return out[
        [
            DATE_COLUMN,
            OPEN_COLUMN,
            HIGH_COLUMN,
            LOW_COLUMN,
            CLOSE_COLUMN,
            VOLUME_COLUMN,
            VALUE_COLUMN,
            "daily_return",
            "future_ret_1d",
            "future_ret_5d",
            "future_ret_20d",
            "return_lag_1d",
            "volatility_20d",
            "log_vol_total",
            "log_val_total",
        ]
    ]


def map_to_next_trading_date(
    sentiment_dates: pd.Series,
    trading_dates: pd.Series,
) -> pd.Series:
    trading_date_values = pd.to_datetime(trading_dates).sort_values().to_numpy(
        dtype="datetime64[ns]"
    )
    sentiment_date_values = pd.to_datetime(sentiment_dates).to_numpy(dtype="datetime64[ns]")

    # Strictly next trading date: news on day t is used from the next market session.
    positions = np.searchsorted(
        trading_date_values,
        sentiment_date_values,
        side="right",
    )

    mapped_dates = np.full(len(sentiment_date_values), np.datetime64("NaT"), dtype="datetime64[ns]")
    valid_positions = positions < len(trading_date_values)
    mapped_dates[valid_positions] = trading_date_values[positions[valid_positions]]
    return pd.Series(pd.to_datetime(mapped_dates), index=sentiment_dates.index)


def prepare_effective_sentiment(
    sentiment_df: pd.DataFrame,
    trading_dates: pd.Series,
) -> pd.DataFrame:
    out = sentiment_df.copy()
    out[DATE_COLUMN] = pd.to_datetime(out[DATE_COLUMN], errors="coerce").dt.normalize()
    out[ARTICLE_COUNT_COLUMN] = pd.to_numeric(out[ARTICLE_COUNT_COLUMN], errors="coerce")
    out[SENTIMENT_SCORE_COLUMN] = pd.to_numeric(out[SENTIMENT_SCORE_COLUMN], errors="coerce")

    for column in [
        "positive_article_count",
        "negative_article_count",
        "neutral_article_count",
    ]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0)

    out = out.loc[
        out[DATE_COLUMN].notna()
        & out[ARTICLE_COUNT_COLUMN].notna()
        & out[ARTICLE_COUNT_COLUMN].gt(0)
        & out[SENTIMENT_SCORE_COLUMN].notna()
    ].copy()

    out["effective_trading_date"] = map_to_next_trading_date(
        out[DATE_COLUMN],
        trading_dates,
    )
    out = out.loc[out["effective_trading_date"].notna()].copy()
    out["weighted_sentiment"] = out[SENTIMENT_SCORE_COLUMN] * out[ARTICLE_COUNT_COLUMN]

    effective_sentiment = out.groupby("effective_trading_date", sort=True).agg(
        sentiment_calendar_start=(DATE_COLUMN, "min"),
        sentiment_calendar_end=(DATE_COLUMN, "max"),
        source_calendar_day_count=(DATE_COLUMN, "size"),
        article_count=(ARTICLE_COUNT_COLUMN, "sum"),
        weighted_sentiment=("weighted_sentiment", "sum"),
        positive_article_count=("positive_article_count", "sum"),
        negative_article_count=("negative_article_count", "sum"),
        neutral_article_count=("neutral_article_count", "sum"),
    )
    effective_sentiment = effective_sentiment.reset_index()
    effective_sentiment["sentiment_index"] = (
        effective_sentiment["weighted_sentiment"] / effective_sentiment["article_count"]
    )
    effective_sentiment["sentiment_index_z"] = standardize_series(
        effective_sentiment["sentiment_index"]
    )
    effective_sentiment["log_article_count"] = np.log1p(
        effective_sentiment["article_count"]
    )

    return effective_sentiment[
        [
            "effective_trading_date",
            "sentiment_calendar_start",
            "sentiment_calendar_end",
            "source_calendar_day_count",
            "article_count",
            "sentiment_index",
            "sentiment_index_z",
            "positive_article_count",
            "negative_article_count",
            "neutral_article_count",
            "log_article_count",
        ]
    ]


def merge_vnindex_daily_with_sentiment(
    vnindex_daily_df: pd.DataFrame,
    sentiment_daily_df: pd.DataFrame,
) -> pd.DataFrame:
    vnindex_daily = prepare_vnindex_daily(vnindex_daily_df)
    effective_sentiment = prepare_effective_sentiment(
        sentiment_daily_df,
        vnindex_daily[DATE_COLUMN],
    )

    merged_df = effective_sentiment.merge(
        vnindex_daily,
        left_on="effective_trading_date",
        right_on=DATE_COLUMN,
        how="inner",
    )
    merged_df = merged_df.drop(columns=[DATE_COLUMN])
    merged_df = merged_df.rename(columns={"effective_trading_date": DATE_COLUMN})

    ordered_columns = [
        DATE_COLUMN,
        "sentiment_calendar_start",
        "sentiment_calendar_end",
        "source_calendar_day_count",
        "article_count",
        "sentiment_index",
        "sentiment_index_z",
        "positive_article_count",
        "negative_article_count",
        "neutral_article_count",
        "log_article_count",
        OPEN_COLUMN,
        HIGH_COLUMN,
        LOW_COLUMN,
        CLOSE_COLUMN,
        VOLUME_COLUMN,
        VALUE_COLUMN,
        "daily_return",
        "future_ret_1d",
        "future_ret_5d",
        "future_ret_20d",
        "return_lag_1d",
        "volatility_20d",
        "log_vol_total",
        "log_val_total",
    ]
    merged_df = merged_df[ordered_columns]
    return merged_df.sort_values(DATE_COLUMN, kind="mergesort").reset_index(drop=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    vnindex_daily_df = pd.read_csv(VNINDEX_DAILY_PATH, encoding="utf-8-sig")
    sentiment_daily_df = pd.read_parquet(SENTIMENT_DAILY_PATH)

    merged_df = merge_vnindex_daily_with_sentiment(
        vnindex_daily_df,
        sentiment_daily_df,
    )

    OUTPUT_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(OUTPUT_PARQUET_PATH, index=False)
    merged_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

    print("VN-Index daily input:", VNINDEX_DAILY_PATH)
    print("Sentiment daily input:", SENTIMENT_DAILY_PATH)
    print("Output parquet:", OUTPUT_PARQUET_PATH)
    print("Output csv:", OUTPUT_CSV_PATH)
    print("VN-Index daily rows:", len(prepare_vnindex_daily(vnindex_daily_df)))
    print("Sentiment daily rows:", len(sentiment_daily_df))
    print("Merged rows:", len(merged_df))
    print(merged_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
