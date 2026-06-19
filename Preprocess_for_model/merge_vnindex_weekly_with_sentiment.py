from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "Dataset"
VNINDEX_WEEKLY_PATH = DATASET_DIR / "vnindex_weekly_return.parquet"
SENTIMENT_WEEKLY_PATH = DATASET_DIR / "market_sentiment_index_weekly.parquet"
OUTPUT_PARQUET_PATH = DATASET_DIR / "vnindex_weekly_sentiment_merged.parquet"
OUTPUT_CSV_PATH = DATASET_DIR / "vnindex_weekly_sentiment_merged.csv"


def prepare_week_key(df: pd.DataFrame, column: str = "week_end") -> pd.DataFrame:
    out = df.copy()
    out[column] = pd.to_datetime(out[column], errors="coerce").dt.normalize()
    out = out.loc[out[column].notna()].copy()
    return out


def merge_vnindex_weekly_with_sentiment(
    vnindex_weekly_df: pd.DataFrame,
    sentiment_weekly_df: pd.DataFrame,
) -> pd.DataFrame:
    vnindex_weekly_df = prepare_week_key(vnindex_weekly_df)
    sentiment_weekly_df = prepare_week_key(sentiment_weekly_df)

    merged_df = sentiment_weekly_df.merge(
        vnindex_weekly_df,
        on="week_end",
        how="inner",
        suffixes=("_sentiment", "_vnindex"),
    )

    if "week_start_sentiment" in merged_df.columns:
        merged_df = merged_df.rename(columns={"week_start_sentiment": "week_start"})
    if "week_start_vnindex" in merged_df.columns:
        merged_df = merged_df.drop(columns=["week_start_vnindex"])

    merged_df["log_article_count"] = np.log1p(merged_df["article_count"])

    ordered_columns = [
        "week_start",
        "week_end",
        "article_count",
        "sentiment_index",
        "sentiment_index_z",
        "positive_article_count",
        "negative_article_count",
        "neutral_article_count",
        "log_article_count",
        "first_trading_date",
        "last_trading_date",
        "trading_day_count",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "vol_total",
        "val_total",
        "weekly_return",
        "future_ret_1w",
        "future_ret_4w",
        "return_lag_1w",
        "volatility_12w",
        "log_vol_total",
    ]
    existing_columns = [column for column in ordered_columns if column in merged_df.columns]
    remaining_columns = [column for column in merged_df.columns if column not in existing_columns]
    merged_df = merged_df[existing_columns + remaining_columns]

    return merged_df.sort_values("week_end", kind="mergesort").reset_index(drop=True)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    vnindex_weekly_df = pd.read_parquet(VNINDEX_WEEKLY_PATH)
    sentiment_weekly_df = pd.read_parquet(SENTIMENT_WEEKLY_PATH)
    merged_df = merge_vnindex_weekly_with_sentiment(
        vnindex_weekly_df,
        sentiment_weekly_df,
    )

    OUTPUT_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_parquet(OUTPUT_PARQUET_PATH, index=False)
    merged_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")

    print("VN-Index weekly input:", VNINDEX_WEEKLY_PATH)
    print("Sentiment weekly input:", SENTIMENT_WEEKLY_PATH)
    print("Output parquet:", OUTPUT_PARQUET_PATH)
    print("Output csv:", OUTPUT_CSV_PATH)
    print("VN-Index weekly rows:", len(vnindex_weekly_df))
    print("Sentiment weekly rows:", len(sentiment_weekly_df))
    print("Merged rows:", len(merged_df))
    print(merged_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
