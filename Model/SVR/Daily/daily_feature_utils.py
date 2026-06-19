from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


TARGET_COLUMNS = ["future_ret_1d", "future_ret_5d", "future_ret_10d", "future_ret_20d"]

SENTIMENT_FEATURE_COLUMNS = [
    "sentiment_index_z",
    "log_article_count",
    "positive_ratio",
    "negative_ratio",
    "sentiment_lag_1d",
    "sentiment_lag_2d",
    "sentiment_lag_5d",
    "sentiment_ma_5d",
    "sentiment_ma_20d",
    "sentiment_z_shock_1d",
    "sentiment_shock_vs_ma5",
    "extreme_negative_sentiment",
    "negative_ratio_change_1d",
    "news_attention_shock",
]

MARKET_FEATURE_COLUMNS = [
    "daily_return",
    "return_lag_1d",
    "return_lag_5d",
    "return_lag_20d",
    "return_ma_5d",
    "return_ma_20d",
    "return_vol_5d",
    "return_vol_20d",
    "log_vol_total",
    "volume_shock_20d",
    "high_low_range_pct",
    "range_shock_20d",
    "return_shock_z_20d",
    "large_down_day",
]

FORBIDDEN_FEATURE_COLUMNS = set(TARGET_COLUMNS)


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "close_price" in out.columns:
        out["future_ret_10d"] = out["close_price"].shift(-10) / out["close_price"] - 1

    article_columns = {
        "positive_article_count",
        "negative_article_count",
        "neutral_article_count",
        "article_count",
    }
    if article_columns.issubset(out.columns):
        out["positive_ratio"] = safe_divide(out["positive_article_count"], out["article_count"])
        out["negative_ratio"] = safe_divide(out["negative_article_count"], out["article_count"])

    if "sentiment_index" in out.columns:
        out["sentiment_lag_1d"] = out["sentiment_index"].shift(1)
        out["sentiment_lag_2d"] = out["sentiment_index"].shift(2)
        out["sentiment_lag_5d"] = out["sentiment_index"].shift(5)
        out["sentiment_ma_5d"] = out["sentiment_index"].rolling(window=5, min_periods=5).mean()
        out["sentiment_ma_20d"] = out["sentiment_index"].rolling(window=20, min_periods=20).mean()

    if {"sentiment_index", "sentiment_index_z"}.issubset(out.columns):
        out["sentiment_z_shock_1d"] = out["sentiment_index_z"] - out["sentiment_index_z"].shift(1)
        out["sentiment_shock_vs_ma5"] = (
            out["sentiment_index"]
            - out["sentiment_index"].rolling(window=5, min_periods=5).mean()
        )
        out["extreme_negative_sentiment"] = (out["sentiment_index_z"] < -1.5).astype(int)

    if {"negative_ratio", "log_article_count"}.issubset(out.columns):
        out["negative_ratio_change_1d"] = out["negative_ratio"] - out["negative_ratio"].shift(1)
        out["news_attention_shock"] = (
            out["log_article_count"]
            - out["log_article_count"].rolling(window=5, min_periods=5).mean()
        )

    if "daily_return" in out.columns:
        out["return_lag_5d"] = out["daily_return"].shift(5)
        out["return_lag_20d"] = out["daily_return"].shift(20)
        out["return_ma_5d"] = out["daily_return"].rolling(window=5, min_periods=5).mean()
        out["return_ma_20d"] = out["daily_return"].rolling(window=20, min_periods=20).mean()
        out["return_vol_5d"] = out["daily_return"].rolling(window=5, min_periods=5).std()
        out["return_vol_20d"] = out["daily_return"].rolling(window=20, min_periods=20).std()

    if {"high_price", "low_price", "close_price"}.issubset(out.columns):
        out["high_low_range_pct"] = safe_divide(
            out["high_price"] - out["low_price"], out["close_price"]
        )

    if {"log_vol_total", "high_low_range_pct", "daily_return", "return_vol_20d"}.issubset(out.columns):
        out["volume_shock_20d"] = (
            out["log_vol_total"]
            - out["log_vol_total"].rolling(window=20, min_periods=10).mean()
        )
        out["range_shock_20d"] = (
            out["high_low_range_pct"]
            - out["high_low_range_pct"].rolling(window=20, min_periods=10).mean()
        )
        out["return_shock_z_20d"] = safe_divide(out["daily_return"], out["return_vol_20d"])
        out["large_down_day"] = (out["daily_return"] < -0.03).astype(int)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def read_daily_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Daily sentiment data not found: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "date" not in df.columns:
        raise ValueError("Expected a date column in the daily sentiment data.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return add_derived_features(df)


def build_feature_columns(
    df: pd.DataFrame,
    feature_set: str,
    extra_features: Sequence[str] | None = None,
    drop_features: Sequence[str] | None = None,
) -> list[str]:
    feature_set = feature_set.lower().strip()
    extras = list(extra_features or [])
    drops = set(drop_features or [])

    if feature_set == "sentiment":
        features = SENTIMENT_FEATURE_COLUMNS
    elif feature_set == "market":
        features = MARKET_FEATURE_COLUMNS
    elif feature_set == "combined":
        features = SENTIMENT_FEATURE_COLUMNS + MARKET_FEATURE_COLUMNS
    else:
        raise ValueError("feature_set must be sentiment, market, or combined.")

    features = list(dict.fromkeys([*features, *extras]))
    features = [feature for feature in features if feature not in drops]

    leaked = sorted(FORBIDDEN_FEATURE_COLUMNS.intersection(features))
    if leaked:
        raise ValueError(f"Forbidden target columns selected as features: {leaked}")

    missing = [feature for feature in features if feature not in df.columns]
    if missing:
        raise ValueError(f"Missing requested feature columns: {missing}")

    non_numeric = [
        feature for feature in features if not pd.api.types.is_numeric_dtype(df[feature])
    ]
    if non_numeric:
        raise ValueError(f"Feature columns must be numeric: {non_numeric}")
    return features


def load_feature_data(
    path: str | Path,
    target_column: str,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    if target_column not in TARGET_COLUMNS:
        raise ValueError(f"target_column must be one of {TARGET_COLUMNS}")

    df = read_daily_data(path)
    if target_column not in df.columns:
        raise ValueError(f"Target column does not exist: {target_column}")

    required = list(feature_columns) + [target_column, "date"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after feature engineering: {missing}")

    return df.dropna(subset=required).reset_index(drop=True)
