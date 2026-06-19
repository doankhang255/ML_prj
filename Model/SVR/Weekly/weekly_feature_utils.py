from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


TARGET_COLUMNS = ["future_ret_1w", "future_ret_4w"]

SENTIMENT_FEATURE_COLUMNS = [
    "sentiment_index_z",
    "log_article_count",
    "positive_ratio",
    "negative_ratio",
    "sentiment_lag_1w",
    "sentiment_lag_4w",
    "sentiment_ma_8w",
    "sentiment_z_shock_1w",
    "sentiment_shock_vs_ma4",
    "extreme_negative_sentiment",
    "negative_attention",
    "negative_ratio_change_1w",
    "news_attention_shock",
]

MARKET_FEATURE_COLUMNS = [
    "weekly_return",
    "return_lag_1w",
    "return_lag_4w",
    "return_ma_4w",
    "return_ma_12w",
    "return_vol_4w",
    "return_vol_12w",
    "log_vol_total",
    "volume_shock_12w",
    "range_shock_12w",
    "return_shock_z_12w",
    "large_down_week",
    "negative_sentiment_market_stress",
    "negative_sentiment_volume_shock",
]

FORBIDDEN_FEATURE_COLUMNS = set(TARGET_COLUMNS)
DATE_COLUMNS = ["week_start", "week_end", "first_trading_date", "last_trading_date"]


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    article_columns = {
        "positive_article_count",
        "negative_article_count",
        "neutral_article_count",
        "article_count",
    }
    if article_columns.issubset(out.columns):
        out["positive_ratio"] = safe_divide(out["positive_article_count"], out["article_count"])
        out["negative_ratio"] = safe_divide(out["negative_article_count"], out["article_count"])
        out["neutral_ratio"] = safe_divide(out["neutral_article_count"], out["article_count"])

    if "sentiment_index" in out.columns:
        out["sentiment_lag_1w"] = out["sentiment_index"].shift(1)
        out["sentiment_lag_4w"] = out["sentiment_index"].shift(4)
        out["sentiment_ma_8w"] = out["sentiment_index"].rolling(window=8, min_periods=8).mean()

    if "weekly_return" in out.columns:
        out["return_lag_4w"] = out["weekly_return"].shift(4)
        out["return_ma_4w"] = out["weekly_return"].rolling(window=4, min_periods=4).mean()
        out["return_ma_12w"] = out["weekly_return"].rolling(window=12, min_periods=12).mean()
        out["return_vol_4w"] = out["weekly_return"].rolling(window=4, min_periods=4).std()
        out["return_vol_12w"] = out["weekly_return"].rolling(window=12, min_periods=12).std()

    if {"high_price", "low_price", "close_price"}.issubset(out.columns):
        out["high_low_range_pct"] = safe_divide(
            out["high_price"] - out["low_price"],
            out["close_price"],
        )

    if {"sentiment_index", "sentiment_index_z"}.issubset(out.columns):
        out["sentiment_z_shock_1w"] = out["sentiment_index_z"] - out["sentiment_index_z"].shift(1)
        sentiment_ma4 = out["sentiment_index"].rolling(window=4, min_periods=4).mean()
        sentiment_std8 = out["sentiment_index"].rolling(window=8, min_periods=4).std()
        out["sentiment_shock_vs_ma4"] = safe_divide(
            out["sentiment_index"] - sentiment_ma4,
            sentiment_std8,
        )
        out["extreme_negative_sentiment"] = (out["sentiment_index_z"] < -1.5).astype(int)

    if "negative_ratio" in out.columns:
        out["negative_ratio_change_1w"] = out["negative_ratio"] - out["negative_ratio"].shift(1)

    if "log_article_count" in out.columns:
        out["news_attention_ma12"] = out["log_article_count"].rolling(
            window=12, min_periods=6
        ).mean()
        out["news_attention_std12"] = out["log_article_count"].rolling(
            window=12, min_periods=6
        ).std()
        out["news_attention_shock"] = safe_divide(
            out["log_article_count"] - out["news_attention_ma12"],
            out["news_attention_std12"],
        )

    if {
        "negative_ratio_change_1w",
        "news_attention_shock",
    }.issubset(out.columns):
        out["negative_attention"] = (
            out["negative_ratio_change_1w"].clip(lower=0)
            * out["news_attention_shock"].clip(lower=0)
        )

    if {"log_vol_total", "high_low_range_pct", "weekly_return", "return_vol_12w"}.issubset(out.columns):
        volume_ma12 = out["log_vol_total"].rolling(window=12, min_periods=6).mean()
        volume_std12 = out["log_vol_total"].rolling(window=12, min_periods=6).std()
        range_ma12 = out["high_low_range_pct"].rolling(window=12, min_periods=6).mean()
        range_std12 = out["high_low_range_pct"].rolling(window=12, min_periods=6).std()
        return_ma12 = out["weekly_return"].rolling(window=12, min_periods=6).mean()
        out["volume_shock_12w"] = safe_divide(out["log_vol_total"] - volume_ma12, volume_std12)
        out["range_shock_12w"] = safe_divide(out["high_low_range_pct"] - range_ma12, range_std12)
        out["return_shock_z_12w"] = safe_divide(
            (out["weekly_return"] - return_ma12).abs(),
            out["return_vol_12w"],
        )
        out["large_down_week"] = (out["weekly_return"] < -0.03).astype(int)

    interaction_columns = {
        "negative_attention",
        "high_low_range_pct",
        "volume_shock_12w",
    }
    if interaction_columns.issubset(out.columns):
        out["negative_sentiment_market_stress"] = (
            out["negative_attention"] * out["high_low_range_pct"]
        )
        out["negative_sentiment_volume_shock"] = (
            out["negative_attention"] * out["volume_shock_12w"]
        )

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def read_weekly_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Weekly sentiment data not found: {path}")

    df = pd.read_parquet(path)
    for column in DATE_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    if "week_start" not in df.columns:
        raise ValueError("Expected a week_start column in the weekly sentiment data.")
    df = df.sort_values("week_start").reset_index(drop=True)
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

    df = read_weekly_data(path)
    if target_column not in df.columns:
        raise ValueError(f"Target column does not exist: {target_column}")

    required = list(feature_columns) + [target_column, "week_start", "week_end"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after feature engineering: {missing}")

    return df.dropna(subset=required).reset_index(drop=True)
