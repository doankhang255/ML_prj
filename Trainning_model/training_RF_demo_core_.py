from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import certifi
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


# This file implements a full baseline pipeline:
# 1) load raw OHLCV from Mongo
# 2) compute features per ticker
# 3) generate labels per ticker
# 4) merge all ticker events into one model frame
# 5) split train/test by time
# 6) train a RandomForest classifier
# 7) backtest only the trades that pass a probability threshold
#
# Important design choice:
# - features are computed separately for each ticker
# - labels are generated separately for each ticker
# - only after that do we stack all rows together into one training table
# This avoids mixing rolling windows across different tickers.


FINAL_FEATURE_COLS = [
    "ma10_ma50_ratio",
    "close_ma10_ratio",
    "dist_ma10_ma50_pct",
    "ma10_slope_3d",
    "ma50_slope_3d",
    "golden_cross",
    "return_3d",
    "return_5d",
    "rsi",
    "macd_hist",
    "volume_ratio_20",
    "volume_change_pct",
    "atr_pct",
    "rolling_std_10",
    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "close_position",
]

RAW_COLUMNS = [
    "ticker",
    "trading_date",
    "timeframe",
    "open",
    "high",
    "low",
    "close",
    "volume",
]


def parse_args() -> argparse.Namespace:
    """Read CLI arguments.

    Debug note:
    - If the script behaves differently from what you expect, check the
      effective values here first because most pipeline behavior is controlled
      by these arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train and backtest a RandomForest model from MongoDB OHLCV data."
    )
    parser.add_argument("--mongo-uri", default=os.getenv("MONGO_URI"))
    parser.add_argument("--mongo-db", default=os.getenv("MONGO_DB", "stock_ml"))
    parser.add_argument(
        "--mongo-collection",
        default=os.getenv("MONGO_COLLECTION", "raw_ohlcv_daily"),
    )
    parser.add_argument("--timeframe", default="1D")
    parser.add_argument(
        "--tickers",
        default="ALL",
        help="Comma-separated tickers. Use ALL to load the whole universe.",
    )
    parser.add_argument(
        "--candidate-mode",
        choices=["crossover_only", "all_rows"],
        default="crossover_only",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--target-return", type=float, default=0.03)
    parser.add_argument("--stop-loss", type=float, default=-0.02)
    parser.add_argument("--fee-pct", type=float, default=0.003)
    parser.add_argument("--prob-threshold", type=float, default=0.55)
    parser.add_argument("--capital-per-trade", type=float, default=10_000_000)
    parser.add_argument(
        "--ambiguity-mode",
        choices=["conservative", "optimistic"],
        default="conservative",
    )
    parser.add_argument("--output-dir", default="outputs/rf_demo_core")
    parser.add_argument("--export-excel", action="store_true")
    return parser.parse_args()


def parse_ticker_list(tickers_arg: str) -> list[str] | None:
    """Normalize the ticker input from CLI.

    Returns:
    - None when user requests ALL tickers
    - a clean uppercase list otherwise
    """
    if tickers_arg.strip().upper() == "ALL":
        return None
    tickers = [ticker.strip().upper() for ticker in tickers_arg.split(",") if ticker.strip()]
    return tickers or None


def load_raw_from_mongo(
    mongo_uri: str,
    mongo_db: str,
    mongo_collection: str,
    timeframe: str,
    tickers: list[str] | None,
) -> pd.DataFrame:
    """Load raw OHLCV rows from MongoDB and coerce types.

    Why this exists:
    - Mongo can store dates/numbers in inconsistent formats
    - training code is simpler if this function guarantees a clean DataFrame

    Debug note:
    - If row count is lower than expected, inspect query filters, type coercion,
      and dropped rows caused by missing numeric/date values.
    """
    if not mongo_uri:
        raise ValueError("Missing MongoDB URI. Please set MONGO_URI or pass --mongo-uri.")

    client = MongoClient(mongo_uri, tls=True, tlsCAFile=certifi.where())
    try:
        collection = client[mongo_db][mongo_collection]
        query: dict[str, object] = {"timeframe": timeframe}
        if tickers:
            query["ticker"] = {"$in": tickers}

        projection = {column: 1 for column in RAW_COLUMNS}
        projection["_id"] = 0

        rows = list(
            collection.find(query, projection=projection).sort(
                [("ticker", 1), ("trading_date", 1)]
            )
        )
    finally:
        client.close()

    if not rows:
        raise ValueError("Mongo query returned no rows. Check URI, collection, timeframe, or tickers.")

    df = pd.DataFrame(rows)
    missing = sorted(set(RAW_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Mongo data is missing required columns: {missing}")

    df["trading_date"] = pd.to_datetime(df["trading_date"], errors="coerce")
    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["ticker", "trading_date", "open", "high", "low", "close", "volume"])
    return df.sort_values(["ticker", "trading_date"]).reset_index(drop=True)


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Compute Wilder RSI from close prices.

    Wilder RSI works in 2 stages:
    - seed the first average gain/loss with a simple mean over `window` bars
    - from that point onward, update the averages with Wilder smoothing

    This version is usually closer to the RSI shown on charting platforms than
    a plain rolling-mean RSI.
    """
    delta = close.diff()
    gain = delta.clip(lower=0) #hàm clip mọi giá trị < 0 thì = 0
    loss = -delta.clip(upper=0) # hàm clip mọi giá trị > 0 thì = 0

    rsi = pd.Series(np.nan, index=close.index, dtype=float)
    if len(close) <= window:
        return rsi

    avg_gain = pd.Series(np.nan, index=close.index, dtype=float)
    avg_loss = pd.Series(np.nan, index=close.index, dtype=float)

    # Seed Wilder smoothing from the first `window` close-to-close moves.
    avg_gain.iloc[window] = gain.iloc[1 : window + 1].mean() #iloc truy cập theo vị trí sô nguyên 
    avg_loss.iloc[window] = loss.iloc[1 : window + 1].mean()

    for i in range(window + 1, len(close)):
        avg_gain.iloc[i] = ((avg_gain.iloc[i - 1] * (window - 1)) + gain.iloc[i]) / window
        avg_loss.iloc[i] = ((avg_loss.iloc[i - 1] * (window - 1)) + loss.iloc[i]) / window

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # Handle edge cases explicitly so flat/up-only/down-only streaks stay interpretable.
    rsi[(avg_loss == 0) & (avg_gain > 0)] = 100.0
    rsi[(avg_gain == 0) & (avg_loss > 0)] = 0.0
    rsi[(avg_gain == 0) & (avg_loss == 0)] = 50.0
    return rsi


def compute_features_for_ticker(group: pd.DataFrame) -> pd.DataFrame:
    """Compute all model features for one ticker only.

    Very important:
    - This function must receive only one ticker at a time.
    - Rolling indicators such as MA/ATR/RSI must never be computed on a table
      that already mixes multiple tickers together.
    """
    group = group.sort_values("trading_date").copy()

    close = group["close"]
    open_ = group["open"]
    high = group["high"]
    low = group["low"]
    volume = group["volume"]

    ma10 = close.rolling(10, min_periods=10).mean()
    ma50 = close.rolling(50, min_periods=50).mean()
    prev_ma10 = ma10.shift(1)
    prev_ma50 = ma50.shift(1)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False).mean()

    prev_close = close.shift(1) 
    # True range is the building block for ATR. On daily data it captures both
    # intraday range and overnight gaps against the previous close.
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(14, min_periods=14).mean()

    # Use candle range as denominator for candle-shape ratios.
    # Replace 0 by NaN to avoid dividing by zero on flat candles.
    candle_range = (high - low).replace(0, np.nan)
    upper_wick = high - pd.concat([open_, close], axis=1).max(axis=1)
    lower_wick = pd.concat([open_, close], axis=1).min(axis=1) - low

    # Keep ma10/ma50 in the frame for inspection/debugging even though the
    # model itself mainly uses normalized ratios instead of raw MA values.
    group["ma10"] = ma10
    group["ma50"] = ma50
    group["ma10_ma50_ratio"] = ma10 / ma50
    group["close_ma10_ratio"] = close / ma10
    group["dist_ma10_ma50_pct"] = (ma10 - ma50) / ma50
    group["ma10_slope_3d"] = ma10.pct_change(3)
    group["ma50_slope_3d"] = ma50.pct_change(3)
    group["golden_cross"] = ((prev_ma10 <= prev_ma50) & (ma10 > ma50)).astype(int)
    group["return_3d"] = close.pct_change(3)
    group["return_5d"] = close.pct_change(5)
    group["rsi"] = compute_rsi(close, window=14)
    group["macd_hist"] = macd - macd_signal
    group["volume_ratio_20"] = volume / volume.rolling(20, min_periods=20).mean()
    group["volume_change_pct"] = volume.pct_change(1)
    group["atr_pct"] = atr / close
    group["rolling_std_10"] = close.pct_change().rolling(10, min_periods=10).std()
    group["body_ratio"] = (close - open_).abs() / candle_range
    group["upper_wick_ratio"] = upper_wick / candle_range
    group["lower_wick_ratio"] = lower_wick / candle_range
    group["close_position"] = (close - low) / candle_range
    return group


def apply_per_ticker(
    df: pd.DataFrame,
    func,
    **kwargs,
) -> pd.DataFrame:
    """Apply a transformation function to each ticker separately, then stack.

    This is the heart of the 'compute separately, merge later' approach.
    """
    frames = []
    for _, group in df.groupby("ticker", sort=False):
        frames.append(func(group.copy(), **kwargs))
    return pd.concat(frames, ignore_index=True)


def label_trades_for_ticker(
    group: pd.DataFrame,
    horizon: int,
    target_return: float,
    stop_loss: float,
    fee_pct: float,
    ambiguity_mode: str,
) -> pd.DataFrame:
    """Generate trade path and label columns for one ticker.

    Labeling logic:
    - signal happens on row t
    - entry is at open of row t+1
    - we inspect the next `horizon` bars
    - trade exits on:
      1) target
      2) stop
      3) timeout at close of last bar in the horizon

    Current label definition:
    - buy_label = 1 if the simulated trade ends with net_return > 0 after fees
    - buy_label = 0 otherwise

    This means:
    - target_hit and buy_label are NOT the same concept
    - a trade can timeout without hitting target and still end profitable
    """
    group = group.sort_values("trading_date").reset_index(drop=True).copy()

    signal_dates: list[pd.Timestamp | float] = []
    entry_dates: list[pd.Timestamp | float] = []
    exit_dates: list[pd.Timestamp | float] = []
    entry_prices: list[float] = []
    exit_prices: list[float] = []
    realized_return_gross: list[float] = []
    realized_return_net: list[float] = []
    future_max_returns: list[float] = []
    future_min_returns: list[float] = []
    bars_held: list[float] = []
    exit_reasons: list[str | float] = []
    labels: list[float] = []
    target_hits: list[float] = []

    opens = group["open"].to_numpy(dtype=float)
    highs = group["high"].to_numpy(dtype=float)
    lows = group["low"].to_numpy(dtype=float)
    closes = group["close"].to_numpy(dtype=float)
    dates = group["trading_date"].to_numpy()

    # target_return is interpreted as desired NET return after fees.
    # So to hit a net target of 3%, the gross move must exceed 3% + fee.
    required_gross_return = target_return + fee_pct

    for signal_idx in range(len(group)):
        # We cannot label the last rows if there are not enough future bars.
        if signal_idx + horizon >= len(group):
            signal_dates.append(np.nan)
            entry_dates.append(np.nan)
            exit_dates.append(np.nan)
            entry_prices.append(np.nan)
            exit_prices.append(np.nan)
            realized_return_gross.append(np.nan)
            realized_return_net.append(np.nan)
            future_max_returns.append(np.nan)
            future_min_returns.append(np.nan)
            bars_held.append(np.nan)
            exit_reasons.append(np.nan)
            labels.append(np.nan)
            target_hits.append(np.nan)
            continue

        # Entry always happens on the next bar open, not on the signal bar.
        entry_idx = signal_idx + 1
        last_eval_idx = signal_idx + horizon
        entry_price = opens[entry_idx]
        target_price = entry_price * (1 + required_gross_return)
        stop_price = entry_price * (1 + stop_loss)

        future_high = highs[entry_idx : last_eval_idx + 1]
        future_low = lows[entry_idx : last_eval_idx + 1]

        exit_idx = last_eval_idx
        exit_price = closes[last_eval_idx]
        exit_reason = "timeout"
        hit_target = 0

        for future_idx in range(entry_idx, last_eval_idx + 1):
            target_touched = highs[future_idx] >= target_price
            stop_touched = lows[future_idx] <= stop_price

            # On daily data, if both target and stop are touched inside the same
            # bar, we do not know which came first. ambiguity_mode controls the
            # assumption used in that situation.
            if target_touched and stop_touched:
                exit_idx = future_idx
                if ambiguity_mode == "optimistic":
                    exit_price = target_price
                    exit_reason = "ambiguous_target"
                    hit_target = 1
                else:
                    exit_price = stop_price
                    exit_reason = "ambiguous_stop"
                break

            if target_touched:
                exit_idx = future_idx
                exit_price = target_price
                exit_reason = "target"
                hit_target = 1
                break

            if stop_touched:
                exit_idx = future_idx
                exit_price = stop_price
                exit_reason = "stop"
                break

        # Gross return = raw price change from entry to exit.
        # Net return = gross return minus an estimated total round-trip fee.
        gross_return = (exit_price / entry_price) - 1
        net_return = gross_return - fee_pct

        signal_dates.append(pd.Timestamp(dates[signal_idx]))
        entry_dates.append(pd.Timestamp(dates[entry_idx]))
        exit_dates.append(pd.Timestamp(dates[exit_idx]))
        entry_prices.append(entry_price)
        exit_prices.append(exit_price)
        realized_return_gross.append(gross_return)
        realized_return_net.append(net_return)
        future_max_returns.append((future_high.max() / entry_price) - 1)
        future_min_returns.append((future_low.min() / entry_price) - 1)

        # bars_held currently counts from signal bar to exit bar.
        # It is useful for relative holding-length comparison, but it is NOT
        # exactly the same as "days in position" because entry occurs at t+1.
        bars_held.append(exit_idx - signal_idx)
        exit_reasons.append(exit_reason)

        # Current supervised target:
        # profitable after fee => class 1, else class 0.
        labels.append(int(net_return > 0))
        target_hits.append(hit_target)

    group["signal_date"] = signal_dates
    group["entry_date"] = entry_dates
    group["exit_date"] = exit_dates
    group["entry_price"] = entry_prices
    group["exit_price"] = exit_prices
    group["realized_return_gross"] = realized_return_gross
    group["realized_return_net"] = realized_return_net
    group["future_max_return_h"] = future_max_returns
    group["future_min_return_h"] = future_min_returns
    group["bars_held"] = bars_held
    group["exit_reason"] = exit_reasons
    group["buy_label"] = labels
    group["target_hit"] = target_hits
    return group


def build_model_frame(raw_df: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, list[str]]:
    """Build the final supervised-learning table used by the model.

    Steps:
    - feature engineering per ticker
    - labeling per ticker
    - optional candidate filtering
    - NaN cleanup
    - constant-feature removal
    """
    featured = apply_per_ticker(raw_df, compute_features_for_ticker)

    labeled = apply_per_ticker(
        featured,
        label_trades_for_ticker,
        horizon=args.horizon,
        target_return=args.target_return,
        stop_loss=args.stop_loss,
        fee_pct=args.fee_pct,
        ambiguity_mode=args.ambiguity_mode,
    )

    labeled = labeled.replace([np.inf, -np.inf], np.nan)

    # In crossover_only mode, the model is not asked to find all entries.
    # Instead, it acts as a filter on top of MA crossover signals.
    model_df = labeled[labeled["golden_cross"] == 1].copy() if args.candidate_mode == "crossover_only" else labeled.copy()

    required_columns = FINAL_FEATURE_COLS + [
        "buy_label",
        "ticker",
        "trading_date",
        "signal_date",
        "entry_date",
        "exit_date",
        "entry_price",
        "exit_price",
        "realized_return_net",
        "realized_return_gross",
        "exit_reason",
        "bars_held",
        "target_hit",
        "future_max_return_h",
        "future_min_return_h",
    ]
    model_df = model_df.dropna(subset=required_columns).copy()
    model_df["buy_label"] = model_df["buy_label"].astype(int)
    model_df["target_hit"] = model_df["target_hit"].astype(int)

    # Remove columns that are constant in the final training table.
    # Example: when candidate_mode == crossover_only, golden_cross is always 1
    # so it contains no information for the model.
    usable_feature_cols = [
        column
        for column in FINAL_FEATURE_COLS
        if column in model_df.columns and model_df[column].nunique(dropna=True) > 1
    ]
    if not usable_feature_cols:
        raise ValueError("No usable feature columns remain after filtering and constant-column removal.")

    return model_df, usable_feature_cols


def split_by_time(model_df: pd.DataFrame, test_size: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split the final dataset chronologically.

    We do not use random shuffle here because trading problems are highly
    time-dependent. Random splitting would leak future patterns into train.
    """
    unique_dates = sorted(model_df["signal_date"].dropna().unique())
    if len(unique_dates) < 2:
        raise ValueError("Need at least two unique signal dates to create a train/test split.")

    split_idx = int(len(unique_dates) * (1 - test_size))
    split_idx = min(max(split_idx, 1), len(unique_dates) - 1)
    split_date = pd.Timestamp(unique_dates[split_idx])

    train_df = model_df[model_df["signal_date"] < split_date].copy()
    test_df = model_df[model_df["signal_date"] >= split_date].copy()
    if train_df.empty or test_df.empty:
        raise ValueError("Chronological split produced an empty train or test set.")

    return train_df, test_df, split_date


def train_random_forest(train_df: pd.DataFrame, feature_cols: list[str]) -> RandomForestClassifier:
    """Train the RandomForest baseline model."""
    if train_df["buy_label"].nunique() < 2:
        raise ValueError("Training set has only one class. The model cannot be trained.")

    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=6,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=1,
    )
    model.fit(train_df[feature_cols], train_df["buy_label"])
    return model


def evaluate_model(
    model: RandomForestClassifier,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    prob_threshold: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Score the test set and convert probabilities into class predictions.

    Important distinction:
    - pred_proba = model confidence that the trade belongs to class 1
    - pred_label = hard decision after applying prob_threshold
    """
    result_df = test_df.copy()
    result_df["pred_proba"] = model.predict_proba(test_df[feature_cols])[:, 1]

    # prob_threshold controls how strict the model is when saying "buy".
    # Higher threshold -> fewer but more selective trades.
    result_df["pred_label"] = (result_df["pred_proba"] >= prob_threshold).astype(int)

    y_true = result_df["buy_label"].astype(int)
    y_pred = result_df["pred_label"].astype(int)

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["loss", "win"],
        zero_division=0,
        output_dict=True,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "threshold": prob_threshold,
        "test_samples": int(len(result_df)),
        "positive_predictions": int(y_pred.sum()),
        "classification_report": report,
    }
    return result_df, metrics


def run_backtest(
    predictions_df: pd.DataFrame,
    prob_threshold: float,
    capital_per_trade: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Run a simple test-period backtest on model predictions.

    Current backtest rules:
    - take only rows whose predicted probability exceeds prob_threshold
    - allow multiple tickers at the same time
    - forbid overlapping positions on the same ticker
    - use a fixed capital allocation per trade
    """
    sorted_df = predictions_df.sort_values(["entry_date", "ticker"]).copy()

    active_until_by_ticker: dict[str, pd.Timestamp] = {}
    selected_rows = []
    skipped_overlap = 0

    for row in sorted_df.itertuples(index=False):
        # This is where probability becomes an actual trading decision.
        if row.pred_proba < prob_threshold:
            continue

        # Do not open a new position on the same ticker while a previous one
        # is still active in the simplified backtest.
        active_until = active_until_by_ticker.get(row.ticker)
        if active_until is not None and row.entry_date <= active_until:
            skipped_overlap += 1
            continue

        selected_rows.append(row._asdict())
        active_until_by_ticker[row.ticker] = row.exit_date

    selected_df = pd.DataFrame(selected_rows)
    if selected_df.empty:
        backtest_summary = {
            "selected_trades": 0,
            "skipped_due_overlap": skipped_overlap,
            "win_rate": 0.0,
            "avg_return_net": 0.0,
            "median_return_net": 0.0,
            "total_return_sum": 0.0,
            "total_pnl_vnd": 0.0,
            "avg_bars_held": 0.0,
            "target_hit_rate": 0.0,
        }
        return selected_df, backtest_summary

    selected_df = selected_df.sort_values(["exit_date", "ticker"]).reset_index(drop=True)
    selected_df["pnl_vnd"] = selected_df["realized_return_net"] * capital_per_trade
    selected_df["cumulative_pnl_vnd"] = selected_df["pnl_vnd"].cumsum()

    backtest_summary = {
        "selected_trades": int(len(selected_df)),
        "skipped_due_overlap": int(skipped_overlap),
        "win_rate": float((selected_df["realized_return_net"] > 0).mean()),
        "avg_return_net": float(selected_df["realized_return_net"].mean()),
        "median_return_net": float(selected_df["realized_return_net"].median()),
        "total_return_sum": float(selected_df["realized_return_net"].sum()),
        "total_pnl_vnd": float(selected_df["pnl_vnd"].sum()),
        "avg_bars_held": float(selected_df["bars_held"].mean()),
        "target_hit_rate": float(selected_df["target_hit"].mean()),
    }
    return selected_df, backtest_summary


def save_outputs(
    output_dir: Path,
    feature_cols: list[str],
    model: RandomForestClassifier,
    train_df: pd.DataFrame,
    test_predictions_df: pd.DataFrame,
    selected_trades_df: pd.DataFrame,
    summary: dict[str, object],
    export_excel: bool,
) -> None:
    """Persist datasets, reports, and feature importance to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    train_df.to_csv(output_dir / "train_dataset.csv", index=False)
    test_predictions_df.to_csv(output_dir / "test_predictions.csv", index=False)
    selected_trades_df.to_csv(output_dir / "selected_trades.csv", index=False)
    feature_importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    with open(output_dir / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, default=str)

    if export_excel:
        try:
            with pd.ExcelWriter(output_dir / "rf_demo_core_report.xlsx") as writer:
                train_df.to_excel(writer, sheet_name="train_dataset", index=False)
                test_predictions_df.to_excel(writer, sheet_name="test_predictions", index=False)
                selected_trades_df.to_excel(writer, sheet_name="selected_trades", index=False)
                feature_importance_df.to_excel(writer, sheet_name="feature_importance", index=False)
        except ImportError:
            print("Skipping Excel export because openpyxl is not installed.")


def build_summary(
    args: argparse.Namespace,
    raw_df: pd.DataFrame,
    model_df: pd.DataFrame,
    feature_cols: list[str],
    split_date: pd.Timestamp,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metrics: dict[str, object],
    backtest_summary: dict[str, object],
) -> dict[str, object]:
    """Assemble a compact JSON-serializable summary of the whole run."""
    ticker_counts = raw_df.groupby("ticker").size().sort_values(ascending=False)

    return {
        "config": {
            "candidate_mode": args.candidate_mode,
            "timeframe": args.timeframe,
            "horizon": args.horizon,
            "target_return": args.target_return,
            "stop_loss": args.stop_loss,
            "fee_pct": args.fee_pct,
            "prob_threshold": args.prob_threshold,
            "ambiguity_mode": args.ambiguity_mode,
            "capital_per_trade": args.capital_per_trade,
        },
        "data": {
            "raw_rows": int(len(raw_df)),
            "tickers": int(raw_df["ticker"].nunique()),
            "raw_start_date": str(raw_df["trading_date"].min().date()),
            "raw_end_date": str(raw_df["trading_date"].max().date()),
            "candidate_rows_after_feature_filter": int(len(model_df)),
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "split_date": str(split_date.date()),
            "largest_tickers_by_rows": ticker_counts.head(10).to_dict(),
        },
        "features": {
            "requested_feature_count": len(FINAL_FEATURE_COLS),
            "used_feature_count": len(feature_cols),
            "used_feature_cols": feature_cols,
            "dropped_constant_features": sorted(set(FINAL_FEATURE_COLS) - set(feature_cols)),
        },
        "model_metrics": metrics,
        "backtest": backtest_summary,
    }


def print_console_summary(summary: dict[str, object], output_dir: Path) -> None:
    """Print a human-readable summary after the pipeline finishes."""
    data = summary["data"]
    features = summary["features"]
    backtest = summary["backtest"]
    metrics = summary["model_metrics"]

    print("=" * 72)
    print("RF demo core pipeline finished")
    print("=" * 72)
    print(f"Rows loaded from Mongo     : {data['raw_rows']}")
    print(f"Tickers loaded             : {data['tickers']}")
    print(f"Date range                 : {data['raw_start_date']} -> {data['raw_end_date']}")
    print(f"Candidate rows             : {data['candidate_rows_after_feature_filter']}")
    print(f"Train / Test rows          : {data['train_rows']} / {data['test_rows']}")
    print(f"Chronological split date   : {data['split_date']}")
    print(f"Used features              : {features['used_feature_count']}")
    print(f"Dropped constant features  : {', '.join(features['dropped_constant_features']) or 'None'}")
    print(f"Classification accuracy    : {metrics['accuracy']:.4f}")
    print(f"Positive predictions       : {metrics['positive_predictions']}")
    print(f"Selected trades            : {backtest['selected_trades']}")
    print(f"Win rate                   : {backtest['win_rate']:.4f}")
    print(f"Average net return/trade   : {backtest['avg_return_net']:.4f}")
    print(f"Total PnL (VND)            : {backtest['total_pnl_vnd']:.0f}")
    print(f"Output directory           : {output_dir}")


def main() -> None:
    """Run the full end-to-end experiment."""
    args = parse_args()
    output_dir = Path(args.output_dir)
    tickers = parse_ticker_list(args.tickers)

    # 1) Load clean raw data from Mongo.
    raw_df = load_raw_from_mongo(
        mongo_uri=args.mongo_uri,
        mongo_db=args.mongo_db,
        mongo_collection=args.mongo_collection,
        timeframe=args.timeframe,
        tickers=tickers,
    )

    # 2) Build the final supervised dataset.
    model_df, feature_cols = build_model_frame(raw_df, args)

    # 3) Split by time so train only sees the past.
    train_df, test_df, split_date = split_by_time(model_df, args.test_size)

    # 4) Fit the baseline classifier.
    model = train_random_forest(train_df, feature_cols)

    # 5) Turn test rows into probabilities and hard class predictions.
    test_predictions_df, metrics = evaluate_model(
        model=model,
        test_df=test_df,
        feature_cols=feature_cols,
        prob_threshold=args.prob_threshold,
    )

    # 6) Convert selected predictions into a simplified portfolio backtest.
    selected_trades_df, backtest_summary = run_backtest(
        predictions_df=test_predictions_df,
        prob_threshold=args.prob_threshold,
        capital_per_trade=args.capital_per_trade,
    )

    # 7) Save a machine-readable summary for later comparison across runs.
    summary = build_summary(
        args=args,
        raw_df=raw_df,
        model_df=model_df,
        feature_cols=feature_cols,
        split_date=split_date,
        train_df=train_df,
        test_df=test_df,
        metrics=metrics,
        backtest_summary=backtest_summary,
    )

    # 8) Persist outputs and print a short report.
    save_outputs(
        output_dir=output_dir,
        feature_cols=feature_cols,
        model=model,
        train_df=train_df,
        test_predictions_df=test_predictions_df,
        selected_trades_df=selected_trades_df,
        summary=summary,
        export_excel=args.export_excel,
    )
    print_console_summary(summary, output_dir)


if __name__ == "__main__":
    main()
