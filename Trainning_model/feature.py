# Trend
# Trend + Momentum
# Trend + Momentum + Volume
# Trend + Volume + Candlestick 
# Trend + Momentum + Volume + Volatility
# Full feature set


# Trend Features:
# MA10, MA50, ratios, slopes, crossover

# Momentum Features:
# Returns, RSI, MACD

# Volume Features:
# Volume ratio, volume spike

# Volatility Features:
# ATR, range, std

# Candlestick Features:
# Body ratio, wick ratio, close position

trend_feature_cols = [
    # Moving Average
    "ma10",
    "ma20",
    "ma50",

    # Ratio
    "ma10_ma50_ratio",
    "close_ma10_ratio",
    "close_ma50_ratio",

    # Distance (%)
    "dist_close_ma10_pct",
    "dist_close_ma50_pct",
    "dist_ma10_ma50_pct",

    # Slope (trend strength)
    "ma10_slope_1d",
    "ma50_slope_1d",
    "ma10_slope_3d",
    "ma50_slope_3d",

    # Cross signal
    "ma10_above_ma50",
    "golden_cross",
    "death_cross",

    # Time since cross
    "days_since_golden_cross",
    "days_since_death_cross",
]

momentum_feature_cols = [
    # Returns
    "return_1d",
    "return_3d",
    "return_5d",
    "return_10d",

    # RSI
    "rsi",

    # MACD
    "macd",
    "macd_signal",
    "macd_hist",

    # Rate of change
    "roc_5",
]

volume_feature_cols = [
    "volume",

    # Average volume
    "volume_sma10",
    "volume_sma20",

    # Ratio
    "volume_ratio_10",
    "volume_ratio_20",

    # Change
    "volume_change_pct",

    # Spike
    "volume_spike",
]

volatility_feature_cols = [
    # Range
    "high_low_range",
    "high_low_range_pct",

    # ATR
    "atr",
    "atr_pct",

    # Standard deviation
    "rolling_std_10",
    "rolling_std_20",

    # True range
    "true_range",
]

candlestick_feature_cols = [
    # Body
    "body",
    "body_ratio",

    # Wick
    "upper_wick",
    "lower_wick",
    "upper_wick_ratio",
    "lower_wick_ratio",

    # Close position
    "close_position",

    # Direction
    "is_bullish",
    "is_bearish",

    # Gap
    "gap_up",
    "gap_down",
]