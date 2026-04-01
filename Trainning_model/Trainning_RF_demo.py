import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# =========================
# 1) Giả sử df đã có các cột:
# open, high, low, close, volume
# =========================

# Ví dụ:
df = pd.read_csv("stock_data.csv")

# =========================
# 2) Feature Engineering (giữ nguyên như trước)
# =========================

df["ma10"] = df["close"].rolling(10).mean()
df["ma20"] = df["close"].rolling(20).mean()
df["ma50"] = df["close"].rolling(50).mean()

df["ma_gap_10_50"] = (df["ma10"] - df["ma50"]) / df["ma50"]

df["return_1d"] = df["close"].pct_change(1)

df["vol_ma20"] = df["volume"].rolling(20).mean()
df["volume_ratio"] = df["volume"] / df["vol_ma20"]

df["body"] = (df["close"] - df["open"]).abs()
df["range"] = df["high"] - df["low"]
df["body_ratio"] = df["body"] / df["range"].replace(0, np.nan)

# RSI(14)
delta = df["close"].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / loss.replace(0, np.nan)
df["rsi14"] = 100 - (100 / (1 + rs))

# =========================
# 3) Tạo tín hiệu crossover
# =========================

df["prev_ma10"] = df["ma10"].shift(1)
df["prev_ma50"] = df["ma50"].shift(1)

df["buy_crossover"] = (
    (df["prev_ma10"] <= df["prev_ma50"]) &
    (df["ma10"] > df["ma50"])
).astype(int)

# =========================
# 4) LABEL GENERATION MỚI
# Nếu chạm target trước H ngày thì dừng luôn
# =========================

TARGET_RETURN = 0.03   # target lợi nhuận 3% 
MAX_DRAWDOWN = -0.02    # giá giảm tối đa 2% 
H = 5                  # tối đa 5 ngày
total_fee_pct = 0.003  # phí sàn 

future_max_returns = []
future_min_returns = []
labels = []
days_to_target = []
entry_prices = []

for i in range(len(df)):
    # Không đủ dữ liệu tương lai để xét H ngày
    if i + H >= len(df):
        future_max_returns.append(np.nan)
        future_min_returns.append(np.nan)
        labels.append(np.nan)
        days_to_target.append(np.nan)
        entry_price.append(np.nan)
        continue

    entry_price = df["open"].iloc[i+1]
    entry_price.append(entry_price)
    
    # high and low of the next days
    future_high = df["high"].iloc[i + 1 : i + H + 1].values
    future_low = df["low"].iloc[i + 1 : i + H + 1].values

    # max return and min return 
    future_max_return = (future_high.max() - entry_price) / entry_price
    future_min_return = (future_low.min() - entry_price) / entry_price

    future_max_return.append(future_max_return)
    future_min_return.append(future_min_return)

    required_return = TARGET_RETURN + total_fee_pct
    target_price = entry_price * (1 + required_return)

    hit_day = np.nan
    for j, future_high in enumerate(future_high, start=1):
        if future_high >= target_price:
            hit_day = j
            break

    days_to_target.append(hit_day)

    # Label nhị phân
    label = int(
        (future_max_return >= required_return) and
        (future_min_return >= MAX_DRAWDOWN)
    )
    labels.append(label)

df["entry_price"] = entry_prices
df["future_max_return_5d"] = future_max_returns
df["future_min_return_5d"] = future_min_returns
df["days_to_target_5d"] = days_to_target
df["buy_label"] = labels

# =========================
# 5) Chỉ giữ các điểm crossover để train
# =========================

feature_cols = [
    "ma10", "ma20", "ma50",
    "ma_gap_10_50",
    "return_1d",
    "volume_ratio",
    "body_ratio",
    "rsi14"
]

train_df = df[df["buy_crossover"] == 1].copy()

# Bỏ các dòng thiếu dữ liệu
train_df = train_df.dropna(subset = feature_cols + ["buy_label"])
train_df["buy_label"] = train_df["buy_label"].astype(int)

# =========================
# 6) Tạo X và y
# =========================

X = train_df[feature_cols]
y = train_df["buy_label"]

# =========================
# 7) Train / Test
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=42,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# =========================
# 8) Dự đoán
# =========================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# =========================
# 9) Xem kết quả
# =========================

result = train_df.loc[X_test.index, feature_cols + ["buy_label", "days_to_target_5d","future_max_return_5d", "future_min_return_5d"]].copy()
result["pred_label"] = y_pred
result["p_win"] = y_prob

print(result.head(10))