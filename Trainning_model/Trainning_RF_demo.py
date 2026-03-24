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
H = 5                  # tối đa 5 ngày

labels = []
days_to_target = []

for i in range(len(df)):
    # Không đủ dữ liệu tương lai để xét H ngày
    if i + H >= len(df):
        labels.append(np.nan)
        days_to_target.append(np.nan)
        continue

    entry_price = df["close"].iloc[i]
    target_price = entry_price * (1 + TARGET_RETURN)

    # Lấy giá close của H ngày tiếp theo
    future_closes = df["close"].iloc[i + 1 : i + H + 1].values

    label = 0
    hit_day = np.nan

    # Duyệt từng ngày tương lai
    for j, future_price in enumerate(future_closes, start=1):
        # Nếu chạm target sớm thì dừng luôn
        if future_price >= target_price:
            label = 1
            hit_day = j
            break

    labels.append(label)
    days_to_target.append(hit_day)

df["label"] = labels
df["days_to_target"] = days_to_target

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
train_df = train_df.dropna(subset=feature_cols + ["label"])

# =========================
# 6) Tạo X và y
# =========================

X = train_df[feature_cols]
y = train_df["label"]

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

result = train_df.loc[X_test.index, feature_cols + ["label", "days_to_target"]].copy()
result["pred_label"] = y_pred
result["p_win"] = y_prob

print(result.head(10))