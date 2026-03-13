import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 1. Load dữ liệu
df = pd.read_csv("Data_Stock/data_ACB.csv", parse_dates=['time'])

# 2. Tính MA10 và MA50
df['MA10'] = df['close'].rolling(window=10).mean()
df['MA50'] = df['close'].rolling(window=50).mean()

# 3. Xây dựng label cho MA crossover
# 1 = mua, -1 = bán, 0 = giữ
def generate_label(row, prev_row):
    if np.isnan(prev_row['MA10']) or np.isnan(prev_row['MA50']):
        return 0
    # MA10 cắt lên MA50 → mua
    if prev_row['MA10'] < prev_row['MA50'] and row['MA10'] > row['MA50']:
        return 1
    # MA10 cắt xuống MA50 → bán
    elif prev_row['MA10'] > prev_row['MA50'] and row['MA10'] < row['MA50']:
        return -1
    else:
        return 0

df['label'] = 0
for i in range(1, len(df)):
    df.loc[i, 'label'] = generate_label(df.iloc[i], df.iloc[i-1])

# 4. Chọn feature và label
features = ['open', 'high', 'low', 'close', 'MA10', 'MA50']
df = df.dropna()  # bỏ các hàng đầu không có MA50
X = df[features]
y = df['label']

# 5. Chia dữ liệu train/test (train: 2023-04-25 → 2024-07-30)
train_df = df[df['time'] <= '2024-07-30']
X_train = train_df[features]
y_train = train_df['label']

# 6. Khởi tạo và train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 7. Kiểm tra accuracy trên cùng dữ liệu train (để tham khảo)
y_pred_train = rf.predict(X_train)
print("Accuracy trên train data:", accuracy_score(y_train, y_pred_train))
print(classification_report(y_train, y_pred_train))

print("Mô hình Random Forest đã train xong!")