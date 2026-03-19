import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 1. Load dữ liệu
df = pd.read_csv("Data_Stock/data_ACB.csv", parse_dates=['time'])

# 2. Tính MA10 và MA50
df['MA10'] = df['close'].rolling(window=10).mean()
df['MA50'] = df['close'].rolling(window=50).mean()

# 3. Tạo label mua/bán dựa trên MA crossover
def generate_label(row, prev_row):
    if pd.isna(prev_row['MA10']) or pd.isna(prev_row['MA50']):
        return 0
    if prev_row['MA10'] < prev_row['MA50'] and row['MA10'] > row['MA50']:
        return 1
    elif prev_row['MA10'] > prev_row['MA50'] and row['MA10'] < row['MA50']:
        return -1
    else:
        return 0

df['label'] = 0
for i in range(1, len(df)):
    df.loc[i, 'label'] = generate_label(df.iloc[i], df.iloc[i-1])

# 4. Chọn dữ liệu train
train_df = df[df['time'] <= '2024-07-30']

# 5. Vẽ biểu đồ
plt.figure(figsize=(16,6))
plt.plot(train_df['time'], train_df['MA10'], label='MA10', color='blue')
plt.plot(train_df['time'], train_df['MA50'], label='MA50', color='red')

# Vẽ các điểm mua/bán
plt.scatter(train_df[train_df['label']==1]['time'], 
            train_df[train_df['label']==1]['close'], 
            color='green', label='Buy Signal', marker='^', s=100)
plt.scatter(train_df[train_df['label']==-1]['time'], 
            train_df[train_df['label']==-1]['close'], 
            color='black', label='Sell Signal', marker='v', s=100)

# 6. Định dạng trục x theo ngày
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))  # hiển thị 1 tick mỗi 5 ngày
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)
plt.title("MA Crossover - ACB (Train Data)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()