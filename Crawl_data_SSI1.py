import requests
import pandas as pd

# API
url = "https://iboard-query.ssi.com.vn/exchange-index/HNX30?hasHistory=true"

# Token bạn lấy từ DevTools
headers = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://iboard.ssi.com.vn/",
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VybmFtZSI6IjAzOTg1NjUwNDYiLCJ1dWlkIjoiMjNlOGZkNTYtZTA2YS00OWM3LThjN2YtMWYzZjU2ZDkzNWE0IiwiY2hhbm5lbCI6IndlYiIsInN5c3RlbVR5cGUiOiJpYm9hcmQiLCJkZXZpY2VJZCI6IjVFQzU0QzJBLTI3NjQtNDg3MC1BODk5LUVEOTkwNDQ2MUFGNCIsInZlcnNpb24iOiIyIiwiaWF0IjoxNzczMjE2OTQyLCJleHAiOjE3NzMyNDU3NDJ9.tBT2-pVF2z2dYZ9Z6TDnmkKnNFnJFblWn34XNPo4sWc"   # dán token của bạn vào đây
}

# Request API
r = requests.get(url, headers=headers)
data = r.json()

# Lấy dữ liệu history
history = data["data"]["history"]

# Convert sang DataFrame
df = pd.DataFrame(history)

# Đổi tên cột
df = df.rename(columns={
    "t": "time",
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume"
})

# Convert timestamp
df["time"] = pd.to_datetime(df["time"], unit="ms")
print(df.columns)
print(df.head())
# Lấy 30 ngày gần nhất
df = df.tail(30)

# Chỉ giữ OHLCV
df = df[["time","open","high","low","close","volume"]]

print(df)