import ccxt
import pandas as pd

# 1. Kết nối Binance
binance = ccxt.binance({
    'apiKey': 'wVdhljHHMFJMJ0foVKau9zHt4M98N4GGCurpuZuD0e65uKHlciE3lrxHKwPirUsl',
    'secret': 'kjDQ0ssSusTTAXNWqXndUexUcDO9DZ4RuQNHA2ptzwAghVSIZ7PSrx5oFgoExAei',
})

# 2. Lấy dữ liệu lịch sử BTC/USDT
ohlcv = binance.fetch_ohlcv('BTC/USDT', timeframe='1d', limit=100)
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

# df.to_csv('BTC_USDT_1h.csv', index=False)

# # 3. Tính chỉ báo MA đơn giản
df['ma5'] = df['close'].rolling(5).mean()
df['ma20'] = df['close'].rolling(20).mean()

# 4. Tín hiệu mua/bán
df['signal'] = 0
df.loc[df['ma5'] > df['ma20'], 'signal'] = 1  # mua
df.loc[df['ma5'] < df['ma20'], 'signal'] = -1 # bán

pd.set_option('display.max_rows', None)     # Hiển thị tất cả các dòng

print(df[['timestamp', 'close', 'ma5', 'ma20', 'signal']])