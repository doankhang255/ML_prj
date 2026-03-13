import ccxt
import pandas as pd
import os

# Kết nối Binance
binance = ccxt.binance({
    'apiKey': 'wVdhljHHMFJMJ0foVKau9zHt4M98N4GGCurpuZuD0e65uKHlciE3lrxHKwPirUsl',
    'secret': 'kjDQ0ssSusTTAXNWqXndUexUcDO9DZ4RuQNHA2ptzwAghVSIZ7PSrx5oFgoExAei',
})

cryptos = ['BTC/USDT', 'ETH/USDT']

for crypto in cryptos:
    # Lấy dữ liệu lịch sử
    ohlcv = binance.fetch_ohlcv(crypto, timeframe='1d', limit = 1000)
    df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['time'], unit='ms')

    #lưu vào thư mục
    filename = os.path.join("Data_Crypto", f"{crypto}_1d.csv")
    df.to_csv(filename, index=False)

