from vnstock import Quote
from datetime import datetime, timedelta
import pandas as pd
import os
import time
import random
from pathlib import Path

stocks = ["VNINDEX"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "Dataset"

# Ngày kết thúc là hôm nay, bắt đầu là 30 ngày trước
end_date = datetime.today().strftime("%Y-%m-%d")
start_date = (datetime.today() - timedelta(days=4000)).strftime("%Y-%m-%d")

# Tạo folder nếu chưa tồn tại
os.makedirs(OUTPUT_DIR, exist_ok=True)

for symbol in stocks:
    try:
        print(f"Đang lấy dữ liệu của {symbol}...")

        quote = Quote(source="VCI", symbol=symbol)
        df = quote.history(
            start=start_date,
            end=end_date,
            interval="1D"
        )

        if df is None or df.empty:
            print(f"Không có dữ liệu cho {symbol}")
        else:
            filename = OUTPUT_DIR / f"data_{symbol}.csv"
            df.to_csv(filename, index=False, encoding="utf-8-sig")
            print(f"Lấy dữ liệu của {symbol} thành công: {len(df)} dòng")

    except Exception as e:
        print(f"Lấy dữ liệu của {symbol} không thành công: {e}")

    # Nghỉ ngẫu nhiên từ 3 đến 7 giây để tránh bị limit
    sleep_time = random.uniform(3, 7)
    print(f"Nghỉ {sleep_time:.2f} giây trước request tiếp theo...\n")
    time.sleep(sleep_time)
