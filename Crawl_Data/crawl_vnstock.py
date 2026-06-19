from vnstock import Quote
from datetime import datetime
import pandas as pd
import os
import time
import random
from pathlib import Path

stocks = ["VNINDEX"] #VNINDEX

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "Dataset"

start_date = "2010-01-01"
end_date = datetime.today().strftime("%Y-%m-%d")

print("Đang chạy file:", __file__)
print("PROJECT_ROOT:", PROJECT_ROOT)
print("OUTPUT_DIR:", OUTPUT_DIR)
print("start_date:", start_date)
print("end_date:", end_date)

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
            df["time"] = pd.to_datetime(df["time"])

            # Ép lọc lại sau khi API trả dữ liệu
            df = df[
                (df["time"] >= pd.to_datetime(start_date)) &
                (df["time"] <= pd.to_datetime(end_date))
            ]

            df = df.sort_values("time")

            filename = OUTPUT_DIR / f"data_{symbol}.csv"

            if filename.exists():
                filename.unlink()
                print("Đã xóa file cũ:", filename.resolve())

            df.to_csv(filename, index=False, encoding="utf-8-sig")

            print(f"Lấy dữ liệu của {symbol} thành công: {len(df)} dòng")
            print("File đã lưu tại:", filename.resolve())
            print("Ngày đầu:", df["time"].min())
            print("Ngày cuối:", df["time"].max())

    except Exception as e:
        print(f"Lấy dữ liệu của {symbol} không thành công: {e}")

    sleep_time = random.uniform(3, 7)
    print(f"Nghỉ {sleep_time:.2f} giây trước request tiếp theo...\n")
    time.sleep(sleep_time)