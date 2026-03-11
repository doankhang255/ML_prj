from vnstock import Quote
from datetime import datetime, timedelta
import pandas as pd

# Danh sách 20 mã cổ phiếu HOSE
stocks = [
    "VIC", "VCB", "VHM", "BID", "CTG",
    "TCB", "HPG", "MBB", "FPT", "VNM",
    "STB", "LPB", "ACB", "MSN", "VJC",
    "PLX", "SSI", "HVN", "VCK", "SHB"
]

# Ngày kết thúc là hôm nay, bắt đầu là 30 ngày trước
end_date = datetime.today().strftime("%Y-%m-%d")
start_date = (datetime.today() - timedelta(days=1000)).strftime("%Y-%m-%d")

# List để chứa tất cả dữ liệu
all_data_list = []

for symbol in stocks:
    try:
        quote = Quote(source="vci", symbol=symbol)  # có thể đổi "vci" sang "mas" hoặc "ssi"
        df = quote.history(start=start_date, end=end_date)
        filename = f"data_{symbol}.csv"
        df.to_csv(filename, index = False)
        print(f"Đã lấy dữ liệu của {symbol}")
    except Exception as e:
        print(f"Lỗi khi lấy dữ liệu của {symbol}: {e}")

