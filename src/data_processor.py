from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "Dataset"
OUTPUT_DIR = PROJECT_ROOT / "Preprocessed_Data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

COLUMN_MAP = {
    "time": "Date",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume"
}

def parse_number(value: object) -> float:
    """Xử lý và chuyển đổi dữ liệu dạng số (kể cả các chuỗi có định dạng như %, K, M, B)."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)

    text = str(value).strip().replace('"', "")
    if not text:
        return np.nan

    multiplier = 1.0
    suffix = text[-1].upper()
    if text.endswith("%"):
        text = text[:-1]
    elif suffix == "K":
        multiplier = 1_000.0
        text = text[:-1]
    elif suffix == "M":
        multiplier = 1_000_000.0
        text = text[:-1]
    elif suffix == "B":
        multiplier = 1_000_000_000.0
        text = text[:-1]

    try:
        return float(text.replace(",", "")) * multiplier
    except ValueError:
        return np.nan

def parse_dates(values: pd.Series) -> pd.Series:
    """Xử lý định dạng ngày tháng một cách linh hoạt."""
    sample = values.dropna().astype(str).head(1)
    if not sample.empty and "-" in sample.iloc[0] and len(sample.iloc[0].split("-")[0]) == 4:
        return pd.to_datetime(values, errors="coerce")
    return pd.to_datetime(values, dayfirst=True, errors="coerce")

def load_raw_data(input_path: Path) -> pd.DataFrame:
    """Đọc dữ liệu thô, ánh xạ cột và làm sạch kiểu dữ liệu."""
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    df = df.rename(columns=COLUMN_MAP)
    
    required = ["Date", "Open", "High", "Low", "Close"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"File {input_path.name} thiếu các cột bắt buộc: {missing}")

    df["Date"] = parse_dates(df["Date"])
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = df[col].map(parse_number)

    # Loại bỏ các dòng thiếu thông tin cơ bản, sắp xếp theo ngày
    df = df.dropna(subset=required).sort_values("Date").reset_index(drop=True)
    return df

def add_features_and_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Tính toán các đặc trưng kỹ thuật và các cột mục tiêu (Targets) dự đoán."""
    out = df.copy()
    
    # 1. Tính toán Tỷ suất sinh lời cơ bản
    out["Return"] = out["Close"].pct_change()
    out["Return_Pct"] = out["Return"] * 100
    
    # 2. Tạo các cột mục tiêu (Targets) cần dự đoán
    # Ngày mục tiêu
    out["Target_Date_1D"] = out["Date"].shift(-1)
    
    # Target hồi quy giá: Giá đóng cửa ngày mai
    out["Next_Close"] = out["Close"].shift(-1)
    
    # Target hồi quy tỷ suất sinh lời (Ngắn hạn)
    out["Future_Return_1D"] = out["Return"].shift(-1)
    out["Future_Return_1D_Pct"] = out["Future_Return_1D"] * 100
    
    # Tỷ suất sinh lời trong 5 phiên tới (khoảng 1 tuần)
    future_close_5d = out["Close"].shift(-5)
    out["Future_Return_5D"] = (future_close_5d / out["Close"]) - 1
    
    # Tỷ suất sinh lời trong 10 phiên tới (khoảng 2 tuần)
    future_close_10d = out["Close"].shift(-10)
    out["Future_Return_10D"] = (future_close_10d / out["Close"]) - 1
    
    # Target hồi quy tỷ suất sinh lời (Trung hạn)
    # Tỷ suất sinh lời trong 20 phiên tới (khoảng 1 tháng)
    future_close_20d = out["Close"].shift(-20)
    out["Future_Return_20D"] = (future_close_20d / out["Close"]) - 1
    
    # Tỷ suất sinh lời trong 60 phiên tới (khoảng 3 tháng)
    future_close_60d = out["Close"].shift(-60)
    out["Future_Return_60D"] = (future_close_60d / out["Close"]) - 1

    # 3. Đặc trưng giá và nến
    out["High_Low_Spread"] = out["High"] - out["Low"]
    out["Close_Open_Change"] = out["Close"] - out["Open"]
    out["High_Low_Range_Pct"] = (out["High"] - out["Low"]) / out["Close"]
    out["Close_Open_Return"] = (out["Close"] / out["Open"]) - 1
    
    # 4. Đặc trưng thời gian
    out["DayOfWeek"] = out["Date"].dt.dayofweek
    out["Month"] = out["Date"].dt.month
    out["Year"] = out["Date"].dt.year
    
    # 5. Đường trung bình động (Moving Averages)
    out["MA7"] = out["Close"].rolling(window=7, min_periods=7).mean()
    out["MA14"] = out["Close"].rolling(window=14, min_periods=14).mean()
    out["MA30"] = out["Close"].rolling(window=30, min_periods=30).mean()
    
    # Tỷ lệ giá so với đường trung bình
    out["Close_MA7_Ratio"] = (out["Close"] / out["MA7"]) - 1
    out["Close_MA14_Ratio"] = (out["Close"] / out["MA14"]) - 1
    out["Close_MA30_Ratio"] = (out["Close"] / out["MA30"]) - 1
    
    # Tỷ lệ giữa các đường trung bình với nhau
    out["MA7_MA14_Ratio"] = (out["MA7"] / out["MA14"]) - 1
    out["MA7_MA30_Ratio"] = (out["MA7"] / out["MA30"]) - 1
    
    # Trung bình động của Tỷ suất sinh lời
    out["Return_MA3"] = out["Return"].rolling(window=3, min_periods=3).mean()
    out["Return_MA5"] = out["Return"].rolling(window=5, min_periods=5).mean()
    out["Return_MA10"] = out["Return"].rolling(window=10, min_periods=10).mean()
    
    # 6. Độ biến động (Volatility)
    out["Volatility_7"] = out["Close"].rolling(window=7, min_periods=7).std()
    out["Volatility_14"] = out["Close"].rolling(window=14, min_periods=14).std()
    out["Return_Volatility_7"] = out["Return"].rolling(window=7, min_periods=7).std()
    out["Return_Volatility_14"] = out["Return"].rolling(window=14, min_periods=14).std()
    
    # 7. Khối lượng giao dịch (Volume)
    if "Volume" in out.columns:
        out["Volume_Change_Pct"] = out["Volume"].pct_change()
        out["Volume_Ratio_7"] = out["Volume"] / out["Volume"].rolling(window=7, min_periods=7).mean()
        out["Volume_Ratio_14"] = out["Volume"] / out["Volume"].rolling(window=14, min_periods=14).mean()
        
    # 8. Các cột Trễ (Lags)
    for lag in [1, 2, 3, 5]:
        out[f"Close_Lag_{lag}"] = out["Close"].shift(lag)
        out[f"Return_Lag_{lag}"] = out["Return"].shift(lag)
        
    # Xử lý vô cùng (inf) thành NaN, sau đó drop tất cả các dòng chứa NaN
    out = out.replace([np.inf, -np.inf], np.nan)
    return out.dropna().reset_index(drop=True)

def process_all_tickers():
    print("=" * 80)
    print("BẮT ĐẦU TIỀN XỬ LÝ DỮ LIỆU CHO 27 MÃ CỔ PHIẾU")
    print("=" * 80)
    
    # Quét toàn bộ các file data_*.csv
    csv_files = sorted(list(INPUT_DIR.glob("data_*.csv")))
    
    if not csv_files:
        print(f"Không tìm thấy file dữ liệu nào trong thư mục: {INPUT_DIR}")
        return
        
    success_count = 0
    all_processed_dfs = []
    
    for file_path in csv_files:
        ticker = file_path.stem.replace("data_", "")
        output_file_path = OUTPUT_DIR / f"data_{ticker}_processed.csv"
        
        try:
            print(f"Đang xử lý mã: {ticker}...")
            # 1. Đọc và dọn dẹp dữ liệu thô
            raw_df = load_raw_data(file_path)
            
            # 2. Tạo đặc trưng và mục tiêu dự đoán
            processed_df = add_features_and_targets(raw_df)
            processed_df.insert(0, "Ticker", ticker)
            
            # 3. Lưu kết quả file riêng lẻ
            processed_df.to_csv(output_file_path, index=False)
            print(f"-> Hoàn thành {ticker}: {len(raw_df)} dòng thô -> {len(processed_df)} dòng đã xử lý")
            
            all_processed_dfs.append(processed_df)
            success_count += 1
            
        except Exception as e:
            print(f"[LỖI] Không thể xử lý mã {ticker}: {e}")
            
    # 4. Gộp toàn bộ dữ liệu của 27 cổ phiếu vào 1 file tổng hợp
    if all_processed_dfs:
        combined_df = pd.concat(all_processed_dfs, ignore_index=True)
        combined_output_path = OUTPUT_DIR / "all_tickers_processed.csv"
        combined_df.to_csv(combined_output_path, index=False)
        print(f"\n-> Đã gộp và lưu file tổng hợp: {combined_output_path} ({len(combined_df)} dòng)")
            
    print("=" * 80)
    print(f"TIỀN XỬ LÝ HOÀN TẤT: Đã xử lý thành công {success_count}/{len(csv_files)} file.")
    print(f"Kết quả được lưu tại: {OUTPUT_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    process_all_tickers()
