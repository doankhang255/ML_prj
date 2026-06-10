from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# Định nghĩa các đặc trưng kỹ thuật và tham số chia dữ liệu dùng chung
SPLIT_RATIOS = {
    "652510": (0.65, 0.25, 0.10),
    "702010": (0.70, 0.20, 0.10),
    "751510": (0.75, 0.15, 0.10),
}

RETURN_1D_TARGET_COLUMN = "Future_Return_1D"

FEATURE_COLUMNS = [
    "Return",
    "Return_Pct",
    "Return_Lag_1",
    "Return_Lag_2",
    "Return_Lag_3",
    "Return_Lag_5",
    "Return_MA3",
    "Return_MA5",
    "Return_MA10",
    "High_Low_Range_Pct",
    "Close_Open_Return",
    "Close_MA7_Ratio",
    "Close_MA14_Ratio",
    "Close_MA30_Ratio",
    "MA7_MA14_Ratio",
    "MA7_MA30_Ratio",
    "Return_Volatility_7",
    "Return_Volatility_14",
    "Volume_Change_Pct",
    "Volume_Ratio_7",
    "Volume_Ratio_14",
]

# Danh sách 27 mã cổ phiếu cố định để tạo ánh xạ chỉ mục
TICKERS = [
    "ACB", "BID", "BSR", "CTG", "FPT", "GAS", "GVR", "HDB", "HPG", 
    "LPB", "MBB", "MSN", "MWG", "PLX", "SAB", "SHB", "SSI", "STB", 
    "TPB", "VCB", "VHM", "VIB", "VIC", "VJC", "VNM", "VPB", "VRE"
]

TICKER_TO_IDX = {ticker: idx for idx, ticker in enumerate(TICKERS)}

def load_and_split_data(
    csv_path: str | Path,
    split_code: str = "702010",
    target_column: str = RETURN_1D_TARGET_COLUMN
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Đọc dữ liệu và chia dữ liệu theo trình tự thời gian cho riêng từng Ticker để tránh rò rỉ thông tin.
    """
    df = pd.read_csv(csv_path)
    
    # Đảm bảo có cột Ticker
    if "Ticker" not in df.columns:
        filename = Path(csv_path).name
        ticker = "ACB"
        
        for t in TICKERS:
            if t in filename:
                ticker = t
                break
        df.insert(0, "Ticker", ticker)
        
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    # Nhóm theo Ticker để chia dữ liệu theo thời gian độc lập cho từng mã
    for ticker, group in df.groupby("Ticker"):
        group = group.sort_values("Date").reset_index(drop=True)
        n = len(group)
        train_ratio, val_ratio, _ = SPLIT_RATIOS[split_code]
        train_size = int(n * train_ratio)
        val_size = int(n * val_ratio)
        
        train_dfs.append(group.iloc[:train_size].copy())
        val_dfs.append(group.iloc[train_size : train_size + val_size].copy())
        test_dfs.append(group.iloc[train_size + val_size :].copy())
        
    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    return train_df, val_df, test_df

def create_sequence_windows_for_split(
    df_split: pd.DataFrame,
    features_list: list[str],
    target_column: str,
    look_back: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Tạo các cửa sổ chuỗi thời gian 3D cho từng Ticker để tránh trộn lẫn dữ liệu giữa các cổ phiếu.
    Trả về:
    - x_numeric: shape (samples, look_back, num_features)
    - x_ticker: shape (samples,) chứa chỉ mục của Ticker đại diện cho cửa sổ đó
    
    - y: shape (samples, 1) chứa nhãn mục tiêu
    """
    x_win_list = []
    x_ticker_list = []
    y_win_list = []
    
    for ticker, group in df_split.groupby("Ticker"):
        group = group.sort_values("Date").reset_index(drop=True)
        
        scaled_cols = [f"{col}_scaled" for col in features_list]
        x_data = group[scaled_cols].to_numpy(dtype=float)
        y_data = group["y_scaled"].to_numpy(dtype=float)
        ticker_idx = TICKER_TO_IDX.get(ticker, 0)
        
        if len(x_data) < look_back:
            continue
            
        for i in range(look_back - 1, len(x_data)):
            x_win_list.append(x_data[i - look_back + 1 : i + 1])
            x_ticker_list.append(ticker_idx)
            y_win_list.append(y_data[i])
            
    if not x_win_list:
        raise ValueError("Không tạo được bất kỳ cửa sổ dữ liệu nào. Hãy kiểm tra lại tham số look_back.")
        
    return np.array(x_win_list), np.array(x_ticker_list), np.array(y_win_list).reshape(-1, 1)

def get_dataloaders(
    csv_path: str | Path,
    look_back: int = 30,
    batch_size: int = 32,
    split_code: str = "702010",
    target_column: str = RETURN_1D_TARGET_COLUMN,
    scale_target: bool = True
) -> tuple[DataLoader, DataLoader, DataLoader, StandardScaler, StandardScaler | None, dict[str, int]]:
    """
    Pipeline chuẩn bị dữ liệu PyTorch:
    1. Đọc và chia dữ liệu độc lập theo từng Ticker.
    2. Chuẩn hóa đặc trưng dựa trên tập Train tổng của các mã.
    3. Tạo cửa sổ 3D và tách biệt chuỗi thời gian của từng mã.
    4. Trả về PyTorch DataLoader (Train được shuffle, Val/Test giữ nguyên thứ tự) và Ticker mapping.
    """
    train_df, val_df, test_df = load_and_split_data(csv_path, split_code, target_column)
    
    features_list = [col for col in FEATURE_COLUMNS if col in train_df.columns]
    
    # 1. Fit StandardScaler trên toàn bộ các mã thuộc tập Train
    x_scaler = StandardScaler()
    x_train_raw = train_df[features_list].to_numpy(dtype=float)
    x_scaler.fit(x_train_raw)
    
    # 2. Áp dụng chuẩn hóa và gán ngược vào dataframe dưới dạng cột mới
    x_train_scaled = x_scaler.transform(x_train_raw)
    x_val_scaled = x_scaler.transform(val_df[features_list].to_numpy(dtype=float))
    x_test_scaled = x_scaler.transform(test_df[features_list].to_numpy(dtype=float))
    
    for i, col in enumerate(features_list):
        train_df[f"{col}_scaled"] = x_train_scaled[:, i]
        val_df[f"{col}_scaled"] = x_val_scaled[:, i]
        test_df[f"{col}_scaled"] = x_test_scaled[:, i]
        
    # 3. Chuẩn hóa cột nhãn y
    y_scaler = None
    y_train_raw = train_df[[target_column]].to_numpy(dtype=float)
    if scale_target:
        y_scaler = StandardScaler()
        y_train_scaled = y_scaler.fit_transform(y_train_raw)
        y_val_scaled = y_scaler.transform(val_df[[target_column]].to_numpy(dtype=float))
        y_test_scaled = y_scaler.transform(test_df[[target_column]].to_numpy(dtype=float))
        
        train_df["y_scaled"] = y_train_scaled.ravel()
        val_df["y_scaled"] = y_val_scaled.ravel()
        test_df["y_scaled"] = y_test_scaled.ravel()
    else:
        train_df["y_scaled"] = y_train_raw.ravel()
        val_df["y_scaled"] = val_df[target_column].to_numpy(dtype=float)
        test_df["y_scaled"] = test_df[target_column].to_numpy(dtype=float)
        
    # 4. Tạo cửa sổ 3D độc lập theo từng mã cổ phiếu
    x_train_win, ticker_train, y_train_win = create_sequence_windows_for_split(
        train_df, features_list, target_column, look_back
    )
    x_val_win, ticker_val, y_val_win = create_sequence_windows_for_split(
        val_df, features_list, target_column, look_back
    )
    x_test_win, ticker_test, y_test_win = create_sequence_windows_for_split(
        test_df, features_list, target_column, look_back
    )
    
    # 5. Đóng gói TensorDataset
    train_ds = TensorDataset(
        torch.tensor(x_train_win, dtype=torch.float32),
        torch.tensor(ticker_train, dtype=torch.long),
        torch.tensor(y_train_win, dtype=torch.float32)
    )
    val_ds = TensorDataset(
        torch.tensor(x_val_win, dtype=torch.float32),
        torch.tensor(ticker_val, dtype=torch.long),
        torch.tensor(y_val_win, dtype=torch.float32)
    )
    test_ds = TensorDataset(
        torch.tensor(x_test_win, dtype=torch.float32),
        torch.tensor(ticker_test, dtype=torch.long),
        torch.tensor(y_test_win, dtype=torch.float32)
    )
    
    # 6. Tạo DataLoader (shuffle tập Train để tăng khả năng tổng quát hóa)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, x_scaler, y_scaler, TICKER_TO_IDX
