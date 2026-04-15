from vnstock import Quote
from datetime import datetime, timedelta
import pandas as pd
import os

# Danh sách 20 mã cổ phiếu HOSE
stocks = [
        "VCB","VIC","VHM","BID","TCB","CTG","FPT","HPG","GAS","MBB",
        "VPB","VNM","ACB","MSN","MWG","LPB","STB","HVN","GVR","HDB",
        "SAB","BCM","BSR","VRE","VIB","SHB","SSB","VJC","SSI","EIB",
        "BVH","REE","DGC","GEE","TPB","MSB","GEX","POW","KDH","NVL",
        "OCB","PNJ","VCI","VND","GMD","FRT","NAB","PGV","VGC","VIX",
        "KBC","DCM","HCM","VPI","DXG","PDR","SBT","KDC","NLG","DPM",
        "HAG","SIP","VCG","TCH","DHG","VHC","FTS","PVD","LGC","CTR",
        "VSH","BMP","DIG","BWE","SJS","HSG","DBC","BSI","HAH","HDG",
        "PVT","DGW","KOS","BHN","CTD","PC1","IMP","CII","EVF","CMG",
        "VCF","DSE","TMS","PHR","VSC","SCS","TDM","SZC","NKG","GEG"
        ]
# Ngày kết thúc là hôm nay, bắt đầu là 30 ngày trước
end_date = datetime.today().strftime("%Y-%m-%d")
start_date = (datetime.today() - timedelta(days=1000)).strftime("%Y-%m-%d")

# # Tạo folder nếu chưa tồn tại
# if not os.path.exists("Data_Stock"):
#     os.makedirs("Data_Stock")

for symbol in stocks:
    try:
        quote = Quote(source="vci", symbol=symbol)  # có thể đổi "vci" sang "mas" hoặc "ssi"
        df = quote.history(start=start_date, end=end_date)

        filename = os.path.join("Data_Stock",f"data_{symbol}.csv")
        df.to_csv(filename, index = False)
        print(f"lấy dữ liệu của {symbol} thành công")
    except Exception as e:
        print(f"lấy dữ liệu của {symbol} không thành công: {e}")

