import pandas as pd
from pymongo import MongoClient
from pymongo import UpdateOne
import certifi

MONGO_URI = "mongodb+srv://doankhangll255_db_user:EGyi6XqdcCAwxbrf@cluster0.ufdio5k.mongodb.net/?retryWrites=true&w=majority"

client = MongoClient(MONGO_URI, tls=True, tlsCAFile=certifi.where())
db = client["stock_ml"]
col = db["raw_ohlcv_daily"]

col.create_index([("ticker",1), ("trading_date",1), ("timeframe",1)], unique = True) 

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

for ticker in stocks:
    ops = []
    df = pd.read_csv(f"Data_Stock/data_{ticker}.csv")
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"time": "trading_date"})
    df["ticker"] = ticker
    df["timeframe"] = "1D"
    df["source"] = "vnstock"
    records = df[["ticker", "trading_date", "timeframe", "open", "high", "low", "close", "volume", "source"]].to_dict("records")   
    for row in records: 
        ops.append(
            UpdateOne(
                {
                    "ticker": row["ticker"],
                    "trading_date": row["trading_date"],
                    "timeframe": row["timeframe"]
                },
                {"$set": row},
                upsert=True
            )
        )
    col.bulk_write(ops, ordered=False)
    print(f"Push {ticker} thành công")

