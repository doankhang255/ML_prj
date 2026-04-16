from __future__ import annotations

from datetime import date
from pathlib import Path
import os
import time

import certifi
import pandas as pd
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from vnstock import Quote

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


# =========================
# User config
# =========================
# Mongo config is loaded from the root .env file.
MONGO_URI = os.getenv("MONGO_URI", "").strip()
MONGO_DB = os.getenv("MONGO_DB", "stock_ml").strip() or "stock_ml"
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "raw_ohlcv_daily").strip() or "raw_ohlcv_daily"

VNSTOCK_SOURCE = os.getenv("VNSTOCK_SOURCE", "vci").strip() or "vci"
TIMEFRAME = "1D"
INITIAL_START_DATE = date(2022, 1, 1)
REQUEST_DELAY_SECONDS = 0.2
BULK_WRITE_BATCH_SIZE = 1000

PARQUET_ROOT = PROJECT_ROOT / "Parquet_Stock"
PARQUET_FILE_NAME = "ohlcv_daily.parquet"

STOCKS = [
    "VCB", "VIC", "VHM", "BID", "TCB", "CTG", "FPT", "HPG", "GAS", "MBB",
    "VPB", "VNM", "ACB", "MSN", "MWG", "LPB", "STB", "HVN", "GVR", "HDB",
    "SAB", "BCM", "BSR", "VRE", "VIB", "SHB", "SSB", "VJC", "SSI", "EIB",
    "BVH", "REE", "DGC", "GEE", "TPB", "MSB", "GEX", "POW", "KDH", "NVL",
    "OCB", "PNJ", "VCI", "VND", "GMD", "FRT", "NAB", "PGV", "VGC", "VIX",
    "KBC", "DCM", "HCM", "VPI", "DXG", "PDR", "SBT", "KDC", "NLG", "DPM",
    "HAG", "SIP", "VCG", "TCH", "DHG", "VHC", "FTS", "PVD", "LGC", "CTR",
    "VSH", "BMP", "DIG", "BWE", "SJS", "HSG", "DBC", "BSI", "HAH", "HDG",
    "PVT", "DGW", "KOS", "BHN", "CTD", "PC1", "IMP", "CII", "EVF", "CMG",
    "VCF", "DSE", "TMS", "PHR", "VSC", "SCS", "TDM", "SZC", "NKG", "GEG",
]

OUTPUT_COLUMNS = [
    "ticker",
    "trading_date",
    "timeframe",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "source",
]


def ensure_parquet_engine() -> None:
    """Fail fast with a helpful message if parquet support is missing."""
    try:
        import pyarrow  # noqa: F401
        return
    except ImportError:
        pass

    try:
        import fastparquet  # noqa: F401
        return
    except ImportError as exc:
        raise ImportError(
            "Parquet support is missing. Please install pyarrow or fastparquet first."
        ) from exc


def resolve_mongo_uri() -> str:
    """Get Mongo URI from config/env and stop early if missing."""
    mongo_uri = MONGO_URI.strip()
    if not mongo_uri:
        raise ValueError(
            "Missing MongoDB URI. Please set MONGO_URI in the root .env file "
            "before running this script."
        )
    return mongo_uri


def parse_partition_date(path: Path) -> date | None:
    """Extract YYYY/MM/DD from path like year=2026/month=04/day=16/file.parquet."""
    try:
        relative_parts = path.relative_to(PARQUET_ROOT).parts
        pieces = {}
        for part in relative_parts[:-1]:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            pieces[key] = int(value)
        return date(pieces["year"], pieces["month"], pieces["day"])
    except (ValueError, KeyError):
        return None


def get_latest_parquet_date(parquet_root: Path) -> date | None:
    """Return the latest partition date already stored in parquet."""
    if not parquet_root.exists():
        return None

    latest: date | None = None
    for parquet_file in parquet_root.rglob("*.parquet"):
        partition_date = parse_partition_date(parquet_file)
        if partition_date is None:
            continue
        if latest is None or partition_date > latest:
            latest = partition_date
    return latest


def compute_crawl_window(parquet_root: Path) -> tuple[date, date, date | None]:
    """First run crawls from 2022-01-01. Later runs refresh from last saved day."""
    latest_saved_date = get_latest_parquet_date(parquet_root)
    today = date.today()
    start_date = latest_saved_date if latest_saved_date is not None else INITIAL_START_DATE
    if start_date > today:
        start_date = today
    return start_date, today, latest_saved_date


def empty_output_frame() -> pd.DataFrame:
    """Return a DataFrame with the canonical project schema."""
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def normalize_history_frame(raw_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize vnstock output into the schema expected by this project."""
    if raw_df is None or raw_df.empty:
        return empty_output_frame()

    df = raw_df.copy()
    df.columns = [str(column).strip().lower() for column in df.columns]
    df = df.rename(
        columns={
            "time": "trading_date",
            "date": "trading_date",
            "datetime": "trading_date",
        }
    )

    required_columns = {"trading_date", "open", "high", "low", "close", "volume"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"{ticker}: vnstock data is missing required columns: {sorted(missing)}")

    df["trading_date"] = pd.to_datetime(df["trading_date"], errors="coerce").dt.normalize()
    for column in ["open", "high", "low", "close", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["ticker"] = ticker.upper()
    df["timeframe"] = TIMEFRAME
    df["source"] = f"vnstock:{VNSTOCK_SOURCE}"

    df = df[OUTPUT_COLUMNS]
    df = df.dropna(subset=["ticker", "trading_date", "open", "high", "low", "close", "volume"])
    df = df.sort_values(["trading_date", "ticker"]).drop_duplicates(
        subset=["ticker", "trading_date", "timeframe"],
        keep="last",
    )
    return df.reset_index(drop=True)


def fetch_symbol_history(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch one ticker from vnstock and normalize it."""
    print(f"[crawl] {symbol}: fetching {start_date} -> {end_date}")
    quote = Quote(source=VNSTOCK_SOURCE, symbol=symbol)
    raw_df = quote.history(start=start_date.isoformat(), end=end_date.isoformat())
    normalized_df = normalize_history_frame(raw_df, symbol)
    print(f"[crawl] {symbol}: {len(normalized_df)} rows")
    return normalized_df


def fetch_all_symbols(symbols: list[str], start_date: date, end_date: date) -> tuple[pd.DataFrame, list[str]]:
    """Fetch every symbol and keep going even if a few symbols fail."""
    frames: list[pd.DataFrame] = []
    failed_symbols: list[str] = []

    for symbol in symbols:
        try:
            frame = fetch_symbol_history(symbol, start_date, end_date)
            if not frame.empty:
                frames.append(frame)
        except Exception as exc:
            failed_symbols.append(symbol)
            print(f"[crawl] {symbol}: failed -> {exc}")

        if REQUEST_DELAY_SECONDS > 0:
            time.sleep(REQUEST_DELAY_SECONDS)

    if not frames:
        return empty_output_frame(), failed_symbols

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["trading_date", "ticker"]).drop_duplicates(
        subset=["ticker", "trading_date", "timeframe"],
        keep="last",
    )
    return combined.reset_index(drop=True), failed_symbols


def partition_path_for_day(parquet_root: Path, trading_day: date) -> Path:
    """Build the target parquet path for one trading day."""
    return (
        parquet_root
        / f"year={trading_day.year:04d}"
        / f"month={trading_day.month:02d}"
        / f"day={trading_day.day:02d}"
        / PARQUET_FILE_NAME
    )


def align_output_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only expected columns and coerce common dtypes."""
    working = df.copy()
    for column in OUTPUT_COLUMNS:
        if column not in working.columns:
            working[column] = pd.NA

    working = working[OUTPUT_COLUMNS]
    working["trading_date"] = pd.to_datetime(working["trading_date"], errors="coerce").dt.normalize()
    for column in ["open", "high", "low", "close", "volume"]:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    working = working.dropna(subset=["ticker", "trading_date", "open", "high", "low", "close", "volume"])
    working = working.sort_values(["trading_date", "ticker"]).drop_duplicates(
        subset=["ticker", "trading_date", "timeframe"],
        keep="last",
    )
    return working.reset_index(drop=True)


def write_daily_parquet_partitions(df: pd.DataFrame, parquet_root: Path) -> list[Path]:
    """Write one parquet file per trading day under year/month/day folders."""
    ensure_parquet_engine()

    if df.empty:
        return []

    written_files: list[Path] = []
    working = align_output_schema(df)
    working["partition_day"] = working["trading_date"].dt.date

    for trading_day, day_df in working.groupby("partition_day", sort=True):
        target_path = partition_path_for_day(parquet_root, trading_day)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        day_output = day_df.drop(columns=["partition_day"]).copy()
        if target_path.exists():
            existing_df = pd.read_parquet(target_path)
            day_output = pd.concat([existing_df, day_output], ignore_index=True)
            day_output = align_output_schema(day_output)

        day_output.to_parquet(target_path, index=False)
        written_files.append(target_path)
        print(f"[parquet] wrote {len(day_output)} rows -> {target_path}")

    return written_files


def build_mongo_ops(df: pd.DataFrame) -> list[UpdateOne]:
    """Create idempotent upsert operations compatible with the current project schema.

    Important compatibility note:
    - The existing project likely stores trading_date as an ISO date string in Mongo.
    - We keep that format here so unique index matching stays consistent.
    """
    ops: list[UpdateOne] = []

    working = align_output_schema(df)
    for row in working.itertuples(index=False):
        trading_date_str = pd.Timestamp(row.trading_date).strftime("%Y-%m-%d")
        document = {
            "ticker": row.ticker,
            "trading_date": trading_date_str,
            "timeframe": row.timeframe,
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "volume": float(row.volume),
            "source": row.source,
        }
        ops.append(
            UpdateOne(
                {
                    "ticker": document["ticker"],
                    "trading_date": document["trading_date"],
                    "timeframe": document["timeframe"],
                },
                {"$set": document},
                upsert=True,
            )
        )
    return ops


def flush_mongo_batch(collection, ops: list[UpdateOne]) -> tuple[int, int, int]:
    """Write one batch to Mongo and return matched/modified/upserted counts."""
    if not ops:
        return 0, 0, 0

    result = collection.bulk_write(ops, ordered=False)
    return result.matched_count, result.modified_count, result.upserted_count


def push_to_mongo(df: pd.DataFrame, mongo_uri: str) -> dict[str, int]:
    """Upsert the fetched rows into MongoDB."""
    if df.empty:
        return {"matched": 0, "modified": 0, "upserted": 0}

    ops = build_mongo_ops(df)
    matched_total = 0
    modified_total = 0
    upserted_total = 0

    client = MongoClient(mongo_uri, tls=True, tlsCAFile=certifi.where())
    try:
        collection = client[MONGO_DB][MONGO_COLLECTION]
        collection.create_index([("ticker", 1), ("trading_date", 1), ("timeframe", 1)], unique=True)

        batch: list[UpdateOne] = []
        for op in ops:
            batch.append(op)
            if len(batch) >= BULK_WRITE_BATCH_SIZE:
                matched, modified, upserted = flush_mongo_batch(collection, batch)
                matched_total += matched
                modified_total += modified
                upserted_total += upserted
                batch = []

        if batch:
            matched, modified, upserted = flush_mongo_batch(collection, batch)
            matched_total += matched
            modified_total += modified
            upserted_total += upserted
    finally:
        client.close()

    return {
        "matched": matched_total,
        "modified": modified_total,
        "upserted": upserted_total,
    }


def main() -> None:
    ensure_parquet_engine()
    mongo_uri = resolve_mongo_uri()

    start_date, end_date, latest_saved_date = compute_crawl_window(PARQUET_ROOT)
    print("=" * 72)
    print("vnstock -> parquet -> MongoDB pipeline")
    print("=" * 72)
    print(f"Parquet root              : {PARQUET_ROOT}")
    print(f"Initial start date        : {INITIAL_START_DATE}")
    print(f"Latest parquet date       : {latest_saved_date or 'None'}")
    print(f"Crawl window              : {start_date} -> {end_date}")
    print(f"Total tickers             : {len(STOCKS)}")

    fetched_df, failed_symbols = fetch_all_symbols(STOCKS, start_date, end_date)
    if fetched_df.empty:
        print("No rows were fetched. Nothing to write to parquet or push to MongoDB.")
        if failed_symbols:
            print(f"Failed symbols            : {', '.join(failed_symbols)}")
        return

    parquet_files = write_daily_parquet_partitions(fetched_df, PARQUET_ROOT)
    mongo_stats = push_to_mongo(fetched_df, mongo_uri)

    print("=" * 72)
    print("Pipeline finished")
    print("=" * 72)
    print(f"Fetched rows              : {len(fetched_df)}")
    print(f"Parquet files written     : {len(parquet_files)}")
    print(f"Mongo matched             : {mongo_stats['matched']}")
    print(f"Mongo modified            : {mongo_stats['modified']}")
    print(f"Mongo upserted            : {mongo_stats['upserted']}")
    if failed_symbols:
        print(f"Failed symbols            : {', '.join(failed_symbols)}")


if __name__ == "__main__":
    main()
