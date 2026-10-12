"""
preprocess.py  –  Run this ONCE before launching the chart app.
Converts nq_final.csv → compressed Parquet files for all timeframes.
Place this file in the same folder as nq_final.csv.

Usage:
    python preprocess.py
"""

import os
import sys
import time
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CSV_PATH  = os.path.join(BASE_DIR, "nq_final.csv")
DATA_DIR  = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ── Timeframes to pre-aggregate ───────────────────────────────────────────────
TIMEFRAMES = {
    # "1min":  "1min",
    # "5min":  "5min",
    "10min": "10min",
    # "15min": "15min",
    # "30min": "30min",
    # "1h":    "1h",
    # "4h":    "4h",
    # "1d":    "1D",
}

OHLCV_AGG = {
    "open":   "first",
    "high":   "max",
    "low":    "min",
    "close":  "last",
    "volume": "sum",
}


def progress(msg: str) -> None:
    print(f"  {msg}", flush=True)


def load_csv() -> pd.DataFrame:
    progress(f"Reading {CSV_PATH} …")
    t0 = time.time()

    df = pd.read_csv(
        CSV_PATH,
        parse_dates=["datetime"],
        index_col="datetime",
    )

    # Normalise column names
    df.columns = [c.strip().lower() for c in df.columns]

    # Drop symbol column – not needed for charting
    if "symbol" in df.columns:
        df.drop(columns=["symbol"], inplace=True)

    # Ensure UTC-aware index, then convert to UTC naive for simpler handling
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    else:
        df.index = pd.to_datetime(df.index)

    df.sort_index(inplace=True)
    progress(f"Loaded {len(df):,} rows in {time.time()-t0:.1f}s")
    return df


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    resampled = df.resample(rule).agg(OHLCV_AGG).dropna(subset=["close"])
    return resampled


def save_parquet(df: pd.DataFrame, name: str) -> None:
    path = os.path.join(DATA_DIR, f"nq_{name}.parquet")
    df.to_parquet(path, engine="pyarrow", compression="snappy")
    size_mb = os.path.getsize(path) / 1_048_576
    progress(f"Saved  nq_{name}.parquet  ({len(df):,} rows, {size_mb:.1f} MB)")


def main() -> None:
    print("\n=== NQ Chart Preprocessor ===\n")

    if not os.path.exists(CSV_PATH):
        print(f"ERROR: Cannot find {CSV_PATH}")
        print("Make sure nq_final.csv is in the same folder as this script.")
        sys.exit(1)

    df_1min = load_csv()

    print("\nBuilding timeframe parquet files …")
    for name, rule in TIMEFRAMES.items():
        t0 = time.time()
        if name == "1min":
            df_tf = df_1min.copy()
        else:
            df_tf = resample(df_1min, rule)
        save_parquet(df_tf, name)
        progress(f"  └ done in {time.time()-t0:.2f}s")

    print("\n✓ All done! You can now run:  python app.py\n")


if __name__ == "__main__":
    main()