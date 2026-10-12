import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
BASE_FILE = os.path.join(DATA_DIR, "nq_1min.parquet")

# Timeframes to check and their standard resample rules
CHECK_TIMEFRAMES = {
    "5min": "5min",
    "10min": "10min",
    "15min": "15min",
    "30min": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
}

AGG_RULES = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}


def load_and_clean(filepath: str) -> pd.DataFrame:
    df = pd.read_parquet(filepath, engine="pyarrow")

    # Standardize index to datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        for col in ["time", "datetime", "date"]:
            if col in df.columns:
                df = df.set_index(col)
                break

    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = df.index.astype("datetime64[ns]")
    df.index.name = "time"
    df.columns = [c.lower() for c in df.columns]

    return df[["open", "high", "low", "close", "volume"]].sort_index()


def verify_all():
    if not os.path.exists(BASE_FILE):
        print(f"❌ Error: Ground truth file missing at {BASE_FILE}")
        return

    print("=" * 70)
    print("🔍 LOADING 1-MINUTE GROUND TRUTH...")
    print("=" * 70)
    df_1m = load_and_clean(BASE_FILE)
    print(f"✓ Loaded {len(df_1m):,} rows ({df_1m.index[0]} → {df_1m.index[-1]})\n")

    overall_passed = True

    for tf_label, rule in CHECK_TIMEFRAMES.items():
        filename = f"nq_{tf_label}.parquet"
        filepath = os.path.join(DATA_DIR, filename)

        print(f"--- Testing '{filename}' (Rule: {rule}) ---")

        if not os.path.exists(filepath):
            print(f"  ⚠️  FILE MISSING: {filename} not found in {DATA_DIR}\n")
            overall_passed = False
            continue

        try:
            df_actual = load_and_clean(filepath)
        except Exception as e:
            print(f"  ❌ FAILED to read parquet file: {e}\n")
            overall_passed = False
            continue

        # ── Step 1: Generate Ground Truth from 1-min ──────────────────────────
        df_expected = (
            df_1m.resample(rule, closed="left", label="left")
            .agg(AGG_RULES)
            .dropna()
        )

        # ── Step 2: Compare Timestamps ────────────────────────────────────────
        missing_in_actual = df_expected.index.difference(df_actual.index)
        extra_in_actual = df_actual.index.difference(df_expected.index)

        has_time_mismatch = False
        if len(missing_in_actual) > 0:
            print(f"  ❌ Timestamp Mismatch: Missing {len(missing_in_actual)} candles in file.")
            print(f"     Example missing: {missing_in_actual[:3].tolist()}")
            has_time_mismatch = True
        if len(extra_in_actual) > 0:
            print(f"  ❌ Timestamp Mismatch: Found {len(extra_in_actual)} unexpected extra candles.")
            print(f"     Example extra: {extra_in_actual[:3].tolist()}")
            has_time_mismatch = True

        if has_time_mismatch:
            overall_passed = False
            print()
            continue

        # ── Step 3: Compare OHLCV Values ──────────────────────────────────────
        # Align rows on shared timestamps
        df_act_aligned = df_actual.loc[df_expected.index]

        errors = []
        for col in ["open", "high", "low", "close"]:
            diff = np.abs(df_expected[col].values - df_act_aligned[col].values)
            max_diff = np.max(diff)
            # Allow small float32 precision discrepancy (0.01)
            if max_diff > 0.01:
                bad_idx = np.where(diff > 0.01)[0][0]
                bad_time = df_expected.index[bad_idx]
                errors.append(
                    f"'{col}' mismatch! Max diff: {max_diff:.4f} "
                    f"(e.g., at {bad_time}: Expected {df_expected[col].iloc[bad_idx]}, Got {df_act_aligned[col].iloc[bad_idx]})"
                )

        # Compare Volume
        vol_diff = np.abs(df_expected["volume"].values - df_act_aligned["volume"].values)
        if np.max(vol_diff) > 0:
            bad_idx = np.where(vol_diff > 0)[0][0]
            bad_time = df_expected.index[bad_idx]
            errors.append(
                f"'volume' mismatch! (e.g., at {bad_time}: Expected {df_expected['volume'].iloc[bad_idx]}, Got {df_act_aligned['volume'].iloc[bad_idx]})"
            )

        if errors:
            print(f"  ❌ Numerical Discrepancy:")
            for err in errors:
                print(f"     • {err}")
            overall_passed = False
        else:
            print(f"  ✅ PASSED: {len(df_actual):,} bars matched 100% with 1-min aggregation.")

        print()

    print("=" * 70)
    if overall_passed:
        print("🎉 ALL FILES ARE 100% VALID AND SYNCHRONIZED WITH 1-MIN DATA.")
    else:
        print("⚠️  SOME FILES FAILED VERIFICATION. Check output details above.")
    print("=" * 70)


if __name__ == "__main__":
    verify_all()