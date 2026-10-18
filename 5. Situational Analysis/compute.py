import os
import pandas as pd
import numpy as np
import time
from datetime import timedelta
from tqdm import tqdm

DATA_PATH = "../data/nq_1min.parquet"
OUTPUT_PATH = "./raw_trades.parquet"


def get_signal_bars(df_rth, timeframe_mins, nth_bar):
    tf_str = f"{timeframe_mins}min"

    df_resampled = df_rth.resample(
        tf_str,
        closed="left",
        label="left"
    ).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    }).dropna()

    df_resampled["sig_color"] = np.where(
        df_resampled["close"] > df_resampled["open"],
        "Green",
        "Red"
    )

    return df_resampled.groupby(
        df_resampled.index.date
    ).nth(nth_bar - 1)


def main():
    print("🚀 Starting Matrix Pre-Computation...")
    start_time = time.time()

    # 1. Load Data
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Data not found at {DATA_PATH}")
        return

    print("📂 Loading parquet data...")

    df = pd.read_parquet(DATA_PATH)

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df.index = df.index.tz_convert("US/Eastern")

    df_rth = df.between_time("08:30", "15:59")

    daily_stats = df_rth.resample("D").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    }).dropna()

    daily_stats.index = daily_stats.index.date

    daily_stats["prev_close"] = daily_stats["close"].shift(1)
    daily_stats["prev_open"] = daily_stats["open"].shift(1)

    daily_stats["prev_color"] = np.where(
        daily_stats["prev_close"] > daily_stats["prev_open"],
        "Green",
        "Red"
    )

    day_map = {
        0: "Mon",
        1: "Tue",
        2: "Wed",
        3: "Thu",
        4: "Fri"
    }

    daily_stats["dow_raw"] = (
        pd.DatetimeIndex(daily_stats.index)
        .dayofweek
        .map(day_map)
    )

    daily_1m_groups = {
        d: g for d, g in df_rth.groupby(df_rth.index.date)
    }

    # Parameter Grids
    TFs = [1, 5, 10, 15, 30]
    Nths = [1, 2, 3, 4, 5]
    Triggers = [
        "Long Breakout",
        "Short Breakdown",
        "Fade Breakout",
        "Fade Breakdown"
    ]
    SLs = [10.0, 15.0, 20.0, 30.0]
    Ses_Hrs = [1.0, 2.0, 4.0, 6.5]

    results = []

    total_combinations = (
        len(TFs)
        * len(Nths)
        * len(Triggers)
    )

    print(
        f"⏳ Crunching all combinations across history "
        f"({total_combinations:,} combinations)..."
    )

    # Progress bar
    progress = tqdm(
        total=total_combinations,
        desc="Processing",
        unit="combo",
        dynamic_ncols=True
    )

    # Highly Optimized Loop
    for tf in TFs:

        for nth in Nths:

            sig_bars = get_signal_bars(
                df_rth,
                tf,
                nth
            )

            for trigger in Triggers:

                is_long = (
                    "Long" in trigger
                    or trigger == "Fade Breakdown"
                )

                for timestamp_obj, sig_bar in sig_bars.iterrows():

                    date_obj = timestamp_obj.date()

                    if date_obj not in daily_stats.index:
                        continue

                    day_1m = daily_1m_groups.get(date_obj)

                    if day_1m is None:
                        continue

                    sig_end_time = (
                        timestamp_obj
                        + timedelta(minutes=tf)
                    )

                    post_sig = day_1m[
                        day_1m.index >= sig_end_time
                    ]

                    if post_sig.empty:
                        continue

                    # Extract arrays for C-level speed
                    highs = post_sig["high"].values
                    lows = post_sig["low"].values
                    opens = post_sig["open"].values
                    closes = post_sig["close"].values
                    times = post_sig.index

                    entry_idx, entry_price = -1, 0.0

                    if trigger in [
                        "Long Breakout",
                        "Fade Breakout"
                    ]:
                        mask = highs > sig_bar["high"]

                        if not mask.any():
                            continue

                        entry_idx = mask.argmax()

                        entry_price = max(
                            sig_bar["high"],
                            opens[entry_idx]
                        )

                    else:
                        mask = lows < sig_bar["low"]

                        if not mask.any():
                            continue

                        entry_idx = mask.argmax()

                        entry_price = min(
                            sig_bar["low"],
                            opens[entry_idx]
                        )

                    entry_time = times[entry_idx]

                    session_start = pd.Timestamp(
                        f"{date_obj} 08:30:00"
                    ).tz_localize("US/Eastern")

                    # Once entry is found, check all
                    # Session Hours and SLs simultaneously
                    for s_hr in Ses_Hrs:

                        sesh_end = (
                            session_start
                            + timedelta(hours=s_hr)
                        )

                        if entry_time >= sesh_end:
                            continue

                        # Slice data up to session end
                        end_idx = times.searchsorted(
                            sesh_end,
                            side="right"
                        )

                        s_highs = highs[
                            entry_idx:end_idx
                        ]

                        s_lows = lows[
                            entry_idx:end_idx
                        ]

                        s_close = (
                            closes[end_idx - 1]
                            if end_idx > 0
                            else closes[-1]
                        )

                        d_stat = daily_stats.loc[date_obj]
                        yr = timestamp_obj.year

                        for sl in SLs:

                            if is_long:

                                sl_mask = (
                                    s_lows
                                    <= (entry_price - sl)
                                )

                                if sl_mask.any():

                                    sl_idx = sl_mask.argmax()

                                    mfe = (
                                        s_highs[:sl_idx + 1].max()
                                        - entry_price
                                    )

                                    sl_hit = True

                                else:

                                    mfe = (
                                        s_highs.max()
                                        - entry_price
                                    )

                                    sl_hit = False

                                pnl = (
                                    s_close
                                    - entry_price
                                )

                            else:

                                sl_mask = (
                                    s_highs
                                    >= (entry_price + sl)
                                )

                                if sl_mask.any():

                                    sl_idx = sl_mask.argmax()

                                    mfe = (
                                        entry_price
                                        - s_lows[:sl_idx + 1].min()
                                    )

                                    sl_hit = True

                                else:

                                    mfe = (
                                        entry_price
                                        - s_lows.min()
                                    )

                                    sl_hit = False

                                pnl = (
                                    entry_price
                                    - s_close
                                )

                            results.append((
                                date_obj,
                                yr,
                                d_stat["dow_raw"],
                                d_stat["prev_color"],
                                sig_bar["sig_color"],
                                tf,
                                nth,
                                trigger,
                                sl,
                                s_hr,
                                sl_hit,
                                round(mfe, 2),
                                round(pnl, 2)
                            ))

                # Update progress after this TF/Nth/Trigger combination
                progress.update(1)

    progress.close()

    cols = [
        "Date",
        "Year",
        "DOW_Raw",
        "Prev_Color",
        "Sig_Color",
        "TF",
        "Nth",
        "Trigger",
        "SL_Set",
        "Sesh_Hrs",
        "SL_Hit",
        "MFE",
        "Session_PnL"
    ]

    print("\n💾 Saving combinations to Parquet...")

    final_df = pd.DataFrame(
        results,
        columns=cols
    )

    final_df.to_parquet(
        OUTPUT_PATH,
        index=False
    )

    elapsed = round(
        time.time() - start_time,
        2
    )

    print(
        f"✅ Success! Computed "
        f"{len(final_df):,} scenario outcomes "
        f"in {elapsed} seconds."
    )

    print(
        f"Output saved to: {OUTPUT_PATH}"
    )

    print(
        "You may now run `streamlit run app.py`."
    )


if __name__ == "__main__":
    main()