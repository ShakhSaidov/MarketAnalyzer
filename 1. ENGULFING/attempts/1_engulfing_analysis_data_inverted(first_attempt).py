import pandas as pd
import numpy as np
import itertools
import time
import sys
from numba import njit, prange

TRADE_COST_PTS = 0.5

@njit(parallel=True)
def compute_inner_numba(
    bull_base, bear_base,
    ex_bull_mfe, ex_bull_mae, ex_bear_mfe, ex_bear_mae,
    ex_bull_exit, ex_bear_exit,
    year_ids, n_years,
    tp_arr, sl_arr, fwd_idx_arr,
    n_inner
):
    N = len(bull_base)
    n_tp = len(tp_arr)
    n_sl = len(sl_arr)
    n_cols = 7 + n_years * 3
    out = np.zeros((n_inner, n_cols), dtype=np.float64)

    for ic in prange(n_inner):
        fi   = fwd_idx_arr[ic]
        tp   = tp_arr[ic % (n_tp * n_sl) // n_sl]
        sl_v = sl_arr[ic % n_sl]

        total_signals = np.float64(0)
        wins, losses, timeouts = np.float64(0), np.float64(0), np.float64(0)
        sum_mfe, sum_mae, sum_pnl = np.float64(0), np.float64(0), np.float64(0)

        y_wins     = np.zeros(n_years, dtype=np.float64)
        y_losses   = np.zeros(n_years, dtype=np.float64)
        y_timeouts = np.zeros(n_years, dtype=np.float64)

        for i in range(N):
            is_bull = bull_base[i]
            is_bear = bear_base[i]
            if not (is_bull or is_bear):
                continue

            total_signals += 1
            y = year_ids[i]

            if is_bull:
                mfe = ex_bull_mfe[fi, i]
                mae = ex_bull_mae[fi, i]
                pnl = ex_bull_exit[fi, i]
            else:
                mfe = ex_bear_mfe[fi, i]
                mae = ex_bear_mae[fi, i]
                pnl = ex_bear_exit[fi, i]

            is_win  = (mfe >= tp) and (mae < sl_v)
            is_loss = (mae >= sl_v)

            if is_win:
                wins += 1
                y_wins[y] += 1
            elif is_loss:
                losses += 1
                y_losses[y] += 1
            else:
                timeouts += 1
                sum_mfe  += mfe
                sum_mae  += mae
                sum_pnl  += pnl
                y_timeouts[y] += 1

        out[ic, 0] = total_signals
        out[ic, 1] = wins
        out[ic, 2] = losses
        out[ic, 3] = timeouts
        out[ic, 4] = sum_mfe
        out[ic, 5] = sum_mae
        out[ic, 6] = sum_pnl
        for y in range(n_years):
            out[ic, 7 + y*3 + 0] = y_wins[y]
            out[ic, 7 + y*3 + 1] = y_losses[y]
            out[ic, 7 + y*3 + 2] = y_timeouts[y]

    return out


def load_and_prep_data():
    print("1. Loading Parquet Data & Calculating Features...")
    df = pd.read_parquet("data/nq_1min.parquet")

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(np.float32)

    df.index = df.index.tz_localize('UTC').tz_convert('US/Pacific')
    df['year']   = df.index.year
    df['hour']   = df.index.hour
    df['minute'] = df.index.minute

    cond = [
        ((df['hour'] == 0) | (df['hour'] == 1)),
        ((df['hour'] == 6) & (df['minute'] >= 30)) | (df['hour'] == 7) | ((df['hour'] == 8) & (df['minute'] < 30)),
        ((df['hour'] == 8) & (df['minute'] >= 30)) | (df['hour'] == 9) | (df['hour'] == 10) | ((df['hour'] == 11) & (df['minute'] < 30)),
        ((df['hour'] == 11) & (df['minute'] >= 30)) | (df['hour'] == 12),
    ]
    df['session'] = np.select(cond, ['London', 'US_Open', 'Midday', 'US_Close'], default='Other')

    df['total_size']      = df['high'] - df['low']
    df['color']           = np.where(df['close'] >= df['open'], 1, -1)
    df['prev_open']       = df['open'].shift(1)
    df['prev_close']      = df['close'].shift(1)
    df['prev_high']       = df['high'].shift(1)
    df['prev_low']        = df['low'].shift(1)
    df['prev_color']      = df['color'].shift(1)
    df['prev_total_size'] = df['total_size'].shift(1)

    df['streak']      = df['color'].groupby((df['color'] != df['prev_color']).cumsum()).cumsum().abs()
    df['prev_streak'] = df['streak'].shift(1)
    df['sma_50']      = df['close'].rolling(window=50).mean()
    df['size_ratio']  = df['total_size'] / df['prev_total_size'].replace(0, 0.0001)

    df['prev_vol']       = df['volume'].shift(1).replace(0, 1)
    df['vol_multiplier'] = df['volume'] / df['prev_vol']

    print("2. Isolating Engulfing Candles...")
    bull_body = (df['prev_color'] == -1) & (df['color'] == 1)  & (df['close'] >= df['prev_open']) & (df['open'] <= df['prev_close'])
    bear_body = (df['prev_color'] == 1)  & (df['color'] == -1) & (df['close'] <= df['prev_open']) & (df['open'] >= df['prev_close'])
    bull_wick = (df['prev_color'] == -1) & (df['color'] == 1)  & (df['high'] > df['prev_high'])   & (df['low'] < df['prev_low'])
    bear_wick = (df['prev_color'] == 1)  & (df['color'] == -1) & (df['high'] > df['prev_high'])   & (df['low'] < df['prev_low'])

    df['is_bull_body'], df['is_bear_body'] = bull_body, bear_body
    df['is_bull_wick'], df['is_bear_wick'] = bull_wick, bear_wick

    signals_df = df[bull_body | bear_body | bull_wick | bear_wick].copy()
    print(f"   Signals found: {len(signals_df):,}")

    print("3. Calculating Future Outcomes...")
    for mins in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 60]:
        signals_df[f'fwd_max_{mins}']   = df['high'].rolling(window=mins).max().shift(-mins)[signals_df.index]
        signals_df[f'fwd_min_{mins}']   = df['low'].rolling(window=mins).min().shift(-mins)[signals_df.index]
        signals_df[f'fwd_close_{mins}'] = df['close'].shift(-mins)[signals_df.index]

    return signals_df


def run_fast_grid(df):
    param_grid = {
        'engulfing_type':    ['body', 'wick'],
        'min_streak':        [1, 2, 3, 4, 5],
        'size_multi':        [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.5, 3.0],
        'trend_filter':      ['All', 'With_Trend'],
        'vol_filter':        ['All', '1.2x', '1.5x', '2.0x'],
        'session':           ['All', 'US_Open', 'London', 'US_Close'],
        'look_forward_mins': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 60],
        'tp_pts':            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        'sl_pts':            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    }

    fwd_windows = param_grid['look_forward_mins']
    tp_list     = param_grid['tp_pts']
    sl_list     = param_grid['sl_pts']
    n_fwd       = len(fwd_windows)   # 15

    print("\nPrecomputing INVERSE masks and excursions...")

    d            = {col: df[col].values for col in df.columns}
    unique_years = np.sort(df['year'].unique())
    n_years      = len(unique_years)
    year_to_idx  = {y: i for i, y in enumerate(unique_years)}
    N            = len(df)
    entry_price  = d['close'].astype(np.float32)

    base_masks = {
        ('body', 'bull'): d['is_bull_body'].astype(bool),
        ('body', 'bear'): d['is_bear_body'].astype(bool),
        ('wick', 'bull'): d['is_bull_wick'].astype(bool),
        ('wick', 'bear'): d['is_bear_wick'].astype(bool),
    }

    streak_masks  = {s: (d['prev_streak'] >= s) for s in param_grid['min_streak']}
    size_masks    = {m: (d['size_ratio']   >= m) for m in param_grid['size_multi']}
    session_masks = {'All': np.ones(N, dtype=bool)}
    for s in ['US_Open', 'London', 'US_Close']:
        session_masks[s] = (d['session'] == s)
    vol_masks = {
        'All':  np.ones(N, dtype=bool),
        '1.2x': d['vol_multiplier'] >= 1.2,
        '1.5x': d['vol_multiplier'] >= 1.5,
        '2.0x': d['vol_multiplier'] >= 2.0,
    }
    sma_bull_mask = d['close'] > d['sma_50']
    sma_bear_mask = d['close'] < d['sma_50']
    year_ids      = np.array([year_to_idx[y] for y in d['year']], dtype=np.int32)

    # Excursion arrays: shape (n_fwd, N) — INVERSE direction
    ex_bull_mfe  = np.zeros((n_fwd, N), dtype=np.float32)
    ex_bull_mae  = np.zeros((n_fwd, N), dtype=np.float32)
    ex_bear_mfe  = np.zeros((n_fwd, N), dtype=np.float32)
    ex_bear_mae  = np.zeros((n_fwd, N), dtype=np.float32)
    ex_bull_exit = np.zeros((n_fwd, N), dtype=np.float32)
    ex_bear_exit = np.zeros((n_fwd, N), dtype=np.float32)

    for fi, mins in enumerate(fwd_windows):
        f_max   = d[f'fwd_max_{mins}'].astype(np.float32)
        f_min   = d[f'fwd_min_{mins}'].astype(np.float32)
        f_close = d[f'fwd_close_{mins}'].astype(np.float32)
        # Bull Engulfing -> SHORT: profit = down move, risk = up move
        ex_bull_mfe[fi]  = entry_price - f_min
        ex_bull_mae[fi]  = f_max - entry_price
        ex_bull_exit[fi] = entry_price - f_close
        # Bear Engulfing -> LONG: profit = up move, risk = down move
        ex_bear_mfe[fi]  = f_max - entry_price
        ex_bear_mae[fi]  = entry_price - f_min
        ex_bear_exit[fi] = f_close - entry_price

    # Inner combo arrays
    inner_combos = list(itertools.product(range(n_fwd), range(len(tp_list)), range(len(sl_list))))
    n_inner      = len(inner_combos)
    fwd_idx_arr  = np.array([c[0] for c in inner_combos], dtype=np.int32)
    tp_arr       = np.array(tp_list, dtype=np.float32)
    sl_arr       = np.array(sl_list, dtype=np.float32)
    inner_params = [(fwd_windows[c[0]], tp_list[c[1]], sl_list[c[2]]) for c in inner_combos]

    outer_keys   = ['engulfing_type', 'min_streak', 'size_multi', 'trend_filter', 'vol_filter', 'session']
    outer_combos = list(itertools.product(*[param_grid[k] for k in outer_keys]))
    total_outer  = len(outer_combos)
    total_combos = total_outer * n_inner

    print(f"Total combinations: {total_combos:,}  ({total_outer} outer x {n_inner} inner)")

    # -----------------------------------------------------------------------
    # NUMBA WARMUP — dummy arrays must match actual n_fwd (15) not hardcoded 4
    # This was the crash cause in the previous version
    # -----------------------------------------------------------------------
    print("\nCompiling numba JIT function (one-time, ~60-90 seconds)...")
    sys.stdout.flush()
    _dummy = np.zeros(10, dtype=bool)
    _dex   = np.zeros((n_fwd, 10), dtype=np.float32)   # <-- n_fwd, not 4
    _dy    = np.zeros(10, dtype=np.int32)
    _fwd   = np.array([0] * n_inner, dtype=np.int32)
    _tp    = np.array(tp_list, dtype=np.float32)
    _sl    = np.array(sl_list, dtype=np.float32)
    compute_inner_numba(
        _dummy, _dummy,
        _dex, _dex, _dex, _dex, _dex, _dex,
        _dy, n_years, _tp, _sl, _fwd, n_inner
    )
    print("Compilation done. Starting INVERSE grid search...\n")
    print(f"{'Outer':>6}  {'Results':>8}  {'Elapsed':>8}  {'ETA':>10}  {'Rate/s':>8}")
    print("-" * 55)
    sys.stdout.flush()

    results    = []
    start_time = time.time()
    last_print = start_time
    outer_done = 0

    for outer in outer_combos:
        op = dict(zip(outer_keys, outer))

        ctx = (streak_masks[op['min_streak']]
               & size_masks[op['size_multi']]
               & session_masks[op['session']]
               & vol_masks[op['vol_filter']])

        bull_base = base_masks[(op['engulfing_type'], 'bull')] & ctx
        bear_base = base_masks[(op['engulfing_type'], 'bear')] & ctx

        if op['trend_filter'] == 'With_Trend':
            bull_base = bull_base & sma_bull_mask
            bear_base = bear_base & sma_bear_mask

        total_signals = int(bull_base.sum() + bear_base.sum())
        outer_done   += 1

        if total_signals < 50:
            now = time.time()
            if now - last_print >= 10:
                elapsed = now - start_time
                rate    = len(results) / elapsed if elapsed > 0 else 0
                pct     = outer_done / total_outer
                eta     = ((elapsed / pct) - elapsed) if pct > 0 else 0
                print(f"{outer_done:>6}/{total_outer}  {len(results):>8,}  {elapsed:>7.0f}s  {eta/60:>8.1f}min  {rate:>7.0f}/s")
                sys.stdout.flush()
                last_print = now
            continue

        out = compute_inner_numba(
            bull_base.astype(np.bool_), bear_base.astype(np.bool_),
            ex_bull_mfe, ex_bull_mae, ex_bear_mfe, ex_bear_mae,
            ex_bull_exit, ex_bear_exit,
            year_ids, n_years, tp_arr, sl_arr, fwd_idx_arr, n_inner
        )

        for ic in range(n_inner):
            mins, tp, sl = inner_params[ic]
            ts       = int(out[ic, 0])
            wins     = int(out[ic, 1])
            losses   = int(out[ic, 2])
            timeouts = int(out[ic, 3])

            if ts < 50:
                continue

            win_rate     = wins     / ts
            loss_rate    = losses   / ts
            timeout_rate = timeouts / ts
            avg_to_pnl   = (out[ic, 6] / timeouts) if timeouts > 0 else 0.0

            real_tp     = tp  - TRADE_COST_PTS
            real_sl     = sl  + TRADE_COST_PTS
            real_to_pnl = avg_to_pnl - TRADE_COST_PTS
            true_ev     = (win_rate * real_tp) - (loss_rate * real_sl) + (timeout_rate * real_to_pnl)

            row = {
                'Engulf Type': op['engulfing_type'],
                'Session':     op['session'],
                'Trend':       op['trend_filter'],
                'Vol Filter':  op['vol_filter'],
                'Streak':      op['min_streak'],
                'Size Mult':   op['size_multi'],
                'Hold Mins':   mins,
                'TP Pts':      tp,
                'SL Pts':      sl,
                'Total Trades':            ts,
                'INV Win%':                round(win_rate  * 100, 1),
                'INV Loss%':               round(loss_rate * 100, 1),
                'Real EV (Costs Deducted)':round(true_ev, 3),
            }

            for yi, y in enumerate(unique_years):
                y_col   = 7 + yi * 3
                y_wins  = int(out[ic, y_col + 0])
                y_loss  = int(out[ic, y_col + 1])
                y_to    = int(out[ic, y_col + 2])
                y_total = y_wins + y_loss + y_to
                if y_total > 0:
                    y_win_r  = y_wins / y_total
                    y_loss_r = y_loss / y_total
                    y_to_r   = y_to   / y_total
                    y_ev     = (y_win_r * real_tp) - (y_loss_r * real_sl) + (y_to_r * real_to_pnl)
                    row[f'{y} Trades'] = y_total
                    row[f'{y} Win%']   = round(y_win_r * 100, 1)
                    row[f'{y} Real EV']= round(y_ev, 2)
                else:
                    row[f'{y} Trades'] = 0
                    row[f'{y} Win%']   = 0.0
                    row[f'{y} Real EV']= 0.0

            results.append(row)

        # Progress every 10 seconds
        now = time.time()
        if now - last_print >= 10:
            elapsed = now - start_time
            rate    = len(results) / elapsed if elapsed > 0 else 0
            pct     = outer_done / total_outer
            eta     = ((elapsed / pct) - elapsed) if pct > 0 else 0
            print(f"{outer_done:>6}/{total_outer}  {len(results):>8,}  {elapsed:>7.0f}s  {eta/60:>8.1f}min  {rate:>7.0f}/s")
            sys.stdout.flush()
            last_print = now

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f} min).")

    if not results:
        print("No combinations met the minimum trade criteria.")
        return

    res_df = pd.DataFrame(results)
    res_df = res_df.sort_values(by='Real EV (Costs Deducted)', ascending=False).reset_index(drop=True)
    res_df.to_parquet("INVERSE_master_results.parquet")
    print(f"Saved {len(res_df):,} rows to INVERSE_master_results.parquet")


if __name__ == "__main__":
    signals_df = load_and_prep_data()
    run_fast_grid(signals_df)