"""
NQ Engulfing Candle Grid Search — STRICT CHRONOLOGICAL (INVERSE Direction)
"""

import pandas as pd
import numpy as np
import itertools
import time
import sys
from numba import njit, prange

TRADE_COST_PTS = 0.5  # $10 round trip for spread/slippage/commissions

# ---------------------------------------------------------------------------
# NUMBA KERNEL — chronological INVERSE candle-by-candle exit simulation
# ---------------------------------------------------------------------------

@njit(parallel=True)
def compute_inverse_inner_numba(
    fwd_high, fwd_low, fwd_close,
    entry, bull_base, bear_base,
    year_ids, n_years,
    tp_arr, sl_arr, hold_arr,
    n_inner
):
    N      = len(bull_base)
    n_tp   = len(tp_arr)
    n_sl   = len(sl_arr)
    n_hold = len(hold_arr)
    n_cols = 7 + n_years * 3
    out    = np.zeros((n_inner, n_cols), dtype=np.float64)

    for ic in prange(n_inner):
        hold_idx = ic // (n_tp * n_sl)
        tp_idx   = (ic // n_sl) % n_tp
        sl_idx   = ic % n_sl

        hold = hold_arr[hold_idx]
        tp   = tp_arr[tp_idx]
        sl_v = sl_arr[sl_idx]

        total_signals = np.float64(0)
        wins          = np.float64(0)
        losses        = np.float64(0)
        timeouts      = np.float64(0)
        sum_mfe       = np.float64(0)
        sum_mae       = np.float64(0)
        sum_pnl       = np.float64(0)

        y_wins     = np.zeros(n_years, dtype=np.float64)
        y_losses   = np.zeros(n_years, dtype=np.float64)
        y_timeouts = np.zeros(n_years, dtype=np.float64)

        for i in range(N):
            is_bull = bull_base[i]
            is_bear = bear_base[i]
            if not (is_bull or is_bear):
                continue

            total_signals += 1
            y  = year_ids[i]
            ep = entry[i]
            
            # 🚨 INVERSE TARGET LEVELS 🚨
            # Bull Signal -> SHORT -> TP is lower, SL is higher
            # Bear Signal -> LONG  -> TP is higher, SL is lower
            if is_bull:
                tp_lvl = ep - tp
                sl_lvl = ep + sl_v
            else:
                tp_lvl = ep + tp
                sl_lvl = ep - sl_v

            outcome  = 0    
            exit_pnl = np.float64(0)
            mfe      = np.float64(0)
            mae      = np.float64(0)

            # Chronological Walk-Forward
            for c in range(hold):
                h = fwd_high[i, c]
                l = fwd_low[i, c]

                # 🚨 INVERSE MFE/MAE TRACKING 🚨
                if is_bull: # Short
                    curr_mfe = ep - l
                    curr_mae = h - ep
                else: # Long
                    curr_mfe = h - ep
                    curr_mae = ep - l

                if curr_mfe > mfe: mfe = curr_mfe
                if curr_mae > mae: mae = curr_mae

                # 🚨 INVERSE HIT DETECTION 🚨
                if is_bull: # Short
                    tp_hit = (l <= tp_lvl)
                    sl_hit = (h >= sl_lvl)
                else: # Long
                    tp_hit = (h >= tp_lvl)
                    sl_hit = (l <= sl_lvl)

                if tp_hit and not sl_hit:
                    outcome  = 1
                    exit_pnl = tp
                    break
                elif sl_hit and not tp_hit:
                    outcome  = -1
                    exit_pnl = -sl_v
                    break
                elif tp_hit and sl_hit:
                    # Both hit in same minute -> conservative loss
                    outcome  = -1
                    exit_pnl = -sl_v
                    break

            if outcome == 0:
                # Timeout — exit at close of last candle (Inverse logic)
                if is_bull: # Short
                    exit_pnl = ep - fwd_close[i, hold - 1]
                else: # Long
                    exit_pnl = fwd_close[i, hold - 1] - ep

            if outcome == 1:
                wins += 1
                y_wins[y] += 1
            elif outcome == -1:
                losses += 1
                y_losses[y] += 1
            else:
                timeouts += 1
                sum_mfe  += mfe
                sum_mae  += mae
                sum_pnl  += exit_pnl
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

# ---------------------------------------------------------------------------
# Data loading (Identical to your verified Script A)
# ---------------------------------------------------------------------------

def load_and_prep_data():
    print("1. Loading Parquet Data & Calculating Features...")
    df = pd.read_parquet("../data/nq_1min.parquet")

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(np.float32)

    df.index = df.index.tz_localize('UTC').tz_convert('US/Pacific')
    df['year']   = df.index.year
    df['hour']   = df.index.hour
    df['minute'] = df.index.minute

    conditions = [
        ((df['hour'] == 0) | (df['hour'] == 1)),
        ((df['hour'] == 6) & (df['minute'] >= 30)) | (df['hour'] == 7) | ((df['hour'] == 8) & (df['minute'] < 30)),
        ((df['hour'] == 8) & (df['minute'] >= 30)) | (df['hour'] == 9) | (df['hour'] == 10) | ((df['hour'] == 11) & (df['minute'] < 30)),
        ((df['hour'] == 11) & (df['minute'] >= 30)) | (df['hour'] == 12),
    ]
    df['session'] = np.select(conditions, ['London', 'US_Open', 'Midday', 'US_Close'], default='Other')

    df['total_size']      = (df['high'] - df['low'])
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
    df['is_bull_wick'],  df['is_bear_wick']  = bull_wick, bear_wick

    signals_df = df[bull_body | bear_body | bull_wick | bear_wick].copy()
    
    max_hold = 60
    highs  = df['high'].values.astype(np.float32)
    lows   = df['low'].values.astype(np.float32)
    closes = df['close'].values.astype(np.float32)

    signal_positions = df.index.get_indexer(signals_df.index)
    n_total = len(highs)
    valid_signals = signal_positions < (n_total - max_hold)
    
    signals_df = signals_df[valid_signals].copy()
    signal_positions = signal_positions[valid_signals]
    n_signals = len(signal_positions)

    fwd_high  = np.full((n_signals, max_hold), np.nan, dtype=np.float32)
    fwd_low   = np.full((n_signals, max_hold), np.nan, dtype=np.float32)
    fwd_close = np.full((n_signals, max_hold), np.nan, dtype=np.float32)

    for ci in range(max_hold):
        target_pos = signal_positions + ci + 1
        fwd_high[:,  ci] = highs[target_pos]
        fwd_low[:,   ci] = lows[target_pos]
        fwd_close[:, ci] = closes[target_pos]

    return signals_df, fwd_high, fwd_low, fwd_close

# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def run_fast_grid(signals_df, fwd_high, fwd_low, fwd_close):
    param_grid = {
        'engulfing_type':    ['body', 'wick'],
        'min_streak':        [1, 2, 3, 4, 5],
        'size_multi':        [1.1, 1.5, 2.0, 2.5, 3.0],
        'trend_filter':      ['All'],
        'vol_filter':        ['All', '1.5x', '2.0x'],
        'session':           ['All', 'US_Open', 'London', 'US_Close'],
        'look_forward_mins': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 60],
        'tp_pts':            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        'sl_pts':            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
    }

    tp_list   = param_grid['tp_pts']
    sl_list   = param_grid['sl_pts']
    hold_list = param_grid['look_forward_mins']
    n_tp      = len(tp_list)
    n_sl      = len(sl_list)
    n_hold    = len(hold_list)
    n_inner   = n_hold * n_tp * n_sl   

    print("\nPrecomputing all masks...")
    df = signals_df
    d  = {col: df[col].values for col in df.columns}

    unique_years = np.sort(df['year'].unique())
    n_years      = len(unique_years)
    year_to_idx  = {y: i for i, y in enumerate(unique_years)}
    N            = len(df)

    entry_price = d['close'].astype(np.float32)
    year_ids    = np.array([year_to_idx[y] for y in d['year']], dtype=np.int32)

    base_masks = {
        ('body', 'bull'): d['is_bull_body'].astype(bool),
        ('body', 'bear'): d['is_bear_body'].astype(bool),
        ('wick', 'bull'): d['is_bull_wick'].astype(bool),
        ('wick', 'bear'): d['is_bear_wick'].astype(bool),
    }

    streak_masks  = {s: (d['prev_streak'] >= s) for s in param_grid['min_streak']}
    size_masks    = {m: (d['size_ratio']   >= m) for m in param_grid['size_multi']}
    session_masks = {'All': np.ones(N, dtype=bool)}
    for s in ['US_Open', 'London', 'US_Close']: session_masks[s] = (d['session'] == s)
    vol_masks = {
        'All':  np.ones(N, dtype=bool),
        '1.5x': d['vol_multiplier'] >= 1.5,
        '2.0x': d['vol_multiplier'] >= 2.0,
    }
    sma_bull_mask = d['close'] > d['sma_50']
    sma_bear_mask = d['close'] < d['sma_50']

    tp_arr   = np.array(tp_list,   dtype=np.float32)
    sl_arr   = np.array(sl_list,   dtype=np.float32)
    hold_arr = np.array(hold_list, dtype=np.int32)

    inner_combos = list(itertools.product(range(n_hold), range(n_tp), range(n_sl)))
    inner_params = [(hold_list[c[0]], tp_list[c[1]], sl_list[c[2]]) for c in inner_combos]

    outer_keys   = ['engulfing_type', 'min_streak', 'size_multi', 'trend_filter', 'vol_filter', 'session']
    outer_combos = list(itertools.product(*[param_grid[k] for k in outer_keys]))
    total_outer  = len(outer_combos)

    print(f"Total combinations: {total_outer * n_inner:,}  ({total_outer} outer x {n_inner} inner)")

    # Numba warmup 
    print("\nCompiling INVERSE numba JIT function (one-time, ~60-90 seconds)...")
    sys.stdout.flush()
    _n = 5
    _fh = np.ones((_n, 60), dtype=np.float32) * 101
    _fl = np.ones((_n, 60), dtype=np.float32) * 99
    _fc = np.ones((_n, 60), dtype=np.float32) * 100
    _ep = np.ones(_n, dtype=np.float32) * 100
    _bb = np.ones(_n, dtype=bool)
    _bf = np.zeros(_n, dtype=bool)
    _yi = np.zeros(_n, dtype=np.int32)
    compute_inverse_inner_numba(_fh, _fl, _fc, _ep, _bb, _bf, _yi, 2, tp_arr, sl_arr, hold_arr, n_inner)
    
    print("Compilation done. Starting INVERSE grid search...\n")
    print(f"{'Outer':>6}  {'Results':>8}  {'Elapsed':>8}  {'ETA':>10}  {'Rate/s':>8}")
    print("-" * 55)

    results    = []
    start_time = time.time()
    last_print = start_time
    outer_done = 0

    for outer in outer_combos:
        op = dict(zip(outer_keys, outer))

        ctx = (streak_masks[op['min_streak']] & size_masks[op['size_multi']] & session_masks[op['session']] & vol_masks[op['vol_filter']])
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
                print(f"{outer_done:>6}/{total_outer}  {len(results):>8,}  {elapsed:>7.0f}s  {(((elapsed/pct)-elapsed)/60):>8.1f}min  {rate:>7.0f}/s")
                sys.stdout.flush()
                last_print = now
            continue

        out = compute_inverse_inner_numba(
            fwd_high, fwd_low, fwd_close,
            entry_price,
            bull_base.astype(np.bool_),
            bear_base.astype(np.bool_),
            year_ids, n_years,
            tp_arr, sl_arr, hold_arr,
            n_inner
        )

        for ic in range(n_inner):
            mins, tp, sl = inner_params[ic]
            ts       = int(out[ic, 0])
            if ts < 50: continue

            wins, losses, timeouts = int(out[ic, 1]), int(out[ic, 2]), int(out[ic, 3])

            win_rate     = wins     / ts
            loss_rate    = losses   / ts
            timeout_rate = timeouts / ts

            if timeouts > 0:
                avg_to_mfe    = out[ic, 4] / timeouts
                avg_to_mae    = out[ic, 5] / timeouts
                avg_to_pnl    = out[ic, 6] / timeouts
                avg_missed_tp = tp - avg_to_mfe
            else:
                avg_to_mfe = avg_to_mae = avg_to_pnl = avg_missed_tp = 0.0

            # 🚨 INVERSE EXPECTED VALUE (WITH TRADING COSTS) 🚨
            real_tp = tp - TRADE_COST_PTS
            real_sl = sl + TRADE_COST_PTS
            real_to = avg_to_pnl - TRADE_COST_PTS
            true_ev = (win_rate * real_tp) - (loss_rate * real_sl) + (timeout_rate * real_to)

            row = {
                'Engulf Type': op['engulfing_type'], 'Session': op['session'], 'Trend': op['trend_filter'],
                'Vol Filter': op['vol_filter'], 'Streak': op['min_streak'], 'Size Mult': op['size_multi'],
                'Hold Mins': mins, 'TP Pts': tp, 'SL Pts': sl, 'Total Trades': ts,
                'INV Win%': round(win_rate * 100, 1), 'INV Loss%': round(loss_rate * 100, 1), 'INV Timeout%': round(timeout_rate * 100, 1),
                'Timeout Avg MFE': round(avg_to_mfe, 2), 'Timeout Avg MAE': round(avg_to_mae, 2),
                'Real EV (Costs Deducted)': round(true_ev, 3),
            }

            for yi, y in enumerate(unique_years):
                y_col = 7 + yi * 3
                y_wins = int(out[ic, y_col + 0])
                y_loss = int(out[ic, y_col + 1])
                y_to   = int(out[ic, y_col + 2])
                y_total = y_wins + y_loss + y_to
                if y_total > 0:
                    y_win_r  = y_wins / y_total
                    y_loss_r = y_loss / y_total
                    y_to_r   = y_to / y_total
                    y_ev     = (y_win_r * real_tp) - (y_loss_r * real_sl) + (y_to_r * real_to)
                    row[f'{y} Trades'] = y_total
                    row[f'{y} Win%']   = round(y_win_r * 100, 1)
                    row[f'{y} EV']     = round(y_ev, 2)
                else:
                    row[f'{y} Trades'] = 0
                    row[f'{y} Win%']   = 0.0
                    row[f'{y} EV']     = 0.0

            results.append(row)

        now = time.time()
        if now - last_print >= 10:
            elapsed = now - start_time
            pct     = outer_done / total_outer
            print(f"{outer_done:>6}/{total_outer}  {len(results):>8,}  {elapsed:>7.0f}s  {(((elapsed/pct)-elapsed)/60):>8.1f}min")
            sys.stdout.flush()
            last_print = now

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s ({elapsed/60:.1f} min).")

    res_df   = pd.DataFrame(results)
    sort_col = '2025 EV' if '2025 EV' in res_df.columns else 'Real EV (Costs Deducted)'
    res_df   = res_df.sort_values(by=sort_col, ascending=False).reset_index(drop=True)
    res_df.to_parquet("INVERSE_master_results.parquet")
    print(f"Saved {len(res_df):,} rows to INVERSE_master_results.parquet")

if __name__ == "__main__":
    signals_df, fwd_high, fwd_low, fwd_close = load_and_prep_data()
    run_fast_grid(signals_df, fwd_high, fwd_low, fwd_close)