"""
NQ Engulfing: All-In-One Multi-Timeframe Analysis (Normal & Inverse)
"""

import pandas as pd
import numpy as np
import itertools
import time
import sys
from numba import njit, prange

# ==========================================
# 1. SETUP YOUR TIMEFRAMES HERE
# ==========================================
HTF_FILE = "../data/nq_1h.parquet"    
HTF_MINS = 60                       
LTF_FILE = "../data/nq_1min.parquet"  

TRADE_COST_PTS = 0.5  # Spread/Slippage/Commissions deduction
# ==========================================

@njit(parallel=True)
def compute_all_in_one_numba(
    trade_dirs, entry_prices, start_idx_1m, year_ids,
    high_1m, low_1m, close_1m, len_1m,
    n_years, tp_arr, sl_arr, hold_arr, n_inner
):
    N_trades = len(trade_dirs)
    
    # We output 2 rows per combination (Index 0 = Normal, Index 1 = Inverse)
    # Columns: 10 Base + (9 * n_years)
    n_base_cols = 10
    n_yr_cols = 9
    n_cols = n_base_cols + (n_years * n_yr_cols)
    
    out = np.zeros((n_inner * 2, n_cols), dtype=np.float64)

    for ic in prange(n_inner):
        hold_mins = hold_arr[ic]
        tp = tp_arr[ic]
        sl_v = sl_arr[ic]

        # Accumulators: idx 0 = Normal, idx 1 = Inverse
        wins = np.zeros(2, dtype=np.float64)
        losses = np.zeros(2, dtype=np.float64)
        timeouts = np.zeros(2, dtype=np.float64)
        sum_mfe = np.zeros(2, dtype=np.float64)
        sum_mae = np.zeros(2, dtype=np.float64)
        sum_pnl = np.zeros(2, dtype=np.float64)
        sum_missed = np.zeros(2, dtype=np.float64)
        sum_to_mfe = np.zeros(2, dtype=np.float64)
        sum_to_mae = np.zeros(2, dtype=np.float64)

        y_wins = np.zeros((2, n_years), dtype=np.float64)
        y_losses = np.zeros((2, n_years), dtype=np.float64)
        y_timeouts = np.zeros((2, n_years), dtype=np.float64)
        y_mfe = np.zeros((2, n_years), dtype=np.float64)
        y_mae = np.zeros((2, n_years), dtype=np.float64)

        for i in range(N_trades):
            direction = trade_dirs[i] # 1 = Bull, -1 = Bear
            ep = entry_prices[i]
            ltf_start = start_idx_1m[i]
            y = year_ids[i]

            if ltf_start < 0 or ltf_start >= len_1m:
                continue

            # 1. WINDOW POTENTIAL SCAN
            max_h, min_l = -999999.0, 999999.0
            for step in range(hold_mins):
                idx = ltf_start + step
                if idx >= len_1m: break
                if high_1m[idx] > max_h: max_h = high_1m[idx]
                if low_1m[idx] < min_l: min_l = low_1m[idx]

            # Logic Arrays: [Normal, Inverse]
            mfe_arr = np.zeros(2, dtype=np.float64)
            mae_arr = np.zeros(2, dtype=np.float64)

            if direction == 1: # Bull
                mfe_arr[0], mae_arr[0] = max_h - ep, ep - min_l     # Normal = Long
                mfe_arr[1], mae_arr[1] = ep - min_l, max_h - ep     # Inverse = Short
            else: # Bear
                mfe_arr[0], mae_arr[0] = ep - min_l, max_h - ep     # Normal = Short
                mfe_arr[1], mae_arr[1] = max_h - ep, ep - min_l     # Inverse = Long
            
            for mode in range(2):
                sum_mfe[mode] += mfe_arr[mode]
                sum_mae[mode] += mae_arr[mode]
                y_mfe[mode, y] += mfe_arr[mode]
                y_mae[mode, y] += mae_arr[mode]

            # 2. CHRONOLOGICAL EXECUTION
            hit_win = np.zeros(2, dtype=np.bool_)
            hit_loss = np.zeros(2, dtype=np.bool_)

            for step in range(hold_mins):
                idx = ltf_start + step
                if idx >= len_1m: break
                
                curr_h = high_1m[idx]
                curr_l = low_1m[idx]

                # Normal Triggers
                if direction == 1:
                    hit_tp_0 = curr_h >= (ep + tp)
                    hit_sl_0 = curr_l <= (ep - sl_v)
                else:
                    hit_tp_0 = curr_l <= (ep - tp)
                    hit_sl_0 = curr_h >= (ep + sl_v)

                # Inverse Triggers
                if direction == 1: # Short
                    hit_tp_1 = curr_l <= (ep - tp)
                    hit_sl_1 = curr_h >= (ep + sl_v)
                else: # Long
                    hit_tp_1 = curr_h >= (ep + tp)
                    hit_sl_1 = curr_l <= (ep - sl_v)

                triggers = [(hit_tp_0, hit_sl_0), (hit_tp_1, hit_sl_1)]

                for mode in range(2):
                    if not hit_win[mode] and not hit_loss[mode]:
                        tp_hit, sl_hit = triggers[mode]
                        if tp_hit and sl_hit:
                            hit_loss[mode] = True # Conservative tie-breaker
                        elif tp_hit:
                            hit_win[mode] = True
                        elif sl_hit:
                            hit_loss[mode] = True

                if (hit_win[0] or hit_loss[0]) and (hit_win[1] or hit_loss[1]):
                    break # Stop looping if both logics have exited

            # 3. RECORD OUTCOMES
            final_idx = min(ltf_start + hold_mins - 1, len_1m - 1)
            
            pnl_arr = np.zeros(2, dtype=np.float64)
            if direction == 1:
                pnl_arr[0] = close_1m[final_idx] - ep
                pnl_arr[1] = ep - close_1m[final_idx]
            else:
                pnl_arr[0] = ep - close_1m[final_idx]
                pnl_arr[1] = close_1m[final_idx] - ep

            for mode in range(2):
                if hit_win[mode]:
                    wins[mode] += 1
                    y_wins[mode, y] += 1
                    # Potential Missed = Total MFE - TP
                    sum_missed[mode] += (mfe_arr[mode] - tp)
                elif hit_loss[mode]:
                    losses[mode] += 1
                    y_losses[mode, y] += 1
                else:
                    timeouts[mode] += 1
                    y_timeouts[mode, y] += 1
                    sum_pnl[mode] += pnl_arr[mode]
                    # Timeout specific MFE/MAE
                    sum_to_mfe[mode] += mfe_arr[mode]
                    sum_to_mae[mode] += mae_arr[mode]

        # WRITE TO OUTPUT ARRAY
        for mode in range(2):
            out_idx = (ic * 2) + mode
            total_valid = wins[mode] + losses[mode] + timeouts[mode]
            out[out_idx, 0] = total_valid
            out[out_idx, 1] = wins[mode]
            out[out_idx, 2] = losses[mode]
            out[out_idx, 3] = timeouts[mode]
            out[out_idx, 4] = sum_mfe[mode]
            out[out_idx, 5] = sum_mae[mode]
            out[out_idx, 6] = sum_pnl[mode]
            out[out_idx, 7] = sum_missed[mode]
            out[out_idx, 8] = sum_to_mfe[mode]
            out[out_idx, 9] = sum_to_mae[mode]

            for y in range(n_years):
                offset = n_base_cols + (y * n_yr_cols)
                out[out_idx, offset + 0] = y_wins[mode, y]
                out[out_idx, offset + 1] = y_losses[mode, y]
                out[out_idx, offset + 2] = y_timeouts[mode, y]
                out[out_idx, offset + 3] = y_mfe[mode, y]
                out[out_idx, offset + 4] = y_mae[mode, y]

    return out

def load_data():
    print(f"1. Loading HTF ({HTF_FILE}) and LTF ({LTF_FILE})...")
    df_htf = pd.read_parquet(HTF_FILE)
    df_1m = pd.read_parquet(LTF_FILE)

    for df in [df_htf, df_1m]:
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(np.float32)
        df.index = df.index.tz_localize('UTC').tz_convert('US/Pacific')
        df['year'] = df.index.year

    exec_times = df_htf.index + pd.Timedelta(minutes=HTF_MINS)
    df_htf['start_idx_1m'] = df_1m.index.get_indexer(exec_times, method='bfill')

    df_htf['hour'], df_htf['minute'] = df_htf.index.hour, df_htf.index.minute
    cond = [
        ((df_htf['hour'] == 0)),
        ((df_htf['hour'] == 6) & (df_htf['minute'] >= 30)) | ((df_htf['hour'] == 7) & (df_htf['minute'] < 30)),
        ((df_htf['hour'] == 8) & (df_htf['minute'] >= 30)) | (df_htf['hour'] == 9) | (df_htf['hour'] == 10) | ((df_htf['hour'] == 11) & (df_htf['minute'] < 30)),
        ((df_htf['hour'] == 12))
    ]
    df_htf['session'] = np.select(cond, ['London', 'US_Open', 'Midday', 'US_Close'], default='Other')

    df_htf['total_size'] = df_htf['high'] - df_htf['low']
    df_htf['color'] = np.where(df_htf['close'] >= df_htf['open'], 1, -1)
    df_htf['prev_open'], df_htf['prev_close'] = df_htf['open'].shift(1), df_htf['close'].shift(1)
    df_htf['prev_high'], df_htf['prev_low'] = df_htf['high'].shift(1), df_htf['low'].shift(1)
    df_htf['prev_color'] = df_htf['color'].shift(1)
    df_htf['prev_total_size'] = df_htf['total_size'].shift(1)

    df_htf['streak'] = df_htf['color'].groupby((df_htf['color'] != df_htf['prev_color']).cumsum()).cumsum().abs()
    df_htf['prev_streak'] = df_htf['streak'].shift(1)
    df_htf['sma_50'] = df_htf['close'].rolling(window=50).mean()
    df_htf['size_ratio'] = df_htf['total_size'] / df_htf['prev_total_size'].replace(0, 0.0001)
    df_htf['prev_vol'] = df_htf['volume'].shift(1).replace(0, 1)
    df_htf['vol_multiplier'] = df_htf['volume'] / df_htf['prev_vol']

    bull_body = (df_htf['prev_color'] == -1) & (df_htf['color'] == 1) & (df_htf['close'] >= df_htf['prev_open']) & (df_htf['open'] <= df_htf['prev_close'])
    bear_body = (df_htf['prev_color'] == 1) & (df_htf['color'] == -1) & (df_htf['close'] <= df_htf['prev_open']) & (df_htf['open'] >= df_htf['prev_close'])
    bull_wick = (df_htf['prev_color'] == -1) & (df_htf['color'] == 1) & (df_htf['high'] > df_htf['prev_high']) & (df_htf['low'] < df_htf['prev_low'])
    bear_wick = (df_htf['prev_color'] == 1) & (df_htf['color'] == -1) & (df_htf['high'] > df_htf['prev_high']) & (df_htf['low'] < df_htf['prev_low'])

    df_htf['is_bull_body'], df_htf['is_bear_body'] = bull_body, bear_body
    df_htf['is_bull_wick'], df_htf['is_bear_wick'] = bull_wick, bear_wick

    return df_htf, df_1m

def run_htf_grid(df_htf, df_1m):
    param_grid = {
        'engulfing_type':    ['wick', 'body'], 
        'min_streak':        [1, 2, 3, 4, 5],
        'size_multi':        [1.1, 1.5, 2.0],
        'trend_filter':      ['All'],
        'vol_filter':        ['All', '1.5x', '2x'],
        'session':           ['All', 'US_Open', 'US_Close', 'London'],
        
        # You can set very short windows here even for 1-Hour Signals!
        'look_forward_mins': [1, 3, 5, 10, 15, 20, 25, 30, 60], 
        'tp_pts':            [3, 5.0, 10.0, 15.0, 20.0, 25, 30, 35, 40, 45, 50],
        'sl_pts':            [3, 5.0, 10.0, 15.0, 20.0, 25, 30, 35, 40, 45, 50],
    }

    high_1m = df_1m['high'].values.astype(np.float32)
    low_1m = df_1m['low'].values.astype(np.float32)
    close_1m = df_1m['close'].values.astype(np.float32)
    open_1m = df_1m['open'].values.astype(np.float32) 
    len_1m = len(df_1m)

    d = {col: df_htf[col].values for col in df_htf.columns}
    unique_years = np.sort(df_htf['year'].unique())
    n_years = len(unique_years)
    year_to_idx = {y: i for i, y in enumerate(unique_years)}

    base_masks = {
        ('body', 'bull'): d['is_bull_body'].astype(bool),
        ('body', 'bear'): d['is_bear_body'].astype(bool),
        ('wick', 'bull'): d['is_bull_wick'].astype(bool),
        ('wick', 'bear'): d['is_bear_wick'].astype(bool),
    }

    inner_combos = list(itertools.product(param_grid['look_forward_mins'], param_grid['tp_pts'], param_grid['sl_pts']))
    n_inner = len(inner_combos)
    
    hold_arr = np.array([c[0] for c in inner_combos], dtype=np.int32)
    tp_arr = np.array([c[1] for c in inner_combos], dtype=np.float32)
    sl_arr = np.array([c[2] for c in inner_combos], dtype=np.float32)

    outer_keys = ['engulfing_type', 'min_streak', 'size_multi', 'trend_filter', 'vol_filter', 'session']
    outer_combos = list(itertools.product(*[param_grid[k] for k in outer_keys]))
    
    print("\nCompiling Numba JIT (~30 seconds)...")
    sys.stdout.flush()
    _dbool = np.array([1, -1], dtype=np.int8)
    _df = np.array([100.0, 100.0], dtype=np.float32)
    _di = np.array([0, 1], dtype=np.int32)
    compute_all_in_one_numba(_dbool, _df, _di, _di, _df, _df, _df, 2, 1, _df, _df, np.array([1], dtype=np.int32), 1)

    print(f"Starting Scan: {len(outer_combos) * n_inner:,} Combinations (x2 for Inverse) ...\n")
    results = []
    start_time = time.time()
    last_print = start_time

    for outer_done, outer in enumerate(outer_combos):
        op = dict(zip(outer_keys, outer))

        ctx = (d['prev_streak'] >= op['min_streak']) & (d['size_ratio'] >= op['size_multi'])
        if op['session'] != 'All': ctx &= (d['session'] == op['session'])
        if op['vol_filter'] == '1.5x':
            ctx &= (d['vol_multiplier'] >= 1.5)
        elif op['vol_filter'] == '2x':
            ctx &= (d['vol_multiplier'] >= 2.0)

        bull_mask = base_masks[(op['engulfing_type'], 'bull')] & ctx
        bear_mask = base_masks[(op['engulfing_type'], 'bear')] & ctx

        valid_signals = (bull_mask | bear_mask) & (d['start_idx_1m'] >= 0) & (d['start_idx_1m'] < len_1m)
        if valid_signals.sum() < 25: continue

        trade_dirs = np.where(bull_mask[valid_signals], 1, -1).astype(np.int8)
        start_idx_1m = d['start_idx_1m'][valid_signals].astype(np.int32)
        entry_prices = open_1m[start_idx_1m].astype(np.float32)
        year_ids = np.array([year_to_idx[y] for y in d['year'][valid_signals]], dtype=np.int32)

        out = compute_all_in_one_numba(
            trade_dirs, entry_prices, start_idx_1m, year_ids,
            high_1m, low_1m, close_1m, len_1m,
            n_years, tp_arr, sl_arr, hold_arr, n_inner
        )

        n_base_cols = 10
        n_yr_cols = 9

        for ic in range(n_inner):
            mins, tp, sl = inner_combos[ic]
            
            for mode in range(2):
                idx = (ic * 2) + mode
                logic_name = "Normal" if mode == 0 else "Inverse"

                ts = int(out[idx, 0])
                if ts < 25: continue

                wins, losses, timeouts = int(out[idx, 1]), int(out[idx, 2]), int(out[idx, 3])
                win_r, loss_r, to_r = wins / ts, losses / ts, timeouts / ts

                avg_mfe = out[idx, 4] / ts
                avg_mae = out[idx, 5] / ts
                avg_to_pnl = out[idx, 6] / timeouts if timeouts > 0 else 0.0
                
                # New "Missed Potential" & Timeout Extremes
                avg_missed = out[idx, 7] / wins if wins > 0 else 0.0
                avg_to_mfe = out[idx, 8] / timeouts if timeouts > 0 else 0.0
                avg_to_mae = out[idx, 9] / timeouts if timeouts > 0 else 0.0

                real_tp = tp - TRADE_COST_PTS
                real_sl = sl + TRADE_COST_PTS
                real_to = avg_to_pnl - TRADE_COST_PTS
                true_ev = (win_r * real_tp) - (loss_r * real_sl) + (to_r * real_to)

                # RECENT YEARS AGGREGATION (2022+)
                rec_ts, rec_wins, rec_loss, rec_to = 0, 0, 0, 0
                rec_pnl_sum = 0.0
                
                for yi, y in enumerate(unique_years):
                    if y >= 2022:
                        offset = n_base_cols + (yi * n_yr_cols)
                        y_wins = int(out[idx, offset + 0])
                        y_loss = int(out[idx, offset + 1])
                        y_to   = int(out[idx, offset + 2])
                        rec_ts += (y_wins + y_loss + y_to)
                        rec_wins += y_wins
                        rec_loss += y_loss
                        rec_to += y_to
                        
                        # Timeout PnL is offset + 5 (approximated for EV below)
                        # We don't store exact timeout PNL per year to save RAM, using average is close enough for aggregate
                        rec_pnl_sum += (y_to * avg_to_pnl)

                if rec_ts > 0:
                    rec_wr = rec_wins / rec_ts
                    rec_lr = rec_loss / rec_ts
                    rec_tor = rec_to / rec_ts
                    rec_to_pnl = (rec_pnl_sum / rec_to) if rec_to > 0 else 0.0
                    rec_ev = (rec_wr * real_tp) - (rec_lr * real_sl) + (rec_tor * (rec_to_pnl - TRADE_COST_PTS))
                else:
                    rec_wr, rec_lr, rec_ev = 0.0, 0.0, 0.0

                row = {
                    'Logic':       logic_name,
                    'HTF Signal':  f"{HTF_MINS} Min",
                    'Engulf Type': op['engulfing_type'],
                    'Session':     op['session'],
                    'Trend':       op['trend_filter'],
                    'Vol Filter':  op['vol_filter'],
                    'Streak':      op['min_streak'],
                    'Size Mult':   op['size_multi'],
                    'Hold Mins':   mins,
                    'TP Pts':      tp,
                    'SL Pts':      sl,
                    'Total Trades':ts,
                    
                    'Wins (#)':    wins,
                    'Win%':        round(win_r * 100, 1),
                    'Losses (#)':  losses,
                    'Loss%':       round(loss_r * 100, 1),
                    'Timeouts (#)':timeouts,
                    'Timeout%':    round(to_r * 100, 1),
                    
                    'Avg Trade MFE': round(avg_mfe, 2),
                    'Avg Trade MAE': round(avg_mae, 2),
                    
                    # New Metrics
                    'Win Missed Potential': round(avg_missed, 2),
                    'Timeout Max High (MFE)': round(avg_to_mfe, 2),
                    'Timeout Max Low (MAE)': round(avg_to_mae, 2),
                    
                    'Chronological EV': round(true_ev, 3),

                    # Recent Years (2022+) Aggregation
                    '2022+ Trades': rec_ts,
                    '2022+ Wins (#)': rec_wins,
                    '2022+ Win%': round(rec_wr * 100, 1),
                    '2022+ EV': round(rec_ev, 2),
                }

                # Yearly Breakdown (MFE & MAE Added)
                for yi, y in enumerate(unique_years):
                    offset = n_base_cols + (yi * n_yr_cols)
                    y_wins = int(out[idx, offset + 0])
                    y_loss = int(out[idx, offset + 1])
                    y_to   = int(out[idx, offset + 2])
                    y_tot  = y_wins + y_loss + y_to
                    y_mfe  = out[idx, offset + 3]
                    y_mae  = out[idx, offset + 4]

                    if y_tot > 0:
                        y_wr = y_wins / y_tot
                        y_lr = y_loss / y_tot
                        y_tr = y_to / y_tot
                        y_ev = (y_wr * real_tp) - (y_lr * real_sl) + (y_tr * (avg_to_pnl - TRADE_COST_PTS))
                        row[f'{y} Trades'] = y_tot
                        row[f'{y} Win%'] = round(y_wr * 100, 1)
                        row[f'{y} EV'] = round(y_ev, 2)
                        row[f'{y} Avg MFE'] = round(y_mfe / y_tot, 2)
                        row[f'{y} Avg MAE'] = round(y_mae / y_tot, 2)
                    else:
                        row[f'{y} Trades'], row[f'{y} Win%'], row[f'{y} EV'] = 0, 0.0, 0.0
                        row[f'{y} Avg MFE'], row[f'{y} Avg MAE'] = 0.0, 0.0

                results.append(row)

        now = time.time()
        if now - last_print >= 5:
            elapsed = now - start_time
            pct = (outer_done+1) / len(outer_combos)
            print(f"{outer_done+1:>5}/{len(outer_combos)} combos  |  {len(results):>8,} results  |  {elapsed:>5.0f}s elapsed")
            last_print = now

    print(f"\nDone in {time.time() - start_time:.1f}s.")
    if len(results) > 0:
        res_df = pd.DataFrame(results).sort_values(by='Chronological EV', ascending=False).reset_index(drop=True)
        res_df.to_parquet(f"HTF_ALL_IN_ONE_{HTF_MINS}m.parquet")
        print(f"Saved {len(res_df):,} rows to HTF_ALL_IN_ONE_{HTF_MINS}m.parquet")
    else:
        print("No valid setups found.")

if __name__ == "__main__":
    df_htf, df_1m = load_data()
    run_htf_grid(df_htf, df_1m)