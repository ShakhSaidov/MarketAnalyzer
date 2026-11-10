"""
NQ Engulfing: Multi-Timeframe Analysis (HTF Signals -> LTF Execution)
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
HTF_FILE = "../data/nq_1h.parquet"    # Your Higher Timeframe (e.g., 1h, 15min, 5min)
HTF_MINS = 60                       # Ensure this matches the file! (5 for 5m, 15 for 15m, 30 for 30m)
LTF_FILE = "../data/nq_1min.parquet"  # Execution Timeframe (always 1 min)

TRADE_COST_PTS = 0.5  # $10 round trip for spread/slippage/commissions
# ==========================================

@njit(parallel=True)
def compute_htf_ltf_numba(
    trade_dirs, entry_prices, start_idx_1m, year_ids,
    high_1m, low_1m, close_1m, len_1m,
    n_years, tp_arr, sl_arr, hold_arr, n_inner
):
    N_trades = len(trade_dirs)
    n_cols = 7 + n_years * 3
    out = np.zeros((n_inner, n_cols), dtype=np.float64)

    for ic in prange(n_inner):
        hold_mins = hold_arr[ic]
        tp = tp_arr[ic]
        sl_v = sl_arr[ic]

        wins, losses, timeouts = np.float64(0), np.float64(0), np.float64(0)
        sum_mfe, sum_mae, sum_pnl = np.float64(0), np.float64(0), np.float64(0)

        y_wins = np.zeros(n_years, dtype=np.float64)
        y_losses = np.zeros(n_years, dtype=np.float64)
        y_timeouts = np.zeros(n_years, dtype=np.float64)

        for i in range(N_trades):
            direction = trade_dirs[i] # 1 = Bull, -1 = Bear
            ep = entry_prices[i]
            ltf_start = start_idx_1m[i]
            y = year_ids[i]

            # 1. FIND ABSOLUTE POTENTIAL OF THE WINDOW (MFE/MAE)
            max_h = -999999.0
            min_l = 999999.0
            
            for step in range(hold_mins):
                idx = ltf_start + step
                if idx >= len_1m: break
                
                if high_1m[idx] > max_h: max_h = high_1m[idx]
                if low_1m[idx] < min_l: min_l = low_1m[idx]

            if direction == 1: # Bull
                trade_mfe = max_h - ep
                trade_mae = ep - min_l
            else: # Bear
                trade_mfe = ep - min_l
                trade_mae = max_h - ep
            
            sum_mfe += trade_mfe
            sum_mae += trade_mae

            # 2. CHRONOLOGICAL HIT DETECTION (Realistic Outcomes)
            hit_win = False
            hit_loss = False

            for step in range(hold_mins):
                idx = ltf_start + step
                if idx >= len_1m: break
                
                curr_h = high_1m[idx]
                curr_l = low_1m[idx]

                if direction == 1: # Bull
                    hit_tp = curr_h >= (ep + tp)
                    hit_sl = curr_l <= (ep - sl_v)
                else: # Bear
                    hit_tp = curr_l <= (ep - tp)
                    hit_sl = curr_h >= (ep + sl_v)

                if hit_tp and hit_sl:
                    hit_loss = True # Conservative double-hit rule inside 1 minute
                    break
                elif hit_tp:
                    hit_win = True
                    break
                elif hit_sl:
                    hit_loss = True
                    break

            # 3. RECORD OUTCOMES
            if hit_win:
                wins += 1
                y_wins[y] += 1
            elif hit_loss:
                losses += 1
                y_losses[y] += 1
            else:
                timeouts += 1
                y_timeouts[y] += 1
                
                # Timeout PNL uses the exact final minute's close price
                final_idx = min(ltf_start + hold_mins - 1, len_1m - 1)
                if direction == 1:
                    sum_pnl += close_1m[final_idx] - ep
                else:
                    sum_pnl += ep - close_1m[final_idx]

        total_valid = wins + losses + timeouts
        out[ic, 0] = total_valid
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

def load_data():
    print(f"1. Loading HTF ({HTF_FILE}) and LTF ({LTF_FILE})...")
    df_htf = pd.read_parquet(HTF_FILE)
    df_1m = pd.read_parquet(LTF_FILE)

    for df in [df_htf, df_1m]:
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(np.float32)
        df.index = df.index.tz_localize('UTC').tz_convert('US/Pacific')
        df['year'] = df.index.year

    print("2. Mapping HTF signals to 1-Min execution timeline...")
    # The moment the HTF candle closes, the execution starts at the VERY NEXT minute
    exec_times = df_htf.index + pd.Timedelta(minutes=HTF_MINS)
    
    # Safely find the index of that exact 1-minute candle in the LTF dataframe
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
    # ==========================================
    # MANUALLY TWEAK THESE PARAMETERS
    # ==========================================
    param_grid = {
        'engulfing_type':    ['wick', 'body'], 
        'min_streak':        [1, 2, 3,4 ,5],
        'size_multi':        [1.1, 1.5, 2.0],
        'trend_filter':      ['All'],
        'vol_filter':        ['All', '1.5x', '2x'],
        'session':           ['All', 'US_Open', 'US_Close', 'London'],
        
        # You can set very short windows here even for 1-Hour Signals!
        'look_forward_mins': [1, 2, 3, 4, 5, 10, 15, 20, 25, 30, 60], 
        'tp_pts':            [2, 3, 4, 5.0, 6, 7, 8, 9, 10.0, 15.0, 20.0],
        'sl_pts':            [2, 3, 4, 5.0, 6, 7, 8, 9, 10.0, 15.0, 20.0],
    }

    print("\nPreparing Numba Arrays...")
    
    high_1m = df_1m['high'].values.astype(np.float32)
    low_1m = df_1m['low'].values.astype(np.float32)
    close_1m = df_1m['close'].values.astype(np.float32)
    open_1m = df_1m['open'].values.astype(np.float32) # USED FOR PRECISE ENTRY
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

    tp_list, sl_list, hold_list = param_grid['tp_pts'], param_grid['sl_pts'], param_grid['look_forward_mins']
    inner_combos = list(itertools.product(hold_list, tp_list, sl_list))
    n_inner = len(inner_combos)
    
    hold_arr = np.array([c[0] for c in inner_combos], dtype=np.int32)
    tp_arr = np.array([c[1] for c in inner_combos], dtype=np.float32)
    sl_arr = np.array([c[2] for c in inner_combos], dtype=np.float32)

    outer_keys = ['engulfing_type', 'min_streak', 'size_multi', 'trend_filter', 'vol_filter', 'session']
    outer_combos = list(itertools.product(*[param_grid[k] for k in outer_keys]))
    
    print(f"Total Combinations: {len(outer_combos) * n_inner:,}")

    # NUMBA WARMUP
    print("\nCompiling HTF Numba JIT (~30 seconds)...")
    sys.stdout.flush()
    _dummy_dirs = np.array([1, -1], dtype=np.int8)
    _dummy_f = np.array([100.0, 100.0], dtype=np.float32)
    _dummy_i = np.array([0, 1], dtype=np.int32)
    _dummy_h = np.array([1], dtype=np.int32)
    compute_htf_ltf_numba(_dummy_dirs, _dummy_f, _dummy_i, _dummy_i, _dummy_f, _dummy_f, _dummy_f, 2, 1, _dummy_f, _dummy_f, _dummy_h, 1)

    print("Compilation done. Starting HTF Grid Search...\n")

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

        if op['trend_filter'] == 'With_Trend':
            bull_mask &= (d['close'] > d['sma_50'])
            bear_mask &= (d['close'] < d['sma_50'])

        valid_signals = bull_mask | bear_mask
        
        # To avoid index errors, filter out signals where the mapping went beyond our 1-min data array
        valid_bounds = (d['start_idx_1m'] >= 0) & (d['start_idx_1m'] < len_1m)
        valid_signals = valid_signals & valid_bounds
        
        if valid_signals.sum() < 25: 
            continue

        trade_dirs = np.where(bull_mask[valid_signals], 1, -1).astype(np.int8)
        start_idx_1m = d['start_idx_1m'][valid_signals].astype(np.int32)
        
        # 🚨 THE EXECUTION FIX: Entry Price is the OPEN of the immediate next 1-minute candle 🚨
        entry_prices = open_1m[start_idx_1m].astype(np.float32)
        
        year_ids = np.array([year_to_idx[y] for y in d['year'][valid_signals]], dtype=np.int32)

        out = compute_htf_ltf_numba(
            trade_dirs, entry_prices, start_idx_1m, year_ids,
            high_1m, low_1m, close_1m, len_1m,
            n_years, tp_arr, sl_arr, hold_arr, n_inner
        )

        for ic in range(n_inner):
            mins, tp, sl = inner_combos[ic]
            ts = int(out[ic, 0])
            if ts < 25: continue

            wins, losses, timeouts = int(out[ic, 1]), int(out[ic, 2]), int(out[ic, 3])
            
            win_rate = wins / ts
            loss_rate = losses / ts
            timeout_rate = timeouts / ts

            # Potential Tracking (Averaged over ALL trades)
            avg_mfe = out[ic, 4] / ts
            avg_mae = out[ic, 5] / ts
            
            # Timeout PNL calculation
            avg_to_pnl = (out[ic, 6] / timeouts) if timeouts > 0 else 0.0

            # True EV with costs
            real_tp = tp - TRADE_COST_PTS
            real_sl = sl + TRADE_COST_PTS
            real_to = avg_to_pnl - TRADE_COST_PTS
            true_ev = (win_rate * real_tp) - (loss_rate * real_sl) + (timeout_rate * real_to)

            row = {
                'HTF Signal':  f"{HTF_MINS} Min",
                'Engulf Type': op['engulfing_type'],
                'Session':     op['session'],
                'Trend':       op['trend_filter'],
                'Vol Filter':  op['vol_filter'],
                'Streak':      op['min_streak'],
                'Size Mult':   op['size_multi'],
                'Hold Mins (1m)': mins,
                'TP Pts':      tp,
                'SL Pts':      sl,
                'Total Trades':ts,
                'ALL Win%':    round(win_rate * 100, 1),
                'ALL Loss%':   round(loss_rate * 100, 1),
                'ALL Timeout%':round(timeout_rate * 100, 1),
                
                # The "Absolute Potential" Columns
                'Avg Window MFE (Pts)': round(avg_mfe, 2),
                'Avg Window MAE (Pts)': round(avg_mae, 2),
                
                'Chronological EV': round(true_ev, 3),
            }

            for yi, y in enumerate(unique_years):
                y_col = 7 + yi * 3
                y_total = int(out[ic, y_col + 0]) + int(out[ic, y_col + 1]) + int(out[ic, y_col + 2])
                if y_total > 0:
                    y_wr = int(out[ic, y_col + 0]) / y_total
                    y_lr = int(out[ic, y_col + 1]) / y_total
                    y_tr = int(out[ic, y_col + 2]) / y_total
                    row[f'{y} Trades'] = y_total
                    row[f'{y} Win%'] = round(y_wr * 100, 1)
                    row[f'{y} EV'] = round((y_wr * real_tp) - (y_lr * real_sl) + (y_tr * real_to), 2)
                else:
                    row[f'{y} Trades'], row[f'{y} Win%'], row[f'{y} EV'] = 0, 0.0, 0.0
            results.append(row)

        now = time.time()
        if now - last_print >= 5:
            elapsed = now - start_time
            pct = (outer_done+1) / len(outer_combos)
            print(f"{outer_done+1:>5}/{len(outer_combos)} combos  |  {len(results):>8,} results  |  {elapsed:>5.0f}s elapsed")
            sys.stdout.flush()
            last_print = now

    elapsed = time.time() - start_time
    print(f"\nDone in {elapsed:.1f}s.")
    
    if len(results) > 0:
        res_df = pd.DataFrame(results).sort_values(by='Chronological EV', ascending=False).reset_index(drop=True)
        res_df.to_parquet(f"HTF_master_results_{HTF_MINS}m.parquet")
        print(f"Saved {len(res_df):,} rows to HTF_master_results_{HTF_MINS}m.parquet")
    else:
        print("No valid setups found.")

if __name__ == "__main__":
    df_htf, df_1m = load_data()
    run_htf_grid(df_htf, df_1m)