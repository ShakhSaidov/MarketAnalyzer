import pandas as pd
import numpy as np

# ==========================================
# ENTER THE EXACT PARAMETERS FROM YOUR ROW
# ==========================================
PARAMS = {
    'htf_file':          "../data/nq_15min.parquet",  
    'htf_mins':          15,                          
    'ltf_file':          "../data/nq_1min.parquet",
    
    'Logic':             'Normal',  # Change to 'Inverse' to flip trade directions
    
    'engulfing_type':    'wick',
    'session':           'US_Open',
    'trend_filter':      'All', 
    'vol_filter':        'All',   
    'min_streak':        1,
    'size_multi':        1.1,
    
    'hold_mins':         15,
    'tp_pts':            5.0,  
    'sl_pts':            50.0   
}
# ==========================================

def extract_timeline(p):
    print(f"Loading {p['htf_file']} and {p['ltf_file']}...")
    df_htf = pd.read_parquet(p['htf_file'])
    df_1m = pd.read_parquet(p['ltf_file'])

    for df in [df_htf, df_1m]:
        df.index = df.index.tz_localize('UTC').tz_convert('US/Pacific')

    print("Mapping HTF Signals to LTF Execution...")
    exec_times = df_htf.index + pd.Timedelta(minutes=p['htf_mins'])
    df_htf['start_idx_1m'] = df_1m.index.get_indexer(exec_times, method='bfill')

    df_htf['hour'], df_htf['minute'] = df_htf.index.hour, df_htf.index.minute
    
    # Session Logic
    cond = [
        ((df_htf['hour'] == 0)),
        ((df_htf['hour'] == 6) & (df_htf['minute'] >= 30)) | ((df_htf['hour'] == 7) & (df_htf['minute'] < 30)),
        ((df_htf['hour'] == 8) & (df_htf['minute'] >= 30)) | (df_htf['hour'] == 9) | (df_htf['hour'] == 10) | ((df_htf['hour'] == 11) & (df_htf['minute'] < 30)),
        ((df_htf['hour'] == 12))
    ]
    df_htf['session'] = np.select(cond, ['London', 'US_Open', 'Midday', 'US_Close'], default='Other')

    # Features
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

    # Engulfing definitions
    if p['engulfing_type'] == 'body':
        bull_base = (df_htf['prev_color'] == -1) & (df_htf['color'] == 1) & (df_htf['close'] >= df_htf['prev_open']) & (df_htf['open'] <= df_htf['prev_close'])
        bear_base = (df_htf['prev_color'] == 1) & (df_htf['color'] == -1) & (df_htf['close'] <= df_htf['prev_open']) & (df_htf['open'] >= df_htf['prev_close'])
    else:
        bull_base = (df_htf['prev_color'] == -1) & (df_htf['color'] == 1) & (df_htf['high'] > df_htf['prev_high']) & (df_htf['low'] < df_htf['prev_low'])
        bear_base = (df_htf['prev_color'] == 1) & (df_htf['color'] == -1) & (df_htf['high'] > df_htf['prev_high']) & (df_htf['low'] < df_htf['prev_low'])

    # Apply Filters
    ctx = (df_htf['prev_streak'] >= p['min_streak']) & (df_htf['size_ratio'] >= p['size_multi'])
    if p['session'] != 'All': ctx &= (df_htf['session'] == p['session'])
    
    if p['vol_filter'] == '1.5x': ctx &= (df_htf['vol_multiplier'] >= 1.5)
    elif p['vol_filter'] == '2.0x' or p['vol_filter'] == '2x': ctx &= (df_htf['vol_multiplier'] >= 2.0)

    bull_base &= ctx
    bear_base &= ctx

    if p['trend_filter'] == 'With_Trend':
        bull_base &= (df_htf['close'] > df_htf['sma_50'])
        bear_base &= (df_htf['close'] < df_htf['sma_50'])

    # Ensure mapped index doesn't exceed 1m boundaries
    valid_bounds = (df_htf['start_idx_1m'] >= 0) & (df_htf['start_idx_1m'] < len(df_1m))
    bull_base &= valid_bounds
    bear_base &= valid_bounds

    signals = df_htf[bull_base | bear_base].copy()
    print(f"Found {len(signals)} valid signals. Simulating chronological 1m execution...")

    high_1m = df_1m['high'].values
    low_1m = df_1m['low'].values
    close_1m = df_1m['close'].values
    open_1m = df_1m['open'].values
    idx_1m_times = df_1m.index

    results = []

    for idx, row in signals.iterrows():
        is_bull_signal = bull_base.loc[idx]
        ltf_start = int(row['start_idx_1m'])
        
        exec_time = idx_1m_times[ltf_start]
        ep = open_1m[ltf_start]  # Exact Open of next 1-minute candle
        
        # Determine Trade Direction based on Logic mode
        if p['Logic'] == 'Normal':
            trade_dir = 'LONG' if is_bull_signal else 'SHORT'
        else: # Inverse
            trade_dir = 'SHORT' if is_bull_signal else 'LONG'
        
        max_mfe, max_mae = 0.0, 0.0
        outcome = "TIMEOUT"
        exit_time = None
        exit_price = None
        tp_hit, sl_hit = False, False

        # 1. First, find Absolute Window MFE/MAE
        max_h, min_l = -999999.0, 999999.0
        for step in range(p['hold_mins']):
            curr_idx = ltf_start + step
            if curr_idx >= len(df_1m): break
            if high_1m[curr_idx] > max_h: max_h = high_1m[curr_idx]
            if low_1m[curr_idx] < min_l: min_l = low_1m[curr_idx]
            
        if trade_dir == 'LONG':
            window_mfe = max_h - ep
            window_mae = ep - min_l
        else:
            window_mfe = ep - min_l
            window_mae = max_h - ep

        # 2. Chronological Walk-Forward for actual hit detection
        for step in range(p['hold_mins']):
            curr_idx = ltf_start + step
            if curr_idx >= len(df_1m): break
            
            h = high_1m[curr_idx]
            l = low_1m[curr_idx]
            
            if trade_dir == 'LONG':
                curr_mfe, curr_mae = h - ep, ep - l
                hit_tp = h >= (ep + p['tp_pts'])
                hit_sl = l <= (ep - p['sl_pts'])
            else: # SHORT
                curr_mfe, curr_mae = ep - l, h - ep
                hit_tp = l <= (ep - p['tp_pts'])
                hit_sl = h >= (ep + p['sl_pts'])
                
            if curr_mfe > max_mfe: max_mfe = curr_mfe
            if curr_mae > max_mae: max_mae = curr_mae

            if hit_tp and hit_sl:
                outcome = "LOSS (Double Hit Tie-Break)"
                exit_price = ep - p['sl_pts'] if trade_dir == 'LONG' else ep + p['sl_pts']
                exit_time = idx_1m_times[curr_idx]
                break
            elif hit_tp:
                outcome = "WIN"
                exit_price = ep + p['tp_pts'] if trade_dir == 'LONG' else ep - p['tp_pts']
                exit_time = idx_1m_times[curr_idx]
                break
            elif hit_sl:
                outcome = "LOSS"
                exit_price = ep - p['sl_pts'] if trade_dir == 'LONG' else ep + p['sl_pts']
                exit_time = idx_1m_times[curr_idx]
                break

        if outcome == "TIMEOUT":
            final_idx = min(ltf_start + p['hold_mins'] - 1, len(df_1m) - 1)
            exit_time = idx_1m_times[final_idx]
            exit_price = close_1m[final_idx]

        # Calculate Missed Potential
        if outcome == "WIN":
            missed_pts = window_mfe - p['tp_pts']
        else:
            missed_pts = 0.0

        results.append({
            'HTF_Signal_Time': idx,
            'LTF_Exec_Time': exec_time,
            'HTF_Signal': 'Bullish Engulf' if is_bull_signal else 'Bearish Engulf',
            'Direction_Traded': trade_dir,
            'Entry_Price': ep,
            'Outcome': outcome,
            'Exit_Time': exit_time,
            'Exit_Price': round(exit_price, 2),
            'Window_MFE': round(window_mfe, 2),
            'Window_MAE': round(window_mae, 2),
            'Missed_Potential': round(missed_pts, 2)
        })

    res_df = pd.DataFrame(results)
    output_filename = "HTF_verification_timeline_15_tp_18.csv"
    res_df.to_csv(output_filename, index=False)
    
    print("\n" + "="*55)
    print(f"✅ TIMELINE EXTRACTION COMPLETE ({p['Logic']} Logic) ✅")
    print("="*55)
    print(f"Total Trades Evaluated: {len(res_df):,}")
    
    wins = len(res_df[res_df['Outcome'] == 'WIN'])
    losses = len(res_df[res_df['Outcome'].str.contains('LOSS')])
    timeouts = len(res_df[res_df['Outcome'] == 'TIMEOUT'])
    
    print(f"Wins:     {wins:,} ({wins/len(res_df)*100:.1f}%)")
    print(f"Losses:   {losses:,} ({losses/len(res_df)*100:.1f}%)")
    print(f"Timeouts: {timeouts:,} ({timeouts/len(res_df)*100:.1f}%)")
    print("="*55)
    print(f"Saved exact timeline to '{output_filename}'")
    print("Open this CSV and cross-reference HTF_Signal_Time and LTF_Exec_Time on your charts.")

if __name__ == "__main__":
    extract_timeline(PARAMS)