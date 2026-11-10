import pandas as pd
import numpy as np
import itertools
import time

def load_and_prep_data():
    print("1. Loading Parquet Data & Calculating Features...")
    df = pd.read_parquet("data/nq_1min.parquet")
    
    # Time and Session setup
    df.index = df.index.tz_localize('UTC').tz_convert('US/Pacific')
    df['year'] = df.index.year 
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    
    conditions = [
        ((df['hour'] == 0) | (df['hour'] == 1)), 
        ((df['hour'] == 6) & (df['minute'] >= 30)) | (df['hour'] == 7) | ((df['hour'] == 8) & (df['minute'] < 30)), 
        ((df['hour'] == 8) & (df['minute'] >= 30)) | (df['hour'] == 9) | (df['hour'] == 10) | ((df['hour'] == 11) & (df['minute'] < 30)), 
        ((df['hour'] == 11) & (df['minute'] >= 30)) | (df['hour'] == 12) 
    ]
    choices = ['London', 'US_Open', 'Midday', 'US_Close']
    df['session'] = np.select(conditions, choices, default='Other')

    # Basic Features
    df['body'] = (df['close'] - df['open']).abs()
    df['total_size'] = df['high'] - df['low']
    df['color'] = np.where(df['close'] >= df['open'], 1, -1)
    
    # Shifts
    df['prev_open'] = df['open'].shift(1)
    df['prev_close'] = df['close'].shift(1)
    df['prev_high'] = df['high'].shift(1)
    df['prev_low'] = df['low'].shift(1)
    df['prev_color'] = df['color'].shift(1)
    df['prev_total_size'] = df['total_size'].shift(1)
    
    # Context Features
    df['streak'] = df['color'].groupby((df['color'] != df['prev_color']).cumsum()).cumsum().abs()
    df['prev_streak'] = df['streak'].shift(1)
    
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['vol_ma_20'] = df['volume'].rolling(window=20).mean()
    df['vol_spike_ratio'] = df['volume'] / df['vol_ma_20']
    df['size_ratio'] = df['total_size'] / df['prev_total_size'].replace(0, 0.0001)

    # Exhaustion Features
    df['next_total_size'] = df['total_size'].shift(-1)
    df['next_close'] = df['close'].shift(-1)
    df['exh_pct'] = df['next_total_size'] / df['total_size'].replace(0, 0.0001)

    print("2. Isolating Engulfing Candles...")
    bull_body = (df['prev_color'] == -1) & (df['color'] == 1) & (df['close'] >= df['prev_open']) & (df['open'] <= df['prev_close'])
    bear_body = (df['prev_color'] == 1) & (df['color'] == -1) & (df['close'] <= df['prev_open']) & (df['open'] >= df['prev_close'])
    bull_wick = (df['prev_color'] == -1) & (df['color'] == 1) & (df['high'] > df['prev_high']) & (df['low'] < df['prev_low'])
    bear_wick = (df['prev_color'] == 1) & (df['color'] == -1) & (df['high'] > df['prev_high']) & (df['low'] < df['prev_low'])
    
    df['is_bull_body'] = bull_body
    df['is_bear_body'] = bear_body
    df['is_bull_wick'] = bull_wick
    df['is_bear_wick'] = bear_wick
    
    signals_df = df[bull_body | bear_body | bull_wick | bear_wick].copy()

    print("3. Calculating Future Outcomes (in absolute points)...")
    forward_times = [5, 10] 
    for mins in forward_times:
        signals_df[f'fwd_max_{mins}'] = df['high'].rolling(window=mins).max().shift(-mins)[signals_df.index]
        signals_df[f'fwd_min_{mins}'] = df['low'].rolling(window=mins).min().shift(-mins)[signals_df.index]
        signals_df[f'fwd_max_exh_{mins}'] = df['high'].rolling(window=mins).max().shift(-mins - 1)[signals_df.index]
        signals_df[f'fwd_min_exh_{mins}'] = df['low'].rolling(window=mins).min().shift(-mins - 1)[signals_df.index]

    return signals_df

def run_fast_grid(df):
    # ==========================================
    # EDITABLE PARAMETER GRID
    # ==========================================
    param_grid = {
        'engulfing_type': ['body', 'wick'],
        'min_streak': [1, 2, 3, 4], 
        'size_multi': [1.5, 2.0, 2.5, 3.0], 
        'use_exhaustion': [False, True],
        'exhaustion_max_pct': [0.5, 0.3, 0.1], 
        'trend_filter': ['All', 'With_Trend'], 
        'vol_filter': ['All', 'Spike_2x'], 
        'session': ['All', 'US_Open', 'London', 'US_Close'], 
        'look_forward_mins': [5, 10], 
        'tp_pts': [1.0, 2.0, 4.0, 5.0, 7.5, 10.0], # 1 to 10 points
        'sl_pts': [1.0, 2.0, 4.0, 5.0, 7.5, 10.0]  # 1 to 10 points
    }

    d = {col: df[col].values for col in df.columns}
    
    # Automatically detect all years from 2010 to 2026 in the dataset
    unique_years = np.sort(df['year'].unique()) 
    
    combinations = list(itertools.product(*param_grid.values()))
    print(f"\n🚀 Running {len(combinations)} combinations... This might take a few minutes.\n")

    results = []
    start_time = time.time()

    for i, combo in enumerate(combinations):
        p = dict(zip(param_grid.keys(), combo))
        
        # Skip illogical combos (Exhaustion % logic)
        if not p['use_exhaustion'] and p['exhaustion_max_pct'] != param_grid['exhaustion_max_pct'][0]:
            continue 

        # 1. Base Signal
        if p['engulfing_type'] == 'body':
            bull_mask, bear_mask = d['is_bull_body'], d['is_bear_body']
        else:
            bull_mask, bear_mask = d['is_bull_wick'], d['is_bear_wick']

        # 2. Context Filters
        ctx_mask = (d['prev_streak'] >= p['min_streak']) & (d['size_ratio'] >= p['size_multi'])
        if p['vol_filter'] == 'Spike_2x': ctx_mask = ctx_mask & (d['vol_spike_ratio'] >= 2)
        if p['session'] != 'All': ctx_mask = ctx_mask & (d['session'] == p['session'])

        bull_mask, bear_mask = bull_mask & ctx_mask, bear_mask & ctx_mask

        # 3. Trend Filter
        if p['trend_filter'] == 'With_Trend':
            bull_mask = bull_mask & (d['close'] > d['sma_50'])
            bear_mask = bear_mask & (d['close'] < d['sma_50'])

        # 4. Exhaustion & Entry
        if p['use_exhaustion']:
            exh_mask = d['exh_pct'] <= p['exhaustion_max_pct']
            bull_mask, bear_mask = bull_mask & exh_mask, bear_mask & exh_mask
            entry_price = d['next_close']
            f_max = d[f"fwd_max_exh_{p['look_forward_mins']}"]
            f_min = d[f"fwd_min_exh_{p['look_forward_mins']}"]
        else:
            entry_price = d['close']
            f_max = d[f"fwd_max_{p['look_forward_mins']}"]
            f_min = d[f"fwd_min_{p['look_forward_mins']}"]

        total_bull = bull_mask.sum()
        total_bear = bear_mask.sum()
        total_signals = total_bull + total_bear
        
        # Skip setups that barely happen to keep output clean
        if total_signals < 50: 
            continue

        # 5. Calculate Points MFE & MAE for all signals
        bull_mfe = f_max - entry_price
        bull_mae = entry_price - f_min
        bear_mfe = entry_price - f_min
        bear_mae = f_max - entry_price

        # Win calculations
        bull_wins = bull_mask & (bull_mfe >= p['tp_pts']) & (bull_mae < p['sl_pts'])
        bear_wins = bear_mask & (bear_mfe >= p['tp_pts']) & (bear_mae < p['sl_pts'])

        total_wins = bull_wins.sum() + bear_wins.sum()
        win_rate = (total_wins / total_signals) * 100
        ev_pts = (win_rate/100 * p['tp_pts']) - ((100-win_rate)/100 * p['sl_pts'])

        # Calculate Average MFE/MAE across all signals in this setup
        avg_mfe = (bull_mfe[bull_mask].sum() + bear_mfe[bear_mask].sum()) / total_signals
        avg_mae = (bull_mae[bull_mask].sum() + bear_mae[bear_mask].sum()) / total_signals

        # 6. Build Row Data
        row_data = {
            'Engulf Type': p['engulfing_type'],
            'Session': p['session'],
            'Trend': p['trend_filter'],
            'Vol Filter': p['vol_filter'],
            'Streak': p['min_streak'],
            'Size Mult': p['size_multi'],
            'Exhaustion': p['exhaustion_max_pct'] if p['use_exhaustion'] else 'None',
            'Hold Mins': p['look_forward_mins'],
            'TP Points': p['tp_pts'],
            'SL Points': p['sl_pts'],
            'Total Trades': total_signals,
            'Avg MFE (Pts)': round(avg_mfe, 2),
            'Avg MAE (Pts)': round(avg_mae, 2),
            'ALL Win%': round(win_rate, 1),
            'EV (Pts/Trade)': round(ev_pts, 3),
        }

        # 7. Dynamic Yearly Breakdown Loop (2010 to 2026)
        for y in unique_years:
            y_mask = d['year'] == y
            y_total = (bull_mask & y_mask).sum() + (bear_mask & y_mask).sum()
            y_wins = (bull_wins & y_mask).sum() + (bear_wins & y_mask).sum()
            y_win_pct = (y_wins / y_total * 100) if y_total > 0 else 0
            
            row_data[f'{y} Trades'] = y_total
            row_data[f'{y} Win%'] = round(y_win_pct, 1)

        results.append(row_data)

        if i % 10000 == 0 and i > 0:
            print(f"  Processed {i} combinations...")

    print(f"\nDone in {time.time() - start_time:.1f} seconds.")
    
    if len(results) == 0:
        print("No combinations met the minimum trade criteria.")
        return

    # Convert to DataFrame
    res_df = pd.DataFrame(results)
    
    # Sort by the most recent full year (e.g., 2023 or 2024) to prioritize modern strategies
    sort_col = '2025 Win%' if '2025 Win%' in res_df.columns else 'ALL Win%'
    res_df = res_df.sort_values(by=sort_col, ascending=False).reset_index(drop=True)
    
    res_df.to_csv("master_grid_results_v3.csv", index=False)
    print("Full breakdown with ALL parameters and years saved to 'master_grid_results_v3.csv'")

if __name__ == "__main__":
    signals_df = load_and_prep_data()
    run_fast_grid(signals_df)