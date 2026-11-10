import pandas as pd
import numpy as np
import itertools
import time

def load_and_prep_data():
    print("1. Loading Parquet Data & Calculating Features...")
    df = pd.read_parquet("data/nq_1min.parquet")
    
    df.index = df.index.tz_localize('UTC').tz_convert('US/Pacific')
    df['day_of_week'] = df.index.dayofweek 
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

    df['body'] = (df['close'] - df['open']).abs()
    df['total_size'] = df['high'] - df['low']
    df['color'] = np.where(df['close'] >= df['open'], 1, -1)
    
    df['prev_open'] = df['open'].shift(1)
    df['prev_close'] = df['close'].shift(1)
    df['prev_high'] = df['high'].shift(1)
    df['prev_low'] = df['low'].shift(1)
    df['prev_color'] = df['color'].shift(1)
    df['prev_total_size'] = df['total_size'].shift(1)
    
    df['streak'] = df['color'].groupby((df['color'] != df['prev_color']).cumsum()).cumsum().abs()
    df['prev_streak'] = df['streak'].shift(1)
    
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['vol_ma_20'] = df['volume'].rolling(window=20).mean()
    df['vol_spike_ratio'] = df['volume'] / df['vol_ma_20']
    df['size_ratio'] = df['total_size'] / df['prev_total_size'].replace(0, 0.0001)

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

    print("3. Calculating Future Outcomes...")
    forward_times = [5, 10, 15] # Update this list if you add new hold times below
    for mins in forward_times:
        signals_df[f'fwd_max_{mins}'] = df['high'].rolling(window=mins).max().shift(-mins)[signals_df.index]
        signals_df[f'fwd_min_{mins}'] = df['low'].rolling(window=mins).min().shift(-mins)[signals_df.index]
        signals_df[f'fwd_max_exh_{mins}'] = df['high'].rolling(window=mins).max().shift(-mins - 1)[signals_df.index]
        signals_df[f'fwd_min_exh_{mins}'] = df['low'].rolling(window=mins).min().shift(-mins - 1)[signals_df.index]

    return signals_df

def run_fast_grid(df):
    # ADD MORE PARAMETERS TO THIS GRID AS NEEDED
    param_grid = {
        'engulfing_type': ['body', 'wick'],
        'min_streak': [1, 2, 3, 4, 5],
        'size_multi': [1.5, 2.0, 2.5, 3.0],
        'use_exhaustion': [False, True],
        'exhaustion_max_pct': [0.5, 0.25, 0.1], 
        'trend_filter': ['All', 'With_Trend'], 
        'vol_filter': ['All', 'Spike_1.5x'], 
        'session': ['All', 'US_Open', 'London'], 
        'look_forward_mins': [5, 10, 15],
        'tp_pct': [0.01, 0.02, 0.04, 0.05, 0.06, 0.08, 0.1], 
        'sl_pct': [0.01, 0.02, 0.04, 0.05, 0.06, 0.08, 0.1]
    }

    d = {col: df[col].values for col in df.columns}
    combinations = list(itertools.product(*param_grid.values()))
    print(f"\n🚀 Running {len(combinations)} combinations...\n")

    results = []
    start_time = time.time()

    for i, combo in enumerate(combinations):
        p = dict(zip(param_grid.keys(), combo))
        
        if not p['use_exhaustion'] and p['exhaustion_max_pct'] != param_grid['exhaustion_max_pct'][0]:
            continue 

        if p['engulfing_type'] == 'body':
            bull_mask = d['is_bull_body']
            bear_mask = d['is_bear_body']
        else:
            bull_mask = d['is_bull_wick']
            bear_mask = d['is_bear_wick']

        ctx_mask = (d['prev_streak'] >= p['min_streak']) & (d['size_ratio'] >= p['size_multi'])
        if p['vol_filter'] == 'Spike_1.5x': ctx_mask = ctx_mask & (d['vol_spike_ratio'] >= 1.5)
        if p['session'] != 'All': ctx_mask = ctx_mask & (d['session'] == p['session'])

        bull_mask = bull_mask & ctx_mask
        bear_mask = bear_mask & ctx_mask

        if p['trend_filter'] == 'With_Trend':
            bull_mask = bull_mask & (d['close'] > d['sma_50'])
            bear_mask = bear_mask & (d['close'] < d['sma_50'])

        if p['use_exhaustion']:
            exh_mask = d['exh_pct'] <= p['exhaustion_max_pct']
            bull_mask = bull_mask & exh_mask
            bear_mask = bear_mask & exh_mask
            entry_price = d['next_close']
            f_max = d[f"fwd_max_exh_{p['look_forward_mins']}"]
            f_min = d[f"fwd_min_exh_{p['look_forward_mins']}"]
        else:
            entry_price = d['close']
            f_max = d[f"fwd_max_{p['look_forward_mins']}"]
            f_min = d[f"fwd_min_{p['look_forward_mins']}"]

        bull_mfe = (f_max - entry_price) / entry_price * 100
        bull_mae = (entry_price - f_min) / entry_price * 100
        bear_mfe = (entry_price - f_min) / entry_price * 100
        bear_mae = (f_max - entry_price) / entry_price * 100

        bull_wins = bull_mask & (bull_mfe >= p['tp_pct']) & (bull_mae < p['sl_pct'])
        bear_wins = bear_mask & (bear_mfe >= p['tp_pct']) & (bear_mae < p['sl_pct'])

        total_bull = bull_mask.sum()
        total_bear = bear_mask.sum()
        total_signals = total_bull + total_bear
        
        if total_signals < 30: continue

        total_wins = bull_wins.sum() + bear_wins.sum()
        win_rate = (total_wins / total_signals) * 100
        ev_pct = (win_rate/100 * p['tp_pct']) - ((100-win_rate)/100 * p['sl_pct'])

        # Bull specific stats
        bull_wr = (bull_wins.sum() / total_bull * 100) if total_bull > 0 else 0
        bull_ev = (bull_wr/100 * p['tp_pct']) - ((100-bull_wr)/100 * p['sl_pct']) if total_bull > 0 else 0

        # Bear specific stats
        bear_wr = (bear_wins.sum() / total_bear * 100) if total_bear > 0 else 0
        bear_ev = (bear_wr/100 * p['tp_pct']) - ((100-bear_wr)/100 * p['sl_pct']) if total_bear > 0 else 0

        results.append({
            'Engulf Type': p['engulfing_type'],
            'Session': p['session'],
            'Trend': p['trend_filter'],
            'Vol Filter': p['vol_filter'],
            'Streak': p['min_streak'],
            'Size Mult': p['size_multi'],
            'Exhaustion': p['exhaustion_max_pct'] if p['use_exhaustion'] else 'None',
            'Hold': p['look_forward_mins'],
            'TP %': p['tp_pct'],
            'SL %': p['sl_pct'],
            'Total': total_signals,
            'ALL Win%': round(win_rate, 2),
            'ALL EV%': round(ev_pct, 4),
            'Bull #': total_bull,
            'Bull Win%': round(bull_wr, 2),
            'Bull EV%': round(bull_ev, 4),
            'Bear #': total_bear,
            'Bear Win%': round(bear_wr, 2),
            'Bear EV%': round(bear_ev, 4)
        })

        if i % 10000 == 0 and i > 0:
            print(f"  Processed {i} combinations...")

    print(f"\nDone in {time.time() - start_time:.1f} seconds.")
    res_df = pd.DataFrame(results).sort_values(by='ALL EV%', ascending=False).reset_index(drop=True)
    res_df.to_csv("master_grid_results_v3.csv", index=False)
    print("Full breakdown saved to 'master_grid_results_v3.csv'")

if __name__ == "__main__":
    signals_df = load_and_prep_data()
    run_fast_grid(signals_df)