import pandas as pd
import numpy as np

# ==========================================
# PASTE YOUR DISCOVERED SETUP HERE
# ==========================================
PARAMS = {
    'engulfing_type': 'wick',
    'min_streak': 3,
    'size_multi': 1.1,
    'trend_filter': 'All', 
    'vol_filter': 'All',   
    'session': 'US_Open',         
    'look_forward_mins': 60,
    'tp_pts': 1.0,  
    'sl_pts': 7.0   
}
# ==========================================

def extract_timeline(p):
    print("Loading Data for Timeline Extraction...")
    df = pd.read_parquet("data/nq_1min.parquet")
    df.index = df.index.tz_localize('UTC').tz_convert('US/Pacific')
    
    df['hour'], df['minute'] = df.index.hour, df.index.minute
    cond = [
        ((df['hour'] == 0) | (df['hour'] == 1)),
        ((df['hour'] == 6) & (df['minute'] >= 30)) | (df['hour'] == 7) | ((df['hour'] == 8) & (df['minute'] < 30)),
        ((df['hour'] == 8) & (df['minute'] >= 30)) | (df['hour'] == 9) | (df['hour'] == 10) | ((df['hour'] == 11) & (df['minute'] < 30)),
        ((df['hour'] == 11) & (df['minute'] >= 30)) | (df['hour'] == 12)
    ]
    df['session'] = np.select(cond, ['London', 'US_Open', 'Midday', 'US_Close'], default='Other')
    df['total_size'] = df['high'] - df['low']
    df['color'] = np.where(df['close'] >= df['open'], 1, -1)
    df['prev_open'], df['prev_close'] = df['open'].shift(1), df['close'].shift(1)
    df['prev_high'], df['prev_low'] = df['high'].shift(1), df['low'].shift(1)
    df['prev_color'] = df['color'].shift(1)
    df['prev_total_size'] = df['total_size'].shift(1)
    df['streak'] = df['color'].groupby((df['color'] != df['prev_color']).cumsum()).cumsum().abs()
    df['prev_streak'] = df['streak'].shift(1)
    df['size_ratio'] = df['total_size'] / df['prev_total_size'].replace(0, 0.0001)

    # Base Signals
    if p['engulfing_type'] == 'body':
        bull_base = (df['prev_color'] == -1) & (df['color'] == 1) & (df['close'] >= df['prev_open']) & (df['open'] <= df['prev_close'])
        bear_base = (df['prev_color'] == 1) & (df['color'] == -1) & (df['close'] <= df['prev_open']) & (df['open'] >= df['prev_close'])
    else:
        bull_base = (df['prev_color'] == -1) & (df['color'] == 1) & (df['high'] > df['prev_high']) & (df['low'] < df['prev_low'])
        bear_base = (df['prev_color'] == 1) & (df['color'] == -1) & (df['high'] > df['prev_high']) & (df['low'] < df['prev_low'])

    # Context
    ctx = (df['prev_streak'] >= p['min_streak']) & (df['size_ratio'] >= p['size_multi'])
    if p['session'] != 'All': ctx &= (df['session'] == p['session'])
    bull_base, bear_base = bull_base & ctx, bear_base & ctx

    f_max = df['high'].rolling(window=p['look_forward_mins']).max().shift(-p['look_forward_mins'])
    f_min = df['low'].rolling(window=p['look_forward_mins']).min().shift(-p['look_forward_mins'])

    trades_df = df[bull_base | bear_base].copy()
    trades_df['Engulfing_Signal'] = np.where(bull_base[trades_df.index], 'Bullish', 'Bearish')
    
    # 🚨 INVERSE LOGIC: Trade opposite the signal 🚨
    trades_df['Direction_Traded'] = np.where(trades_df['Engulfing_Signal'] == 'Bullish', 'SHORT', 'LONG')

    trades_df['MFE_Pts'] = np.where(trades_df['Direction_Traded'] == 'SHORT', trades_df['close'] - f_min[trades_df.index], f_max[trades_df.index] - trades_df['close'])
    trades_df['MAE_Pts'] = np.where(trades_df['Direction_Traded'] == 'SHORT', f_max[trades_df.index] - trades_df['close'], trades_df['close'] - f_min[trades_df.index])

    trades_df['Outcome'] = 'TIMEOUT'
    win_mask = (trades_df['MFE_Pts'] >= p['tp_pts']) & (trades_df['MAE_Pts'] < p['sl_pts'])
    loss_mask = (trades_df['MAE_Pts'] >= p['sl_pts'])
    
    trades_df.loc[win_mask, 'Outcome'] = 'WIN'
    trades_df.loc[loss_mask, 'Outcome'] = 'LOSS'

    output = trades_df[['Engulfing_Signal', 'Direction_Traded', 'close', 'MFE_Pts', 'MAE_Pts', 'Outcome']].copy()
    output.rename(columns={'close': 'Entry_Price'}, inplace=True)
    output.index.name = 'Date_Time_PST'
    
    output.to_csv("inverse_timeline_check.csv")
    print(f"\nSaved {len(output)} trades to 'inverse_timeline_check.csv'.")
    print("Here are the last 15 trades:")
    print(output.tail(15))

if __name__ == "__main__":
    extract_timeline(PARAMS)