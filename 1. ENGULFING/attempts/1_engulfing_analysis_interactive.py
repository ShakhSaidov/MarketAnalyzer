import streamlit as st
import pandas as pd
import numpy as np

# --- 1. APP SETUP & DATA LOADING ---
st.set_page_config(page_title="NQ Engulfing Research Lab", layout="wide")
st.title("📈 NQ Candlestick Research Lab")

@st.cache_data # This caches the 5.4M rows in RAM so the app is lightning fast
def load_data():
    # Load 1-min data
    df = pd.read_parquet("data/nq_1min.parquet")
    
    # Convert UTC to US/Pacific (PST/PDT) so your time filters match your brain
    df.index = df.index.tz_localize('UTC').tz_convert('US/Pacific')
    
    # Pre-calculate base candle features
    df['body'] = (df['close'] - df['open']).abs()
    df['total_size'] = df['high'] - df['low']
    df['color'] = np.where(df['close'] >= df['open'], 1, -1)
    
    # Shifted previous values
    df['prev_open'] = df['open'].shift(1)
    df['prev_close'] = df['close'].shift(1)
    df['prev_high'] = df['high'].shift(1)
    df['prev_low'] = df['low'].shift(1)
    df['prev_color'] = df['color'].shift(1)
    df['prev_body'] = df['body'].shift(1)
    df['prev_total_size'] = df['total_size'].shift(1)
    
    # Calculate consecutive streaks (Opposite colors beforehand)
    # This counts consecutive reds before a green, or greens before a red
    streak = df['color'].groupby((df['color'] != df['prev_color']).cumsum()).cumsum()
    df['streak'] = streak.abs()
    df['prev_streak'] = df['streak'].shift(1) # Preceding streak length
    
    return df

df_raw = load_data()

# --- 2. SIDEBAR CONTROLS (YOUR FILTERS) ---
st.sidebar.header("1. Define the Context")
session_filter = st.sidebar.multiselect(
    "Time of Day (PST)",
    options=["All", "Opening Hour (06:30-07:30)", "London Open (00:00-01:00)", "Power Hour (12:00-13:00)"],
    default="All"
)
min_preceding_candles = st.sidebar.slider("Min. Preceding Opposite Candles", 1, 10, 2)

st.sidebar.header("2. Define the Signal")
engulfing_type = st.sidebar.radio("Engulfing Definition", ["Option A: Body Only", "Option B: Body + Wicks"])
size_multiplier = st.sidebar.slider("Signal Candle Size vs Prev Candle", 1.0, 4.0, 1.5, 0.1)

st.sidebar.header("3. Define the Outcome")
look_forward_mins = st.sidebar.slider("Time Limit (Minutes to hold)", 1, 60, 5)
take_profit = st.sidebar.slider("Take Profit (Points)", 1.0, 20.0, 5.0, 0.5)
stop_loss = st.sidebar.slider("Stop Loss (Points)", 1.0, 10.0, 2.0, 0.5)

# --- 3. APPLY LOGIC & FILTERS ---
df = df_raw.copy()

# A. Build the Signal Logic
if engulfing_type == "Option A: Body Only":
    # Green engulfs Red body
    bullish_engulfing = (df['prev_color'] == -1) & (df['color'] == 1) & (df['close'] >= df['prev_open']) & (df['open'] <= df['prev_close'])
    # Red engulfs Green body
    bearish_engulfing = (df['prev_color'] == 1) & (df['color'] == -1) & (df['close'] <= df['prev_open']) & (df['open'] >= df['prev_close'])
else:
    # Green engulfs Red completely (Wicks included)
    bullish_engulfing = (df['prev_color'] == -1) & (df['color'] == 1) & (df['high'] > df['prev_high']) & (df['low'] < df['prev_low'])
    # Red engulfs Green completely
    bearish_engulfing = (df['prev_color'] == 1) & (df['color'] == -1) & (df['high'] > df['prev_high']) & (df['low'] < df['prev_low'])

# B. Apply Context Conditions (Streak & Size)
size_condition = df['total_size'] >= (df['prev_total_size'] * size_multiplier)
streak_condition = df['prev_streak'] >= min_preceding_candles

df['is_bull_signal'] = bullish_engulfing & size_condition & streak_condition
df['is_bear_signal'] = bearish_engulfing & size_condition & streak_condition

# C. Filter by Session (PST Time)
if "All" not in session_filter:
    mask = pd.Series(False, index=df.index)
    if "Opening Hour (06:30-07:30)" in session_filter:
        mask |= (df.index.hour == 6) & (df.index.minute >= 30) | (df.index.hour == 7) & (df.index.minute <= 30)
    if "London Open (00:00-01:00)" in session_filter:
        mask |= (df.index.hour == 0)
    if "Power Hour (12:00-13:00)" in session_filter:
        mask |= (df.index.hour == 12)
    df = df[mask]

# --- 4. CALCULATE FUTURE EXCURSION (MFE/MAE) ---
# Rolling forward max/min for the next 'look_forward_mins'
# .shift(-W) pulls future W-period data into the current row
forward_max = df_raw['high'].rolling(window=look_forward_mins).max().shift(-look_forward_mins)
forward_min = df_raw['low'].rolling(window=look_forward_mins).min().shift(-look_forward_mins)

# Map forward metrics back to our filtered DataFrame
df['forward_max'] = forward_max[df.index]
df['forward_min'] = forward_min[df.index]

# Calculate Excursions from the Close of the signal candle
df['MFE_Bull'] = df['forward_max'] - df['close']
df['MAE_Bull'] = df['close'] - df['forward_min']

df['MFE_Bear'] = df['close'] - df['forward_min']
df['MAE_Bear'] = df['forward_max'] - df['close']

# Determine Wins/Losses
# A win is when MFE hits Take Profit BEFORE time runs out, and MAE doesn't stop us out.
# (Note: In 1min data, we assume if MFE > TP and MAE < SL, it's highly favorable. For strictness, if BOTH are hit, we count it as a stop-out loss to be conservative).
df['Bull_Win'] = (df['MFE_Bull'] >= take_profit) & (df['MAE_Bull'] < stop_loss)
df['Bear_Win'] = (df['MFE_Bear'] >= take_profit) & (df['MAE_Bear'] < stop_loss)

# --- 5. RESULTS & DASHBOARD ---
bull_signals = df[df['is_bull_signal']]
bear_signals = df[df['is_bear_signal']]

total_bull = len(bull_signals)
total_bear = len(bear_signals)
bull_wins = bull_signals['Bull_Win'].sum()
bear_wins = bear_signals['Bear_Win'].sum()

st.header("Results Dashboard")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Signals Found", total_bull + total_bear)
with col2:
    overall_win_rate = ((bull_wins + bear_wins) / (total_bull + total_bear) * 100) if (total_bull + total_bear) > 0 else 0
    st.metric("Overall Win Rate", f"{overall_win_rate:.2f}%")
with col3:
    # A simple Risk/Reward math check
    rr_ratio = take_profit / stop_loss
    st.metric("Risk / Reward Ratio", f"{rr_ratio:.2f} : 1")

st.divider()

st.subheader("Breakdown by Direction")
bc1, bc2 = st.columns(2)
with bc1:
    st.markdown("### 🟢 Bullish Signals")
    st.write(f"**Occurrences:** {total_bull}")
    if total_bull > 0:
        st.write(f"**Win Rate:** {(bull_wins / total_bull * 100):.2f}%")
        st.write(f"**Wins:** {bull_wins} | **Losses/Timeouts:** {total_bull - bull_wins}")

with bc2:
    st.markdown("### 🔴 Bearish Signals")
    st.write(f"**Occurrences:** {total_bear}")
    if total_bear > 0:
        st.write(f"**Win Rate:** {(bear_wins / total_bear * 100):.2f}%")
        st.write(f"**Wins:** {bear_wins} | **Losses/Timeouts:** {total_bear - bear_wins}")

st.divider()
st.subheader("Sample of Winning Trades (Bullish)")
if total_bull > 0:
    st.dataframe(bull_signals[bull_signals['Bull_Win']][['open', 'high', 'low', 'close', 'total_size', 'prev_streak', 'MFE_Bull', 'MAE_Bull']].head(10))