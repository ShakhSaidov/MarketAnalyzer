import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import itertools

st.set_page_config(page_title="NQ Quant Lab: Dynamic Matrix", layout="wide")

RAW_TRADES_PATH = "./raw_trades.parquet"

@st.cache_data
def load_raw_trades():
    if not os.path.exists(RAW_TRADES_PATH):
        return None
    return pd.read_parquet(RAW_TRADES_PATH)

df = load_raw_trades()

if df is None:
    st.error(f"❌ Missing file: {RAW_TRADES_PATH}. Please run `python compute.py` first.")
    st.stop()

st.title("🔬 NQ Quant Lab: Dynamic OLAP Matrix")
st.markdown("Dynamically aggregate dimensions to boost sample sizes, adjust custom R:R metrics, and verify year-over-year stability.")

# ==========================================
# 1. SIDEBAR: GLOBAL FILTERS
# ==========================================
st.sidebar.header("1. Global Data Filters")
min_yr, max_yr = int(df['Year'].min()), int(df['Year'].max())
year_range = st.sidebar.slider("Year Range", min_value=min_yr, max_value=max_yr, value=(min_yr, max_yr))

tf_list = st.sidebar.multiselect("Timeframes", df['TF'].unique(), default=[5])
nth_list = st.sidebar.multiselect("N-th Bar", df['Nth'].unique(), default=[1, 2])
action_list = st.sidebar.multiselect("Triggers", df['Trigger'].unique(), default=["Long Breakout", "Short Breakdown"])
sl_list = st.sidebar.multiselect("Stop Losses", df['SL_Set'].unique(), default=[15.0])
session_list = st.sidebar.multiselect("Session Hrs", df['Sesh_Hrs'].unique(), default=[2.0])
days_filter = st.sidebar.multiselect("Include Days", ['Mon', 'Tue', 'Wed', 'Thu', 'Fri'], default=['Mon', 'Tue', 'Wed', 'Thu', 'Fri'])

# Filter Data Immediately
mask = (
    (df['Year'] >= year_range[0]) & (df['Year'] <= year_range[1]) &
    (df['TF'].isin(tf_list)) & (df['Nth'].isin(nth_list)) &
    (df['Trigger'].isin(action_list)) & (df['SL_Set'].isin(sl_list)) &
    (df['Sesh_Hrs'].isin(session_list)) & (df['DOW_Raw'].isin(days_filter))
)
filtered_df = df[mask].copy()

if filtered_df.empty:
    st.warning("No data found for these filters.")
    st.stop()

# ==========================================
# 2. SIDEBAR: CUSTOM R:R METRICS
# ==========================================
st.sidebar.header("2. Risk/Reward Potential")
rr_multiplier = st.sidebar.slider(
    "Target Multiplier (x SL)", 
    min_value=1.0, max_value=5.0, value=2.0, step=0.5,
    help="Defines what constitutes a 'Potential Win'. E.g., 2.0 x 15pt SL = 30pt Target."
)

filtered_df['Target_Pts'] = filtered_df['SL_Set'] * rr_multiplier
filtered_df['Is_Potential_Win'] = filtered_df['MFE'] >= filtered_df['Target_Pts']
filtered_df['Is_Loss'] = filtered_df['SL_Hit'] & (~filtered_df['Is_Potential_Win'])

# ==========================================
# 3. SIDEBAR: DYNAMIC GROUPING ENGINE
# ==========================================
st.sidebar.header("3. Grouping Engine (Dimensions)")
st.sidebar.info("Uncheck a box to AGGREGATE that dimension (combining data to boost sample size).")

col1, col2 = st.sidebar.columns(2)
split_tf = col1.checkbox("Split by TF", value=True)
split_nth = col2.checkbox("Split by N-th Bar", value=True)
split_trig = col1.checkbox("Split by Trigger", value=True)
split_sl = col2.checkbox("Split by Stop Loss", value=True)
split_prev = col1.checkbox("Split Prev Day Color", value=False) # Aggregated by default!
split_sig = col2.checkbox("Split Signal Color", value=False)  # Aggregated by default!

dow_mode = st.sidebar.radio("Day of Week Grouping Mode:", [
    "Combined as One (Aggregated)", 
    "Individual Days (Split)", 
    "Powerset (All Combinations)"
])

# Build Grouping Columns dynamically
group_cols = []
if split_tf: group_cols.append('TF')
if split_nth: group_cols.append('Nth')
if split_trig: group_cols.append('Trigger')
if split_sl: group_cols.append('SL_Set')
if split_prev: group_cols.append('Prev_Color')
if split_sig: group_cols.append('Sig_Color')

# ==========================================
# MATRIX CALCULATION
# ==========================================
matrix_rows = []

def process_group(name_tuple, group, dow_label):
    total = len(group)
    if total < 5: return None
    
    potential_wins = group['Is_Potential_Win'].sum()
    sl_losses = group['Is_Loss'].sum()
    win_rate = potential_wins / total
    loss_rate = sl_losses / total
    
    # Calculate Mechanical Expectancy Proxy
    # Assuming rigid execution: Win = Target Pts, Loss = - SL Pts, Tie/Timeout = Session PnL
    avg_sl = group['SL_Set'].mean()
    avg_target = group['Target_Pts'].mean()
    
    expectancy = (win_rate * avg_target) - (loss_rate * avg_sl)
    
    row = {'DOW_Group': dow_label}
    
    # Map the dynamic group columns back to the row
    if isinstance(name_tuple, tuple):
        for col_name, val in zip(group_cols, name_tuple):
            row[col_name] = val
    else: # Only 1 grouping column edge case
        if group_cols: row[group_cols[0]] = name_tuple

    row.update({
        'Occurrences': total,
        'Survival_Rate': round(100.0 - (group['SL_Hit'].mean() * 100), 1),
        f'Win_Rate_{rr_multiplier}x': round(win_rate * 100, 1),
        'Loss_Rate': round(loss_rate * 100, 1),
        'Exp_Pts_Per_Trade': round(expectancy, 2),
        'Avg_MFE': round(group['MFE'].mean(), 1)
    })
    return row

with st.spinner("Crunching Dynamic OLAP Cube..."):
    # MODE 1: Powerset
    if dow_mode == "Powerset (All Combinations)":
        dow_combinations = []
        for r in range(1, len(days_filter) + 1):
            dow_combinations.extend(itertools.combinations(days_filter, r))
            
        for dow_combo in dow_combinations:
            combo_name = "+".join(dow_combo)
            subset = filtered_df[filtered_df['DOW_Raw'].isin(dow_combo)]
            
            if not group_cols: # If EVERYTHING is aggregated except DOW
                row = process_group(None, subset, combo_name)
                if row: matrix_rows.append(row)
            else:
                grouped = subset.groupby(group_cols)
                for name, group in grouped:
                    row = process_group(name, group, combo_name)
                    if row: matrix_rows.append(row)
                    
    # MODE 2: Individual Days
    elif dow_mode == "Individual Days (Split)":
        temp_group_cols = ['DOW_Raw'] + group_cols
        grouped = filtered_df.groupby(temp_group_cols)
        for name, group in grouped:
            dow_label = name[0]
            sub_name = name[1:] if len(name) > 2 else (name[1] if len(name) == 2 else None)
            row = process_group(sub_name, group, dow_label)
            if row: matrix_rows.append(row)
            
    # MODE 3: Combined as One
    else: 
        combo_name = "+".join(days_filter)
        if not group_cols:
            row = process_group(None, filtered_df, combo_name)
            if row: matrix_rows.append(row)
        else:
            grouped = filtered_df.groupby(group_cols)
            for name, group in grouped:
                row = process_group(name, group, combo_name)
                if row: matrix_rows.append(row)

matrix_df = pd.DataFrame(matrix_rows)

# ==========================================
# UI RENDERING
# ==========================================
if matrix_df.empty:
    st.warning("Not enough data to form valid statistical groups.")
    st.stop()

# Reorder columns dynamically so metrics are always at the end
metric_cols = ['Occurrences', 'Survival_Rate', f'Win_Rate_{rr_multiplier}x', 'Loss_Rate', 'Exp_Pts_Per_Trade', 'Avg_MFE']
front_cols = [c for c in matrix_df.columns if c not in metric_cols]
matrix_df = matrix_df[front_cols + metric_cols]

matrix_df = matrix_df.sort_values(by=['Exp_Pts_Per_Trade', f'Win_Rate_{rr_multiplier}x'], ascending=False)

tab1, tab2, tab3 = st.tabs(["📊 Master Aggregate Matrix", "📅 Year-Over-Year Consistency", "📉 Deep Dive Distributions"])

with tab1:
    st.markdown("### Master Aggregate Matrix")
    st.markdown("""
    By unchecking grouping options in the sidebar, you combine variables. Notice how **Occurrences** shoot up, 
    giving you highly robust statistical confidence.
    * **Exp_Pts_Per_Trade**: Calculated Expectancy. If you rigidly took profits at your Target and cut losses at your SL, how many points do you average per trade over time?
    """)
    st.dataframe(
        matrix_df, 
        use_container_width=True, 
        hide_index=True,
        column_config={
            f'Win_Rate_{rr_multiplier}x': st.column_config.ProgressColumn(f"Hit {rr_multiplier}x SL Target %", format="%.1f%%", min_value=0, max_value=100),
            'Survival_Rate': st.column_config.ProgressColumn("SL Survival %", format="%.1f%%", min_value=0, max_value=100),
            'Exp_Pts_Per_Trade': st.column_config.NumberColumn("Est. Expectancy (Pts)", format="%.2f")
        }
    )

with tab2:
    st.markdown("### Year-Over-Year Stability")
    st.markdown("Select a baseline configuration below to see if the statistical edge holds up across different market regimes (Bull Markets, Bear Markets, Low Volatility).")
    
    # Let user select one setup from their generated matrix to analyze YoY
    if len(front_cols) > 0:
        st.info("Select a specific configuration to test its yearly stability:")
        
        cols = st.columns(min(len(front_cols), 4))
        yoy_filters = {}
        for i, col_name in enumerate(front_cols):
            with cols[i % 4]:
                unique_vals = matrix_df[col_name].unique()
                yoy_filters[col_name] = st.selectbox(f"{col_name}", unique_vals)
                
        # Filter raw data for YoY based on selection
        yoy_mask = pd.Series(True, index=filtered_df.index)
        for col_name, val in yoy_filters.items():
            if col_name == 'DOW_Group':
                # Handle dow group string
                days_in_group = val.split('+')
                yoy_mask = yoy_mask & filtered_df['DOW_Raw'].isin(days_in_group)
            else:
                yoy_mask = yoy_mask & (filtered_df[col_name] == val)
                
        yoy_df = filtered_df[yoy_mask]
        
        if not yoy_df.empty:
            yoy_grouped = yoy_df.groupby('Year').agg(
                Occurrences=('Year', 'count'),
                Potential_Wins=('Is_Potential_Win', 'sum'),
                Losses=('Is_Loss', 'sum'),
                Avg_MFE=('MFE', 'mean')
            ).reset_index()
            
            yoy_grouped['Win_Rate'] = (yoy_grouped['Potential_Wins'] / yoy_grouped['Occurrences']) * 100
            
            fig_yoy = px.bar(
                yoy_grouped, x='Year', y='Win_Rate', 
                title=f"Target Hit Rate ({rr_multiplier}x SL) by Year",
                text=yoy_grouped['Win_Rate'].apply(lambda x: f"{x:.1f}%"),
                color='Win_Rate', color_continuous_scale='RdYlGn'
            )
            fig_yoy.update_traces(textposition='outside')
            st.plotly_chart(fig_yoy, use_container_width=True)
            st.dataframe(yoy_grouped.round(1), use_container_width=True, hide_index=True)
        else:
            st.warning("No data matches this exact configuration.")

with tab3:
    st.markdown("### Risk vs. Reward Distribution (For Selected Setup)")
    if 'yoy_df' in locals() and not yoy_df.empty:
        fig_scatter = px.scatter(
            yoy_df, x="Year", y="MFE", color="Is_Loss",
            color_discrete_map={True: "red", False: "green"},
            title="Max Favorable Excursion over Time",
            opacity=0.6
        )
        # Draw target line
        avg_target = yoy_df['Target_Pts'].mean()
        fig_scatter.add_hline(y=avg_target, line_dash="dash", line_color="blue", annotation_text=f"{rr_multiplier}x Target")
        st.plotly_chart(fig_scatter, use_container_width=True)