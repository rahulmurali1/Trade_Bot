import upstox_client
import pandas as pd
import time
import numpy as np
from datetime import datetime, timedelta
from upstox_client.rest import ApiException

# =============================
# CONFIG
# =============================
api = upstox_client.HistoryV3Api()
instrument = "NSE_EQ|INE002A01018"
stock_name = "RELIANCE"
interval_type = "minutes"
interval = "5"
start_date = datetime(2022,2,2)
end_date = datetime(2026,3,10)
entry_time = "09:15:00"
end_time = "14:30:00"
RR = 1.5
MIN_RISK_PCT = 0.003
# MAX_RISK_PCT = 0.01

# =============================
# DOWNLOAD DATA
# =============================
all_data = []
current = start_date
while current < end_date:
    next_date = current + timedelta(days=25)
    if next_date > end_date:
        next_date = end_date
    try:
        print(f"Fetching {current.date()} → {next_date.date()}")
        response = api.get_historical_candle_data1(
            instrument,
            interval_type,
            interval,
            next_date.strftime("%Y-%m-%d"),
            current.strftime("%Y-%m-%d")
        )
        if response.data and response.data.candles:
            all_data.extend(response.data.candles)
        time.sleep(0.5)
        current = next_date
    except ApiException as e:
        print("API error:", e)
        print("Retrying in 5 seconds...")
        time.sleep(2)

# =============================
# CREATE DATAFRAME
# =============================
df = pd.DataFrame(all_data, columns=[
    "time","open","high","low","close","volume","oi"
])
df.drop_duplicates(inplace=True)
df["time"] = df["time"].str.replace("+05:30","",regex=False)
df["time"] = pd.to_datetime(df["time"])
df["date"] = df["time"].dt.date
df["clock_time"] = df["time"].dt.time.astype(str)
df = df.drop(columns=["time","oi"], errors="ignore")
df = df[[
    "date","clock_time","open","high","low","close","volume"
]]
df = df.sort_values(by=["date","clock_time"]).reset_index(drop=True)

# =============================
# BASIC FEATURES
# =============================
df["avg_price"] = (df["open"] + df["close"]) / 2
df["avg_order_value"] = df["avg_price"] * df["volume"]

# =============================
# NEW FEATURE COLUMNS
# =============================
df["first_15min_range"] = np.nan
# df["opening_range_pct"] = np.nan
df["distance_from_pp"] = np.nan
df["distance_from_s1"] = np.nan
df["distance_from_r1"] = np.nan
df["price_vs_ema20"] = np.nan
df["price_vs_ema50"] = np.nan
# df["relative_volume"] = np.nan
# df["volume_spike"] = np.nan

# =============================
# EMA FEATURES
# =============================
df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

# =============================
# DAILY OHLC
# =============================
daily_ohlc = df.groupby("date").agg({
    "open":"first",
    "high":"max",
    "low":"min",
    "close":"last"
}).reset_index()
# Red close or green close
df["first_candle_pct"] = np.nan
mask = df["clock_time"] == "09:15:00"
df.loc[mask, "first_candle_pct"] = (
    (df.loc[mask, "close"] - df.loc[mask, "open"]) / df.loc[mask, "open"] * 100
).round(2)

# =============================
# DAILY VOLUME BASELINE
# =============================
daily_volume = df.groupby("date")["volume"].sum()
avg_volume_20 = daily_volume.rolling(20).mean()

# =============================
# CPR LEVELS
# =============================
daily_ohlc["PP"] = (daily_ohlc["high"] + daily_ohlc["low"] + daily_ohlc["close"]) / 3
daily_ohlc["BC"] = (daily_ohlc["high"] + daily_ohlc["low"]) / 2
daily_ohlc["R1"] = 2*daily_ohlc["PP"] - daily_ohlc["low"]
daily_ohlc["S1"] = 2*daily_ohlc["PP"] - daily_ohlc["high"]
daily_ohlc["S2"] = daily_ohlc["PP"] - (daily_ohlc["high"] - daily_ohlc["low"])
daily_ohlc[["PP","BC","S1","S2","R1"]] = \
daily_ohlc[["PP","BC","S1","S2","R1"]].shift(1)
df = df.merge(
    daily_ohlc[['date','PP','BC','S1','S2','R1']],
    on="date",
    how="left"
)
# =============================
# ROUNDING
# =============================
dec1_cols = [
    "open","high","low","close",
    "avg_price","PP","BC","S1","S2"
]
df[dec1_cols] = df[dec1_cols].round(1)
df["avg_order_value"] = df["avg_order_value"].round(0)

# =============================
# CPR DISTANCE FEATURES
# =============================
df["PP_to_BC_pct"] = abs(df["PP"]-df["BC"]) / df["PP"] * 100
df["PP_to_S1_pct"] = abs(df["PP"]-df["S1"]) / df["PP"] * 100
df["PP_to_S2_pct"] = abs(df["PP"]-df["S2"]) / df["PP"] * 100
df[["PP_to_BC_pct","PP_to_S1_pct","PP_to_S2_pct"]] = \
df[["PP_to_BC_pct","PP_to_S1_pct","PP_to_S2_pct"]].round(2)

# =============================
# KEEP CPR ONLY FOR 09:15
# =============================
cpr_cols = [
    "PP","BC","S1","S2",
    "PP_to_BC_pct","PP_to_S1_pct","PP_to_S2_pct"
]
df.loc[df["clock_time"] != "09:15:00", cpr_cols] = np.nan

# =============================
# PP INTERACTION FEATURES
# =============================
df["open_above_pp"] = np.nan
df["touch_pp_directional"] = np.nan
tolerance_pct = 0.1 
mask = df["clock_time"] == "09:15:00"
tol = df.loc[mask, "PP"] * tolerance_pct / 100
# Column 1: whether open is above PP
df.loc[mask, "open_above_pp"] = (
    df.loc[mask, "open"] > df.loc[mask, "PP"]
).astype(int)
# Column 2: directional PP touch
df.loc[mask, "touch_pp_directional"] = np.where(
    df.loc[mask, "open"] > df.loc[mask, "PP"],      # open above PP
    (df.loc[mask, "low"] <= df.loc[mask, "PP"] + tol),    # check downside touch
    (df.loc[mask, "high"] >= df.loc[mask, "PP"] - tol)    # check upside touch
).astype(int)
#closing above PP
df["close_above_pp"] = np.nan
mask = df["clock_time"] == "09:15:00"
df.loc[mask, "close_above_pp"] = (
    df.loc[mask, "close"] > df.loc[mask, "PP"]
).astype(int)

# =============================
# AVG PRICE DISTANCE FROM PP
# =============================
df["avg_pp_dist_pct"] = np.nan
mask = df["clock_time"] == "09:15:00"
df.loc[mask, "avg_pp_dist_pct"] = (
    abs(df.loc[mask, "avg_price"] - df.loc[mask, "PP"]) / df.loc[mask, "PP"] * 100
).round(2)

# =============================
# PREVIOUS DAY CLOSE MAP
# =============================
daily_close = df.groupby("date")["close"].last()
prev_close_map = daily_close.shift(1)
df["prev_close"] = np.nan
df["gap_pct"] = np.nan

# =============================
# TRADE LABEL
# =============================
df["trade_label"] = np.nan
# =============================
# PROCESS EACH DAY
# =============================
for date, group in df.groupby("date"):
    candle = group[group["clock_time"] == entry_time]
    if candle.empty:
        continue
    idx = candle.index[0]
        # =============================
    # FIRST 15 MIN RANGE
    # =============================
    first15 = group[group["clock_time"] < "09:30:00"]
    entry = candle["close"].values[0]
    open = candle['open'].values[0]
    low_915 = candle["low"].values[0]
    high_915 = candle["high"].values[0]
    if not first15.empty:
        high15 = first15["high"].max()
        low15 = first15["low"].min()
        range_pct = (high15 - low15) / entry * 100
        df.at[idx,"first_15min_range"] = round(range_pct,2)
        # df.at[idx,"opening_range_pct"] = round((high15 - low15) / entry * 100,2)
        
    # =============================
    # PREV CLOSE + GAP %
    # =============================
    if date in prev_close_map.index:
        prev_close = prev_close_map.loc[date]
        if pd.notna(prev_close):
            df.at[idx,"prev_close"] = prev_close
            gap_pct = (open - prev_close) / prev_close * 100
            df.at[idx,"gap_pct"] = round(gap_pct,2)

    # =============================
    # RISK CALCULATION
    # =============================
    risk = max(entry - low_915, high_915 - entry)
    risk *= 1.05
    risk_pct = risk / entry
    # if risk_pct < MIN_RISK_PCT or risk_pct > MAX_RISK_PCT:
    if risk_pct < MIN_RISK_PCT:
        df.at[idx,"trade_label"] = 2
        continue
    sl_bull = entry - risk
    sl_bear = entry + risk
    target_bull = entry + RR*risk
    target_bear = entry - RR*risk
    future = group[
        (group["clock_time"] > entry_time) &
        (group["clock_time"] <= end_time)
    ]
    bull_target_time = None
    bull_sl_time = None
    bear_target_time = None
    bear_sl_time = None
    for _,row in future.iterrows():
        high = row["high"]
        low = row["low"]
        t = row["clock_time"]
        # Bullish events
        if bull_sl_time is None and low <= sl_bull:
            bull_sl_time = t
        if bull_target_time is None and high >= target_bull:
            bull_target_time = t
        # Bearish events
        if bear_sl_time is None and high >= sl_bear:
            bear_sl_time = t
        if bear_target_time is None and low <= target_bear:
            bear_target_time = t
        if (bull_target_time or bull_sl_time) and (bear_target_time or bear_sl_time):
            break
    bull_success = False
    bear_success = False
    if bull_target_time and (not bull_sl_time or bull_target_time < bull_sl_time):
        bull_success = True
    if bear_target_time and (not bear_sl_time or bear_target_time < bear_sl_time):
        bear_success = True
            # =============================
    # DISTANCE FROM PIVOT LEVELS
    # =============================
    pp = candle["PP"].values[0]
    s1 = candle["S1"].values[0]
    r1 = candle["R1"].values[0]  # approximate R1 if not present

    if pd.notna(pp):
        df.at[idx,"distance_from_pp"] = round((entry - pp) / entry * 100,2)

    if pd.notna(s1):
        df.at[idx,"distance_from_s1"] = round((entry - s1) / entry * 100,2)

    if pd.notna(r1):
        df.at[idx,"distance_from_r1"] = round((entry - r1) / entry * 100,2)
    # PRICE VS EMA
    # =============================
    ema20 = df.loc[idx,"ema20"]
    ema50 = df.loc[idx,"ema50"]

    if pd.notna(ema20):
        df.at[idx,"price_vs_ema20"] = round((entry - ema20) / ema20 * 100,2)

    if pd.notna(ema50):
        df.at[idx,"price_vs_ema50"] = round((entry - ema50) / ema50 * 100,2)       
    # =============================
    # VOLUME SPIKE
    # =============================
    # =============================
# VOLUME SPIKE
# # =============================
#     vol_series = group["volume"]

#     vol_915 = candle["volume"].values[0]

#     avg_vol = vol_series.rolling(20, min_periods=1).mean().shift(1)
#     avg_vol_915 = avg_vol.loc[idx]

#     if pd.notna(avg_vol_915) and avg_vol_915 > 0:
#         spike = vol_915 / avg_vol_915
#         df.at[idx,"volume_spike"] = round(spike,2)
    # =============================
    # RELATIVE VOLUME
    # =============================
    # if date in avg_volume_20.index:
    #     avg_vol20 = avg_volume_20.loc[date]

    #     if pd.notna(avg_vol20):
    #         today_vol = group["volume"].sum()
    #         df.at[idx,"relative_volume"] = round(today_vol / avg_vol20,2)
    # =============================
    # FINAL LABEL
    # =============================
    if bull_success and not bear_success:
        df.at[idx,"trade_label"] = 1
    elif bear_success and not bull_success:
        df.at[idx,"trade_label"] = 0
    elif bull_success and bear_success:
        if bull_target_time < bear_target_time:
            df.at[idx,"trade_label"] = 1
        else:
            df.at[idx,"trade_label"] = 0
    else:
        df.at[idx,"trade_label"] = 2
            # =============================


# =============================
# FINAL CLEAN
# =============================
df["date"] = pd.to_datetime(df["date"]).dt.date

# =============================
# SAVE
# =============================
df = df.drop(columns=["ema20","ema50"], errors="ignore")
file_name = f"{stock_name}_v14_dataset.xlsx"
df.to_excel(file_name,index=False)
print("Dataset saved:",file_name)
print("Original label distribution:")
print(df["trade_label"].value_counts())



# Give me the complete updated code as per the below code with same logic to get dataset for these 10 stocks: HDFC Bank (NSE_EQ|INE040A01034), Reliance Industries (NSE_EQ|INE002A01018), ICICI Bank (NSE_EQ|INE090A01021), Infosys (NSE_EQ|INE009A01021), Bharti Airtel (NSE_EQ|INE397D01024), Tata Consultancy Services (NSE_EQ|INE467B01029), Larsen & Toubro (NSE_EQ|INE018A01030), ITC Limited (NSE_EQ|INE154A01025), State Bank of India (NSE_EQ|INE062A01020), Axis Bank (NSE_EQ|INE238A01034)

