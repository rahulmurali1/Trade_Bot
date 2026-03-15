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

entry_time = "09:20:00"
prev_time = "09:15:00"
end_time = "14:30:00"

RR = 2
MIN_RISK_PCT = 0.002
MAX_RISK_PCT = 0.01

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

        time.sleep(1)
        current = next_date

    except ApiException as e:

        print("API error:", e)
        print("Retrying in 5 seconds...")
        time.sleep(5)

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

# =============================
# BASIC FEATURES
# =============================

df["avg_price"] = (df["open"] + df["close"]) / 2
df["avg_order_value"] = df["avg_price"] * df["volume"]

# =============================
# DAILY OHLC
# =============================

daily_ohlc = df.groupby("date").agg({
    "open":"first",
    "high":"max",
    "low":"min",
    "close":"last"
}).reset_index()

# =============================
# CPR LEVELS
# =============================

daily_ohlc["PP"] = (daily_ohlc["high"] + daily_ohlc["low"] + daily_ohlc["close"]) / 3
daily_ohlc["BC"] = (daily_ohlc["high"] + daily_ohlc["low"]) / 2
daily_ohlc["TC"] = 2*daily_ohlc["PP"] - daily_ohlc["BC"]

daily_ohlc["S1"] = 2*daily_ohlc["PP"] - daily_ohlc["high"]
daily_ohlc["R1"] = 2*daily_ohlc["PP"] - daily_ohlc["low"]
daily_ohlc["S2"] = daily_ohlc["PP"] - (daily_ohlc["high"] - daily_ohlc["low"])
daily_ohlc["R2"] = daily_ohlc["PP"] + (daily_ohlc["high"] - daily_ohlc["low"])

df = df.merge(
    daily_ohlc[['date','PP','BC','TC','S1','R1','S2','R2']],
    on="date",
    how="left"
)

# =============================
# SORT
# =============================

df = df.sort_values(by=["date","clock_time"]).reset_index(drop=True)

# =============================
# ROUNDING
# =============================

dec1_cols = [
    "open","high","low","close",
    "avg_price","PP","BC","TC","S1","R1","S2","R2"
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
# KEEP CPR ONLY FOR 9:20
# =============================

cpr_cols = [
    "PP","BC","TC","S1","R1","S2","R2",
    "PP_to_BC_pct","PP_to_S1_pct","PP_to_S2_pct"
]

df.loc[df["clock_time"] != "09:20:00", cpr_cols] = np.nan

# =============================
# PREVIOUS DAY CLOSE MAP
# =============================

daily_close = df.groupby("date")["close"].last()
prev_close_map = daily_close.shift(1)

df["prev_close"] = np.nan
df["gap_pct"] = np.nan

# =============================
# TRADE COLUMNS
# =============================

df["trade_label"] = np.nan
df["trade_type"] = ""

df["entry_price"] = np.nan
df["stop_loss"] = np.nan
df["target_price"] = np.nan

df["bull_target_time"] = None
df["bull_sl_time"] = None
df["bear_target_time"] = None
df["bear_sl_time"] = None

# Distance features
df["dist_915_high_pct"] = np.nan
df["dist_915_low_pct"] = np.nan
df["dist_915_max_pct"] = np.nan

# =============================
# PROCESS EACH DAY
# =============================

for date, group in df.groupby("date"):

    candle_915 = group[group["clock_time"] == prev_time]
    candle_920 = group[group["clock_time"] == entry_time]

    if candle_915.empty or candle_920.empty:
        continue

    idx = candle_920.index[0]

    entry = candle_920["open"].values[0]

    low_915 = candle_915["low"].values[0]
    high_915 = candle_915["high"].values[0]

    # =============================
    # PREV CLOSE + GAP %
    # =============================

    if date in prev_close_map.index:

        prev_close = prev_close_map.loc[date]

        if pd.notna(prev_close):

            df.at[idx,"prev_close"] = prev_close

            gap_pct = (entry - prev_close) / prev_close * 100
            df.at[idx,"gap_pct"] = round(gap_pct,3)

    # =============================
    # DISTANCE FEATURES
    # =============================

    dist_high_pct = abs(entry - high_915) / entry * 100
    dist_low_pct = abs(entry - low_915) / entry * 100
    dist_max_pct = max(dist_high_pct, dist_low_pct)

    df.at[idx,"dist_915_high_pct"] = round(dist_high_pct,3)
    df.at[idx,"dist_915_low_pct"] = round(dist_low_pct,3)
    df.at[idx,"dist_915_max_pct"] = round(dist_max_pct,3)

    # =============================
    # RISK CALCULATION
    # =============================

    risk = max(entry - low_915, high_915 - entry)
    risk_pct = risk / entry

    if risk_pct < MIN_RISK_PCT or risk_pct > MAX_RISK_PCT:

        df.at[idx,"trade_label"] = 2
        df.at[idx,"trade_type"] = "no_signal"
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

        if bull_target_time is None and high >= target_bull:
            bull_target_time = t

        if bull_sl_time is None and low <= sl_bull:
            bull_sl_time = t

        if bear_target_time is None and low <= target_bear:
            bear_target_time = t

        if bear_sl_time is None and high >= sl_bear:
            bear_sl_time = t

    df.at[idx,"bull_target_time"] = bull_target_time
    df.at[idx,"bull_sl_time"] = bull_sl_time
    df.at[idx,"bear_target_time"] = bear_target_time
    df.at[idx,"bear_sl_time"] = bear_sl_time

    bull_success = False
    bear_success = False

    if bull_target_time and (not bull_sl_time or bull_target_time < bull_sl_time):
        bull_success = True

    if bear_target_time and (not bear_sl_time or bear_target_time < bear_sl_time):
        bear_success = True

    if bull_success and not bear_success:

        df.loc[idx,["trade_label","trade_type","entry_price","stop_loss","target_price"]] = \
        [1,"bullish",entry,sl_bull,target_bull]

    elif bear_success and not bull_success:

        df.loc[idx,["trade_label","trade_type","entry_price","stop_loss","target_price"]] = \
        [0,"bearish",entry,sl_bear,target_bear]

    elif bull_success and bear_success:

        if bull_target_time < bear_target_time:

            df.loc[idx,["trade_label","trade_type","entry_price","stop_loss","target_price"]] = \
            [1,"bullish",entry,sl_bull,target_bull]

        else:

            df.loc[idx,["trade_label","trade_type","entry_price","stop_loss","target_price"]] = \
            [0,"bearish",entry,sl_bear,target_bear]

    else:

        df.at[idx,"trade_label"] = 2
        df.at[idx,"trade_type"] = "no_signal"

# =============================
# FINAL CLEAN
# =============================

df[["entry_price","stop_loss","target_price"]] = \
df[["entry_price","stop_loss","target_price"]].round(1)

df["date"] = pd.to_datetime(df["date"]).dt.date

# =============================
# SAVE
# =============================
# df2 = df[df["clock_time"] == "09:20:00"].copy()

# df2 = df[df["clock_time"].isin(["09:15:00", "09:20:00"])].copy()
# df2 = df.sort_values(by=["date", "clock_time"]).reset_index(drop=True)

file_name = f"{stock_name}_dataset.xlsx"

df.to_excel(file_name,index=False)

print("Dataset saved:",file_name)