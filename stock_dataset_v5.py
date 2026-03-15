# Reducing features from individual 9.15 candle characteristcs to a first candle range

import upstox_client
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
from upstox_client.rest import ApiException

# =============================
# CONFIG
# =============================
api = upstox_client.HistoryV3Api()

stocks = {
    0:("HDFCBANK","NSE_EQ|INE040A01034"),
    1:("RELIANCE","NSE_EQ|INE002A01018"),
    2:("ICICIBANK","NSE_EQ|INE090A01021"),
    3:("INFY","NSE_EQ|INE009A01021"),
    4:("BHARTIARTL","NSE_EQ|INE397D01024"),
    5:("TCS","NSE_EQ|INE467B01029"),
    6:("LT","NSE_EQ|INE018A01030"),
    7:("ITC","NSE_EQ|INE154A01025"),
    8:("SBIN","NSE_EQ|INE062A01020"),
    9:("AXISBANK","NSE_EQ|INE238A01034")
}

interval_type="minutes"
interval="5"

start_date=datetime(2022,2,1)
end_date=datetime(2026,3,10)

entry_time="09:15:00"
end_time="14:30:00"

RR=2
MIN_RISK_PCT=0.003

master_df=[]

# =============================
# LOOP STOCKS
# =============================
for stock_id,(stock_name,instrument) in stocks.items():
    print("\nProcessing:",stock_name)
    all_data=[]
    current=start_date

    # =============================
    # DOWNLOAD DATA
    # =============================
    while current < end_date:
        next_date=current+timedelta(days=25)
        if next_date>end_date:
            next_date=end_date
        try:
            print(f"{stock_name}: {current.date()} → {next_date.date()}")
            response=api.get_historical_candle_data1(
                instrument,
                interval_type,
                interval,
                next_date.strftime("%Y-%m-%d"),
                current.strftime("%Y-%m-%d")
            )
            if response.data and response.data.candles:
                all_data.extend(response.data.candles)
            current=next_date
            time.sleep(0.4)
        except ApiException as e:
            print("API error:",e)
            time.sleep(2)

    # =============================
    # DATAFRAME
    # =============================
    df=pd.DataFrame(all_data,columns=["time","open","high","low","close","volume","oi"])
    df.drop_duplicates(inplace=True)
    df["time"]=df["time"].str.replace("+05:30","",regex=False)
    df["time"]=pd.to_datetime(df["time"])
    df["date"]=df["time"].dt.date
    df["clock_time"]=df["time"].dt.time.astype(str)
    df=df.drop(columns=["time","oi"],errors="ignore")
    df=df[["date","clock_time","open","high","low","close","volume"]]
    df=df.sort_values(["date","clock_time"]).reset_index(drop=True)

    # =============================
    # BASIC FEATURES
    # =============================
    df["avg_price"]=(df["open"]+df["close"])/2
    df["avg_order_value"]=df["avg_price"].round(0)*df["volume"]
    df["ema20"]=df["close"].ewm(span=20,adjust=False).mean()
    df["ema50"]=df["close"].ewm(span=50,adjust=False).mean()
    df["ema20_ema50_pct_diff"]=((df["ema20"]-df["ema50"])/df["ema50"])*100

    # =============================
    # DAILY OHLC & PIVOTS
    # =============================
    daily=df.groupby("date").agg({"open":"first","high":"max","low":"min","close":"last"}).reset_index()
    daily["prev_day_high"]=daily["high"].shift(1)
    daily["prev_day_low"]=daily["low"].shift(1)
    daily["prev_day_close"]=daily["close"].shift(1)

    daily["PP"]=(daily["high"]+daily["low"]+daily["close"])/3
    daily["BC"]=(daily["high"]+daily["low"])/2
    daily["TC"]=(daily["PP"]-daily["BC"]) + daily["PP"]
    daily["S1"]=2*daily["PP"]-daily["high"]
    daily["S2"]=daily["PP"]-(daily["high"]-daily["low"])
    daily["R1"]=2*daily["PP"]-daily["low"]
    daily["R2"]=daily["PP"]+(daily["high"]-daily["low"])
    daily["cpr_range"]=abs(daily["TC"]-daily["BC"])/daily["PP"]*100
    daily["r1_s1_range"]=daily["R1"]-daily["S1"]
    daily["r2_s2_range"]=daily["R2"]-daily["S2"]
    daily["pp_vs_prev_pp_pct"]=((daily["PP"]-daily["PP"].shift(1))/daily["PP"].shift(1)*100)

    pivot_cols=["PP","BC","TC","S1","S2","R1","R2","cpr_range","r1_s1_range","r2_s2_range","pp_vs_prev_pp_pct"]
    daily[pivot_cols] = daily[pivot_cols].shift(1)

    df=df.merge(
        daily[["date","prev_day_high","prev_day_low","prev_day_close","PP","S1","R1", "cpr_range", "pp_vs_prev_pp_pct"]],
        on="date",
        how="left"
    )

    # =============================
    # FEATURES
    # =============================
    feature_cols=[
        "first_candle_range_pct","range_vs_prevclose","range_vs_prevhigh","range_vs_prevlow",
        "distance_from_pp","distance_from_s1","distance_from_r1",
        "price_vs_ema20","price_vs_ema50","gap_pct"
    ]
    for col in feature_cols:
        df[col]=np.nan
    df["trade_label"]=np.nan

    # =============================
    # DAY LOOP
    # =============================
    for date,group in df.groupby("date"):
        candle=group[group["clock_time"]==entry_time]
        if candle.empty:
            continue
        idx=candle.index[0]
        entry=candle["close"].values[0]
        open_price=candle["open"].values[0]
        high=candle["high"].values[0]
        low=candle["low"].values[0]

        prev_close=candle["prev_day_close"].values[0]
        prev_high=candle["prev_day_high"].values[0]
        prev_low=candle["prev_day_low"].values[0]

        pp=candle["PP"].values[0]
        s1=candle["S1"].values[0]
        r1=candle["R1"].values[0]
        ema20=candle["ema20"].values[0]
        ema50=candle["ema50"].values[0]

        # -----------------------------
        # Directional first candle range
        # -----------------------------
        first_range_signed = entry - open_price
        first_candle_mid = (high+low)/2

        df.at[idx,"first_candle_range_pct"] = (entry-open_price)/open_price*100
        if pd.notna(prev_close):
            df.at[idx,"range_vs_prevclose"] = (first_candle_mid-prev_close)/prev_close*100
        if pd.notna(prev_high):
            df.at[idx,"range_vs_prevhigh"] = (first_candle_mid-prev_high)/prev_high*100
        if pd.notna(prev_low):
            df.at[idx,"range_vs_prevlow"] = (first_candle_mid-prev_low)/prev_low*100
        if pd.notna(pp):
            # df.at[idx,"range_vs_pp"] = first_candle_mid/pp*100
            # df.at[idx,"mid_vs_pp"] = (first_candle_mid - pp)/pp*100
            df.at[idx,"distance_from_pp"] = (entry-pp)/pp*100
        if pd.notna(s1):
            # df.at[idx,"range_vs_s1"] = first_candle_mid/s1*100
            # df.at[idx,"mid_vs_s1"] = (first_candle_mid - s1)/s1*100
            df.at[idx,"distance_from_s1"] = (entry-s1)/s1*100
        if pd.notna(r1):
            # df.at[idx,"range_vs_r1"] = first_candle_mid/r1*100
            # df.at[idx,"mid_vs_r1"] = (first_candle_mid - r1)/r1*100
            df.at[idx,"distance_from_r1"] = (entry-r1)/r1*100
        if pd.notna(ema20):
            # df.at[idx,"range_vs_ema20"] = first_candle_mid-ema20/ema20*100
            df.at[idx,"price_vs_ema20"] = (entry-ema20)/ema20*100
        if pd.notna(ema50):
            # df.at[idx,"range_vs_ema50"] = first_candle_mid-ema50/ema50*100
            df.at[idx,"price_vs_ema50"] = (entry-ema50)/ema50*100
        if pd.notna(prev_close):
            df.at[idx,"gap_pct"] = (open_price-prev_close)/prev_close*100

        # -----------------------------
        # LABEL LOGIC (0/1)
        # -----------------------------
        risk=max(entry-low,high-entry)*1.05
        if risk/entry<MIN_RISK_PCT:
            df.at[idx,"trade_label"]=4
            continue

        sl_bull=entry-risk
        sl_bear=entry+risk
        target_bull=entry+RR*risk
        target_bear=entry-RR*risk

        future=group[(group["clock_time"]>entry_time) & (group["clock_time"]<=end_time)]
        bull_target=bull_sl=bear_target=bear_sl=None
        for _,row in future.iterrows():
            if bull_sl is None and row["low"]<=sl_bull: bull_sl=row["clock_time"]
            if bull_target is None and row["high"]>=target_bull: bull_target=row["clock_time"]
            if bear_sl is None and row["high"]>=sl_bear: bear_sl=row["clock_time"]
            if bear_target is None and row["low"]<=target_bear: bear_target=row["clock_time"]

        if bear_target and (not bear_sl or bear_target < bear_sl):
            df.at[idx,"trade_label"]=0
        elif bull_target and (not bull_sl or bull_target < bull_sl):
            df.at[idx,"trade_label"]=1
        else:
            close_230 = future[future["clock_time"]==end_time]
            if not close_230.empty:
                close_price = close_230["close"].values[0]
                df.at[idx,"trade_label"] = 1 if close_price > entry*1.001 else 0
            else:
                df.at[idx,"trade_label"]=4

    # keep entry rows only
    df = df[df["clock_time"]==entry_time].copy()
    df["stock_id"]=stock_id
    df["stock_name"]=stock_name
    master_df.append(df)

# =============================
# FINAL DATASET
# =============================
final_df=pd.concat(master_df)
final_df = final_df[final_df["trade_label"].isin([0,1])]
final_df["trade_label"]=final_df["trade_label"].astype(int)

# rounding
exclude_cols=["trade_label","stock_id"]
num_cols=final_df.select_dtypes(include=["float64","float32","int64"]).columns
num_cols=[c for c in num_cols if c not in exclude_cols]
final_df[num_cols] = final_df[num_cols].round(2)

drop_cols=["open","high","low","close","volume","avg_price","ema20","ema50",
           "prev_day_close","prev_day_high","prev_day_low","PP","S1","R1"]
final_df = final_df.drop(columns=drop_cols,errors="ignore")

final_df.to_excel("top10_stocks_dataset_v14.xlsx",index=False)
print("Dataset saved")
print("\nLabel distribution:")
print(final_df["trade_label"].value_counts())