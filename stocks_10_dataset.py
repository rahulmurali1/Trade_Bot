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
    df=pd.DataFrame(all_data,columns=[
    "time","open","high","low","close","volume","oi"])

    df.drop_duplicates(inplace=True)

    df["time"]=df["time"].str.replace("+05:30","",regex=False)
    df["time"]=pd.to_datetime(df["time"])

    df["date"]=df["time"].dt.date
    df["clock_time"]=df["time"].dt.time.astype(str)
    df = df.drop(columns=["time","oi"], errors="ignore")
    df = df[[
        "date","clock_time","open","high","low","close","volume"
    ]]
    df=df.sort_values(["date","clock_time"]).reset_index(drop=True)

    # =============================
    # BASIC FEATURES
    # =============================
    df["avg_price"]=(df["open"]+df["close"])/2
    df["avg_order_value"]=df["avg_price"].round(0) *df["volume"]

    # =============================
    # EMA
    # =============================
    df["ema20"]=df["close"].ewm(span=20,adjust=False).mean().round(2)
    df["ema50"]=df["close"].ewm(span=50,adjust=False).mean().round(2)

    df["ema20_ema50_pct_diff"] = ((df["ema20"] - df["ema50"]) / df["ema50"]) * 100

    # =============================
    # DAILY OHLC
    # =============================
    daily=df.groupby("date").agg({
    "open":"first",
    "high":"max",
    "low":"min",
    "close":"last"}).reset_index()

    daily["prev_day_high"]=daily["high"].shift(1)
    daily["prev_day_low"]=daily["low"].shift(1)
    daily["prev_day_close"]=daily["close"].shift(1)

    # =============================
    # CPR
    # =============================
    daily["PP"]=(daily["high"]+daily["low"]+daily["close"])/3
    daily["BC"]=(daily["high"]+daily["low"])/2
    daily["TC"] = 2*daily["PP"] - daily["BC"]
    daily["S1"]=2*daily["PP"]-daily["high"]
    daily["S2"]=daily["PP"]-(daily["high"]-daily["low"])
    daily["R1"]=2*daily["PP"]-daily["low"]
    daily["R2"] = daily["PP"] + (daily["high"] - daily["low"])
    daily["cpr_range"] = daily["TC"] - daily["BC"]
    daily["r1_s1_range"] = daily["R1"] - daily["S1"]
    daily["r2_s2_range"] = daily["R2"] - daily["S2"]
    daily["pp_vs_prev_pp_pct"] = (
        (daily["PP"] - daily["PP"].shift(1)) / daily["PP"].shift(1) * 100
    )
    
    daily[["PP","BC","TC","S1","S2","R1", "R2", "cpr_range","r1_s1_range","r2_s2_range", "pp_vs_prev_pp_pct"]]=daily[["PP","BC","TC","S1","S2","R1", "R2", "cpr_range","r1_s1_range","r2_s2_range", "pp_vs_prev_pp_pct"]].shift(1)
    # pivot_cols = [
    # "PP","BC","TC",
    # "S1","S2","R1","R2",
    # "cpr_range","r1_s1_range","r2_s2_range"
    # ]

    # daily[pivot_cols] = daily[pivot_cols].shift(1)

    df = df.merge(
    daily[["date","prev_day_close", "PP","BC","TC","S1","S2","R1","R2",
           "cpr_range","r1_s1_range","r2_s2_range", "pp_vs_prev_pp_pct"]],
    on="date",
    how="left"
    )
    df["cpr_range_pct"] = df["cpr_range"] / df["PP"] * 100
    df["r1_s1_range_pct"] = df["r1_s1_range"] / df["PP"] * 100
    df["r2_s2_range_pct"] = df["r2_s2_range"] / df["PP"] * 100

    # =============================
    # CPR DISTANCE FEATURES
    # =============================
    df["PP_to_BC_pct"]=abs(df["PP"]-df["BC"])/df["PP"]*100
    df["PP_to_S1_pct"]=abs(df["PP"]-df["S1"])/df["PP"]*100
    df["PP_to_S2_pct"]=abs(df["PP"]-df["S2"])/df["PP"]*100
    # =============================
    # INTERACTION FEATURES
    # =============================
    # df["open_above_pp"]=(df["open"]>df["PP"]).astype(int)
    # df["close_above_pp"]=(df["close"]>df["PP"]).astype(int)

    # tol=df["PP"]*0.001

    # df["touch_pp_directional"]=np.where(
    #     df["open"]>df["PP"],
    #     (df["low"]<=df["PP"]+tol),
    #     (df["high"]>=df["PP"]-tol)
    # ).astype(int)

    df["avg_pp_dist_pct"]=abs(df["avg_price"]-df["PP"])/df["PP"]*100

    # =============================
    # FEATURE COLUMNS
    # =============================
    df["first_candle_pct"]=np.nan
    df["first_15min_range"]=np.nan
    df["gap_pct"]=np.nan

    df["distance_from_pp"]=np.nan
    df["distance_from_s1"]=np.nan
    df["distance_from_r1"]=np.nan

    df["price_vs_ema20"]=np.nan
    df["price_vs_ema50"]=np.nan

    df["pp_above_prev_close"]=np.nan
    df["open915_vs_prevclose_pct"]=np.nan
    df["close915_vs_prevclose_pct"]=np.nan
    df["low915_vs_prevclose_pct"]=np.nan
    df["high915_vs_prevclose_pct"]=np.nan
    df["open915_vs_pp_pct"] = np.nan
    df["close915_vs_pp_pct"] = np.nan
    df["low915_vs_pp_pct"] = np.nan
    df["high915_vs_pp_pct"] = np.nan
    df["open915_vs_close915_pct"] = np.nan
    df["open915_vs_high915_pct"] = np.nan
    df["open915_vs_low915_pct"] = np.nan
    df["trade_label"]=np.nan

    prev_close_map=daily.set_index("date")["prev_day_close"]

    grouped=df.groupby("date",sort=False)

    # =============================
    # DAY LOOP
    # =============================
    for date,group in grouped:

        candle=group[group["clock_time"]==entry_time]
        if candle.empty:
            continue

        idx=candle.index[0]

        entry=candle["close"].values[0]
        open_price=candle["open"].values[0]

        low_915=candle["low"].values[0]
        high_915=candle["high"].values[0]
        df.at[idx,"open915_vs_close915_pct"] = (
            (entry - open_price) / open_price * 100
        )

        df.at[idx,"open915_vs_high915_pct"] = (
            (high_915 - open_price) / open_price * 100
        )

        df.at[idx,"open915_vs_low915_pct"] = (
            (low_915 - open_price) / open_price * 100
        )

        # first candle %
        df.at[idx,"first_candle_pct"]=(
        (entry-open_price)/open_price*100)

        # opening range
        first15=group[group["clock_time"]<"09:30:00"]

        if not first15.empty:
            high15=first15["high"].max()
            low15=first15["low"].min()
            df.at[idx,"first_15min_range"]=(
            (high15-low15)/entry*100)

        # gap
        if date in prev_close_map.index:
            prev_close=prev_close_map.loc[date]
            if pd.notna(prev_close):
                df.at[idx,"gap_pct"]=(
                (open_price-prev_close)/prev_close*100)

        # distance features
        pp=candle["PP"].values[0]
        s1=candle["S1"].values[0]
        r1=candle["R1"].values[0]
        if pd.notna(pp):

            df.at[idx,"open915_vs_pp_pct"] = (
                (open_price - pp) / pp * 100
            )
            

            df.at[idx,"close915_vs_pp_pct"] = (
                (entry - pp) / pp * 100
            )

            df.at[idx,"low915_vs_pp_pct"] = (
                (low_915 - pp) / pp * 100
            )

            df.at[idx,"high915_vs_pp_pct"] = (
                (high_915 - pp) / pp * 100
            )
            df.at[idx,"distance_from_pp"]=(entry-pp)/pp*100

        if pd.notna(s1):
            df.at[idx,"distance_from_s1"]=(entry-s1)/s1*100

        if pd.notna(r1):
            df.at[idx,"distance_from_r1"]=(entry-r1)/r1*100

        # EMA distance
        ema20=df.loc[idx,"ema20"]
        ema50=df.loc[idx,"ema50"]

        if pd.notna(ema20):
            df.at[idx,"price_vs_ema20"]=(entry-ema20)/ema20*100

        if pd.notna(ema50):
            df.at[idx,"price_vs_ema50"]=(entry-ema50)/ema50*100

        # PP vs prev close
        prev_close=candle["prev_day_close"].values[0]
        if pd.notna(prev_close):
            df.at[idx,"open915_vs_prevclose_pct"] = (
                (open_price - prev_close) / prev_close * 100
            )
            df.at[idx,"close915_vs_prevclose_pct"] = (
                (entry - prev_close) / prev_close * 100
            )
            df.at[idx,"low915_vs_prevclose_pct"] = (
                (low_915 - prev_close) / prev_close * 100
            )
            df.at[idx,"high915_vs_prevclose_pct"] = (
                (high_915 - prev_close) / prev_close * 100
            )
        if pd.notna(pp) and pd.notna(prev_close):
            df.at[idx,"pp_above_prev_close"]=1 if pp>prev_close else 0
        
        # =============================
        # LABEL LOGIC
        # =============================
        risk=max(entry-low_915,high_915-entry)
        risk*=1.05

        if risk/entry<MIN_RISK_PCT:
            df.at[idx,"trade_label"]=2
            continue

        sl_bull=entry-risk
        sl_bear=entry+risk

        target_bull=entry+RR*risk
        target_bear=entry-RR*risk

        future=group[
        (group["clock_time"]>entry_time) &
        (group["clock_time"]<=end_time)]

        bull_target=None
        bull_sl=None
        bear_target=None
        bear_sl=None

        for _,row in future.iterrows():
            if bull_sl is None and row["low"]<=sl_bull:
                bull_sl=row["clock_time"]
            if bull_target is None and row["high"]>=target_bull:
                bull_target=row["clock_time"]
            if bear_sl is None and row["high"]>=sl_bear:
                bear_sl=row["clock_time"]
            if bear_target is None and row["low"]<=target_bear:
                bear_target=row["clock_time"]
        bull_success=False
        bear_success=False

        if bull_target and (not bull_sl or bull_target<bull_sl):
                bull_success=True

        if bear_target and (not bear_sl or bear_target<bear_sl):
                bear_success=True

        if bull_success and not bear_success:
            df.at[idx,"trade_label"]=1
        elif bear_success and not bull_success:
            df.at[idx,"trade_label"]=0
        elif bull_success and bear_success:
            df.at[idx,"trade_label"]=1 if bull_target<bear_target else 0
        else:
            df.at[idx,"trade_label"]=2

    # keep entry rows only
    df=df[df["clock_time"]==entry_time].copy()
    df["stock_id"]=stock_id
    df["stock_name"]=stock_name
    master_df.append(df)

# =============================
# FINAL DATASET
# =============================
final_df=pd.concat(master_df)

cols=list(final_df.columns)
cols.remove("stock_id")
cols.remove("stock_name")

idx=cols.index("trade_label")

cols.insert(idx,"stock_id")
cols.insert(idx+1,"stock_name")
# =============================
# ROUND NUMERIC FEATURES
# =============================

exclude_cols = ["trade_label","stock_id", "avg_order_value", ]
num_cols = final_df.select_dtypes(include=["float64","float32","int64"]).columns
num_cols = [c for c in num_cols if c not in exclude_cols]
final_df[num_cols] = final_df[num_cols].round(2)
final_df=final_df[cols]

drop_cols = [
"open","high","low","close","volume",
"avg_price","ema20","ema50",
"prev_day_close",
"PP","BC","TC","S1","S2","R1","R2",
"PP_to_BC_pct","PP_to_S1_pct","PP_to_S2_pct",
"avg_pp_dist_pct","gap_pct"
]

final_df = final_df.drop(columns=drop_cols, errors="ignore")

final_df.to_excel("top10_stocks_dataset_v7.xlsx",index=False)
print("Dataset saved")
print("\nLabel distribution:")
print(final_df["trade_label"].value_counts())