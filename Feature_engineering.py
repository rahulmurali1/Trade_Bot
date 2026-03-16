import pandas as pd
import numpy as np

# =============================
# CONFIG
# =============================

raw_file="top10_raw_dataset.xlsx"
file_name="top10_data_feature3.xlsx"

entry_time="09:15:00"
end_time="15:00:00"

# =============================
# LOAD RAW DATA
# =============================

df=pd.read_excel(raw_file)

df=df.sort_values(["stock_id","date","clock_time"]).reset_index(drop=True)

master_df=[]

# =============================
# LOOP STOCKS
# =============================

for stock_id,stock_df in df.groupby("stock_id"):

    stock_name=stock_df["stock_name"].iloc[0]
    df=stock_df.copy()

    # =============================
    # BASIC FEATURES
    # =============================

    df["avg_price"]=(df["open"]+df["close"])/2
    df["ema20"]=df["close"].ewm(span=20,adjust=False).mean()
    df["ema50"]=df["close"].ewm(span=50,adjust=False).mean()

    # =============================
    # DAILY OHLC
    # =============================

    daily=df.groupby("date").agg({
        "open":"first",
        "high":"max",
        "low":"min",
        "close":"last"
    }).reset_index()

    daily["prev_day_high"]=daily["high"].shift(1)
    daily["prev_day_low"]=daily["low"].shift(1)
    daily["prev_day_close"]=daily["close"].shift(1)

    daily["PP"]=(daily["high"]+daily["low"]+daily["close"])/3
    daily["BC"]=(daily["high"]+daily["low"])/2
    daily["TC"]=(daily["PP"]-daily["BC"])+daily["PP"]

    daily["S1"]=2*daily["PP"]-daily["high"]
    daily["S2"]=daily["PP"]-(daily["high"]-daily["low"])
    daily["R1"]=2*daily["PP"]-daily["low"]
    daily["R2"]=daily["PP"]+(daily["high"]-daily["low"])

    daily["pp_vs_prev_pp"]=(daily["PP"]>daily["PP"].shift(1)).astype(int)

    daily["cpr_range"]=abs(daily["TC"]-daily["BC"])/daily["PP"]*100

    pivot_cols=["PP","BC","TC","S1","S2","R1","R2","pp_vs_prev_pp"]
    daily[pivot_cols]=daily[pivot_cols].shift(1)
    daily["cpr_range"]=abs(daily["TC"]-daily["BC"])/daily["PP"]*100
    df=df.merge(
        daily[[
            "date",
            "prev_day_high",
            "prev_day_low",
            "prev_day_close",
            "cpr_range",
            "PP",
            "S1",
            "R1",
            "pp_vs_prev_pp"
        ]],
        on="date",
        how="left"
    )

    # =============================
    # FEATURE COLUMNS
    # =============================

    feature_cols=[
        "gap_pct",
        "gap_dir",
        "first_candle_body_pct",
        "bullish_close",
        "open_close_prevday_relation",
        "open_close_PP_relation",
        "cpr_range",
        "dist_PP_price",
        "upper_wick",
        "lower_wick"
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

        # -----------------------------
        # first candle body %
        # -----------------------------

        if high!=low:
            candle_body_pct=abs((open_price-entry)/(high-low))
        else:
            candle_body_pct=0

        df.at[idx,"first_candle_body_pct"]=candle_body_pct

        # df.at[idx,"first_candle_range_pct"]=(entry-open_price)/open_price*100

        if pd.notna(prev_close):
            df.at[idx, "gap_dir"] = 1 if open_price>prev_close else 0
            df.at[idx,"gap_pct"]=abs((open_price-prev_close)/prev_close*100)

        df.at[idx,"bullish_close"]=1 if entry>open_price else 0
        bull_close = df.at[idx, "bullish_close"]

        # -----------------------------
        # Open Close vs Previous Day
        # -----------------------------

        if pd.notna(prev_high) and pd.notna(prev_low):

            if prev_low < open_price < prev_high and prev_low < entry < prev_high:
                df.at[idx,"open_close_prevday_relation"]=0

            elif prev_low < open_price < prev_high and entry < prev_low:
                df.at[idx,"open_close_prevday_relation"]=1

            elif prev_low < open_price < prev_high and entry > prev_high:
                df.at[idx,"open_close_prevday_relation"]=2

            elif open_price > prev_high and entry > prev_high:
                df.at[idx,"open_close_prevday_relation"]=3

            elif open_price < prev_low and entry < prev_low:
                df.at[idx,"open_close_prevday_relation"]=4

            elif open_price > prev_high and prev_low < entry < prev_high:
                df.at[idx,"open_close_prevday_relation"]=5

            elif open_price < prev_low and prev_low < entry < prev_high:
                df.at[idx,"open_close_prevday_relation"]=6
        if pd.notna(open_price) and pd.notna(entry) and pd.notna(pp):
            if open_price>pp and entry>pp:
                df.at[idx, "open_close_PP_relation"]=0
            elif open_price>pp and entry<pp:
                df.at[idx, "open_close_PP_relation"]=1
            elif open_price<pp and entry>pp:
                df.at[idx, "open_close_PP_relation"]=2
            else:
                df.at[idx, "open_close_PP_relation"]=3
        # Distance between PP and Entry price in abs
        if pd.notna(entry) and pd.notna(pp):
            df.at[idx, "dist_PP_price"]= abs((entry-pp)/pp*100)
        # Distance of upperwick and lower wick
        if pd.notna(open_price)and pd.notna(entry) and pd.notna(high) and pd.notna(low):
            if bull_close:
                df.at[idx, "upper_wick"] = (high-entry)/entry*100
                df.at[idx, "lower_wick"] = (open_price-low)/open_price*100
            else:
                df.at[idx, "upper_wick"] = (high-open_price)/open_price*100
                df.at[idx, "lower_wick"] = (entry-low)/entry*100
        # -----------------------------
        # LABEL LOGIC (UNCHANGED)
        # -----------------------------

        future=group[
            (group["clock_time"]>entry_time) &
            (group["clock_time"]<=end_time)
        ]

        close_150=future[future["clock_time"]==end_time]

        if not close_150.empty:
            close_price=close_150["close"].values[0]

            df.at[idx,"trade_label"]=1 if close_price>entry*1.003 else 0
        else:
            df.at[idx,"trade_label"]=2

    # keep entry rows only
    df=df[df["clock_time"]==entry_time].copy()

    master_df.append(df)

# =============================
# FINAL DATASET
# =============================

final_df=pd.concat(master_df)

# final_df["first_raw_candle_seg"]=pd.qcut(
#     final_df["first_candle_range_pct"],
#     q=3,
#     labels=[0,1,2],
#     duplicates="drop"
# )

# final_df["gap_segment"]=pd.qcut(
#     final_df["gap_pct"],
#     q=4,
#     labels=[0,1,2,3],
#     duplicates="drop"
# )

# final_df["first_candle_body_seg"]=pd.qcut(
#     final_df["first_candle_body_pct"],
#     q=3,
#     labels=[0,1,2],
#     duplicates="drop"
# )

final_df=final_df[final_df["trade_label"].isin([0,1])]
final_df["trade_label"]=final_df["trade_label"].astype(int)

# rounding
exclude_cols=["trade_label","stock_id"]

num_cols=final_df.select_dtypes(
    include=["float64","float32","int64"]
).columns

num_cols=[c for c in num_cols if c not in exclude_cols]

final_df[num_cols]=final_df[num_cols].round(2)

drop_cols=[
"high","low","close","volume","avg_price",
"ema20","ema50","prev_day_close","prev_day_high",
"prev_day_low","PP","S1","R1"
]

final_df=final_df.drop(columns=drop_cols,errors="ignore")

final_df.to_excel(file_name,index=False)

print("Dataset saved")
print("\nLabel distribution:")
print(final_df["trade_label"].value_counts())