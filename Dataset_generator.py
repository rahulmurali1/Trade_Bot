import upstox_client
import pandas as pd
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

raw_file="top10_raw_dataset.xlsx"

master_df=[]

# =============================
# DOWNLOAD DATA
# =============================

for stock_id,(stock_name,instrument) in stocks.items():

    print("Downloading:",stock_name)

    all_data=[]
    current=start_date

    while current < end_date:

        next_date=current+timedelta(days=28)
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
            time.sleep(0.3)

        except ApiException as e:
            print("API error:",e)
            time.sleep(2)

    # =============================
    # DATAFRAME
    # =============================

    df=pd.DataFrame(
        all_data,
        columns=["time","open","high","low","close","volume","oi"]
    )

    df.drop_duplicates(inplace=True)

    df["time"]=df["time"].str.replace("+05:30","",regex=False)
    df["time"]=pd.to_datetime(df["time"])

    df["date"]=df["time"].dt.date
    df["clock_time"]=df["time"].dt.time.astype(str)

    df=df.drop(columns=["time","oi"],errors="ignore")
    df=df[["date","clock_time","open","high","low","close","volume"]]
    df=df.sort_values(["date","clock_time"]).reset_index(drop=True)
    df["stock_id"]=stock_id
    df["stock_name"]=stock_name

    master_df.append(df)

# =============================
# SAVE RAW DATA
# =============================

raw_df=pd.concat(master_df)

raw_df.to_excel(raw_file,index=False)

print("Raw dataset saved:",raw_file)