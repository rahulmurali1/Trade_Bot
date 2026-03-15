import upstox_client
import pandas as pd
import time
from datetime import datetime, timedelta
from upstox_client.rest import ApiException

api = upstox_client.HistoryV3Api()

instrument = "NSE_EQ|INE758T01015"   # NSE_EQ|INE040A01034 = HDFCBANK
interval_type = "minutes"
interval = "5"

start_date = datetime(2022,2,2)
end_date = datetime(2026,3,10)

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

        candles = response.data.candles

        all_data.extend(candles)

        time.sleep(1)   # gap between API calls

        current = next_date

    except ApiException as e:

        print("API error:", e)
        print("Retrying in 5 seconds...")

        time.sleep(5)


df = pd.DataFrame(all_data, columns=[
    "time","open","high","low","close","volume","oi"
])

df.drop_duplicates(inplace=True)

df["time"] = df["time"].str.replace("+05:30", "", regex=False)

# Convert to datetime
df["time"] = pd.to_datetime(df["time"])

# Create separate date and time columns
df["date"] = df["time"].dt.date
df["clock_time"] = df["time"].dt.time

# Drop original time column
df = df.drop(columns=["time"])

# Remove OI column
if "oi" in df.columns:
    df = df.drop(columns=["oi"])

# Reorder columns
df = df[[
    "date",
    "clock_time",
    "open",
    "high",
    "low",
    "close",
    "volume"
]]

# Calculate average price
df["avg_price"] = (df["open"] + df["close"]) / 2

# Calculate average order value
df["avg_order_value"] = df["avg_price"] * df["volume"]

df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['clock_time'].astype(str))

# Daily OHLC aggregation
daily_ohlc = df.groupby('date').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last'
}).reset_index()

# CPR calculation (classic)
daily_ohlc['PP'] = (daily_ohlc['high'] + daily_ohlc['low'] + daily_ohlc['close']) / 3
daily_ohlc['BC'] = (daily_ohlc['high'] + daily_ohlc['low']) / 2
daily_ohlc['TC'] = 2 * daily_ohlc['PP'] - daily_ohlc['BC']

# Support / Resistance levels
daily_ohlc['S1'] = 2 * daily_ohlc['PP'] - daily_ohlc['high']
daily_ohlc['R1'] = 2 * daily_ohlc['PP'] - daily_ohlc['low']
daily_ohlc['S2'] = daily_ohlc['PP'] - (daily_ohlc['high'] - daily_ohlc['low'])
daily_ohlc['R2'] = daily_ohlc['PP'] + (daily_ohlc['high'] - daily_ohlc['low'])

df = df.merge(
    daily_ohlc[['date', 'PP', 'BC', 'TC', 'S1', 'R1', 'S2', 'R2']],
    on='date',
    how='left'
)

# df['date'] = pd.to_datetime(df['date'])
# df['clock_time'] = pd.to_datetime(df['clock_time'], format='%H:%M:%S').dt.time

# Sort data from oldest to newest
df = df.sort_values(by=['date', 'clock_time'])

# Round CPR values to 1 decimal
dec1_columns = ['open','high', 'low', 'close', 'avg_price','PP','BC','TC','S1','R1','S2','R2']
df[dec1_columns] = df[dec1_columns].round(1)
dec0_columns = ['avg_order_value']
df[dec0_columns] = df[dec0_columns].round(0)

if "datetime" in df.columns:
    df = df.drop(columns=["datetime"])

# Save cleaned file
df.to_excel("eternal_hist.xlsx", index=False)

print("Saved full dataset")