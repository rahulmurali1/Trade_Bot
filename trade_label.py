import pandas as pd
import numpy as np

# =========================
# LOAD DATA
# =========================

df = pd.read_excel("eternal_hist_ft.xlsx")

# =========================
# INITIALIZE COLUMNS
# =========================

df['trade_label'] = np.nan
df['trade_type'] = ""

df['entry_price'] = np.nan
df['stop_loss'] = np.nan
df['target_price'] = np.nan

df['bull_target_time'] = None
df['bull_sl_time'] = None
df['bear_target_time'] = None
df['bear_sl_time'] = None

# =========================
# PARAMETERS
# =========================

entry_time = "09:20:00"
prev_time = "09:15:00"
end_time = "14:30:00"

RR = 2

MIN_RISK_PCT = 0.005   # 0.5%
MAX_RISK_PCT = 0.01    # 1%

# =========================
# PROCESS EACH DAY
# =========================

for date, group in df.groupby("date"):

    candle_915 = group[group['clock_time'] == prev_time]
    candle_920 = group[group['clock_time'] == entry_time]

    if candle_915.empty or candle_920.empty:
        continue

    idx = candle_920.index[0]

    entry = candle_920['open'].values[0]

    low_915 = candle_915['low'].values[0]
    high_915 = candle_915['high'].values[0]

    # =========================
    # SYMMETRIC RISK
    # =========================

    risk = max(entry - low_915, high_915 - entry)

    risk_pct = risk / entry

    # =========================
    # MIN + MAX RISK FILTER
    # =========================

    if risk_pct < MIN_RISK_PCT or risk_pct > MAX_RISK_PCT:

        df.at[idx,'trade_label'] = 2
        df.at[idx,'trade_type'] = "no_signal"
        continue

    # =========================
    # DEFINE SL & TARGETS
    # =========================

    sl_bull = entry - risk
    sl_bear = entry + risk

    target_bull = entry + RR * risk
    target_bear = entry - RR * risk

    future = group[
        (group['clock_time'] > entry_time) &
        (group['clock_time'] <= end_time)
    ]

    bull_target_time = None
    bull_sl_time = None
    bear_target_time = None
    bear_sl_time = None

    # =========================
    # SCAN FUTURE CANDLES
    # =========================

    for i, row in future.iterrows():

        high = row['high']
        low = row['low']
        time = row['clock_time']

        if bull_target_time is None and high >= target_bull:
            bull_target_time = time

        if bull_sl_time is None and low <= sl_bull:
            bull_sl_time = time

        if bear_target_time is None and low <= target_bear:
            bear_target_time = time

        if bear_sl_time is None and high >= sl_bear:
            bear_sl_time = time

    df.at[idx,'bull_target_time'] = bull_target_time
    df.at[idx,'bull_sl_time'] = bull_sl_time
    df.at[idx,'bear_target_time'] = bear_target_time
    df.at[idx,'bear_sl_time'] = bear_sl_time

    # =========================
    # SUCCESS CHECK
    # =========================

    bull_success = False
    bear_success = False

    if bull_target_time is not None:
        if bull_sl_time is None or bull_target_time < bull_sl_time:
            bull_success = True

    if bear_target_time is not None:
        if bear_sl_time is None or bear_target_time < bear_sl_time:
            bear_success = True

# =========================
# FINAL DECISION (3-class: 0,1,2)
# =========================

    if bull_success and not bear_success:

        df.at[idx,'trade_label'] = 1
        df.at[idx,'trade_type'] = "bullish"
        df.at[idx,'entry_price'] = entry
        df.at[idx,'stop_loss'] = sl_bull
        df.at[idx,'target_price'] = target_bull

    elif bear_success and not bull_success:

        df.at[idx,'trade_label'] = 0
        df.at[idx,'trade_type'] = "bearish"
        df.at[idx,'entry_price'] = entry
        df.at[idx,'stop_loss'] = sl_bear
        df.at[idx,'target_price'] = target_bear

    elif bull_success and bear_success:

        if bull_target_time < bear_target_time:

            df.at[idx,'trade_label'] = 1
            df.at[idx,'trade_type'] = "bullish"
            df.at[idx,'entry_price'] = entry
            df.at[idx,'stop_loss'] = sl_bull
            df.at[idx,'target_price'] = target_bull

        else:

            df.at[idx,'trade_label'] = 0
            df.at[idx,'trade_type'] = "bearish"
            df.at[idx,'entry_price'] = entry
            df.at[idx,'stop_loss'] = sl_bear
            df.at[idx,'target_price'] = target_bear

    else:

        df.at[idx,'trade_label'] = 2
        df.at[idx,'trade_type'] = "no_signal"

# =========================
# CLEAN DATA
# =========================

cols = ['entry_price','stop_loss','target_price']
df[cols] = df[cols].round(1)

df['date'] = pd.to_datetime(df['date']).dt.date

# =========================
# SAVE DATASET
# =========================

df.to_excel("eternal_hist_ft_lb.xlsx", index=False)

print("Dataset generated successfully.")