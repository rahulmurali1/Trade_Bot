import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

# =====================================
# CONFIGURATION
# =====================================
MODEL_PATH = "COALINDIA_dataset_model.json"
DATA_FILE = "RELIANCE_dataset.xlsx"  # make sure dataset has both candles
OUTPUT_FILE = "reliance_result_finalv8.xlsx"

CAPITAL = 50000
RISK_PER_TRADE = 0.01
RR = 2
MIN_SL_PCT = 0.2
MAX_SL_PCT = 1
BROKERAGE_PCT = 300  # 0.09% per side
SLIPPAGE_PCT = 0.0003    # 0.03% slippage

# =====================================
# LOAD MODEL
# =====================================
model = XGBClassifier()
model.load_model(MODEL_PATH)

# =====================================
# LOAD DATA (keep both 09:15 and 09:20)
# =====================================
df = pd.read_excel(DATA_FILE)

# =====================================
# FILTER 09:20 CANDLE FOR FEATURES
# =====================================
feature_df = df[df['clock_time'] == "09:20:00"].copy()
features = [
    'open', 'volume','avg_price','avg_order_value',
    'PP','PP_to_BC_pct', 'PP_to_S1_pct', 'PP_to_S2_pct', 'gap_pct', 'dist_915_high_pct', 'dist_915_low_pct', 'dist_915_max_pct'
]

X = feature_df[features]
feature_df['pred_label'] = model.predict(X)
feature_df['pred_prob'] = model.predict_proba(X).tolist()

# =====================================
# TRADE SIMULATION
# =====================================
profits, quantities, exit_reason = [], [], []
entry_prices, sl_prices, targets = [], [], []
sl_times, target_times = [], []
sl_pct_list = []
trade_values, brokerages = [], []

capital = CAPITAL

for idx, row in feature_df.iterrows():
    pred = row['pred_label']
    if pred == 2:
        # Skip no_signal trades
        profits.append(0)
        quantities.append(0)
        exit_reason.append("NO_TRADE")
        entry_prices.append(None)
        sl_prices.append(None)
        targets.append(None)
        sl_times.append(None)
        target_times.append(None)
        sl_pct_list.append(None)
        trade_values.append(None)
        brokerages.append(None)
        continue

    date = row['date']
    entry_price = row['open']

    # Apply slippage
    if pred == 1:
        entry_price *= (1 + SLIPPAGE_PCT)
    else:
        entry_price *= (1 - SLIPPAGE_PCT)
    entry_prices.append(entry_price)

    # =========================
    # GET 09:15 CANDLE FOR SL
    # =========================
    candle_915 = df[(df['date'] == date) & (df['clock_time'] == "09:15:00")]
    if candle_915.empty:
        profits.append(0)
        quantities.append(0)
        exit_reason.append("NO_915_DATA")
        sl_prices.append(None)
        targets.append(None)
        sl_times.append(None)
        target_times.append(None)
        sl_pct_list.append(None)
        trade_values.append(None)
        brokerages.append(None)
        continue

    high_915 = candle_915['high'].values[0]
    low_915 = candle_915['low'].values[0]

    # =========================
    # SL DISTANCE
    # =========================
    sl_distance = max(entry_price - low_915, high_915 - entry_price)
    sl_pct = (sl_distance / entry_price) * 100
    sl_pct_list.append(sl_pct)

    # SL % filter
    if sl_pct < MIN_SL_PCT:
        profits.append(0)
        quantities.append(0)
        exit_reason.append("SL_TOO_SMALL")
        sl_prices.append(None)
        targets.append(None)
        sl_times.append(None)
        target_times.append(None)
        trade_values.append(None)
        brokerages.append(None)
        continue
    if sl_pct > MAX_SL_PCT:
        profits.append(0)
        quantities.append(0)
        exit_reason.append("SL_TOO_BIG")
        sl_prices.append(None)
        targets.append(None)
        sl_times.append(None)
        target_times.append(None)
        trade_values.append(None)
        brokerages.append(None)
        continue

    # =========================
    # POSITION SIZE
    # =========================
    risk_amount = 1500
    qty = max(int(risk_amount / sl_distance), 1)
    quantities.append(qty)

    # =========================
    # SL + TARGET
    # =========================
    if pred == 1:
        sl = entry_price - sl_distance
        target = entry_price + RR * sl_distance
    else:
        sl = entry_price + sl_distance
        target = entry_price - RR * sl_distance
    sl_prices.append(sl)
    targets.append(target)

    # =========================
    # FUTURE CANDLES
    # =========================
    future = df[(df['date'] == date) & (df['clock_time'] > "09:20:00") & (df['clock_time'] <= "14:30:00")]
    trade_closed = False
    sl_time, target_time = None, None

    for _, f in future.iterrows():
        high, low, t = f['high'], f['low'], f['clock_time']

        if pred == 1:
            if low <= sl and high >= target:
                sl_time, target_time = t, t
                profit = -sl_distance * qty
                exit_reason.append("BOTH_HIT_SL_ASSUMED")
                trade_closed = True
                break
            elif low <= sl:
                sl_time = t
                profit = -sl_distance * qty
                exit_reason.append("SL_HIT")
                trade_closed = True
                break
            elif high >= target:
                target_time = t
                profit = RR * sl_distance * qty
                exit_reason.append("TARGET_HIT")
                trade_closed = True
                break

        if pred == 0:
            if high >= sl and low <= target:
                sl_time, target_time = t, t
                profit = -sl_distance * qty
                exit_reason.append("BOTH_HIT_SL_ASSUMED")
                trade_closed = True
                break
            elif high >= sl:
                sl_time = t
                profit = -sl_distance * qty
                exit_reason.append("SL_HIT")
                trade_closed = True
                break
            elif low <= target:
                target_time = t
                profit = RR * sl_distance * qty
                exit_reason.append("TARGET_HIT")
                trade_closed = True
                break

    # =========================
    # TIME EXIT
    # =========================
    if not trade_closed:
        candle_230 = df[(df['date'] == date) & (df['clock_time'] == "14:30:00")]
        if not candle_230.empty:
            exit_price = candle_230['close'].values[0]
            profit = (exit_price - entry_price) * qty if pred == 1 else (entry_price - exit_price) * qty
            exit_reason.append("TIME_EXIT")
        else:
            profit = 0
            exit_reason.append("NO_230_DATA")

    # =========================
    # TRANSACTION COSTS
    # =========================
    trade_value = entry_price * qty
    brokerage = BROKERAGE_PCT # both sides
    profit -= brokerage  # subtract brokerage

    trade_values.append(trade_value)
    brokerages.append(brokerage)
    profits.append(profit)
    sl_times.append(sl_time)
    target_times.append(target_time)
    capital += profit

# =====================================
# SAVE RESULTS
# =====================================
feature_df['entry_price'] = entry_prices
feature_df['sl_price'] = sl_prices
feature_df['target_price'] = targets
feature_df['quantity'] = quantities
feature_df['sl_distance_pct'] = sl_pct_list
feature_df['sl_hit_time'] = sl_times
feature_df['target_hit_time'] = target_times
feature_df['exit_reason'] = exit_reason
feature_df['trade_profit'] = profits
feature_df['trade_value'] = trade_values
feature_df['brokerage'] = brokerages

# =====================================
# PERFORMANCE METRICS
# =====================================
feature_df['equity_curve'] = feature_df['trade_profit'].cumsum()
feature_df['peak'] = feature_df['equity_curve'].cummax()
feature_df['drawdown'] = feature_df['equity_curve'] - feature_df['peak']
max_dd = feature_df['drawdown'].min()

trades = feature_df[feature_df['exit_reason'].isin(["SL_HIT","TARGET_HIT","TIME_EXIT","BOTH_HIT_SL_ASSUMED"])]
total_trades = len(trades)
wins = len(trades[trades['trade_profit'] > 0])
losses = len(trades[trades['trade_profit'] < 0])
total_profit = trades['trade_profit'].sum()
win_rate = wins / total_trades if total_trades > 0 else 0

print("\n========== TRADE PERFORMANCE ==========")
print(f"Total trades taken: {total_trades}")
print(f"Winning trades: {wins}")
print(f"Losing trades: {losses}")
print(f"Win rate: {round(win_rate*100,2)}%")
print(f"Total PnL (₹): {round(total_profit,2)}")
print(f"Max Drawdown: {round(max_dd,2)}")

# =====================================
# MODEL ACCURACY
# =====================================
feature_df['correct_prediction'] = feature_df['trade_label'] == feature_df['pred_label']
accuracy = feature_df['correct_prediction'].sum() / len(feature_df)
print("\n========== MODEL ACCURACY ==========")
print(f"Total predictions: {len(feature_df)}")
print(f"Correct predictions: {feature_df['correct_prediction'].sum()}")
print(f"Accuracy: {round(accuracy*100,2)}%")
print(classification_report(feature_df['trade_label'], feature_df['pred_label']))
print(confusion_matrix(feature_df['trade_label'], feature_df['pred_label']))

# =====================================
# SAVE BACKTEST FILE
# =====================================
feature_df.to_excel(OUTPUT_FILE, index=False)
print("\nBacktest results saved to", OUTPUT_FILE)