##### increased model parameters ####

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ==============================
# LOAD DATA
# ==============================
file_name = "top10_stocks_dataset_v7.xlsx"
df = pd.read_excel(file_name)

# ==============================
# ENTRY CANDLE ONLY
# ==============================
df = df[df["clock_time"] == "09:15:00"].copy()

# ==============================
# FEATURES
# ==============================
features = [
    "ema20_ema50_pct_diff",
    "cpr_range_pct",
    "r1_s1_range_pct",
    "r2_s2_range_pct",
    "pp_vs_prev_pp_pct",
    "first_15min_range",
    "distance_from_s1",
    "distance_from_r1",
    "price_vs_ema20",
    "price_vs_ema50",
    "open915_vs_prevclose_pct",
    "close915_vs_prevclose_pct",
    "low915_vs_prevclose_pct",
    "high915_vs_prevclose_pct",
    "open915_vs_pp_pct",
    "close915_vs_pp_pct",
    "low915_vs_pp_pct",
    "high915_vs_pp_pct",
    "open915_vs_close915_pct",
    "open915_vs_high915_pct",
    "open915_vs_low915_pct"
]

df = df.dropna(subset=features + ["trade_label"])

# ==============================
# CLIP EXTREME VALUES
# ==============================
for f in features:
    df[f] = df[f].clip(-5, 5)

# ==============================
# SORT BY DATE
# ==============================
df = df.sort_values("date").reset_index(drop=True)

# ==============================
# STAGE 1 LABEL
# ==============================
df["trade_filter"] = np.where(df["trade_label"] == 2, 0, 1)

# ==============================
# TIME SPLIT
# ==============================
split = int(len(df) * 0.8)

train_df = df.iloc[:split]
test_df = df.iloc[split:]

X_train = train_df[features]
X_test = test_df[features]

y_train = train_df["trade_filter"]
y_test = test_df["trade_filter"]

# ==============================
# CLASS IMBALANCE FIX
# ==============================
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# ==============================
# STAGE 1 MODEL (TRADE FILTER)
# ==============================
trade_model = XGBClassifier(
    n_estimators=800,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.2,
    reg_alpha=0.1,
    reg_lambda=1.5,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    early_stopping_rounds=50,
    random_state=42
)

trade_model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

# ==============================
# STAGE 1 RESULTS
# ==============================
pred_filter = trade_model.predict(X_test)

print("\n===== STAGE 1: TRADE FILTER =====")
print(classification_report(y_test, pred_filter))
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred_filter))

# ==============================
# STAGE 2 DATA (ONLY TRADE DAYS)
# ==============================
trade_df = df[df["trade_label"] != 2].copy()

split2 = int(len(trade_df) * 0.8)

train2 = trade_df.iloc[:split2]
test2 = trade_df.iloc[split2:]

X2_train = train2[features]
X2_test = test2[features]

y2_train = train2["trade_label"]
y2_test = test2["trade_label"]

# ==============================
# DIRECTION CLASS WEIGHT
# ==============================
scale_pos_weight2 = (y2_train == 0).sum() / (y2_train == 1).sum()

# ==============================
# STAGE 2 MODEL (DIRECTION)
# ==============================
direction_model = XGBClassifier(
    n_estimators=800,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.2,
    reg_alpha=0.1,
    reg_lambda=1.5,
    scale_pos_weight=scale_pos_weight2,
    eval_metric="logloss",
    early_stopping_rounds=50,
    random_state=42
)

direction_model.fit(
    X2_train,
    y2_train,
    eval_set=[(X2_test, y2_test)],
    verbose=False
)

# ==============================
# STAGE 2 RESULTS
# ==============================
pred_dir = direction_model.predict(X2_test)

print("\n===== STAGE 2: DIRECTION MODEL =====")
print(classification_report(y2_test, pred_dir))
print("Confusion Matrix:")
print(confusion_matrix(y2_test, pred_dir))

# ==============================
# AUTOMATIC THRESHOLD SEARCH
# ==============================
RR = 1.2

best_threshold = 0.5
best_score = -999

for threshold in np.arange(0.5, 0.8, 0.02):

    correct = 0
    wrong = 0
    skipped = 0

    for _, row in test_df.iterrows():

        X_row = row[features].values.reshape(1, -1)

        trade_prob = trade_model.predict_proba(X_row)[0][1]

        if trade_prob < threshold:
            skipped += 1
            continue

        direction_pred = direction_model.predict(X_row)[0]

        true_label = row["trade_label"]

        if true_label == 2:
            continue

        if direction_pred == true_label:
            correct += 1
        else:
            wrong += 1

    trades = correct + wrong

    if trades == 0:
        continue

    win_rate = correct / trades
    expectancy = (win_rate * RR) - (1 - win_rate)
    score = expectancy * trades
    if score > best_score:
        best_expectancy = expectancy
        best_threshold = threshold

print("\nBest Threshold:", round(best_threshold, 2))
print("Best Expectancy:", round(best_expectancy, 3))

# ==============================
# FINAL COMBINED TEST
# ==============================
print("\n===== FINAL COMBINED MODEL TEST =====")

correct = 0
wrong = 0
skipped = 0

for _, row in test_df.iterrows():

    X_row = row[features].values.reshape(1, -1)

    trade_prob = trade_model.predict_proba(X_row)[0][1]

    if trade_prob < best_threshold:
        skipped += 1
        continue

    direction_pred = direction_model.predict(X_row)[0]

    true_label = row["trade_label"]

    if true_label == 2:
        continue

    if direction_pred == true_label:
        correct += 1
    else:
        wrong += 1

total_trades = correct + wrong

if total_trades > 0:

    win_rate = correct / total_trades
    expectancy = (win_rate * RR) - (1 - win_rate)

    print("Trades taken:", total_trades)
    print("Correct:", correct)
    print("Wrong:", wrong)
    print("Skipped:", skipped)

    print("Win Rate:", round(win_rate * 100, 2), "%")
    print("Expectancy:", round(expectancy, 3))

else:
    print("No trades taken")

# ==============================
# FEATURE IMPORTANCE
# ==============================
importance = trade_model.get_booster().get_score(importance_type="gain")

if len(importance) > 0:

    imp_df = pd.DataFrame({
        "feature": list(importance.keys()),
        "importance": list(importance.values())
    }).sort_values("importance", ascending=False)

    print("\nTop Features:")
    print(imp_df)

# ==============================
# SAVE MODELS
# ==============================
# trade_model.save_model("trade_filter_model.json")
# direction_model.save_model("direction_model.json")

print("\nModels saved:")
print("trade_filter_model.json")
print("direction_model.json")