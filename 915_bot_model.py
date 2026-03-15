##### currently using ######

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ==============================
# LOAD DATA
# ==============================
file_name = "top10_stocks_dataset_v8.xlsx"
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
    "cpr_range",
    "r1_s1_range",
    "r2_s2_range",
    "pp_vs_prev_pp_pct",
    "cpr_range_pct",
    "r1_s1_range_pct",
    "r2_s2_range_pct",
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
    "open915_vs_low915_pct",
    "nr4",
    "nr7",
    "open915_vs_prevhigh_pct",
    "open915_vs_prevlow_pct",
    "close915_vs_prevhigh_pct",
    "close915_vs_prevlow_pct",
    "high915_vs_prevhigh_pct",
    "low915_vs_prevhigh_pct",
    "high915_vs_prevlow_pct",
    "low915_vs_prevlow_pct"
]

df = df.dropna(subset=features + ["trade_label"])

# ==============================
# SORT BY DATE
# ==============================
df = df.sort_values("date").reset_index(drop=True)

# ==============================
# STAGE 1 LABEL
# ==============================
# 0 = No Trade
# 1 = Trade
df["trade_filter"] = np.where(df["trade_label"] == 2, 0, 1)

X = df[features]
y_filter = df["trade_filter"]

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
# CLASS WEIGHTS
# ==============================
classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))
sample_weights = y_train.map(class_weights)

# ==============================
# STAGE 1 MODEL
# ==============================
trade_model = XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

trade_model.fit(
    X_train,
    y_train,
    sample_weight=sample_weights
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
# STAGE 2 DATA (TRADE DAYS ONLY)
# ==============================
trade_df = df[df["trade_label"] != 2].copy()

X2 = trade_df[features]
y_direction = trade_df["trade_label"]

# ==============================
# SPLIT
# ==============================
split2 = int(len(trade_df) * 0.8)

X2_train = X2.iloc[:split2]
X2_test = X2.iloc[split2:]

y2_train = y_direction.iloc[:split2]
y2_test = y_direction.iloc[split2:]

# ==============================
# CLASS WEIGHTS
# ==============================
classes2 = np.unique(y2_train)

weights2 = compute_class_weight(
    class_weight="balanced",
    classes=classes2,
    y=y2_train
)

class_weights2 = dict(zip(classes2, weights2))
sample_weights2 = y2_train.map(class_weights2)

# ==============================
# STAGE 2 MODEL
# ==============================
direction_model = XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)

direction_model.fit(
    X2_train,
    y2_train,
    sample_weight=sample_weights2
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
# COMBINED MODEL TEST
# ==============================
print("\n===== COMBINED MODEL TEST =====")

TRADE_THRESHOLD = 0.6

correct = 0
wrong = 0
skipped = 0

for _, row in test_df.iterrows():

    X_row = row[features].values.reshape(1, -1)

    trade_prob = trade_model.predict_proba(X_row)[0][1]

    # Skip if model says no trade
    if trade_prob < TRADE_THRESHOLD:
        skipped += 1
        continue

    # Predict direction
    direction_pred = direction_model.predict(X_row)[0]

    true_label = row["trade_label"]

    # Skip evaluation if actual label is no trade
    if true_label == 2:
        continue

    if direction_pred == true_label:
        correct += 1
    else:
        wrong += 1

total_trades = correct + wrong

if total_trades > 0:

    win_rate = correct / total_trades

    print("Trades taken:", total_trades)
    print("Correct:", correct)
    print("Wrong:", wrong)
    print("Skipped:", skipped)

    print("Win Rate:", round(win_rate * 100, 2), "%")

else:
    print("No trades taken")

# ==============================
# EXPECTANCY CALCULATION
# ==============================
RR = 1.2

if total_trades > 0:

    expectancy = (win_rate * RR) - (1 - win_rate)

    print("Expectancy:", round(expectancy, 3))

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

# print("\nModels saved:")
# print("trade_filter_model.json")
# print("direction_model.json")



# ===== STAGE 1: TRADE FILTER =====
#               precision    recall  f1-score   support

#            0       0.83      0.66      0.74      1525
#            1       0.37      0.60      0.46       501

#     accuracy                           0.65      2026
#    macro avg       0.60      0.63      0.60      2026
# weighted avg       0.72      0.65      0.67      2026

# Confusion Matrix:
# [[1011  514]
#  [ 202  299]]

# ===== STAGE 2: DIRECTION MODEL =====
#               precision    recall  f1-score   support

#            0       0.57      0.61      0.59       323
#            1       0.59      0.54      0.56       331

#     accuracy                           0.58       654
#    macro avg       0.58      0.58      0.58       654
# weighted avg       0.58      0.58      0.58       654

# Confusion Matrix:
# [[198 125]
#  [152 179]]

# ===== COMBINED MODEL TEST =====
# Trades taken: 170
# Correct: 94
# Wrong: 76
# Skipped: 1619
# Win Rate: 55.29 %
# Expectancy: 0.216

# Top Features:
#                       feature  importance
# 21    open915_vs_close915_pct    9.757295
# 8           first_15min_range    7.356680
# 23      open915_vs_low915_pct    5.950242
# 22     open915_vs_high915_pct    5.623361
# 15    low915_vs_prevclose_pct    4.668792
# 16   high915_vs_prevclose_pct    4.528027
# 7             r2_s2_range_pct    4.511446
# 18         close915_vs_pp_pct    4.286005
# 14  close915_vs_prevclose_pct    4.214751
# 19           low915_vs_pp_pct    4.198237
# 4           pp_vs_prev_pp_pct    4.088943
# 12             price_vs_ema50    4.075571
# 6             r1_s1_range_pct    4.065240
# 11             price_vs_ema20    4.039145
# 5               cpr_range_pct    4.027388
# 10           distance_from_r1    4.010630
# 1                   cpr_range    3.981544
# 13   open915_vs_prevclose_pct    3.966268
# 20          high915_vs_pp_pct    3.917695
# 9            distance_from_s1    3.875503
# 0        ema20_ema50_pct_diff    3.844175
# 17          open915_vs_pp_pct    3.809981
# 2                 r1_s1_range    3.793360
# 3                 r2_s2_range    3.683933

# Models saved:
# trade_filter_model.json
# direction_model.json