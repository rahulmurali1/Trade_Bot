### balancing training data to check if accuracy improves ######

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.utils.class_weight import compute_class_weight

# ==============================
# LOAD DATA
# ==============================
file_name = "top10_stocks_dataset_v8.xlsx"
df = pd.read_excel(file_name)
df["trade_label"] = df["trade_label"].astype(int)

print("Original label distribution:")
print(df["trade_label"].value_counts())

# ==============================
# USE ONLY ENTRY CANDLE (09:15)
# ==============================
df = df[df["clock_time"] == "09:15:00"].copy()

# ==============================
# FEATURES (ADD YOUR FEATURES HERE)
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

# Drop rows missing any feature
if len(features) > 0:
    df = df.dropna(subset=features)

# ==============================
# LABEL
# ==============================
# 0 = bearish
# 1 = bullish
# 2 = no signal
df = df.dropna(subset=["trade_label"])

df = df.sort_values("date")
X = df[features] if len(features) > 0 else pd.DataFrame(index=df.index)
y = df["trade_label"]

# ==============================
# TIME BASED SPLIT
# ==============================

split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# ==============================
# CLASS WEIGHTS
# ==============================
# ==============================
# BALANCED TRAINING DATA
# ==============================

train_df = X_train.copy()
train_df["trade_label"] = y_train.values

print("\nTrain distribution BEFORE balancing:")
print(train_df["trade_label"].value_counts())

# Find smallest class
min_class_size = train_df["trade_label"].value_counts().min()

balanced_train = (
    train_df
    .groupby("trade_label")
    .sample(n=min_class_size, random_state=42)
)

print("\nTrain distribution AFTER balancing:")
print(balanced_train["trade_label"].value_counts())

X_train = balanced_train[features]
y_train = balanced_train["trade_label"]

# ==============================
# MODEL
# ==============================
model = XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="multi:softprob",
    eval_metric="mlogloss",
    num_class=3,
    tree_method="hist",
    random_state=42
)

# ==============================
# TRAIN
# ==============================
model.fit(
    X_train,
    y_train
)
# ==============================
# FEATURE IMPORTANCE
# ==============================

importance = model.feature_importances_

if len(importance) > 0:

    imp_df = pd.DataFrame({
        "feature": features,
        "importance": importance
    }).sort_values("importance", ascending=False)

    print("\nTop Features:")
    print(imp_df)
# ==============================
# PREDICTIONS
# ==============================
pred = model.predict(X_test)

print("\n========= MODEL RESULTS =========")

print(classification_report(y_test, pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))

# ==============================
# SAMPLE PREDICTION
# ==============================
if len(features) > 0:

    sample = X_test.iloc[-1:]

    probs = model.predict_proba(sample)[0]

    print("\nPrediction probabilities:")

    print(f"Bearish: {probs[0]:.2f}")
    print(f"Bullish: {probs[1]:.2f}")
    print(f"No signal: {probs[2]:.2f}")

    pred_class = model.predict(sample)[0]

    if pred_class == 0:
        print("Prediction: Bearish trade")
    elif pred_class == 1:
        print("Prediction: Bullish trade")
    else:
        print("Prediction: No trade")

# ==============================
# SAVE MODEL
# ==============================
# model.save_model(f"{file_name}.json")

# print("\nModel saved: reliance_trade_model.json")

# Original label distribution:
# trade_label
# 2.0    404
# 0.0    315
# 1.0    294
# Name: count, dtype: int64

# Class weights: {np.float64(0.0): np.float64(0.9263721552878179), np.float64(1.0): np.float64(0.9815602836879432), np.float64(2.0): np.float64(1.108974358974359)}

# Top Features:
#                  feature  importance
# 15        close_above_pp    0.951183
# 2       first_candle_pct    0.888891
# 13         open_above_pp    0.768159
# 9        relative_volume    0.734319
# 3      first_15min_range    0.707345
# 17               gap_pct    0.682428
# 5       distance_from_s1    0.681804
# 6       distance_from_r1    0.678616
# 8         price_vs_ema50    0.675754
# 7         price_vs_ema20    0.663140
# 12          PP_to_S2_pct    0.646458
# 0                 volume    0.640867
# 4       distance_from_pp    0.640347
# 16       avg_pp_dist_pct    0.631567
# 1        avg_order_value    0.631198
# 11          PP_to_S1_pct    0.609527
# 14  touch_pp_directional    0.581859
# 10          PP_to_BC_pct    0.548738

# ========= MODEL RESULTS =========
#               precision    recall  f1-score   support

#          0.0       0.48      0.56      0.52        55
#          1.0       0.46      0.57      0.51        58
#          2.0       0.62      0.38      0.47        60

#     accuracy                           0.50       173
#    macro avg       0.52      0.51      0.50       173
# weighted avg       0.52      0.50      0.50       173

# Confusion Matrix:
# [[31 19  5]
#  [16 33  9]
#  [18 19 23]]

# Prediction probabilities:
# Bearish: 0.24
# Bullish: 0.27
# No signal: 0.49
# Prediction: No trade