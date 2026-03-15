#### This one works better ####

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

np.random.seed(42)

# ==============================
# LOAD DATA
# ==============================
file_name = "top10_stocks_dataset_v9.xlsx"

df = pd.read_excel(file_name)

print("Original label distribution:")
print(df["trade_label"].value_counts())

# ==============================
# FILTER ENTRY CANDLE
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

# Drop rows with missing features
df = df.dropna(subset=features)

# Drop rows with missing label
df = df.dropna(subset=["trade_label"])

# Convert label to integer
df["trade_label"] = df["trade_label"].astype(int)

# ==============================
# SORT BY DATE (IMPORTANT)
# ==============================
df = df.sort_values("date")

# ==============================
# CREATE DATASET
# ==============================
X = df[features]
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
classes = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = dict(zip(classes, weights))

print("\nClass weights:", class_weights)

sample_weights = y_train.map(class_weights)

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
    y_train,
    sample_weight=sample_weights
)

# ==============================
# FEATURE IMPORTANCE
# ==============================
importance = model.feature_importances_

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
# model.save_model("trade_model.json")
# print("\nModel saved.")