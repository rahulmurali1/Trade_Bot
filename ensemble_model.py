import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ==============================
# LOAD DATA
# ==============================
file_name = "top10_stocks_dataset_v14.xlsx"
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
    "avg_order_value",
    "first_candle_range_pct",
    "range_vs_prevclose",
    "range_vs_prevhigh",
    "range_vs_prevlow",
    "distance_from_pp",
    "distance_from_s1",
    "distance_from_r1",
    "cpr_range",
    "pp_vs_prev_pp_pct"
]

# ==============================
# DATA CLEANING
# ==============================
df = df.dropna(subset=features + ["trade_label"])
df["trade_label"] = df["trade_label"].astype(int)
df = df[df["trade_label"].isin([0, 1])]  # Only 0 and 1

# ==============================
# CREATE TRAIN/TEST SPLIT
# ==============================
X = df[features]
y = df["trade_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# ==============================
# RANDOM FOREST MODEL
# ==============================
rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# ==============================
# TRAIN MODEL
# ==============================
rf_model.fit(X_train, y_train)

# ==============================
# FEATURE IMPORTANCE
# ==============================
importance = rf_model.feature_importances_
imp_df = pd.DataFrame({
    "feature": features,
    "importance": importance
}).sort_values("importance", ascending=False)

print("\nTop Features:")
print(imp_df)

# ==============================
# PREDICTIONS
# ==============================
pred = rf_model.predict(X_test)
pred_probs = rf_model.predict_proba(X_test)

print("\n========= MODEL RESULTS =========")
print(classification_report(y_test, pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))

# ==============================
# SAMPLE PREDICTION
# ==============================
sample = X_test.iloc[-1:]
sample_probs = rf_model.predict_proba(sample)[0]
sample_class = rf_model.predict(sample)[0]

print("\nSample Prediction Probabilities:")
print(f"Class 0 probability: {sample_probs[0]:.2f}")
print(f"Class 1 probability: {sample_probs[1]:.2f}")
print(f"\nSample Prediction Result: {'Bullish' if sample_class == 1 else 'Bearish'}")

# ==============================
# SAVE MODEL
# ==============================
# import joblib
# joblib.dump(rf_model, "rf_trade_model.pkl")
print("\nModel saved as 'rf_trade_model.pkl'")