import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# ==============================
# LOAD DATA
# ==============================
file_name = "COALINDIA_dataset"
df = pd.read_excel(f"{file_name}.xlsx")


print("Trade label distribution (original):")
print(df['trade_label'].value_counts())

# ==============================
# FILTER ONLY 9:20 CANDLES
# ==============================
# df = df[df['clock_time'] == "09:20:00"].copy()

# ==============================
# FEATURES
# ==============================
features = [
    'open', 'volume','avg_price','avg_order_value',
    'PP','PP_to_BC_pct', 'PP_to_S1_pct', 'PP_to_S2_pct', 'gap_pct', 'dist_915_high_pct', 'dist_915_low_pct', 'dist_915_max_pct'
]
df = df.dropna(subset=features)

# ==============================
# MULTI-CLASS LABELS
# ==============================
# 0 = bearish, 1 = bullish, 2 = no_signal
df['trade_label_encoded'] = df['trade_label']

# Remove any remaining NaN
df = df.dropna(subset=['trade_label_encoded'])
df = df.sort_values("date").reset_index(drop=True)

X = df[features]
y = df['trade_label_encoded']

# ==============================
# TRAIN / TEST SPLIT
# ==============================
split = int(len(df) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# ==============================
# CALCULATE CLASS WEIGHTS
# ==============================
classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))
print("\nClass weights:", class_weights)

# Create sample weights for XGBoost
sample_weights = y_train.map(class_weights)

# ==============================
# XGBOOST MULTI-CLASS MODEL
# ==============================
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    objective='multi:softprob',  # multi-class probability
    eval_metric='mlogloss',
    num_class=3,
    random_state=42
)

model.fit(X_train, y_train, sample_weight=sample_weights)

importance = model.get_booster().get_score(importance_type="gain")

imp_df = pd.DataFrame({
    "feature": list(importance.keys()),
    "importance": list(importance.values())
}).sort_values("importance", ascending=False)

print("\nTop Features:")
print(imp_df)

# ==============================
# PREDICTIONS & METRICS
# ==============================
pred = model.predict(X_test)

print("\n========== MULTI-CLASS TRADE LABEL MODEL ==========")
print(classification_report(y_test, pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred))

# ==============================
# EXAMPLE PREDICTION
# ==============================
sample = df[features].iloc[-1:].copy()
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
    print("Prediction: No signal / skip trade")

# Save the trained model to a file
model.save_model(f"{file_name}_model.json")
print(f"Model saved successfully as {file_name}_model.json")


# -------------------RESULT----------------------------

# Trade label distribution (original):
# trade_label
# 2.0    488
# 0.0    309
# 1.0    217
# Name: count, dtype: int64

# Class weights: {np.float64(0.0): np.float64(1.0944669365721997), np.float64(1.0): np.float64(1.553639846743295), np.float64(2.0): np.float64(0.6931623931623931)}

# ========== MULTI-CLASS TRADE LABEL MODEL ==========
#               precision    recall  f1-score   support

#          0.0       0.77      0.74      0.75        62
#          1.0       0.57      0.81      0.67        43
#          2.0       0.78      0.65      0.71        98

#     accuracy                           0.71       203
#    macro avg       0.71      0.74      0.71       203
# weighted avg       0.73      0.71      0.72       203

# Confusion Matrix:
# [[46  4 12]
#  [ 2 35  6]
#  [12 22 64]]

# Prediction probabilities:
# Bearish: 0.01
# Bullish: 0.96
# No signal: 0.04
# Prediction: Bullish trade