# Multi class model prediction with OHLC, and CPR


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight

# ==============================
# LOAD DATA
# ==============================
df = pd.read_excel("reliance_final_ml_datasetv2.xlsx")

print("Trade label distribution (original):")
print(df['trade_label'].value_counts())

# ==============================
# FILTER ONLY 9:20 CANDLES
# ==============================
df = df[df['clock_time'] == "09:20:00"].copy()

# ==============================
# FEATURES
# ==============================
features = [
    'open','high','low','close',
    'volume','avg_price','avg_order_value',
    'PP','BC','TC','S1','R1','S2','R2'
]

# ==============================
# MULTI-CLASS LABELS
# ==============================
# 0 = bearish, 1 = bullish, 2 = no_signal
df['trade_label_encoded'] = df['trade_label']

# Remove any remaining NaN
df = df.dropna(subset=['trade_label_encoded'])

X = df[features]
y = df['trade_label_encoded']

# ==============================
# TRAIN / TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

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


#--------------Results---------------------
# Trade label distribution (original):
# trade_label
# 2.0    488
# 0.0    309
# 1.0    217
# Name: count, dtype: int64

# Class weights: {np.float64(0.0): np.float64(1.0944669365721997), np.float64(1.0): np.float64(1.553639846743295), np.float64(2.0): np.float64(0.6931623931623931)}

# ========== MULTI-CLASS TRADE LABEL MODEL ==========
#               precision    recall  f1-score   support

#          0.0       0.50      0.45      0.47        62
#          1.0       0.41      0.37      0.39        43
#          2.0       0.58      0.64      0.61        98

#     accuracy                           0.53       203
#    macro avg       0.50      0.49      0.49       203
# weighted avg       0.52      0.53      0.52       203

# Confusion Matrix:
# [[28  5 29]
#  [11 16 16]
#  [17 18 63]]

# Prediction probabilities:
# Bearish: 0.12
# Bullish: 0.37
# No signal: 0.51
# Prediction: No signal / skip trade
