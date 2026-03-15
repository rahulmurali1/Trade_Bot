import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

# ==============================
# LOAD DATA
# ==============================

df = pd.read_excel("reliance_label_outcome.xlsx")

print("Trade label distribution:")
print(df['trade_label'].value_counts())

# ==============================
# FILTER ONLY 9:20 CANDLES
# ==============================

df = df[df['clock_time'] == "09:20:00"].copy()

# ==============================
# KEEP ONLY REAL TRADES
# ==============================

# remove:
# 0 = no_trade_day
# 1 = autosquare_off_day

df = df[df['trade_label'].isin([2,3])].copy()

# ==============================
# ENCODE LABELS
# ==============================

# Direction
# 2 = bearish -> 0
# 3 = bullish -> 1

df['trade_label_encoded'] = df['trade_label'].map({
    2: 0,
    3: 1
})

# Outcome
# 0 = SL hit
# 1 = Target hit

df['trade_outcome_encoded'] = df['trade_outcome'].map({
    0: 0,
    1: 1
})

# Remove any remaining NaN
df = df.dropna(subset=['trade_label_encoded','trade_outcome_encoded'])

# ==============================
# FEATURES
# ==============================

features = [
    'open','high','low','close',
    'volume','avg_price','avg_order_value',
    'PP','BC','TC','S1','R1','S2','R2'
]

# ==============================
# MODEL 1 : TRADE DIRECTION
# ==============================

X_dir = df[features]
y_dir = df['trade_label_encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X_dir,
    y_dir,
    test_size=0.2,
    random_state=42,
    stratify=y_dir
)

model_direction = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    objective='binary:logistic',
    eval_metric='logloss'
)

model_direction.fit(X_train, y_train)

pred_dir = model_direction.predict(X_test)

print("========== TRADE DIRECTION MODEL ==========")
print(classification_report(y_test, pred_dir))
print("Confusion Matrix:")
print(confusion_matrix(y_test, pred_dir))

# ==============================
# MODEL 2 : TRADE OUTCOME
# ==============================

X_out = df[features]
y_out = df['trade_outcome_encoded']

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X_out,
    y_out,
    test_size=0.2,
    random_state=42,
    stratify=y_dir
)

model_outcome = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    objective='binary:logistic',
    eval_metric='logloss'
)

model_outcome.fit(X2_train, y2_train)

pred_out = model_outcome.predict(X2_test)

print("\n========== TRADE OUTCOME MODEL ==========")
print(classification_report(y2_test, pred_out))
print("Confusion Matrix:")
print(confusion_matrix(y2_test, pred_out))

# ==============================
# EXAMPLE PREDICTION
# ==============================

sample = df[features].iloc[-1:].copy()

direction_prob = model_direction.predict_proba(sample)[0]

print("\nDirection probabilities:")
print("Bearish:", direction_prob[0])
print("Bullish:", direction_prob[1])

direction_pred = model_direction.predict(sample)[0]

if direction_pred == 1:
    print("Prediction: Bullish trade")
else:
    print("Prediction: Bearish trade")

# ==============================
# OUTCOME PREDICTION
# ==============================

outcome_prob = model_outcome.predict_proba(sample)[0]

print("\nOutcome probabilities:")
print("SL hit:", outcome_prob[0])
print("Target hit:", outcome_prob[1])

outcome_pred = model_outcome.predict(sample)[0]

if outcome_pred == 1:
    print("Prediction: Target likely to hit")
else:
    print("Prediction: Stop loss likely")