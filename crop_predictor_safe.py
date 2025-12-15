import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    r2_score,
    mean_absolute_error,
)
import pickle

# ----------------------------------
# Python version check
# ----------------------------------
if sys.version_info < (3, 8):
    print("Warning: Python 3.8+ is recommended. Current:", sys.version)
    print("Running with Python:", sys.version.split()[0])

# ----------------------------------
# Locate dataset
# ----------------------------------
DATAFILE = "crop_recommendation.csv"
print(f"\nLooking for dataset file: {DATAFILE}")
if not os.path.isfile(DATAFILE):
    print(
        f"ERROR: Dataset file '{DATAFILE}' not found in current working directory:\n {os.getcwd()}"
    )
    sys.exit(1)

# ----------------------------------
# Load dataset
# ----------------------------------
try:
    data = pd.read_csv(DATAFILE)
    print("Dataset loaded. Shape:", data.shape)
except Exception:
    print("ERROR reading CSV file.")
    raise

# ----------------------------------
# Rename columns if needed
# ----------------------------------
column_map = {
    "Nitrogen": "N",
    "Phosphorus": "P",
    "Potassium": "K",
    "Temperature": "temperature",
    "Humidity": "humidity",
    "pH_Value": "ph",
    "Rainfall": "rainfall",
    "Crop": "label",
}
data = data.rename(columns=column_map)

required_columns = {
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
    "label",
    "rates",
    "yield",
}
missing = required_columns - set(data.columns)
if missing:
    print(f"ERROR: Dataset is missing columns: {missing}")
    sys.exit(1)

print("Column check passed. Using these columns:")
print(list(data.columns))

print("\nLabel distribution:")
print(data["label"].value_counts())

# -------------------------------
# MODEL 1: CROP RECOMMENDATION
# -------------------------------
print("\n=== Training crop recommendation model (classification) ===")

X_cls = data[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]]
y_cls = data["label"]

print("\nClassification feature dtypes:")
print(X_cls.dtypes)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)

clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
)
clf.fit(X_train_c, y_train_c)
acc = clf.score(X_test_c, y_test_c)
print(f"\nCrop model accuracy on test set: {acc*100:.2f}%")

print("\nClassification report:")
print(classification_report(y_test_c, clf.predict(X_test_c)))

print("\nConfusion matrix:")
print(confusion_matrix(y_test_c, clf.predict(X_test_c)))

# -------------------------------
# MODEL 2: YIELD (AND RATE) PREDICTION
# -------------------------------
print("\n=== Training yield prediction model (regression) ===")

# Encode crop label as one-hot so regressor can use it
data_reg = pd.get_dummies(data, columns=["label"], drop_first=False)

feature_cols_reg = (
    ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    + [c for c in data_reg.columns if c.startswith("label_")]
)
X_reg = data_reg[feature_cols_reg]
y_yield = data_reg["yield"]

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_yield, test_size=0.2, random_state=42
)

reg_yield = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
)
reg_yield.fit(X_train_r, y_train_r)

y_pred_yield = reg_yield.predict(X_test_r)
print("\nYield regression performance:")
print("R2 score:", r2_score(y_test_r, y_pred_yield))
print("MAE:", mean_absolute_error(y_test_r, y_pred_yield))

# (Optional) separate regressor for rates
y_rate = data_reg["rates"]
X_train_rr, X_test_rr, y_train_rr, y_test_rr = train_test_split(
    X_reg, y_rate, test_size=0.2, random_state=42
)
reg_rate = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
)
reg_rate.fit(X_train_rr, y_train_rr)
y_pred_rate = reg_rate.predict(X_test_rr)
print("\nRate regression performance:")
print("R2 score:", r2_score(y_test_rr, y_pred_rate))
print("MAE:", mean_absolute_error(y_test_rr, y_pred_rate))

# -------------------------------
# SAVE MODELS
# -------------------------------
out_cls = "crop_model.pkl"
out_yield = "yield_model.pkl"
out_rate = "rate_model.pkl"

with open(out_cls, "wb") as f:
    pickle.dump(clf, f)
with open(out_yield, "wb") as f:
    pickle.dump(reg_yield, f)
with open(out_rate, "wb") as f:
    pickle.dump(reg_rate, f)

print(f"\nSaved models: {out_cls}, {out_yield}, {out_rate}")

# -------------------------------
# DEMO FUNCTIONS
# -------------------------------
def predict_crop(N, P, K, temperature, humidity, ph, rainfall):
    df = pd.DataFrame(
        [[N, P, K, temperature, humidity, ph, rainfall]],
        columns=["N", "P", "K", "temperature", "humidity", "ph", "rainfall"],
    )
    return clf.predict(df)[0]


def predict_yield_and_rate(N, P, K, temperature, humidity, ph, rainfall, crop_label):
    # one-hot encode crop label in same way as training
    base = {
        "N": N,
        "P": P,
        "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall,
    }
    for c in [c for c in data_reg.columns if c.startswith("label_")]:
        base[c] = 0
    col_name = f"label_{crop_label}"
    if col_name in base:
        base[col_name] = 1
    df = pd.DataFrame([base], columns=feature_cols_reg)
    pred_yield = reg_yield.predict(df)[0]
    pred_rate = reg_rate.predict(df)[0]
    return pred_yield, pred_rate


print("\nDemo end-to-end prediction (classification + yield/rate):")
try:
    crop_demo = predict_crop(42, 0, 0, 25, 80, 6.5, 200)
    y_demo, r_demo = predict_yield_and_rate(42, 0, 0, 25, 80, 6.5, 200, crop_demo)
    print("Predicted crop:", crop_demo)
    print("Predicted yield:", y_demo)
    print("Predicted rate:", r_demo)
except Exception:
    print("Demo prediction failed.")
    raise

print("\nAll done.")
