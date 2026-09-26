import pandas as pd
import numpy as np
import json
import time
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import joblib


# ============================================================
# PATHS
# ============================================================

BASE = Path(r"D:\cs427-ids-project")

DATA_FILE = BASE / "data" / "processed" / "5g_nidd_processed.csv"

MODEL_DIR = BASE / "models"
RESULTS_DIR = BASE / "results"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

FEATURES = [
    "Rate",
    "Packet_Count",
    "Mean_Packet_Size",
    "TTL",
    "TCP",
    "UDP",
    "ICMP"
]

TARGET = "Target"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("RANDOM FOREST BASELINE")
print("=" * 80)

print("\nLoading processed 5G-NIDD...")

df = pd.read_csv(DATA_FILE)

print(f"Rows loaded: {len(df):,}")


# ============================================================
# CHECK DATA
# ============================================================

print("\nChecking data...")

print("\nMissing values:")
print(df[FEATURES + [TARGET]].isna().sum())

print("\nTarget distribution:")
print(df[TARGET].value_counts().sort_index())


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df[FEATURES]
y = df[TARGET]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 80)
print("TRAIN / TEST SPLIT")
print("=" * 80)

print("\nUsing 80% training / 20% testing")
print("Stratified split")
print(f"Random state: {RANDOM_STATE}")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"\nTraining rows: {len(X_train):,}")
print(f"Testing rows:  {len(X_test):,}")


print("\nTraining target distribution:")
print(y_train.value_counts().sort_index())

print("\nTesting target distribution:")
print(y_test.value_counts().sort_index())


# ============================================================
# RANDOM FOREST
# ============================================================

print("\n" + "=" * 80)
print("TRAINING RANDOM FOREST")
print("=" * 80)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight=None,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

print("\nModel settings:")
print("Algorithm: Random Forest")
print("Trees: 100")
print("Class weight: None")
print("Max features: sqrt")
print("Parallel processing: all CPU cores")

print("\nTraining...")

start_time = time.time()

model.fit(X_train, y_train)

training_time = time.time() - start_time

print(f"\nTraining completed in {training_time:.2f} seconds")


# ============================================================
# PREDICTION
# ============================================================

print("\nMaking predictions...")

prediction_start = time.time()

y_pred = model.predict(X_test)

prediction_time = time.time() - prediction_start

print(f"Prediction completed in {prediction_time:.2f} seconds")


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

cm = confusion_matrix(y_test, y_pred)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 80)
print("IN-DOMAIN RESULTS: 5G-NIDD TEST SET")
print("=" * 80)

print(f"\nAccuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Benign", "Malicious"],
        digits=4,
        zero_division=0
    )
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 80)
print("FEATURE IMPORTANCE")
print("=" * 80)

importance = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\n")
print(importance.to_string(index=False))


# ============================================================
# MODEL SIZE
# ============================================================

model_path = MODEL_DIR / "random_forest_baseline.joblib"

joblib.dump(model, model_path)

model_size_mb = model_path.stat().st_size / (1024 * 1024)

print("\n" + "=" * 80)
print("MODEL INFORMATION")
print("=" * 80)

print(f"\nModel saved to:")
print(model_path)

print(f"\nModel size: {model_size_mb:.2f} MB")
print(f"Training time: {training_time:.2f} seconds")
print(f"Prediction time: {prediction_time:.2f} seconds")


# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "dataset": "5G-NIDD",
    "experiment": "In-domain baseline",
    "model": "Random Forest",
    "n_estimators": 100,
    "random_state": RANDOM_STATE,
    "features": FEATURES,
    "training_rows": int(len(X_train)),
    "testing_rows": int(len(X_test)),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "training_time_seconds": float(training_time),
    "prediction_time_seconds": float(prediction_time),
    "model_size_mb": float(model_size_mb),
    "confusion_matrix": cm.tolist()
}

results_path = RESULTS_DIR / "baseline_5g_results.json"

with open(results_path, "w") as f:
    json.dump(results, f, indent=4)


importance_path = RESULTS_DIR / "baseline_feature_importance.csv"

importance.to_csv(
    importance_path,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("\nResults saved to:")
print(results_path)

print("\nFeature importance saved to:")
print(importance_path)

print("\n" + "=" * 80)
print("BASELINE TRAINING COMPLETE")
print("=" * 80)