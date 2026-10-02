import pandas as pd
import numpy as np
import json
import time
from pathlib import Path

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

BASE = Path(__file__).resolve().parent.parent

MODEL_FILE = BASE / "models" / "random_forest_baseline.joblib"
CIC_FILE = BASE / "data" / "processed" / "cic_iot2023_processed.csv"

RESULTS_DIR = BASE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

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
# LOAD MODEL
# ============================================================

print("=" * 80)
print("CROSS-DOMAIN EVALUATION")
print("=" * 80)

print("\nLoading trained Random Forest...")

model = joblib.load(MODEL_FILE)

print("Model loaded successfully.")
print(f"Model: {MODEL_FILE}")


# ============================================================
# LOAD CIC DATA
# ============================================================

print("\n" + "=" * 80)
print("LOADING CICIoT2023")
print("=" * 80)

print("\nLoading processed CICIoT2023...")

df = pd.read_csv(
    CIC_FILE,
    low_memory=False
)

print(f"Rows loaded: {len(df):,}")


# ============================================================
# DATA CHECK
# ============================================================

print("\nChecking data...")

print("\nMissing values:")
print(df[FEATURES + [TARGET]].isna().sum())

print("\nTarget distribution:")
print(df[TARGET].value_counts().sort_index())


# ============================================================
# FEATURES AND TARGET
# ============================================================

X_cic = df[FEATURES]
y_cic = df[TARGET]


# ============================================================
# CROSS-DOMAIN PREDICTION
# ============================================================

print("\n" + "=" * 80)
print("CROSS-DOMAIN PREDICTION")
print("=" * 80)

print("\nIMPORTANT:")
print("The Random Forest was trained ONLY on 5G-NIDD.")
print("No CICIoT2023 data is used for training.")
print("No retraining is performed.")
print("No model parameters are changed.")

print("\nMaking predictions...")

start_time = time.time()

y_pred = model.predict(X_cic)

prediction_time = time.time() - start_time

print(
    f"\nPrediction completed in "
    f"{prediction_time:.2f} seconds"
)


# ============================================================
# OVERALL METRICS
# ============================================================

accuracy = accuracy_score(
    y_cic,
    y_pred
)

precision = precision_score(
    y_cic,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_cic,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_cic,
    y_pred,
    zero_division=0
)

cm = confusion_matrix(
    y_cic,
    y_pred
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 80)
print("CROSS-DOMAIN RESULTS: CICIoT2023")
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
        y_cic,
        y_pred,
        target_names=["Benign", "Malicious"],
        digits=4,
        zero_division=0
    )
)


# ============================================================
# PERFORMANCE COMPARISON
# ============================================================

# 5G-NIDD baseline results from previous experiment
baseline_file = RESULTS_DIR / "baseline_5g_results.json"

with open(baseline_file, "r") as f:
    baseline = json.load(f)


baseline_accuracy = baseline["accuracy"]
baseline_precision = baseline["precision"]
baseline_recall = baseline["recall"]
baseline_f1 = baseline["f1_score"]


accuracy_change = accuracy - baseline_accuracy
precision_change = precision - baseline_precision
recall_change = recall - baseline_recall
f1_change = f1 - baseline_f1


print("\n" + "=" * 80)
print("IN-DOMAIN VS CROSS-DOMAIN")
print("=" * 80)

comparison = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score"
    ],
    "5G-NIDD In-Domain": [
        baseline_accuracy,
        baseline_precision,
        baseline_recall,
        baseline_f1
    ],
    "CICIoT2023 Cross-Domain": [
        accuracy,
        precision,
        recall,
        f1
    ],
    "Change": [
        accuracy_change,
        precision_change,
        recall_change,
        f1_change
    ]
})

print(
    comparison.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# PERFORMANCE DROP
# ============================================================

print("\n" + "=" * 80)
print("PERFORMANCE CHANGE")
print("=" * 80)

print(
    f"\nAccuracy change:  "
    f"{accuracy_change:+.4f} "
    f"({accuracy_change * 100:+.2f} percentage points)"
)

print(
    f"Precision change: "
    f"{precision_change:+.4f} "
    f"({precision_change * 100:+.2f} percentage points)"
)

print(
    f"Recall change:    "
    f"{recall_change:+.4f} "
    f"({recall_change * 100:+.2f} percentage points)"
)

print(
    f"F1-score change:  "
    f"{f1_change:+.4f} "
    f"({f1_change * 100:+.2f} percentage points)"
)


# ============================================================
# ATTACK-TYPE ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("CICIoT2023 PERFORMANCE BY ATTACK TYPE")
print("=" * 80)

attack_results = []

# Use the original CIC label if available.
# The processed file does not contain it, so this section
# will be populated by matching row positions later.

print(
    "\nThe processed CIC file contains only the common features "
    "and binary Target."
)

print(
    "Attack-type analysis requires the original CIC labels."
)

print(
    "The script will therefore save the overall cross-domain "
    "results first."
)


# ============================================================
# SAVE OVERALL RESULTS
# ============================================================

results = {
    "experiment": "Cross-domain evaluation",
    "training_dataset": "5G-NIDD",
    "testing_dataset": "CICIoT2023",
    "model": "Random Forest",
    "features": FEATURES,
    "testing_rows": int(len(X_cic)),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "prediction_time_seconds": float(prediction_time),
    "confusion_matrix": cm.tolist(),

    "baseline_5g": {
        "accuracy": float(baseline_accuracy),
        "precision": float(baseline_precision),
        "recall": float(baseline_recall),
        "f1_score": float(baseline_f1)
    },

    "change": {
        "accuracy": float(accuracy_change),
        "precision": float(precision_change),
        "recall": float(recall_change),
        "f1_score": float(f1_change)
    }
}


results_file = RESULTS_DIR / "cross_domain_results.json"

with open(results_file, "w") as f:
    json.dump(
        results,
        f,
        indent=4
    )


comparison_file = RESULTS_DIR / "in_domain_vs_cross_domain.csv"

comparison.to_csv(
    comparison_file,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 80)
print("CROSS-DOMAIN EVALUATION COMPLETE")
print("=" * 80)

print("\nResults saved to:")
print(results_file)

print("\nComparison saved to:")
print(comparison_file)