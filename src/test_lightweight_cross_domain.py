import os
import time
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"D:\cs427-ids-project"

PROCESSED_DATA = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "cic_iot2023_processed.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)


# ============================================================
# FEATURES
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


# ============================================================
# MODELS TO TEST
# ============================================================

TREE_COUNTS = [10, 25, 50, 100]


# ============================================================
# LOAD CICIoT2023
# ============================================================

print("=" * 80)
print("LIGHTWEIGHT CROSS-DOMAIN TEST")
print("=" * 80)

print("\nLoading CICIoT2023 processed data...")

df = pd.read_csv(PROCESSED_DATA)

print(f"Rows loaded: {len(df):,}")

X = df[FEATURES]
y = df["Target"]

print("\nFeatures:")
print(FEATURES)

print("\nTarget distribution:")
print(y.value_counts().sort_index())


# ============================================================
# TEST EACH MODEL
# ============================================================

results = []

for trees in TREE_COUNTS:

    print("\n" + "=" * 80)
    print(f"TESTING RANDOM FOREST: {trees} TREES")
    print("=" * 80)

    model_path = os.path.join(
        MODEL_DIR,
        f"random_forest_{trees}_trees.joblib"
    )

    if not os.path.exists(model_path):

        # The baseline uses a different filename.
        if trees == 100:
            model_path = os.path.join(
                MODEL_DIR,
                "random_forest_baseline.joblib"
            )

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        continue

    print(f"\nLoading model:")
    print(model_path)

    model = joblib.load(model_path)

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print("\nMaking predictions...")

    start_time = time.time()

    y_pred = model.predict(X)

    prediction_time = time.time() - start_time

    print(f"Prediction time: {prediction_time:.4f} seconds")

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(y, y_pred)

    precision = precision_score(
        y,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y,
        y_pred,
        zero_division=0
    )

    cm = confusion_matrix(y, y_pred)

    # --------------------------------------------------------
    # Model size
    # --------------------------------------------------------

    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    print("\nResults:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print(f"\nModel size: {model_size_mb:.2f} MB")

    results.append({
        "Model": f"Random Forest ({trees} trees)",
        "Trees": trees,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Prediction_Time_Seconds": prediction_time,
        "Model_Size_MB": model_size_mb
    })


# ============================================================
# COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("CROSS-DOMAIN MODEL COMPARISON")
print("=" * 80)

if not results_df.empty:

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    output_path = os.path.join(
        RESULTS_DIR,
        "lightweight_cross_domain_comparison.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print("\nComparison saved to:")
    print(output_path)

else:

    print("No models were tested.")


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 80)
print("CROSS-DOMAIN EXPERIMENT COMPLETE")
print("=" * 80)

print("\nIMPORTANT:")
print("No model was retrained.")
print("No model settings were changed.")
print("The original 100-tree baseline was NOT changed.")