import os
import time
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "5g_nidd_processed.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


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

TREE_COUNTS = [
    10,
    25,
    50
]

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("LIGHTWEIGHT RANDOM FOREST EXPERIMENT")
print("=" * 80)

print("\nLoading 5G-NIDD processed data...")

df = pd.read_csv(DATA_PATH)

print(f"Rows loaded: {len(df):,}")

X = df[FEATURES]
y = df[TARGET]


# ============================================================
# SAME TRAIN/TEST SPLIT AS BASELINE
# ============================================================

print("\nCreating the same 80/20 stratified split used by baseline...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE
)

print(f"Training samples: {len(X_train):,}")
print(f"Testing samples:  {len(X_test):,}")


# ============================================================
# BASELINE RESULTS
# ============================================================

baseline_path = os.path.join(
    RESULTS_DIR,
    "baseline_5g_results.json"
)

baseline_results = {}

if os.path.exists(baseline_path):

    import json

    with open(
        baseline_path,
        "r"
    ) as f:
        baseline_results = json.load(f)

    print("\nExisting baseline results loaded.")

else:

    print("\nWARNING: baseline_5g_results.json not found.")


# ============================================================
# EXPERIMENT
# ============================================================

results = []

for n_trees in TREE_COUNTS:

    print("\n" + "=" * 80)
    print(f"TESTING RANDOM FOREST: {n_trees} TREES")
    print("=" * 80)

    model = RandomForestClassifier(
        n_estimators=n_trees,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight=None,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\nTraining model...")

    train_start = time.time()

    model.fit(
        X_train,
        y_train
    )

    train_time = time.time() - train_start

    print(
        f"Training time: {train_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print("Making predictions...")

    predict_start = time.time()

    predictions = model.predict(
        X_test
    )

    predict_time = time.time() - predict_start

    print(
        f"Prediction time: {predict_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        f"random_forest_{n_trees}_trees.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    model_size_mb = (
        os.path.getsize(model_path)
        / (1024 * 1024)
    )

    print("\nResults:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Model size: {model_size_mb:.2f} MB")

    results.append({
        "Model": f"Random Forest ({n_trees} trees)",
        "Trees": n_trees,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Training_Time_Seconds": train_time,
        "Prediction_Time_Seconds": predict_time,
        "Model_Size_MB": model_size_mb
    })


# ============================================================
# ADD BASELINE
# ============================================================

if baseline_results:

    baseline_row = {
        "Model": "Random Forest (100 trees) - Baseline",
        "Trees": 100,
        "Accuracy": baseline_results.get("accuracy"),
        "Precision": baseline_results.get("precision"),
        "Recall": baseline_results.get("recall"),
        "F1": baseline_results.get("f1_score"),
        "Training_Time_Seconds": baseline_results.get("training_time_seconds"),
        "Prediction_Time_Seconds": baseline_results.get("prediction_time_seconds"),
        "Model_Size_MB": 147.14
    }

    results.insert(
        0,
        baseline_row
    )


# ============================================================
# COMPARISON TABLE
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    200
)

pd.set_option(
    "display.float_format",
    "{:.4f}".format
)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_path = os.path.join(
    RESULTS_DIR,
    "lightweight_model_comparison.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

print("\n" + "=" * 80)
print("EXPERIMENT COMPLETE")
print("=" * 80)

print("\nComparison saved to:")
print(output_path)

print("\nModels saved to:")
print(MODEL_DIR)

print("\nIMPORTANT:")
print("The original 100-tree baseline model was NOT changed.")