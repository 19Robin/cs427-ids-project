from pathlib import Path
import time
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from xgboost import XGBClassifier


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PROCESSED_DIR = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

FIVEG_FILE = PROCESSED_DIR / "5g_nidd_processed.csv"
CIC_FILE = PROCESSED_DIR / "cic_iot2023_processed.csv"


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

TARGET = "Target"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("OTHER MACHINE LEARNING MODELS")
print("=" * 70)

print("\nLoading 5G-NIDD...")

fiveg = pd.read_csv(FIVEG_FILE)

print(f"5G-NIDD rows: {len(fiveg):,}")

X = fiveg[FEATURES]
y = fiveg[TARGET]


print("\nLoading CICIoT2023...")

cic = pd.read_csv(CIC_FILE)

print(f"CICIoT2023 rows: {len(cic):,}")

X_cross = cic[FEATURES]
y_cross = cic[TARGET]


# ============================================================
# SAME TRAIN/TEST SPLIT AS RANDOM FOREST
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("\n5G-NIDD split:")
print(f"Training rows: {len(X_train):,}")
print(f"Testing rows:  {len(X_test):,}")


# ============================================================
# MODELS
# ============================================================

models = {

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=1.0,
        colsample_bytree=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )
}


# ============================================================
# RESULTS STORAGE
# ============================================================

results = []


# ============================================================
# TRAIN AND TEST MODELS
# ============================================================

for model_name, model in models.items():

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    print("\nTraining...")

    start = time.time()

    model.fit(
        X_train,
        y_train
    )

    training_time = time.time() - start

    print(
        f"Training time: {training_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # MODEL SIZE
    # --------------------------------------------------------

    model_path = (
        MODELS_DIR /
        f"{model_name.lower().replace(' ', '_')}.joblib"
    )

    joblib.dump(
        model,
        model_path
    )

    model_size_mb = (
        model_path.stat().st_size /
        (1024 * 1024)
    )

    print(
        f"Model size: {model_size_mb:.2f} MB"
    )

    # --------------------------------------------------------
    # IN-DOMAIN TEST
    # --------------------------------------------------------

    print("\nTesting on 5G-NIDD...")

    start = time.time()

    predictions = model.predict(
        X_test
    )

    prediction_time = time.time() - start

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

    print(
        f"Accuracy:  {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1:        {f1:.4f}"
    )

    print(
        f"Prediction time: {prediction_time:.4f} seconds"
    )

    # --------------------------------------------------------
    # CROSS-DOMAIN TEST
    # --------------------------------------------------------

    print("\nTesting on CICIoT2023...")

    start = time.time()

    cross_predictions = model.predict(
        X_cross
    )

    cross_prediction_time = time.time() - start

    cross_accuracy = accuracy_score(
        y_cross,
        cross_predictions
    )

    cross_precision = precision_score(
        y_cross,
        cross_predictions,
        zero_division=0
    )

    cross_recall = recall_score(
        y_cross,
        cross_predictions,
        zero_division=0
    )

    cross_f1 = f1_score(
        y_cross,
        cross_predictions,
        zero_division=0
    )

    print(
        f"Accuracy:  {cross_accuracy:.4f}"
    )

    print(
        f"Precision: {cross_precision:.4f}"
    )

    print(
        f"Recall:    {cross_recall:.4f}"
    )

    print(
        f"F1:        {cross_f1:.4f}"
    )

    print(
        f"Prediction time: "
        f"{cross_prediction_time:.4f} seconds"
    )

    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    results.append({

        "Model": model_name,

        "Model_Size_MB": round(
            model_size_mb,
            2
        ),

        "Training_Time_Sec": round(
            training_time,
            4
        ),

        "InDomain_Prediction_Time_Sec": round(
            prediction_time,
            4
        ),

        "InDomain_Accuracy": round(
            accuracy,
            4
        ),

        "InDomain_Precision": round(
            precision,
            4
        ),

        "InDomain_Recall": round(
            recall,
            4
        ),

        "InDomain_F1": round(
            f1,
            4
        ),

        "CrossDomain_Prediction_Time_Sec": round(
            cross_prediction_time,
            4
        ),

        "CrossDomain_Accuracy": round(
            cross_accuracy,
            4
        ),

        "CrossDomain_Precision": round(
            cross_precision,
            4
        ),

        "CrossDomain_Recall": round(
            cross_recall,
            4
        ),

        "CrossDomain_F1": round(
            cross_f1,
            4
        )
    })


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

output_file = (
    RESULTS_DIR /
    "other_models_comparison.csv"
)

results_df.to_csv(
    output_file,
    index=False
)


# ============================================================
# DISPLAY FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)

print("\nResults saved to:")

print(output_file)

print("\nModel files saved to:")

for model_name in models:

    filename = (
        f"{model_name.lower().replace(' ', '_')}.joblib"
    )

    print(
        MODELS_DIR / filename
    )

print("\nDone.")