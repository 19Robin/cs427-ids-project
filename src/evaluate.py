"""
evaluate.py

Loads the model trained in train.py (no retraining happens here) and
evaluates it two ways:
  1. In-domain  -> held-out split of Domain A (5G-NIDD)
  2. Cross-domain -> Domain B (UNSW-NB15), completely unseen by the model

Prints and saves the comparison metrics.
"""

import json
import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from data_loader import load_domain_b

MODEL_PATH = "outputs/models/lightweight_ids_model.pkl"


def get_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }


def evaluate():
    model = joblib.load(MODEL_PATH)

    # --- In-domain evaluation ---
    X_test_indomain = pd.read_csv("data/processed/domain_a_test_X.csv")
    y_test_indomain = pd.read_csv("data/processed/domain_a_test_y.csv").squeeze()
    pred_indomain = model.predict(X_test_indomain)
    metrics_indomain = get_metrics(y_test_indomain, pred_indomain)
    cm_indomain = confusion_matrix(y_test_indomain, pred_indomain).tolist()

    # --- Cross-domain evaluation (NO retraining) ---
    X_b, y_b = load_domain_b()
    pred_crossdomain = model.predict(X_b)
    metrics_crossdomain = get_metrics(y_b, pred_crossdomain)
    cm_crossdomain = confusion_matrix(y_b, pred_crossdomain).tolist()

    results = {
        "in_domain": metrics_indomain,
        "cross_domain": metrics_crossdomain,
        "performance_drop": {
            k: metrics_indomain[k] - metrics_crossdomain[k]
            for k in metrics_indomain
        },
        "confusion_matrix_in_domain": cm_indomain,
        "confusion_matrix_cross_domain": cm_crossdomain,
    }

    with open("outputs/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("=== IN-DOMAIN RESULTS ===")
    for k, v in metrics_indomain.items():
        print(f"{k}: {v:.4f}")

    print("\n=== CROSS-DOMAIN RESULTS ===")
    for k, v in metrics_crossdomain.items():
        print(f"{k}: {v:.4f}")

    print("\n=== PERFORMANCE DROP ===")
    for k, v in results["performance_drop"].items():
        print(f"{k}: dropped by {v:.4f} ({v*100:.1f} percentage points)")

    print("\nFull results saved to outputs/results.json")
    return results


if __name__ == "__main__":
    evaluate()
