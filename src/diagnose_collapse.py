"""
diagnose_collapse.py

Investigates WHY cross-domain recall collapsed to ~0. Checks two possible
explanations:
  1. Feature scale mismatch between domains (a preprocessing issue)
  2. Genuine semantic mismatch -- UNSW-NB15's attack types don't produce
     the same traffic-volume signature 5G-NIDD's attacks do (a real
     generalization failure, which is the actual finding of this project)

Run from project root: python src/diagnose_collapse.py
"""

import sys
sys.path.insert(0, "src")

import pandas as pd
import joblib
from data_loader import load_domain_a, load_domain_b

MODEL_PATH = "outputs/models/lightweight_ids_model.pkl"


def compare_feature_distributions():
    X_a, y_a = load_domain_a()
    X_b, y_b = load_domain_b()

    print("=" * 70)
    print("FEATURE DISTRIBUTION COMPARISON (Domain A vs Domain B)")
    print("=" * 70)
    print("\nIf a feature's ranges are wildly different between domains,")
    print("that points to a scale/preprocessing issue rather than a")
    print("genuine attack-signature difference.\n")

    comparison = pd.DataFrame({
        "A_mean": X_a.mean(),
        "A_median": X_a.median(),
        "A_max": X_a.max(),
        "B_mean": X_b.mean(),
        "B_median": X_b.median(),
        "B_max": X_b.max(),
    })
    print(comparison.round(2))
    return X_a, y_a, X_b, y_b


def check_predictions_by_attack_type(unsw_path="data/raw/unsw_nb15.csv"):
    """If UNSW-NB15 has an attack_cat column, break down predictions by
    attack type to see if the model misses ALL attack types equally, or
    only specific ones (which would support the 'different attack
    signature' explanation)."""
    model = joblib.load(MODEL_PATH)
    X_b, y_b = load_domain_b(unsw_path)
    raw_df = pd.read_csv(unsw_path)

    preds = model.predict(X_b)

    if "attack_cat" in raw_df.columns:
        breakdown = pd.DataFrame({
            "attack_cat": raw_df["attack_cat"],
            "true_label": y_b,
            "predicted": preds
        })
        print("\n" + "=" * 70)
        print("PREDICTIONS BROKEN DOWN BY ATTACK CATEGORY")
        print("=" * 70)
        summary = breakdown.groupby("attack_cat").apply(
            lambda g: pd.Series({
                "count": len(g),
                "predicted_as_attack": (g["predicted"] == 1).sum(),
                "detection_rate": (g["predicted"] == 1).mean() if g["true_label"].iloc[0] == 1 else None
            })
        )
        print(summary)


if __name__ == "__main__":
    compare_feature_distributions()
    check_predictions_by_attack_type()
