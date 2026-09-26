import os
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"D:\cs427-ids-project"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "random_forest_baseline.joblib"
)

CIC_DIR = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "cic_iot"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

FILES = [
    "Merged01.csv",
    "Merged02.csv",
    "Merged03.csv",
    "Merged04.csv",
    "Merged05.csv"
]

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
# LOAD MODEL
# ============================================================

print("=" * 80)
print("CICIoT2023 ATTACK-TYPE ANALYSIS")
print("=" * 80)

print("\nLoading trained Random Forest...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")
print(f"Model: {MODEL_PATH}")

print("\nIMPORTANT:")
print("The Random Forest was trained ONLY on 5G-NIDD.")
print("No CICIoT2023 data is used for training.")
print("No retraining is performed.")
print("No model parameters are changed.")


# ============================================================
# LOAD AND PREPARE CICIoT2023
# ============================================================

print("\n" + "=" * 80)
print("LOADING ORIGINAL CICIoT2023 DATA")
print("=" * 80)

all_data = []

for filename in FILES:

    filepath = os.path.join(CIC_DIR, filename)

    print(f"\nLoading: {filename}")

    df = pd.read_csv(filepath, low_memory=False)

    print(f"Original rows: {len(df):,}")

    # --------------------------------------------------------
    # Create common features
    # --------------------------------------------------------

    # Rate
    df["Rate"] = pd.to_numeric(
        df["Rate"],
        errors="coerce"
    )

    # Replace infinite values
    df["Rate"] = df["Rate"].replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Packet count
    df["Packet_Count"] = pd.to_numeric(
        df["Number"],
        errors="coerce"
    )

    # Mean packet size
    df["Mean_Packet_Size"] = pd.to_numeric(
        df["AVG"],
        errors="coerce"
    )

    # TTL
    df["TTL"] = pd.to_numeric(
        df["Time_To_Live"],
        errors="coerce"
    )

    # Protocol indicators
    df["TCP"] = pd.to_numeric(
        df["TCP"],
        errors="coerce"
    )

    df["UDP"] = pd.to_numeric(
        df["UDP"],
        errors="coerce"
    )

    df["ICMP"] = pd.to_numeric(
        df["ICMP"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Keep only required columns
    # --------------------------------------------------------

    required_columns = FEATURES + ["Label"]

    df = df[required_columns]

    # --------------------------------------------------------
    # Remove invalid rows using same logic as preprocessing
    # --------------------------------------------------------

    before = len(df)

    df = df.dropna(
        subset=FEATURES + ["Label"]
    )

    after = len(df)

    removed = before - after

    print(f"Rows after cleaning: {after:,}")
    print(f"Rows removed: {removed:,}")

    all_data.append(df)


# ============================================================
# COMBINE FILES
# ============================================================

print("\n" + "=" * 80)
print("COMBINING DATA")
print("=" * 80)

data = pd.concat(
    all_data,
    ignore_index=True
)

print(f"\nTotal rows: {len(data):,}")

print("\nAttack labels found:")

print(
    data["Label"]
    .value_counts()
    .to_string()
)


# ============================================================
# CREATE BINARY TARGET
# ============================================================

# CICIoT2023 uses BENIGN for normal traffic.
# Every other label represents an attack.

data["Target"] = (
    data["Label"]
    .str.upper()
    .ne("BENIGN")
    .astype(int)
)


# ============================================================
# PREDICTION
# ============================================================

print("\n" + "=" * 80)
print("CROSS-DOMAIN PREDICTION BY ATTACK TYPE")
print("=" * 80)

X = data[FEATURES]

print(f"\nFeatures used: {FEATURES}")
print(f"Rows to predict: {len(X):,}")

print("\nMaking predictions...")

start_time = time.time()

predictions = model.predict(X)

prediction_time = time.time() - start_time

print(
    f"Prediction completed in {prediction_time:.2f} seconds"
)

data["Prediction"] = predictions


# ============================================================
# OVERALL CHECK
# ============================================================

overall_accuracy = accuracy_score(
    data["Target"],
    data["Prediction"]
)

print("\n" + "=" * 80)
print("OVERALL RESULT CHECK")
print("=" * 80)

print(
    f"\nAccuracy: {overall_accuracy:.4f}"
)


# ============================================================
# ATTACK-TYPE DETECTION RATE
# ============================================================

print("\n" + "=" * 80)
print("PERFORMANCE BY ATTACK TYPE")
print("=" * 80)

attack_rows = []

attack_labels = sorted(
    data.loc[
        data["Label"].str.upper() != "BENIGN",
        "Label"
    ].unique()
)

for attack_type in attack_labels:

    attack_data = data[
        data["Label"] == attack_type
    ]

    total = len(attack_data)

    detected = (
        attack_data["Prediction"] == 1
    ).sum()

    missed = (
        attack_data["Prediction"] == 0
    ).sum()

    detection_rate = detected / total

    attack_rows.append({
        "Attack_Type": attack_type,
        "Samples": total,
        "Detected_Malicious": detected,
        "Missed_As_Benign": missed,
        "Detection_Rate": detection_rate
    })


attack_results = pd.DataFrame(
    attack_rows
)

attack_results = attack_results.sort_values(
    "Detection_Rate",
    ascending=True
).reset_index(drop=True)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print(
    "\n"
    + attack_results.to_string(
        index=False,
        formatters={
            "Detection_Rate": "{:.4f}".format
        }
    )
)


# ============================================================
# BENIGN ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("BENIGN TRAFFIC CHECK")
print("=" * 80)

benign = data[
    data["Label"].str.upper() == "BENIGN"
]

benign_total = len(benign)

benign_correct = (
    benign["Prediction"] == 0
).sum()

benign_false_positive = (
    benign["Prediction"] == 1
).sum()

benign_accuracy = (
    benign_correct / benign_total
)

print(f"\nBenign samples: {benign_total:,}")
print(f"Correctly classified as benign: {benign_correct:,}")
print(f"Incorrectly classified as malicious: {benign_false_positive:,}")
print(f"Benign detection accuracy: {benign_accuracy:.4f}")


# ============================================================
# SAVE RESULTS
# ============================================================

output_path = os.path.join(
    RESULTS_DIR,
    "attack_type_results.csv"
)

attack_results.to_csv(
    output_path,
    index=False
)

print("\n" + "=" * 80)
print("ATTACK-TYPE ANALYSIS COMPLETE")
print("=" * 80)

print(f"\nResults saved to:")
print(output_path)