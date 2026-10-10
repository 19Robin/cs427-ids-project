"""
run_improved_experiments.py

Second round of experiments, run after the original baseline scripts.

1. Data quality (proposal section 4.3)
   - removes duplicate records from both datasets
   - measures how many 5G-NIDD test rows also appeared in the training set
     under the ORIGINAL (non-deduplicated) split, i.e. potential leakage

2. Main experiment, repeated with the fixes promised in the proposal
   - deduplicated data
   - class balancing (class_weight="balanced" / scale_pos_weight)
   - Logistic Regression with feature scaling (fitted on 5G-NIDD train only)
   - a depth-limited 10-tree Random Forest as a genuinely small model
   - model size in KB, per-flow inference latency, confusion matrices,
     balanced accuracy, specificity and MCC
   - per-attack-type detection for BOTH 5G-NIDD (in-domain) and CICIoT2023

3. Diagnosis / mitigation experiments (all trained on 5G-NIDD only)
   A1  CICIoT2023 protocol flags rounded to 0/1 (feature alignment)
   A2  A1 + Packet_Count removed (feature with the largest definitional mismatch)
   B   CICIoT2023 restricted to attack types comparable to 5G-NIDD attacks
   C   each dataset standardised with its OWN statistics (unlabelled target data)
   D   1% of labelled CICIoT2023 added to training (this IS retraining;
       reported separately, tested on the remaining 99%)

4. Saves a small set of REAL flows from both test sets for the dashboard's
   Edge IDS Monitor, and the depth-limited model as models/edge_ids_model.joblib.

Requires data/processed/*.csv from preprocess_data.py. The per-attack and
experiment B results need the Attack_Type column added to preprocess_data.py,
so re-run preprocessing first if your processed files predate it.

Run:  python src/run_improved_experiments.py
"""

from pathlib import Path
import io
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    confusion_matrix
)
from xgboost import XGBClassifier


# ============================================================
# PATHS AND SETTINGS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FIVEG_FILE = BASE_DIR / "data" / "processed" / "5g_nidd_processed.csv"
CIC_FILE = BASE_DIR / "data" / "processed" / "cic_iot2023_processed.csv"

OUT_DIR = BASE_DIR / "results" / "improved"
MODELS_DIR = BASE_DIR / "models"

OUT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "Rate",
    "Packet_Count",
    "Mean_Packet_Size",
    "TTL",
    "TCP",
    "UDP",
    "ICMP"
]

PROTOCOLS = ["TCP", "UDP", "ICMP"]
TARGET = "Target"
ATTACK = "Attack_Type"
RANDOM_STATE = 42

EDGE_MODEL_NAME = "Random Forest (10 trees, depth ≤ 12)"

# CICIoT2023 labels whose attack family also exists in 5G-NIDD
# (5G-NIDD: ICMP/UDP/SYN/HTTP flood, slow-rate DoS, SYN/TCP-connect/UDP scan).
COMPARABLE_IN_5G = {
    "DDOS-ICMP_FLOOD",
    "DDOS-UDP_FLOOD",
    "DOS-UDP_FLOOD",
    "DDOS-SYN_FLOOD",
    "DOS-SYN_FLOOD",
    "DDOS-HTTP_FLOOD",
    "DOS-HTTP_FLOOD",
    "DDOS-SLOWLORIS",
    "RECON-PORTSCAN",
}


# ============================================================
# HELPERS
# ============================================================

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def make_models(y_train):
    """Same four algorithms as before, now with class balancing."""
    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())

    return {
        "Random Forest (10 trees)": RandomForestClassifier(
            n_estimators=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        EDGE_MODEL_NAME: RandomForestClassifier(
            n_estimators=10,
            max_depth=12,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "Logistic Regression (scaled)": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE
            )
        ),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=negatives / max(positives, 1),
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
    }


def metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Rows": int(len(y_true)),
        "Malicious_Share": float(np.mean(y_true)),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Predicted_Malicious_Share": float(np.mean(y_pred)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def size_kb(model):
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    return buffer.getbuffer().nbytes / 1024


def latency_ms(model, X, repeats=200):
    """
    Median time to classify ONE flow (as an edge IDS would), single-threaded:
    with n_jobs=-1 the thread start-up cost would dominate a one-row prediction.
    """
    if "n_jobs" in model.get_params():
        model.set_params(n_jobs=1)
    rows = X.sample(n=min(repeats, len(X)), random_state=RANDOM_STATE)
    times = []
    for i in range(len(rows)):
        one = rows.iloc[[i]]
        start = time.perf_counter()
        model.predict(one)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


def binarise_protocols(X):
    """Round CICIoT2023's fractional protocol flags to the 0/1 form used in 5G-NIDD."""
    X = X.copy()
    values = X[PROTOCOLS].to_numpy()
    winner = values.argmax(axis=1)
    keep = values.max(axis=1) >= 0.5
    one_hot = np.zeros_like(values)
    one_hot[np.arange(len(values)), winner] = 1
    one_hot[~keep] = 0
    X[PROTOCOLS] = one_hot
    return X


def row_hash(df):
    return pd.util.hash_pandas_object(df[FEATURES], index=False)


def per_attack(model_name, test_set, attack_labels, y_pred):
    frame = pd.DataFrame({ATTACK: attack_labels.to_numpy(), "Pred": y_pred})
    grouped = frame.groupby(ATTACK)["Pred"].agg(["size", "sum"]).reset_index()
    grouped.columns = [ATTACK, "Samples", "Predicted_Malicious"]
    grouped["Predicted_Malicious_Rate"] = grouped["Predicted_Malicious"] / grouped["Samples"]
    grouped.insert(0, "Test_Set", test_set)
    grouped.insert(0, "Model", model_name)
    return grouped


results = []
per_attack_rows = []
model_info = []


def record(experiment, model_name, test_set, y_true, y_pred):
    row = {"Experiment": experiment, "Model": model_name, "Test_Set": test_set}
    row.update(metrics(y_true, y_pred))
    results.append(row)
    print(f"  {model_name:40s} {test_set:32s} "
          f"F1={row['F1']:.4f} Recall={row['Recall']:.4f} "
          f"BalAcc={row['Balanced_Accuracy']:.4f}")


# ============================================================
# LOAD DATA
# ============================================================

section("LOADING PROCESSED DATA")

fiveg = pd.read_csv(FIVEG_FILE, low_memory=False)
cic = pd.read_csv(CIC_FILE, low_memory=False)

has_5g_attack = ATTACK in fiveg.columns
has_cic_attack = ATTACK in cic.columns

print(f"5G-NIDD rows:    {len(fiveg):,}  (attack types available: {has_5g_attack})")
print(f"CICIoT2023 rows: {len(cic):,}  (attack types available: {has_cic_attack})")

if not (has_5g_attack and has_cic_attack):
    print("\nNOTE: re-run src/preprocess_data.py to add Attack_Type; "
          "per-attack results and experiment B need it.")


# ============================================================
# 1. DATA QUALITY
# ============================================================

section("1. DATA QUALITY: DUPLICATES AND TRAIN/TEST OVERLAP")

quality = {}

# Overlap under the ORIGINAL split (no deduplication), same settings as before
X_tr0, X_te0, y_tr0, y_te0 = train_test_split(
    fiveg[FEATURES], fiveg[TARGET],
    test_size=0.20, stratify=fiveg[TARGET], random_state=RANDOM_STATE
)
train_hashes = set(row_hash(X_tr0))
overlap = row_hash(X_te0).isin(train_hashes)
quality["original_split_test_rows"] = int(len(X_te0))
quality["original_split_test_rows_seen_in_train"] = int(overlap.sum())
quality["original_split_test_overlap_share"] = float(overlap.mean())
print(f"Original split: {overlap.mean() * 100:.2f}% of 5G-NIDD test rows have a "
      f"feature vector that also appears in the training rows.")

# Feature vectors that occur with BOTH labels (cannot be separated by any model)
for name, df in [("5g_nidd", fiveg), ("ciciot2023", cic)]:
    labels_per_vector = df.groupby(row_hash(df))[TARGET].nunique()
    conflicting_vectors = labels_per_vector[labels_per_vector > 1].index
    conflicting_rows = row_hash(df).isin(set(conflicting_vectors)).sum()
    quality[f"{name}_rows_with_conflicting_labels"] = int(conflicting_rows)
    quality[f"{name}_conflicting_label_share"] = float(conflicting_rows / len(df))
    print(f"{name}: {conflicting_rows / len(df) * 100:.2f}% of rows share their exact "
          f"feature vector with a row of the opposite label.")

# Deduplicate on features + label
for name, df in [("5g_nidd", fiveg), ("ciciot2023", cic)]:
    quality[f"{name}_rows_before_dedup"] = int(len(df))

fiveg = fiveg.drop_duplicates(subset=FEATURES + [TARGET]).reset_index(drop=True)
cic = cic.drop_duplicates(subset=FEATURES + [TARGET]).reset_index(drop=True)

for name, df in [("5g_nidd", fiveg), ("ciciot2023", cic)]:
    quality[f"{name}_rows_after_dedup"] = int(len(df))
    quality[f"{name}_malicious_share_after_dedup"] = float(df[TARGET].mean())
    before = quality[f"{name}_rows_before_dedup"]
    print(f"{name}: {before:,} -> {len(df):,} rows after removing duplicates "
          f"({(1 - len(df) / before) * 100:.1f}% removed); "
          f"malicious share {df[TARGET].mean() * 100:.2f}%")


# ============================================================
# 2. MAIN EXPERIMENT (deduplicated + balanced)
# ============================================================

section("2. MAIN EXPERIMENT: DEDUPLICATED DATA + CLASS BALANCING")

train_df, test_df = train_test_split(
    fiveg, test_size=0.20, stratify=fiveg[TARGET], random_state=RANDOM_STATE
)
X_train, y_train = train_df[FEATURES], train_df[TARGET]
X_test, y_test = test_df[FEATURES], test_df[TARGET]
X_cic, y_cic = cic[FEATURES], cic[TARGET]

quality["train_rows"] = int(len(train_df))
quality["test_rows"] = int(len(test_df))

trained = {}

for model_name, model in make_models(y_train).items():
    start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start
    trained[model_name] = model

    record("Main", model_name, "5G-NIDD test (in-domain)", y_test, model.predict(X_test))
    cic_pred = model.predict(X_cic)
    record("Main", model_name, "CICIoT2023 (cross-domain)", y_cic, cic_pred)

    model_info.append({
        "Model": model_name,
        "Model_Size_KB": size_kb(model),
        "Training_Time_Sec": train_time,
        "Latency_ms_per_flow": latency_ms(model, X_test),
    })

    if has_5g_attack:
        per_attack_rows.append(per_attack(
            model_name, "5G-NIDD test (in-domain)",
            test_df[ATTACK], model.predict(X_test)
        ))
    if has_cic_attack:
        per_attack_rows.append(per_attack(
            model_name, "CICIoT2023 (cross-domain)", cic[ATTACK], cic_pred
        ))

joblib.dump(trained[EDGE_MODEL_NAME], MODELS_DIR / "edge_ids_model.joblib")


# ============================================================
# 3. DIAGNOSIS / MITIGATION EXPERIMENTS
# ============================================================

section("3A. FEATURE ALIGNMENT")

X_cic_bin = binarise_protocols(X_cic)

print("A1: CICIoT2023 protocol flags rounded to 0/1 (same models, no retraining)")
for model_name, model in trained.items():
    record("A1: protocol flags rounded to 0/1", model_name,
           "CICIoT2023 (cross-domain)", y_cic, model.predict(X_cic_bin))

print("A2: A1 + Packet_Count removed (models trained on 5G-NIDD without it)")
reduced = [f for f in FEATURES if f != "Packet_Count"]
for model_name, model in make_models(y_train).items():
    model.fit(X_train[reduced], y_train)
    record("A2: + Packet_Count removed", model_name,
           "5G-NIDD test (in-domain)", y_test, model.predict(X_test[reduced]))
    record("A2: + Packet_Count removed", model_name,
           "CICIoT2023 (cross-domain)", y_cic, model.predict(X_cic_bin[reduced]))


section("3B. COMPARABLE ATTACK TYPES ONLY")

if has_cic_attack:
    labels = cic[ATTACK].str.upper()
    mask = (cic[TARGET] == 0) | labels.isin(COMPARABLE_IN_5G)
    print(f"Subset: {mask.sum():,} rows (benign + {len(COMPARABLE_IN_5G)} comparable attack types)")
    for model_name, model in trained.items():
        record("B: comparable attacks only", model_name,
               "CICIoT2023 comparable subset", y_cic[mask], model.predict(X_cic[mask]))
else:
    print("Skipped: CICIoT2023 Attack_Type column not available.")


section("3C. PER-DOMAIN STANDARDISATION")

print("Each dataset is standardised with its own mean/std. The CICIoT2023 scaler")
print("uses unlabelled CICIoT2023 features only; no CICIoT2023 labels are used.")

scale_cols = ["Rate", "Packet_Count", "Mean_Packet_Size", "TTL"]


def standardise(X, scaler):
    X = X.copy()
    X[scale_cols] = scaler.transform(X[scale_cols])
    return X


source_scaler = StandardScaler().fit(X_train[scale_cols])
target_scaler = StandardScaler().fit(X_cic_bin[scale_cols])

X_train_s = standardise(X_train, source_scaler)
X_test_s = standardise(X_test, source_scaler)
X_cic_s = standardise(X_cic_bin, target_scaler)

for model_name, model in make_models(y_train).items():
    model.fit(X_train_s, y_train)
    record("C: per-domain standardisation", model_name,
           "5G-NIDD test (in-domain)", y_test, model.predict(X_test_s))
    record("C: per-domain standardisation", model_name,
           "CICIoT2023 (cross-domain)", y_cic, model.predict(X_cic_s))


section("3D. SMALL LABELLED CICIoT2023 SAMPLE (RETRAINING)")

print("This experiment DOES retrain: 1% of labelled CICIoT2023 is added to the")
print("5G-NIDD training data. Evaluated on the other 99% of CICIoT2023 only.")

cic_small, cic_rest = train_test_split(
    cic, train_size=0.01, stratify=cic[TARGET], random_state=RANDOM_STATE
)
X_mix = pd.concat([X_train, binarise_protocols(cic_small[FEATURES])], ignore_index=True)
y_mix = pd.concat([y_train, cic_small[TARGET]], ignore_index=True)
X_rest = binarise_protocols(cic_rest[FEATURES])

quality["experiment_d_cic_training_rows"] = int(len(cic_small))
quality["experiment_d_cic_test_rows"] = int(len(cic_rest))

for model_name, model in make_models(y_mix).items():
    model.fit(X_mix, y_mix)
    record("D: + 1% labelled CICIoT2023 (retrained)", model_name,
           "5G-NIDD test (in-domain)", y_test, model.predict(X_test))
    record("D: + 1% labelled CICIoT2023 (retrained)", model_name,
           "CICIoT2023 remaining 99%", cic_rest[TARGET], model.predict(X_rest))


# ============================================================
# 4. REAL FLOWS FOR THE EDGE IDS MONITOR
# ============================================================

section("4. REAL FLOW SAMPLES FOR THE DASHBOARD")

stream_parts = []
for dataset, frame in [("5G-NIDD", test_df), ("CICIoT2023", cic)]:
    sample = frame.sample(n=min(300, len(frame)), random_state=RANDOM_STATE).copy()
    sample.insert(0, "Dataset", dataset)
    if ATTACK not in sample.columns:
        sample[ATTACK] = np.where(sample[TARGET] == 1, "Malicious", "Benign")
    stream_parts.append(sample[["Dataset"] + FEATURES + [TARGET, ATTACK]])

stream = pd.concat(stream_parts, ignore_index=True)
stream.to_csv(OUT_DIR / "stream_samples.csv", index=False)
print(f"Saved {len(stream)} real flows (5G-NIDD test split + CICIoT2023).")


# ============================================================
# SAVE
# ============================================================

section("SAVING RESULTS")

pd.DataFrame(results).to_csv(OUT_DIR / "experiment_results.csv", index=False)
pd.DataFrame(model_info).to_csv(OUT_DIR / "model_info.csv", index=False)

if per_attack_rows:
    pd.concat(per_attack_rows, ignore_index=True).to_csv(
        OUT_DIR / "per_attack_results.csv", index=False
    )

quality["edge_model"] = EDGE_MODEL_NAME
with open(OUT_DIR / "data_quality.json", "w") as f:
    json.dump(quality, f, indent=4)

print(f"Results saved to {OUT_DIR}")
print(f"Edge model saved to {MODELS_DIR / 'edge_ids_model.joblib'}")

section("SUMMARY: MAIN EXPERIMENT")
summary = pd.DataFrame(results)
summary = summary[summary["Experiment"] == "Main"][
    ["Model", "Test_Set", "F1", "Recall", "Balanced_Accuracy", "MCC"]
]
print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\n" + pd.DataFrame(model_info).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
