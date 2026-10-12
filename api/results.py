from pathlib import Path
import json
import math
from datetime import datetime

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
IMPROVED_DIR = RESULTS_DIR / "improved"

FEATURES = ["Rate", "Packet_Count", "Mean_Packet_Size", "TTL", "TCP", "UDP", "ICMP"]
CONTINUOUS_FEATURES = ["Rate", "Packet_Count", "Mean_Packet_Size", "TTL"]

# Khan et al. (2025), DL-IID, IEEE Access 13: quantised model size
DL_IID_SIZE_KB = 108.42

# CICIoT2023 attacks with a comparable attack type in 5G-NIDD
COMPARABLE_IN_5G = {
    "DDOS-ICMP_FLOOD": "ICMP Flood",
    "DDOS-UDP_FLOOD": "UDP Flood",
    "DOS-UDP_FLOOD": "UDP Flood",
    "DDOS-SYN_FLOOD": "SYN Flood",
    "DOS-SYN_FLOOD": "SYN Flood",
    "DDOS-HTTP_FLOOD": "HTTP Flood",
    "DOS-HTTP_FLOOD": "HTTP Flood",
    "DDOS-SLOWLORIS": "Slowrate DoS",
    "RECON-PORTSCAN": "SYN / TCP Connect / UDP Scan",
}


# ============================================================
# JSON HELPERS
# ============================================================

def clean(value):
    """Make numpy / pandas / datetime values JSON-safe (NaN and inf -> null)."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    return value


def read_csv(path):
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    return frame.loc[:, ~frame.columns.duplicated()]


def read_json(path):
    if not path.exists():
        return None
    with open(path) as file:
        return json.load(file)


def records(frame):
    return None if frame is None else clean(frame.to_dict(orient="records"))


# ============================================================
# DERIVED VALUES
# ============================================================

def confusion_summary(cm):
    """[[TN, FP], [FN, TP]] with 0 = benign, 1 = malicious (sklearn order)."""
    if cm is None:
        return None
    (tn, fp), (fn, tp) = cm
    total = tn + fp + fn + tp
    benign, malicious = tn + fp, fn + tp
    prior = malicious / total
    specificity = tn / benign if benign else None
    recall = tp / malicious if malicious else None
    return {
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "total": total, "benign": benign, "malicious": malicious,
        "malicious_share": prior,
        "predicted_malicious_share": (fp + tp) / total,
        "accuracy": (tn + tp) / total,
        "specificity": specificity,
        "recall": recall,
        "balanced_accuracy": (specificity + recall) / 2,
        # trivial classifier that labels every record malicious
        "trivial_accuracy": prior,
        "trivial_precision": prior,
        "trivial_f1": 2 * prior / (1 + prior),
    }


def metric_block(row, prefix=""):
    return {
        "accuracy": row[f"{prefix}Accuracy"],
        "precision": row[f"{prefix}Precision"],
        "recall": row[f"{prefix}Recall"],
        "f1": row[f"{prefix}F1"],
    }


def cic_category(label):
    label = label.upper()
    if label.startswith("DDOS"):
        return "DDoS"
    if label.startswith("DOS"):
        return "DoS"
    if label.startswith("MIRAI"):
        return "Mirai"
    if label.startswith("RECON") or label == "VULNERABILITYSCAN":
        return "Reconnaissance"
    if label in {"DNS_SPOOFING", "MITM-ARPSPOOFING"}:
        return "Spoofing"
    if label == "DICTIONARYBRUTEFORCE":
        return "Brute force"
    return "Web-based"


# ============================================================
# PAYLOAD
# ============================================================

def build_payload():
    baseline = read_json(RESULTS_DIR / "baseline_5g_results.json")
    cross = read_json(RESULTS_DIR / "cross_domain_results.json")
    lightweight = read_csv(RESULTS_DIR / "lightweight_model_comparison.csv")
    cross_rf = read_csv(RESULTS_DIR / "lightweight_cross_domain_comparison.csv")
    other = read_csv(RESULTS_DIR / "other_models_comparison.csv")
    importance = read_csv(RESULTS_DIR / "baseline_feature_importance.csv")
    attacks = read_csv(RESULTS_DIR / "attack_type_results.csv")
    feature_shift = read_csv(RESULTS_DIR / "feature_distribution_shift.csv")
    protocol_shift = read_csv(RESULTS_DIR / "protocol_distribution_shift.csv")
    demo_samples = read_csv(RESULTS_DIR / "demo_samples_5g_nidd.csv")

    cm_5g = confusion_summary(baseline["confusion_matrix"]) if baseline else None
    cm_cic = confusion_summary(cross["confusion_matrix"]) if cross else None

    # ---------------- datasets ----------------
    datasets = {
        "fiveg": None if not baseline else {
            "rows": baseline["training_rows"] + baseline["testing_rows"],
            "train_rows": baseline["training_rows"],
            "test_rows": baseline["testing_rows"],
            "malicious_share": cm_5g["malicious_share"],
        },
        "cic": None if not cross else {
            "rows": cross["testing_rows"],
            "malicious_share": cm_cic["malicious_share"],
            "benign_rows": cm_cic["benign"],
            "attack_types": None if attacks is None else int(len(attacks)),
        },
    }

    # ---------------- Random Forest by number of trees ----------------
    rf_trees = None
    if lightweight is not None:
        lw = lightweight.sort_values("Trees")
        cross_by_trees = {} if cross_rf is None else {
            int(r["Trees"]): r for _, r in cross_rf.iterrows()
        }
        rf_trees = []
        for _, r in lw.iterrows():
            trees = int(r["Trees"])
            c = cross_by_trees.get(trees)
            rf_trees.append({
                "trees": trees,
                "model": r["Model"],
                "in_domain": metric_block(r),
                "cross_domain": None if c is None else metric_block(c),
                "size_mb": r["Model_Size_MB"],
                "train_seconds": r["Training_Time_Seconds"],
                "predict_seconds_in": r["Prediction_Time_Seconds"],
                "predict_seconds_cross": None if c is None else c["Prediction_Time_Seconds"],
            })

    # ---------------- four algorithms ----------------
    algorithms = []
    if lightweight is not None and cross_rf is not None:
        rf_in = lightweight[lightweight["Trees"] == 10]
        rf_out = cross_rf[cross_rf["Trees"] == 10]
        if not rf_in.empty and not rf_out.empty:
            rf_in, rf_out = rf_in.iloc[0], rf_out.iloc[0]
            algorithms.append({
                "model": "Random Forest (10 trees)",
                "size_mb": rf_in["Model_Size_MB"],
                "train_seconds": rf_in["Training_Time_Seconds"],
                "in_domain": metric_block(rf_in),
                "cross_domain": metric_block(rf_out),
            })
    if other is not None:
        for _, r in other.iterrows():
            algorithms.append({
                "model": r["Model"],
                # other_models_comparison.csv rounds sizes to 2 decimals
                "size_mb": r["Model_Size_MB"],
                "train_seconds": r["Training_Time_Sec"],
                "in_domain": metric_block(r, "InDomain_"),
                "cross_domain": metric_block(r, "CrossDomain_"),
            })
    for a in algorithms:
        a["f1_drop_pp"] = (a["in_domain"]["f1"] - a["cross_domain"]["f1"]) * 100

    # ---------------- attacks ----------------
    attack_rows = None
    if attacks is not None:
        attack_rows = []
        for _, r in attacks.iterrows():
            attack_rows.append({
                "attack": r["Attack_Type"],
                "samples": r["Samples"],
                "detected": r["Detected_Malicious"],
                "missed": r["Missed_As_Benign"],
                "rate": r["Detection_Rate"],
                "category": cic_category(r["Attack_Type"]),
                "comparable": COMPARABLE_IN_5G.get(r["Attack_Type"].upper()),
            })

    # ---------------- feature shift ----------------
    shift_rows = None
    if feature_shift is not None:
        shift_rows = []
        fs = feature_shift.set_index("Feature")
        for feature in FEATURES:
            if feature not in fs.index:
                continue
            r = fs.loc[feature]
            pooled = math.sqrt((r["5G_Std"] ** 2 + r["CIC_Std"] ** 2) / 2)
            shift_rows.append({
                "feature": feature,
                "fiveg": {k: r[f"5G_{k.capitalize()}"] for k in ["mean", "median", "std", "min", "max"]},
                "cic": {k: r[f"CIC_{k.capitalize()}"] for k in ["mean", "median", "std", "min", "max"]},
                "smd": (r["CIC_Mean"] - r["5G_Mean"]) / pooled if pooled else None,
                "mean_ratio": r.get("CIC_to_5G_Mean_Ratio"),
            })

    protocol_rows = None
    if protocol_shift is not None:
        protocol_rows = [
            {"protocol": r["Protocol"], "fiveg": r["5G_NIDD_Mean"], "cic": r["CICIoT2023_Mean"]}
            for _, r in protocol_shift.iterrows()
        ]

    # ---------------- improved experiments ----------------
    improved_results = read_csv(IMPROVED_DIR / "experiment_results.csv")
    improved_info = read_csv(IMPROVED_DIR / "model_info.csv")
    improved = None
    if improved_results is not None and improved_info is not None:
        improved = {
            "quality": read_json(IMPROVED_DIR / "data_quality.json"),
            "results": records(improved_results),
            "model_info": records(improved_info),
            "per_attack": records(read_csv(IMPROVED_DIR / "per_attack_results.csv")),
        }

    return clean({
        "generated_at": datetime.now(),
        "features": FEATURES,
        "baseline": baseline,
        "cross": cross,
        "confusion": {"fiveg": cm_5g, "cic": cm_cic},
        "datasets": datasets,
        "rf_trees": rf_trees,
        "algorithms": algorithms or None,
        "feature_importance": records(importance),
        "attacks": attack_rows,
        "feature_shift": shift_rows,
        "protocol_shift": protocol_rows,
        "improved": improved,
        "demo_samples": records(demo_samples),
        "paper": {"dl_iid_size_kb": DL_IID_SIZE_KB, "citation": "Khan et al., IEEE Access 13 (2025)"},
    })


def feature_ranges():
    """5G-NIDD min/max/median per continuous feature (for the prediction demo)."""
    fs = read_csv(RESULTS_DIR / "feature_distribution_shift.csv")
    if fs is None:
        return {}
    fs = fs.set_index("Feature")
    return {
        f: {"min": float(fs.loc[f, "5G_Min"]), "max": float(fs.loc[f, "5G_Max"]),
            "median": float(fs.loc[f, "5G_Median"])}
        for f in CONTINUOUS_FEATURES if f in fs.index
    }
