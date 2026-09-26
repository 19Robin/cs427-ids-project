import os
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"D:\cs427-ids-project"

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)


FIVE_G_PATH = os.path.join(
    DATA_DIR,
    "5g_nidd_processed.csv"
)

CIC_PATH = os.path.join(
    DATA_DIR,
    "cic_iot2023_processed.csv"
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
# LOAD DATA
# ============================================================

print("=" * 80)
print("FEATURE DISTRIBUTION SHIFT ANALYSIS")
print("=" * 80)

print("\nLoading 5G-NIDD...")
five_g = pd.read_csv(FIVE_G_PATH)

print(f"5G-NIDD rows: {len(five_g):,}")

print("\nLoading CICIoT2023...")
cic = pd.read_csv(CIC_PATH)

print(f"CICIoT2023 rows: {len(cic):,}")


# ============================================================
# CREATE SUMMARY STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("CALCULATING FEATURE STATISTICS")
print("=" * 80)

results = []

for feature in FEATURES:

    five_values = pd.to_numeric(
        five_g[feature],
        errors="coerce"
    )

    cic_values = pd.to_numeric(
        cic[feature],
        errors="coerce"
    )

    # Remove invalid values
    five_values = five_values.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    cic_values = cic_values.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    five_median = five_values.median()
    cic_median = cic_values.median()

    five_mean = five_values.mean()
    cic_mean = cic_values.mean()

    five_std = five_values.std()
    cic_std = cic_values.std()

    five_min = five_values.min()
    cic_min = cic_values.min()

    five_max = five_values.max()
    cic_max = cic_values.max()

    # Median ratio
    if five_median != 0:
        median_ratio = cic_median / five_median
    else:
        median_ratio = np.nan

    # Mean ratio
    if five_mean != 0:
        mean_ratio = cic_mean / five_mean
    else:
        mean_ratio = np.nan

    results.append({
        "Feature": feature,

        "5G_Mean": five_mean,
        "CIC_Mean": cic_mean,

        "5G_Median": five_median,
        "CIC_Median": cic_median,

        "5G_Std": five_std,
        "CIC_Std": cic_std,

        "5G_Min": five_min,
        "CIC_Min": cic_min,

        "5G_Max": five_max,
        "CIC_Max": cic_max,

        "CIC_to_5G_Median_Ratio": median_ratio,
        "CIC_to_5G_Mean_Ratio": mean_ratio
    })


results_df = pd.DataFrame(results)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 80)
print("FEATURE DISTRIBUTION COMPARISON")
print("=" * 80)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:.4f}".format)

print(results_df.to_string(index=False))


# ============================================================
# PROTOCOL PREVALENCE
# ============================================================

print("\n" + "=" * 80)
print("PROTOCOL FEATURE COMPARISON")
print("=" * 80)

protocol_results = []

for feature in ["TCP", "UDP", "ICMP"]:

    five_mean = five_g[feature].mean()
    cic_mean = cic[feature].mean()

    protocol_results.append({
        "Protocol": feature,
        "5G_NIDD_Mean": five_mean,
        "CICIoT2023_Mean": cic_mean,
        "Difference": cic_mean - five_mean
    })

protocol_df = pd.DataFrame(protocol_results)

print(
    protocol_df.to_string(index=False)
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_path = os.path.join(
    RESULTS_DIR,
    "feature_distribution_shift.csv"
)

results_df.to_csv(
    output_path,
    index=False
)

protocol_path = os.path.join(
    RESULTS_DIR,
    "protocol_distribution_shift.csv"
)

protocol_df.to_csv(
    protocol_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

print("\nFeature results saved to:")
print(output_path)

print("\nProtocol results saved to:")
print(protocol_path)