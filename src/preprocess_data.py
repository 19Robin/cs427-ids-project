import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

BASE = Path(r"D:\cs427-ids-project")

FIVEG_FILE = BASE / "data" / "raw" / "5g_nidd.csv"
CIC_DIR = BASE / "data" / "raw" / "cic_iot"

PROCESSED_DIR = BASE / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42


# ============================================================
# 5G-NIDD PREPROCESSING
# ============================================================

print("=" * 80)
print("5G-NIDD PREPROCESSING")
print("=" * 80)

print("\nLoading 5G-NIDD...")
fiveg = pd.read_csv(FIVEG_FILE, low_memory=False)

print(f"Original rows: {len(fiveg):,}")


# ------------------------------------------------------------
# Create common features
# ------------------------------------------------------------

fiveg_processed = pd.DataFrame()

# 1. Rate
fiveg_processed["Rate"] = pd.to_numeric(
    fiveg["Rate"],
    errors="coerce"
)

# 2. Packet count
fiveg_processed["Packet_Count"] = pd.to_numeric(
    fiveg["TotPkts"],
    errors="coerce"
)

# 3. Mean packet size
fiveg_processed["Mean_Packet_Size"] = (
    pd.to_numeric(fiveg["TotBytes"], errors="coerce")
    /
    pd.to_numeric(
        fiveg["TotPkts"],
        errors="coerce"
    ).replace(0, np.nan)
)

# 4. TTL
fiveg_processed["TTL"] = pd.to_numeric(
    fiveg["sTtl"],
    errors="coerce"
)


# ------------------------------------------------------------
# Protocol indicators
# ------------------------------------------------------------

proto = fiveg["Proto"].astype(str).str.lower().str.strip()

fiveg_processed["TCP"] = (
    proto == "tcp"
).astype(int)

fiveg_processed["UDP"] = (
    proto == "udp"
).astype(int)

fiveg_processed["ICMP"] = (
    proto.isin(["icmp", "ipv6-icmp"])
).astype(int)


# ------------------------------------------------------------
# Binary label
# ------------------------------------------------------------

fiveg_processed["Target"] = (
    fiveg["Label"]
    .astype(str)
    .str.lower()
    .eq("malicious")
    .astype(int)
)


# ------------------------------------------------------------
# Remove invalid values
# ------------------------------------------------------------

fiveg_processed = fiveg_processed.replace(
    [np.inf, -np.inf],
    np.nan
)

before = len(fiveg_processed)

fiveg_processed = fiveg_processed.dropna(
    subset=[
        "Rate",
        "Packet_Count",
        "Mean_Packet_Size",
        "TTL",
        "TCP",
        "UDP",
        "ICMP",
        "Target"
    ]
)

removed = before - len(fiveg_processed)

print(f"\nRows removed because of missing/invalid values: {removed:,}")
print(f"Rows remaining: {len(fiveg_processed):,}")


# ------------------------------------------------------------
# Save 5G processed dataset
# ------------------------------------------------------------

fiveg_output = PROCESSED_DIR / "5g_nidd_processed.csv"

fiveg_processed.to_csv(
    fiveg_output,
    index=False
)

print(f"\nSaved:")
print(fiveg_output)


# ============================================================
# CICIoT2023 PREPROCESSING
# ============================================================

print("\n" + "=" * 80)
print("CICIoT2023 PREPROCESSING")
print("=" * 80)


cic_files = sorted(CIC_DIR.glob("Merged*.csv"))

print(f"\nCIC files found: {len(cic_files)}")

cic_processed_parts = []


for file in cic_files:

    print(f"\nProcessing {file.name}...")

    df = pd.read_csv(
        file,
        low_memory=False
    )

    processed = pd.DataFrame()

    # --------------------------------------------------------
    # Common features
    # --------------------------------------------------------

    processed["Rate"] = pd.to_numeric(
        df["Rate"],
        errors="coerce"
    )

    processed["Packet_Count"] = pd.to_numeric(
        df["Number"],
        errors="coerce"
    )

    processed["Mean_Packet_Size"] = pd.to_numeric(
        df["AVG"],
        errors="coerce"
    )

    processed["TTL"] = pd.to_numeric(
        df["Time_To_Live"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Protocol indicators
    # --------------------------------------------------------

    processed["TCP"] = pd.to_numeric(
        df["TCP"],
        errors="coerce"
    )

    processed["UDP"] = pd.to_numeric(
        df["UDP"],
        errors="coerce"
    )

    processed["ICMP"] = pd.to_numeric(
        df["ICMP"],
        errors="coerce"
    )

    # --------------------------------------------------------
    # Binary label
    #
    # BENIGN = 0
    # Everything else = 1
    # --------------------------------------------------------

    processed["Target"] = (
        df["Label"]
        .astype(str)
        .str.upper()
        .ne("BENIGN")
        .astype(int)
    )

    processed = processed.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Remove invalid rows
    before = len(processed)

    processed = processed.dropna(
        subset=[
            "Rate",
            "Packet_Count",
            "Mean_Packet_Size",
            "TTL",
            "TCP",
            "UDP",
            "ICMP",
            "Target"
        ]
    )

    removed = before - len(processed)

    print(f"Rows: {before:,}")
    print(f"Invalid rows removed: {removed:,}")
    print(f"Rows remaining: {len(processed):,}")

    cic_processed_parts.append(processed)


# ------------------------------------------------------------
# Combine CIC files
# ------------------------------------------------------------

print("\nCombining CICIoT2023 files...")

cic_processed = pd.concat(
    cic_processed_parts,
    ignore_index=True
)

print(
    f"Total CIC processed rows: "
    f"{len(cic_processed):,}"
)


# ------------------------------------------------------------
# Save CIC processed dataset
# ------------------------------------------------------------

cic_output = PROCESSED_DIR / "cic_iot2023_processed.csv"

cic_processed.to_csv(
    cic_output,
    index=False
)

print(f"\nSaved:")
print(cic_output)


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 80)
print("FINAL DATA CHECK")
print("=" * 80)

feature_columns = [
    "Rate",
    "Packet_Count",
    "Mean_Packet_Size",
    "TTL",
    "TCP",
    "UDP",
    "ICMP"
]


print("\nFeatures:")
print(feature_columns)


print("\n5G shape:")
print(fiveg_processed.shape)

print("\nCIC shape:")
print(cic_processed.shape)


print("\n5G target distribution:")
print(
    fiveg_processed["Target"]
    .value_counts()
    .sort_index()
)

print("\nCIC target distribution:")
print(
    cic_processed["Target"]
    .value_counts()
    .sort_index()
)


print("\n5G feature summary:")
print(
    fiveg_processed[feature_columns]
    .describe()
    .round(4)
)


print("\nCIC feature summary:")
print(
    cic_processed[feature_columns]
    .describe()
    .round(4)
)


print("\n" + "=" * 80)
print("PREPROCESSING COMPLETE")
print("=" * 80)