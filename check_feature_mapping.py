import pandas as pd
import os
import numpy as np

# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"D:\cs427-ids-project"
NIDD_FILE = os.path.join(BASE_DIR, "data", "raw", "5g_nidd.csv")
CIC_DIR = os.path.join(BASE_DIR, "data", "raw", "cic_iot")

CIC_FILES = [
    "Merged01.csv",
    "Merged02.csv",
    "Merged03.csv",
    "Merged04.csv",
    "Merged05.csv"
]

# ============================================================
# LOAD 5G-NIDD
# ============================================================

print("=" * 70)
print("LOADING 5G-NIDD")
print("=" * 70)

nidd = pd.read_csv(NIDD_FILE, low_memory=False)

print(f"Rows: {len(nidd):,}")
print(f"Columns: {len(nidd.columns)}")

# ============================================================
# LOAD CICIoT2023 SAMPLE
# ============================================================

print("\n" + "=" * 70)
print("LOADING CICIoT2023")
print("=" * 70)

cic_parts = []

for filename in CIC_FILES:
    path = os.path.join(CIC_DIR, filename)

    print(f"Loading {filename}...")

    df = pd.read_csv(path, low_memory=False)

    print(f"  Rows: {len(df):,}")

    cic_parts.append(df)

cic = pd.concat(cic_parts, ignore_index=True)

print(f"\nCombined CICIoT2023 rows: {len(cic):,}")
print(f"Columns: {len(cic.columns)}")

# ============================================================
# HELPER FUNCTION
# ============================================================

def numeric_stats(df, column):
    """Return useful statistics for a numeric column."""

    if column not in df.columns:
        return None

    series = pd.to_numeric(df[column], errors="coerce").dropna()

    if len(series) == 0:
        return None

    return {
        "count": len(series),
        "missing": df[column].isna().sum(),
        "min": series.min(),
        "max": series.max(),
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "p25": series.quantile(0.25),
        "p75": series.quantile(0.75),
    }


def print_stats(dataset_name, df, column):
    stats = numeric_stats(df, column)

    print(f"\n{dataset_name}: {column}")

    if stats is None:
        print("  No usable numeric data")
        return

    print(f"  Count:    {stats['count']:,}")
    print(f"  Missing:  {stats['missing']:,}")
    print(f"  Min:      {stats['min']:.6g}")
    print(f"  Max:      {stats['max']:.6g}")
    print(f"  Mean:     {stats['mean']:.6g}")
    print(f"  Median:   {stats['median']:.6g}")
    print(f"  Std:      {stats['std']:.6g}")
    print(f"  P25:      {stats['p25']:.6g}")
    print(f"  P75:      {stats['p75']:.6g}")


# ============================================================
# 1. RATE
# ============================================================

print("\n" + "=" * 70)
print("MAPPING 1: Rate  <->  Rate")
print("=" * 70)

print_stats("5G-NIDD", nidd, "Rate")
print_stats("CICIoT2023", cic, "Rate")


# ============================================================
# 2. PACKET COUNT
# ============================================================

print("\n" + "=" * 70)
print("MAPPING 2: TotPkts  <->  Number")
print("=" * 70)

print_stats("5G-NIDD", nidd, "TotPkts")
print_stats("CICIoT2023", cic, "Number")


# ============================================================
# 3. PROTOCOL
# ============================================================

print("\n" + "=" * 70)
print("MAPPING 3: Proto  <->  Protocol Type")
print("=" * 70)

print("\n5G-NIDD Proto values:")
print(nidd["Proto"].value_counts(dropna=False))

print("\nCICIoT2023 Protocol Type values:")

protocol_counts = cic["Protocol Type"].value_counts(dropna=False).sort_index()

print(protocol_counts)


# ============================================================
# 4. TOTAL BYTES / TOTAL SUM
# ============================================================

print("\n" + "=" * 70)
print("MAPPING 4: TotBytes  <->  Tot sum")
print("=" * 70)

print_stats("5G-NIDD", nidd, "TotBytes")
print_stats("CICIoT2023", cic, "Tot sum")


# ============================================================
# 5. PACKET SIZE
# ============================================================

print("\n" + "=" * 70)
print("MAPPING 5: Packet Size Features")
print("=" * 70)

print("\n5G-NIDD packet-size features:")

for column in [
    "sMeanPktSz",
    "dMeanPktSz",
    "sMeanPktSz",
    "dMeanPktSz"
]:
    if column in nidd.columns:
        print_stats("5G-NIDD", nidd, column)

print("\nCICIoT2023 packet-size features:")

for column in [
    "Min",
    "Max",
    "AVG",
    "Std",
    "Tot size"
]:
    if column in cic.columns:
        print_stats("CICIoT2023", cic, column)


# ============================================================
# 6. DURATION
# ============================================================

print("\n" + "=" * 70)
print("MAPPING 6: Duration")
print("=" * 70)

for column in ["Dur", "RunTime", "Mean", "Sum", "Min", "Max"]:
    if column in nidd.columns:
        print_stats("5G-NIDD", nidd, column)


# ============================================================
# 7. TIME / GAP FEATURES
# ============================================================

print("\n" + "=" * 70)
print("MAPPING 7: Timing Features")
print("=" * 70)

for column in [
    "SrcGap",
    "DstGap"
]:
    if column in nidd.columns:
        print_stats("5G-NIDD", nidd, column)

if "IAT" in cic.columns:
    print_stats("CICIoT2023", cic, "IAT")


# ============================================================
# 8. UNIQUE VALUES FOR IMPORTANT FEATURES
# ============================================================

print("\n" + "=" * 70)
print("UNIQUE VALUE CHECK")
print("=" * 70)

important_columns = [
    "Rate",
    "TotPkts",
    "TotBytes",
    "sMeanPktSz",
    "dMeanPktSz",
    "SrcGap",
    "DstGap"
]

print("\n5G-NIDD:")

for column in important_columns:
    if column in nidd.columns:
        series = pd.to_numeric(nidd[column], errors="coerce")

        print(
            f"  {column:<15} "
            f"unique={series.nunique(dropna=True):,}"
        )

important_cic_columns = [
    "Rate",
    "Number",
    "Tot sum",
    "Min",
    "Max",
    "AVG",
    "Std",
    "Tot size",
    "IAT"
]

print("\nCICIoT2023:")

for column in important_cic_columns:
    if column in cic.columns:
        series = pd.to_numeric(cic[column], errors="coerce")

        print(
            f"  {column:<15} "
            f"unique={series.nunique(dropna=True):,}"
        )


# ============================================================
# 9. CORRELATION CHECK
# ============================================================

print("\n" + "=" * 70)
print("CORRELATION CHECK")
print("=" * 70)

nidd_corr_columns = [
    "Rate",
    "TotPkts",
    "TotBytes",
    "sMeanPktSz",
    "dMeanPktSz",
    "SrcGap",
    "DstGap"
]

nidd_corr_columns = [
    c for c in nidd_corr_columns
    if c in nidd.columns
]

print("\n5G-NIDD correlation matrix:")

print(
    nidd[nidd_corr_columns]
    .apply(pd.to_numeric, errors="coerce")
    .corr()
    .round(3)
)

cic_corr_columns = [
    "Rate",
    "Number",
    "Tot sum",
    "Min",
    "Max",
    "AVG",
    "Std",
    "Tot size",
    "IAT"
]

cic_corr_columns = [
    c for c in cic_corr_columns
    if c in cic.columns
]

print("\nCICIoT2023 correlation matrix:")

print(
    cic[cic_corr_columns]
    .apply(pd.to_numeric, errors="coerce")
    .corr()
    .round(3)
)


# ============================================================
# 10. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("INITIAL MAPPING SUMMARY")
print("=" * 70)

print("""
Candidate mappings to investigate:

1. Rate       <-> Rate
2. TotPkts    <-> Number
3. Proto      <-> Protocol Type
4. TotBytes   <-> Tot sum
5. Packet-size features
6. Duration features
7. SrcGap/DstGap <-> IAT

IMPORTANT:
These are NOT confirmed mappings yet.

The statistics above are being used to help us determine
whether the features are actually comparable.

Next we will verify the feature definitions and units before
creating the final common feature set.
""")

print("=" * 70)
print("CHECK COMPLETE")
print("=" * 70)