import pandas as pd
import os
import numpy as np

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NIDD_FILE = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "5g_nidd.csv"
)

CIC_DIR = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "cic_iot"
)

CIC_FILES = [
    "Merged01.csv",
    "Merged02.csv",
    "Merged03.csv",
    "Merged04.csv",
    "Merged05.csv"
]


# ============================================================
# LOAD DATASETS
# ============================================================

print("=" * 75)
print("LOADING DATASETS")
print("=" * 75)

print("\nLoading 5G-NIDD...")
nidd = pd.read_csv(
    NIDD_FILE,
    low_memory=False
)

print(f"5G-NIDD rows: {len(nidd):,}")
print(f"5G-NIDD columns: {len(nidd.columns)}")

print("\nLoading CICIoT2023...")

cic_parts = []

for filename in CIC_FILES:

    path = os.path.join(
        CIC_DIR,
        filename
    )

    print(f"  Loading {filename}...")

    df = pd.read_csv(
        path,
        low_memory=False
    )

    cic_parts.append(df)

cic = pd.concat(
    cic_parts,
    ignore_index=True
)

print(f"\nCICIoT2023 rows: {len(cic):,}")
print(f"CICIoT2023 columns: {len(cic.columns)}")


# ============================================================
# STATISTICS FUNCTION
# ============================================================

def stats(df, column):

    if column not in df.columns:
        return None

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    values = values.replace(
        [np.inf, -np.inf],
        np.nan
    ).dropna()

    if len(values) == 0:
        return None

    return {
        "count": len(values),
        "missing": df[column].isna().sum(),
        "min": values.min(),
        "max": values.max(),
        "mean": values.mean(),
        "median": values.median(),
        "std": values.std(),
        "p25": values.quantile(0.25),
        "p75": values.quantile(0.75)
    }


def show_stats(dataset, df, column):

    result = stats(df, column)

    print(f"\n{dataset}: {column}")

    if result is None:
        print("  No usable numeric data.")
        return

    print(f"  Count:   {result['count']:,}")
    print(f"  Missing: {result['missing']:,}")
    print(f"  Min:     {result['min']:.6g}")
    print(f"  Max:     {result['max']:.6g}")
    print(f"  Mean:    {result['mean']:.6g}")
    print(f"  Median:  {result['median']:.6g}")
    print(f"  Std:     {result['std']:.6g}")
    print(f"  P25:     {result['p25']:.6g}")
    print(f"  P75:     {result['p75']:.6g}")


# ============================================================
# 1. PACKET COUNTS
# ============================================================

print("\n" + "=" * 75)
print("1. PACKET COUNT FEATURES")
print("=" * 75)

for col in [
    "TotPkts",
    "SrcPkts",
    "DstPkts"
]:

    show_stats(
        "5G-NIDD",
        nidd,
        col
    )

show_stats(
    "CICIoT2023",
    cic,
    "Number"
)


# ============================================================
# 2. BYTE FEATURES
# ============================================================

print("\n" + "=" * 75)
print("2. BYTE / TRAFFIC VOLUME FEATURES")
print("=" * 75)

for col in [
    "TotBytes",
    "SrcBytes",
    "DstBytes"
]:

    show_stats(
        "5G-NIDD",
        nidd,
        col
    )

for col in [
    "Tot sum",
    "Tot size"
]:

    show_stats(
        "CICIoT2023",
        cic,
        col
    )


# ============================================================
# 3. RATE FEATURES
# ============================================================

print("\n" + "=" * 75)
print("3. RATE FEATURES")
print("=" * 75)

for col in [
    "Rate",
    "SrcRate",
    "DstRate"
]:

    show_stats(
        "5G-NIDD",
        nidd,
        col
    )

show_stats(
    "CICIoT2023",
    cic,
    "Rate"
)


# ============================================================
# 4. PACKET SIZE
# ============================================================

print("\n" + "=" * 75)
print("4. PACKET SIZE FEATURES")
print("=" * 75)

for col in [
    "sMeanPktSz",
    "dMeanPktSz"
]:

    show_stats(
        "5G-NIDD",
        nidd,
        col
    )

for col in [
    "Min",
    "Max",
    "AVG",
    "Std",
    "Tot size"
]:

    show_stats(
        "CICIoT2023",
        cic,
        col
    )


# ============================================================
# 5. TTL / HOPS
# ============================================================

print("\n" + "=" * 75)
print("5. TTL / HOP FEATURES")
print("=" * 75)

for col in [
    "sTtl",
    "dTtl",
    "sHops",
    "dHops"
]:

    show_stats(
        "5G-NIDD",
        nidd,
        col
    )


# ============================================================
# 6. LOSS FEATURES
# ============================================================

print("\n" + "=" * 75)
print("6. PACKET LOSS FEATURES")
print("=" * 75)

for col in [
    "Loss",
    "SrcLoss",
    "DstLoss",
    "pLoss"
]:

    show_stats(
        "5G-NIDD",
        nidd,
        col
    )


# ============================================================
# 7. TCP FLAG / PACKET FEATURES
# ============================================================

print("\n" + "=" * 75)
print("7. TCP FLAG FEATURES")
print("=" * 75)

print("\n5G-NIDD TCP-related columns:")

for col in [
    "Proto",
    "SrcTCPBase",
    "DstTCPBase",
    "TcpRtt",
    "SynAck",
    "AckDat"
]:

    if col in nidd.columns:

        print(f"\n{col}")

        print(
            nidd[col]
            .value_counts(dropna=False)
            .head(15)
        )


print("\nCICIoT2023 TCP-related features:")

for col in [
    "ack_count",
    "syn_count",
    "fin_count",
    "rst_count",
    "TCP"
]:

    if col in cic.columns:

        print(f"\n{col}")

        if pd.api.types.is_numeric_dtype(cic[col]):

            show_stats(
                "CICIoT2023",
                cic,
                col
            )

        else:

            print(
                cic[col]
                .value_counts(dropna=False)
                .head(15)
            )


# ============================================================
# 8. PROTOCOL INDICATORS
# ============================================================

print("\n" + "=" * 75)
print("8. PROTOCOL FEATURES")
print("=" * 75)

print("\n5G-NIDD protocol distribution:")

print(
    nidd["Proto"]
    .value_counts(dropna=False)
)


print("\nCICIoT2023 protocol columns:")

for col in [
    "TCP",
    "UDP",
    "ICMP",
    "IGMP",
    "ARP",
    "IPv",
    "LLC"
]:

    if col in cic.columns:

        print(f"\n{col}")

        print(
            cic[col]
            .value_counts(dropna=False)
            .head(10)
        )


print("\nCICIoT2023 Protocol Type:")

print(
    cic["Protocol Type"]
    .value_counts(dropna=False)
    .sort_index()
)


# ============================================================
# 9. DERIVED FEATURES FOR 5G-NIDD
# ============================================================

print("\n" + "=" * 75)
print("9. DERIVED 5G-NIDD FEATURES")
print("=" * 75)

# Overall mean packet size
nidd["Derived_MeanPktSz"] = np.where(
    nidd["TotPkts"] > 0,
    (
        nidd["SrcPkts"] * nidd["sMeanPktSz"]
        +
        nidd["DstPkts"] * nidd["dMeanPktSz"]
    ) / nidd["TotPkts"],
    np.nan
)

show_stats(
    "5G-NIDD",
    nidd,
    "Derived_MeanPktSz"
)


# Source packet percentage
nidd["Derived_SrcPktRatio"] = np.where(
    nidd["TotPkts"] > 0,
    nidd["SrcPkts"] / nidd["TotPkts"],
    np.nan
)

# Destination packet percentage
nidd["Derived_DstPktRatio"] = np.where(
    nidd["TotPkts"] > 0,
    nidd["DstPkts"] / nidd["TotPkts"],
    np.nan
)

show_stats(
    "5G-NIDD",
    nidd,
    "Derived_SrcPktRatio"
)

show_stats(
    "5G-NIDD",
    nidd,
    "Derived_DstPktRatio"
)


# ============================================================
# 10. RELATIONSHIP CHECKS
# ============================================================

print("\n" + "=" * 75)
print("10. INTERNAL RELATIONSHIPS")
print("=" * 75)

print("\n5G-NIDD:")

nidd_relationships = [
    [
        "TotPkts",
        "TotBytes"
    ],
    [
        "SrcPkts",
        "SrcBytes"
    ],
    [
        "DstPkts",
        "DstBytes"
    ],
    [
        "TotPkts",
        "Rate"
    ],
    [
        "TotBytes",
        "Rate"
    ],
    [
        "SrcPkts",
        "sMeanPktSz"
    ],
    [
        "DstPkts",
        "dMeanPktSz"
    ]
]

for a, b in nidd_relationships:

    if a in nidd.columns and b in nidd.columns:

        temp = nidd[[a, b]].apply(
            pd.to_numeric,
            errors="coerce"
        )

        correlation = temp[a].corr(
            temp[b]
        )

        print(
            f"  {a:<15} <-> {b:<15} "
            f"correlation = {correlation:.4f}"
        )


print("\nCICIoT2023:")

cic_relationships = [
    [
        "Number",
        "Tot sum"
    ],
    [
        "Number",
        "Rate"
    ],
    [
        "Tot sum",
        "AVG"
    ],
    [
        "AVG",
        "Max"
    ],
    [
        "AVG",
        "Min"
    ],
    [
        "AVG",
        "Std"
    ]
]

for a, b in cic_relationships:

    if a in cic.columns and b in cic.columns:

        temp = cic[[a, b]].apply(
            pd.to_numeric,
            errors="coerce"
        )

        correlation = temp[a].corr(
            temp[b]
        )

        print(
            f"  {a:<15} <-> {b:<15} "
            f"correlation = {correlation:.4f}"
        )


# ============================================================
# 11. MISSINGNESS
# ============================================================

print("\n" + "=" * 75)
print("11. MISSINGNESS CHECK")
print("=" * 75)

print("\n5G-NIDD:")

for col in [
    "SrcPkts",
    "DstPkts",
    "SrcBytes",
    "DstBytes",
    "SrcRate",
    "DstRate",
    "sTtl",
    "dTtl",
    "sHops",
    "dHops",
    "Loss",
    "SrcLoss",
    "DstLoss",
    "pLoss"
]:

    if col in nidd.columns:

        missing = nidd[col].isna().sum()

        percentage = (
            missing / len(nidd)
        ) * 100

        print(
            f"  {col:<15} "
            f"{missing:>10,} missing "
            f"({percentage:>6.2f}%)"
        )


print("\nCICIoT2023:")

for col in [
    "ack_count",
    "syn_count",
    "fin_count",
    "rst_count",
    "TCP",
    "UDP",
    "ICMP",
    "IGMP",
    "ARP",
    "IPv",
    "LLC"
]:

    if col in cic.columns:

        missing = cic[col].isna().sum()

        percentage = (
            missing / len(cic)
        ) * 100

        print(
            f"  {col:<15} "
            f"{missing:>10,} missing "
            f"({percentage:>6.2f}%)"
        )


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n" + "=" * 75)
print("EXTENDED MAPPING CHECK COMPLETE")
print("=" * 75)

print("""
Do NOT modify or merge the datasets yet.

The next step is to use these results together with the
published feature definitions to select the final common
feature set.

Potential categories being investigated:

- Rate
- Packet count
- Protocol
- Packet size
- Traffic volume
- TCP flags
- Packet loss
- TTL / hops
- Timing

No model training has been performed.
""")

print("=" * 75)