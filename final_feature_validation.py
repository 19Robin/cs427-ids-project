import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(r"D:\cs427-ids-project")

FIVEG_FILE = BASE / "data" / "raw" / "5g_nidd.csv"
CIC_DIR = BASE / "data" / "raw" / "cic_iot"


def stats(series):
    s = pd.to_numeric(series, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).dropna()

    return {
        "count": len(s),
        "min": s.min(),
        "median": s.median(),
        "mean": s.mean(),
        "max": s.max(),
        "missing": series.isna().sum(),
        "inf": np.isinf(pd.to_numeric(series, errors="coerce")).sum()
    }


print("=" * 80)
print("FINAL FEATURE MAPPING VALIDATION")
print("=" * 80)


# ============================================================
# 1. LOAD 5G-NIDD
# ============================================================

print("\n[1] Loading 5G-NIDD...")
fiveg = pd.read_csv(FIVEG_FILE, low_memory=False)

print(f"Rows: {len(fiveg):,}")
print(f"Columns: {len(fiveg.columns)}")


# ============================================================
# 2. LOAD CICIoT2023
# ============================================================

print("\n[2] Loading CICIoT2023 files...")

cic_files = sorted(CIC_DIR.glob("Merged*.csv"))

cic_list = []

for file in cic_files:
    print(f"Loading {file.name}...")
    df = pd.read_csv(file, low_memory=False)
    cic_list.append(df)

cic = pd.concat(cic_list, ignore_index=True)

print(f"CIC rows: {len(cic):,}")
print(f"CIC columns: {len(cic.columns)}")


# ============================================================
# 3. RATE
# ============================================================

print("\n" + "=" * 80)
print("FEATURE 1: RATE")
print("=" * 80)

print("\n5G-NIDD Rate:")
print(stats(fiveg["Rate"]))

print("\nCICIoT2023 Rate:")
print(stats(cic["Rate"]))


# ============================================================
# 4. PACKET COUNT
# ============================================================

print("\n" + "=" * 80)
print("FEATURE 2: PACKET COUNT")
print("=" * 80)

print("\n5G-NIDD TotPkts:")
print(stats(fiveg["TotPkts"]))

print("\nCICIoT2023 Number:")
print(stats(cic["Number"]))

print("\nCIC Number value distribution:")
print(
    cic["Number"]
    .value_counts()
    .sort_index()
    .head(20)
)


# ============================================================
# 5. MEAN PACKET SIZE
# ============================================================

print("\n" + "=" * 80)
print("FEATURE 3: MEAN PACKET SIZE")
print("=" * 80)

# Direct calculation from total bytes / total packets
fiveg["Calculated_MeanPktSz"] = (
    fiveg["TotBytes"] /
    fiveg["TotPkts"].replace(0, np.nan)
)

# Weighted source + destination packet size
fiveg["Weighted_MeanPktSz"] = (
    (
        fiveg["SrcPkts"] * fiveg["sMeanPktSz"]
        + fiveg["DstPkts"] * fiveg["dMeanPktSz"]
    )
    /
    fiveg["TotPkts"].replace(0, np.nan)
)

print("\n5G calculated mean packet size:")
print(stats(fiveg["Calculated_MeanPktSz"]))

print("\n5G weighted mean packet size:")
print(stats(fiveg["Weighted_MeanPktSz"]))

print("\nCIC AVG:")
print(stats(cic["AVG"]))


# Correlation between the two 5G calculations
comparison = fiveg[
    ["Calculated_MeanPktSz", "Weighted_MeanPktSz"]
].replace([np.inf, -np.inf], np.nan).dropna()

print("\nCorrelation between the two 5G mean-packet-size calculations:")
print(
    comparison["Calculated_MeanPktSz"]
    .corr(comparison["Weighted_MeanPktSz"])
)


# Difference
difference = (
    comparison["Calculated_MeanPktSz"]
    - comparison["Weighted_MeanPktSz"]
).abs()

print("\nAbsolute difference between calculations:")
print(difference.describe())


# ============================================================
# 6. TTL
# ============================================================

print("\n" + "=" * 80)
print("FEATURE 4: TTL")
print("=" * 80)

print("\n5G-NIDD source TTL (sTtl):")
print(stats(fiveg["sTtl"]))

print("\nCICIoT2023 Time_To_Live:")
print(stats(cic["Time_To_Live"]))


# ============================================================
# 7. PROTOCOL
# ============================================================

print("\n" + "=" * 80)
print("FEATURE 5: PROTOCOL")
print("=" * 80)

print("\n5G protocol distribution:")
print(fiveg["Proto"].value_counts(dropna=False))


print("\n5G protocol categories:")
print(sorted(fiveg["Proto"].dropna().astype(str).unique()))


print("\nCIC protocol columns:")

protocol_columns = [
    "TCP",
    "UDP",
    "ICMP",
    "IGMP",
    "ARP",
    "IPv",
    "LLC"
]

for col in protocol_columns:
    if col in cic.columns:
        print(f"\n{col}:")
        print(stats(cic[col]))


# ============================================================
# 8. CREATE COMMON PROTOCOL CLASS
# ============================================================

print("\n" + "=" * 80)
print("COMMON PROTOCOL REPRESENTATION")
print("=" * 80)


def map_5g_protocol(proto):
    if pd.isna(proto):
        return "Other"

    proto = str(proto).lower().strip()

    if proto == "tcp":
        return "TCP"

    if proto == "udp":
        return "UDP"

    if proto in ["icmp", "ipv6-icmp"]:
        return "ICMP"

    return "Other"


fiveg["Common_Protocol"] = fiveg["Proto"].apply(map_5g_protocol)

print("\n5G common protocol distribution:")
print(fiveg["Common_Protocol"].value_counts())


# For CIC, choose the protocol with the highest proportion.
def cic_protocol(row):
    values = {
        "TCP": row.get("TCP", 0),
        "UDP": row.get("UDP", 0),
        "ICMP": row.get("ICMP", 0),
        "Other": max(
            row.get("IGMP", 0),
            row.get("ARP", 0),
            row.get("IPv", 0),
            row.get("LLC", 0)
        )
    }

    return max(values, key=values.get)


cic["Common_Protocol"] = cic.apply(cic_protocol, axis=1)

print("\nCIC common protocol distribution:")
print(cic["Common_Protocol"].value_counts())


# ============================================================
# 9. TCP / UDP / ICMP INDICATORS
# ============================================================

print("\n" + "=" * 80)
print("PROTOCOL INDICATORS")
print("=" * 80)

fiveg["TCP"] = (fiveg["Common_Protocol"] == "TCP").astype(int)
fiveg["UDP"] = (fiveg["Common_Protocol"] == "UDP").astype(int)
fiveg["ICMP"] = (fiveg["Common_Protocol"] == "ICMP").astype(int)

print("\n5G protocol indicator averages:")
print(
    fiveg[["TCP", "UDP", "ICMP"]]
    .mean()
)


print("\nCIC protocol indicator averages:")

for col in ["TCP", "UDP", "ICMP"]:
    print(f"{col}: {cic[col].mean():.6f}")


# ============================================================
# 10. MISSING VALUES IN FINAL FEATURES
# ============================================================

print("\n" + "=" * 80)
print("MISSING VALUES")
print("=" * 80)

fiveg_final = pd.DataFrame({
    "Rate": fiveg["Rate"],
    "Packet_Count": fiveg["TotPkts"],
    "Mean_Packet_Size": fiveg["Calculated_MeanPktSz"],
    "TTL": fiveg["sTtl"],
    "TCP": fiveg["TCP"],
    "UDP": fiveg["UDP"],
    "ICMP": fiveg["ICMP"]
})

cic_final = pd.DataFrame({
    "Rate": cic["Rate"],
    "Packet_Count": cic["Number"],
    "Mean_Packet_Size": cic["AVG"],
    "TTL": cic["Time_To_Live"],
    "TCP": cic["TCP"],
    "UDP": cic["UDP"],
    "ICMP": cic["ICMP"]
})


print("\n5G final feature missing values:")
print(fiveg_final.isna().sum())

print("\nCIC final feature missing values:")
print(cic_final.isna().sum())


# ============================================================
# 11. NON-FINITE VALUES
# ============================================================

print("\n" + "=" * 80)
print("NON-FINITE VALUES")
print("=" * 80)

print("\n5G non-finite values:")
print(
    fiveg_final.apply(
        lambda col: np.isinf(pd.to_numeric(col, errors="coerce")).sum()
    )
)

print("\nCIC non-finite values:")
print(
    cic_final.apply(
        lambda col: np.isinf(pd.to_numeric(col, errors="coerce")).sum()
    )
)


# ============================================================
# 12. LABEL CHECK
# ============================================================

print("\n" + "=" * 80)
print("LABEL CHECK")
print("=" * 80)

print("\n5G labels:")
print(fiveg["Label"].value_counts())

print("\n5G attack types:")
print(fiveg["Attack Type"].value_counts())

print("\nCIC labels:")
print(cic["Label"].value_counts().head(40))


# ============================================================
# 13. FINAL PROPOSED FEATURE SET
# ============================================================

print("\n" + "=" * 80)
print("FINAL PROPOSED COMMON FEATURES")
print("=" * 80)

features = [
    "Rate",
    "Packet_Count",
    "Mean_Packet_Size",
    "TTL",
    "TCP",
    "UDP",
    "ICMP"
]

for i, feature in enumerate(features, 1):
    print(f"{i}. {feature}")


print("\n" + "=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)