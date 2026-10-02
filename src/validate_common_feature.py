import os
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FIVE_G_PATH = os.path.join(
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


# ============================================================
# LOAD 5G-NIDD
# ============================================================

print("=" * 80)
print("COMMON FEATURE MAPPING VALIDATION")
print("=" * 80)

print("\nLoading 5G-NIDD...")

five_g = pd.read_csv(
    FIVE_G_PATH,
    low_memory=False
)

print(f"Rows: {len(five_g):,}")


# ============================================================
# LOAD CICIoT2023
# ============================================================

print("\nLoading CICIoT2023 files...")

cic_files = [
    "Merged01.csv",
    "Merged02.csv",
    "Merged03.csv",
    "Merged04.csv",
    "Merged05.csv"
]

cic_parts = []

for filename in cic_files:

    path = os.path.join(
        CIC_DIR,
        filename
    )

    print(f"Loading {filename}...")

    df = pd.read_csv(
        path,
        low_memory=False
    )

    cic_parts.append(df)


cic = pd.concat(
    cic_parts,
    ignore_index=True
)

print(f"CICIoT2023 rows: {len(cic):,}")


# ============================================================
# 1. RATE
# ============================================================

print("\n" + "=" * 80)
print("1. RATE")
print("=" * 80)

five_rate = pd.to_numeric(
    five_g["Rate"],
    errors="coerce"
)

cic_rate = pd.to_numeric(
    cic["Rate"],
    errors="coerce"
)

print("\n5G-NIDD Rate:")
print(f"  Median: {five_rate.median():,.4f}")
print(f"  Mean:   {five_rate.mean():,.4f}")
print(f"  Min:    {five_rate.min():,.4f}")
print(f"  Max:    {five_rate.max():,.4f}")

print("\nCICIoT2023 Rate:")
print(f"  Median: {cic_rate.median():,.4f}")
print(f"  Mean:   {cic_rate.mean():,.4f}")
print(f"  Min:    {cic_rate.min():,.4f}")
print(f"  Max:    {cic_rate.max():,.4f}")

print("\nAssessment:")
print("Rate exists in both datasets.")
print("However, its calculation context may differ between datasets.")


# ============================================================
# 2. PACKET COUNT
# ============================================================

print("\n" + "=" * 80)
print("2. PACKET COUNT")
print("=" * 80)

five_packets = pd.to_numeric(
    five_g["TotPkts"],
    errors="coerce"
)

cic_packets = pd.to_numeric(
    cic["Number"],
    errors="coerce"
)

print("\n5G-NIDD TotPkts:")
print(f"  Median: {five_packets.median():,.4f}")
print(f"  Mean:   {five_packets.mean():,.4f}")
print(f"  Min:    {five_packets.min():,.4f}")
print(f"  Max:    {five_packets.max():,.4f}")

print("\nCICIoT2023 Number:")
print(f"  Median: {cic_packets.median():,.4f}")
print(f"  Mean:   {cic_packets.mean():,.4f}")
print(f"  Min:    {cic_packets.min():,.4f}")
print(f"  Max:    {cic_packets.max():,.4f}")

print("\nAssessment:")
print("The features are related to packet count.")
print("However, their aggregation/window definitions differ.")
print("This is an APPROXIMATE mapping, not an identical feature.")


# ============================================================
# 3. MEAN PACKET SIZE
# ============================================================

print("\n" + "=" * 80)
print("3. MEAN PACKET SIZE")
print("=" * 80)

five_bytes = pd.to_numeric(
    five_g["TotBytes"],
    errors="coerce"
)

five_count = pd.to_numeric(
    five_g["TotPkts"],
    errors="coerce"
)

five_mean_size = (
    five_bytes / five_count
)

five_mean_size = five_mean_size.replace(
    [np.inf, -np.inf],
    np.nan
)

cic_mean_size = pd.to_numeric(
    cic["AVG"],
    errors="coerce"
)

print("\n5G-NIDD derived mean packet size:")
print(f"  Median: {five_mean_size.median():,.4f}")
print(f"  Mean:   {five_mean_size.mean():,.4f}")
print(f"  Min:    {five_mean_size.min():,.4f}")
print(f"  Max:    {five_mean_size.max():,.4f}")

print("\nCICIoT2023 AVG:")
print(f"  Median: {cic_mean_size.median():,.4f}")
print(f"  Mean:   {cic_mean_size.mean():,.4f}")
print(f"  Min:    {cic_mean_size.min():,.4f}")
print(f"  Max:    {cic_mean_size.max():,.4f}")

print("\nAssessment:")
print("5G mean packet size is derived from TotBytes / TotPkts.")
print("CIC AVG directly represents average packet size.")
print("This is a reasonably strong semantic mapping.")


# ============================================================
# 4. TTL
# ============================================================

print("\n" + "=" * 80)
print("4. TTL")
print("=" * 80)

five_ttl = pd.to_numeric(
    five_g["sTtl"],
    errors="coerce"
)

cic_ttl = pd.to_numeric(
    cic["Time_To_Live"],
    errors="coerce"
)

print("\n5G-NIDD sTtl:")
print(f"  Median: {five_ttl.median():,.4f}")
print(f"  Mean:   {five_ttl.mean():,.4f}")
print(f"  Min:    {five_ttl.min():,.4f}")
print(f"  Max:    {five_ttl.max():,.4f}")
print(f"  Missing: {five_ttl.isna().sum():,}")

print("\nCICIoT2023 Time_To_Live:")
print(f"  Median: {cic_ttl.median():,.4f}")
print(f"  Mean:   {cic_ttl.mean():,.4f}")
print(f"  Min:    {cic_ttl.min():,.4f}")
print(f"  Max:    {cic_ttl.max():,.4f}")
print(f"  Missing: {cic_ttl.isna().sum():,}")

print("\nAssessment:")
print("TTL is a strong semantic mapping.")
print("Both represent packet TTL information.")


# ============================================================
# 5. PROTOCOL
# ============================================================

print("\n" + "=" * 80)
print("5. PROTOCOL FEATURES")
print("=" * 80)

proto = five_g["Proto"].astype(str).str.lower()

five_tcp = (proto == "tcp").astype(int)
five_udp = (proto == "udp").astype(int)
five_icmp = (
    proto == "icmp"
).astype(int)

print("\n5G-NIDD protocol prevalence:")

print(
    f"  TCP:  {five_tcp.mean():.4f}"
)

print(
    f"  UDP:  {five_udp.mean():.4f}"
)

print(
    f"  ICMP: {five_icmp.mean():.4f}"
)

print("\nCICIoT2023 protocol feature means:")

for feature in ["TCP", "UDP", "ICMP"]:

    values = pd.to_numeric(
        cic[feature],
        errors="coerce"
    )

    print(
        f"  {feature}: {values.mean():.4f}"
    )

print("\nAssessment:")
print("Protocol information exists in both datasets.")
print("However, the representation differs.")
print("5G uses protocol categories per record.")
print("CICIoT2023 uses protocol-related window features.")
print("Therefore this is an APPROXIMATE mapping.")


# ============================================================
# 6. FEATURES WE SHOULD NOT USE
# ============================================================

print("\n" + "=" * 80)
print("6. FEATURES NOT USED")
print("=" * 80)

print("""
The following mappings were considered but rejected:

SrcRate / DstRate
    CICIoT2023 CSV does not contain these exact fields.

TotBytes / Tot sum
    Different aggregation meanings.

SrcGap / DstGap / IAT
    Missingness and directional definitions differ.

TCP flag counts
    5G does not provide equivalent packet-level flag counts.

Loss / Hops
    No strong direct CICIoT2023 equivalent.
""")


# ============================================================
# FINAL ASSESSMENT
# ============================================================

print("\n" + "=" * 80)
print("FINAL COMMON FEATURE ASSESSMENT")
print("=" * 80)

print("""
Feature              Mapping Quality
-------------------------------------
Rate                 Reasonable
Packet_Count         Approximate
Mean_Packet_Size     Strong
TTL                  Strong
TCP                  Approximate
UDP                  Approximate
ICMP                 Approximate
-------------------------------------

The selected 7-feature representation is suitable for a
cross-domain baseline, but several mappings are approximate.

This limitation should be clearly documented in the research.
""")

print("=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)