"""
data_loader.py

Loads and cleans the two datasets used in this project:
  - Domain A (5G-NIDD)   -> used for training + in-domain testing
  - Domain B (UNSW-NB15) -> used only for cross-domain testing (no training)

IMPORTANT: This file currently contains PLACEHOLDER loading functions using
synthetic data, so the rest of the pipeline can be built and tested before
the real datasets are downloaded. Replace the body of each function with
real pd.read_csv(...) calls once you have the actual CSV files in data/raw/.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

RANDOM_SEED = 42

# Fixed protocol encoding shared by BOTH datasets. This matters: if each
# dataset were encoded independently (e.g. via pandas .cat.codes), "tcp"
# could end up as a different number in each dataset purely by chance,
# silently corrupting the cross-domain comparison. Using one fixed map
# for both loaders guarantees "tcp" always means the same number
# everywhere. Anything not listed here (UNSW-NB15 has many more protocol
# types than 5G-NIDD) falls into the "other" bucket.
PROTOCOL_MAP = {"tcp": 1, "udp": 2, "icmp": 3}
PROTOCOL_OTHER = 0


def encode_protocol(series):
    return series.astype(str).str.lower().map(PROTOCOL_MAP).fillna(PROTOCOL_OTHER).astype(int)


def load_domain_a(path="data/raw/5g_nidd.csv"):
    """
    Load and clean the 5G-NIDD dataset (training domain).

    5G-NIDD is Argus-based network flow data. We select a subset of features
    that conceptually also exist in UNSW-NB15 (duration, packet counts, byte
    counts, rate, protocol, TTL, loss, TCP timing), so the same trained model
    can later be applied to UNSW-NB15's differently-named columns without
    needing to retrain -- this shared feature set is what makes the
    cross-domain comparison possible at all.
    """
    df = pd.read_csv(path)

    # Build the shared feature schema, renamed to common names
    shared = pd.DataFrame({
        "duration": df["Dur"],
        "src_packets": df["SrcPkts"],
        "dst_packets": df["DstPkts"],
        "src_bytes": df["SrcBytes"],
        "dst_bytes": df["DstBytes"],
        "rate": df["Rate"],
        "src_ttl": df["sTtl"],
        "dst_ttl": df["dTtl"],
        "src_loss": df["SrcLoss"],
        "dst_loss": df["DstLoss"],
        "tcp_rtt": df["TcpRtt"],
        "synack": df["SynAck"],
        "ackdat": df["AckDat"],
        "protocol": df["Proto"],
    })

    # Fill any missing numeric values with 0 (common for fields like TCP
    # timing that don't apply to non-TCP protocols such as ICMP/UDP)
    numeric_cols = shared.columns.drop("protocol")
    shared[numeric_cols] = shared[numeric_cols].fillna(0)

    # Encode protocol using the shared fixed map (see top of file) so
    # "tcp"/"udp"/"icmp" get identical codes in both datasets.
    shared["protocol"] = encode_protocol(shared["protocol"])

    # Label: Benign -> 0, Malicious -> 1
    y = (df["Label"] == "Malicious").astype(int)
    y.name = "label"

    return shared, y


def load_domain_b(path="data/raw/unsw_nb15.csv"):
    """
    Load and clean the UNSW-NB15 dataset (cross-domain test domain).

    Maps UNSW-NB15's columns onto the exact same shared feature schema used
    in load_domain_a(), so the model trained on 5G-NIDD can be evaluated on
    this dataset directly, with no retraining.
    """
    df = pd.read_csv(path)

    shared = pd.DataFrame({
        "duration": df["dur"],
        "src_packets": df["spkts"],
        "dst_packets": df["dpkts"],
        "src_bytes": df["sbytes"],
        "dst_bytes": df["dbytes"],
        "rate": df["rate"],
        "src_ttl": df["sttl"],
        "dst_ttl": df["dttl"],
        "src_loss": df["sloss"],
        "dst_loss": df["dloss"],
        "tcp_rtt": df["tcprtt"],
        "synack": df["synack"],
        "ackdat": df["ackdat"],
        "protocol": df["proto"],
    })

    numeric_cols = shared.columns.drop("protocol")
    shared[numeric_cols] = shared[numeric_cols].fillna(0)

    # Encode protocol using the same shared fixed map used in load_domain_a,
    # so "tcp"/"udp"/"icmp" mean the same numeric value in both datasets.
    shared["protocol"] = encode_protocol(shared["protocol"])

    # Label is already 0 (normal) / 1 (attack) in this dataset
    y = df["label"].astype(int)
    y.name = "label"

    return shared, y


if __name__ == "__main__":
    X_a, y_a = load_domain_a()
    X_b, y_b = load_domain_b()
    print(f"Domain A (5G-NIDD): {X_a.shape[0]} rows, {X_a.shape[1]} features, "
          f"{y_a.mean()*100:.1f}% attack samples")
    print(f"Domain B (UNSW-NB15): {X_b.shape[0]} rows, {X_b.shape[1]} features, "
          f"{y_b.mean()*100:.1f}% attack samples")