"""
inspect_data.py

Run this once, after placing your downloaded CSV files in data/raw/,
to see the actual column names, shapes, and label column in each dataset.
This tells us exactly what data_loader.py needs to work with.

Usage (from project root):
    python src/inspect_data.py
"""

import pandas as pd
import os

RAW_DIR = "data/raw"


def inspect(filename):
    path = os.path.join(RAW_DIR, filename)
    if not os.path.exists(path):
        print(f"[!] {filename} not found in {RAW_DIR}/ -- skipping")
        return

    print(f"\n{'='*60}")
    print(f"FILE: {filename}")
    print('='*60)

    # Read just a sample first in case the file is huge
    df = pd.read_csv(path, nrows=2000)

    print(f"Shape (first 2000 rows sample): {df.shape}")
    print(f"\nColumn names:")
    for col in df.columns:
        print(f"  - {col}  (dtype: {df[col].dtype})")

    print(f"\nFirst 3 rows:")
    print(df.head(3).to_string())

    # Try to guess which column is the label
    likely_label_cols = [c for c in df.columns if c.lower() in
                          ['label', 'attack_cat', 'attack type', 'class', 'target']]
    if likely_label_cols:
        print(f"\nLikely label column(s): {likely_label_cols}")
        for col in likely_label_cols:
            print(f"  {col} unique values: {df[col].unique()[:20]}")


if __name__ == "__main__":
    print("Looking for CSV files in data/raw/...")
    files = [f for f in os.listdir(RAW_DIR) if f.endswith('.csv')]
    if not files:
        print("No CSV files found in data/raw/. Download the datasets first.")
    else:
        print(f"Found: {files}")
        for f in files:
            inspect(f)
