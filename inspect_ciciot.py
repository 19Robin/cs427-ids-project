import pandas as pd
from pathlib import Path

DATA_DIR = Path(r"D:\cs427-ids-project\data\raw\cic_iot")

files = sorted(DATA_DIR.glob("Merged*.csv"))

print("=" * 80)
print("CICIoT2023 DATASET INSPECTION")
print("=" * 80)

print(f"\nFiles found: {len(files)}")

for file in files:
    print(f"\n{'=' * 80}")
    print(f"FILE: {file.name}")
    print(f"{'=' * 80}")

    try:
        df = pd.read_csv(file, low_memory=False)

        print(f"Rows: {len(df):,}")
        print(f"Columns: {len(df.columns)}")

        print("\nColumns:")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:3}. {col}")

        print("\nData types:")
        print(df.dtypes.to_string())

        print("\nMissing values:")
        missing = df.isnull().sum()
        missing = missing[missing > 0]

        if len(missing) == 0:
            print("  No missing values")
        else:
            print(missing.to_string())

        print("\nDuplicate rows:")
        print(f"  {df.duplicated().sum():,}")

        # Look for possible label columns
        label_columns = [
            col for col in df.columns
            if col.lower() in ["label", "class", "attack", "target"]
        ]

        if label_columns:
            for label_col in label_columns:
                print(f"\nLabel distribution ({label_col}):")
                print(df[label_col].value_counts(dropna=False).to_string())

        print("\nFirst 3 rows:")
        print(df.head(3).to_string())

    except Exception as e:
        print(f"ERROR reading {file.name}: {e}")

print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)
