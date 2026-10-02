import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data" / "raw"

# Find CSV files
csv_files = list(DATA_DIR.glob("*.csv"))

print("CSV files found:")
for file in csv_files:
    print(" -", file.name)

if not csv_files:
    print("\nNo CSV files found.")
    exit()

# Use the first CSV for now
file = csv_files[0]

print("\nLoading:", file)
print("Please wait...")

df = pd.read_csv(file)

print("\n==============================")
print("DATASET SHAPE")
print("==============================")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\n==============================")
print("COLUMN NAMES")
print("==============================")

for i, column in enumerate(df.columns, 1):
    print(f"{i}. {column}")

print("\n==============================")
print("FIRST 5 ROWS")
print("==============================")
print(df.head())

print("\n==============================")
print("DATA TYPES")
print("==============================")
print(df.dtypes)

print("\n==============================")
print("MISSING VALUES")
print("==============================")

missing = df.isnull().sum()
missing = missing[missing > 0]

if len(missing) == 0:
    print("No missing values found.")
else:
    print(missing)

print("\n==============================")
print("DUPLICATES")
print("==============================")
print("Duplicate rows:", df.duplicated().sum())


print("\n==============================")
print("LABEL DISTRIBUTION")
print("==============================")
print(df["Label"].value_counts(dropna=False))

print("\n==============================")
print("ATTACK TYPE DISTRIBUTION")
print("==============================")
print(df["Attack Type"].value_counts(dropna=False))

print("\n==============================")
print("ATTACK TOOL DISTRIBUTION")
print("==============================")
print(df["Attack Tool"].value_counts(dropna=False))

print("\n==============================")
print("UNIQUE VALUES - OBJECT COLUMNS")
print("==============================")

object_columns = df.select_dtypes(include="object").columns

for column in object_columns:
    print(f"\n{column}:")
    print(df[column].value_counts(dropna=False).head(20))



    print("\n==============================")
print("MISSING VALUE ANALYSIS")
print("==============================")

missing_count = df.isnull().sum()
missing_percent = (missing_count / len(df)) * 100

missing_table = pd.DataFrame({
    "Missing Count": missing_count,
    "Missing %": missing_percent
})

missing_table = missing_table[
    missing_table["Missing Count"] > 0
].sort_values(
    "Missing %",
    ascending=False
)

print(missing_table.to_string())



print("\n==============================")
print("NUMERICAL FEATURE SUMMARY")
print("==============================")

numeric_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

print(df[numeric_columns].describe().T.to_string())



print("\n==============================")
print("POTENTIAL DUPLICATE FEATURES")
print("==============================")

columns_to_check = [
    "Dur",
    "RunTime",
    "Mean",
    "Sum",
    "Min",
    "Max"
]

for i in range(len(columns_to_check)):
    for j in range(i + 1, len(columns_to_check)):
        col1 = columns_to_check[i]
        col2 = columns_to_check[j]

        identical = df[col1].equals(df[col2])

        print(f"{col1} == {col2}: {identical}")




print("\n==============================")
print("MISSINGNESS BY LABEL")
print("==============================")

check_columns = [
    "dVid",
    "sVid",
    "DstWin",
    "DstTCPBase",
    "SrcWin",
    "dTos",
    "dHops",
    "dDSb",
    "dTtl",
    "DstGap",
    "SrcGap",
    "SrcTCPBase"
]

for column in check_columns:
    missing_by_label = (
        df.groupby("Label")[column]
        .apply(lambda x: x.isna().mean() * 100)
    )

    print(f"\n{column}:")
    print(missing_by_label)