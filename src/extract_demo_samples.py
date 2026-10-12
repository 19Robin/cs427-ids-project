from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = BASE_DIR / "data" / "processed" / "5g_nidd_processed.csv"
OUTPUT_FILE = BASE_DIR / "results" / "demo_samples_5g_nidd.csv"

FEATURES = [
    "Rate",
    "Packet_Count",
    "Mean_Packet_Size",
    "TTL",
    "TCP",
    "UDP",
    "ICMP"
]

TARGET = "Target"
RANDOM_STATE = 42
SAMPLES_PER_CLASS = 5


df = pd.read_csv(DATA_FILE)

X_train, X_test, y_train, y_test = train_test_split(
    df[FEATURES],
    df[TARGET],
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=df[TARGET]
)

test = X_test.copy()
test[TARGET] = y_test

samples = pd.concat(
    [
        test[test[TARGET] == label].sample(
            n=min(SAMPLES_PER_CLASS, int((test[TARGET] == label).sum())),
            random_state=RANDOM_STATE
        )
        for label in sorted(test[TARGET].unique())
    ],
    ignore_index=True
)

samples.to_csv(OUTPUT_FILE, index=False)

print(f"Saved {len(samples)} real 5G-NIDD test records to:")
print(OUTPUT_FILE)
print(samples.to_string(index=False))
