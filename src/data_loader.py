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


def load_domain_a(path="data/raw/5g_nidd.csv"):
    """
    Load and clean the 5G-NIDD dataset (training domain).

    TODO once the real dataset is downloaded:
        df = pd.read_csv(path)
        # drop irrelevant columns (e.g. IDs, timestamps if not useful)
        # handle missing values
        # encode the label column as 0 (benign) / 1 (attack)
        # separate into X (features) and y (label)
        return X, y
    """
    print("[PLACEHOLDER] Using synthetic data to stand in for 5G-NIDD.")
    X, y = make_classification(
        n_samples=5000, n_features=20, n_informative=12, n_redundant=4,
        weights=[0.7, 0.3], flip_y=0.02, class_sep=1.5, random_state=1
    )
    columns = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=columns)
    y_series = pd.Series(y, name="label")
    return X_df, y_series


def load_domain_b(path="data/raw/unsw_nb15.csv"):
    """
    Load and clean the UNSW-NB15 dataset (cross-domain test domain).

    TODO once the real dataset is downloaded:
        df = pd.read_csv(path)
        # UNSW-NB15 has different column names/features than 5G-NIDD --
        # you will need to select/align a comparable feature subset,
        # or retrain domain A's model using only shared feature types.
        return X, y
    """
    print("[PLACEHOLDER] Using synthetic data to stand in for UNSW-NB15.")
    X, y = make_classification(
        n_samples=5000, n_features=20, n_informative=12, n_redundant=4,
        weights=[0.6, 0.4], flip_y=0.05, class_sep=0.9, random_state=2
    )
    columns = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=columns)
    y_series = pd.Series(y, name="label")
    return X_df, y_series


if __name__ == "__main__":
    X_a, y_a = load_domain_a()
    X_b, y_b = load_domain_b()
    print(f"Domain A (5G-NIDD): {X_a.shape[0]} rows, {X_a.shape[1]} features, "
          f"{y_a.mean()*100:.1f}% attack samples")
    print(f"Domain B (UNSW-NB15): {X_b.shape[0]} rows, {X_b.shape[1]} features, "
          f"{y_b.mean()*100:.1f}% attack samples")
