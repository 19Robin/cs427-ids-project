"""
train.py

Trains the lightweight model ONCE, on Domain A (5G-NIDD) only.
The trained model is saved so evaluate.py can load it later without
retraining -- this matters because the whole point of the project is
testing one fixed, trained model against unseen data, not retraining
per dataset.
"""

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from data_loader import load_domain_a

RANDOM_SEED = 42
MODEL_PATH = "outputs/models/lightweight_ids_model.pkl"


def train_model():
    X_a, y_a = load_domain_a()

    # Split domain A into train / held-out in-domain test set
    X_train, X_test, y_train, y_test = train_test_split(
        X_a, y_a, test_size=0.2, random_state=RANDOM_SEED, stratify=y_a
    )

    print(f"Training on {len(X_train)} samples...")
    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=RANDOM_SEED
    )
    model.fit(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Save the in-domain test split too, so evaluate.py uses the exact
    # same held-out data every time (not a new random split)
    X_test.to_csv("data/processed/domain_a_test_X.csv", index=False)
    y_test.to_csv("data/processed/domain_a_test_y.csv", index=False)
    print("In-domain test split saved to data/processed/")

    return model


if __name__ == "__main__":
    train_model()
