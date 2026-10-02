"""
live_predictor.py

Loads an EXISTING trained model (default: models/random_forest_10_trees.joblib)
and classifies live flow records. The model is only used for inference:
nothing here calls fit() and live traffic is never used for training.

Before use, the model is checked against what the live features produce:

  * feature_names_in_ must equal FEATURES (same names, same order).
    The 10-tree Random Forest stores
    ['Rate', 'Packet_Count', 'Mean_Packet_Size', 'TTL', 'TCP', 'UDP', 'ICMP'].
  * classes_ must contain 0 (benign) and 1 (malicious); preprocess_data.py sets
    Target = 1 when the 5G-NIDD Label is "Malicious", else 0. The malicious
    probability is read from the predict_proba column of class 1 (looked up via
    classes_, not assumed to be column 1).
  * For a Random Forest, predict_proba averages each tree's leaf class
    proportions (leaves can be mixed because identical 5G-NIDD feature vectors
    occur with both labels). It is a model score, not a calibrated probability.
"""

from pathlib import Path

import joblib
import numpy as np

try:
    from .live_features import FEATURES, records_to_model_input
except ImportError:  # run directly as a script
    from live_features import FEATURES, records_to_model_input


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "random_forest_10_trees.joblib"

BENIGN = 0
MALICIOUS = 1
LABELS = {BENIGN: "BENIGN", MALICIOUS: "MALICIOUS"}


class ModelCompatibilityError(Exception):
    """The model file is missing or does not match the live features."""


class LivePredictor:

    def __init__(self, model_path=DEFAULT_MODEL_PATH, model=None):
        self.model_path = Path(model_path)
        if model is None:
            if not self.model_path.exists():
                raise ModelCompatibilityError(
                    f"Model file not found: {self.model_path}. Model files are "
                    "git-ignored; create it with `python src/test_lightweight_models.py`."
                )
            try:
                model = joblib.load(self.model_path)
            except Exception as error:
                raise ModelCompatibilityError(
                    f"Could not load {self.model_path.name}: {error}"
                ) from error
        self.model = model
        self._check_model()
        self.malicious_column = list(self.model.classes_).index(MALICIOUS)

    def _check_model(self):
        model = self.model
        for attribute in ("predict", "predict_proba", "classes_"):
            if not hasattr(model, attribute):
                raise ModelCompatibilityError(
                    f"The model has no '{attribute}'; is it a trained classifier?"
                )

        names = getattr(model, "feature_names_in_", None)
        if names is not None and list(names) != FEATURES:
            raise ModelCompatibilityError(
                f"Model expects features {list(names)} but the live extractor "
                f"produces {FEATURES}."
            )
        if getattr(model, "n_features_in_", len(FEATURES)) != len(FEATURES):
            raise ModelCompatibilityError(
                f"Model expects {model.n_features_in_} features, live extractor "
                f"produces {len(FEATURES)}."
            )

        classes = [int(c) for c in model.classes_]
        if sorted(classes) != [BENIGN, MALICIOUS]:
            raise ModelCompatibilityError(
                f"Model classes are {classes}; expected [0 (benign), 1 (malicious)]."
            )

    def describe(self):
        return {
            "file": self.model_path.name,
            "type": type(self.model).__name__,
            "trees": getattr(self.model, "n_estimators", None),
            "features": list(getattr(self.model, "feature_names_in_", FEATURES)),
            "classes": [int(c) for c in self.model.classes_],
        }

    def predict(self, records):
        """
        Classify flow records (dicts from FlowAggregator.flush()).
        Returns (predictions, malicious_probabilities) as numpy arrays.
        """
        X = records_to_model_input(records)
        if X.empty:
            return np.array([], dtype=int), np.array([], dtype=float)
        if not np.isfinite(X.to_numpy()).all():
            # Should not happen (the extractor never produces NaN/inf), but a
            # bad row must not crash the monitor: replace and carry on.
            X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        predictions = self.model.predict(X).astype(int)
        probabilities = self.model.predict_proba(X)[:, self.malicious_column]
        return predictions, probabilities
