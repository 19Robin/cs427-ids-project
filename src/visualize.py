"""
visualize.py

Reads outputs/results.json (produced by evaluate.py) and generates:
  - a bar chart comparing in-domain vs cross-domain metrics
  - confusion matrix heatmaps for both conditions

Saves everything to outputs/figures/
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_PATH = "outputs/results.json"
FIGURES_DIR = "outputs/figures"


def make_charts():
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    indomain = results["in_domain"]
    crossdomain = results["cross_domain"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # --- Bar chart comparison ---
    labels = list(indomain.keys())
    indomain_vals = list(indomain.values())
    crossdomain_vals = list(crossdomain.values())
    x = np.arange(len(labels))
    width = 0.35

    axes[0].bar(x - width/2, indomain_vals, width, label="In-Domain", color="#4C72B0")
    axes[0].bar(x + width/2, crossdomain_vals, width, label="Cross-Domain", color="#DD8452")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("In-Domain vs Cross-Domain Performance")
    axes[0].legend()
    axes[0].set_ylabel("Score")

    # --- Confusion matrices ---
    cm1 = np.array(results["confusion_matrix_in_domain"])
    sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues", ax=axes[1],
                xticklabels=["Benign", "Attack"], yticklabels=["Benign", "Attack"])
    axes[1].set_title("Confusion Matrix - In-Domain")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    cm2 = np.array(results["confusion_matrix_cross_domain"])
    sns.heatmap(cm2, annot=True, fmt="d", cmap="Oranges", ax=axes[2],
                xticklabels=["Benign", "Attack"], yticklabels=["Benign", "Attack"])
    axes[2].set_title("Confusion Matrix - Cross-Domain")
    axes[2].set_xlabel("Predicted")
    axes[2].set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/comparison_results.png", dpi=150)
    print(f"Chart saved to {FIGURES_DIR}/comparison_results.png")


if __name__ == "__main__":
    make_charts()
