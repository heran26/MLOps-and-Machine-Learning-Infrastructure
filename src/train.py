"""
train.py
Trains a CANDIDATE model and saves it + its metrics under models/.

This script never touches the "production" model directly -- evaluate.py
is the only thing allowed to promote a candidate to production, and only
if it proves itself better on held-out data.

Run locally, in Colab, or (this is the point) automatically inside the
GitHub Actions runner -- same script, same result, no local PC required.
"""

import json
import os
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from preprocess import (
    build_preprocessor, load_and_split, RANDOM_STATE,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES,
)

MODELS_DIR = "../models"  # relative to src/, matches workflow's working-directory
N_BINS = 10  # decile bins for numeric drift reference


def build_reference_stats(X_train):
    """
    Snapshot of the TRAINING feature distribution, used later by the drift
    monitor to tell whether live traffic looks statistically different.

    Numeric features: decile bin edges + the proportion of training rows
    falling in each bin (roughly 10% each, by construction).
    Categorical features: proportion of training rows in each category.
    """
    stats = {"numeric": {}, "categorical": {}}

    for col in NUMERIC_FEATURES:
        values = X_train[col].astype(float).values
        bin_edges = np.quantile(values, np.linspace(0, 1, N_BINS + 1)).tolist()
        # Guard against duplicate edges (happens with skewed/discrete columns)
        bin_edges = sorted(set(bin_edges))
        if len(bin_edges) < 2:
            bin_edges = [values.min(), values.max() + 1e-9]
        counts, _ = np.histogram(values, bins=bin_edges)
        proportions = (counts / counts.sum()).tolist()
        stats["numeric"][col] = {"bin_edges": bin_edges, "reference_proportions": proportions}

    for col in CATEGORICAL_FEATURES:
        proportions = X_train[col].value_counts(normalize=True).to_dict()
        stats["categorical"][col] = proportions

    return stats


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading and splitting data...")
    X_train, X_test, y_train, y_test = load_and_split()
    print(f"Training rows: {len(X_train)} | Test rows (held out): {len(X_test)}")

    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", GradientBoostingClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
        )),
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)

    train_preds = pipeline.predict(X_train)
    train_acc = accuracy_score(y_train, train_preds)
    print(f"Train accuracy: {train_acc:.4f}")

    candidate_path = os.path.join(MODELS_DIR, "candidate_model.pkl")
    joblib.dump(pipeline, candidate_path)
    print(f"Candidate model saved to {candidate_path}")

    with open(os.path.join(MODELS_DIR, "candidate_metrics.json"), "w") as f:
        json.dump({"train_accuracy": train_acc}, f, indent=2)

    print("Building reference distribution snapshot for drift monitoring...")
    reference_stats = build_reference_stats(X_train)
    with open(os.path.join(MODELS_DIR, "candidate_reference_stats.json"), "w") as f:
        json.dump(reference_stats, f, indent=2)
    print("Saved candidate_reference_stats.json")


if __name__ == "__main__":
    main()
