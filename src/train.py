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
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from preprocess import build_preprocessor, load_and_split, RANDOM_STATE

MODELS_DIR = "../models"  # relative to src/, matches workflow's working-directory


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


if __name__ == "__main__":
    main()