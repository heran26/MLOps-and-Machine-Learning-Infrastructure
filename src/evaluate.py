"""
evaluate.py
Scores the candidate model on the held-out test set, compares it against the
CURRENT production model, and only promotes the candidate if it genuinely
improves accuracy. This is the "only deploy if accuracy improves" gate.

Writes deploy=true/false and accuracy=<value> to $GITHUB_OUTPUT so the
workflow can branch on the result in later steps (commit + deploy).
"""

import json
import os
import shutil
import joblib
from sklearn.metrics import accuracy_score, f1_score

from preprocess import load_and_split

MODELS_DIR = "../models"
CANDIDATE_MODEL = os.path.join(MODELS_DIR, "candidate_model.pkl")
PRODUCTION_MODEL = os.path.join(MODELS_DIR, "production_model.pkl")
PRODUCTION_METRICS = os.path.join(MODELS_DIR, "production_metrics.json")


def set_github_output(name: str, value: str):
    gh_output = os.environ.get("GITHUB_OUTPUT")
    if gh_output:
        with open(gh_output, "a") as f:
            f.write(f"{name}={value}\n")
    print(f"{name}={value}")


def main():
    _, X_test, _, y_test = load_and_split()

    candidate = joblib.load(CANDIDATE_MODEL)
    candidate_preds = candidate.predict(X_test)
    candidate_acc = accuracy_score(y_test, candidate_preds)
    candidate_f1 = f1_score(y_test, candidate_preds)
    print(f"Candidate test accuracy: {candidate_acc:.4f} | F1: {candidate_f1:.4f}")

    if os.path.exists(PRODUCTION_MODEL) and os.path.exists(PRODUCTION_METRICS):
        with open(PRODUCTION_METRICS) as f:
            prod_metrics = json.load(f)
        production_acc = prod_metrics.get("test_accuracy", 0.0)
        print(f"Current production test accuracy: {production_acc:.4f}")
    else:
        production_acc = 0.0
        print("No production model yet -- candidate becomes production if it beats a 0.0 baseline.")

    deploy = candidate_acc > production_acc

    if deploy:
        print("Candidate IMPROVES on production -> promoting candidate to production.")
        shutil.copyfile(CANDIDATE_MODEL, PRODUCTION_MODEL)
        with open(PRODUCTION_METRICS, "w") as f:
            json.dump({"test_accuracy": candidate_acc, "test_f1": candidate_f1}, f, indent=2)
    else:
        print("Candidate does NOT improve on production -> keeping current production model.")

    set_github_output("deploy", "true" if deploy else "false")
    set_github_output("accuracy", f"{candidate_acc:.4f}")


if __name__ == "__main__":
    main()
