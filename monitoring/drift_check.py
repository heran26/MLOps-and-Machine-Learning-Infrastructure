"""
drift_check.py

Fetches recently logged live predictions from the deployed API, compares
their feature distribution against the training-time reference distribution
using the Population Stability Index (PSI), and separately checks prediction
fairness across sensitive groups using the four-fifths rule.

If either check fails, this script triggers the existing retraining pipeline
(mlops-pipeline.yml) via the GitHub Actions REST API.

--- Methodology notes (for your own write-up) ---

Population Stability Index (PSI): a standard metric in credit-risk and MLOps
practice for detecting distribution shift between two samples of the same
variable. For each bin, PSI accumulates (live% - reference%) * ln(live% / reference%).
Conventional thresholds (Siddiqi, 2006, credit scoring literature):
    PSI < 0.10  -> no significant shift
    0.10-0.25   -> moderate shift, worth investigating
    PSI > 0.25  -> significant shift, model likely degrading on new data
This script uses 0.25 as the "retrain" trigger threshold (configurable below).

Four-fifths rule (80% rule): a standard adverse-impact heuristic from
employment discrimination law (US EEOC, 1978 Uniform Guidelines), widely
reused in the algorithmic fairness literature as a simple, interpretable
screen (e.g. Feldman et al., 2015, "Certifying and Removing Disparate Impact").
Flags a group if its positive-outcome rate is less than 80% of the highest
group's rate. This is a coarse screen, not a substitute for a full fairness
audit (equalized odds, calibration, etc.) -- worth discussing as a limitation
in any write-up.
"""

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

API_BASE_URL = os.environ["API_BASE_URL"].rstrip("/")
MONITOR_API_KEY = os.environ.get("MONITOR_API_KEY", "")
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPOSITORY = os.environ.get("GITHUB_REPOSITORY", "")  # "owner/repo", auto-set in Actions

REFERENCE_STATS_PATH = "models/reference_stats.json"
HISTORY_DIR = "monitoring/history"

MIN_SAMPLES = int(os.environ.get("MIN_SAMPLES", "30"))       # skip check if too little live data
PSI_ALERT_THRESHOLD = float(os.environ.get("PSI_ALERT_THRESHOLD", "0.25"))
FAIRNESS_RATIO_THRESHOLD = float(os.environ.get("FAIRNESS_RATIO_THRESHOLD", "0.8"))

SENSITIVE_ATTRIBUTES = ["sex", "race"]  # groups checked for fairness


def fetch_live_data() -> pd.DataFrame:
    headers = {"X-API-Key": MONITOR_API_KEY} if MONITOR_API_KEY else {}
    resp = requests.get(f"{API_BASE_URL}/monitoring/raw-data", headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    records = payload.get("records", [])
    if not records:
        return pd.DataFrame()
    rows = []
    for r in records:
        row = dict(r["features"])
        row["prediction"] = r["prediction"]
        row["probability"] = r["probability"]
        rows.append(row)
    return pd.DataFrame(rows)


def compute_psi_numeric(live_values, bin_edges, reference_proportions) -> float:
    counts, _ = np.histogram(live_values, bins=bin_edges)
    total = counts.sum()
    if total == 0:
        return 0.0
    live_proportions = counts / total
    eps = 1e-6
    psi = 0.0
    for live_p, ref_p in zip(live_proportions, reference_proportions):
        live_p = max(live_p, eps)
        ref_p = max(ref_p, eps)
        psi += (live_p - ref_p) * np.log(live_p / ref_p)
    return float(psi)


def compute_psi_categorical(live_series, reference_proportions: dict) -> float:
    live_counts = live_series.value_counts(normalize=True).to_dict()
    categories = set(live_counts) | set(reference_proportions)
    eps = 1e-6
    psi = 0.0
    for cat in categories:
        live_p = max(live_counts.get(cat, 0.0), eps)
        ref_p = max(reference_proportions.get(cat, 0.0), eps)
        psi += (live_p - ref_p) * np.log(live_p / ref_p)
    return float(psi)


def compute_bias_report(live_df: pd.DataFrame) -> dict:
    selection_rates = {}
    flagged_groups = []

    for attr in SENSITIVE_ATTRIBUTES:
        if attr not in live_df.columns:
            continue
        rates = live_df.groupby(attr)["prediction"].mean().to_dict()
        selection_rates[attr] = rates

        if len(rates) < 2:
            continue
        max_rate = max(rates.values())
        if max_rate == 0:
            continue
        for group, rate in rates.items():
            ratio = rate / max_rate
            if ratio < FAIRNESS_RATIO_THRESHOLD:
                flagged_groups.append({
                    "attribute": attr, "group": group,
                    "selection_rate": rate, "ratio_to_max_group": ratio,
                })

    return {
        "selection_rates": selection_rates,
        "flagged_groups": flagged_groups,
        "bias_flag": len(flagged_groups) > 0,
    }


def trigger_retrain():
    if not GITHUB_TOKEN or not GITHUB_REPOSITORY:
        print("No GITHUB_TOKEN/GITHUB_REPOSITORY available -- skipping auto-trigger "
              "(expected if running outside GitHub Actions).")
        return
    url = f"https://api.github.com/repos/{GITHUB_REPOSITORY}/actions/workflows/mlops-pipeline.yml/dispatches"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }
    resp = requests.post(url, headers=headers, json={"ref": "main"}, timeout=30)
    if resp.status_code == 204:
        print("Retraining pipeline triggered successfully.")
    else:
        print(f"Failed to trigger retraining: {resp.status_code} {resp.text}")


def publish_report(report: dict):
    headers = {"X-API-Key": MONITOR_API_KEY} if MONITOR_API_KEY else {}
    headers["Content-Type"] = "application/json"
    try:
        resp = requests.post(f"{API_BASE_URL}/monitoring/report", headers=headers, json=report, timeout=30)
        if resp.status_code == 200:
            print("Report published to dashboard.")
        else:
            print(f"Could not publish report to dashboard: {resp.status_code} {resp.text}")
    except requests.RequestException as e:
        print(f"Could not reach dashboard endpoint: {e}")


def main():
    with open(REFERENCE_STATS_PATH) as f:
        reference_stats = json.load(f)

    print("Fetching live prediction data from deployed API...")
    live_df = fetch_live_data()
    n_samples = len(live_df)
    print(f"Fetched {n_samples} live records.")

    if n_samples < MIN_SAMPLES:
        print(f"Fewer than {MIN_SAMPLES} live samples -- skipping statistical check "
              f"(not enough data for a meaningful comparison yet).")
        sys.exit(0)

    feature_psi = {}
    for col, ref in reference_stats["numeric"].items():
        if col not in live_df.columns:
            continue
        feature_psi[col] = compute_psi_numeric(
            live_df[col].astype(float).values, ref["bin_edges"], ref["reference_proportions"]
        )
    for col, ref in reference_stats["categorical"].items():
        if col not in live_df.columns:
            continue
        feature_psi[col] = compute_psi_categorical(live_df[col], ref)

    max_psi = max(feature_psi.values()) if feature_psi else 0.0
    drift_flag = max_psi > PSI_ALERT_THRESHOLD

    bias_report = compute_bias_report(live_df)

    alert = drift_flag or bias_report["bias_flag"]

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_samples": n_samples,
        "feature_psi": feature_psi,
        "max_psi": max_psi,
        "drift_flag": drift_flag,
        "bias": bias_report,
        "alert": alert,
    }

    print(json.dumps(report, indent=2, default=str))

    os.makedirs(HISTORY_DIR, exist_ok=True)
    filename = f"{HISTORY_DIR}/{report['timestamp'].replace(':', '-')}.json"
    with open(filename, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved report to {filename}")

    publish_report(report)

    if alert:
        print("ALERT: drift or bias threshold exceeded -> triggering retraining pipeline.")
        trigger_retrain()
    else:
        print("All checks within normal range -- no action needed.")


if __name__ == "__main__":
    main()
