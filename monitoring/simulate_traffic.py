"""
simulate_traffic.py

Sends synthetic requests to the deployed API so the monitoring dashboard has
something real to show without waiting for actual users. Two modes:

  --mode normal   Samples that look like the training distribution (no drift
                   expected). Use this to confirm the monitor reports "OK".

  --mode shifted   Deliberately skews age, hours-per-week, and occupation mix
                   away from the training distribution. Use this to prove the
                   PSI check actually catches a real shift -- this is the
                   more useful one for a portfolio demo/screenshot.

Usage:
    python monitoring/simulate_traffic.py --url https://your-app.onrender.com --mode shifted --n 150
"""

import argparse
import random
import time

import requests

WORKCLASSES = ["Private", "Self-emp-not-inc", "Local-gov", "State-gov", "Federal-gov"]
EDUCATIONS = ["Bachelors", "HS-grad", "Some-college", "Masters", "Assoc-voc"]
MARITAL = ["Never-married", "Married-civ-spouse", "Divorced"]
OCCUPATIONS = ["Adm-clerical", "Exec-managerial", "Prof-specialty", "Craft-repair", "Sales"]
RELATIONSHIPS = ["Not-in-family", "Husband", "Wife", "Own-child", "Unmarried"]
RACES = ["White", "Black", "Asian-Pac-Islander"]
SEXES = ["Male", "Female"]
COUNTRIES = ["United-States", "Mexico", "Philippines", "Germany"]


def random_record(shifted: bool) -> dict:
    if shifted:
        # Deliberately different from the ~48k-row training distribution:
        # skews older, longer hours, narrower occupation mix.
        age = random.randint(50, 75)
        hours_per_week = random.randint(55, 80)
        occupation = random.choice(["Exec-managerial", "Prof-specialty"])
    else:
        age = random.randint(20, 60)
        hours_per_week = random.randint(20, 50)
        occupation = random.choice(OCCUPATIONS)

    return {
        "age": age,
        "workclass": random.choice(WORKCLASSES),
        "fnlwgt": random.randint(20000, 300000),
        "education": random.choice(EDUCATIONS),
        "education_num": random.randint(9, 16),
        "marital_status": random.choice(MARITAL),
        "occupation": occupation,
        "relationship": random.choice(RELATIONSHIPS),
        "race": random.choice(RACES),
        "sex": random.choice(SEXES),
        "capital_gain": random.choice([0, 0, 0, 5000, 15000]),
        "capital_loss": 0,
        "hours_per_week": hours_per_week,
        "native_country": random.choice(COUNTRIES),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Base URL of the deployed API")
    parser.add_argument("--mode", choices=["normal", "shifted"], default="normal")
    parser.add_argument("--n", type=int, default=100, help="Number of requests to send")
    args = parser.parse_args()

    endpoint = args.url.rstrip("/") + "/predict"
    print(f"Sending {args.n} '{args.mode}' requests to {endpoint} ...")

    ok = 0
    for i in range(args.n):
        record = random_record(shifted=(args.mode == "shifted"))
        try:
            resp = requests.post(endpoint, json=record, timeout=15)
            if resp.status_code == 200:
                ok += 1
            else:
                print(f"[{i}] HTTP {resp.status_code}: {resp.text[:200]}")
        except requests.RequestException as e:
            print(f"[{i}] request failed: {e}")
        if i % 20 == 0:
            print(f"  ...{i}/{args.n} sent")
        time.sleep(0.05)  # be gentle on the free-tier instance

    print(f"Done. {ok}/{args.n} succeeded.")
    print("Now run the drift check (locally or via the GitHub Actions workflow) to see the report.")


if __name__ == "__main__":
    main()
