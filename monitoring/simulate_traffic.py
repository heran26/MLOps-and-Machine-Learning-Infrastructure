"""
simulate_traffic.py

Sends synthetic requests to the deployed API so the monitoring dashboard has
something real to show without waiting for actual users. Two modes:

  --mode normal    Samples drawn from the ACTUAL training distribution
                    (models/reference_stats.json). This is the correct
                    no-drift baseline -- uniform-random values would look
                    "drifted" against real, skewed training data even with
                    no real shift happening, so this mode reproduces the
                    real proportions instead.

  --mode shifted    Deliberately skews age, hours-per-week, and occupation
                    away from the training distribution, to prove the PSI
                    check actually catches a real shift.

Usage:
    python monitoring/simulate_traffic.py --url https://your-app.onrender.com --mode normal --n 100
    python monitoring/simulate_traffic.py --url https://your-app.onrender.com --mode shifted --n 150
"""

import argparse
import json
import random
import time

import requests

REFERENCE_STATS_PATH = "models/reference_stats.json"

# Fallback category lists, only used if reference_stats.json can't be found
# (e.g. running before the main pipeline has ever committed it).
FALLBACK_CATEGORICAL = {
    "workclass": ["Private", "Self-emp-not-inc", "Local-gov", "State-gov", "Federal-gov"],
    "education": ["Bachelors", "HS-grad", "Some-college", "Masters", "Assoc-voc"],
    "marital_status": ["Never-married", "Married-civ-spouse", "Divorced"],
    "occupation": ["Adm-clerical", "Exec-managerial", "Prof-specialty", "Craft-repair", "Sales"],
    "relationship": ["Not-in-family", "Husband", "Wife", "Own-child", "Unmarried"],
    "race": ["White", "Black", "Asian-Pac-Islander"],
    "sex": ["Male", "Female"],
    "native_country": ["United-States", "Mexico", "Philippines", "Germany"],
}


def load_reference_stats():
    try:
        with open(REFERENCE_STATS_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"WARNING: {REFERENCE_STATS_PATH} not found -- falling back to made-up "
              f"proportions. Run the main pipeline once first (Actions > MLOps CI/CD "
              f"Pipeline > Run workflow) so this file exists, then re-run for a proper "
              f"no-drift baseline.")
        return None


def weighted_choice(proportions: dict):
    categories = list(proportions.keys())
    weights = list(proportions.values())
    return random.choices(categories, weights=weights, k=1)[0]


def sample_numeric(ref: dict) -> float:
    """Pick a bin according to its real proportion, then a value inside that bin."""
    edges = ref["bin_edges"]
    proportions = ref["reference_proportions"]
    bin_idx = random.choices(range(len(proportions)), weights=proportions, k=1)[0]
    low, high = edges[bin_idx], edges[bin_idx + 1]
    return random.uniform(low, high)


def random_record(shifted: bool, reference_stats) -> dict:
    if shifted:
        # Deliberately different from the ~48k-row training distribution:
        # skews older, longer hours, narrower occupation mix.
        age = random.randint(50, 75)
        hours_per_week = random.randint(55, 80)
        occupation = random.choice(["Exec-managerial", "Prof-specialty"])
        workclass = random.choice(["Private", "Self-emp-not-inc"])
        education = random.choice(["Masters", "Bachelors"])
        marital_status = random.choice(["Never-married", "Divorced"])
        relationship = random.choice(["Not-in-family", "Unmarried"])
        race = random.choice(["White", "Black", "Asian-Pac-Islander"])
        sex = random.choice(["Male", "Female"])
        native_country = random.choice(["United-States", "Mexico", "Germany"])
        fnlwgt = random.randint(20000, 300000)
        education_num = random.randint(9, 16)
        capital_gain = random.choice([0, 0, 5000, 15000])

    elif reference_stats:
        num = reference_stats["numeric"]
        cat = reference_stats["categorical"]
        age = round(sample_numeric(num["age"]))
        hours_per_week = round(sample_numeric(num["hours_per_week"]))
        fnlwgt = round(sample_numeric(num["fnlwgt"]))
        education_num = round(sample_numeric(num["education_num"]))
        capital_gain = round(sample_numeric(num["capital_gain"]))
        workclass = weighted_choice(cat["workclass"])
        education = weighted_choice(cat["education"])
        marital_status = weighted_choice(cat["marital_status"])
        occupation = weighted_choice(cat["occupation"])
        relationship = weighted_choice(cat["relationship"])
        race = weighted_choice(cat["race"])
        sex = weighted_choice(cat["sex"])
        native_country = weighted_choice(cat["native_country"])

    else:
        # No reference_stats.json available -- crude fallback, will likely
        # still show some drift since it's not weighted to real proportions.
        age = random.randint(20, 60)
        hours_per_week = random.randint(20, 50)
        fnlwgt = random.randint(20000, 300000)
        education_num = random.randint(9, 16)
        capital_gain = random.choice([0, 0, 0, 5000, 15000])
        workclass = random.choice(FALLBACK_CATEGORICAL["workclass"])
        education = random.choice(FALLBACK_CATEGORICAL["education"])
        marital_status = random.choice(FALLBACK_CATEGORICAL["marital_status"])
        occupation = random.choice(FALLBACK_CATEGORICAL["occupation"])
        relationship = random.choice(FALLBACK_CATEGORICAL["relationship"])
        race = random.choice(FALLBACK_CATEGORICAL["race"])
        sex = random.choice(FALLBACK_CATEGORICAL["sex"])
        native_country = random.choice(FALLBACK_CATEGORICAL["native_country"])

    return {
        "age": int(age),
        "workclass": workclass,
        "fnlwgt": int(fnlwgt),
        "education": education,
        "education_num": int(education_num),
        "marital_status": marital_status,
        "occupation": occupation,
        "relationship": relationship,
        "race": race,
        "sex": sex,
        "capital_gain": int(capital_gain),
        "capital_loss": 0,
        "hours_per_week": int(hours_per_week),
        "native_country": native_country,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Base URL of the deployed API")
    parser.add_argument("--mode", choices=["normal", "shifted"], default="normal")
    parser.add_argument("--n", type=int, default=100, help="Number of requests to send")
    args = parser.parse_args()

    reference_stats = load_reference_stats()
    endpoint = args.url.rstrip("/") + "/predict"
    print(f"Sending {args.n} '{args.mode}' requests to {endpoint} ...")

    ok = 0
    for i in range(args.n):
        record = random_record(shifted=(args.mode == "shifted"), reference_stats=reference_stats)
        try:
            resp = requests.post(endpoint, json=record, timeout=30)
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