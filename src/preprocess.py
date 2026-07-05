"""
preprocess.py
Shared data loading & preprocessing logic used by both train.py and evaluate.py.

Using a fixed RANDOM_STATE guarantees train.py and evaluate.py always produce
the exact same train/test split, so evaluate.py scores the candidate model on
a held-out set it never trained on -- this is what makes the accuracy
comparison in the CI/CD pipeline trustworthy.

Swap DATA_URL / COLUMNS / feature lists for your own dataset when you're
ready -- everything downstream (train.py, evaluate.py, serving/app.py) only
depends on the functions in this file, not on this specific dataset.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

RANDOM_STATE = 42

# UCI "Adult" census income dataset: ~48,800 rows, binary classification
# (predict whether someone earns >$50K/year). Good stand-in for "enough data
# for a real accuracy signal" -- swap for your own CSV/URL when ready.
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income",
]

NUMERIC_FEATURES = [
    "age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"
]
CATEGORICAL_FEATURES = [
    "workclass", "education", "marital_status", "occupation",
    "relationship", "race", "sex", "native_country",
]


def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_URL, names=COLUMNS, sep=r",\s*", engine="python", na_values="?")
    df = df.dropna()
    df["income"] = (df["income"].str.strip() == ">50K").astype(int)
    return df


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def load_and_split():
    df = load_raw_data()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df["income"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test
