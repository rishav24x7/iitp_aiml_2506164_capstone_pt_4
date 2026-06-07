"""Train and save the churn model from the committed dataset.

Run once before starting the API:  python train_model.py
Produces model.pkl (a dict with the fitted sklearn Pipeline, feature lists, and decision threshold)
that app/main.py loads at startup. This keeps the Part 4 repo fully self-contained and reproducible.

The modelling approach mirrors Part 3: leakage-safe features from rfm_modeling_snapshot.csv, a tuned
XGBoost pipeline, and a recall-leaning threshold chosen on the validation split.
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, precision_recall_curve
from xgboost import XGBClassifier

DATA = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model.pkl"
TARGET = "churn_next_60d"
RANDOM_STATE = 42


def build_pipeline(numeric, categorical):
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])
    clf = XGBClassifier(n_estimators=600, max_depth=3, learning_rate=0.02,
                        subsample=0.8, colsample_bytree=0.8, reg_lambda=3.0,
                        min_child_weight=8, eval_metric="logloss", random_state=RANDOM_STATE)
    return Pipeline([("pre", pre), ("clf", clf)])


def main():
    df = pd.read_csv(DATA / "rfm_modeling_snapshot.csv")

    leak_or_id = ["customer_id", "snapshot_date", "split", TARGET]
    features = [c for c in df.columns if c not in leak_or_id]
    categorical = [c for c in features if df[c].dtype == "object"]
    numeric = [c for c in features if c not in categorical]

    # Leakage guard — fail loudly if a forbidden column sneaks in.
    assert {TARGET, "snapshot_date", "split"}.isdisjoint(features), "Leakage: forbidden column in features"

    train = df[df.split == "train"]
    val = df[df.split == "validation"]

    model = build_pipeline(numeric, categorical)
    model.fit(train[features], train[TARGET])

    val_proba = model.predict_proba(val[features])[:, 1]
    auc = roc_auc_score(val[TARGET], val_proba)

    # Recall-leaning threshold: F1-optimal point on the validation PR curve.
    prec, rec, thr = precision_recall_curve(val[TARGET], val_proba)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    threshold = float(thr[int(np.nanargmax(f1[:-1]))])

    joblib.dump(
        {"model": model, "features": features, "numeric": numeric,
         "categorical": categorical, "threshold": threshold},
        MODEL_PATH,
    )
    print(json.dumps({"saved": str(MODEL_PATH), "validation_roc_auc": round(auc, 4),
                      "threshold": round(threshold, 4), "n_features": len(features)}, indent=2))


if __name__ == "__main__":
    main()
